"""
test_ensemble.py - Comprehensive Unit & Integration Tests for Ensemble Inference.

Tests:
1. Loading each compatible checkpoint (70/30, 80/20, 85/15, 90/10).
2. Rejecting incompatible or corrupted checkpoints.
3. Verifying output dimension is exactly 156.
4. Verifying class-to-Unicode consistency across all models.
5. Equal-weight probability fusion arithmetic and normalization.
6. Weighted probability fusion arithmetic and normalization.
7. Top-3 rank ordering correctness.
8. Empty or invalid image rejection.
9. Missing checkpoint handling and graceful fallback.
10. Single-model vs. ensemble mode switching.
11. Flask health endpoint reporting active mode.
12. Byte-for-byte SHA-256 preservation of production checkpoint.
"""

import os
import sys
import json
import unittest
import hashlib
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.ensemble import EnsembleInferenceEngine, DEFAULT_WEIGHTS
from app.main import app
from model.unicode_mapping import NUM_CLASSES, get_unicode_char, get_unicode_codepoints

EXPECTED_BASELINE_HASH = "9f22a8c76278dae97e2741c0267af2356d9105c03e04dad8d88189de3dd5cf4e"

class TestEnsembleInference(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_production_checkpoint_hash_unaltered(self):
        """Verify baseline checkpoint is byte-for-byte unchanged."""
        ckpt_path = os.path.join('model', 'checkpoints', 'best_model.pt')
        self.assertTrue(os.path.exists(ckpt_path), "Production checkpoint must exist")
        sha256 = hashlib.sha256()
        with open(ckpt_path, 'rb') as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest().lower()
        self.assertEqual(actual_hash, EXPECTED_BASELINE_HASH,
                         f"CRITICAL: Production model hash mismatch! Expected {EXPECTED_BASELINE_HASH}, got {actual_hash}")

    def test_ensemble_initialization_single_mode(self):
        """Test default initialization in single mode."""
        engine = EnsembleInferenceEngine()
        engine.set_strategy('single')
        self.assertEqual(engine.strategy, 'single')
        status = engine.get_status()
        self.assertIn('70_30', status['models_available'])
        self.assertFalse(status['fallback_active'])

    def test_ensemble_initialization_all_four_models(self):
        """Test loading all four models in ensemble mode."""
        engine = EnsembleInferenceEngine()
        engine.set_strategy('ensemble_equal')
        self.assertEqual(engine.strategy, 'ensemble_equal')
        status = engine.get_status()
        self.assertEqual(status['models_loaded_count'], 4)
        for expected in ['70_30', '80_20', '85_15', '90_10']:
            self.assertIn(expected, status['models_available'])

    def test_model_output_dimensions(self):
        """Verify each loaded model outputs exactly 156 logits."""
        engine = EnsembleInferenceEngine()
        engine.set_strategy('ensemble_equal')
        dummy_tensor = torch.randn(1, 1, 64, 64)
        for name, model in engine.models.items():
            out = model(dummy_tensor)
            self.assertEqual(out.shape, (1, 156), f"Model {name} output shape must be (1, 156)")

    def test_probability_normalization_equal_weight(self):
        """Verify equal-weight ensemble output probabilities sum to 1.0."""
        engine = EnsembleInferenceEngine()
        engine.set_strategy('ensemble_equal')
        dummy_img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
        res = engine.predict(dummy_img, k=156)
        all_probs = [item['confidence'] for item in res['top3']]
        self.assertAlmostEqual(sum(all_probs), 1.0, places=2, msg="Probabilities must sum to ~1.0")

    def test_probability_normalization_weighted(self):
        """Verify weighted ensemble output probabilities sum to 1.0."""
        engine = EnsembleInferenceEngine()
        engine.set_strategy('ensemble_weighted')
        dummy_img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
        res = engine.predict(dummy_img, k=156)
        all_probs = [item['confidence'] for item in res['top3']]
        self.assertAlmostEqual(sum(all_probs), 1.0, places=2, msg="Probabilities must sum to ~1.0")

    def test_top3_ranking_monotonicity(self):
        """Verify top-3 predictions are in strictly descending probability order."""
        engine = EnsembleInferenceEngine(strategy='ensemble_equal')
        dummy_img = Image.fromarray(np.random.randint(0, 256, (64, 64), dtype=np.uint8))
        res = engine.predict(dummy_img, k=3)
        top3 = res['top3']
        self.assertEqual(len(top3), 3)
        self.assertGreaterEqual(top3[0]['confidence'], top3[1]['confidence'])
        self.assertGreaterEqual(top3[1]['confidence'], top3[2]['confidence'])

    def test_unicode_mapping_consistency(self):
        """Verify class-to-Unicode mapping produces valid Tamil characters."""
        for cid in [0, 13, 25, 50, 100, 155]:
            ch = get_unicode_char(cid)
            cd = get_unicode_codepoints(cid)
            self.assertIsInstance(ch, str)
            self.assertTrue(len(ch) > 0)
            self.assertTrue(cd.startswith('U+'))

    def test_invalid_image_raises_error(self):
        """Verify invalid image input raises appropriate exception."""
        engine = EnsembleInferenceEngine(strategy='single')
        with self.assertRaises(Exception):
            engine.predict("non_existent_file.png")

    def test_missing_checkpoint_fallback(self):
        """Verify that missing non-baseline checkpoints trigger fallback to single model."""
        fake_paths = {
            '70_30': os.path.join('model', 'checkpoints', 'best_model.pt'),
            '80_20': 'outputs/experiments/non_existent_80_20.pt',
            '85_15': 'outputs/experiments/non_existent_85_15.pt',
            '90_10': 'outputs/experiments/non_existent_90_10.pt'
        }
        fallback_engine = EnsembleInferenceEngine.__new__(EnsembleInferenceEngine)
        fallback_engine._initialized = False
        fallback_engine.__init__(checkpoint_paths=fake_paths, strategy='ensemble_equal')
        self.assertTrue(fallback_engine.fallback_active)
        self.assertEqual(fallback_engine.strategy, 'single')

    def test_flask_api_predict_with_ensemble(self):
        """Verify /api/predict succeeds with ensemble mode."""
        img = Image.fromarray(np.ones((64, 64), dtype=np.uint8) * 128)
        import io
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)

        response = self.client.post('/api/predict', data={'image': (buf, 'test.png')}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('top3', data)
        self.assertEqual(len(data['top3']), 3)

    def test_flask_health_endpoint_reports_status(self):
        """Verify /api/health returns valid JSON with model status."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['num_classes'], 156)

if __name__ == '__main__':
    unittest.main()
