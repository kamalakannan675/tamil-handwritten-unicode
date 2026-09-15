"""
test_augmentation.py - Unit tests for Tamil OCR data augmentation pipeline.
"""

import unittest
import torch
from model.augmentation import (
    get_augmentation_transform,
    RandomStrokeVariation,
    RandomTensorNoise
)
from model.dataset import uTHCDDataset

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        # Create a dummy character tensor (1, 64, 64) with background 0.0 and a central square stroke
        self.dummy_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
        self.dummy_tensor[0, 20:44, 20:44] = 1.0

    def test_get_augmentation_transform_none(self):
        """Mode 'none' must return None (identity)."""
        for m in ['none', 'None', 'off', 'identity']:
            self.assertIsNone(get_augmentation_transform(m))

    def test_get_augmentation_transform_light(self):
        """Mode 'light' must return callable transform."""
        t = get_augmentation_transform('light')
        self.assertTrue(callable(t))

    def test_get_augmentation_transform_strong(self):
        """Mode 'strong' must return callable transform."""
        t = get_augmentation_transform('strong')
        self.assertTrue(callable(t))

    def test_invalid_mode_raises_error(self):
        """Unknown mode should raise ValueError."""
        with self.assertRaises(ValueError):
            get_augmentation_transform('invalid_mode_xyz')

    def test_light_augmentation_preserves_shape_and_range(self):
        """Light transform must maintain (1, 64, 64) shape and [0.0, 1.0] bounds."""
        transform = get_augmentation_transform('light')
        for _ in range(5):
            out = transform(self.dummy_tensor)
            self.assertEqual(out.shape, (1, 64, 64))
            self.assertEqual(out.dtype, torch.float32)
            self.assertGreaterEqual(out.min().item(), 0.0)
            self.assertLessEqual(out.max().item(), 1.0)

    def test_strong_augmentation_preserves_shape_and_range(self):
        """Strong transform must maintain (1, 64, 64) shape and [0.0, 1.0] bounds."""
        transform = get_augmentation_transform('strong')
        for _ in range(5):
            out = transform(self.dummy_tensor)
            self.assertEqual(out.shape, (1, 64, 64))
            self.assertEqual(out.dtype, torch.float32)
            self.assertGreaterEqual(out.min().item(), 0.0)
            self.assertLessEqual(out.max().item(), 1.0)

    def test_stroke_variation_transform(self):
        """RandomStrokeVariation must preserve tensor shape and bounds."""
        stroke_var = RandomStrokeVariation(p=1.0)
        for _ in range(5):
            out = stroke_var(self.dummy_tensor)
            self.assertEqual(out.shape, (1, 64, 64))
            self.assertGreaterEqual(out.min().item(), 0.0)
            self.assertLessEqual(out.max().item(), 1.0)

    def test_noise_transform(self):
        """RandomTensorNoise must preserve tensor shape and bounds."""
        noise_var = RandomTensorNoise(p=1.0, std=0.03)
        out = noise_var(self.dummy_tensor)
        self.assertEqual(out.shape, (1, 64, 64))
        self.assertGreaterEqual(out.min().item(), 0.0)
        self.assertLessEqual(out.max().item(), 1.0)

    def test_unaugmented_val_and_test_datasets(self):
        """Validation and test datasets must default to transform=None."""
        val_ds = uTHCDDataset(split='val', max_samples=10)
        test_ds = uTHCDDataset(split='test', max_samples=10)
        self.assertIsNone(val_ds.transform)
        self.assertIsNone(test_ds.transform)

if __name__ == '__main__':
    unittest.main()
