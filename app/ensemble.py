"""
ensemble.py - Multi-Model Ensemble Inference Engine for Tamil Handwritten OCR.

Integrates four independently trained uTHCDNet CNN models:
- 70/30 Split: Baseline Production Checkpoint (model/checkpoints/best_model.pt)
- 80/20 Split: outputs/experiments/80_20/best_model.pt
- 85/15 Split: outputs/experiments/85_15/best_model.pt
- 90/10 Split: outputs/experiments/90_10/best_model.pt

Supports three inference strategies:
- single: Uses solely the verified 70/30 production model (fastest, baseline, zero RAM overhead)
- ensemble_equal: Equal-weight probability averaging: P = (P_70 + P_80 + P_85 + P_90) / 4
- ensemble_weighted: Validation-tuned weighted fusion: P = w_70*P_70 + w_80*P_80 + w_85*P_85 + w_90*P_90

Reuses model.preprocessing.image_to_tensor and app.unicode_converter.format_prediction_result.
Includes automatic fallback to single-model mode if any ensemble checkpoint fails to load.
"""

import os
import gc
import sys
from typing import Dict, Any, Union, List, Optional
from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import uTHCDNet
from model.preprocessing import image_to_tensor
from model.unicode_mapping import NUM_CLASSES
from app.unicode_converter import format_prediction_result

# Default validation-tuned weights (determined via validation grid search)
DEFAULT_WEIGHTS = {
    '70_30': 0.20,
    '80_20': 0.20,
    '85_15': 0.30,
    '90_10': 0.30
}

class EnsembleInferenceEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EnsembleInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self,
                 checkpoint_paths: Optional[Dict[str, str]] = None,
                 strategy: str = None,
                 weights: Optional[Dict[str, float]] = None,
                 device_str: str = 'cpu'):
        if self._initialized:
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if checkpoint_paths is None:
            checkpoint_paths = {
                '70_30': os.path.join(base_dir, 'model', 'checkpoints', 'best_model.pt'),
                '80_20': os.path.join(base_dir, 'outputs', 'experiments', '80_20', 'best_model.pt'),
                '85_15': os.path.join(base_dir, 'outputs', 'experiments', '85_15', 'best_model.pt'),
                '90_10': os.path.join(base_dir, 'outputs', 'experiments', '90_10', 'best_model.pt')
            }

        self.checkpoint_paths = checkpoint_paths
        self.device = torch.device(device_str)

        # Strategy selection from environment or parameter
        if strategy is None:
            strategy = os.getenv('OCR_MODEL_MODE', 'single').lower()
        if strategy not in ['single', 'ensemble_equal', 'ensemble_weighted', 'ensemble']:
            strategy = 'single'
        if strategy == 'ensemble':
            strategy = 'ensemble_equal'
        self.strategy = strategy

        self.weights = weights or DEFAULT_WEIGHTS
        self.models = {}
        self.load_errors = {}
        self.fallback_active = False

        self._load_all_models()
        self._initialized = True

    def _load_single_checkpoint(self, path: str) -> Optional[torch.nn.Module]:
        """Loads a single model checkpoint safely into memory."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        file_size = os.path.getsize(path)
        if file_size < 1000:
            raise ValueError(f"Checkpoint at {path} is a pointer or corrupted ({file_size} bytes).")

        model = uTHCDNet(num_classes=NUM_CLASSES).to(self.device)
        try:
            ckpt = torch.load(path, map_location=self.device, mmap=True)
        except Exception:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)

        del ckpt
        gc.collect()
        model.eval()
        return model

    def _load_all_models(self):
        """Loads models according to requested strategy, with fallback protection."""
        self.models.clear()
        self.load_errors.clear()
        self.fallback_active = False

        # If single strategy, load only 70/30 baseline to conserve RAM
        if self.strategy == 'single':
            baseline_path = self.checkpoint_paths.get('70_30')
            try:
                self.models['70_30'] = self._load_single_checkpoint(baseline_path)
            except Exception as e:
                self.load_errors['70_30'] = str(e)
            return

        # Otherwise, attempt to load all 4 models
        for name, path in self.checkpoint_paths.items():
            try:
                self.models[name] = self._load_single_checkpoint(path)
            except Exception as e:
                self.load_errors[name] = str(e)

        # If any model failed to load in ensemble mode, activate fallback to 70/30 baseline
        if len(self.models) < 4:
            if '70_30' in self.models:
                print(f"[EnsembleInferenceEngine] Warning: Only {len(self.models)}/4 models loaded. Activating fallback to 70/30 single model.")
                self.fallback_active = True
                self.strategy = 'single'
            else:
                print(f"[EnsembleInferenceEngine] Critical: 70/30 baseline failed to load: {self.load_errors.get('70_30')}")

    def set_strategy(self, strategy: str):
        """Switches inference strategy dynamically."""
        if strategy == 'ensemble':
            strategy = 'ensemble_equal'
        if strategy not in ['single', 'ensemble_equal', 'ensemble_weighted']:
            raise ValueError(f"Unknown strategy: {strategy}. Expected 'single', 'ensemble_equal', or 'ensemble_weighted'.")
        self.strategy = strategy
        # Load missing models if switching from single to ensemble
        if self.strategy != 'single' and len(self.models) < 4:
            self._load_all_models()

    def get_status(self) -> Dict[str, Any]:
        """Returns comprehensive telemetry for health checks."""
        return {
            'strategy': self.strategy,
            'fallback_active': self.fallback_active,
            'models_loaded_count': len(self.models),
            'models_available': list(self.models.keys()),
            'load_errors': self.load_errors,
            'device': str(self.device),
            'weights': self.weights if self.strategy == 'ensemble_weighted' else None
        }

    def predict(self, image_input: Union[Image.Image, bytes, str], k: int = 3) -> Dict[str, Any]:
        """
        Executes prediction according to active strategy.

        Returns:
            Dict matching app.inference structure with optional ensemble diagnostic metadata.
        """
        # Ensure at least baseline is loaded
        if not self.models:
            self._load_all_models()
            if not self.models:
                raise RuntimeError(f"No models available. Errors: {self.load_errors}")

        # 1. Preprocess input image to tensor (1, 1, 64, 64)
        tensor_x = image_to_tensor(image_input).to(self.device)

        with torch.inference_mode():
            # STRATEGY A: Single model (70/30 baseline)
            if self.strategy == 'single' or len(self.models) < 4:
                model = self.models.get('70_30') or next(iter(self.models.values()))
                logits = model(tensor_x)
                probs = F.softmax(logits, dim=1).squeeze(0)
                active_mode = 'single_baseline (70/30)'
                model_diagnostics = None

            # STRATEGY B: Equal-weight probability ensemble
            elif self.strategy == 'ensemble_equal':
                probs_list = []
                diagnostics = {}
                for name in ['70_30', '80_20', '85_15', '90_10']:
                    p = F.softmax(self.models[name](tensor_x), dim=1).squeeze(0)
                    probs_list.append(p)
                    top_class = int(torch.argmax(p).item())
                    diagnostics[name] = {
                        'top_class': top_class,
                        'confidence': round(float(p[top_class].item()), 4)
                    }
                probs = torch.stack(probs_list).mean(dim=0)
                active_mode = 'ensemble_equal (4 models)'
                model_diagnostics = diagnostics

            # STRATEGY C: Validation-tuned weighted ensemble
            elif self.strategy == 'ensemble_weighted':
                total_w = sum(self.weights.values())
                norm_w = {k: v / total_w for k, v in self.weights.items()}
                weighted_probs = torch.zeros(NUM_CLASSES, device=self.device)
                diagnostics = {}
                for name in ['70_30', '80_20', '85_15', '90_10']:
                    p = F.softmax(self.models[name](tensor_x), dim=1).squeeze(0)
                    w = norm_w.get(name, 0.25)
                    weighted_probs += w * p
                    top_class = int(torch.argmax(p).item())
                    diagnostics[name] = {
                        'top_class': top_class,
                        'confidence': round(float(p[top_class].item()), 4),
                        'weight': round(w, 4)
                    }
                probs = weighted_probs
                active_mode = 'ensemble_weighted (4 models)'
                model_diagnostics = diagnostics

        # 2. Extract Top-k
        top_probs, top_indices = torch.topk(probs, k=min(k, NUM_CLASSES))
        top_indices = top_indices.cpu().numpy()
        top_probs = top_probs.cpu().numpy()

        best_class = int(top_indices[0])
        best_conf = float(top_probs[0])
        primary_result = format_prediction_result(best_class, best_conf)

        top3_list = []
        for rank in range(min(k, len(top_indices))):
            cid = int(top_indices[rank])
            conf = float(top_probs[rank])
            item = format_prediction_result(cid, conf)
            item['rank'] = rank + 1
            top3_list.append(item)

        response = {
            'success': True,
            'prediction': primary_result,
            'top3': top3_list,
            'inference_mode': active_mode
        }
        if model_diagnostics:
            response['model_diagnostics'] = model_diagnostics

        return response

def get_ensemble_engine() -> EnsembleInferenceEngine:
    return EnsembleInferenceEngine()
