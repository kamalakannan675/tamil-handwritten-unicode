"""
inference.py - Inference Engine for Tamil Handwritten Character Recognition.

Loads the trained PyTorch CNN model once as a singleton, caches it in memory,
and executes CPU inference for image uploads or drawing canvas inputs.
Returns structured predictions with Model Confidence and Top-3 alternatives.
"""

import os
import sys
from typing import Dict, Any, Union
from PIL import Image
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import uTHCDNet
from model.preprocessing import image_to_tensor
from model.unicode_mapping import NUM_CLASSES
from app.unicode_converter import format_prediction_result

class OCRInferenceEngine:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OCRInferenceEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, checkpoint_path: str = None,
                 device_str: str = 'cpu'):
        if self._initialized:
            return

        if checkpoint_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_path = os.path.join(base_dir, 'model', 'checkpoints', 'best_model.pt')

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device_str)
        self.model = None
        self._load_model()
        self._initialized = True


    def _load_model(self):
        """Loads model weights once into memory."""
        self.model = uTHCDNet(num_classes=NUM_CLASSES).to(self.device)

        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded trained checkpoint: {self.checkpoint_path}")
        else:
            print(f"Warning: Checkpoint '{self.checkpoint_path}' not found. Initialized with fresh weights.")

        self.model.eval()

    def reload_checkpoint(self, checkpoint_path: str = None):
        """Allows hot-reloading a new checkpoint after training."""
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        self._load_model()

    def predict(self, image_input: Union[Image.Image, bytes, str]) -> Dict[str, Any]:
        """
        Runs inference on an input image.

        Args:
            image_input: PIL Image, raw image bytes, or filepath.

        Returns:
            Dict containing top prediction and top-3 alternatives.
        """
        if self.model is None:
            self._load_model()

        # 1. Preprocess into tensor (1, 1, 64, 64)
        tensor_x = image_to_tensor(image_input).to(self.device)

        # 2. Forward pass & softmax top-3
        with torch.no_grad():
            top_indices, top_probs = self.model.predict_top_k(tensor_x, k=3)

        top_indices = top_indices.cpu().numpy()[0]
        top_probs = top_probs.cpu().numpy()[0]

        # 3. Format primary prediction
        best_class = int(top_indices[0])
        best_conf = float(top_probs[0])
        primary_result = format_prediction_result(best_class, best_conf)

        # 4. Format top 3 alternatives
        top3_list = []
        for rank in range(3):
            cid = int(top_indices[rank])
            conf = float(top_probs[rank])
            item = format_prediction_result(cid, conf)
            item['rank'] = rank + 1
            top3_list.append(item)

        return {
            'success': True,
            'prediction': primary_result,
            'top3': top3_list
        }

# Global singleton accessor
def get_inference_engine() -> OCRInferenceEngine:
    return OCRInferenceEngine()
