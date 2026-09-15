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
        self.load_error = None
        self._load_model()
        self._initialized = True


    def _load_model(self):
        """Loads model weights safely into memory with memory-mapping and LFS checks."""
        import gc
        try:
            self.model = uTHCDNet(num_classes=NUM_CLASSES).to(self.device)

            if not os.path.exists(self.checkpoint_path):
                err = f"Checkpoint file '{self.checkpoint_path}' not found."
                print(f"Warning: {err}", flush=True)
                self.load_error = err
                self.model = None
                return

            file_size = os.path.getsize(self.checkpoint_path)
            print(f"Inspecting checkpoint: {self.checkpoint_path} ({file_size} bytes)", flush=True)

            # If file is < 1000 bytes, it is likely an unpulled Git LFS pointer text file
            if file_size < 1000:
                print(f"Notice: Checkpoint is {file_size} bytes (possible Git LFS pointer). Attempting git lfs pull...", flush=True)
                try:
                    import subprocess
                    res = subprocess.run(['git', 'lfs', 'pull'], capture_output=True, text=True, timeout=60)
                    print(f"git lfs pull result: {res.returncode}, stdout: {res.stdout.strip()}, stderr: {res.stderr.strip()}", flush=True)
                    file_size = os.path.getsize(self.checkpoint_path)
                except Exception as lfs_err:
                    print(f"git lfs pull attempt error: {lfs_err}", flush=True)

            if file_size < 1000:
                with open(self.checkpoint_path, 'r', errors='ignore') as f:
                    pointer_text = f.read(300).strip()
                err = f"Model checkpoint is a Git LFS pointer ({file_size} bytes), not the binary model file. Details: {pointer_text}"
                print(f"ERROR: {err}", flush=True)
                self.model = None
                self.load_error = err
                return

            # Load checkpoint using mmap=True for memory efficiency on constrained cloud instances
            print("Loading checkpoint weights with torch.load...", flush=True)
            checkpoint = None
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, mmap=True)
            except Exception as mmap_err:
                print(f"mmap load failed ({mmap_err}); falling back to standard load...", flush=True)
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                self.model.load_state_dict(checkpoint)

            # Explicitly delete checkpoint dictionary and run garbage collection to reclaim memory
            del checkpoint
            gc.collect()

            self.model.eval()
            self.load_error = None
            print(f"Successfully loaded trained checkpoint: {self.checkpoint_path}", flush=True)

        except Exception as e:
            import traceback
            trace = traceback.format_exc()
            print(f"Failed to load checkpoint: {trace}", flush=True)
            self.model = None
            self.load_error = str(e)

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
            if self.model is None:
                raise RuntimeError(self.load_error or "Model weights could not be loaded into memory.")

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
