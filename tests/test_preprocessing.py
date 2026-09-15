"""
test_preprocessing.py - Unit tests for image preprocessing pipeline.
"""

import unittest
import numpy as np
import torch
from PIL import Image

from model.preprocessing import (
    preprocess_pil_image,
    preprocess_raw_dataset_image,
    image_to_tensor,
    TARGET_SIZE
)

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_raw_dataset_image(self):
        """Test conversion of raw dataset uint8 image to normalized float array."""
        raw = np.full((64, 64), 255, dtype=np.uint8)  # all white background
        raw[20:40, 20:40] = 0                          # dark stroke
        norm = preprocess_raw_dataset_image(raw)

        self.assertEqual(norm.shape, (64, 64))
        self.assertEqual(norm.dtype, np.float32)
        self.assertAlmostEqual(norm[0, 0], 0.0, places=4)   # background inverted to 0.0
        self.assertAlmostEqual(norm[30, 30], 1.0, places=4) # ink stroke inverted to 1.0

    def test_preprocess_pil_image_aspect_ratio_and_padding(self):
        """Test non-square PIL image aspect ratio preservation and centering."""
        # Create non-square rectangle image (120x60)
        img = Image.new('RGB', (120, 60), color=(255, 255, 255))
        # Draw some dark pixels in the middle
        pixels = img.load()
        for x in range(40, 80):
            for y in range(20, 40):
                pixels[x, y] = (0, 0, 0)

        norm = preprocess_pil_image(img)
        self.assertEqual(norm.shape, TARGET_SIZE)
        self.assertGreaterEqual(norm.min(), 0.0)
        self.assertLessEqual(norm.max(), 1.0)

    def test_alpha_channel_handling(self):
        """Test transparent PNG alpha flattening."""
        rgba = Image.new('RGBA', (100, 100), color=(0, 0, 0, 0))  # fully transparent
        norm = preprocess_pil_image(rgba)
        self.assertEqual(norm.shape, TARGET_SIZE)
        # Flattened transparent background should be treated as white -> normalized to 0.0
        self.assertAlmostEqual(float(norm.mean()), 0.0, places=2)

    def test_image_to_tensor(self):
        """Test tensor creation and batch/channel dimensions."""
        img = Image.new('L', (80, 80), color=255)
        tensor = image_to_tensor(img)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (1, 1, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_invalid_input_handling(self):
        """Test error raised when invalid object is passed."""
        with self.assertRaises(TypeError):
            image_to_tensor(12345)

if __name__ == '__main__':
    unittest.main()
