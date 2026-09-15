"""
augmentation.py - Configurable data augmentation pipeline for Tamil Handwritten OCR.

Grounding & Constraints:
- Designed specifically for 64x64 grayscale isolated character tensors:
  Shape: (1, 64, 64), float32 in [0.0, 1.0], where background=0.0 and ink=1.0.
- CPU-friendly transforms (no CUDA dependency, zero RAM bloat).
- Character-identity preserving:
  * NO vertical or horizontal flips (Tamil characters are strictly non-symmetric).
  * Controlled rotations and translations reflecting natural human handwriting slant and positioning.
  * Border fill is strictly 0.0 (background).

Modes:
- 'none':   No augmentation (identity).
- 'light':  Mild rotation (+/-6 deg), translation (+/-4%), scale (0.95-1.05), shear (+/-4 deg).
- 'strong': Moderate rotation (+/-10 deg), translation (+/-7%), scale (0.90-1.10), shear (+/-8 deg),
            and slight stroke variation (random dilation/blur).
"""

import random
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class RandomStrokeVariation(object):
    """
    Simulates slight pen thickness variation (dilation or thinning) using CPU-friendly min/max pooling.
    Applied with probability p on ink=1.0 / background=0.0 tensors.
    """
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        choice = random.choice(['dilate', 'erode'])
        orig_dim = tensor.dim()
        if orig_dim == 3:
            t = tensor.unsqueeze(0)  # (1, 1, 64, 64)
        else:
            t = tensor

        if choice == 'dilate':
            # Max-pooling with 3x3 kernel expands dark ink strokes slightly
            t_padded = F.pad(t, (1, 1, 1, 1), mode='constant', value=0.0)
            dilated = F.max_pool2d(t_padded, kernel_size=3, stride=1, padding=0)
            alpha = random.uniform(0.3, 0.6)
            out = (1.0 - alpha) * t + alpha * dilated
        else:
            # Min-pooling thins stroke slightly: min(t) = -max(-t)
            t_padded = F.pad(-t, (1, 1, 1, 1), mode='constant', value=0.0)
            eroded = -F.max_pool2d(t_padded, kernel_size=3, stride=1, padding=0)
            alpha = random.uniform(0.2, 0.4)
            out = (1.0 - alpha) * t + alpha * eroded

        out = torch.clamp(out, 0.0, 1.0)
        if orig_dim == 3:
            out = out.squeeze(0)
        return out

class RandomTensorNoise(object):
    """Adds very slight paper texture Gaussian noise without obscuring the glyph."""
    def __init__(self, p: float = 0.25, std: float = 0.02):
        self.p = p
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)

def get_augmentation_transform(mode: str = 'none') -> Optional[Callable]:
    """
    Factory function returning a torchvision Compose pipeline for the requested mode.

    Args:
        mode: One of 'none', 'light', or 'strong'.

    Returns:
        Callable torchvision.transforms.Compose or None if mode is 'none'.
    """
    mode = (mode or 'none').lower().strip()

    if mode in ('none', 'off', 'identity'):
        return None

    elif mode == 'light':
        return T.Compose([
            T.RandomRotation(degrees=(-6, 6), fill=0.0),
            T.RandomAffine(
                degrees=0,
                translate=(0.04, 0.04),
                scale=(0.95, 1.05),
                shear=(-4, 4),
                fill=0.0
            )
        ])

    elif mode == 'strong':
        return T.Compose([
            T.RandomRotation(degrees=(-10, 10), fill=0.0),
            T.RandomAffine(
                degrees=0,
                translate=(0.07, 0.07),
                scale=(0.90, 1.10),
                shear=(-8, 8),
                fill=0.0
            ),
            RandomStrokeVariation(p=0.30),
            RandomTensorNoise(p=0.25, std=0.02)
        ])

    else:
        raise ValueError(f"Unknown augmentation mode '{mode}'. Expected 'none', 'light', or 'strong'.")
