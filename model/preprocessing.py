"""
preprocessing.py - Image preprocessing pipeline for Tamil Handwritten Character Recognition.

Includes:
1. Image loading & decoding
2. RGBA to RGB alpha-channel handling against a white background
3. Grayscale conversion
4. Ink/Foreground bounding box detection & tight cropping
5. Aspect-ratio preserving resize with square padding to (64, 64)
6. Inversion & Normalization (Background 255 -> 0.0, Stroke 0 -> 1.0)
7. PyTorch tensor conversion (1, 64, 64)
"""

import io
from typing import Union
import numpy as np
from PIL import Image, ImageOps
import torch

TARGET_SIZE = (64, 64)

def preprocess_pil_image(image: Image.Image, target_size=TARGET_SIZE) -> np.ndarray:
    """
    Process a PIL Image into a normalized (64, 64) float32 numpy array.
    Background is 0.0, ink strokes are positive (up to 1.0).
    """
    # 1. Handle transparency if present (RGBA / LA / P with alpha)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        # Create solid white canvas
        white_bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        # Composite image over white canvas
        composite = Image.alpha_composite(white_bg, rgba)
        gray = composite.convert('L')
    else:
        gray = image.convert('L')

    # Convert to numpy array for foreground analysis
    arr = np.array(gray, dtype=np.uint8)

    # 2. Invert check: in uTHCD, background is white (~255) and ink is dark (~0).
    # If a user drew white strokes on a black background (e.g. inverted canvas), detect and fix it.
    # Determine corner pixel brightness
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    corner_mean = float(np.mean(corners))
    if corner_mean < 128:
        # User supplied dark background with bright strokes -> invert so background is white
        arr = 255 - arr
        gray = Image.fromarray(arr)

    # 3. Foreground detection: find ink bounding box (pixels significantly darker than background)
    # Threshold for dark ink (typically < 220 in 0-255 scale)
    ink_mask = arr < 220
    if np.any(ink_mask):
        rows = np.any(ink_mask, axis=1)
        cols = np.any(ink_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add small margin/padding around bounding box (5% of box dimension)
        h, w = rmax - rmin + 1, cmax - cmin + 1
        pad_y = max(2, int(h * 0.08))
        pad_x = max(2, int(w * 0.08))

        rmin = max(0, rmin - pad_y)
        rmax = min(arr.shape[0] - 1, rmax + pad_y)
        cmin = max(0, cmin - pad_x)
        cmax = min(arr.shape[1] - 1, cmax + pad_x)

        cropped = gray.crop((cmin, rmin, cmax + 1, rmax + 1))
    else:
        # Blank or entirely white image
        cropped = gray

    # 4. Aspect-ratio preserving resize to fit within target_size with margin
    target_w, target_h = target_size
    margin = 8  # 4 pixels border margin inside 64x64
    inner_w = target_w - margin
    inner_h = target_h - margin

    c_w, c_h = cropped.size
    ratio = min(inner_w / max(1, c_w), inner_h / max(1, c_h))
    new_w = max(1, int(c_w * ratio))
    new_h = max(1, int(c_h * ratio))

    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 5. Paste centered onto a 64x64 white background (255)
    canvas = Image.new('L', target_size, 255)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    # 6. Normalize: Invert so background is 0.0 and ink stroke is positive (up to 1.0)
    norm_arr = (255.0 - np.array(canvas, dtype=np.float32)) / 255.0

    return norm_arr

def preprocess_raw_dataset_image(arr_uint8: np.ndarray) -> np.ndarray:
    """
    Directly preprocess an image from the uTHCD HDF5 dataset.
    Input: uint8 array of shape (64, 64), where 255 is background and 0 is ink.
    Output: float32 array of shape (64, 64) normalized so background is 0.0 and ink is 1.0.
    """
    if arr_uint8.dtype != np.uint8:
        arr_uint8 = arr_uint8.astype(np.uint8)
    return (255.0 - arr_uint8.astype(np.float32)) / 255.0

def image_to_tensor(image_input: Union[Image.Image, bytes, str, np.ndarray]) -> torch.Tensor:
    """
    Converts any input (PIL Image, bytes, filepath, or numpy array)
    into a PyTorch tensor with shape (1, 1, 64, 64), float32.
    """
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
        norm = preprocess_pil_image(pil_img)
    elif isinstance(image_input, str):
        pil_img = Image.open(image_input)
        norm = preprocess_pil_image(pil_img)
    elif isinstance(image_input, Image.Image):
        norm = preprocess_pil_image(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.shape == (64, 64) and image_input.dtype == np.uint8:
            norm = preprocess_raw_dataset_image(image_input)
        else:
            pil_img = Image.fromarray(image_input)
            norm = preprocess_pil_image(pil_img)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Add channel and batch dimension: (1, 1, 64, 64)
    tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(torch.float32)
    return tensor
