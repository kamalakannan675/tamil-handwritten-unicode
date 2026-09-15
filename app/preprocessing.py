"""
app/preprocessing.py - Web application preprocessing wrapper.

Re-exports core preprocessing logic from model.preprocessing.
"""

from model.preprocessing import (
    preprocess_pil_image,
    preprocess_raw_dataset_image,
    image_to_tensor,
    TARGET_SIZE,
)

__all__ = [
    'preprocess_pil_image',
    'preprocess_raw_dataset_image',
    'image_to_tensor',
    'TARGET_SIZE',
]
