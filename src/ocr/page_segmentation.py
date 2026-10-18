"""
page_segmentation.py - Lightweight, Zero-Dependency Full-Page Document Segmentation.

Uses pure NumPy and Pillow (no OpenCV or heavy C-dependencies) to operate within
Render's 512MB RAM and 500MB slug size limits.

Pipeline:
1. Otsu Adaptive Binarization (Foreground text = 1, Background = 0).
2. Horizontal Projection Profile Line Segmentation.
3. Vertical Projection Profile Word Segmentation with gap thresholding.
4. Hierarchical Document Tree: Page -> Lines -> Words (with bounding boxes).
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image

def otsu_threshold(gray_np: np.ndarray) -> int:
    """
    Computes optimal threshold using Otsu's method on a 2D grayscale numpy array.
    """
    hist, _ = np.histogram(gray_np, bins=256, range=(0, 256))
    total_pixels = gray_np.size
    current_max = 0.0
    threshold = 128
    
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total_pixels - w_b
        if w_f == 0:
            break
            
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        
        # Between class variance
        var_between = w_b * w_f * ((m_b - m_f) ** 2)
        if var_between > current_max:
            current_max = var_between
            threshold = t
            
    return threshold

def binarize_image(pil_img: Image.Image) -> np.ndarray:
    """
    Binarizes image such that handwritten strokes are 1 and background is 0.
    """
    gray = pil_img.convert('L')
    gray_np = np.array(gray, dtype=np.uint8)
    thresh = otsu_threshold(gray_np)
    # Dark strokes have lower intensity than bright paper background (<= thresh)
    binary = (gray_np <= thresh).astype(np.uint8)
    return binary

def moving_average(arr: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Smooths 1D projection profile array."""
    if len(arr) < window_size:
        return arr
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

def segment_lines(binary: np.ndarray,
                  min_line_height: int = 16,
                  blank_ratio: float = 0.005) -> List[Tuple[int, int]]:
    """
    Segments page into horizontal lines using projection profiles.
    
    Args:
        binary: 2D array (H, W) where 1=stroke, 0=background.
        min_line_height: Minimum height in pixels for a valid line.
        blank_ratio: Fraction of width below which a row is considered blank space.
        
    Returns:
        List of (y_top, y_bottom) tuples.
    """
    h, w = binary.shape
    # Horizontal projection: sum of stroke pixels per row
    proj = np.sum(binary, axis=1)
    smoothed = moving_average(proj, window_size=max(3, h // 100))
    threshold = w * blank_ratio
    
    lines = []
    in_line = False
    start_y = 0
    
    for y in range(h):
        val = smoothed[y]
        if val > threshold:
            if not in_line:
                in_line = True
                start_y = y
        else:
            if in_line:
                in_line = False
                if (y - start_y) >= min_line_height:
                    lines.append((start_y, y))
                    
    if in_line and (h - start_y) >= min_line_height:
        lines.append((start_y, h))
        
    return lines

def segment_words_in_line(line_binary: np.ndarray,
                          min_word_width: int = 16,
                          space_gap: int = 15) -> List[Tuple[int, int]]:
    """
    Segments a single text line strip into word boxes using vertical projection.
    
    Args:
        line_binary: 2D array of line strip (H, W).
        min_word_width: Minimum width in pixels for a valid word.
        space_gap: Consecutive blank vertical columns required to split two words.
        
    Returns:
        List of (x_left, x_right) tuples.
    """
    h, w = line_binary.shape
    proj = np.sum(line_binary, axis=0)
    smoothed = moving_average(proj, window_size=3)
    
    word_boxes = []
    in_word = False
    start_x = 0
    blank_counter = 0
    
    for x in range(w):
        if smoothed[x] > 0:
            if not in_word:
                in_word = True
                start_x = x
            blank_counter = 0
        else:
            if in_word:
                blank_counter += 1
                if blank_counter >= space_gap:
                    end_x = x - blank_counter + 1
                    if (end_x - start_x) >= min_word_width:
                        word_boxes.append((start_x, end_x))
                    in_word = False
                    blank_counter = 0
                    
    if in_word:
        end_x = w - blank_counter
        if (end_x - start_x) >= min_word_width:
            word_boxes.append((start_x, end_x))
            
    return word_boxes

class PageSegmenter:
    """
    High-level segmenter converting a full-page document into structured lines and words.
    """
    def __init__(self, min_line_height: int = 16, min_word_width: int = 16, space_gap: int = 15):
        self.min_line_height = min_line_height
        self.min_word_width = min_word_width
        self.space_gap = space_gap
        
    def process_page(self, pil_image: Image.Image) -> List[Dict[str, Any]]:
        """
        Segments a full-page PIL image into ordered lines and words.
        
        Returns:
            List of Line dictionaries, each containing:
            - 'line_idx': int
            - 'bbox': (x, y, w, h)
            - 'words': List of Word dicts with 'word_idx', 'bbox', 'crop' (PIL Image)
        """
        binary = binarize_image(pil_image)
        line_bounds = segment_lines(binary, min_line_height=self.min_line_height)
        
        result_lines = []
        for line_idx, (y1, y2) in enumerate(line_bounds):
            line_binary = binary[y1:y2, :]
            word_bounds = segment_words_in_line(
                line_binary,
                min_word_width=self.min_word_width,
                space_gap=self.space_gap
            )
            
            line_words = []
            for word_idx, (x1, x2) in enumerate(word_bounds):
                # Add slight padding around word crop for robust OCR
                pad_x = 2
                pad_y = 2
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(pil_image.width, x2 + pad_x)
                crop_y2 = min(pil_image.height, y2 + pad_y)
                
                word_crop = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                
                line_words.append({
                    'word_idx': word_idx,
                    'bbox': [int(crop_x1), int(crop_y1), int(crop_x2 - crop_x1), int(crop_y2 - crop_y1)],
                    'crop': word_crop
                })
                
            if line_words:
                result_lines.append({
                    'line_idx': line_idx,
                    'bbox': [0, int(y1), int(pil_image.width), int(y2 - y1)],
                    'words': line_words
                })
                
        return result_lines
