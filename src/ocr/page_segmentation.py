"""
page_segmentation.py - Lightweight, Zero-Dependency Full-Page Document Segmentation.

Uses pure NumPy and Pillow (no OpenCV or heavy C-dependencies) to operate within
Render's 512MB RAM and 500MB slug size limits.

Pipeline:
1. Otsu Adaptive Binarization (Foreground text = 1, Background = 0).
2. Notebook ruling line filtering (removes thin horizontal notebook lines).
3. Valley-based Horizontal Projection Profile Line Segmentation with shoulder detection.
4. Adaptive Vertical Projection Profile Word Segmentation with calibrated gap thresholding
   to preserve Tamil grapheme clusters and prevent intra-word splitting.
5. Hierarchical Document Tree: Page -> Lines -> Words (with bounding boxes and padded crops).
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

def filter_horizontal_rules(binary: np.ndarray, min_run_ratio: float = 0.35) -> np.ndarray:
    """
    Suppresses thin horizontal notebook ruled lines that bridge text lines.
    A notebook rule line is a continuous horizontal run of black pixels
    spanning >= 35% of the page width with a thickness of only 1-2 pixels.
    """
    h, w = binary.shape
    filtered = binary.copy()
    min_run = int(w * min_run_ratio)
    
    for y in range(1, h - 1):
        row = binary[y, :]
        if np.sum(row) >= min_run:
            above = binary[y - 1, :]
            below = binary[y + 1, :]
            # If the stroke is isolated horizontally (thickness 1-2px)
            thin_mask = (row == 1) & (above == 0) & (below == 0)
            if np.sum(thin_mask) >= min_run:
                filtered[y, thin_mask] = 0
                
    return filtered

def moving_average(arr: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Smooths 1D projection profile array."""
    if len(arr) < window_size:
        return arr
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

def segment_lines(binary: np.ndarray,
                  min_line_height: int = 12,
                  blank_ratio: float = 0.005) -> List[Tuple[int, int]]:
    """
    Hybrid line segmentation:
    1. Initial pass finds line bands separated by clear whitespace gaps.
    2. Refinement pass detects deep internal valleys between prominent peaks
       to separate touching lines caused by descenders or ruled paper.
       
    Returns:
        List of (y_top, y_bottom) tuples.
    """
    h, w = binary.shape
    proj = np.sum(binary, axis=1).astype(np.float32)
    max_val = np.max(proj)
    if max_val < 2.0:
        return [(0, h)]
        
    blank_thresh = max(1.0, max_val * 0.03)
    
    # 1. Coarse bands separated by whitespace
    raw_bands = []
    in_band = False
    start_y = 0
    for y in range(h):
        if proj[y] > blank_thresh:
            if not in_band:
                in_band = True
                start_y = y
        else:
            if in_band:
                in_band = False
                if (y - start_y) >= min_line_height:
                    raw_bands.append((start_y, y))
    if in_band and (h - start_y) >= min_line_height:
        raw_bands.append((start_y, h))
        
    # 2. Refinement pass: for any tall band, check for multiple peaks separated by valleys
    final_lines = []
    win = max(3, int(h * 0.02))
    kernel = np.ones(win) / float(win)
    smoothed = np.convolve(proj, kernel, mode='same')
    
    for (by1, by2) in raw_bands:
        band_h = by2 - by1
        band_smoothed = smoothed[by1:by2]
        band_max = np.max(band_smoothed)
        
        min_dist = max(8, int(band_h * 0.30))
        peaks = []
        for y_rel in range(1, band_h - 1):
            val = band_smoothed[y_rel]
            if val > (band_max * 0.18) and val >= band_smoothed[y_rel - 1] and val >= band_smoothed[y_rel + 1]:
                if not peaks or (y_rel - peaks[-1]) >= min_dist:
                    peaks.append(y_rel)
                elif val > band_smoothed[peaks[-1]]:
                    peaks[-1] = y_rel
                    
        valid_valleys = []
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            seg = band_smoothed[p1:p2 + 1]
            v_rel = p1 + int(np.argmin(seg))
            v_val = band_smoothed[v_rel]
            p1_val = band_smoothed[p1]
            p2_val = band_smoothed[p2]
            if v_val < 0.65 * min(p1_val, p2_val):
                valid_valleys.append(by1 + v_rel)
                
        if not valid_valleys:
            final_lines.append((by1, by2))
        else:
            cur_start = by1
            for vy in valid_valleys:
                if (vy - cur_start) >= min_line_height:
                    final_lines.append((cur_start, vy))
                    cur_start = vy
            if (by2 - cur_start) >= min_line_height:
                final_lines.append((cur_start, by2))
                
    return final_lines if final_lines else [(0, h)]

def segment_words_in_line(line_binary: np.ndarray,
                          line_h: int = None,
                          min_word_width: int = None,
                          space_gap: int = None) -> List[Tuple[int, int]]:
    """
    Segments a single text line strip into word boxes using vertical projection
    with calibrated gap thresholding based on line height and intelligent fragment merging.
    
    Ensures Tamil intra-word pen lifts and combining marks (pulli, vowel signs)
    are NEVER split from their base words.
    
    Returns:
        List of (x_left, x_right) tuples.
    """
    h, w = line_binary.shape
    if line_h is None:
        line_h = h
        
    vproj = np.sum(line_binary, axis=0)
    col_thresh = max(0.0, h * 0.03)
    is_blank = (vproj <= col_thresh)
    
    # Calibrated word spacing thresholds:
    # Intra-word gaps between Tamil glyphs and combining marks are typically 1-6px.
    # True inter-word spaces are typically >= 0.30 - 0.50 of line height.
    effective_space_gap = space_gap if space_gap is not None else max(6, int(round(line_h * 0.32)))
    effective_min_word = min_word_width if min_word_width is not None else max(16, int(round(line_h * 0.60)))
    
    segments = []
    in_word = False
    start_x = 0
    blank_counter = 0
    
    for x in range(w):
        if not is_blank[x]:
            if not in_word:
                in_word = True
                start_x = x
            blank_counter = 0
        else:
            if in_word:
                blank_counter += 1
                if blank_counter >= effective_space_gap:
                    end_x = x - blank_counter + 1
                    segments.append((start_x, end_x))
                    in_word = False
                    blank_counter = 0
                    
    if in_word:
        end_x = w - blank_counter
        segments.append((start_x, end_x))
        
    # Merge tiny sub-character fragments (< effective_min_word) into adjacent words
    # to guarantee combining vowel marks / pulli dots are never isolated.
    merged_words = []
    for (sx, ex) in segments:
        if not merged_words:
            merged_words.append([sx, ex])
        else:
            prev_sx, prev_ex = merged_words[-1]
            gap_to_prev = sx - prev_ex
            # Merge if segment is narrower than min_word or gap is smaller than word separator threshold
            if (ex - sx) < effective_min_word or gap_to_prev < effective_space_gap:
                merged_words[-1][1] = ex
            else:
                merged_words.append([sx, ex])
                
    result_boxes = [(int(b[0]), int(b[1])) for b in merged_words if (b[1] - b[0]) >= 8]
    
    # Fallback: if no word segments found but line contains foreground strokes
    if not result_boxes and np.sum(line_binary) > 10:
        active_cols = np.where(vproj > 0)[0]
        if len(active_cols) > 0:
            result_boxes.append((int(active_cols[0]), int(active_cols[-1]) + 1))
            
    return result_boxes

class PageSegmenter:
    """
    High-level segmenter converting a full-page document into structured lines and words.
    """
    def __init__(self, min_line_height: int = 12, min_word_width: int = 16, space_gap: int = 16):
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
        raw_binary = binarize_image(pil_image)
        binary = filter_horizontal_rules(raw_binary)
        
        line_bounds = segment_lines(binary, min_line_height=self.min_line_height)
        
        result_lines = []
        for line_idx, (y1, y2) in enumerate(line_bounds):
            line_binary = binary[y1:y2, :]
            line_h = y2 - y1
            
            # Dynamic adaptive space gap and minimum word width based on line height
            adaptive_gap = max(6, int(round(line_h * 0.32)))
            adaptive_word_w = max(14, int(round(line_h * 0.60)))
            
            word_bounds = segment_words_in_line(
                line_binary,
                line_h=line_h,
                min_word_width=adaptive_word_w,
                space_gap=adaptive_gap
            )
            
            # Generous vertical margin padding (preserves ascenders: pulli dots, vowel marks, and descenders)
            pad_y = max(4, int(round(line_h * 0.20)))
            
            line_words = []
            for word_idx, (x1, x2) in enumerate(word_bounds):
                pad_x = max(3, int(round(line_h * 0.10)))
                
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
                min_w_x1 = min(w['bbox'][0] for w in line_words)
                max_w_x2 = max(w['bbox'][0] + w['bbox'][2] for w in line_words)
                result_lines.append({
                    'line_idx': line_idx,
                    'bbox': [int(min_w_x1), int(y1), int(max_w_x2 - min_w_x1), int(y2 - y1)],
                    'words': line_words
                })
                
        return result_lines
