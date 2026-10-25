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

from typing import List, Tuple, Dict, Any, Optional
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

def detect_interline_rule_rows(binary: np.ndarray,
                               line_bounds: List[Tuple[int, int]],
                               min_run_ratio: float = 0.30) -> set:
    """
    Identifies y-coordinates of horizontal notebook ruling lines strictly
    located in the inter-line gaps (OUTSIDE detected text line bands).
    This guarantees that genuine horizontal strokes within text words
    (such as in 'எழுத்தெல்லாம்') are never misidentified as ruling lines.
    """
    h, w = binary.shape
    text_rows = set()
    for y1, y2 in line_bounds:
        text_rows.update(range(y1, y2 + 1))

    rule_threshold = int(w * min_run_ratio)
    row_sums = np.sum(binary, axis=1)

    interline_rule_rows = set()
    for y in range(h):
        if y not in text_rows and row_sums[y] >= rule_threshold:
            interline_rule_rows.add(y)

    return interline_rule_rows


def _label_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Labels 8-connected components in a 2D binary array.
    Uses scipy.ndimage.label if available; otherwise falls back to a zero-dependency BFS.
    """
    try:
        from scipy.ndimage import label as sp_label
        structure = np.ones((3, 3), dtype=int)
        labeled, num_features = sp_label(binary, structure=structure)
        return labeled, int(num_features)
    except ImportError:
        pass

    h, w = binary.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    current_label = 0
    visited = np.zeros((h, w), dtype=bool)

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    foreground_pts = np.argwhere(binary > 0)
    for pt in foreground_pts:
        y0, x0 = int(pt[0]), int(pt[1])
        if visited[y0, x0]:
            continue
        current_label += 1
        queue = [(y0, x0)]
        visited[y0, x0] = True
        head = 0
        while head < len(queue):
            cy, cx = queue[head]
            head += 1
            labeled[cy, cx] = current_label
            for dy, dx in neighbors:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and binary[ny, nx] > 0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    return labeled, current_label


def filter_boundary_artifacts(
    binary: np.ndarray,
    gray: Optional[np.ndarray] = None,
    min_artifact_area: int = 400,
    min_solidity: float = 0.40,
    max_laplacian_var: float = 40.0
) -> np.ndarray:
    """
    Conservative connected-component filtering for non-text artifacts touching outer image boundaries.

    Identifies massive, smooth finger, thumb, or shadow blobs that touch the extreme image edges
    (x=0, y=0, x=W-1, y=H-1) and masks them to background (0), while strictly preserving legitimate
    thin handwritten Tamil strokes touching or approaching page margins.

    Safety constraints:
    1. NEVER filters any component that does not touch an outer image border.
    2. NEVER filters thin handwriting strokes (area < min_artifact_area, low solidity, or high edge variance).
    3. Operates on a copy of the binary array, never modifying the input array in place.
    """
    if binary is None or binary.size == 0 or np.sum(binary) == 0:
        return binary.copy() if binary is not None else np.zeros((0, 0), dtype=np.uint8)

    h, w = binary.shape
    labeled, num_features = _label_components(binary)
    if num_features == 0:
        return binary.copy()

    cleaned = binary.copy()

    for comp_idx in range(1, num_features + 1):
        comp_mask = (labeled == comp_idx)
        area = int(np.sum(comp_mask))
        if area < min_artifact_area:
            continue

        ys, xs = np.where(comp_mask)
        y1, y2 = int(np.min(ys)), int(np.max(ys))
        x1, x2 = int(np.min(xs)), int(np.max(xs))
        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1

        touches_border = (y1 == 0 or x1 == 0 or y2 == h - 1 or x2 == w - 1)
        if not touches_border:
            continue

        bbox_area = box_w * box_h
        solidity = area / max(1, bbox_area)

        # Condition 1: Massive solid blob touching boundary (e.g. thumb / finger holding paper)
        is_massive_border_blob = (
            area >= 1000 and
            box_w >= 30 and
            box_h >= 30 and
            solidity >= min_solidity
        )

        # Condition 2: Smooth, inkless shadow gradient touching boundary
        is_low_contrast_shadow = False
        if gray is not None and area >= min_artifact_area and (box_w >= 20 or box_h >= 20):
            crop_gray = gray[y1:y2+1, x1:x2+1].astype(np.float32)
            if crop_gray.shape[0] >= 3 and crop_gray.shape[1] >= 3:
                p = np.pad(crop_gray, 1, mode='edge')
                lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1])
                lap_var = float(np.var(lap))
                if lap_var < max_laplacian_var:
                    is_low_contrast_shadow = True

        if is_massive_border_blob or is_low_contrast_shadow:
            cleaned[comp_mask] = 0

    return cleaned

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

def compute_bounded_line_crop_y(line_bounds: List[Tuple[int, int]],
                                line_idx: int,
                                img_height: int,
                                rule_rows: set = None) -> Tuple[int, int]:
    """
    Computes vertically bounded crop coordinates (crop_y1, crop_y2) for line at line_idx.

    Guarantees:
    1. Zero overlap with adjacent line crops (crop_y2[i] <= crop_y1[i+1]).
    2. Preserves Tamil ascenders, pulli dots, and descenders up to available inter-line gap.
    3. First line (top margin) and last line (bottom margin) are safely bounded within [0, img_height].
    4. Degenerate, zero-gap, or inverted line bounds never cause negative or zero-height crops.
    """
    if not line_bounds or line_idx < 0 or line_idx >= len(line_bounds):
        return 0, max(1, img_height)

    y1, y2 = line_bounds[line_idx]
    y1 = max(0, min(img_height, y1))
    y2 = max(y1, min(img_height, y2))
    line_h = max(1, y2 - y1)

    desired_pad_y = max(2, int(round(line_h * 0.18)))

    # Distance to preceding line (or top of image)
    if line_idx > 0:
        prev_y2 = max(0, min(img_height, line_bounds[line_idx - 1][1]))
        gap_above = max(0, y1 - prev_y2)
        pad_top = min(desired_pad_y, gap_above // 2)
    else:
        gap_above = y1
        pad_top = min(desired_pad_y, gap_above)

    # Distance to succeeding line (or bottom of image)
    if line_idx < len(line_bounds) - 1:
        next_y1 = max(0, min(img_height, line_bounds[line_idx + 1][0]))
        gap_below = max(0, next_y1 - y2)
        pad_bot = min(desired_pad_y, gap_below // 2)
    else:
        gap_below = max(0, img_height - y2)
        pad_bot = min(desired_pad_y, gap_below)

    crop_y1 = max(0, y1 - pad_top)
    crop_y2 = min(img_height, y2 + pad_bot)

    # Ruling-line awareness: clamp padding if a rule row exists in the gap
    if rule_rows:
        # Check gap above (crop_y1 to y1)
        for y in range(y1 - 1, crop_y1 - 1, -1):
            if y in rule_rows:
                crop_y1 = y + 1  # Clamp immediately below the ruling row
                break

        # Check gap below (y2 to crop_y2)
        for y in range(y2 + 1, crop_y2 + 1):
            if y in rule_rows:
                crop_y2 = y - 1  # Clamp immediately above the ruling row
                break

    if crop_y2 <= crop_y1:
        crop_y2 = min(img_height, crop_y1 + 1)

    return crop_y1, crop_y2

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
    # True inter-word spaces are typically >= 0.16 - 0.25 of line height (min 7px).
    effective_space_gap = space_gap if space_gap is not None else max(7, int(round(line_h * 0.18)))

    # Minimum width for a standalone word vs a tiny sub-character fragment (e.g. pulli dot, detached kombu)
    # A standalone base consonant or consonant-vowel (e.g. நீ, தீ, வா) is >= 0.20 * line_h (min 12px)
    effective_min_word = min_word_width if min_word_width is not None else max(12, int(round(line_h * 0.22)))

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
    # to guarantee combining vowel marks / pulli dots are never isolated,
    # while preserving words that have width >= effective_min_word and clear whitespace separation.
    merged_words = []
    for (sx, ex) in segments:
        seg_w = ex - sx
        if not merged_words:
            merged_words.append([sx, ex])
        else:
            prev_sx, prev_ex = merged_words[-1]
            prev_w = prev_ex - prev_sx
            gap_to_prev = sx - prev_ex

            # Merge if gap is smaller than word gap threshold, OR if either segment is a sub-character fragment
            if gap_to_prev < effective_space_gap or seg_w < effective_min_word or prev_w < effective_min_word:
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
    def __init__(self, min_line_height: int = 12, min_word_width: int = 16, space_gap: int = 16, filter_boundaries: bool = True):
        self.filter_boundaries = filter_boundaries
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

        if self.filter_boundaries:
            gray_np = np.array(pil_image.convert('L'), dtype=np.uint8)
            binary = filter_boundary_artifacts(binary, gray=gray_np)

        line_bounds = segment_lines(binary, min_line_height=self.min_line_height)
        rule_rows = detect_interline_rule_rows(raw_binary, line_bounds)

        result_lines = []
        for line_idx, (y1, y2) in enumerate(line_bounds):
            line_binary = binary[y1:y2, :]
            line_h = y2 - y1

            # Dynamic adaptive space gap and minimum word width based on line height
            adaptive_gap = max(7, int(round(line_h * 0.18)))
            adaptive_word_w = max(12, int(round(line_h * 0.22)))

            word_bounds = segment_words_in_line(
                line_binary,
                line_h=line_h,
                min_word_width=adaptive_word_w,
                space_gap=adaptive_gap
            )

            # Bounded vertical crop coordinates avoiding adjacent line stroke intrusion
            crop_y1, crop_y2 = compute_bounded_line_crop_y(line_bounds, line_idx, pil_image.height, rule_rows=rule_rows)

            line_words = []
            for word_idx, (x1, x2) in enumerate(word_bounds):
                pad_x = max(3, int(round(line_h * 0.10)))

                crop_x1 = max(0, x1 - pad_x)
                crop_x2 = min(pil_image.width, x2 + pad_x)

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

    def assess_image_quality(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Lightweight pre-flight image quality and line-height assessment.
        Estimates the median height of detected text lines using the existing
        line-segmentation pipeline without running expensive word segmentation
        or CRNN neural network inference.

        Returns:
            Dict containing:
            - 'line_count': int
            - 'line_heights': List[int]
            - 'median_line_height': float
            - 'quality_status': 'good' | 'caution_low_res' | 'severe_low_res' | 'no_text_detected'
            - 'warning': Optional[str]
            - 'can_proceed': bool (Always True; never rejects images)
        """
        raw_binary = binarize_image(pil_image)
        binary = filter_horizontal_rules(raw_binary)
        if getattr(self, 'filter_boundaries', True):
            gray_np = np.array(pil_image.convert('L'), dtype=np.uint8)
            binary = filter_boundary_artifacts(binary, gray=gray_np)

        # Check if the image contains negligible foreground pixels (e.g. blank page or solid background)
        total_foreground = int(np.sum(binary))
        if total_foreground < 25:
            return {
                'line_count': 0,
                'line_heights': [],
                'median_line_height': 0.0,
                'quality_status': 'no_text_detected',
                'warning': (
                    "No clear text lines detected. Please ensure the document is clearly visible, "
                    "oriented right-side up, and well-lit. You can still continue with this image."
                ),
                'can_proceed': True
            }

        # Use a slightly lower line-height threshold to capture very small text lines (down to 8px)
        line_bounds = segment_lines(binary, min_line_height=max(8, self.min_line_height // 2))

        # Filter out potential spurious full-image boundary fallback if foreground is sparse
        valid_lines = []
        for (y1, y2) in line_bounds:
            line_fg = np.sum(binary[y1:y2, :])
            if line_fg >= 15: # At least minimal stroke activity
                valid_lines.append((y1, y2))

        if not valid_lines:
            return {
                'line_count': 0,
                'line_heights': [],
                'median_line_height': 0.0,
                'quality_status': 'no_text_detected',
                'warning': (
                    "No clear text lines detected. Please ensure the document is clearly visible, "
                    "oriented right-side up, and well-lit. You can still continue with this image."
                ),
                'can_proceed': True
            }

        line_heights = [int(y2 - y1) for (y1, y2) in valid_lines]
        median_h = float(np.median(line_heights))

        if median_h < 20.0:
            status = 'severe_low_res'
            warning = (
                f"Severe low-resolution handwriting detected (estimated median line height: {int(round(median_h))} px). "
                "Text strokes are extremely small, which may cause Tamil characters or vowel marks to be misrecognized. "
                "For better results, take a closer, well-lit photograph and keep the text in focus. You can still continue with this image."
            )
        elif 20.0 <= median_h < 30.0:
            status = 'caution_low_res'
            warning = (
                f"Low-resolution handwriting detected (estimated median line height: {int(round(median_h))} px). "
                "Some text lines are very small, so Tamil characters or vowel marks may be misrecognized. "
                "For better results, take a closer, well-lit photograph and keep the text in focus. You can still continue with this image."
            )
        else:
            status = 'good'
            warning = None

        return {
            'line_count': len(valid_lines),
            'line_heights': line_heights,
            'median_line_height': round(median_h, 1),
            'quality_status': status,
            'warning': warning,
            'can_proceed': True
        }
