"""
test_boundary_filtering.py - Unit tests for conservative boundary-artifact filtering.

Verifies:
1. Large, smooth artificial component touching top-left boundary is filtered out.
2. Shadow-like component at the left boundary is filtered out when edge contrast is low.
3. Genuine thin handwriting touching the left boundary is preserved.
4. Genuine handwriting touching the right boundary is preserved.
5. Genuine handwriting touching the top or bottom boundary is preserved.
6. Ordinary images without boundary artifacts retain their expected segmentation.
7. Filtering does not modify the original input image array in place.
8. Empty images and degenerate components do not crash the pipeline.
"""

import unittest
import numpy as np
from PIL import Image, ImageDraw

from src.ocr.page_segmentation import (
    filter_boundary_artifacts,
    _label_components,
    PageSegmenter,
    segment_lines
)

class TestBoundaryFiltering(unittest.TestCase):
    def test_01_large_smooth_component_at_top_left_is_removed(self):
        """1. A large, smooth artificial component touching the top-left boundary is removed."""
        h, w = 200, 300
        binary = np.zeros((h, w), dtype=np.uint8)
        gray = np.full((h, w), 255, dtype=np.uint8)

        # Artificial thumb block at top-left: y in [0, 60], x in [0, 80], area = 4800px
        binary[0:60, 0:80] = 1
        gray[0:60, 0:80] = 140 # smooth gray tone, low variance

        # Legitimate text stroke in middle: y in [100, 120], x in [50, 150]
        binary[100:104, 50:150] = 1
        gray[100:104, 50:150] = 20

        cleaned = filter_boundary_artifacts(binary, gray=gray)

        # The thumb at [0:60, 0:80] must be removed
        self.assertEqual(np.sum(cleaned[0:60, 0:80]), 0)
        # The text stroke in the middle must be completely preserved
        self.assertEqual(np.sum(cleaned[100:104, 50:150]), np.sum(binary[100:104, 50:150]))

    def test_02_shadow_like_component_at_left_boundary_is_removed(self):
        """2. A low-contrast shadow-like component at the left boundary is handled and removed."""
        h, w = 150, 200
        binary = np.zeros((h, w), dtype=np.uint8)
        gray = np.full((h, w), 240, dtype=np.uint8)

        # Shadow region touching left edge: x in [0, 40], y in [20, 80]
        binary[20:80, 0:40] = 1
        # Gradient shadow with almost zero Laplacian variance
        for x in range(40):
            gray[20:80, x] = int(180 + x * 0.5)

        cleaned = filter_boundary_artifacts(binary, gray=gray, max_laplacian_var=40.0)
        self.assertEqual(np.sum(cleaned[20:80, 0:40]), 0)

    def test_03_genuine_thin_handwriting_touching_left_boundary_is_preserved(self):
        """3. Genuine thin handwriting touching the left boundary is strictly preserved."""
        h, w = 150, 200
        binary = np.zeros((h, w), dtype=np.uint8)
        gray = np.full((h, w), 255, dtype=np.uint8)

        # Thin letter stroke (e.g. 'ர' or 'க') touching left edge: x in [0, 15], width=3px stroke
        binary[40:70, 0:3] = 1   # vertical stem touching x=0
        binary[40:43, 0:15] = 1  # horizontal bar
        gray[40:70, 0:3] = 10
        gray[40:43, 0:15] = 10

        cleaned = filter_boundary_artifacts(binary, gray=gray)
        # Must be preserved because area is small (<400) and stroke is thin (low solidity)
        self.assertEqual(np.sum(cleaned), np.sum(binary))

    def test_04_genuine_handwriting_touching_right_boundary_is_preserved(self):
        """4. Genuine handwriting touching the right boundary is preserved."""
        h, w = 150, 200
        binary = np.zeros((h, w), dtype=np.uint8)
        gray = np.full((h, w), 255, dtype=np.uint8)

        # Trailing punctuation or flourish touching right edge x=w-1
        binary[50:54, 185:w] = 1
        gray[50:54, 185:w] = 15

        cleaned = filter_boundary_artifacts(binary, gray=gray)
        self.assertEqual(np.sum(cleaned), np.sum(binary))

    def test_05_genuine_handwriting_touching_top_or_bottom_is_preserved(self):
        """5. Genuine handwriting touching the top or bottom boundary (ascender/descender) is preserved."""
        h, w = 150, 200
        binary = np.zeros((h, w), dtype=np.uint8)
        gray = np.full((h, w), 255, dtype=np.uint8)

        # Ascender loop touching y=0 (thin stroke)
        binary[0:15, 60:63] = 1
        # Descender loop touching y=h-1 (thin stroke)
        binary[h-20:h, 120:123] = 1

        cleaned = filter_boundary_artifacts(binary, gray=gray)
        self.assertEqual(np.sum(cleaned), np.sum(binary))

    def test_06_ordinary_image_without_boundary_artifacts_retains_segmentation(self):
        """6. An ordinary image without boundary artifacts retains its expected segmentation."""
        im = Image.new("RGB", (300, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(im)
        # Draw 2 clean text lines in middle of page
        draw.rectangle([(30, 40), (200, 65)], fill=(0, 0, 0))
        draw.rectangle([(30, 110), (220, 135)], fill=(0, 0, 0))

        segmenter = PageSegmenter(min_line_height=12, filter_boundaries=True)
        res = segmenter.process_page(im)

        self.assertEqual(len(res), 2, "Should retain exactly 2 lines for clean document")
        self.assertAlmostEqual(res[0]['bbox'][1], 40, delta=5)
        self.assertAlmostEqual(res[1]['bbox'][1], 110, delta=5)

    def test_07_filtering_does_not_modify_original_binary_in_place(self):
        """7. Filtering does not modify the original input image array."""
        h, w = 100, 100
        binary = np.zeros((h, w), dtype=np.uint8)
        binary[0:50, 0:50] = 1 # Massive border block

        binary_copy = binary.copy()
        cleaned = filter_boundary_artifacts(binary)

        # Original binary must remain unchanged
        np.testing.assert_array_equal(binary, binary_copy)
        self.assertNotEqual(np.sum(cleaned), np.sum(binary))

    def test_08_empty_and_degenerate_inputs_handled_safely(self):
        """8. Empty images and degenerate components do not crash the pipeline."""
        # None input
        self.assertEqual(filter_boundary_artifacts(None).size, 0)
        # Empty array
        empty = np.zeros((0, 0), dtype=np.uint8)
        self.assertEqual(filter_boundary_artifacts(empty).shape, (0, 0))
        # All zeros
        all_zeros = np.zeros((50, 50), dtype=np.uint8)
        res = filter_boundary_artifacts(all_zeros)
        self.assertEqual(np.sum(res), 0)
        # Single pixel
        single = np.ones((1, 1), dtype=np.uint8)
        res_single = filter_boundary_artifacts(single)
        self.assertEqual(res_single.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()
