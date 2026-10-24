"""
test_bounded_padding.py - Targeted Regression Tests for Bounded Inter-Line Vertical Padding.

Verifies:
1. Adjacent line crops do not overlap (crop_y2[i] <= crop_y1[i+1]).
2. Tight line spacing (gaps of 0, 1, 2 px, and negative gaps) is handled safely.
3. The first and last lines remain strictly within image bounds [0, img_height].
4. Empty or unusual line-boundary input does not crash or raise unhandled exceptions.
5. Existing Tamil diacritics and descenders are preserved when space permits.
6. The real notebook regression image does not introduce Line 1 pixels into Line 0 crops.
7. End-to-end PageSegmenter produces non-overlapping word crop bounding boxes.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.page_segmentation import (
    compute_bounded_line_crop_y,
    detect_interline_rule_rows,
    PageSegmenter,
    binarize_image,
    filter_horizontal_rules,
    segment_lines
)

class TestBoundedInterLinePadding(unittest.TestCase):
    def test_01_adjacent_line_crops_never_overlap(self):
        """1. Mathematically guarantees crop_y2[i] <= crop_y1[i+1] across diverse gap sizes."""
        img_h = 500
        test_cases = [
            [(30, 60), (70, 100), (120, 150)],       # Gaps: 10, 20
            [(40, 80), (82, 120), (124, 160)],       # Tight gaps: 2, 4
            [(50, 100), (101, 150), (152, 200)],     # Very tight gaps: 1, 1
            [(100, 200), (200, 300), (300, 400)],    # Zero gaps: 0, 0 (touching)
            [(20, 120), (220, 320)]                  # Large gaps: 100
        ]
        
        for lines in test_cases:
            crops = [compute_bounded_line_crop_y(lines, idx, img_h) for idx in range(len(lines))]
            for i in range(len(crops) - 1):
                cur_y2 = crops[i][1]
                next_y1 = crops[i + 1][0]
                self.assertLessEqual(
                    cur_y2, next_y1,
                    f"Adjacent line overlap detected in {lines}: Line {i} y2={cur_y2} > Line {i+1} y1={next_y1}"
                )

    def test_02_tight_line_spacing_handled_safely(self):
        """2. Tight spacing (gap=0, 1, 2) and overlapping inputs do not produce inverted crops."""
        img_h = 300
        # Touching lines (gap=0)
        touching_lines = [(50, 100), (100, 150)]
        c0 = compute_bounded_line_crop_y(touching_lines, 0, img_h)
        c1 = compute_bounded_line_crop_y(touching_lines, 1, img_h)
        self.assertLessEqual(c0[1], c1[0])
        self.assertGreaterEqual(c0[1], c0[0])
        self.assertGreaterEqual(c1[1], c1[0])
        
        # Inverted / overlapping line bounds (defensive handling)
        overlapping_lines = [(50, 105), (95, 150)] # overlap of 10px
        co0 = compute_bounded_line_crop_y(overlapping_lines, 0, img_h)
        co1 = compute_bounded_line_crop_y(overlapping_lines, 1, img_h)
        self.assertLessEqual(co0[0], co0[1])
        self.assertLessEqual(co1[0], co1[1])

    def test_03_first_and_last_lines_within_bounds(self):
        """3. Crops for top-touching and bottom-touching lines remain strictly in [0, img_height]."""
        img_h = 200
        lines = [(0, 40), (60, 120), (160, 200)]
        
        c_first = compute_bounded_line_crop_y(lines, 0, img_h)
        self.assertEqual(c_first[0], 0, "Top line touching y=0 should not have negative crop_y1")
        self.assertLessEqual(c_first[1], lines[1][0])
        
        c_last = compute_bounded_line_crop_y(lines, 2, img_h)
        self.assertEqual(c_last[1], img_h, "Bottom line touching height should not exceed img_height")
        self.assertGreaterEqual(c_last[0], lines[1][1])

    def test_04_empty_or_unusual_line_boundary_input(self):
        """4. Unusual inputs (empty list, out of bounds index, zero-height lines) return safe fallbacks."""
        img_h = 150
        # Empty list
        cy1, cy2 = compute_bounded_line_crop_y([], 0, img_h)
        self.assertEqual((cy1, cy2), (0, img_h))
        
        # Out of bounds indices
        lines = [(20, 60)]
        self.assertEqual(compute_bounded_line_crop_y(lines, -1, img_h), (0, img_h))
        self.assertEqual(compute_bounded_line_crop_y(lines, 5, img_h), (0, img_h))
        
        # Zero-height degenerate line
        degen = [(50, 50)]
        d_y1, d_y2 = compute_bounded_line_crop_y(degen, 0, img_h)
        self.assertLess(d_y1, d_y2, "Degenerate line should produce a positive-height crop")

    def test_05_tamil_diacritics_and_descenders_preserved(self):
        """5. Sufficient inter-line space allows generous padding for ascenders and descenders."""
        img_h = 600
        # Line height = 60, generous gaps of 50px
        lines = [(50, 110), (160, 220)]
        c0 = compute_bounded_line_crop_y(lines, 0, img_h)
        
        line_h = 110 - 50
        expected_pad = max(2, int(round(line_h * 0.18))) # 11px
        
        # Gap below is 50 -> half is 25 > expected_pad (11)
        # So padding below should reach full desired_pad_y (11px)
        self.assertEqual(c0[1] - 110, expected_pad, "Should use full desired padding when space permits")

    def test_06_real_notebook_regression_crop_boundaries(self):
        """6. Real notebook image does not introduce Line 1 pixels into Line 0 crops."""
        candidates = [
            "outputs/verification/hand-written.png",
            os.path.join(os.path.dirname(__file__), "..", "outputs", "verification", "hand-written.png")
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)
        if img_path is None:
            self.skipTest("hand-written.png regression image not found.")
            
        im = Image.open(img_path).convert('RGB')
        raw_binary = binarize_image(im)
        binary = filter_horizontal_rules(raw_binary)
        lines = segment_lines(binary, min_line_height=12)
        
        self.assertGreaterEqual(len(lines), 2, "Notebook image should have at least 2 lines")
        
        # In hand-written.png: Line 0 is (34, 59), Line 1 begins at 61
        c0_y1, c0_y2 = compute_bounded_line_crop_y(lines, 0, im.height)
        c1_y1, c1_y2 = compute_bounded_line_crop_y(lines, 1, im.height)
        
        self.assertLessEqual(c0_y2, c1_y1, "Line 0 bottom must not encroach into Line 1 top")
        self.assertLessEqual(c0_y2, lines[1][0], "Line 0 crop must not cross Line 1 detected start")
        # Line 1 starts at y=61; Line 0 crop must be <= 60
        self.assertLessEqual(c0_y2, 60, f"Line 0 crop y2 is {c0_y2}, expected <= 60 to prevent encroaching into Line 1")

    def test_07_page_segmenter_integration(self):
        """7. Full PageSegmenter produces non-overlapping line and word bounding boxes."""
        # Create a synthetic 3-line document with tight 3px gaps
        img = Image.new('L', (300, 150), color=250)
        draw = ImageDraw.Draw(img)
        # Line 0: y=20..45 (h=25)
        draw.rectangle([20, 20, 100, 45], fill=20)
        draw.rectangle([120, 20, 220, 45], fill=20)
        # Line 1: y=48..73 (gap=3px, h=25)
        draw.rectangle([25, 48, 110, 73], fill=20)
        draw.rectangle([130, 48, 200, 73], fill=20)
        # Line 2: y=76..101 (gap=3px, h=25)
        draw.rectangle([30, 76, 150, 101], fill=20)
        
        segmenter = PageSegmenter(min_line_height=10)
        doc = segmenter.process_page(img)
        
        self.assertEqual(len(doc), 3, "Expected 3 detected lines")
        for i in range(len(doc) - 1):
            cur_line_words = doc[i]['words']
            next_line_words = doc[i + 1]['words']
            
            max_cur_y2 = max(w['bbox'][1] + w['bbox'][3] for w in cur_line_words)
            min_next_y1 = min(w['bbox'][1] for w in next_line_words)
            
            self.assertLessEqual(
                max_cur_y2, min_next_y1,
                f"Word crops between Line {i} and Line {i+1} overlap vertically: {max_cur_y2} > {min_next_y1}"
            )

    def test_08_ruling_aware_padding_clamping(self):
        """8. Ruling lines in inter-line gaps clamp crop boundaries strictly before the rule row."""
        img_h = 300
        lines = [(40, 80), (120, 160)]
        # Rule line at y=31 (above line 0) and at y=100 (in gap between line 0 and 1)
        rule_rows = {31, 100}
        
        # Line 0 default would be y1=40 - 7 = 33, y2=80 + 7 = 87
        # With rule at y=100, gap below is clamped if pad reached it, but 87 < 100
        # Now test with rule at y=35 (inside top pad region 33..40):
        rule_rows_tight = {35, 85}
        c0_tight = compute_bounded_line_crop_y(lines, 0, img_h, rule_rows=rule_rows_tight)
        # Top must clamp immediately below y=35 -> y=36
        self.assertEqual(c0_tight[0], 36, f'Expected crop_y1 to be clamped to 36, got {c0_tight[0]}')
        # Bottom must clamp immediately above y=85 -> y=84
        self.assertEqual(c0_tight[1], 84, f'Expected crop_y2 to be clamped to 84, got {c0_tight[1]}')
        
    def test_09_detect_interline_rule_rows_ignores_text_strokes(self):
        """9. Ruling line detector ignores horizontal strokes that lie inside text lines."""
        # 100x100 synthetic binary image with text line at y=30..60
        bin_img = np.zeros((100, 100), dtype=np.uint8)
        # Draw a heavy horizontal text stroke inside line band at y=45
        bin_img[45, 10:90] = 1 # 80px run (80% ratio)
        # Draw a real notebook ruling line outside text band at y=80
        bin_img[80, 5:95] = 1 # 90px run
        
        lines = [(30, 60)]
        detected_rules = detect_interline_rule_rows(bin_img, lines, min_run_ratio=0.30)
        
        self.assertIn(80, detected_rules, 'Rule at y=80 outside text band must be detected')
        self.assertNotIn(45, detected_rules, 'Text stroke at y=45 inside text band must NEVER be detected as a rule')
        
    def test_10_unruled_document_padding_parity(self):
        """10. Plain unruled documents (rule_rows=None or empty) yield identical crops to baseline."""
        img_h = 400
        lines = [(20, 60), (90, 140), (180, 240)]
        for idx in range(len(lines)):
            c_base = compute_bounded_line_crop_y(lines, idx, img_h)
            c_none = compute_bounded_line_crop_y(lines, idx, img_h, rule_rows=None)
            c_empty = compute_bounded_line_crop_y(lines, idx, img_h, rule_rows=set())
            self.assertEqual(c_base, c_none)
            self.assertEqual(c_base, c_empty)

if __name__ == '__main__':
    unittest.main()
