"""
test_adaptive_word_segmentation.py - Unit tests for adaptive Tamil word segmentation.

Verifies:
1. Short single-character words ('நீ' and 'தான்') remain separate when there is a word gap.
2. Legitimate short Tamil words ('தீ', 'பூ', 'வா', 'கை', 'போ') are not merged solely due to narrow width.
3. Detached marks (pulli dot, isolated vowel sign) remain merged into the parent word.
4. Closely spaced characters within a single word are never over-segmented into separate words.
5. Existing word ordering and bounding-box validity are strictly preserved.
"""

import unittest
import numpy as np

from src.ocr.page_segmentation import segment_words_in_line

class TestAdaptiveWordSegmentation(unittest.TestCase):
    def test_01_short_word_nee_and_than_remain_separate(self):
        """1. 'நீ' (width 32px) and 'தான்' (width 92px) separated by a 16px gap remain two distinct words."""
        line_h = 75
        w = 200
        line_binary = np.zeros((line_h, w), dtype=np.uint8)

        # Word 1: 'நீ' occupying x in [10, 42] (width 32px, height 50px)
        line_binary[15:65, 10:42] = 1

        # Whitespace gap: x in [42, 58] (gap = 16px)

        # Word 2: 'தான்' occupying x in [58, 150] (width 92px, height 50px)
        line_binary[15:65, 58:150] = 1

        boxes = segment_words_in_line(line_binary, line_h=line_h)

        self.assertEqual(len(boxes), 2, f"Expected 2 separate word boxes, got {len(boxes)}: {boxes}")
        # Word 1 bounds
        self.assertAlmostEqual(boxes[0][0], 10, delta=2)
        self.assertAlmostEqual(boxes[0][1], 42, delta=2)
        # Word 2 bounds
        self.assertAlmostEqual(boxes[1][0], 58, delta=2)
        self.assertAlmostEqual(boxes[1][1], 150, delta=2)

    def test_02_other_short_tamil_words_preserved(self):
        """2. Legitimate short Tamil words ('தீ', 'வா', 'கை') with width ~25px are not merged across word gaps."""
        line_h = 60
        w = 300
        line_binary = np.zeros((line_h, w), dtype=np.uint8)

        # Word 1: 'வா' [10, 35] (width 25px)
        line_binary[10:50, 10:35] = 1
        # Gap: [35, 50] (gap 15px)
        # Word 2: 'கை' [50, 78] (width 28px)
        line_binary[10:50, 50:78] = 1
        # Gap: [78, 95] (gap 17px)
        # Word 3: 'போ' [95, 125] (width 30px)
        line_binary[10:50, 95:125] = 1

        boxes = segment_words_in_line(line_binary, line_h=line_h)
        self.assertEqual(len(boxes), 3, f"Expected 3 words for short Tamil words, got {len(boxes)}: {boxes}")
        self.assertLess(boxes[0][1], boxes[1][0])
        self.assertLess(boxes[1][1], boxes[2][0])

    def test_03_detached_pulli_dot_remains_associated_with_word(self):
        """3. Detached pulli dot (width 4px) separated by a 3px gap remains merged into parent word."""
        line_h = 60
        w = 150
        line_binary = np.zeros((line_h, w), dtype=np.uint8)

        # Base consonant 'க' in [10, 40]
        line_binary[20:55, 10:40] = 1
        # Detached pulli dot at top: x in [45, 49], y in [5, 10] (width 4px, gap 5px)
        line_binary[5:10, 45:49] = 1

        boxes = segment_words_in_line(line_binary, line_h=line_h)
        # Must be merged into 1 word box spanning [10, 49]
        self.assertEqual(len(boxes), 1, f"Detached pulli dot should be merged into parent word, got {boxes}")
        self.assertAlmostEqual(boxes[0][0], 10, delta=2)
        self.assertGreaterEqual(boxes[0][1], 48)

    def test_04_closely_spaced_intra_word_characters_not_over_segmented(self):
        """4. Characters within a single word with intra-word pen lifts (gaps 2-5px) are not split."""
        line_h = 60
        w = 200
        line_binary = np.zeros((line_h, w), dtype=np.uint8)

        # Word 'உலகம்': letter 1 [10, 30], gap [30, 34] (4px), letter 2 [34, 55], gap [55, 59] (4px), letter 3 [59, 85]
        line_binary[15:55, 10:30] = 1
        line_binary[15:55, 34:55] = 1
        line_binary[15:55, 59:85] = 1

        boxes = segment_words_in_line(line_binary, line_h=line_h)
        self.assertEqual(len(boxes), 1, f"Intra-word gaps of 4px should not split word, got: {boxes}")
        self.assertAlmostEqual(boxes[0][0], 10, delta=2)
        self.assertAlmostEqual(boxes[0][1], 85, delta=2)

    def test_05_word_ordering_and_bounding_box_validity_preserved(self):
        """5. Word boxes are strictly non-overlapping, in left-to-right order, and have positive dimensions."""
        line_h = 50
        w = 300
        line_binary = np.zeros((line_h, w), dtype=np.uint8)

        # Draw 4 words with various spacings
        line_binary[10:45, 10:45] = 1
        line_binary[10:45, 65:110] = 1
        line_binary[10:45, 130:190] = 1
        line_binary[10:45, 210:280] = 1

        boxes = segment_words_in_line(line_binary, line_h=line_h)
        self.assertEqual(len(boxes), 4)
        for i in range(len(boxes)):
            x1, x2 = boxes[i]
            self.assertGreater(x2, x1, "Word box must have positive width")
            if i > 0:
                prev_x1, prev_x2 = boxes[i - 1]
                self.assertLessEqual(prev_x2, x1, "Word boxes must be strictly ordered without overlap")

if __name__ == "__main__":
    unittest.main()
