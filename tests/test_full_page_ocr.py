"""
test_full_page_ocr.py - Unit and Integration Tests for Full-Page Document OCR Pipeline.
"""

import os
import sys
# Bootstrap: ensure the project root is on sys.path when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from PIL import Image, ImageDraw

from src.ocr.page_segmentation import (
    otsu_threshold,
    binarize_image,
    segment_lines,
    segment_words_in_line,
    PageSegmenter
)
from src.ocr.full_page_ocr import TamilFullPageOCR

class TestFullPageOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic 2-line handwritten document image
        cls.img_w = 400
        cls.img_h = 200
        cls.test_img = Image.new('L', (cls.img_w, cls.img_h), color=240) # white/off-white paper
        draw = ImageDraw.Draw(cls.test_img)
        
        # Line 1: Two words
        # Word 1: (30, 20) to (100, 50)
        draw.rectangle([30, 20, 100, 50], fill=20) # dark ink
        # Word 2: (140, 20) to (220, 50)
        draw.rectangle([140, 20, 220, 50], fill=20)
        
        # Line 2: One word
        # Word 1: (30, 100) to (180, 140)
        draw.rectangle([30, 100, 180, 140], fill=20)
        
        cls.vocab_path = "reports/tamil_ctc_char_vocab.json"

    def test_otsu_thresholding(self):
        """Verify Otsu threshold correctly separates dark ink from bright paper."""
        gray_np = np.array(self.test_img, dtype=np.uint8)
        thresh = otsu_threshold(gray_np)
        self.assertGreaterEqual(thresh, 20)
        self.assertLess(thresh, 240)
        
        binary = binarize_image(self.test_img)
        # Background is white (240) -> should be 0 in binary
        self.assertEqual(binary[0, 0], 0)
        # Ink is dark (20) -> should be 1 in binary
        self.assertEqual(binary[30, 50], 1)

    def test_line_and_word_segmentation(self):
        """Verify segmentation detects 2 lines and correct word counts."""
        binary = binarize_image(self.test_img)
        lines = segment_lines(binary, min_line_height=10)
        self.assertEqual(len(lines), 2)
        
        # Line 1 should have 2 words separated by a 40px gap
        line1_binary = binary[lines[0][0]:lines[0][1], :]
        words_l1 = segment_words_in_line(line1_binary, min_word_width=10, space_gap=15)
        self.assertEqual(len(words_l1), 2)
        
        # Line 2 should have 1 word
        line2_binary = binary[lines[1][0]:lines[1][1], :]
        words_l2 = segment_words_in_line(line2_binary, min_word_width=10, space_gap=15)
        self.assertEqual(len(words_l2), 1)

    def test_page_segmenter_class(self):
        """Verify PageSegmenter hierarchical document output."""
        segmenter = PageSegmenter(min_line_height=10, min_word_width=10, space_gap=15)
        doc_structure = segmenter.process_page(self.test_img)
        
        self.assertEqual(len(doc_structure), 2)
        self.assertEqual(len(doc_structure[0]['words']), 2)
        self.assertEqual(len(doc_structure[1]['words']), 1)
        
        # Verify word crop dimensions
        w0 = doc_structure[0]['words'][0]['crop']
        self.assertIsInstance(w0, Image.Image)
        self.assertGreater(w0.width, 10)
        self.assertGreater(w0.height, 10)

    def test_full_page_ocr_pipeline(self):
        """Verify end-to-end full page OCR processing."""
        if not os.path.exists(self.vocab_path):
            self.skipTest(f"Vocab file {self.vocab_path} missing.")
            
        ocr = TamilFullPageOCR(vocab_path=self.vocab_path)
        result = ocr.process_document(self.test_img)
        
        self.assertIn('full_text', result)
        self.assertEqual(result['total_lines'], 2)
        self.assertEqual(result['total_words'], 3)
        self.assertEqual(len(result['lines']), 2)
        self.assertIsInstance(result['full_text'], str)

if __name__ == '__main__':
    unittest.main()
