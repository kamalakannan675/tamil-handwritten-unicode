"""
test_word_refusion.py - Unit and Integration Tests for Experimental Word Re-Fusion Prototype.
"""

import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from PIL import Image, ImageDraw

from src.ocr.full_page_ocr import (
    TamilFullPageOCR,
    INDEPENDENT_SHORT_WORDS,
    is_word_refusion_enabled
)

class TestWordRefusionPrototype(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_path = "reports/tamil_ctc_char_vocab.json"
        if not os.path.exists(cls.vocab_path):
            raise unittest.SkipTest(f"Vocab file {cls.vocab_path} missing.")
        cls.ocr = TamilFullPageOCR(vocab_path=cls.vocab_path, device="cpu")

    def test_feature_flag_resolution(self):
        """Verify is_word_refusion_enabled follows strict opt-in rules and defaults to False."""
        # Explicit arguments take precedence
        self.assertTrue(is_word_refusion_enabled(True))
        self.assertFalse(is_word_refusion_enabled(False))

        # When explicit argument is None, consult environment variable
        with patch.dict(os.environ, {}, clear=True):
            # Missing env var -> default False
            self.assertFalse(is_word_refusion_enabled())
            self.assertFalse(is_word_refusion_enabled(None))

        with patch.dict(os.environ, {'ENABLE_WORD_REFUSION': ''}):
            self.assertFalse(is_word_refusion_enabled())

        with patch.dict(os.environ, {'ENABLE_WORD_REFUSION': '   '}):
            self.assertFalse(is_word_refusion_enabled())

        with patch.dict(os.environ, {'ENABLE_WORD_REFUSION': 'false'}):
            self.assertFalse(is_word_refusion_enabled())

        with patch.dict(os.environ, {'ENABLE_WORD_REFUSION': '0'}):
            self.assertFalse(is_word_refusion_enabled())

        with patch.dict(os.environ, {'ENABLE_WORD_REFUSION': 'disabled'}):
            self.assertFalse(is_word_refusion_enabled())

        with patch.dict(os.environ, {'ENABLE_WORD_REFUSION': 'random_string'}):
            self.assertFalse(is_word_refusion_enabled())

        # Valid truthy values
        for truthy in ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES']:
            with patch.dict(os.environ, {'ENABLE_WORD_REFUSION': truthy}):
                self.assertTrue(is_word_refusion_enabled())

    def test_independent_short_words_safeguard(self):
        """Verify legitimate independent words (e.g. 'நீ', 'தான்') are protected from fusion."""
        self.assertIn('நீ', INDEPENDENT_SHORT_WORDS)
        self.assertIn('தான்', INDEPENDENT_SHORT_WORDS)
        self.assertIn('தீ', INDEPENDENT_SHORT_WORDS)
        self.assertIn('வா', INDEPENDENT_SHORT_WORDS)
        self.assertIn('போ', INDEPENDENT_SHORT_WORDS)
        # Bare consonants must NOT be in the independent words whitelist
        self.assertNotIn('ப', INDEPENDENT_SHORT_WORDS)
        self.assertNotIn('க', INDEPENDENT_SHORT_WORDS)
        self.assertNotIn('த', INDEPENDENT_SHORT_WORDS)

    def test_refuse_adjacent_word_segments_safeguard_logic(self):
        """Verify refuse_adjacent_word_segments skips pairs where both are independent words."""
        page_img = Image.new('RGB', (300, 100), color=(255, 255, 255))
        line_bbox = [0, 0, 300, 50]
        # Two words: 'நீ' at [10, 10, 30, 40], 'தான்' at [45, 10, 60, 40]. Gap = 5px.
        base_words = [
            {'word_idx': 0, 'bbox': [10, 10, 30, 40], 'text': 'நீ', 'crop': page_img.crop((10, 10, 40, 50))},
            {'word_idx': 1, 'bbox': [45, 10, 60, 40], 'text': 'தான்', 'crop': page_img.crop((45, 10, 105, 50))}
        ]
        new_words, fusion_events = self.ocr.refuse_adjacent_word_segments(page_img, line_bbox, base_words)
        # Both words must remain independent
        self.assertEqual(len(new_words), 2)
        self.assertEqual(new_words[0]['text'], 'நீ')
        self.assertEqual(new_words[1]['text'], 'தான்')
        self.assertEqual(len(fusion_events), 0)

    def test_refusion_large_gap_not_evaluated(self):
        """Verify adjacent words separated by normal gaps (> 0.15 * line_h) are never evaluated."""
        page_img = Image.new('RGB', (400, 100), color=(255, 255, 255))
        line_bbox = [0, 0, 400, 50]
        # Gap = 80 - 40 = 40px (0.80 * line_h)
        base_words = [
            {'word_idx': 0, 'bbox': [10, 10, 30, 40], 'text': 'உலகம்', 'crop': page_img.crop((10, 10, 40, 50))},
            {'word_idx': 1, 'bbox': [80, 10, 60, 40], 'text': 'என்பது', 'crop': page_img.crop((80, 10, 140, 50))}
        ]
        new_words, fusion_events = self.ocr.refuse_adjacent_word_segments(page_img, line_bbox, base_words)
        self.assertEqual(len(new_words), 2)
        self.assertEqual(len(fusion_events), 0)

    def test_refusion_defensive_exception_handling(self):
        """Verify exceptions during union crop inference fallback safely to baseline words."""
        page_img = Image.new('RGB', (300, 100), color=(255, 255, 255))
        line_bbox = [0, 0, 300, 50]
        # Two words that would trigger fusion: 'ப' (bare consonant) and 'யணம்'
        base_words = [
            {'word_idx': 0, 'bbox': [10, 10, 20, 40], 'text': 'ப', 'crop': page_img.crop((10, 10, 30, 50))},
            {'word_idx': 1, 'bbox': [33, 10, 60, 40], 'text': 'யணம்', 'crop': page_img.crop((33, 10, 93, 50))}
        ]

        with patch.object(self.ocr, 'recognize_word_crop', side_effect=RuntimeError("Simulated CUDA/Memory Error")):
            new_words, fusion_events = self.ocr.refuse_adjacent_word_segments(page_img, line_bbox, base_words)

        # Baseline words must be preserved despite exception
        self.assertEqual(len(new_words), 2)
        self.assertEqual(new_words[0]['text'], 'ப')
        self.assertEqual(new_words[1]['text'], 'யணம்')
        # Fusion event recorded with exception details
        self.assertEqual(len(fusion_events), 1)
        self.assertFalse(fusion_events[0]['accepted'])
        self.assertIn("Exception during union crop inference: RuntimeError", fusion_events[0]['reason'])

    def test_refusion_flag_disabled_preserves_baseline(self):
        """Verify enable_word_refusion=False bypasses re-fusion entirely."""
        test_img = Image.new('L', (400, 200), color=240)
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([30, 20, 100, 50], fill=20)
        draw.rectangle([140, 20, 220, 50], fill=20)

        res_baseline = self.ocr.process_document(test_img, enable_word_refusion=False)
        self.assertIn('full_text', res_baseline)
        self.assertIn('lines', res_baseline)
        for line in res_baseline['lines']:
            self.assertNotIn('fusion_events', line)

    def test_process_document_default_disabled(self):
        """Verify process_document defaults to disabled when no flag and no env var provided."""
        test_img = Image.new('L', (400, 200), color=240)
        draw = ImageDraw.Draw(test_img)
        draw.rectangle([30, 20, 100, 50], fill=20)
        draw.rectangle([140, 20, 220, 50], fill=20)

        with patch.dict(os.environ, {}, clear=True):
            res_default = self.ocr.process_document(test_img)
            for line in res_default['lines']:
                self.assertNotIn('fusion_events', line)

    def test_incident_image_refusion_evaluation(self):
        """Verify re-fusion corrects 'ப யணம்' to 'பயணம்' and 'நீ ரடை' to 'நீரேடை' on the incident image."""
        incident_path = r"C:\Users\VS\.gemini\antigravity\brain\7bbb0c6e-6666-4556-9311-dd1d71efba7c\.user_uploaded\media_1792847294856_3135e84c.jpg"
        if not os.path.exists(incident_path):
            self.skipTest(f"Incident image not found: {incident_path}")

        res = self.ocr.process_document(incident_path, enable_word_refusion=True)
        full_text = res['full_text']
        lines = full_text.split('\n')

        # Line 0 should contain 'நீரேடை' instead of 'நீ ரடை'
        self.assertIn('நீரேடை', lines[0])
        self.assertNotIn('நீ ரடை', lines[0])

        # Line 4 should contain 'பயணம்' instead of 'ப யணம்'
        self.assertIn('பயணம்', lines[4])
        self.assertNotIn('ப யணம்', lines[4])

        # Line 3 should preserve 'தான்' without fusing into 'ந்தான்'
        self.assertIn('தான்', lines[3])

if __name__ == '__main__':
    unittest.main()
