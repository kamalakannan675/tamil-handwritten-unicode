"""
test_tamil_unicode_ocr.py - Comprehensive Unit Tests for Tamil Unicode Decoding,
Combining Mark Integrity, Word Spacing, and Full-Page Document Reconstruction.
"""

import os
import sys
import unittest
import unicodedata
import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.vocab import TamilCTCCharacterVocabulary, TAMIL_COMBINING_MARKS
from src.inference.ctc_decoder import (
    ctc_greedy_decode,
    levenshtein_distance,
    compute_cer,
    compute_wer,
    compute_exact_accuracy
)
from src.ocr.page_segmentation import (
    binarize_image,
    segment_lines,
    segment_words_in_line,
    PageSegmenter
)
from src.ocr.full_page_ocr import TamilFullPageOCR

class TestTamilUnicodeOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab_path = "reports/tamil_ctc_char_vocab.json"
        cls.vocab = TamilCTCCharacterVocabulary(cls.vocab_path)
        cls.ocr = TamilFullPageOCR(
            checkpoint_path="checkpoints/crnn/best_model.pt",
            vocab_path=cls.vocab_path,
            device="cpu"
        )

    def test_unicode_nfc_normalization(self):
        """Verify that decomposed Tamil vowel signs are normalized to canonical NFC form."""
        decomposed = "அங்கே\u0bbeர்"
        expected_nfc = unicodedata.normalize('NFC', decomposed)
        self.assertEqual(expected_nfc, "அங்கோர்")
        
        encoded = self.vocab.encode(expected_nfc)
        decoded = self.vocab.decode(encoded)
        self.assertEqual(decoded, expected_nfc)
        self.assertEqual(unicodedata.is_normalized('NFC', decoded), True)

    def test_ctc_repeated_characters_and_blanks(self):
        """Verify CTC decoder preserves identical adjacent characters when separated by blank."""
        # Sequence: [15, 15, 0, 15] -> should decode to 'கக' (two 'க' characters)
        T = 4
        num_classes = len(self.vocab)
        mock_logits = torch.full((T, 1, num_classes), -10.0)
        mock_logits[0, 0, 15] = 10.0 # 'க'
        mock_logits[1, 0, 15] = 10.0 # 'க' (collapsed)
        mock_logits[2, 0, 0] = 10.0  # <blank>
        mock_logits[3, 0, 15] = 10.0 # 'க' (second 'க')
        
        decoded = ctc_greedy_decode(mock_logits, self.vocab, blank_idx=0)
        self.assertEqual(decoded[0], "கக")

        # Sequence with pulli: [15, 0, 49, 0, 15] -> 'க' + '்' + 'க' = 'க்க'
        T2 = 5
        mock_logits2 = torch.full((T2, 1, num_classes), -10.0)
        mock_logits2[0, 0, 15] = 10.0 # 'க'
        mock_logits2[1, 0, 0] = 10.0  # blank
        mock_logits2[2, 0, 49] = 10.0 # '்'
        mock_logits2[3, 0, 0] = 10.0  # blank
        mock_logits2[4, 0, 15] = 10.0 # 'க'
        decoded2 = ctc_greedy_decode(mock_logits2, self.vocab, blank_idx=0)
        self.assertEqual(decoded2[0], "க்க")

    def test_combining_mark_assembly(self):
        """Verify that orphaned combining marks are safely attached to the preceding word without spaces."""
        words_with_split = ["அவ", "ை", "எவ்வளவு", "கெட்டவை"]
        assembled = self.ocr.assemble_line_words(words_with_split)
        self.assertEqual(assembled, "அவை எவ்வளவு கெட்டவை")
        
        pulli_split = ["ப", "்"]
        assembled_pulli = self.ocr.assemble_line_words(pulli_split)
        self.assertEqual(assembled_pulli, "ப்")

    def test_no_intra_word_spaces_in_multicharacter_words(self):
        """Verify that known multi-character Tamil words are decoded as continuous tokens without internal spaces."""
        test_words = ["வணக்கம்", "தமிழ்", "நாடு", "கூட்டுகிறார்", "விளைவதில்லை"]
        for word in test_words:
            encoded = self.vocab.encode(word)
            T = len(encoded)
            num_classes = len(self.vocab)
            logits = torch.full((T, 1, num_classes), -10.0)
            for t_idx, token_id in enumerate(encoded):
                logits[t_idx, 0, token_id] = 10.0
            decoded = ctc_greedy_decode(logits, self.vocab)[0]
            self.assertEqual(decoded, word)
            self.assertNotIn(" ", decoded, f"Intra-word space detected in '{word}' -> '{decoded}'")

    def test_word_segmentation_gap_calibration(self):
        """Verify that small intra-word gaps (2-4px) are NOT treated as word boundaries."""
        line_w = 200
        line_h = 30
        line_img = Image.new('L', (line_w, line_h), color=255)
        draw = ImageDraw.Draw(line_img)
        
        # Word 1: Part A (10..40), Part B (43..70) -> 3px gap (intra-word pen lift)
        draw.rectangle([10, 5, 40, 25], fill=0)
        draw.rectangle([43, 5, 70, 25], fill=0)
        
        # Word 2: (95..150) -> 25px gap (inter-word space)
        draw.rectangle([95, 5, 150, 25], fill=0)
        
        bin_line = binarize_image(line_img)
        words = segment_words_in_line(bin_line, line_h=line_h)
        
        self.assertEqual(len(words), 2, f"Expected 2 words, but got {len(words)}: {words}")
        self.assertLess(words[0][0], 15)
        self.assertGreater(words[0][1], 65)

    def test_cer_wer_metrics_nfc_consistency(self):
        """Verify CER and WER return 0.0 for matching NFC and NFD representations."""
        word_nfc = unicodedata.normalize('NFC', "அங்கோர்")
        word_nfd = unicodedata.normalize('NFD', "அங்கோர்")
        
        cer = compute_cer([word_nfc], [word_nfd])
        wer = compute_wer([word_nfc], [word_nfd])
        acc = compute_exact_accuracy([word_nfc], [word_nfd])
        
        self.assertEqual(cer, 0.0)
        self.assertEqual(wer, 0.0)
        self.assertEqual(acc, 1.0)

    def test_empty_and_whitespace_only_handling(self):
        """Verify robust handling of blank images without crashing or producing invalid characters."""
        blank_img = Image.new('RGB', (200, 100), color=(255, 255, 255))
        res = self.ocr.process_document(blank_img)
        self.assertEqual(res['total_lines'], 0)
        self.assertEqual(res['total_words'], 0)
        self.assertEqual(res['full_text'], "")

if __name__ == '__main__':
    unittest.main()
