"""
test_unicode_validator.py - Comprehensive Unit Tests for Non-Destructive Tamil Unicode Validation.

Verifies:
1. Valid Tamil vocabulary and classical poetry have zero false positives (empty anomaly lists).
2. Spurious non-text sequences like 'உிட்' are detected as INVALID_VOWEL_COMBINATION.
3. Orphaned leading combining marks / viramas are flagged as LEADING_COMBINING_MARK.
4. Consecutive viramas (double pulli) are flagged as CONSECUTIVE_VIRAMA.
5. Consecutive dependent vowel signs are flagged as MULTIPLE_VOWEL_SIGNS.
6. Non-destructive guarantee: validator never alters, deletes, or mutates the underlying text.
7. FullPageOCR integration: process_document attaches anomaly metadata to lines and document result.
"""

import os
import sys
import unittest
from PIL import Image

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.full_page_ocr import validate_tamil_unicode_sequence, TamilFullPageOCR

class TestTamilUnicodeValidator(unittest.TestCase):
    def test_01_valid_tamil_words_zero_anomalies(self):
        """1. Standard Tamil words, consonants, independent vowels, and words with viramas produce zero anomalies."""
        valid_samples = [
            "அம்மா",
            "ஆடு",
            "இலை",
            "உலகு",
            "ஊர்",
            "எழுத்தெல்லாம்",
            "ஐந்து",
            "ஒன்று",
            "ஔவை",
            "அகர",
            "முதல",
            "முதல்",
            "பகவன்",
            "முதற்றே",
            "திருக்குறள்",
            "தமிழ்",
            "கற்க கசடறக் கற்பவை கற்றபின்",
            "நிற்க அதற்குத் தக"
        ]
        for word in valid_samples:
            anomalies = validate_tamil_unicode_sequence(word)
            self.assertEqual(
                len(anomalies), 0,
                f"Valid Tamil word '{word}' produced unexpected anomalies: {anomalies}"
            )

    def test_02_detect_invalid_vowel_combinations(self):
        """2. Detects illegal combinations of independent vowel + combining mark/virama (e.g. 'உிட்')."""
        # The observed Thirukkural real-world spurious line error
        spurious_line = "உிட்"
        anomalies = validate_tamil_unicode_sequence(spurious_line)
        self.assertGreaterEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]['type'], 'INVALID_VOWEL_COMBINATION')
        self.assertEqual(anomalies[0]['sequence'], 'உி')
        self.assertIn('U+0B89', anomalies[0]['codepoints'])
        self.assertIn('U+0BBF', anomalies[0]['codepoints'])

        # Other invalid independent vowel combinations
        other_cases = ["அா", "இீ", "ஊ்", "எை"]
        for seq in other_cases:
            anoms = validate_tamil_unicode_sequence(seq)
            self.assertGreaterEqual(len(anoms), 1)
            self.assertEqual(anoms[0]['type'], 'INVALID_VOWEL_COMBINATION')

    def test_03_detect_leading_combining_marks(self):
        """3. Detects orphaned combining marks or pulli at the start of tokens."""
        cases = [
            ("்வரி", 'LEADING_COMBINING_MARK', "Leading virama"),
            ("ாப்பம்", 'LEADING_COMBINING_MARK', "Leading vowel sign aa"),
            ("ிளமை", 'LEADING_COMBINING_MARK', "Leading vowel sign i")
        ]
        for text, expected_type, desc in cases:
            anomalies = validate_tamil_unicode_sequence(text)
            self.assertGreaterEqual(len(anomalies), 1, f"Failed to detect anomaly in {desc}")
            types = [a['type'] for a in anomalies]
            self.assertIn(expected_type, types, f"Expected {expected_type} in {types}")

    def test_04_detect_consecutive_viramas(self):
        """4. Detects consecutive pulli marks (double virama)."""
        text = "க்்"
        anomalies = validate_tamil_unicode_sequence(text)
        self.assertGreaterEqual(len(anomalies), 1)
        types = [a['type'] for a in anomalies]
        self.assertIn('CONSECUTIVE_VIRAMA', types)

    def test_05_detect_multiple_vowel_signs(self):
        """5. Detects consecutive dependent vowel signs on a base character."""
        text = "காி"  # க + ா + ி
        anomalies = validate_tamil_unicode_sequence(text)
        self.assertGreaterEqual(len(anomalies), 1)
        types = [a['type'] for a in anomalies]
        self.assertIn('MULTIPLE_VOWEL_SIGNS', types)

    def test_06_non_destructive_guarantee(self):
        """6. Verifies that input text is never mutated, shortened, or altered by validation."""
        original = "உிட் அகர முதல"
        anomalies = validate_tamil_unicode_sequence(original)
        self.assertIsInstance(anomalies, list)
        # Original string must be completely identical
        self.assertEqual(original, "உிட் அகர முதல")

    def test_07_empty_whitespace_and_latin_text(self):
        """7. Handles empty string, pure whitespace, Latin text, and numbers gracefully without errors."""
        self.assertEqual(validate_tamil_unicode_sequence(""), [])
        self.assertEqual(validate_tamil_unicode_sequence("   "), [])
        self.assertEqual(validate_tamil_unicode_sequence("Hello World 123!"), [])

    def test_08_full_page_ocr_schema_backward_compatibility(self):
        """8. Validates that process_document attaches anomalies without breaking existing keys."""
        # Use mock image
        img = Image.new('RGB', (100, 40), color=(255, 255, 255))
        ocr = TamilFullPageOCR(
            checkpoint_path="checkpoints/crnn/best_model.pt",
            vocab_path="reports/tamil_ctc_char_vocab.json",
            device='cpu'
        )
        res = ocr.process_document(img)
        # Verify required legacy keys exist
        self.assertIn('full_text', res)
        self.assertIn('total_lines', res)
        self.assertIn('total_words', res)
        self.assertIn('lines', res)
        self.assertIn('image_size', res)
        self.assertIn('preflight', res)
        # Verify new additive key exists
        self.assertIn('anomalies', res)
        self.assertIsInstance(res['anomalies'], list)
        for line in res['lines']:
            self.assertIn('anomalies', line)
            self.assertIsInstance(line['anomalies'], list)

if __name__ == '__main__':
    unittest.main()
