"""
test_unicode_virama_repair.py - Targeted Unit Tests for Tamil Virama Post-Processing Repair.

Verifies:
1. Positive repair: Misplaced virama sequence in 'திரு்குறள்' -> 'திருக்குறள்'.
2. Idempotence: Already correct 'திருக்குறள்' and repeated passes remain 100% unchanged.
3. Preservation of verified ground-truth words ('அகர', 'முதல', 'எழுத்தெல்லாம்', 'ஆதி', 'பகவன்', 'முதற்றே', 'உலகு').
4. Negative controls: Diverse valid Tamil words with viramas (pulli) remain strictly unaffected.
5. Edge cases: Empty strings, whitespace, single characters, punctuation, numbers, non-Tamil text.
6. Pipeline integration: FullPageOCR line assembly correctly applies virama repair.
"""

import unittest
import unicodedata

from src.data.vocab import (
    repair_geminate_virama,
    TAMIL_VIRAMA,
    TAMIL_CONSONANTS,
    TAMIL_VOWEL_SIGNS
)
from src.ocr.full_page_ocr import TamilFullPageOCR


class TestUnicodeViramaRepair(unittest.TestCase):

    def test_01_positive_geminate_virama_repair(self):
        """Positive test: Malformed 'திரு்குறள்' must be repaired to 'திருக்குறள்'."""
        malformed = "திரு்குறள்"
        expected = "திருக்குறள்"
        
        # Verify input has the exact illegal [VOWEL_SIGN U] + [VIRAMA] sequence
        raw_chars = list(malformed)
        self.assertEqual(raw_chars[3], '\u0bc1')  # ு
        self.assertEqual(raw_chars[4], '\u0bcd')  # ்
        self.assertEqual(raw_chars[5], '\u0b95')  # க
        self.assertEqual(raw_chars[6], '\u0bc1')  # ு
        
        repaired = repair_geminate_virama(malformed)
        self.assertEqual(repaired, expected)
        self.assertTrue(unicodedata.is_normalized('NFC', repaired))

    def test_02_idempotence(self):
        """Idempotence test: Valid 'திருக்குறள்' and repeated repair calls must not alter the string."""
        correct = "திருக்குறள்"
        self.assertEqual(repair_geminate_virama(correct), correct)
        
        # Repeated application on malformed input must be stable
        malformed = "திரு்குறள்"
        first_pass = repair_geminate_virama(malformed)
        second_pass = repair_geminate_virama(first_pass)
        third_pass = repair_geminate_virama(second_pass)
        self.assertEqual(first_pass, correct)
        self.assertEqual(second_pass, correct)
        self.assertEqual(third_pass, correct)

    def test_03_preserves_verified_sample_words(self):
        """Regression test: All 7 other words from verified sample must remain completely unaffected."""
        sample_words = [
            "அகர",
            "முதல",
            "எழுத்தெல்லாம்",
            "ஆதி",
            "பகவன்",
            "முதற்றே",
            "உலகு"
        ]
        for word in sample_words:
            repaired = repair_geminate_virama(word)
            self.assertEqual(repaired, word, f"Unintended alteration of ground-truth word '{word}' -> '{repaired}'")

    def test_04_negative_controls_valid_viramas(self):
        """Negative controls: Valid Tamil words with viramas (pure consonants) must not change."""
        valid_words = [
            "தமிழ்",
            "வணக்கம்",
            "நன்றி",
            "கற்க",
            "கசடறக்",
            "கற்பவை",
            "கற்றபின்",
            "நிற்க",
            "அதற்குத்",
            "தக",
            "சொல்",
            "பால்",
            "பண்",
            "கண்",
            "மண்",
            "வெற்றி",
            "முயற்சி",
            "அன்பு",
            "இன்பம்",
            "செய்கை",
            "எண்ணித்",
            "துணிக",
            "கருமம்"
        ]
        for word in valid_words:
            repaired = repair_geminate_virama(word)
            self.assertEqual(repaired, word, f"Valid Tamil word '{word}' altered to '{repaired}'")

    def test_05_edge_cases(self):
        """Edge cases: Empty strings, whitespace, isolated characters, numbers, Latin script."""
        edge_cases = [
            "",
            "   ",
            "\n\t",
            "அ",
            "க",
            "க்",
            "்",
            "12345",
            "3.14159",
            "Tamil OCR Project 2026",
            "Special characters: !@#$%^&*()_+=-{}[]|;:,.<>?/",
            "Mixed: அகர 123 English முதல"
        ]
        for item in edge_cases:
            repaired = repair_geminate_virama(item)
            self.assertEqual(repaired, item, f"Edge case '{item}' altered to '{repaired}'")

    def test_06_pipeline_line_assembly_integration(self):
        """Integration test: PageSegmenter and FullPageOCR assemble_line_words repairs tokens."""
        raw_tokens = ["திரு்குறள்", "அகர", "முதல"]
        assembled = TamilFullPageOCR.assemble_line_words(raw_tokens)
        self.assertEqual(assembled, "திருக்குறள் அகர முதல")


if __name__ == '__main__':
    unittest.main()
