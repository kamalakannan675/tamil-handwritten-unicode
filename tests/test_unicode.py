"""
test_unicode.py - Unit tests for Tamil Class-to-Unicode mapping.
"""

import unittest
from model.unicode_mapping import (
    CLASS_TO_UNICODE,
    UNICODE_TO_CLASS,
    NUM_CLASSES,
    get_unicode_char,
    get_unicode_codepoints,
    get_class_id
)

class TestUnicodeMapping(unittest.TestCase):
    def test_total_classes(self):
        """Verify exactly 156 classes exist, indexed 0 to 155."""
        self.assertEqual(NUM_CLASSES, 156)
        self.assertEqual(len(CLASS_TO_UNICODE), 156)
        self.assertEqual(min(CLASS_TO_UNICODE.keys()), 0)
        self.assertEqual(max(CLASS_TO_UNICODE.keys()), 155)

    def test_verified_paper_mappings(self):
        """Verify ground truth mappings from Figures 11, 28, 29 of base paper."""
        # Class 0: Tamil Vowel Sign AA (ா)
        self.assertEqual(get_unicode_char(0), 'ா')
        self.assertEqual(get_unicode_codepoints(0), 'U+0BBE')

        # Class 1: Tamil Letter A (அ)
        self.assertEqual(get_unicode_char(1), 'அ')
        self.assertEqual(get_unicode_codepoints(1), 'U+0B85')

        # Class 2: Tamil Letter AA (ஆ)
        self.assertEqual(get_unicode_char(2), 'ஆ')
        self.assertEqual(get_unicode_codepoints(2), 'U+0B86')

        # Class 13: Tamil Sign Visarga / Ayudha Ezhuthu (ஃ)
        self.assertEqual(get_unicode_char(13), 'ஃ')
        self.assertEqual(get_unicode_codepoints(13), 'U+0B83')

        # Class 14: Tamil Letter K + Pulli (க்)
        self.assertEqual(get_unicode_char(14), 'க்')
        self.assertEqual(get_unicode_codepoints(14), 'U+0B95 U+0BCD')

        # Class 15: Tamil Letter KA (க)
        self.assertEqual(get_unicode_char(15), 'க')
        self.assertEqual(get_unicode_codepoints(15), 'U+0B95')

        # Class 146: Tamil Shri (ஸ்ரீ)
        self.assertEqual(get_unicode_char(146), 'ஸ்ரீ')
        self.assertEqual(get_unicode_codepoints(146), 'U+0BB8 U+0BCD U+0BB0 U+0BC0')

        # Class 153: Tamil Vowel Sign E (ெ)
        self.assertEqual(get_unicode_char(153), 'ெ')
        self.assertEqual(get_unicode_codepoints(153), 'U+0BC6')

        # Class 154: Tamil Vowel Sign EE (ே)
        self.assertEqual(get_unicode_char(154), 'ே')
        self.assertEqual(get_unicode_codepoints(154), 'U+0BC7')

        # Class 155: Tamil Vowel Sign AI (ை)
        self.assertEqual(get_unicode_char(155), 'ை')
        self.assertEqual(get_unicode_codepoints(155), 'U+0BC8')

    def test_reverse_lookup(self):
        """Verify reverse glyph lookup to class ID."""
        self.assertEqual(get_class_id('அ'), 1)
        self.assertEqual(get_class_id('க'), 15)
        self.assertEqual(get_class_id('ஸ்ரீ'), 146)
        self.assertIsNone(get_class_id('XYZ'))

    def test_codepoint_hex_reconstruction(self):
        """Verify that every character equals its reconstructed hex string."""
        for cid in range(NUM_CLASSES):
            info = CLASS_TO_UNICODE[cid]
            char = info['char']
            hex_parts = info['hex'].split()
            reconstructed = ''.join(chr(int(h, 16)) for h in hex_parts)
            self.assertEqual(char, reconstructed, f"Mismatch at class {cid}")

if __name__ == '__main__':
    unittest.main()
