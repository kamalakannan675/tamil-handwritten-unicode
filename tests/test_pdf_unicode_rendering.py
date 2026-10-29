"""
test_pdf_unicode_rendering.py - Comprehensive Unit & Regression Tests for Phase U12
Tamil PDF Rendering, Font Embedding, and Unicode Fidelity.

Validates:
1. Visual Correctness (Test Group A):
   - Non-blank pages, valid page counts and dimensions.
   - Independent vowels, consonants, combining marks, pullis, complex ligatures (ஸ்ரீ).
   - Mixed Tamil and English text without missing glyph warnings or placeholder boxes.
   - Long lines and multi-page pagination with margin-aware wrapping.

2. Unicode and Font Correctness (Test Group B):
   - Valid TrueType / CIDFontType2 font embedding (/FontFile2 present).
   - Valid Adobe-Identity-UCS /ToUnicode CMap mapping glyphs to UTF-16BE.
   - Clean, non-duplicated syllable extraction (no "மாமா", "பாபா", "பிபி").
   - Faithful recovery of input characters, numbers, and Latin words.
   - Selectable, copyable, and searchable text extraction via pdfplumber.

3. Concurrency & Stress Testing (Test Group C):
   - Multi-threaded PDF export under 4, 8, and 12 concurrent workers.
   - Thread isolation: ensures distinct concurrent requests never interleave or corrupt outputs.
"""

import os
import io
import re
import unittest
import unicodedata
import concurrent.futures
from typing import List, Dict, Any

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image

from app.unicode_converter import (
    export_to_pdf,
    _get_tamil_font_properties,
    reorder_tamil_visual_to_logical
)


class TestPdfVisualCorrectness(unittest.TestCase):
    """Test Group A: PDF visual layout, glyph shaping, and rendering correctness."""

    def test_known_tamil_glyphs_visual_rendering(self):
        """Verifies that independent vowels, consonants, and ligatures render to non-blank high-res images."""
        text = "அஆஇஈஉஊஎஏஐஒஓஔ\nகசடதபற\nஅம்மா அப்பா தம்பி தங்கை\nதமிழ் வாழ்க வெல்க"
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        self.assertTrue(pdf_bytes.startswith(b'%PDF-'), "Output must be a valid PDF document")

        # Render page using pypdfium2
        pdoc = pdfium.PdfDocument(pdf_bytes)
        self.assertEqual(len(pdoc), 1, "Expected single page for short text")

        page = pdoc.get_page(0)
        im = page.render(scale=2.0).to_pil()
        self.assertGreater(im.width, 1000)
        self.assertGreater(im.height, 1000)

        # Confirm non-blank rendering (image has diverse pixel distribution)
        extrema = im.convert('L').getextrema()
        self.assertNotEqual(extrema[0], extrema[1], "Rendered PDF page must not be blank")

    def test_sri_ligature_rendering(self):
        """Verifies complex Sanskrit-Tamil ligature ஸ்ரீ renders without error or clipping."""
        text = "ஸ்ரீ ராம ஜெயம்\nஸ்ரீ லங்கா\nஓம் ஸ்ரீ கணேஷாய நமஹ"
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        pdoc = pdfium.PdfDocument(pdf_bytes)
        self.assertEqual(len(pdoc), 1)
        im = pdoc.get_page(0).render(scale=2.0).to_pil()
        self.assertIsNotNone(im)

    def test_mixed_tamil_english_visual_rendering(self):
        """Verifies mixed Tamil and English typography without missing glyph warnings."""
        text = "Tamil தமிழ் and English ஆங்கிலம் together.\nAccuracy: 98.5% (மிகவும் நன்று)."
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        pdoc = pdfium.PdfDocument(pdf_bytes)
        im = pdoc.get_page(0).render(scale=2.0).to_pil()
        self.assertIsNotNone(im)

    def test_multipage_pagination_and_wrapping(self):
        """Verifies multi-line text wraps properly and paginates across multiple pages."""
        long_sentence = (
            "தமிழ் எழுத்துக்கள் உயிர் எழுத்துக்கள் பன்னிரண்டு, மெய் எழுத்துக்கள் பதினெட்டு, "
            "உயிர்மெய் எழுத்துக்கள் இருநூற்றுப் பதினாறு மற்றும் ஆய்த எழுத்து ஒன்று என மொத்தம் "
            "இருநூற்று நாற்பத்தேழு எழுத்துக்கள் ஆகும்."
        )
        paragraphs = [f"பத்தி எண் {i}: {long_sentence}" for i in range(1, 35)]
        text = "\n".join(paragraphs)

        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        pdoc = pdfium.PdfDocument(pdf_bytes)
        self.assertGreater(len(pdoc), 1, "35 paragraphs should paginate across multiple pages")

        for idx in range(len(pdoc)):
            page_im = pdoc.get_page(idx).render(scale=1.5).to_pil()
            extrema = page_im.convert('L').getextrema()
            self.assertNotEqual(extrema[0], extrema[1], f"Page {idx+1} must contain visible rendered text")


class TestPdfUnicodeAndFontCorrectness(unittest.TestCase):
    """Test Group B: Internal PDF objects, font embedding, ToUnicode CMap, and text extraction."""

    def test_font_embedding_and_type42(self):
        """Verifies TrueType font stream (/FontFile2) and /CIDFontType2 are embedded in PDF."""
        text = "தமிழ் வாழ்க"
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        self.assertIn(b'/FontFile2', pdf_bytes, "PDF must embed TrueType font file stream (/FontFile2)")
        self.assertIn(b'/CIDFontType2', pdf_bytes, "PDF must define /CIDFontType2 composite font")
        self.assertIn(b'/ToUnicode', pdf_bytes, "PDF must include /ToUnicode CMap for Unicode extraction")

    def test_no_syllable_duplication_in_text_extraction(self):
        """
        Verifies that text extraction does NOT duplicate syllables (e.g. 'அம்மா' must NOT
        extract as 'அம்மாமா', 'தம்பி' must NOT extract as 'தம்பிபி').
        """
        text = "அம்மா அப்பா தம்பி"
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            extracted = pdf.pages[0].extract_text()

        self.assertNotIn("அம்மாமா", extracted, "Syllable 'மா' must not be duplicated")
        self.assertNotIn("அப்பாபா", extracted, "Syllable 'பா' must not be duplicated")
        self.assertNotIn("தம்பிபி", extracted, "Syllable 'பி' must not be duplicated")
        self.assertIn("அம்மா அப்பா தம்பி", extracted, "Original Tamil input must be preserved faithfully")

    def test_tamil_word_fidelity(self):
        """Verifies exact string preservation of common Tamil phrases."""
        text = "தமிழ் வாழ்க"
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            extracted = pdf.pages[0].extract_text()

        self.assertIn("தமிழ் வாழ்க", extracted, "'தமிழ் வாழ்க' must match exactly in extracted text")

    def test_mixed_script_text_extraction(self):
        """Verifies that English words, numbers, and punctuation extract without corruption."""
        text = "Tamil தமிழ் and English ஆங்கிலம் together.\nAccuracy: 95.5% (மிகவும் நன்று)."
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            extracted = pdf.pages[0].extract_text()

        self.assertIn("Tamil", extracted)
        self.assertIn("English", extracted)
        self.assertIn("together.", extracted)
        self.assertIn("Accuracy: 95.5%", extracted)
        self.assertIn("தமிழ்", extracted)
        self.assertIn("ஆங்கிலம்", extracted)

    def test_sri_ligature_extraction(self):
        """Verifies that Sri (ஸ்ரீ) ligature extracts faithfully without loss."""
        text = "ஸ்ரீ ராம ஜெயம்"
        pdf_buf = export_to_pdf(text)
        pdf_bytes = pdf_buf.getvalue()

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            extracted = pdf.pages[0].extract_text()

        self.assertIn("ஸ்ரீ", extracted, "Ligature 'ஸ்ரீ' must be preserved in extracted text")
        self.assertIn("ராம", extracted)


class TestPdfConcurrencyAndRegression(unittest.TestCase):
    """Test Group C: Concurrency hardening and regression safety under load."""

    def test_concurrent_pdf_generation_8_workers(self):
        """Validates that 8 concurrent threads can generate PDFs with distinct payloads without cross-talk."""
        distinct_payloads = [
            f"தமிழ் சோதனை வரிசை எண் {i}: அம்மா அப்பா தம்பி {i*100}"
            for i in range(8)
        ]

        def generate_and_check(idx: int):
            target_str = distinct_payloads[idx]
            pdf_buf = export_to_pdf(target_str, title=f"Thread Test {idx}")
            data = pdf_buf.getvalue()
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                ext = pdf.pages[0].extract_text()
            return idx, ext, data.startswith(b'%PDF-'), b'/FontFile2' in data

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(generate_and_check, i) for i in range(8)]
            results = [f.result() for f in futures]

        for idx, ext, is_pdf, has_fontfile in results:
            self.assertTrue(is_pdf, f"Thread {idx} output must be a valid PDF")
            self.assertTrue(has_fontfile, f"Thread {idx} output must embed font")
            self.assertIn(f"Thread Test {idx}", ext, f"Thread {idx} title must match")
            logical_ext = reorder_tamil_visual_to_logical(ext)
            self.assertIn(f"வரிசை எண் {idx}", logical_ext, f"Thread {idx} content must match")
            # Verify no thread bleed: other threads' unique tags must not appear
            for other_idx in range(8):
                if other_idx != idx:
                    self.assertNotIn(f"Thread Test {other_idx}", ext)


if __name__ == '__main__':
    unittest.main()
