"""
test_preflight_image_quality.py - Comprehensive Unit and Integration Tests
for Image Quality Pre-Flight Checks in Tamil Handwritten Unicode OCR.

Tests:
1. Median line height below 20 pixels (severe_low_res warning).
2. Median line height between 20 and 29 pixels (caution_low_res warning).
3. Median line height of 30 pixels or greater (good status, no warning).
4. Image with no detected text lines (no_text_detected status).
5. Large image containing very small text (proves check inspects text lines, not image dimensions).
6. Normal-resolution handwritten page (good status).
7. User's existing notebook regression example (evaluates real 260x178 image).
8. Confirmation that users can continue OCR after a warning (can_proceed=True).
9. Confirmation that pre-flight check does not modify the production checkpoint SHA-256.
10. Flask API endpoint POST /api/ocr/preflight test.
"""

import os
import sys
import io
import hashlib
import unittest
import numpy as np
from PIL import Image, ImageDraw

# Bootstrap: ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr.page_segmentation import PageSegmenter
from app.main import app, get_full_page_ocr_engine

def create_synthetic_page(line_height: int, num_lines: int = 3, img_width: int = 500, img_height: int = None):
    """Creates a synthetic handwritten page with black rectangular strokes on off-white paper."""
    gap = 25
    if img_height is None:
        img_height = 40 + num_lines * (line_height + gap)
    img = Image.new('L', (img_width, img_height), color=245)
    draw = ImageDraw.Draw(img)
    
    y = 20
    for _ in range(num_lines):
        # Two word blocks per line
        draw.rectangle([30, y, 180, y + line_height], fill=20)
        draw.rectangle([210, y, 380, y + line_height], fill=20)
        y += line_height + gap
        
    return img

class TestPreflightImageQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segmenter = PageSegmenter(min_line_height=10)
        cls.expected_sha256 = "cb7147047c86cdb256a927b9dac76b8dad91ad12751b2f76a4366dd3877c8eeb"
        cls.ckpt_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "crnn", "best_model.pt")
        cls.notebook_img_path = r"C:\Users\VS\.gemini\antigravity\brain\7bbb0c6e-6666-4556-9311-dd1d71efba7c\.user_uploaded\media_1792341662721_48bf2e00.png"
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_median_line_height_below_20px(self):
        """1. Median line height below 20 pixels triggers severe_low_res warning."""
        img = create_synthetic_page(line_height=14, num_lines=3)
        res = self.segmenter.assess_image_quality(img)
        
        self.assertEqual(res['quality_status'], 'severe_low_res')
        self.assertLess(res['median_line_height'], 20.0)
        self.assertIsNotNone(res['warning'])
        self.assertIn('Severe low-resolution handwriting detected', res['warning'])
        self.assertIn('px', res['warning'])
        self.assertTrue(res['can_proceed'])

    def test_02_median_line_height_between_20_and_29px(self):
        """2. Median line height between 20 and 29 pixels triggers caution_low_res warning."""
        img = create_synthetic_page(line_height=24, num_lines=3)
        res = self.segmenter.assess_image_quality(img)
        
        self.assertEqual(res['quality_status'], 'caution_low_res')
        self.assertGreaterEqual(res['median_line_height'], 20.0)
        self.assertLess(res['median_line_height'], 30.0)
        self.assertIsNotNone(res['warning'])
        self.assertIn('Low-resolution handwriting detected', res['warning'])
        self.assertIn('px', res['warning'])
        self.assertTrue(res['can_proceed'])

    def test_03_median_line_height_30px_or_greater(self):
        """3. Median line height of 30 pixels or greater yields good status with no warning."""
        img = create_synthetic_page(line_height=42, num_lines=3)
        res = self.segmenter.assess_image_quality(img)
        
        self.assertEqual(res['quality_status'], 'good')
        self.assertGreaterEqual(res['median_line_height'], 30.0)
        self.assertIsNone(res['warning'])
        self.assertTrue(res['can_proceed'])

    def test_04_image_with_no_detected_lines(self):
        """4. An image with no detected text lines yields no_text_detected status."""
        blank_white = Image.new('RGB', (400, 300), color=(255, 255, 255))
        res = self.segmenter.assess_image_quality(blank_white)
        
        self.assertEqual(res['quality_status'], 'no_text_detected')
        self.assertEqual(res['line_count'], 0)
        self.assertEqual(res['median_line_height'], 0.0)
        self.assertIn('No clear text lines detected', res['warning'])
        self.assertTrue(res['can_proceed'])

    def test_05_large_image_containing_very_small_text(self):
        """5. A large canvas (1600x1200) containing tiny text correctly triggers low-res warning."""
        # 1600x1200 image where text lines are only 14px high
        img = create_synthetic_page(line_height=14, num_lines=4, img_width=1600, img_height=1200)
        res = self.segmenter.assess_image_quality(img)
        
        # Must evaluate actual text line heights, NOT canvas dimensions
        self.assertIn(res['quality_status'], ['severe_low_res', 'caution_low_res'])
        self.assertLess(res['median_line_height'], 30.0)
        self.assertTrue(res['can_proceed'])

    def test_06_normal_resolution_handwritten_page(self):
        """6. Normal-resolution page (50px lines) passes preflight check cleanly."""
        img = create_synthetic_page(line_height=55, num_lines=4, img_width=800, img_height=600)
        res = self.segmenter.assess_image_quality(img)
        
        self.assertEqual(res['quality_status'], 'good')
        self.assertGreaterEqual(res['median_line_height'], 30.0)
        self.assertIsNone(res['warning'])
        self.assertTrue(res['can_proceed'])

    def test_07_user_notebook_regression_example(self):
        """7. Evaluates pre-flight on user's real notebook test image."""
        if not os.path.exists(self.notebook_img_path):
            self.skipTest("Notebook regression image not found at specified brain path.")
            
        real_img = Image.open(self.notebook_img_path)
        res = self.segmenter.assess_image_quality(real_img)
        
        self.assertEqual(res['line_count'], 5)
        # Measured median is 20.0px, triggering caution_low_res
        self.assertIn(res['quality_status'], ['caution_low_res', 'severe_low_res'])
        self.assertLess(res['median_line_height'], 30.0)
        self.assertIsNotNone(res['warning'])
        self.assertTrue(res['can_proceed'])

    def test_08_users_can_continue_ocr_after_warning(self):
        """8. Confirmation that users can proceed with full OCR after receiving a preflight warning."""
        img = create_synthetic_page(line_height=18, num_lines=2)
        ocr = get_full_page_ocr_engine()
        
        # Preflight warning is active
        preflight = ocr.assess_image_quality(img)
        self.assertTrue(preflight['can_proceed'])
        self.assertIsNotNone(preflight['warning'])
        
        # Document OCR runs and returns complete result without crashing or blocking
        result = ocr.process_document(img)
        self.assertIn('full_text', result)
        self.assertIn('preflight', result)
        self.assertTrue(result['preflight']['can_proceed'])
        self.assertEqual(result['total_lines'], 2)

    def test_09_checkpoint_sha256_unmodified_by_preflight(self):
        """9. Confirmation that preflight checks never alter the production model checkpoint."""
        if not os.path.exists(self.ckpt_path):
            self.skipTest(f"Checkpoint not found at {self.ckpt_path}")
            
        # Run preflight multiple times on different images
        for h in [12, 22, 45]:
            img = create_synthetic_page(line_height=h, num_lines=2)
            self.segmenter.assess_image_quality(img)
            
        # Verify SHA-256 hash using safe chunk iteration
        hasher = hashlib.sha256()
        with open(self.ckpt_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hasher.update(chunk)
        current_hash = hasher.hexdigest()
        
        self.assertEqual(current_hash, self.expected_sha256,
                         f"Checkpoint SHA-256 altered! Expected {self.expected_sha256}, got {current_hash}")

    def test_10_api_preflight_route(self):
        """10. Test POST /api/ocr/preflight HTTP endpoint."""
        img = create_synthetic_page(line_height=22, num_lines=3)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        resp = self.client.post(
            '/api/ocr/preflight',
            data={'image': (buf, 'test_note.png')},
            content_type='multipart/form-data'
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data['success'])
        self.assertTrue(data['can_proceed'])
        self.assertEqual(data['quality_status'], 'caution_low_res')
        self.assertIn('warning', data)

if __name__ == '__main__':
    unittest.main()
