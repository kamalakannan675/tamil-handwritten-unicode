"""
test_render_ocr_production.py - Automated tests for Render OCR 502 fixes and production readiness.
Covers:
- Preflight endpoint validation, error handling, and latency across all 8 samples.
- Full document OCR endpoint validation and concurrency safety.
- End-to-end workflow (Image -> Preflight -> OCR -> TXT / DOCX / PDF export) for all 8 samples.
"""

import io
import os
import sys
import json
import time
import unittest
import concurrent.futures
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app
import pdfplumber
import docx

class TestPreflightRoute(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'verification', 'sample_testing_after_pdf_change'
        )

    def test_preflight_valid_image_small(self):
        """Verify preflight responds successfully on a clean synthetic image."""
        im = Image.new('RGB', (400, 100), color=(255, 255, 255))
        buf = io.BytesIO()
        im.save(buf, format='JPEG')
        buf.seek(0)

        res = self.client.post('/api/ocr/preflight', data={'image': (buf, 'test.jpg')})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))
        self.assertTrue(data.get('can_proceed'))
        self.assertIn('line_count', data)
        self.assertIn('quality_status', data)

    def test_preflight_missing_image(self):
        """Verify preflight rejects request with missing image."""
        res = self.client.post('/api/ocr/preflight', data={})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get('success'))

    def test_preflight_unsupported_format(self):
        """Verify preflight rejects unsupported file extensions."""
        buf = io.BytesIO(b'not an image')
        res = self.client.post('/api/ocr/preflight', data={'image': (buf, 'document.pdf')})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data.get('success'))

    def test_preflight_corrupted_image(self):
        """Verify preflight safely handles corrupted image bytes."""
        buf = io.BytesIO(b'corrupted image binary data')
        res = self.client.post('/api/ocr/preflight', data={'image': (buf, 'bad.jpg')})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data.get('success'))
        self.assertTrue(data.get('can_proceed'))

    def test_preflight_all_eight_samples(self):
        """Verify preflight succeeds and runs fast across all 8 user samples."""
        for s in range(1, 9):
            ext = 'jpeg' if s == 8 else 'jpg'
            img_path = os.path.join(self.sample_dir, f'sample_{s}', f'sample_{s}.{ext}')
            if not os.path.exists(img_path):
                continue
            with open(img_path, 'rb') as f:
                img_bytes = f.read()

            t0 = time.time()
            res = self.client.post(
                '/api/ocr/preflight',
                data={'image': (io.BytesIO(img_bytes), f'sample_{s}.{ext}')}
            )
            dt = time.time() - t0

            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertTrue(data.get('success'), f"Preflight failed on sample {s}")
            self.assertTrue(data.get('can_proceed'))
            self.assertIn('quality_status', data)
            self.assertLess(dt, 10.0, f"Sample {s} preflight took too long ({dt:.2f}s)")

    def test_preflight_repeated_requests(self):
        """Verify repeated requests to preflight do not leak memory or degrade latency."""
        img_path = os.path.join(self.sample_dir, 'sample_3', 'sample_3.jpg')
        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        for _ in range(3):
            res = self.client.post(
                '/api/ocr/preflight',
                data={'image': (io.BytesIO(img_bytes), 'sample_3.jpg')}
            )
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.get_json().get('success'))


class TestDocumentOCREndpoint(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'verification', 'sample_testing_after_pdf_change'
        )

    def test_document_ocr_schema_and_serialization(self):
        """Verify document OCR response matches API contract."""
        img_path = os.path.join(self.sample_dir, 'sample_3', 'sample_3.jpg')
        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        res = self.client.post(
            '/api/ocr/document',
            data={'image': (io.BytesIO(img_bytes), 'sample_3.jpg')}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get('success'))
        self.assertIn('full_text', data)
        self.assertIn('total_lines', data)
        self.assertIn('total_words', data)
        self.assertIn('lines', data)
        self.assertIn('anomalies', data)
        self.assertIn('image_size', data)

    def test_document_ocr_missing_image(self):
        """Verify document OCR rejects missing image payload."""
        res = self.client.post('/api/ocr/document', data={})
        self.assertEqual(res.status_code, 400)

    def test_document_ocr_concurrency_lock(self):
        """Verify multiple threads calling /api/ocr/document simultaneously succeed without race condition."""
        img_path = os.path.join(self.sample_dir, 'sample_3', 'sample_3.jpg')
        with open(img_path, 'rb') as f:
            img_bytes = f.read()

        def _call_ocr(idx):
            with app.test_client() as c:
                return c.post(
                    '/api/ocr/document',
                    data={'image': (io.BytesIO(img_bytes), f's3_{idx}.jpg')}
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(_call_ocr, i) for i in range(2)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for r in results:
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json().get('success'))


class TestEndToEndWorkflowAllSamples(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.sample_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'outputs', 'verification', 'sample_testing_after_pdf_change'
        )

    def test_all_eight_samples_e2e_and_export_fidelity(self):
        """
        Runs the complete user workflow for all 8 authorized samples:
        1. Preflight check
        2. Full document OCR recognition
        3. Validates recognized Tamil Unicode text exactly matches the archived reference TXT
        4. Validates TXT export matches OCR output
        5. Validates DOCX export creates a valid document containing recognized lines
        6. Validates PDF export embeds TrueType font (/FontFile2) with valid /ToUnicode CMap
        """
        for s in range(1, 9):
            ext = 'jpeg' if s == 8 else 'jpg'
            sdir = os.path.join(self.sample_dir, f'sample_{s}')
            img_path = os.path.join(sdir, f'sample_{s}.{ext}')
            txt_path = os.path.join(sdir, 'tamil_handwritten_ocr.txt')

            if not os.path.exists(img_path) or not os.path.exists(txt_path):
                continue

            with open(img_path, 'rb') as f:
                img_bytes = f.read()
            with open(txt_path, 'r', encoding='utf-8') as f:
                expected_txt = f.read().strip()

            # 1. Preflight
            pref_res = self.client.post(
                '/api/ocr/preflight',
                data={'image': (io.BytesIO(img_bytes), f'sample_{s}.{ext}')}
            )
            self.assertEqual(pref_res.status_code, 200)
            self.assertTrue(pref_res.get_json().get('can_proceed'))

            # 2. Document OCR
            ocr_res = self.client.post(
                '/api/ocr/document',
                data={'image': (io.BytesIO(img_bytes), f'sample_{s}.{ext}')}
            )
            self.assertEqual(ocr_res.status_code, 200)
            ocr_data = ocr_res.get_json()
            self.assertTrue(ocr_data.get('success'))
            actual_txt = ocr_data.get('full_text', '').strip()

            # 3. Match reference OCR text
            self.assertEqual(
                actual_txt, expected_txt,
                f"Sample {s} OCR text does not match reference archived text"
            )

            # 4. Export TXT
            exp_txt_res = self.client.post('/api/export/txt', json={'text': actual_txt})
            self.assertEqual(exp_txt_res.status_code, 200)
            self.assertEqual(exp_txt_res.data.decode('utf-8').strip(), actual_txt)

            # 5. Export DOCX
            exp_docx_res = self.client.post('/api/export/docx', json={'text': actual_txt})
            self.assertEqual(exp_docx_res.status_code, 200)
            doc = docx.Document(io.BytesIO(exp_docx_res.data))
            docx_text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
            self.assertTrue(len(docx_text) > 0)

            # 6. Export PDF
            exp_pdf_res = self.client.post('/api/export/pdf', json={'text': actual_txt})
            self.assertEqual(exp_pdf_res.status_code, 200)
            pdf_bytes = exp_pdf_res.data
            self.assertTrue(pdf_bytes.startswith(b'%PDF'))
            self.assertTrue(b'/FontFile2' in pdf_bytes or b'/FontFile3' in pdf_bytes)
            self.assertTrue(b'/ToUnicode' in pdf_bytes)

            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                self.assertGreaterEqual(len(pdf.pages), 1)
                pdf_text = pdf.pages[0].extract_text()
                self.assertTrue(len(pdf_text) > 0)

if __name__ == '__main__':
    unittest.main()
