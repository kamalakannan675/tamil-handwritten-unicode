"""
test_app.py - Unit tests for Flask web application routes and endpoints.
"""

import unittest
import io
import json
import base64
from PIL import Image

from app.main import app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        """Verify home page loads successfully with 200 OK."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Tamil Handwritten OCR', response.data)
        self.assertIn(b'Upload Image', response.data)
        self.assertIn(b'Draw Character', response.data)

    def test_health_route(self):
        """Verify health check endpoint returns valid JSON."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['num_classes'], 156)

    def test_predict_multipart_valid_image(self):
        """Verify predict endpoint handles valid multipart image."""
        # Create small test image
        img = Image.new('RGB', (64, 64), color=(255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        response = self.client.post(
            '/api/predict',
            data={'image': (img_bytes, 'test_sample.png')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)
        self.assertIn('unicode', data['prediction'])
        self.assertIn('confidence', data['prediction'])
        self.assertIn('top3', data)
        self.assertEqual(len(data['top3']), 3)

    def test_predict_invalid_extension(self):
        """Verify rejection of disallowed file formats."""
        txt_file = io.BytesIO(b"Not an image")
        response = self.client.post(
            '/api/predict',
            data={'image': (txt_file, 'test.exe')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('Unsupported file extension', data['error'])

    def test_predict_empty_payload(self):
        """Verify error handling on empty POST request."""
        response = self.client.post('/api/predict', data={})
        self.assertEqual(response.status_code, 400)

    def test_export_txt(self):
        """Verify TXT export returns UTF-8 plain text file."""
        response = self.client.post(
            '/api/export/txt',
            json={'text': 'தமிழ்'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers.get('Content-Disposition', ''))
        self.assertEqual(response.data.decode('utf-8'), 'தமிழ்')

    def test_export_docx(self):
        """Verify DOCX export returns valid word document file."""
        response = self.client.post(
            '/api/export/docx',
            json={'text': 'க'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers.get('Content-Disposition', ''))

    def test_export_pdf(self):
        """Verify PDF export returns valid PDF file."""
        response = self.client.post(
            '/api/export/pdf',
            json={'text': 'தமிழ்'}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn('attachment', response.headers.get('Content-Disposition', ''))
        self.assertTrue(response.data.startswith(b'%PDF'))

    def test_predict_base64_canvas(self):
        """Verify predict endpoint handles base64 canvas data URIs."""
        img = Image.new('RGB', (100, 100), color=(255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        b64_str = "data:image/png;base64," + base64.b64encode(img_bytes.getvalue()).decode('utf-8')

        response = self.client.post(
            '/api/predict',
            json={'image': b64_str}
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('prediction', data)

    def test_predict_jpeg_and_bmp(self):
        """Verify support for JPEG and BMP formats."""
        for fmt, ext in [('JPEG', 'test.jpg'), ('BMP', 'test.bmp')]:
            img = Image.new('RGB', (64, 64), color=(255, 255, 255))
            img_bytes = io.BytesIO()
            img.save(img_bytes, format=fmt)
            img_bytes.seek(0)

            response = self.client.post(
                '/api/predict',
                data={'image': (img_bytes, ext)},
                content_type='multipart/form-data'
            )
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertTrue(data['success'])

    def test_predict_corrupted_image(self):
        """Verify handling of corrupted image files."""
        corrupted_bytes = io.BytesIO(b"\x89PNG\r\n\x1a\nCorruptedGarbageData")
        response = self.client.post(
            '/api/predict',
            data={'image': (corrupted_bytes, 'corrupted.png')},
            content_type='multipart/form-data'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('Corrupted or unreadable image file', data['error'])

    def test_export_empty_text_rejection(self):
        """Verify export endpoints reject empty text with 400."""
        for endpoint in ['/api/export/txt', '/api/export/docx', '/api/export/pdf']:
            response = self.client.post(endpoint, json={'text': ''})
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('No text provided', data['error'])

if __name__ == '__main__':
    unittest.main()

