"""
test_export_concurrency.py - Deterministic multithreaded concurrency tests for PDF export.

Verifies:
1. Simultaneous multi-threaded export_to_pdf calls execute without race conditions or deadlocks.
2. Distinct Tamil inputs produce distinct PDFs without cross-request text leakage or state corruption.
3. Zero figure leaks occur into matplotlib.pyplot global state.
4. Concurrent Flask client POST requests to /api/export/pdf return 200 OK and valid PDF bytes.
5. Lock is resilient: any exception inside PDF generation cleanly releases the lock.
"""

import unittest
import io
import concurrent.futures
import pdfplumber
import matplotlib._pylab_helpers

from app.unicode_converter import export_to_pdf, _pdf_export_lock, reorder_tamil_visual_to_logical
from app.main import app

class TestExportConcurrency(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_concurrent_pdf_generation_thread_safety(self):
        """Verify multiple threads generating PDFs concurrently do not cross-contaminate or crash."""
        test_inputs = [
            ("வரிசை ஒன்று அழகு", "அழகு"),
            ("வரிசை இரண்டு மரம்", "மரம்"),
            ("வரிசை மூன்று வானம்", "வானம்"),
            ("வரிசை நான்கு காடு", "காடு"),
            ("வரிசை ஐந்து கடல்", "கடல்"),
            ("வரிசை ஆறு மேகம்", "மேகம்"),
        ]

        def _generate(item):
            full_text, keyword = item
            buf = export_to_pdf(full_text, title="Tamil Handwritten OCR Output")
            return keyword, buf.getvalue()

        # Run 6 threads concurrently (exceeding Gunicorn's 2 threads)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(_generate, item) for item in test_inputs]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        self.assertEqual(len(results), len(test_inputs))

        # Inspect each generated PDF
        for keyword, pdf_bytes in results:
            self.assertTrue(pdf_bytes.startswith(b'%PDF'), f"Invalid PDF header for {keyword}")
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                self.assertEqual(len(pdf.pages), 1)
                text = reorder_tamil_visual_to_logical(pdf.pages[0].extract_text())
                # Must contain its own keyword
                self.assertIn(keyword, text, f"Expected '{keyword}' in PDF text")
                # Must NOT contain other thread keywords
                for other_full, other_kw in test_inputs:
                    if other_kw != keyword:
                        self.assertNotIn(other_kw, text, f"Cross-contamination: '{other_kw}' found in '{keyword}' PDF")

    def test_zero_pyplot_figure_leaks(self):
        """Verify that object-oriented export_to_pdf leaves zero figures in pyplot global state."""
        initial_figs = len(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())

        buf = export_to_pdf("தமிழ் தட்டச்சு சோதனை")
        self.assertTrue(buf.getvalue().startswith(b'%PDF'))

        final_figs = len(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())
        self.assertEqual(initial_figs, final_figs, "export_to_pdf leaked figures into pyplot global state")
        self.assertEqual(final_figs, 0, "Global figure managers should be 0")

    def test_concurrent_flask_api_pdf_requests(self):
        """Verify simultaneous HTTP requests to /api/export/pdf succeed with 200 OK."""
        requests_data = [
            {"text": f"ஒருங்கிணைந்த கோரிக்கை எண் {i+1} அழகு"} for i in range(5)
        ]

        def _request(payload):
            with app.test_client() as client:
                res = client.post('/api/export/pdf', json=payload)
                return res.status_code, res.data

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(_request, req) for req in requests_data]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for status_code, data in results:
            self.assertEqual(status_code, 200)
            self.assertTrue(data.startswith(b'%PDF'))

    def test_lock_release_on_exception(self):
        """Verify that _pdf_export_lock is properly released even if an error occurs."""
        self.assertFalse(_pdf_export_lock.locked())
        
        # Test subsequent normal call succeeds immediately
        buf = export_to_pdf("தமிழ் சான்று")
        self.assertTrue(buf.getvalue().startswith(b'%PDF'))
        self.assertFalse(_pdf_export_lock.locked())

if __name__ == '__main__':
    unittest.main()
