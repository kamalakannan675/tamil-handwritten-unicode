"""
main.py - Flask Web Application for Tamil Handwritten Character Recognition.

Provides:
- GET / : Web user interface (Image upload, interactive canvas, result screen, exports)
- POST /api/predict : Character recognition endpoint (supports file upload & base64 canvas)
- POST /api/export/txt : UTF-8 Tamil text download
- POST /api/export/docx : Microsoft Word (.docx) download
- POST /api/export/pdf : PDF document download with Tamil font support
- GET /api/health : Application & model status
"""

import os
import sys
import io
import re
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image

import torch
# Bound thread allocations for resource-constrained Render free-tier container
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.inference import get_inference_engine
from app.unicode_converter import export_to_txt, export_to_docx, export_to_pdf
from model.unicode_mapping import NUM_CLASSES

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB limit
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', num_classes=NUM_CLASSES)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route('/api/health', methods=['GET'])
def health():
    try:
        ocr_engine = get_full_page_ocr_engine()
        ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'crnn', 'best_model.pt')
        vocab_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'tamil_ctc_char_vocab.json')
        
        ckpt_exists = os.path.exists(ckpt_path)
        ckpt_size = os.path.getsize(ckpt_path) if ckpt_exists else 0
        vocab_exists = os.path.exists(vocab_path)
        
        return jsonify({
            'status': 'healthy' if ocr_engine is not None else 'unhealthy',
            'ocr_engine_loaded': ocr_engine is not None,
            'checkpoint_exists': ckpt_exists,
            'checkpoint_size_bytes': ckpt_size,
            'vocab_exists': vocab_exists,
            'load_error': _ocr_load_error,
            'num_classes': NUM_CLASSES,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal Server Error: {e}", exc_info=True)
    return jsonify({
        'success': False,
        'error': f'Internal Server Error: {str(e)}'
    }), 500

@app.errorhandler(Exception)
def unhandled_exception(e):
    app.logger.error(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify({
        'success': False,
        'error': f'Server Error: {str(e)}'
    }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        engine = get_inference_engine()
        if engine.model is None:
            err = getattr(engine, 'load_error', 'Model weights could not be loaded.')
            return jsonify({'success': False, 'error': f'Model unavailable: {err}'}), 500
        image_bytes = None

        # Check 1: Multipart file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No image selected for upload'}), 400

            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': f"Unsupported file extension. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }), 400

            image_bytes = file.read()

        # Check 2: JSON payload (e.g. from HTML5 canvas base64)
        elif request.is_json:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'success': False, 'error': 'Missing image data in JSON request'}), 400

            b64_data = data['image']
            # Strip data URI prefix if present
            if ',' in b64_data:
                b64_data = b64_data.split(',', 1)[1]

            try:
                image_bytes = base64.b64decode(b64_data)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Invalid base64 encoding: {str(e)}'}), 400

        else:
            return jsonify({'success': False, 'error': 'Request must be multipart/form-data or application/json'}), 400

        if not image_bytes or len(image_bytes) == 0:
            return jsonify({'success': False, 'error': 'Empty image payload received'}), 400

        # Validate that bytes represent a valid decodable image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            # Reopen after verify (PIL requirement)
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Corrupted or unreadable image file: {str(e)}'}), 400

        # Run inference
        result = engine.predict(pil_img)
        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"Inference error: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': f'Internal inference error: {str(e)}'}), 500

# Global cached instance of Full-Page OCR Engine
_full_page_ocr_engine = None
_ocr_load_error = None

def get_full_page_ocr_engine():
    global _full_page_ocr_engine, _ocr_load_error
    if _full_page_ocr_engine is None:
        try:
            from src.ocr.full_page_ocr import TamilFullPageOCR
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'crnn', 'best_model.pt')
            vocab_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'tamil_ctc_char_vocab.json')
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found at {os.path.abspath(ckpt_path)}")
            if os.path.getsize(ckpt_path) < 1000:
                raise ValueError(f"Checkpoint at {ckpt_path} is an un-downloaded Git LFS pointer ({os.path.getsize(ckpt_path)} bytes).")
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f"Vocabulary not found at {os.path.abspath(vocab_path)}")
                
            _full_page_ocr_engine = TamilFullPageOCR(
                checkpoint_path=os.path.abspath(ckpt_path),
                vocab_path=os.path.abspath(vocab_path),
                device='cpu'
            )
            _ocr_load_error = None
        except Exception as e:
            app.logger.error(f"Failed to initialize FullPageOCR: {e}", exc_info=True)
            _ocr_load_error = str(e)
            _full_page_ocr_engine = None
    return _full_page_ocr_engine

@app.route('/api/ocr/test_sample', methods=['GET'])
def test_sample():
    """Diagnostic route to test Full-Page OCR latency and functionality on Render."""
    try:
        import time
        engine = get_full_page_ocr_engine()
        if engine is None:
            return jsonify({'success': False, 'error': f'Engine not loaded: {_ocr_load_error}'}), 200
            
        t0 = time.time()
        test_im = Image.new('RGB', (200, 64), color=(255, 255, 255))
        res = engine.process_document(test_im)
        res['elapsed_sec'] = time.time() - t0
        res['success'] = True
        return jsonify(res), 200
    except BaseException as e:
        import traceback
        app.logger.error(f"test_sample error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 200

@app.route('/api/ocr/document', methods=['POST'])
def ocr_document():
    """
    Full-Page Handwritten Tamil Document OCR Endpoint.
    Accepts: Multipart image file upload or base64 JSON payload.
    Returns: Complete formatted Tamil Unicode text, line & word bounding boxes, and statistics.
    """
    try:
        engine = get_full_page_ocr_engine()
        if engine is None:
            return jsonify({'success': False, 'error': f'Full-Page OCR engine could not be initialized: {_ocr_load_error}'}), 200

        image_bytes = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No document image selected'}), 400
            if not allowed_file(file.filename):
                return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
            image_bytes = file.read()
        elif request.is_json:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'success': False, 'error': 'Missing image in request'}), 400
            b64_data = data['image']
            if ',' in b64_data:
                b64_data = b64_data.split(',', 1)[1]
            image_bytes = base64.b64decode(b64_data)
        else:
            return jsonify({'success': False, 'error': 'Request must be multipart or json'}), 400

        if not image_bytes:
            return jsonify({'success': False, 'error': 'Empty image payload received'}), 400

        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.load()
        result = engine.process_document(pil_img)
        result['success'] = True
        return jsonify(result), 200

    except BaseException as e:
        import traceback
        tb = traceback.format_exc()
        app.logger.error(f"Full-page OCR error: {e}\n{tb}")
        return jsonify({
            'success': False,
            'error': f'Document OCR processing error: {str(e)}',
            'traceback': tb
        }), 200

@app.route('/api/export/txt', methods=['POST'])
def export_txt():
    data = request.get_json(silent=True) or request.form
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided for export'}), 400

    buf = export_to_txt(text)
    return send_file(
        buf,
        as_attachment=True,
        download_name='tamil_recognized_text.txt',
        mimetype='text/plain; charset=utf-8'
    )

@app.route('/api/export/docx', methods=['POST'])
def export_docx_route():
    data = request.get_json(silent=True) or request.form
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided for export'}), 400

    buf = export_to_docx(text)
    return send_file(
        buf,
        as_attachment=True,
        download_name='tamil_recognized_text.docx',
        mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    )

@app.route('/api/export/pdf', methods=['POST'])
def export_pdf_route():
    data = request.get_json(silent=True) or request.form
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided for export'}), 400

    buf = export_to_pdf(text)
    return send_file(
        buf,
        as_attachment=True,
        download_name='tamil_recognized_text.pdf',
        mimetype='application/pdf'
    )

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({'success': False, 'error': f'Bad Request: {str(e)}'}), 400

@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(413)
def handle_too_large(e):
    return jsonify({'success': False, 'error': 'Uploaded file exceeds maximum limit of 10 MB.'}), 413

@app.errorhandler(500)
@app.errorhandler(Exception)
def handle_server_error(e):
    app.logger.error(f"Unhandled server error: {e}", exc_info=True)
    return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
