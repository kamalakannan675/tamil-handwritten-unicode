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
    engine = get_inference_engine()
    return jsonify({
        'status': 'healthy',
        'model_loaded': engine.model is not None,
        'checkpoint_path': engine.checkpoint_path,
        'num_classes': NUM_CLASSES,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])

def predict():
    try:
        engine = get_inference_engine()
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
