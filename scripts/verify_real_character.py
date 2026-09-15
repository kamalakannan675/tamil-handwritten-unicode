"""
scripts/verify_real_character.py - Verifies real handwritten test image inference through Flask.
"""

import os
import sys
import io
import json
import h5py
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app
from model.unicode_mapping import get_unicode_char, get_unicode_codepoints

def run_real_character_test():
    # 1. Extract a real handwritten test sample from HDF5
    with h5py.File('data/hdf5_uTHCD_compressed.h5', 'r') as f:
        y_test = f['Test Data/y_test']
        x_test = f['Test Data/x_test']
        sample_idx = 180  # Class 1 ('அ')
        true_label = int(y_test[sample_idx])
        img_data = x_test[sample_idx]

    img = Image.fromarray(img_data, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    # 2. Test through Flask Test Client
    app.config['TESTING'] = True
    client = app.test_client()

    res = client.post(
        '/api/predict',
        data={'image': (io.BytesIO(buf.getvalue()), 'real_test_char.png')},
        content_type='multipart/form-data'
    )
    assert res.status_code == 200, f"Predict failed with status {res.status_code}"
    data = json.loads(res.data)
    assert data['success'] is True

    pred = data['prediction']
    top3 = data['top3']

    print("=" * 70)
    print("REAL TEST CHARACTER INFERENCE VERIFICATION")
    print("=" * 70)
    print(f"True Label:       Class {true_label} ({get_unicode_char(true_label)} - {get_unicode_codepoints(true_label)})")
    print(f"Predicted Class:  Class {pred['class_id']} ({pred['unicode']} - {pred['codepoints']})")
    print(f"Model Confidence: {pred['confidence_percent']}%")
    top3_strs = [f"{t['unicode']} ({t['confidence_percent']}%)" for t in top3]
    print(f"Top-3 Candidates: {top3_strs}")
    assert pred['class_id'] == true_label, f"Mismatch: expected {true_label}, got {pred['class_id']}"
    print("[PASS] EXACT MATCH ON REAL HANDWRITTEN TEST SAMPLE!")

    # 3. Test Editing & Export
    edited_text = pred['unicode'] + 'ன்பு'  # "அன்பு" (Love)
    for fmt in ['txt', 'docx', 'pdf']:
        exp_res = client.post(f'/api/export/{fmt}', json={'text': edited_text})
        assert exp_res.status_code == 200
        assert 'attachment' in exp_res.headers.get('Content-Disposition', '')
        print(f"[PASS] Export {fmt.upper()}: Valid file generated ({len(exp_res.data)} bytes)")

    print("=" * 70)
    print("ALL REAL GLYPH INFERENCE AND EXPORT CHECKS PASSED (100%)")
    print("=" * 70)

if __name__ == '__main__':
    run_real_character_test()
