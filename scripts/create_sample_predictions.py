"""
create_sample_predictions.py - Run real test dataset predictions against the trained model.

Displays actual test samples, ground truth classes & Unicode characters,
predicted classes & Unicode characters, model confidence %, and correctness status.
Saves structured predictions to outputs/predictions/sample_predictions.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.dataset import uTHCDDataset
from model.model import uTHCDNet
from model.unicode_mapping import get_unicode_char, get_unicode_codepoints, NUM_CLASSES

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def run_sample_predictions(checkpoint_path: str = 'model/checkpoints/best_model.pt',
                           hdf5_path: str = 'data/hdf5_uTHCD_compressed.h5',
                           num_samples: int = 20,
                           seed: int = 42):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' does not exist. Please train a model first.")
        sys.exit(1)

    device = torch.device('cpu')
    model = uTHCDNet(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    test_dataset = uTHCDDataset(hdf5_path=hdf5_path, split='test', seed=seed)

    # Select representative samples
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(len(test_dataset), size=num_samples, replace=False)

    print("=" * 75)
    print("REAL TEST SET SAMPLE PREDICTIONS (Tamil Handwritten Unicode OCR)")
    print("=" * 75)

    results = []

    for rank, idx in enumerate(sample_indices, 1):
        tensor_img, true_label = test_dataset[idx]
        batch_x = tensor_img.unsqueeze(0).to(device)  # (1, 1, 64, 64)

        with torch.no_grad():
            top_indices, top_probs = model.predict_top_k(batch_x, k=3)

        pred_label = int(top_indices[0, 0].item())
        confidence = float(top_probs[0, 0].item()) * 100.0
        status = "CORRECT" if pred_label == true_label else "INCORRECT"

        actual_char = get_unicode_char(true_label)
        actual_code = get_unicode_codepoints(true_label)
        pred_char = get_unicode_char(pred_label)
        pred_code = get_unicode_codepoints(pred_label)

        # Build top-3 alternatives
        top3_list = []
        for k_i in range(3):
            cid = int(top_indices[0, k_i].item())
            prob = float(top_probs[0, k_i].item()) * 100.0
            top3_list.append({
                'rank': k_i + 1,
                'class_id': cid,
                'char': get_unicode_char(cid),
                'codepoints': get_unicode_codepoints(cid),
                'confidence_percent': round(prob, 2)
            })

        print(f"Sample #{rank:<2} [Index {idx}]:")
        print(f"  Actual Class:    {true_label:<4} ({actual_char} | {actual_code})")
        print(f"  Predicted Class: {pred_label:<4} ({pred_char} | {pred_code})")
        print(f"  Model Confidence: {confidence:.2f}%")
        print(f"  Status:          {status}")
        print(f"  Top 3:           1. {top3_list[0]['char']} ({top3_list[0]['confidence_percent']}%), "
              f"2. {top3_list[1]['char']} ({top3_list[1]['confidence_percent']}%), "
              f"3. {top3_list[2]['char']} ({top3_list[2]['confidence_percent']}%)")
        print("-" * 75)

        results.append({
            'sample_index': int(idx),
            'actual_class': true_label,
            'actual_unicode': actual_char,
            'actual_codepoint': actual_code,
            'predicted_class': pred_label,
            'predicted_unicode': pred_char,
            'predicted_codepoint': pred_code,
            'confidence_percent': round(confidence, 2),
            'status': status,
            'top3': top3_list
        })

    os.makedirs('outputs/predictions', exist_ok=True)
    out_file = 'outputs/predictions/sample_predictions.json'
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved structured sample predictions to: {os.path.abspath(out_file)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Create sample predictions from real test samples")
    parser.add_argument('--checkpoint', type=str, default='model/checkpoints/best_model.pt')
    parser.add_argument('--dataset', type=str, default='data/hdf5_uTHCD_compressed.h5')
    parser.add_argument('--num-samples', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='outputs/predictions/sample_predictions.json')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_sample_predictions(
        checkpoint_path=args.checkpoint,
        hdf5_path=args.dataset,
        num_samples=args.num_samples,
        seed=args.seed
    )
    if args.output != 'outputs/predictions/sample_predictions.json':
        import shutil
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        shutil.copyfile('outputs/predictions/sample_predictions.json', args.output)
        print(f"Copied predictions to: {os.path.abspath(args.output)}")

