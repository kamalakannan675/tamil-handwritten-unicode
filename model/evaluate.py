"""
evaluate.py - Comprehensive evaluation script for uTHCD character recognition model.

Evaluates trained checkpoint on actual test dataset (Test Data in HDF5).
Calculates:
- Top-1 and Top-3 Accuracy
- Precision, Recall, F1-Score (Macro and Weighted)
- Per-class classification report saved to CSV
- Confusion Matrix visualization saved to PNG
- Top-10 most confused character pairs (Error Analysis report) saved to JSON
"""

import os
import sys
import json
import time
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.dataset import uTHCDDataset
from model.model import uTHCDNet
from model.unicode_mapping import get_unicode_char, get_unicode_codepoints, NUM_CLASSES

def get_tamil_font():
    for font_name in ['Nirmala UI', 'Latha', 'Vijaya', 'Arial Unicode MS']:
        try:
            return FontProperties(family=font_name)
        except Exception:
            continue
    return FontProperties()

def evaluate_model(checkpoint_path: str = 'model/checkpoints/best_model.pt',
                   hdf5_path: str = 'data/hdf5_uTHCD_compressed.h5',
                   max_test: int = 1560,
                   batch_size: int = 32,
                   prefix: str = '',
                   output_dir: str = 'outputs/reports',
                   device_str: str = 'cpu'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)

    print("=" * 60)
    print("uTHCD MODEL TEST EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"Dataset:    {os.path.abspath(hdf5_path)}")
    print(f"Device:     {device}")
    print(f"Max Test:   {max_test if max_test > 0 else 'Full (28,080)'}")
    print("=" * 60)

    # 1. Load Model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Train a model first.")

    model = uTHCDNet(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load Test Dataset & Run Inference (Chunked for ultra-low latency on CPU)
    print(f"Loading and evaluating test samples from HDF5...")
    all_targets = []
    all_preds = []
    top3_correct = 0
    total = 0

    with h5py.File(hdf5_path, 'r') as f:
        x_test_dset = f['Test Data/x_test']
        y_test_dset = f['Test Data/y_test']
        n_total_test = x_test_dset.shape[0]

        eval_count = n_total_test if (max_test is None or max_test <= 0) else min(max_test, n_total_test)
        print(f"Total test samples to evaluate: {eval_count} (out of {n_total_test})")

        chunk_size = 1000
        for start_idx in range(0, eval_count, chunk_size):
            end_idx = min(start_idx + chunk_size, eval_count)
            raw_chunk_x = x_test_dset[start_idx:end_idx]
            raw_chunk_y = y_test_dset[start_idx:end_idx]

            # Vectorized preprocessing
            tensor_chunk = (255.0 - torch.from_numpy(raw_chunk_x).float()) / 255.0
            tensor_chunk = tensor_chunk.unsqueeze(1).to(device)

            with torch.no_grad():
                for b_i in range(0, tensor_chunk.size(0), batch_size):
                    b_end = min(b_i + batch_size, tensor_chunk.size(0))
                    batch_x = tensor_chunk[b_i:b_end]
                    batch_y = raw_chunk_y[b_i:b_end]

                    probs = model.predict_proba(batch_x)
                    top_probs, top_indices = torch.topk(probs, k=3, dim=1)
                    top_indices_np = top_indices.cpu().numpy()

                    for i in range(len(batch_y)):
                        t_lbl = int(batch_y[i])
                        p_lbl = int(top_indices_np[i, 0])
                        all_targets.append(t_lbl)
                        all_preds.append(p_lbl)
                        if t_lbl in top_indices_np[i]:
                            top3_correct += 1
                        total += 1

            pct = (total / eval_count) * 100.0
            print(f"  Progress: {total}/{eval_count} ({pct:.1f}%) evaluated...")

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # 3. Calculate Overall Metrics
    top1_accuracy = (np.sum(all_targets == all_preds) / max(1, total)) * 100.0
    top3_accuracy = (top3_correct / max(1, total)) * 100.0

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0
    )

    print("\n" + "=" * 60)
    print("MEASURED TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Top-1 Accuracy:     {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy:     {top3_accuracy:.2f}%")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f}")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")
    print("=" * 60)

    # 4. Save metrics.json
    metrics_data = {
        'total_test_samples': int(total),
        'top1_accuracy_percent': round(float(top1_accuracy), 2),
        'top3_accuracy_percent': round(float(top3_accuracy), 2),
        'macro_precision': round(float(precision_macro), 4),
        'macro_recall': round(float(recall_macro), 4),
        'macro_f1': round(float(f1_macro), 4),
        'weighted_f1': round(float(f1_weighted), 4),
        'checkpoint_evaluated': os.path.basename(checkpoint_path)
    }
    prefix = prefix + '_' if prefix else ''
    metrics_path = os.path.join(output_dir, f'{prefix}metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved: {metrics_path}")

    # 5. Generate and Save Classification Report CSV
    clf_dict = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    report_csv_path = os.path.join(output_dir, f'{prefix}classification_report.csv')
    with open(report_csv_path, 'w', encoding='utf-8') as f:
        f.write("class_id,tamil_char,codepoints,precision,recall,f1_score,support\n")
        for k, v in clf_dict.items():
            if k.isdigit():
                cid = int(k)
                ch = get_unicode_char(cid)
                cd = get_unicode_codepoints(cid)
                f.write(f"{cid},{ch},{cd},{v['precision']:.4f},{v['recall']:.4f},{v['f1-score']:.4f},{v['support']}\n")
    print(f"Saved: {report_csv_path}")

    # 6. Confusion Matrix and Error Analysis
    present_classes = np.unique(np.concatenate([all_targets, all_preds]))
    cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_CLASSES))

    # Identify most frequent confusion pairs (true != pred)
    confusion_pairs = []
    for true_c in range(NUM_CLASSES):
        for pred_c in range(NUM_CLASSES):
            if true_c != pred_c and cm[true_c, pred_c] > 0:
                confusion_pairs.append({
                    'true_class_id': int(true_c),
                    'true_char': get_unicode_char(true_c),
                    'true_codepoints': get_unicode_codepoints(true_c),
                    'predicted_class_id': int(pred_c),
                    'predicted_char': get_unicode_char(pred_c),
                    'predicted_codepoints': get_unicode_codepoints(pred_c),
                    'error_count': int(cm[true_c, pred_c])
                })

    confusion_pairs.sort(key=lambda x: x['error_count'], reverse=True)
    top_confusions = confusion_pairs[:20]

    error_analysis_path = os.path.join(output_dir, f'{prefix}error_analysis.json')
    with open(error_analysis_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_misclassifications': int(np.sum(all_targets != all_preds)),
            'most_frequently_confused_pairs': top_confusions
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {error_analysis_path}")

    # 7. Plot Confusion Matrix
    cm_path = os.path.join(output_dir, f'{prefix}confusion_matrix.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'156-Class Tamil Handwritten Character Confusion Matrix ({total} Samples)', fontsize=12, fontweight='bold')
    plt.colorbar()
    plt.xlabel('Predicted Class ID', fontsize=10)
    plt.ylabel('True Class ID', fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")

    return metrics_data

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained uTHCD model")
    parser.add_argument('--checkpoint-path', type=str, default='model/checkpoints/best_model.pt')
    parser.add_argument('--dataset-path', type=str, default='data/hdf5_uTHCD_compressed.h5')
    parser.add_argument('--max-test', type=int, default=1560,
                        help="Max test samples to evaluate (0 for full 28,080)")
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--output-dir', type=str, default='outputs/reports')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate_model(
        checkpoint_path=args.checkpoint_path,
        hdf5_path=args.dataset_path,
        max_test=args.max_test,
        batch_size=args.batch_size,
        prefix=args.prefix,
        output_dir=args.output_dir,
        device_str=args.device
    )
