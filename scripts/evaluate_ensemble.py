"""
evaluate_ensemble.py - Rigorous Evaluation of Multi-Model Ensemble vs. Individual Models.

Evaluates:
1. 70/30 Baseline Model
2. 80/20 Model
3. 85/15 Model
4. 90/10 Model
5. Equal-Weight Ensemble (P = (P_70 + P_80 + P_85 + P_90) / 4)
6. Validation-Tuned Weighted Ensemble (P = w_70*P_70 + w_80*P_80 + w_85*P_85 + w_90*P_90)

Evaluates on:
A) Protocol A (Common Reference Benchmark): 70/30 Test Partition (28,080 test samples)
B) Protocol B (Strictly Leak-Free Held-out Subset): 3,141 samples NEVER seen in the training partition of ANY model.

Outputs:
- outputs/experiments/ensemble/ensemble_metrics.json
- outputs/experiments/ensemble/ensemble_comparison.csv
- outputs/experiments/ensemble/ensemble_classification_report.csv
- outputs/experiments/ensemble/ensemble_confusion_matrix.png
- outputs/experiments/ensemble/ensemble_analysis.md
"""

import os
import gc
import json
import csv
import time
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model import uTHCDNet
from model.unicode_mapping import NUM_CLASSES, get_unicode_char, get_unicode_codepoints

def load_model(path: str, device: torch.device):
    model = uTHCDNet(num_classes=NUM_CLASSES).to(device)
    try:
        ckpt = torch.load(path, map_location=device, mmap=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    model.eval()
    return model

def compute_metrics(y_true, y_pred, top3_correct, total):
    top1_acc = float(np.mean(y_true == y_pred) * 100.0)
    top3_acc = float((top3_correct / total) * 100.0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'top1_accuracy_percent': round(top1_acc, 2),
        'top3_accuracy_percent': round(top3_acc, 2),
        'macro_precision': round(float(p_macro), 4),
        'macro_recall': round(float(r_macro), 4),
        'macro_f1': round(float(f1_macro), 4),
        'weighted_f1': round(float(f1_weighted), 4),
        'total_samples': total
    }

def run_evaluation(device_str: str = 'cpu', batch_size: int = 128):
    out_dir = 'outputs/experiments/ensemble'
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device_str if (device_str != 'cuda' or torch.cuda.is_available()) else 'cpu')

    print(f"=== Running Ensemble Evaluation on {device} (batch_size={batch_size}) ===")

    # 1. Load the 4 models
    ckpt_paths = {
        '70_30': 'model/checkpoints/best_model.pt',
        '80_20': 'outputs/experiments/80_20/best_model.pt',
        '85_15': 'outputs/experiments/85_15/best_model.pt',
        '90_10': 'outputs/experiments/90_10/best_model.pt'
    }

    t_start_load = time.time()
    models = {name: load_model(path, device) for name, path in ckpt_paths.items()}
    load_time = time.time() - t_start_load
    print(f"[OK] Loaded all 4 models in {load_time:.2f}s")

    # 2. Load 70/30 Test Partition
    with h5py.File('data/hdf5_uTHCD_compressed.h5', 'r') as f:
        test_images = f['Test Data/x_test'][:]
        test_labels = f['Test Data/y_test'][:]

    total_samples = len(test_images)
    print(f"[OK] Loaded {total_samples} test samples from 70/30 test set.")

    # 3. Batch inference to collect logits and probabilities for each model
    num_batches = int(np.ceil(total_samples / batch_size))
    probs_dict = {name: [] for name in models.keys()}

    t_start_inf = time.time()
    with torch.inference_mode():
        for b_idx in range(num_batches):
            start = b_idx * batch_size
            end = min(start + batch_size, total_samples)
            # Correct uTHCD normalization: (255.0 - raw_chunk_x) / 255.0
            batch_np = (255.0 - test_images[start:end].astype(np.float32)) / 255.0
            batch_t = torch.from_numpy(batch_np).unsqueeze(1).to(device)

            for name, model in models.items():
                logits = model(batch_t)
                p = F.softmax(logits, dim=1).cpu().numpy()
                probs_dict[name].append(p)

    inf_time_total = time.time() - t_start_inf
    print(f"[OK] Completed inference across all 4 models in {inf_time_total:.2f}s ({inf_time_total/total_samples*1000:.2f} ms/sample)")

    # Concatenate probabilities
    for name in probs_dict:
        probs_dict[name] = np.concatenate(probs_dict[name], axis=0)

    # 4. Generate Ensemble Probability Matrices
    # Equal weight:
    p_equal = (probs_dict['70_30'] + probs_dict['80_20'] + probs_dict['85_15'] + probs_dict['90_10']) / 4.0

    # Validation-tuned weights:
    # Based on validation accuracy (70: 96.37%, 80: 96.99%, 85: 96.70%, 90: 96.68%)
    # w_70=0.20, w_80=0.25, w_85=0.25, w_90=0.30
    w = {'70_30': 0.20, '80_20': 0.25, '85_15': 0.25, '90_10': 0.30}
    p_weighted = (
        w['70_30'] * probs_dict['70_30'] +
        w['80_20'] * probs_dict['80_20'] +
        w['85_15'] * probs_dict['85_15'] +
        w['90_10'] * probs_dict['90_10']
    )

    all_prob_sets = {
        '70/30 (Baseline)': probs_dict['70_30'],
        '80/20 Model': probs_dict['80_20'],
        '85/15 Model': probs_dict['85_15'],
        '90/10 Model': probs_dict['90_10'],
        'Equal-Weight Ensemble': p_equal,
        'Weighted Ensemble': p_weighted
    }

    # Evaluate each on the full 28,080 test samples (Common Benchmark Protocol)
    results_common = []
    print("\n--- RESULTS ON 70/30 COMMON BENCHMARK (28,080 SAMPLES) ---")
    for name, p_matrix in all_prob_sets.items():
        top1_preds = np.argmax(p_matrix, axis=1)
        top3_preds = np.argsort(p_matrix, axis=1)[:, -3:]
        top3_hits = sum(test_labels[i] in top3_preds[i] for i in range(total_samples))

        metrics = compute_metrics(test_labels, top1_preds, top3_hits, total_samples)
        metrics['Configuration'] = name
        results_common.append(metrics)
        print(f"{name:25s} | Top-1: {metrics['top1_accuracy_percent']:.2f}% | Top-3: {metrics['top3_accuracy_percent']:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

    # Evaluate on the 3,141 STRICTLY LEAK-FREE samples
    # Compute the leak-free indices
    import hashlib
    def get_h(img, lbl):
        return f"{hashlib.md5(img.tobytes()).hexdigest()}_{lbl}"

    print("\nIdentifying leak-free samples for Protocol B...")
    train_hashes = set()
    with h5py.File('data/hdf5_uTHCD_compressed.h5', 'r') as f:
        x, y = f['Train Data/x_train'][:55000], f['Train Data/y_train'][:55000]
        for i in range(len(x)): train_hashes.add(get_h(x[i], y[i]))
    with h5py.File('data/hdf5_uTHCD_80_20_split.h5', 'r') as f:
        x, y = f['Train Data/Images'][:64584], f['Train Data/Labels'][:64584]
        for i in range(len(x)): train_hashes.add(get_h(x[i], y[i]))
    with h5py.File('data/hdf5_uTHCD_85_15_split.h5', 'r') as f:
        x, y = f['Train Data/Images'][:69638], f['Train Data/Labels'][:69638]
        for i in range(len(x)): train_hashes.add(get_h(x[i], y[i]))
    with h5py.File('data/hdf5_uTHCD_90_10_split.h5', 'r') as f:
        x, y = f['Train Data/Images'][:73710], f['Train Data/Labels'][:73710]
        for i in range(len(x)): train_hashes.add(get_h(x[i], y[i]))

    unseen_idx = [i for i in range(total_samples) if get_h(test_images[i], test_labels[i]) not in train_hashes]
    unseen_total = len(unseen_idx)
    print(f"Strictly unseen samples across ALL models: {unseen_total}")

    y_true_unseen = test_labels[unseen_idx]
    results_unseen = []
    print("\n--- RESULTS ON STRICTLY LEAK-FREE SUBSET (3,141 SAMPLES) ---")
    for name, p_matrix in all_prob_sets.items():
        p_sub = p_matrix[unseen_idx]
        top1_preds = np.argmax(p_sub, axis=1)
        top3_preds = np.argsort(p_sub, axis=1)[:, -3:]
        top3_hits = sum(y_true_unseen[i] in top3_preds[i] for i in range(unseen_total))

        metrics = compute_metrics(y_true_unseen, top1_preds, top3_hits, unseen_total)
        metrics['Configuration'] = name
        results_unseen.append(metrics)
        print(f"{name:25s} | Top-1: {metrics['top1_accuracy_percent']:.2f}% | Top-3: {metrics['top3_accuracy_percent']:.2f}% | Macro F1: {metrics['macro_f1']:.4f}")

    # Save comparison CSV
    csv_path = os.path.join(out_dir, 'ensemble_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Evaluation_Protocol', 'Configuration', 'Total_Samples', 'Top1_Acc_Percent', 'Top3_Acc_Percent', 'Macro_Precision', 'Macro_Recall', 'Macro_F1', 'Weighted_F1'])
        for r in results_common:
            writer.writerow(['Common_70_30_Test', r['Configuration'], r['total_samples'], r['top1_accuracy_percent'], r['top3_accuracy_percent'], r['macro_precision'], r['macro_recall'], r['macro_f1'], r['weighted_f1']])
        for r in results_unseen:
            writer.writerow(['Strictly_Leak_Free', r['Configuration'], r['total_samples'], r['top1_accuracy_percent'], r['top3_accuracy_percent'], r['macro_precision'], r['macro_recall'], r['macro_f1'], r['weighted_f1']])
    print(f"[OK] Saved CSV: {csv_path}")

    # Save JSON
    json_path = os.path.join(out_dir, 'ensemble_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'common_benchmark_28080': results_common,
            'leak_free_benchmark_3141': results_unseen,
            'weights_used': w
        }, f, indent=2)
    print(f"[OK] Saved JSON: {json_path}")

    # Generate Confusion Matrix for Weighted Ensemble
    p_best = p_weighted
    y_pred_best = np.argmax(p_best, axis=1)
    cm = confusion_matrix(test_labels, y_pred_best, labels=range(NUM_CLASSES))

    cm_path = os.path.join(out_dir, 'ensemble_confusion_matrix.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Four-Model Ensemble Confusion Matrix (28,080 Samples)', fontsize=12, fontweight='bold')
    plt.colorbar()
    plt.xlabel('Predicted Class ID', fontsize=10)
    plt.ylabel('True Class ID', fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[OK] Saved Confusion Matrix: {cm_path}")

    # Generate Classification Report CSV for Weighted Ensemble
    clf_dict = classification_report(test_labels, y_pred_best, output_dict=True, zero_division=0)
    rep_path = os.path.join(out_dir, 'ensemble_classification_report.csv')
    with open(rep_path, 'w', encoding='utf-8') as f:
        f.write("class_id,tamil_char,codepoints,precision,recall,f1_score,support\n")
        for k, v in clf_dict.items():
            if k.isdigit():
                cid = int(k)
                ch = get_unicode_char(cid)
                cd = get_unicode_codepoints(cid)
                f.write(f"{cid},{ch},{cd},{v['precision']:.4f},{v['recall']:.4f},{v['f1-score']:.4f},{v['support']}\n")
    print(f"[OK] Saved Classification Report: {rep_path}")

    return results_common, results_unseen

if __name__ == '__main__':
    run_evaluation()
