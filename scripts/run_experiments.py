"""
run_experiments.py - Multi-model experiment runner and comparison generator.

Orchestrates and compiles reports across:
- Experiment A: Baseline CNN (No Augmentation - 5K/5 epochs)
- Experiment B: Augmented Light (Light Augmentation - 5K/5 epochs)
- Experiment C: Augmented Strong (Strong Augmentation - 5K/5 epochs)
- Experiment D: Augmented Light Full (Full 55,000 Training Samples, 7,870 Validation, Extended Epochs)

All models are evaluated on the complete 28,080-image unaugmented test set.
Generates model_comparison.json and model_comparison.csv.
"""

import os
import sys
import json
import csv
import argparse
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.unicode_mapping import NUM_CLASSES

EXPERIMENTS_DIR = 'outputs/reports/experiments'
CHECKPOINTS_DIR = 'model/checkpoints/experiments'

def compile_comparison_report(output_dir: str = 'outputs/reports') -> Dict[str, Any]:
    """
    Compiles comparison report from all completed experiments in outputs/reports/experiments/.
    Saves model_comparison.json and model_comparison.csv with full metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    exp_order = ['baseline', 'augmented_light', 'augmented_strong', 'augmented_light_full']
    exp_labels = {
        'baseline': 'Baseline CNN (No Augmentation - 5K/5 ep)',
        'augmented_light': 'Augmented Light (5K/5 ep)',
        'augmented_strong': 'Augmented Strong (5K/5 ep)',
        'augmented_light_full': 'Augmented Light (Full Training - 55K)'
    }

    comparison_rows = []

    for exp_id in exp_order:
        rep_dir = os.path.join(EXPERIMENTS_DIR, exp_id)
        ckpt_dir = os.path.join(CHECKPOINTS_DIR, exp_id)
        metrics_file = os.path.join(rep_dir, 'metrics.json')
        meta_file = os.path.join(ckpt_dir, 'training_metadata.json')

        if not os.path.exists(metrics_file):
            continue

        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        training_samples = 5000
        val_samples = 1000
        epochs = 5
        training_time = 0.0
        augmentation = 'none'

        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                training_samples = meta.get('dataset_split', {}).get('train_samples_used', training_samples)
                val_samples = meta.get('dataset_split', {}).get('val_samples_used', val_samples)
                epochs = meta.get('epochs_completed', meta.get('epochs', epochs))
                training_time = meta.get('total_training_time_seconds', 0.0)
                augmentation = meta.get('augmentation_mode', exp_id.replace('augmented_', ''))

        top1 = metrics.get('top1_accuracy_percent', 0.0)
        top3 = metrics.get('top3_accuracy_percent', 0.0)
        macro_prec = metrics.get('macro_precision', 0.0)
        macro_rec = metrics.get('macro_recall', 0.0)
        macro_f1 = metrics.get('macro_f1', 0.0)
        weighted_f1 = metrics.get('weighted_f1', 0.0)
        total_test = metrics.get('total_test_samples', 28080)

        row = {
            'experiment_id': exp_id,
            'model_name': exp_labels.get(exp_id, exp_id),
            'training_samples': training_samples,
            'validation_samples': val_samples,
            'epochs': epochs,
            'augmentation': augmentation,
            'total_test_samples': total_test,
            'top1_accuracy_percent': top1,
            'top3_accuracy_percent': top3,
            'macro_precision': macro_prec,
            'macro_recall': macro_rec,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'training_time_seconds': training_time
        }
        comparison_rows.append(row)

    # Determine best model based on Top-1 test accuracy
    best_model = None
    baseline_top1 = None
    if comparison_rows:
        for r in comparison_rows:
            if r['experiment_id'] == 'baseline':
                baseline_top1 = r['top1_accuracy_percent']
        sorted_by_top1 = sorted(comparison_rows, key=lambda x: x['top1_accuracy_percent'], reverse=True)
        best_model = sorted_by_top1[0]

    augmentation_improved = False
    improvement_delta = 0.0
    if best_model and baseline_top1 is not None and best_model['experiment_id'] != 'baseline':
        if best_model['top1_accuracy_percent'] > baseline_top1:
            augmentation_improved = True
            improvement_delta = round(best_model['top1_accuracy_percent'] - baseline_top1, 2)

    report = {
        'project_title': 'Tamil Handwritten Character Recognition and Unicode Conversion System',
        'comparison_table': comparison_rows,
        'baseline_top1_percent': baseline_top1,
        'best_model_id': best_model['experiment_id'] if best_model else None,
        'best_model_name': best_model['model_name'] if best_model else None,
        'best_top1_accuracy_percent': best_model['top1_accuracy_percent'] if best_model else None,
        'augmentation_improved_accuracy': augmentation_improved,
        'top1_improvement_delta_percent': improvement_delta,
        'evaluation_scope': 'Complete 28,080-image unaugmented test set (180 samples/class across 156 classes)'
    }

    # Save JSON
    json_path = os.path.join(output_dir, 'model_comparison.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save CSV with full requested columns
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Experiment', 'Training Samples', 'Validation Samples', 'Epochs',
            'Augmentation', 'Top-1 (%)', 'Top-3 (%)', 'Macro Precision',
            'Macro Recall', 'Macro F1', 'Weighted F1', 'Training Time (s)'
        ])
        for r in comparison_rows:
            writer.writerow([
                r['model_name'],
                r['training_samples'],
                r['validation_samples'],
                r['epochs'],
                r['augmentation'],
                f"{r['top1_accuracy_percent']:.2f}%",
                f"{r['top3_accuracy_percent']:.2f}%",
                f"{r['macro_precision']:.4f}",
                f"{r['macro_recall']:.4f}",
                f"{r['macro_f1']:.4f}",
                f"{r['weighted_f1']:.4f}",
                f"{r['training_time_seconds']:.2f}"
            ])

    print("=" * 88)
    print("MODEL COMPARISON SUMMARY (Tamil Handwritten Unicode OCR)")
    print("=" * 88)
    print(f"{'Model':<38}{'Aug':<8}{'Train/Val':<14}{'Top-1 (%)':<11}{'Top-3 (%)':<11}{'Macro F1':<9}")
    print("-" * 88)
    for r in comparison_rows:
        tv_str = f"{r['training_samples']}/{r['validation_samples']}"
        print(f"{r['model_name']:<38}{r['augmentation']:<8}{tv_str:<14}{r['top1_accuracy_percent']:<11.2f}{r['top3_accuracy_percent']:<11.2f}{r['macro_f1']:<9.4f}")
    print("=" * 88)
    if best_model:
        print(f"Best Model: {best_model['model_name']} (Top-1: {best_model['top1_accuracy_percent']}%)")
        if augmentation_improved:
            print(f"Outcome: Augmentation IMPROVED performance by +{improvement_delta}% over baseline.")
        else:
            print(f"Outcome: Baseline remained superior/equivalent (difference: {improvement_delta}%).")
    print(f"Saved: {os.path.abspath(json_path)}")
    print(f"Saved: {os.path.abspath(csv_path)}")

    return report

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model comparison utility")
    parser.add_argument('--action', type=str, default='compare',
                        choices=['compare'],
                        help="Action to perform")
    parser.add_argument('--output-dir', type=str, default='outputs/reports')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if args.action == 'compare':
        compile_comparison_report(output_dir=args.output_dir)
