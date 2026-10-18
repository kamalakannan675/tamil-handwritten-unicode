"""
compile_multi_split_comparison.py

Compiles and compares training & evaluation results across all 4 split configurations:
- 70/30: 55,000 train / 7,870 val / 28,080 test
- 80/20: 64,584 train / 7,176 val / 19,190 test
- 85/15: 69,638 train / 7,738 val / 13,574 test
- 90/10: 73,710 train / 8,190 val / 9,050 test

Pure Python standard library implementation (no pandas dependency).
Outputs:
- outputs/reports/multi_split_comparison.csv
- outputs/reports/multi_split_comparison.json
- outputs/reports/multi_split_comparison.md
"""

import os
import sys
import json
import csv

def format_table(headers, rows):
    """Formats headers and rows as an aligned plain-text table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    data_lines = [
        " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{sep_line}\n" + "\n".join(data_lines)

def format_markdown_table(headers, rows):
    """Formats headers and rows as a GitHub-flavored Markdown table."""
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines = [
        "| " + " | ".join(str(val) for val in row) + " |"
        for row in rows
    ]
    return f"{header_line}\n{sep_line}\n" + "\n".join(data_lines)

def compile_comparison(base_dir: str = 'outputs/experiments',
                       reports_dir: str = 'outputs/reports') -> list:
    os.makedirs(reports_dir, exist_ok=True)
    splits = ['70_30', '80_20', '85_15', '90_10']
    rows = []

    # Fallback paths for 70/30 production model if not in outputs/experiments/70_30
    fallback_70_30 = {
        'metrics': 'outputs/reports/experiments/augmented_light_full/metrics.json',
        'metadata': 'model/checkpoints/training_metadata.json'
    }

    for s in splits:
        met_path = os.path.join(base_dir, s, 'metrics.json')
        meta_path = os.path.join(base_dir, s, 'training_metadata.json')

        if not os.path.exists(met_path) and s == '70_30':
            if os.path.exists(fallback_70_30['metrics']):
                met_path = fallback_70_30['metrics']
            if os.path.exists(fallback_70_30['metadata']):
                meta_path = fallback_70_30['metadata']

        if not os.path.exists(met_path):
            print(f"[-] Split {s}: Metrics not found at {met_path}")
            continue

        with open(met_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

        dataset_split = meta.get('dataset_split', {})
        train_samples = dataset_split.get('train_samples_used', 'N/A')
        val_samples = dataset_split.get('val_samples_used', 'N/A')
        test_samples = metrics.get('total_test_samples', 'N/A')

        best_epoch = meta.get('best_epoch', 'N/A')
        best_val_acc = meta.get('best_val_accuracy', 'N/A')
        top1 = metrics.get('top1_accuracy_percent', 'N/A')
        top3 = metrics.get('top3_accuracy_percent', 'N/A')
        macro_prec = metrics.get('macro_precision', 'N/A')
        macro_rec = metrics.get('macro_recall', 'N/A')
        macro_f1 = metrics.get('macro_f1', 'N/A')
        weighted_f1 = metrics.get('weighted_f1', 'N/A')
        train_time = meta.get('total_training_time_seconds', 'N/A')

        rows.append({
            'Split': s.replace('_', '/'),
            'Train Samples': train_samples,
            'Val Samples': val_samples,
            'Test Samples': test_samples,
            'Best Epoch': best_epoch,
            'Best Val Acc (%)': best_val_acc,
            'Test Top-1 Acc (%)': top1,
            'Test Top-3 Acc (%)': top3,
            'Macro Precision': macro_prec,
            'Macro Recall': macro_rec,
            'Macro F1': macro_f1,
            'Weighted F1': weighted_f1,
            'Training Time (s)': train_time
        })

    if not rows:
        print("[!] No completed experiment metrics found to compile.")
        return []

    headers = list(rows[0].keys())
    matrix = [[r[h] for h in headers] for r in rows]

    print("\n" + "=" * 90)
    print("CONSOLIDATED MULTI-SPLIT BENCHMARK COMPARISON")
    print("=" * 90)
    print(format_table(headers, matrix))
    print("=" * 90)

    # Save CSV
    csv_path = os.path.join(reports_dir, 'multi_split_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Saved CSV: {csv_path}")

    # Save JSON
    json_path = os.path.join(reports_dir, 'multi_split_comparison.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    print(f"[OK] Saved JSON: {json_path}")

    # Save Markdown Table
    md_path = os.path.join(reports_dir, 'multi_split_comparison.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Split Benchmark Comparison (uTHCD Tamil OCR)\n\n")
        f.write(format_markdown_table(headers, matrix))
        f.write("\n")
    print(f"[OK] Saved Markdown: {md_path}")

    return rows

if __name__ == '__main__':
    compile_comparison()
