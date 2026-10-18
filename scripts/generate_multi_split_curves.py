"""
generate_multi_split_curves.py

Generates publication-quality dual and 4-panel comparative visualization
across all four experimental dataset splits:
- 70/30 (Baseline Benchmark Split)
- 80/20 Split
- 85/15 Split
- 90/10 Split
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_multi_split_plots():
    out_dir = 'outputs/reports'
    os.makedirs(out_dir, exist_ok=True)

    splits = ['70_30', '80_20', '85_15', '90_10']
    split_labels = ['70/30 Split', '80/20 Split', '85/15 Split', '90/10 Split']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    histories = {}

    # Load 70/30 history
    meta_70_30 = 'model/checkpoints/training_metadata.json'
    if os.path.exists(meta_70_30):
        with open(meta_70_30, 'r', encoding='utf-8') as f:
            histories['70_30'] = json.load(f).get('history', [])

    # Load 80/20, 85/15, 90/10 histories
    for s in ['80_20', '85_15', '90_10']:
        h_path = f'outputs/experiments/{s}/training_history.json'
        if os.path.exists(h_path):
            with open(h_path, 'r', encoding='utf-8') as f:
                histories[s] = json.load(f)

    # Load metrics
    metrics = {}
    with open('outputs/reports/multi_split_comparison.json', 'r', encoding='utf-8') as f:
        comp_rows = json.load(f)
        for r in comp_rows:
            key = r['Split'].replace('/', '_')
            metrics[key] = r

    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Validation Accuracy vs Epochs
    ax1 = axes[0, 0]
    for s, label, color in zip(splits, split_labels, colors):
        if s in histories and histories[s]:
            epochs = [r['epoch'] for r in histories[s]]
            val_acc = [r['val_acc'] for r in histories[s]]
            ax1.plot(epochs, val_acc, marker='o', label=f"{label} (Peak: {max(val_acc):.2f}%)", color=color, linewidth=2, markersize=4)
    ax1.set_title('Validation Accuracy (%) vs. Epochs Across Dataset Splits', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', fontsize=10)

    # Panel 2: Training Loss vs Epochs
    ax2 = axes[0, 1]
    for s, label, color in zip(splits, split_labels, colors):
        if s in histories and histories[s]:
            epochs = [r['epoch'] for r in histories[s]]
            train_loss = [r['train_loss'] for r in histories[s]]
            ax2.plot(epochs, train_loss, marker='s', label=f"{label}", color=color, linewidth=2, markersize=3, linestyle='--')
    ax2.set_title('Training Cross-Entropy Loss vs. Epochs Across Dataset Splits', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=10)

    # Panel 3: Test Top-1 and Top-3 Accuracy Comparison
    ax3 = axes[1, 0]
    x_positions = range(len(splits))
    top1_vals = [metrics[s]['Test Top-1 Acc (%)'] for s in splits]
    top3_vals = [metrics[s]['Test Top-3 Acc (%)'] for s in splits]
    width = 0.35

    bar1 = ax3.bar([p - width/2 for p in x_positions], top1_vals, width, label='Top-1 Test Accuracy (%)', color='#2b5c8f')
    bar2 = ax3.bar([p + width/2 for p in x_positions], top3_vals, width, label='Top-3 Test Accuracy (%)', color='#56b4e9')

    for b in bar1:
        yval = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2.0, yval + 0.4, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for b in bar2:
        yval = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2.0, yval + 0.4, f"{yval:.2f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_title('Test Set Evaluation: Top-1 vs. Top-3 Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_xticks(list(x_positions))
    ax3.set_xticklabels(split_labels, fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_ylim(85, 102)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.legend(loc='lower right', fontsize=10)

    # Panel 4: Macro F1-Score & Sample Distribution
    ax4 = axes[1, 1]
    f1_vals = [metrics[s]['Macro F1'] for s in splits]
    train_counts = [metrics[s]['Train Samples'] for s in splits]

    color_line = '#e66101'
    ax4.plot(x_positions, f1_vals, marker='D', color=color_line, linewidth=2.5, markersize=8, label='Macro F1-Score')
    for i, txt in enumerate(f1_vals):
        ax4.annotate(f"{txt:.4f}", (i, txt), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, fontweight='bold', color=color_line)

    ax4.set_title('Macro F1-Score Across Dataset Split Configurations', fontsize=12, fontweight='bold')
    ax4.set_xticks(list(x_positions))
    ax4.set_xticklabels([f"{l}\n({c:,} train)" for l, c in zip(split_labels, train_counts)], fontsize=10)
    ax4.set_ylabel('Macro F1-Score', fontsize=11)
    ax4.set_ylim(0.90, 0.98)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='lower right', fontsize=10)

    plt.suptitle('uTHCD Tamil Handwritten OCR: Multi-Split Empirical Benchmark Study\n(Comparison of 70/30, 80/20, 85/15, and 90/10 Partitioning Strategies)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_png = os.path.join(out_dir, 'multi_split_comparison_curves.png')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated multi-split comparative curves: {output_png}")

if __name__ == '__main__':
    generate_multi_split_plots()
