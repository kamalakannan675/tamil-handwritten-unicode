"""
visualize_dataset.py - Script to visualize sample images from the uTHCD dataset.

Extracts representative samples across different classes, formats them into a clean
visual grid with Class ID, Unicode Character, and Unicode Hex Code, and saves
the output image to outputs/reports/dataset_visualization.png.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for headless servers
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.unicode_mapping import get_unicode_char, get_unicode_codepoints, NUM_CLASSES

# Try to detect a Tamil font on Windows if available (Nirmala UI, Latha, etc.)
def get_tamil_font():
    for font_name in ['Nirmala UI', 'Latha', 'Vijaya', 'Arial Unicode MS']:
        try:
            fp = FontProperties(family=font_name)
            return fp
        except Exception:
            continue
    return FontProperties()

def generate_visualization(hdf5_path='data/hdf5_uTHCD_compressed.h5',
                           output_path='outputs/reports/dataset_visualization.png',
                           num_rows=6, num_cols=6):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    font_prop = get_tamil_font()

    with h5py.File(hdf5_path, 'r') as f:
        x_train = f['Train Data/x_train']
        y_train = f['Train Data/y_train'][:]

        # Select a diverse set of classes spaced evenly across all 156 classes
        total_samples = num_rows * num_cols
        classes_to_show = np.linspace(0, NUM_CLASSES - 1, total_samples, dtype=int)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2.2, num_rows * 2.5))
        fig.suptitle("uTHCD Tamil Handwritten Character Dataset Samples", fontsize=16, fontweight='bold', y=0.99)

        for i, class_id in enumerate(classes_to_show):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]

            # Find first index with this class
            indices = np.where(y_train == class_id)[0]
            if len(indices) > 0:
                sample_img = x_train[indices[0]]
            else:
                sample_img = np.zeros((64, 64), dtype=np.uint8)

            tamil_char = get_unicode_char(class_id)
            codepoints = get_unicode_codepoints(class_id)

            ax.imshow(sample_img, cmap='gray')
            ax.set_title(f"Class {class_id}: {tamil_char}\n{codepoints}", fontsize=9, fontproperties=font_prop)
            ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"Dataset visualization grid saved successfully to: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    generate_visualization()
