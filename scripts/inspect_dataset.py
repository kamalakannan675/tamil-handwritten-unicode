"""
inspect_dataset.py - Script to dynamically inspect and verify data/hdf5_uTHCD_compressed.h5

Calculates actual dataset structure, groups, dataset shapes, dtypes,
pixel ranges, label ranges, sample counts, and class distributions.
"""

import os
import sys
import h5py
import numpy as np

# Configure standard output to utf-8 for Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def inspect_dataset(hdf5_path='data/hdf5_uTHCD_compressed.h5'):
    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 dataset file not found at '{hdf5_path}'.")
        sys.exit(1)

    print("=" * 50)
    print("uTHCD DATASET INSPECTION")
    print("=" * 50)
    print(f"HDF5 path:\n{os.path.abspath(hdf5_path)}\n")

    with h5py.File(hdf5_path, 'r') as f:
        groups = list(f.keys())
        print(f"Groups:\n{groups}\n")

        # Validate Train Data group
        if 'Train Data' not in f or 'Test Data' not in f:
            print("Error: Expected 'Train Data' and 'Test Data' groups in HDF5.")
            sys.exit(1)

        g_train = f['Train Data']
        g_test = f['Test Data']

        x_train = g_train['x_train']
        y_train = g_train['y_train']
        x_test = g_test['x_test']
        y_test = g_test['y_test']

        n_train = x_train.shape[0]
        n_test = x_test.shape[0]

        print(f"Training images:\n{n_train} samples, shape: {x_train.shape}, dtype: {x_train.dtype}")
        print(f"Training labels:\n{y_train.shape[0]} labels, shape: {y_train.shape}, dtype: {y_train.dtype}\n")

        print(f"Test images:\n{n_test} samples, shape: {x_test.shape}, dtype: {x_test.dtype}")
        print(f"Test labels:\n{y_test.shape[0]} labels, shape: {y_test.shape}, dtype: {y_test.dtype}\n")

        print(f"Image shape:\n{x_train.shape[1:]} (Height x Width, Grayscale)")
        print(f"Pixel dtype:\n{x_train.dtype}")

        # Compute pixel statistics from a representative sample without loading all into memory
        sample_batch = x_train[:1000]
        p_min = int(sample_batch.min())
        p_max = int(sample_batch.max())
        p_mean = float(sample_batch.mean())
        print(f"Pixel range:\n[{p_min}, {p_max}] (Sample mean: {p_mean:.2f}, 255=Background, 0=Ink)\n")

        # Class labels analysis
        y_train_arr = y_train[:]
        y_test_arr = y_test[:]

        unique_train = np.unique(y_train_arr)
        unique_test = np.unique(y_test_arr)
        total_unique = np.union1d(unique_train, unique_test)

        print(f"Unique classes:\n{len(total_unique)} classes (Train: {len(unique_train)}, Test: {len(unique_test)})")
        print(f"Class range:\n[{total_unique.min()}, {total_unique.max()}] (0-indexed: 0 to 155)\n")

        # Class distribution summary
        train_counts = np.bincount(y_train_arr, minlength=156)
        test_counts = np.bincount(y_test_arr, minlength=156)

        print("=" * 50)
        print("CLASS DISTRIBUTION SUMMARY")
        print("=" * 50)
        print(f"Train samples per class: min={train_counts.min()}, max={train_counts.max()}, mean={train_counts.mean():.2f}")
        print(f"Test samples per class:  min={test_counts.min()}, max={test_counts.max()}, mean={test_counts.mean():.2f} (Uniform: exactly 180 per class)")
        print(f"Total dataset samples:   {n_train + n_test} (Train: {n_train}, Test: {n_test})")
        print("=" * 50)

    return {
        'hdf5_path': hdf5_path,
        'groups': groups,
        'n_train': n_train,
        'n_test': n_test,
        'image_shape': x_train.shape[1:],
        'pixel_dtype': str(x_train.dtype),
        'pixel_min': p_min,
        'pixel_max': p_max,
        'num_classes': len(total_unique),
        'class_min': int(total_unique.min()),
        'class_max': int(total_unique.max()),
    }

if __name__ == '__main__':
    inspect_dataset()
