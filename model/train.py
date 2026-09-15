"""
train.py - Training script for Tamil Handwritten Character Recognition CNN.

Features:
- CPU training strictly optimized for low-resource environments (4 GB RAM, Intel Pentium, HDD).
- Mini-batch gradient descent (Adam optimizer, lr=0.001, batch_size=32 as per paper Table 3).
- Memory-efficient HDF5 chunk streaming (avoids RAM exhaustion and HDD random-seek thrashing).
- Validation-accuracy-based checkpointing (preserves best model strictly based on validation performance).
- Early stopping monitor (patience on validation accuracy).
- Real-time training & validation metrics logging.
- Exports training_history.json, training_history.csv, training_metadata.json, and training_curve.png.
"""

import os
import sys
import time
import json
import csv
import argparse
from datetime import datetime
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.dataset import uTHCDDataset
from model.model import uTHCDNet
from model.unicode_mapping import NUM_CLASSES
from model.augmentation import get_augmentation_transform

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_training_curves(history, output_path, best_epoch=None):
    """Generates and saves dual-panel loss and accuracy training curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        epochs = [r['epoch'] for r in history]
        train_loss = [r['train_loss'] for r in history]
        val_loss = [r['val_loss'] for r in history]
        train_acc = [r['train_acc'] for r in history]
        val_acc = [r['val_acc'] for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 1. Loss curves
        ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, val_loss, 'r--s', label='Val Loss', linewidth=2, markersize=4)
        if best_epoch:
            ax1.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2,
                        label=f'Best Val Epoch ({best_epoch})')
        ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right')

        # 2. Accuracy curves
        ax2.plot(epochs, train_acc, 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
        ax2.plot(epochs, val_acc, 'r--s', label='Val Accuracy', linewidth=2, markersize=4)
        if best_epoch:
            ax2.axvline(x=best_epoch, color='green', linestyle=':', linewidth=2,
                        label=f'Best Val Epoch ({best_epoch})')
        ax2.set_title('Classification Accuracy (%) vs. Epochs', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"Saved Training Curve to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not generate training curve plot: {e}")

def save_history_csv(history, csv_path):
    """Saves training history records to a CSV file."""
    if not history:
        return
    fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch_time_seconds']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"Saved Training History CSV to: {csv_path}")

def train_epoch_chunked(model: nn.Module, hdf5_path: str, train_indices: np.ndarray,
                        batch_size: int, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                        transform, device: torch.device, chunk_size: int = 2500) -> tuple:
    """
    Memory-efficient chunk streaming training loop.
    Reads contiguous chunks from HDF5 (fast sequential disk I/O), permutes within chunk,
    and updates model weights in mini-batches. Peak RAM is strictly bounded under 60 MB.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    n_samples = len(train_indices)
    num_chunks = int(np.ceil(n_samples / chunk_size))
    # Randomly permute chunk processing order each epoch
    chunk_order = np.random.permutation(num_chunks)

    with h5py.File(hdf5_path, 'r') as f:
        x_dset = f['Train Data/x_train']
        y_dset = f['Train Data/y_train']

        for c_order_idx, c_idx in enumerate(chunk_order):
            start_i = c_idx * chunk_size
            end_i = min(start_i + chunk_size, n_samples)
            if start_i >= end_i:
                continue

            idx_slice = train_indices[start_i:end_i]
            c_start = int(idx_slice[0])
            c_end = int(idx_slice[-1]) + 1

            # Contiguous chunk read
            raw_x = x_dset[c_start:c_end]
            raw_y = y_dset[c_start:c_end]
            c_len = len(raw_y)

            # Vectorized normalization
            tensor_x = (255.0 - torch.from_numpy(raw_x).float()) / 255.0
            tensor_x = tensor_x.unsqueeze(1)
            tensor_y = torch.from_numpy(raw_y).long()

            # Random permutation of samples within the current chunk
            perm = torch.randperm(c_len)

            for b_i in range(0, c_len, batch_size):
                b_end = min(b_i + batch_size, c_len)
                batch_idxs = perm[b_i:b_end]
                bx = tensor_x[batch_idxs].to(device)
                by = tensor_y[batch_idxs].to(device)

                if transform is not None:
                    bx = transform(bx)

                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * bx.size(0)
                _, preds = torch.max(logits, 1)
                correct += (preds == by).sum().item()
                total += bx.size(0)

            # Periodic progress output inside long epochs
            if (c_order_idx + 1) % 5 == 0 or (c_order_idx + 1) == num_chunks:
                pct = (total / n_samples) * 100.0
                curr_loss = running_loss / max(1, total)
                curr_acc = (correct / max(1, total)) * 100.0
                print(f"    [Train Progress] {total}/{n_samples} ({pct:.1f}%) | Batch Loss: {curr_loss:.4f} | Batch Acc: {curr_acc:.2f}%")

    epoch_loss = running_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_epoch_chunked(model: nn.Module, hdf5_path: str, val_indices: np.ndarray,
                           batch_size: int, criterion: nn.Module,
                           device: torch.device, chunk_size: int = 2500) -> tuple:
    """
    Fast sequential chunk evaluation on the validation split. Unaugmented.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    n_samples = len(val_indices)

    with h5py.File(hdf5_path, 'r') as f:
        x_dset = f['Train Data/x_train']
        y_dset = f['Train Data/y_train']

        for start_i in range(0, n_samples, chunk_size):
            end_i = min(start_i + chunk_size, n_samples)
            idx_slice = val_indices[start_i:end_i]
            c_start = int(idx_slice[0])
            c_end = int(idx_slice[-1]) + 1

            raw_x = x_dset[c_start:c_end]
            raw_y = y_dset[c_start:c_end]
            c_len = len(raw_y)

            tensor_x = (255.0 - torch.from_numpy(raw_x).float()) / 255.0
            tensor_x = tensor_x.unsqueeze(1)
            tensor_y = torch.from_numpy(raw_y).long()

            for b_i in range(0, c_len, batch_size):
                b_end = min(b_i + batch_size, c_len)
                bx = tensor_x[b_i:b_end].to(device)
                by = tensor_y[b_i:b_end].to(device)

                logits = model(bx)
                loss = criterion(logits, by)

                running_loss += loss.item() * bx.size(0)
                _, preds = torch.max(logits, 1)
                correct += (preds == by).sum().item()
                total += bx.size(0)

    val_loss = running_loss / max(1, total)
    val_acc = (correct / max(1, total)) * 100.0
    return val_loss, val_acc

def run_training(args):
    set_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if getattr(args, 'smoke_test', False):
        args.max_train = 128
        args.max_val = 64
        args.epochs = 1
        print("[SMOKE TEST MODE ACTIVATED: max_train=128, max_val=64, epochs=1]")

    ckpt_dir = args.checkpoint_dir
    if getattr(args, 'experiment_name', None):
        ckpt_dir = os.path.join(args.checkpoint_dir, args.experiment_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs('outputs/reports', exist_ok=True)

    # Check total available samples in dataset
    with h5py.File(args.dataset_path, 'r') as f:
        total_train_available = f['Train Data/x_train'].shape[0]  # 62870

    val_size = 7870
    train_size = total_train_available - val_size  # 55000

    # Determine train & val sample count
    actual_train_count = train_size if (args.max_train is None or args.max_train <= 0) else min(args.max_train, train_size)
    actual_val_count = val_size if (args.max_val is None or args.max_val <= 0) else min(args.max_val, val_size)

    # Deterministic slicing per paper Listing 1
    # Train is first 55000 (or first actual_train_count)
    train_indices = np.arange(0, actual_train_count)
    # Val is last 7870 (or slice of actual_val_count)
    val_start = train_size
    val_indices = np.arange(val_start, val_start + actual_val_count)

    print("=" * 70)
    print("uTHCD TAMIL HANDWRITTEN CHARACTER RECOGNITION - TRAINING")
    print("=" * 70)
    print(f"Device:               {device}")
    print(f"Random Seed:          {args.seed}")
    print(f"Target Epochs:        {args.epochs}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Learning Rate:        {args.learning_rate}")
    print(f"Augmentation:         {args.augmentation}")
    print(f"Training Samples:     {actual_train_count} (out of {train_size})")
    print(f"Validation Samples:   {actual_val_count} (out of {val_size}, strictly unaugmented)")
    print(f"Checkpoint Directory: {os.path.abspath(ckpt_dir)}")
    print(f"Early Stopping:       Patience = {args.patience} epochs")
    print("=" * 70)

    train_transform = get_augmentation_transform(args.augmentation)

    model = uTHCDNet(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    start_total_time = time.time()

    best_checkpoint_path = os.path.join(ckpt_dir, 'best_model.pt')
    metadata_path = os.path.join(ckpt_dir, 'training_metadata.json')
    history_json_path = os.path.join(ckpt_dir, 'training_history.json')
    history_csv_path = os.path.join(ckpt_dir, 'training_history.csv')
    curve_png_path = os.path.join(ckpt_dir, 'training_curve.png')

    print("\nStarting Epoch Iterations...")
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<16}{'Time (s)':<10}")
    print("-" * 78)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch_chunked(
            model=model,
            hdf5_path=args.dataset_path,
            train_indices=train_indices,
            batch_size=args.batch_size,
            criterion=criterion,
            optimizer=optimizer,
            transform=train_transform,
            device=device,
            chunk_size=args.chunk_size
        )

        val_loss, val_acc = evaluate_epoch_chunked(
            model=model,
            hdf5_path=args.dataset_path,
            val_indices=val_indices,
            batch_size=args.batch_size,
            criterion=criterion,
            device=device,
            chunk_size=args.chunk_size
        )

        elapsed = time.time() - t0

        record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'train_acc': round(train_acc, 2),
            'val_loss': round(val_loss, 4),
            'val_acc': round(val_acc, 2),
            'epoch_time_seconds': round(elapsed, 2)
        }
        history.append(record)

        print(f"{epoch:<8}{train_loss:<14.4f}{train_acc:<16.2f}{val_loss:<14.4f}{val_acc:<16.2f}{elapsed:<10.2f}")

        # Checkpoint if validation accuracy strictly improved (secondary: val_loss)
        is_best = False
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            is_best = True
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': NUM_CLASSES,
                'arch': 'uTHCDNet'
            }, best_checkpoint_path)
            print(f"  --> Checkpoint saved: New best val accuracy {val_acc:.2f}% (Epoch {epoch})")
        else:
            epochs_without_improvement += 1
            print(f"  --> Validation did not improve ({epochs_without_improvement}/{args.patience} epochs)")

        # Save progress files after every epoch so nothing is lost if interrupted
        with open(history_json_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        save_history_csv(history, history_csv_path)
        plot_training_curves(history, curve_png_path, best_epoch=best_epoch)

        # Early stopping check
        if args.patience > 0 and epochs_without_improvement >= args.patience and epoch >= getattr(args, 'min_epochs', 5):
            print(f"\n[EARLY STOPPING ACTIVATED] Validation accuracy did not improve for {args.patience} consecutive epochs.")
            print(f"Terminating training early. Best checkpoint preserved from Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%).")
            break

    total_training_time = time.time() - start_total_time
    print("-" * 78)
    print(f"Training finished in {total_training_time:.2f} seconds ({total_training_time/60:.2f} mins).")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Best Checkpoint: {best_checkpoint_path}")

    metadata = {
        'project_title': 'Tamil Handwritten Character Recognition and Unicode Conversion System',
        'dataset_path': os.path.abspath(args.dataset_path),
        'dataset_split': {
            'train_samples_used': actual_train_count,
            'val_samples_used': actual_val_count,
            'total_available_train': 55000,
            'total_available_val': 7870,
            'total_available_test': 28080
        },
        'architecture': 'uTHCDNet (Figure 13 baseline: 32C3-MP2-64C3-MP2-1024FC-512FC-156FC)',
        'num_classes': NUM_CLASSES,
        'image_size': [64, 64],
        'augmentation_mode': args.augmentation,
        'experiment_name': getattr(args, 'experiment_name', 'default'),
        'optimizer': 'Adam',
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs_completed': len(history),
        'target_epochs': args.epochs,
        'random_seed': args.seed,
        'device': str(device),
        'best_epoch': best_epoch,
        'best_val_accuracy': round(best_val_acc, 2),
        'best_val_loss': round(best_val_loss, 4),
        'total_training_time_seconds': round(total_training_time, 2),
        'history': history,
        'timestamp': datetime.now().isoformat()
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved Metadata to: {metadata_path}")

    return metadata

def parse_args():
    parser = argparse.ArgumentParser(description="Train uTHCD CNN model with streaming")
    parser.add_argument('--dataset-path', type=str, default='data/hdf5_uTHCD_compressed.h5')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--augmentation', type=str, default='light',
                        choices=['none', 'light', 'strong'],
                        help="Data augmentation mode: none, light, or strong")
    parser.add_argument('--experiment-name', type=str, default='augmented_light_full',
                        help="Subfolder name for experiment checkpoints")
    parser.add_argument('--smoke-test', action='store_true',
                        help="Run quick smoke test (128 train, 64 val, 1 epoch)")
    parser.add_argument('--max-train', type=int, default=0,
                        help="Max training samples to use (0 for full 55,000)")
    parser.add_argument('--max-val', type=int, default=0,
                        help="Max validation samples to use (0 for full 7,870)")
    parser.add_argument('--chunk-size', type=int, default=2500,
                        help="Streaming chunk size (samples in RAM at once)")
    parser.add_argument('--patience', type=int, default=4,
                        help="Early stopping patience epochs on validation accuracy")
    parser.add_argument('--min-epochs', type=int, default=5,
                        help="Minimum epochs before early stopping can trigger")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-dir', type=str, default='model/checkpoints/experiments')
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run_training(args)
