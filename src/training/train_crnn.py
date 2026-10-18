"""
train_crnn.py - End-to-End Training and Evaluation Pipeline for Tamil CRNN Word OCR.

Features:
1. PyTorch Automatic Mixed Precision (AMP) with GradScaler for fast GPU training.
2. CTC Loss with zero_infinity=True to guard against length instability.
3. Resumable checkpointing (saves 'latest_checkpoint.pt' and 'best_model.pt').
4. Full validation loop calculating CTC Loss, Character Error Rate (CER), Word Error Rate (WER),
   and Exact Match Accuracy.
5. Early stopping based on Validation CER.
6. Support for loading from raw nested ZIP or pre-extracted NVMe directories.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.data.vocab import TamilCTCCharacterVocabulary
from src.data.dataset import TamilWordDataset, pad_collate_fn
from src.models.crnn import TamilCRNN
from src.inference.ctc_decoder import (
    ctc_greedy_decode,
    compute_cer,
    compute_wer,
    compute_exact_accuracy
)

def evaluate_validation(model: nn.Module,
                        val_loader: DataLoader,
                        criterion: nn.Module,
                        vocab: TamilCTCCharacterVocabulary,
                        device: torch.device,
                        max_batches: Optional[int] = None) -> Dict[str, float]:
    """
    Evaluates validation partition for Loss, CER, WER, and Exact Match.
    """
    model.eval()
    total_val_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches and b_idx >= max_batches:
                break
                
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            texts = batch['texts']
            
            # Forward pass: shape (T, B, num_classes)
            logits = model(images)
            log_probs = logits.log_softmax(2)
            
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            if torch.isfinite(loss):
                total_val_loss += loss.item()
                
            # Decode predictions
            decoded = ctc_greedy_decode(logits, vocab, blank_idx=0)
            all_predictions.extend(decoded)
            all_targets.extend(texts)
            
    batches_evaluated = b_idx + 1 if 'b_idx' in locals() else 1
    avg_val_loss = total_val_loss / max(1, batches_evaluated)
    val_cer = compute_cer(all_predictions, all_targets)
    val_wer = compute_wer(all_predictions, all_targets)
    val_acc = compute_exact_accuracy(all_predictions, all_targets)
    
    return {
        'val_loss': avg_val_loss,
        'val_cer': val_cer,
        'val_wer': val_wer,
        'val_acc': val_acc,
        'num_samples': len(all_targets)
    }

def train_one_epoch(model: nn.Module,
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: torch.cuda.amp.GradScaler,
                    device: torch.device,
                    use_amp: bool = True,
                    log_interval: int = 50) -> float:
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['images'].to(device)
        targets = batch['targets'].to(device)
        target_lengths = batch['target_lengths'].to(device)
        input_lengths = batch['input_lengths'].to(device)
        
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(images)
                log_probs = logits.log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] Skipped batch {batch_idx} due to non-finite loss: {loss.item()}")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
        running_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            avg_batch_loss = running_loss / (batch_idx + 1)
            print(f"  Step [{batch_idx + 1}/{total_batches}] - Loss: {avg_batch_loss:.4f}")
            
    return running_loss / max(1, total_batches)

def main():
    parser = argparse.ArgumentParser(description="Train Tamil CRNN Word Recognition Model")
    parser.add_argument("--zip_path", type=str, default="datasets/raw/tamil_word_dataset.zip",
                        help="Path to full tamil_word_dataset.zip archive")
    parser.add_argument("--extracted_dir", type=str, default=None,
                        help="Path to extracted dataset folder containing train.txt, val.txt, etc.")
    parser.add_argument("--vocab_path", type=str, default="reports/tamil_ctc_char_vocab.json",
                        help="Path to precomputed tamil_ctc_char_vocab.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/crnn",
                        help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Total training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Adam learning rate")
    parser.add_argument("--hidden_size", type=int, default=256, help="BiLSTM hidden size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pt")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit train samples (for smoke test)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit val samples (for smoke test)")
    parser.add_argument("--no_amp", action="store_true", help="Disable PyTorch AMP mixed precision")
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Device Selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Tamil CRNN Training Initialized ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # 2. Vocabulary Initialization
    vocab = TamilCTCCharacterVocabulary()
    if os.path.exists(args.vocab_path):
        vocab.load_from_json(args.vocab_path)
        print(f"Loaded vocabulary from {args.vocab_path}: {len(vocab)} classes.")
    else:
        # Fallback build from zip
        if args.zip_path and os.path.exists(args.zip_path):
            import zipfile
            with zipfile.ZipFile(args.zip_path, 'r') as z:
                with z.open('vocab.txt') as f:
                    words = f.read().decode('utf-8').splitlines()
            vocab.build_from_words(words)
            print(f"Built vocabulary from zip: {len(vocab)} classes.")
        else:
            raise FileNotFoundError(f"Neither {args.vocab_path} nor {args.zip_path} exists.")
            
    num_classes = len(vocab)
    assert num_classes == 73, f"Expected 73 classes (72 chars + 1 blank), got {num_classes}"
    
    # 3. Datasets and DataLoaders
    print("Initializing datasets...")
    train_dataset = TamilWordDataset(
        split='train',
        zip_path=args.zip_path if not args.extracted_dir else None,
        extracted_dir=args.extracted_dir,
        vocab=vocab,
        max_samples=args.max_train_samples
    )
    val_dataset = TamilWordDataset(
        split='val',
        zip_path=args.zip_path if not args.extracted_dir else None,
        extracted_dir=args.extracted_dir,
        vocab=vocab,
        max_samples=args.max_val_samples
    )
    print(f"Dataset loaded: Train = {len(train_dataset):,} samples, Val = {len(val_dataset):,} samples")
    
    # On Windows, num_workers > 0 can sometimes cause issues with zipfile handles; set to 0 if Windows and no extracted_dir
    workers = 0 if (sys.platform == 'win32' and not args.extracted_dir) else args.num_workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=workers,
        pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=workers,
        pin_memory=(device.type == 'cuda')
    )
    
    # 4. Model, Criterion, Optimizer, Scheduler, Scaler
    model = TamilCRNN(num_classes=num_classes, in_channels=1, hidden_size=args.hidden_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    start_epoch = 1
    best_val_cer = float('inf')
    early_stop_counter = 0
    metrics_history = []
    
    # 5. Checkpoint Resume
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    
    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming training from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_cer = checkpoint.get('best_val_cer', float('inf'))
        metrics_history = checkpoint.get('metrics_history', [])
        print(f"Resumed at epoch {start_epoch}. Best Val CER so far: {best_val_cer:.4f}")
        
    print(f"\n--- Starting Training ({args.num_epochs} Epochs) ---")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.num_epochs} (LR: {current_lr:.6f})")
        
        # Train
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp
        )
        
        # Validate
        val_metrics = evaluate_validation(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            vocab=vocab,
            device=device
        )
        
        epoch_duration = time.time() - epoch_start
        print(f"Epoch {epoch} Completed in {epoch_duration:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Val CER: {val_metrics['val_cer']:.4f} | "
              f"Val WER: {val_metrics['val_wer']:.4f} | "
              f"Val Acc: {val_metrics['val_acc']*100:.2f}%")
              
        # Step LR Scheduler based on validation CER
        scheduler.step(val_metrics['val_cer'])
        
        # Record metrics
        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_cer': val_metrics['val_cer'],
            'val_wer': val_metrics['val_wer'],
            'val_acc': val_metrics['val_acc'],
            'learning_rate': current_lr,
            'duration_seconds': epoch_duration
        }
        metrics_history.append(record)
        
        # Save latest checkpoint
        latest_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_cer': best_val_cer,
            'metrics_history': metrics_history,
            'num_classes': num_classes,
            'hidden_size': args.hidden_size
        }
        torch.save(latest_ckpt, latest_ckpt_path)
        
        # Check for best model improvement
        if val_metrics['val_cer'] < best_val_cer:
            improvement = best_val_cer - val_metrics['val_cer']
            best_val_cer = val_metrics['val_cer']
            early_stop_counter = 0
            print(f"  -> Best model updated! Val CER decreased by {improvement:.4f} to {best_val_cer:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
                'num_classes': num_classes,
                'hidden_size': args.hidden_size,
                'vocab_size': len(vocab)
            }, best_model_path)
        else:
            early_stop_counter += 1
            print(f"  -> No CER improvement. Early stopping counter: {early_stop_counter}/{args.patience}")
            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
        # Save training log JSON
        with open(os.path.join(args.checkpoint_dir, "training_log.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'best_val_cer': best_val_cer,
                'total_epochs': epoch,
                'total_time_seconds': time.time() - start_time,
                'history': metrics_history
            }, f, indent=2)
            
    total_training_time = time.time() - start_time
    print(f"\n=== Training Finished in {total_training_time/60:.2f} mins ===")
    print(f"Best Validation CER: {best_val_cer:.4f}")
    print(f"Best checkpoint: {best_model_path}")

if __name__ == "__main__":
    main()
