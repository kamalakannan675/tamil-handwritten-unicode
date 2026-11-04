"""
experiments/u16_dual_dataset/train_dual_crnn.py - Fine-tune Production CRNN on Dual Tamil Datasets.

Features:
1. Starts from independent replica of Production Checkpoint (production_crnn_baseline_copy.pt).
2. 50/50 Balanced Dual-Stream Batching: 64 legacy samples + 64 HW-UC samples per batch.
3. Independent validation tracking on:
   - Old Dataset (tamil_word_dataset)
   - New Dataset (IIIT-Indic-HW-UC)
   - Combined Joint Validation
4. Checkpoint persistence tracking:
   - latest_checkpoint.pt
   - best_model_val_loss_combined.pt (Primary)
   - best_model_word_acc_combined.pt
   - best_model_cer_combined.pt
   - best_model_val_old.pt
   - best_model_val_new.pt
5. Conservative fine-tuning learning rate (5e-5) with CosineAnnealingLR.
"""

import os
import sys
import time
import json
import hashlib
import argparse
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Bootstrap repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.models.crnn import TamilCRNN
from src.data.vocab import TamilCTCCharacterVocabulary
from src.data.dataset import TamilWordDataset, pad_collate_fn
from src.data.dataset_hw_uc import IIITIndicHWUCDataset
from src.inference.ctc_decoder import (
    ctc_greedy_decode,
    compute_cer,
    compute_wer,
    compute_exact_accuracy
)
from experiments.u16_dual_dataset.dual_dataset import DualStreamDataLoader

class CRNNDataParallel(nn.DataParallel):
    def gather(self, outputs, output_device):
        return torch.cat([o.to(output_device) for o in outputs], dim=1)

def evaluate_split(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   vocab: TamilCTCCharacterVocabulary,
                   device: torch.device,
                   split_name: str,
                   use_amp: bool = True,
                   max_batches: Optional[int] = None) -> Dict[str, float]:
    """Evaluates a validation partition."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    batches = len(loader)
    
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches and b_idx >= max_batches:
                break
                
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            texts = batch['texts']
            
            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    log_probs = logits.log_softmax(2)
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)
            else:
                logits = model(images)
                log_probs = logits.log_softmax(2)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                
            if torch.isfinite(loss):
                total_loss += loss.item()
                
            try:
                decoded = ctc_greedy_decode(logits, vocab, blank_idx=0, input_lengths=input_lengths)
            except TypeError:
                decoded = ctc_greedy_decode(logits, vocab, blank_idx=0)
            all_preds.extend(decoded)
            all_targets.extend(texts)
            
    n_batches = b_idx + 1 if 'b_idx' in locals() else 1
    avg_loss = total_loss / max(1, n_batches)
    cer = compute_cer(all_preds, all_targets)
    wer = compute_wer(all_preds, all_targets)
    acc = compute_exact_accuracy(all_preds, all_targets)
    
    return {
        'split': split_name,
        'val_loss': avg_loss,
        'val_cer': cer,
        'val_wer': wer,
        'val_acc': acc,
        'num_samples': len(all_targets)
    }

def main():
    parser = argparse.ArgumentParser(description="U16 Dual-Dataset Production CRNN Fine-Tuning")
    parser.add_argument("--legacy_zip", type=str, default="datasets/raw/tamil_word_dataset.zip",
                        help="Path to legacy tamil_word_dataset.zip")
    parser.add_argument("--legacy_extracted", type=str, default=None,
                        help="Path to pre-extracted legacy dataset folder")
    parser.add_argument("--hw_uc_root", type=str, default="datasets/iiit_indic_hw_uc/tamil",
                        help="Path to IIIT-Indic-HW-UC root directory")
    parser.add_argument("--train_gt_hw_uc", type=str, default=None,
                        help="Explicit path to HW-UC train split")
    parser.add_argument("--val_gt_hw_uc", type=str, default=None,
                        help="Explicit path to HW-UC val split")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/crnn/tamil_ctc_char_vocab.json",
                        help="Path to production 73-class character vocabulary")
    parser.add_argument("--initial_checkpoint", type=str,
                        default="experiments/u16_dual_dataset/checkpoints/production_crnn_baseline_copy.pt",
                        help="Initial checkpoint to fine-tune from")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="/kaggle/working",
                        help="Directory to save fine-tuned checkpoints directly")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Total batch size (split evenly between datasets)")
    parser.add_argument("--num_epochs", type=int, default=15, help="Total fine-tuning epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial fine-tuning learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum cosine annealing learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit training samples for testing")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit validation samples for testing")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pt")
    
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== U16 Dual-Dataset Fine-Tuning Pipeline Initialized ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Initial Checkpoint: {args.initial_checkpoint}")
    print(f"Batch Size: {args.batch_size} (Dual stream: {args.batch_size//2} Old + {args.batch_size//2} New)")
    print(f"Learning Rate: {args.learning_rate} -> {args.min_lr} (Cosine Annealing)")
    
    # 1. Load Vocabulary
    vocab = TamilCTCCharacterVocabulary(args.vocab_path)
    print(f"Loaded production vocabulary: {len(vocab)} classes.")
    
    # 2. Datasets
    print("\n--- Initializing Dual-Stream Datasets ---")
    batch_a = args.batch_size // 2
    batch_b = args.batch_size // 2
    
    legacy_ext = args.legacy_extracted
    if not legacy_ext and os.path.exists("datasets/extracted") and os.path.exists("datasets/extracted/train.txt"):
        legacy_ext = "datasets/extracted"
        print("Auto-detected pre-extracted legacy archives at datasets/extracted.")
    
    # Stream A (Legacy)
    train_ds_a = TamilWordDataset(
        split='train',
        zip_path=args.legacy_zip if not legacy_ext else None,
        extracted_dir=legacy_ext,
        vocab=vocab,
        max_samples=args.max_train_samples
    )
    val_ds_a = TamilWordDataset(
        split='val',
        zip_path=args.legacy_zip if not legacy_ext else None,
        extracted_dir=legacy_ext,
        vocab=vocab,
        max_samples=args.max_val_samples
    )
    
    # Stream B (HW-UC)
    t_gt = args.train_gt_hw_uc or os.path.join(args.hw_uc_root, 'file', 'train_doc_split_85_15.txt')
    if not os.path.exists(t_gt):
        t_gt = os.path.join(args.hw_uc_root, 'file', 'train_gt.txt')
        
    v_gt = args.val_gt_hw_uc or os.path.join(args.hw_uc_root, 'file', 'val_doc_split_85_15.txt')
    if not os.path.exists(v_gt):
        v_gt = os.path.join(args.hw_uc_root, 'file', 'val_gt.txt')
        
    train_ds_b = IIITIndicHWUCDataset(
        root_dir=args.hw_uc_root,
        split='train',
        gt_path=t_gt,
        vocab=vocab,
        max_samples=args.max_train_samples
    )
    
    # Phase U16-C4: Filter verified CTC-invalid HW-UC training samples from actual manifest audit
    VERIFIED_CTC_INVALID_PATHS = [
        '1_New_Annoatation_Google/01_tamil_word_images/1882/image/259481_12.jpg',
        '1_New_Annoatation_Google/01_tamil_word_images/1874/image/256787_9.jpg',
        '1_New_Annoatation_Google/01_tamil_word_images/1959/image/268658_6.jpg',
        '1_New_Annoatation_Google/01_tamil_word_images/1959/image/268661_12.jpg',
    ]
    norm_excluded_paths = {os.path.normpath(p).replace('\\', '/') for p in VERIFIED_CTC_INVALID_PATHS}
    
    path_counts = {p: 0 for p in norm_excluded_paths}
    filtered_records = []
    for r in train_ds_b.records:
        norm_r = os.path.normpath(r[0]).replace('\\', '/')
        if norm_r in norm_excluded_paths:
            path_counts[norm_r] += 1
        else:
            filtered_records.append(r)
            
    if args.max_train_samples is None:
        missing_paths = [p for p, c in path_counts.items() if c == 0]
        duplicated_paths = [p for p, c in path_counts.items() if c > 1]
        if missing_paths or duplicated_paths:
            details = []
            if missing_paths:
                details.append(f"Missing expected paths (count=0): {missing_paths}")
            if duplicated_paths:
                details.append(f"Duplicated expected paths (count>1): {duplicated_paths}")
            raise ValueError(f"HW-UC sample exclusion verification failed: {'; '.join(details)}")
            
        train_ds_b.records = filtered_records
        assert len(train_ds_b) == 142931, f"Expected 142,931 clean samples in train_ds_b, got {len(train_ds_b)}"
        assert not any(os.path.normpath(r[0]).replace('\\', '/') in norm_excluded_paths for r in train_ds_b.records), \
            "Excluded path remains in train_ds_b.records"
        print(f"Verified and excluded exactly {len(VERIFIED_CTC_INVALID_PATHS)} invalid HW-UC training samples. Retained: {len(train_ds_b):,}")
    else:
        train_ds_b.records = filtered_records

    val_ds_b = IIITIndicHWUCDataset(
        root_dir=args.hw_uc_root,
        split='val',
        gt_path=v_gt,
        vocab=vocab,
        max_samples=args.max_val_samples
    )
    
    print(f"Dataset A (Legacy): Train={len(train_ds_a):,}, Val={len(val_ds_a):,}")
    print(f"Dataset B (HW-UC):  Train={len(train_ds_b):,}, Val={len(val_ds_b):,}")
    
    # Dual-Stream Training Loader
    train_loader = DualStreamDataLoader(
        train_ds_a,
        train_ds_b,
        batch_size_a=batch_a,
        batch_size_b=batch_b,
        shuffle=True
    )
    print(f"Dual-Stream Training Batches per Epoch: {len(train_loader):,}")
    
    # Separate Validation Loaders
    val_loader_a = DataLoader(val_ds_a, batch_size=64, shuffle=False, collate_fn=pad_collate_fn)
    val_loader_b = DataLoader(val_ds_b, batch_size=64, shuffle=False, collate_fn=pad_collate_fn)
    
    # 3. Model Setup & Weights Loading
    raw_model = TamilCRNN(num_classes=len(vocab), in_channels=1, hidden_size=256).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=2048.0)
    
    latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_checkpoint_u16d.pt")
    if not os.path.exists(latest_ckpt_path):
        latest_ckpt_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
        
    start_epoch = 1
    best_loss_comb = float('inf')
    best_acc_comb = 0.0
    best_cer_comb = float('inf')
    best_acc_old = 0.0
    best_acc_new = 0.0
    best_score_dual = 0.0
    metrics_history = []
    
    if args.resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming from: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        st = ckpt['model_state_dict']
        if any(k.startswith('module.') for k in st.keys()):
            st = {k[7:] if k.startswith('module.') else k: v for k, v in st.items()}
        raw_model.load_state_dict(st)
        
        if 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print("Restored optimizer state.")
        if 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print("Restored scheduler state.")
        if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None and use_amp:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print("Restored scaler state.")
            
        start_epoch = ckpt['epoch'] + 1
        best_loss_comb = ckpt.get('best_loss_comb', float('inf'))
        best_acc_comb = ckpt.get('best_acc_comb', 0.0)
        best_cer_comb = ckpt.get('best_cer_comb', float('inf'))
        best_acc_old = ckpt.get('best_acc_old', 0.0)
        best_acc_new = ckpt.get('best_acc_new', 0.0)
        best_score_dual = ckpt.get('best_score_dual', 0.0)
        metrics_history = ckpt.get('metrics_history', [])
        print(f"Resumed at epoch {start_epoch}. Best Comb Loss: {best_loss_comb:.4f}, Best Comb Acc: {best_acc_comb*100:.2f}%, Best Score: {best_score_dual:.4f}")
    else:
        print(f"Loading initial production weights from: {args.initial_checkpoint}")
        init_ckpt = torch.load(args.initial_checkpoint, map_location=device, weights_only=False)
        st = init_ckpt['model_state_dict']
        if any(k.startswith('module.') for k in st.keys()):
            st = {k[7:] if k.startswith('module.') else k: v for k, v in st.items()}
        raw_model.load_state_dict(st)
        print("Successfully loaded initial production CRNN weights.")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 1:
        print(f"Enabling Dual-GPU DataParallel across {num_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
        model = CRNNDataParallel(raw_model)
    else:
        model = raw_model
        
    # Write pre-training configuration & environment metadata
    training_config = {
        'initial_checkpoint': args.initial_checkpoint,
        'checkpoint_dir': args.checkpoint_dir,
        'batch_size': args.batch_size,
        'batch_size_legacy': batch_a,
        'batch_size_hw_uc': batch_b,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'min_lr': args.min_lr,
        'weight_decay': 1e-5,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'patience': args.patience,
        'grad_clip': 5.0,
        'ctc_blank': 0,
        'ctc_zero_infinity': True,
        'vocab_size': len(vocab),
        'device': str(device),
        'num_gpus': num_gpus,
        'use_amp': use_amp,
        'dataset_counts': {
            'legacy_train': len(train_ds_a),
            'legacy_val': len(val_ds_a),
            'hw_uc_train': len(train_ds_b),
            'hw_uc_val': len(val_ds_b),
            'total_train': len(train_ds_a) + len(train_ds_b),
            'total_val': len(val_ds_a) + len(val_ds_b)
        }
    }
    with open(os.path.join(args.checkpoint_dir, "training_config.json"), 'w', encoding='utf-8') as f:
        json.dump(training_config, f, indent=2)

    gpu_env = {
        'device_type': device.type,
        'cuda_available': torch.cuda.is_available(),
        'device_count': num_gpus,
        'devices': [
            {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'total_memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)
            } for i in range(num_gpus)
        ] if torch.cuda.is_available() else []
    }
    with open(os.path.join(args.checkpoint_dir, "gpu_environment.json"), 'w', encoding='utf-8') as f:
        json.dump(gpu_env, f, indent=2)
        
    # Checkpoint output paths
    p_latest_u16d = os.path.join(args.checkpoint_dir, "latest_checkpoint_u16d.pt")
    p_latest_generic = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    p_best_u16d = os.path.join(args.checkpoint_dir, "best_model_u16d.pt")
    p_best_generic = os.path.join(args.checkpoint_dir, "best_model.pt")
    p_best_loss_comb = os.path.join(args.checkpoint_dir, "best_model_val_loss_combined.pt")
    p_best_acc_comb = os.path.join(args.checkpoint_dir, "best_model_word_acc_combined.pt")
    p_best_cer_comb = os.path.join(args.checkpoint_dir, "best_model_cer_combined.pt")
    p_best_val_old = os.path.join(args.checkpoint_dir, "best_model_val_old.pt")
    p_best_val_new = os.path.join(args.checkpoint_dir, "best_model_val_new.pt")
    
    print(f"\n--- Starting Dual-Dataset Fine-Tuning ({args.num_epochs} Epochs) ---")
    start_time = time.time()
    early_stop_counter = 0
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{args.num_epochs} (LR: {current_lr:.6f})")
        
        # Training loop
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
                with torch.amp.autocast('cuda'):
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
                    print(f"[Warning] Skipped batch {batch_idx} due to non-finite loss")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
            running_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                avg_b_loss = running_loss / (batch_idx + 1)
                print(f"  Step [{batch_idx + 1}/{total_batches}] - Loss: {avg_b_loss:.4f}", flush=True)
                
        train_loss = running_loss / max(1, total_batches)
        
        # Validation across both sets
        print("  Evaluating Old Dataset Validation...")
        metrics_old = evaluate_split(model, val_loader_a, criterion, vocab, device, "legacy")
        print("  Evaluating New Dataset Validation...")
        metrics_new = evaluate_split(model, val_loader_b, criterion, vocab, device, "hw_uc")
        
        # Combined metrics
        comb_samples = metrics_old['num_samples'] + metrics_new['num_samples']
        comb_loss = (metrics_old['val_loss'] * metrics_old['num_samples'] + metrics_new['val_loss'] * metrics_new['num_samples']) / max(1, comb_samples)
        comb_cer = (metrics_old['val_cer'] * metrics_old['num_samples'] + metrics_new['val_cer'] * metrics_new['num_samples']) / max(1, comb_samples)
        comb_wer = (metrics_old['val_wer'] * metrics_old['num_samples'] + metrics_new['val_wer'] * metrics_new['num_samples']) / max(1, comb_samples)
        comb_acc = (metrics_old['val_acc'] * metrics_old['num_samples'] + metrics_new['val_acc'] * metrics_new['num_samples']) / max(1, comb_samples)
        
        epoch_dur = time.time() - epoch_start
        print(f"Epoch {epoch} ({epoch_dur:.1f}s) | Train Loss: {train_loss:.4f}")
        print(f"  Old Val:  Loss={metrics_old['val_loss']:.4f} | CER={metrics_old['val_cer']:.4f} | Acc={metrics_old['val_acc']*100:.2f}%")
        print(f"  New Val:  Loss={metrics_new['val_loss']:.4f} | CER={metrics_new['val_cer']:.4f} | Acc={metrics_new['val_acc']*100:.2f}%")
        print(f"  Combined: Loss={comb_loss:.4f} | CER={comb_cer:.4f} | Acc={comb_acc*100:.2f}%")
        
        scheduler.step()
        
        # Save checkpoints
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        
        def save_ckpt(path, metric_name, val):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'metric_saved_by': metric_name,
                'metric_value': val,
                'metrics_old': metrics_old,
                'metrics_new': metrics_new,
                'metrics_comb': {'val_loss': comb_loss, 'val_cer': comb_cer, 'val_wer': comb_wer, 'val_acc': comb_acc},
                'source_initial_checkpoint': args.initial_checkpoint
            }, path)
            
        # 1. Latest checkpoint (save both latest_checkpoint_u16d.pt and latest_checkpoint.pt)
        ckpt_payload = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'best_loss_comb': best_loss_comb,
            'best_acc_comb': best_acc_comb,
            'best_cer_comb': best_cer_comb,
            'best_acc_old': best_acc_old,
            'best_acc_new': best_acc_new,
            'best_score_dual': best_score_dual,
            'metrics_history': metrics_history
        }
        torch.save(ckpt_payload, p_latest_u16d)
        torch.save(ckpt_payload, p_latest_generic)
        _chk_latest = torch.load(p_latest_u16d, map_location='cpu', weights_only=False)
        assert 'model_state_dict' in _chk_latest, "Saved latest checkpoint missing model_state_dict"
        
        # Check improvements & Documented Checkpoint Selection Rule Protecting Both Datasets
        improved = False
        
        # Primary Selection Criterion: Balanced Dual Retention Score
        # Baseline legacy validation accuracy is 89.28%.
        # Candidate must protect legacy accuracy >= 88.0% (within 1.28% margin of baseline).
        acc_old = metrics_old['val_acc']
        acc_new = metrics_new['val_acc']
        if acc_old >= 0.880:
            score = 2.0 * (acc_old * acc_new) / (acc_old + acc_new + 1e-8)
        else:
            score = 2.0 * (acc_old * acc_new) / (acc_old + acc_new + 1e-8) * (acc_old / 0.880)
            
        if score > best_score_dual or not os.path.exists(p_best_u16d):
            diff = score - best_score_dual
            best_score_dual = score
            improved = True
            print(f"  -> Best U16-D Dual Candidate updated! Score: {best_score_dual:.4f} (Legacy: {acc_old*100:.2f}%, HW-UC: {acc_new*100:.2f}%)")
            save_ckpt(p_best_u16d, 'balanced_dual_retention_score', best_score_dual)
            save_ckpt(p_best_generic, 'balanced_dual_retention_score', best_score_dual)
            
            # Verify saved best model can be loaded successfully
            _chk_best = torch.load(p_best_u16d, map_location='cpu', weights_only=False)
            assert 'model_state_dict' in _chk_best, "Saved best model missing model_state_dict"
            
            # Update u16d_model_sha256.txt
            h_b = hashlib.sha256()
            with open(p_best_u16d, 'rb') as f_b:
                while chunk := f_b.read(65536):
                    h_b.update(chunk)
            cur_sha = h_b.hexdigest().upper()
            sha_txt_p = os.path.join(args.checkpoint_dir, "u16d_model_sha256.txt")
            with open(sha_txt_p, 'w', encoding='utf-8') as f_s:
                f_s.write(f"{cur_sha}  best_model_u16d.pt\n")

        if comb_loss < best_loss_comb:
            diff = best_loss_comb - comb_loss
            best_loss_comb = comb_loss
            print(f"  -> Best Combined Loss updated! Decreased by {diff:.4f} to {best_loss_comb:.4f}")
            save_ckpt(p_best_loss_comb, 'combined_loss', comb_loss)
            
        if comb_acc > best_acc_comb:
            diff = (comb_acc - best_acc_comb) * 100
            best_acc_comb = comb_acc
            print(f"  -> Best Combined Word Acc updated! Increased by {diff:.2f}% to {best_acc_comb*100:.2f}%")
            save_ckpt(p_best_acc_comb, 'combined_word_acc', comb_acc)
            
        if comb_cer < best_cer_comb:
            best_cer_comb = comb_cer
            save_ckpt(p_best_cer_comb, 'combined_cer', comb_cer)
            
        if metrics_old['val_acc'] > best_acc_old:
            best_acc_old = metrics_old['val_acc']
            save_ckpt(p_best_val_old, 'old_val_acc', best_acc_old)
            
        if metrics_new['val_acc'] > best_acc_new:
            best_acc_new = metrics_new['val_acc']
            save_ckpt(p_best_val_new, 'new_val_acc', best_acc_new)
            
        if improved:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break
                
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'comb_loss': comb_loss,
            'comb_cer': comb_cer,
            'comb_wer': comb_wer,
            'comb_acc': comb_acc,
            'old_loss': metrics_old['val_loss'],
            'old_cer': metrics_old['val_cer'],
            'old_wer': metrics_old['val_wer'],
            'old_acc': metrics_old['val_acc'],
            'new_loss': metrics_new['val_loss'],
            'new_cer': metrics_new['val_cer'],
            'new_wer': metrics_new['val_wer'],
            'new_acc': metrics_new['val_acc'],
            'score_dual': score,
            'learning_rate': current_lr,
            'duration': epoch_dur
        })
        
        # 2. Append to epoch_metrics.csv
        csv_path = os.path.join(args.checkpoint_dir, "epoch_metrics.csv")
        csv_header = "epoch,learning_rate,train_loss,old_loss,old_cer,old_wer,old_acc,new_loss,new_cer,new_wer,new_acc,comb_loss,comb_cer,comb_wer,comb_acc,duration_sec\n"
        row = f"{epoch},{current_lr:.8f},{train_loss:.6f},{metrics_old['val_loss']:.6f},{metrics_old['val_cer']:.6f},{metrics_old['val_wer']:.6f},{metrics_old['val_acc']:.6f},{metrics_new['val_loss']:.6f},{metrics_new['val_cer']:.6f},{metrics_new['val_wer']:.6f},{metrics_new['val_acc']:.6f},{comb_loss:.6f},{comb_cer:.6f},{comb_wer:.6f},{comb_acc:.6f},{epoch_dur:.2f}\n"
        if epoch == 1 or not os.path.exists(csv_path):
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(csv_header + row)
        else:
            with open(csv_path, 'a', encoding='utf-8') as f:
                f.write(row)
                
        # 3. Save validation_metrics.json
        val_metrics_data = {
            'last_epoch': epoch,
            'legacy_validation': metrics_old,
            'hw_uc_validation': metrics_new,
            'combined_validation': {
                'val_loss': comb_loss,
                'val_cer': comb_cer,
                'val_wer': comb_wer,
                'val_acc': comb_acc,
                'num_samples': comb_samples
            },
            'best_scores': {
                'best_loss_comb': best_loss_comb,
                'best_acc_comb': best_acc_comb,
                'best_cer_comb': best_cer_comb,
                'best_acc_old': best_acc_old,
                'best_acc_new': best_acc_new,
                'best_score_dual': best_score_dual
            }
        }
        with open(os.path.join(args.checkpoint_dir, "validation_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(val_metrics_data, f, indent=2)

        # 4. Save training_log.json
        with open(os.path.join(args.checkpoint_dir, "training_log.json"), 'w', encoding='utf-8') as f:
            json.dump({
                'total_epochs': epoch,
                'best_loss_comb': best_loss_comb,
                'best_acc_comb': best_acc_comb,
                'best_acc_old': best_acc_old,
                'best_acc_new': best_acc_new,
                'best_score_dual': best_score_dual,
                'history': metrics_history
            }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n=== Fine-Tuning Completed in {total_time/60:.2f} mins ===")
    print(f"Best Combined Loss: {best_loss_comb:.4f}")
    print(f"Best Combined Acc:  {best_acc_comb*100:.2f}%")
    print(f"Best Old Acc:       {best_acc_old*100:.2f}%")
    print(f"Best New Acc:       {best_acc_new*100:.2f}%")

    # Final Output Verification (Section 5 Requirement)
    print("\n=== Final Model Output Verification ===")
    assert os.path.exists(p_best_u16d), f"CRITICAL: Main model {p_best_u16d} missing!"
    best_sz = os.path.getsize(p_best_u16d)
    assert best_sz > 0, f"CRITICAL: Main model {p_best_u16d} is empty (0 bytes)!"
    print(f"1. Main model confirmed: {p_best_u16d} ({best_sz:,} bytes) [PASS]")
    
    # Safe loading check
    loaded_best = torch.load(p_best_u16d, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in loaded_best, "Missing model_state_dict in best_model_u16d.pt"
    print("2. Safe checkpoint load verification: [PASS]")
    
    # Architecture and 73-class vocabulary check
    eval_model = TamilCRNN(num_classes=len(vocab), in_channels=1, hidden_size=256)
    eval_model.load_state_dict(loaded_best['model_state_dict'])
    print(f"3. Architecture & 73-class vocabulary compatibility: [PASS] ({len(vocab)} classes)")
    
    # Checksum computation
    h_best = hashlib.sha256()
    with open(p_best_u16d, 'rb') as f_best:
        while chunk := f_best.read(65536):
            h_best.update(chunk)
    final_sha = h_best.hexdigest().upper()
    baseline_sha = "CB7147047C86CDB256A927B9DAC76B8DAD91AD12751B2F76A4366DD3877C8EEB"
    if final_sha == baseline_sha:
        print("4. Baseline comparison: Notice - identical to baseline weights (no parameter delta)")
    else:
        print("4. Baseline comparison: Verified fine-tuned weights differ from baseline [PASS]")
    print(f"5. Exact absolute file path: {os.path.abspath(p_best_u16d)}")
    print(f"6. SHA-256 Checksum: {final_sha}")
    
    sha256_path = os.path.join(args.checkpoint_dir, "u16d_model_sha256.txt")
    with open(sha256_path, 'w', encoding='utf-8') as f_s:
        f_s.write(f"{final_sha}  best_model_u16d.pt\n")
    print(f"7. SHA-256 recorded to: {sha256_path} [PASS]")
    
    # Confirm presence of required reports
    required_reports = [
        "training_log.json",
        "epoch_metrics.csv",
        "validation_metrics.json",
        "training_config.json",
        "gpu_environment.json",
        "u16d_model_sha256.txt"
    ]
    for rep in required_reports:
        rep_path = os.path.join(args.checkpoint_dir, rep)
        assert os.path.exists(rep_path), f"Missing required report: {rep_path}"
        print(f"  Verified report: {rep} ({os.path.getsize(rep_path):,} bytes) [PASS]")
    print("8. Required reports presence check: ALL REQUIRED REPORTS VERIFIED [PASS]")

if __name__ == '__main__':
    main()
