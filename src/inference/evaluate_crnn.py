"""
evaluate_crnn.py - Comprehensive Evaluation Script for Tamil CRNN Model.

Evaluates trained checkpoint on validation or test partitions:
- Computes Character Error Rate (CER)
- Computes Word Error Rate (WER)
- Computes Exact Match Accuracy
- Computes Average CTC Loss
- Exports qualitative prediction samples to JSON for error analysis.
"""

import os
import sys
# Bootstrap: ensure project root is on sys.path when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    levenshtein_distance,
    compute_cer,
    compute_wer,
    compute_exact_accuracy
)

def evaluate_checkpoint(checkpoint_path: str,
                        split: str = 'test',
                        zip_path: str = "datasets/raw/tamil_word_dataset.zip",
                        extracted_dir: str = None,
                        vocab_path: str = "reports/tamil_ctc_char_vocab.json",
                        batch_size: int = 32,
                        max_samples: int = None,
                        output_json: str = None) -> Dict[str, Any]:
    """
    Evaluates a saved checkpoint on the requested dataset split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating Checkpoint: {checkpoint_path} on split: {split} ===")
    print(f"Device: {device}")
    
    # 1. Load Vocabulary
    vocab = TamilCTCCharacterVocabulary()
    if os.path.exists(vocab_path):
        vocab.load_from_json(vocab_path)
    else:
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
        
    num_classes = len(vocab)
    
    # 2. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    hidden_size = ckpt.get('hidden_size', 256)
    
    model = TamilCRNN(num_classes=num_classes, in_channels=1, hidden_size=hidden_size).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded model state dict (epoch {ckpt.get('epoch', 'N/A')}).")
    
    # 3. Load Dataset
    dataset = TamilWordDataset(
        split=split,
        zip_path=zip_path if not extracted_dir else None,
        extracted_dir=extracted_dir,
        vocab=vocab,
        max_samples=max_samples
    )
    print(f"Dataset split '{split}' loaded with {len(dataset):,} samples.")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
        num_workers=0 if sys.platform == 'win32' else 2
    )
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    total_loss = 0.0
    all_preds = []
    all_targets = []
    sample_comparisons = []
    
    start_time = time.time()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            images = batch['images'].to(device)
            targets = batch['targets'].to(device)
            target_lengths = batch['target_lengths'].to(device)
            input_lengths = batch['input_lengths'].to(device)
            texts = batch['texts']
            img_paths = batch['img_paths']
            
            logits = model(images)
            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            if torch.isfinite(loss):
                total_loss += loss.item()
                
            decoded = ctc_greedy_decode(logits, vocab, blank_idx=0)
            
            for p, t, path in zip(decoded, texts, img_paths):
                all_preds.append(p)
                all_targets.append(t)
                if len(sample_comparisons) < 50:
                    sample_comparisons.append({
                        'image_path': path,
                        'ground_truth': t,
                        'prediction': p,
                        'exact_match': (p == t),
                        'char_edit_distance': levenshtein_distance(list(p), list(t))
                    })
                    
            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(loader):
                print(f"  Processed [{b_idx + 1}/{len(loader)}] batches...")
                
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, len(loader))
    cer = compute_cer(all_preds, all_targets)
    wer = compute_wer(all_preds, all_targets)
    acc = compute_exact_accuracy(all_preds, all_targets)
    
    results = {
        'checkpoint_path': checkpoint_path,
        'split': split,
        'num_samples': len(all_targets),
        'avg_loss': avg_loss,
        'cer': cer,
        'wer': wer,
        'exact_match_accuracy': acc,
        'elapsed_seconds': elapsed,
        'samples_per_second': len(all_targets) / max(0.001, elapsed),
        'sample_comparisons': sample_comparisons
    }
    
    print("\n=== Evaluation Results ===")
    print(f"Split: {split}")
    print(f"Total Samples: {len(all_targets):,}")
    print(f"Average CTC Loss: {avg_loss:.4f}")
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")
    print(f"Exact Match Accuracy: {acc * 100:.2f}%")
    print(f"Inference Speed: {results['samples_per_second']:.1f} samples/sec")
    
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_json}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tamil CRNN Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--zip_path", type=str, default="datasets/raw/tamil_word_dataset.zip")
    parser.add_argument("--extracted_dir", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default="reports/tamil_ctc_char_vocab.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    
    args = parser.parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        zip_path=args.zip_path,
        extracted_dir=args.extracted_dir,
        vocab_path=args.vocab_path,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_json=args.output_json
    )

if __name__ == "__main__":
    main()
