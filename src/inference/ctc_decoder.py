"""
ctc_decoder.py - CTC Decoding and Evaluation Metrics for Tamil OCR.

Implements:
1. ctc_greedy_decode: Argmax best-path decoding with CTC blank removal, deduplication,
   and Unicode NFC canonical normalization.
2. levenshtein_distance: Character and word-level edit distance.
3. compute_cer: Character Error Rate with NFC normalization.
4. compute_wer: Word Error Rate with NFC normalization.
5. compute_exact_accuracy: Exact sequence match accuracy.
"""

from typing import List, Tuple, Optional, Union
import unicodedata
import torch
from src.data.vocab import TamilCTCCharacterVocabulary

def ctc_greedy_decode(logits: torch.Tensor,
                      vocab: TamilCTCCharacterVocabulary,
                      blank_idx: int = 0,
                      input_lengths: Optional[Union[List[int], torch.Tensor]] = None) -> List[str]:
    """
    Greedy best-path decoding for CTC output with Unicode NFC normalization.
    
    Args:
        logits: Tensor of shape (T, B, num_classes) or (B, T, num_classes)
        vocab: Instance of TamilCTCCharacterVocabulary
        blank_idx: Index of CTC blank token (default: 0)
        input_lengths: Optional valid horizontal sequence lengths per sample T_b (W // 4).
                       If provided, truncates predictions to ignore padded background frames.
        
    Returns:
        List of decoded Unicode strings of length B.
    """
    if logits.dim() == 2:
        logits_b_first = logits.unsqueeze(0)
    elif logits.dim() == 3:
        if input_lengths is not None:
            n_samples = len(input_lengths)
            if logits.size(1) == n_samples:
                logits_b_first = logits.permute(1, 0, 2)
            elif logits.size(0) == n_samples:
                logits_b_first = logits
            else:
                logits_b_first = logits.permute(1, 0, 2) if logits.size(0) != logits.size(1) else logits
        else:
            # Default CRNN output convention is (T, B, C)
            logits_b_first = logits.permute(1, 0, 2) if logits.size(0) != logits.size(1) else logits
    else:
        logits_b_first = logits

    predictions = torch.argmax(logits_b_first, dim=-1).cpu().numpy() # (B, T)
    
    # Convert input_lengths to list of ints if provided
    lengths_list = None
    if input_lengths is not None:
        if isinstance(input_lengths, torch.Tensor):
            lengths_list = input_lengths.cpu().tolist()
        else:
            lengths_list = list(input_lengths)

    decoded_texts: List[str] = []
    for b_idx, seq in enumerate(predictions):
        if lengths_list is not None and b_idx < len(lengths_list):
            valid_len = int(lengths_list[b_idx])
            seq = seq[:valid_len]

        collapsed_tokens: List[int] = []
        prev_token = None
        for token in seq:
            # Collapse consecutive identical tokens
            if token != prev_token:
                if token != blank_idx:
                    collapsed_tokens.append(int(token))
                prev_token = token
        raw_decoded = vocab.decode(collapsed_tokens)
        norm_decoded = unicodedata.normalize('NFC', raw_decoded)
        decoded_texts.append(norm_decoded)
        
    return decoded_texts

def levenshtein_distance(seq1: List, seq2: List) -> int:
    """
    Standard dynamic programming Levenshtein distance between two sequences.
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
        
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
            
    return dp[n][m]

def compute_cer(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Character Error Rate (CER) across a corpus:
    CER = sum(Levenshtein(pred, gt)) / sum(len(gt))
    Strict Unicode NFC normalization applied to prevent decomposed vowel discrepancies.
    """
    total_distance = 0
    total_length = 0
    
    for pred, gt in zip(predictions, ground_truths):
        norm_pred = unicodedata.normalize('NFC', pred)
        norm_gt = unicodedata.normalize('NFC', gt)
        
        pred_chars = list(norm_pred)
        gt_chars = list(norm_gt)
        
        total_distance += levenshtein_distance(pred_chars, gt_chars)
        total_length += len(gt_chars)
        
    if total_length == 0:
        return 0.0
    return total_distance / total_length

def compute_wer(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Word Error Rate (WER) across a corpus:
    WER = sum(Levenshtein(pred_words, gt_words)) / sum(len(gt_words))
    """
    total_distance = 0
    total_length = 0
    
    for pred, gt in zip(predictions, ground_truths):
        norm_pred = unicodedata.normalize('NFC', pred)
        norm_gt = unicodedata.normalize('NFC', gt)
        
        pred_words = norm_pred.split()
        gt_words = norm_gt.split()
        
        total_distance += levenshtein_distance(pred_words, gt_words)
        total_length += len(gt_words)
        
    if total_length == 0:
        return 0.0
    return total_distance / total_length

def compute_exact_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Exact Match Accuracy: percentage of predictions identical to ground truth under NFC normalization.
    """
    if not ground_truths:
        return 0.0
        
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        norm_pred = unicodedata.normalize('NFC', pred)
        norm_gt = unicodedata.normalize('NFC', gt)
        if norm_pred == norm_gt:
            correct += 1
            
    return correct / len(ground_truths)
