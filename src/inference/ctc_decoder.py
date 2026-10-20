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

from typing import List, Tuple
import unicodedata
import torch
from src.data.vocab import TamilCTCCharacterVocabulary

def ctc_greedy_decode(logits: torch.Tensor,
                      vocab: TamilCTCCharacterVocabulary,
                      blank_idx: int = 0) -> List[str]:
    """
    Greedy best-path decoding for CTC output with Unicode NFC normalization.
    
    Args:
        logits: Tensor of shape (T, B, num_classes) or (B, T, num_classes)
        vocab: Instance of TamilCTCCharacterVocabulary
        blank_idx: Index of CTC blank token (default: 0)
        
    Returns:
        List of decoded Unicode strings of length B.
    """
    if logits.dim() == 3 and logits.size(0) != logits.size(1):
        logits_b_first = logits.permute(1, 0, 2)
    elif logits.dim() == 2:
        logits_b_first = logits.unsqueeze(0)
    else:
        logits_b_first = logits

    predictions = torch.argmax(logits_b_first, dim=-1).cpu().numpy() # (B, T)
    
    decoded_texts: List[str] = []
    for seq in predictions:
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

def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Character Error Rate: sum(edit_distance(pred, target)) / sum(len(target)).
    Applies Unicode NFC normalization before calculating CER.
    """
    total_distance = 0
    total_target_chars = 0
    for pred, target in zip(predictions, targets):
        p_norm = unicodedata.normalize('NFC', pred)
        t_norm = unicodedata.normalize('NFC', target)
        p_chars = list(p_norm)
        t_chars = list(t_norm)
        dist = levenshtein_distance(p_chars, t_chars)
        total_distance += dist
        total_target_chars += max(1, len(t_chars))
        
    return total_distance / max(1, total_target_chars)

def compute_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Compute Word Error Rate: sum(edit_distance(pred_words, target_words)) / sum(len(target_words)).
    Applies Unicode NFC normalization before calculating WER.
    """
    total_distance = 0
    total_target_words = 0
    for pred, target in zip(predictions, targets):
        p_norm = unicodedata.normalize('NFC', pred)
        t_norm = unicodedata.normalize('NFC', target)
        p_words = p_norm.split()
        t_words = t_norm.split()
        dist = levenshtein_distance(p_words, t_words)
        total_distance += dist
        total_target_words += max(1, len(t_words))
        
    return total_distance / max(1, total_target_words)

def compute_exact_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Exact string match accuracy with Unicode NFC normalization.
    """
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, t in zip(predictions, targets)
        if unicodedata.normalize('NFC', p.strip()) == unicodedata.normalize('NFC', t.strip())
    )
    return correct / len(predictions)
