"""
experiments/u16_dual_dataset/dual_dataset.py - 50/50 Balanced Dual-Stream Dataset & Batching.

Combines:
- Stream A: Legacy tamil_word_dataset (word lexicon indexed from ZIP)
- Stream B: New IIIT-Indic-HW-UC (unconstrained word/line crops from disk)

In each training batch of size 128:
- Exactly 64 samples are drawn from Stream A.
- Exactly 64 samples are drawn from Stream B.
Epoch length is determined by Stream B (ceil(142,935 / 64) = 2,234 steps).
Stream A reshuffles and smoothly cycles when exhausted.
"""

import os
import sys
import math
import random
from typing import Dict, Any, List, Optional, Tuple, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import TamilWordDataset, pad_collate_fn
from src.data.dataset_hw_uc import IIITIndicHWUCDataset
from src.data.vocab import TamilCTCCharacterVocabulary

class BalancedDualStreamBatchSampler:
    """
    Yields batches of indices or coordinates such that each batch contains
    exactly half samples from Dataset A and half from Dataset B.
    """
    def __init__(self,
                 len_a: int,
                 len_b: int,
                 batch_size_a: int = 64,
                 batch_size_b: int = 64,
                 shuffle: bool = True):
        self.len_a = len_a
        self.len_b = len_b
        self.batch_size_a = batch_size_a
        self.batch_size_b = batch_size_b
        self.shuffle = shuffle
        
        # Epoch length governed by the larger dataset
        self.num_batches = math.ceil(max(self.len_a / self.batch_size_a, self.len_b / self.batch_size_b))
        
    def __len__(self) -> int:
        return self.num_batches
        
    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        # Generate permutation indices
        indices_a = list(range(self.len_a))
        indices_b = list(range(self.len_b))
        
        if self.shuffle:
            random.shuffle(indices_a)
            random.shuffle(indices_b)
            
        pos_a = 0
        pos_b = 0
        
        for _ in range(self.num_batches):
            # Batch from A
            batch_a = []
            while len(batch_a) < self.batch_size_a:
                needed = self.batch_size_a - len(batch_a)
                chunk = indices_a[pos_a : pos_a + needed]
                batch_a.extend(chunk)
                pos_a += len(chunk)
                if pos_a >= self.len_a:
                    if self.shuffle:
                        random.shuffle(indices_a)
                    pos_a = 0
                    
            # Batch from B
            batch_b = []
            while len(batch_b) < self.batch_size_b:
                needed = self.batch_size_b - len(batch_b)
                chunk = indices_b[pos_b : pos_b + needed]
                batch_b.extend(chunk)
                pos_b += len(chunk)
                if pos_b >= self.len_b:
                    if self.shuffle:
                        random.shuffle(indices_b)
                    pos_b = 0
                    
            yield batch_a, batch_b

def dual_pad_collate_fn(items_a: List[Dict[str, Any]], items_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combines two lists of items (from Dataset A and Dataset B) into a unified padded batch.
    Ensures per-sample valid CTC length calculation.
    """
    batch = items_a + items_b
    batch_size = len(batch)
    max_w = max(item['width'] for item in batch)
    
    # Pad to nearest multiple of 4
    if max_w % 4 != 0:
        max_w = ((max_w // 4) + 1) * 4
        
    target_height = batch[0]['image'].shape[1] # 64
    padded_images = torch.zeros(batch_size, 1, target_height, max_w, dtype=torch.float32)
    
    targets = []
    target_lengths = []
    texts = []
    img_paths = []
    widths = []
    datasets = []
    
    for i, item in enumerate(batch):
        w = item['width']
        widths.append(w)
        padded_images[i, :, :, :w] = item['image']
        targets.extend(item['target'].tolist())
        target_lengths.append(item['target_length'])
        texts.append(item['text'])
        img_paths.append(item['img_path'])
        datasets.append(item.get('dataset', 'unknown'))
        
    # True physical CNN downsampled sequence length: T = width // 4
    # NEVER artificially inflate using max(target_length, ...)
    input_lengths = torch.tensor([item['width'] // 4 for item in batch], dtype=torch.long)
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)
    
    # Enforce strict CTC alignment: T >= L + R for every individual sample
    for item in batch:
        t_seq = item['target'].tolist()
        l_len = len(t_seq)
        r_rep = sum(1 for k in range(l_len - 1) if t_seq[k] == t_seq[k+1])
        t_min = l_len + r_rep
        t_actual = item['width'] // 4
        if t_actual < t_min:
            raise ValueError(
                f"CTC alignment violation in batch: sample '{item['img_path']}' "
                f"(text: '{item['text']}') has physical frames T={t_actual} < "
                f"required minimum T_min={t_min} (L={l_len}, R={r_rep}, width={item['width']}px)"
            )
    
    return {
        'images': padded_images, # (B, 1, 64, max_w)
        'targets': torch.tensor(targets, dtype=torch.long),
        'target_lengths': torch.tensor(target_lengths, dtype=torch.long),
        'input_lengths': input_lengths,
        'widths': widths,
        'texts': texts,
        'img_paths': img_paths,
        'datasets': datasets,
        'batch_size_a': len(items_a),
        'batch_size_b': len(items_b)
    }

class DualStreamDataLoader:
    """
    DataLoader wrapper that coordinates balanced dual-stream batching.
    """
    def __init__(self,
                 dataset_a: Dataset,
                 dataset_b: Dataset,
                 batch_size_a: int = 64,
                 batch_size_b: int = 64,
                 shuffle: bool = True):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.batch_size_a = batch_size_a
        self.batch_size_b = batch_size_b
        self.sampler = BalancedDualStreamBatchSampler(
            len(dataset_a),
            len(dataset_b),
            batch_size_a=batch_size_a,
            batch_size_b=batch_size_b,
            shuffle=shuffle
        )
        
    def __len__(self) -> int:
        return len(self.sampler)
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for indices_a, indices_b in self.sampler:
            items_a = [self.dataset_a[idx] for idx in indices_a]
            items_b = [self.dataset_b[idx] for idx in indices_b]
            yield dual_pad_collate_fn(items_a, items_b)
