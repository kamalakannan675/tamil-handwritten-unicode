"""
dataset.py - Memory-Efficient PyTorch Dataset and Collate for Tamil Handwritten Word Recognition.

Supports:
1. Direct lazy access to images from nested ZIPs (e.g. datasets/raw/tamil_word_dataset.zip -> val.zip)
   OR from extracted disk directories (e.g. /kaggle/working/data/extracted/val/1.jpg).
2. Parsing train.txt, val.txt, test.txt mapping: '<rel_image_path>, <vocab_id>'.
3. Resolving vocab_id to authentic Tamil Unicode string using vocab.txt.
4. Aspect-ratio preserving resizing to target_height (default 64) and proportional width scaling.
5. Inverting and normalizing grayscale images to [0.0, 1.0].
6. Variable-width batching with pad_collate_fn ensuring valid CTC sequence lengths.
"""

import os
import io
import zipfile
from typing import List, Tuple, Dict, Optional, Union, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.vocab import TamilCTCCharacterVocabulary

class TamilWordDataset(Dataset):
    def __init__(self,
                 split: str,
                 zip_path: Optional[str] = None,
                 extracted_dir: Optional[str] = None,
                 vocab: Optional[TamilCTCCharacterVocabulary] = None,
                 vocab_txt_path: Optional[str] = None,
                 target_height: int = 64,
                 max_width: int = 512,
                 max_samples: Optional[int] = None):
        """
        Args:
            split: 'train', 'val', or 'test'
            zip_path: Path to main zip archive (e.g. datasets/raw/tamil_word_dataset.zip)
            extracted_dir: Path to directory containing extracted split folders (e.g. /kaggle/working/extracted)
            vocab: Pre-loaded TamilCTCCharacterVocabulary instance
            vocab_txt_path: Path to vocab.txt if extracted
            target_height: Fixed image height (default: 64)
            max_width: Maximum image width after aspect-ratio scaling (default: 512)
            max_samples: Optional limit on samples for debugging/smoke testing
        """
        self.split = split.lower()
        assert self.split in ['train', 'val', 'test'], f"Invalid split: {split}"
        
        self.zip_path = zip_path
        self.extracted_dir = extracted_dir
        self.target_height = target_height
        self.max_width = max_width
        
        # Load Vocabulary
        if vocab is not None:
            self.vocab = vocab
        elif vocab_txt_path and os.path.exists(vocab_txt_path):
            with open(vocab_txt_path, 'r', encoding='utf-8') as f:
                words = f.read().splitlines()
            self.vocab = TamilCTCCharacterVocabulary()
            self.vocab.build_from_words(words)
        elif self.zip_path and os.path.exists(self.zip_path):
            with zipfile.ZipFile(self.zip_path, 'r') as z:
                with z.open('vocab.txt') as f:
                    words = f.read().decode('utf-8').splitlines()
            self.vocab = TamilCTCCharacterVocabulary()
            self.vocab.build_from_words(words)
        else:
            raise ValueError("Either vocab, vocab_txt_path, or valid zip_path must be provided.")
            
        # Load Word Lexicon (the full word list that maps vocab_id -> Tamil word string)
        # Priority: explicit vocab_txt_path -> extracted_dir/vocab.txt -> nested zip
        self.word_lexicon: List[str] = []
        if vocab_txt_path and os.path.exists(vocab_txt_path):
            with open(vocab_txt_path, 'r', encoding='utf-8') as f:
                self.word_lexicon = f.read().splitlines()
        elif self.extracted_dir:
            vocab_in_dir = os.path.join(self.extracted_dir, 'vocab.txt')
            if os.path.exists(vocab_in_dir):
                with open(vocab_in_dir, 'r', encoding='utf-8') as f:
                    self.word_lexicon = f.read().splitlines()
        if not self.word_lexicon and self.zip_path and os.path.exists(self.zip_path):
            with zipfile.ZipFile(self.zip_path, 'r') as z:
                with z.open('vocab.txt') as f:
                    self.word_lexicon = f.read().decode('utf-8').splitlines()

        if not self.word_lexicon:
            raise ValueError(
                "Word lexicon (vocab.txt) could not be loaded. "
                "Provide vocab_txt_path=, or ensure vocab.txt exists in extracted_dir or zip_path."
            )
                    
        # Parse split mapping file (e.g. val.txt)
        txt_filename = f"{self.split}.txt"
        lines = []
        if self.extracted_dir and os.path.exists(os.path.join(self.extracted_dir, txt_filename)):
            with open(os.path.join(self.extracted_dir, txt_filename), 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f.read().splitlines() if l.strip()]
        elif self.zip_path and os.path.exists(self.zip_path):
            with zipfile.ZipFile(self.zip_path, 'r') as z:
                with z.open(txt_filename) as f:
                    lines = [l.strip() for l in f.read().decode('utf-8').splitlines() if l.strip()]
        else:
            raise FileNotFoundError(f"Could not locate {txt_filename} in zip or extracted directory.")
            
        self.records: List[Tuple[str, int, str]] = []
        for line_no, line in enumerate(lines, start=1):
            parts = line.split(',')
            if len(parts) != 2:
                continue
            img_rel_path = parts[0].strip()
            try:
                vocab_id = int(parts[1].strip())
            except ValueError:
                continue
                
            if 0 <= vocab_id < len(self.word_lexicon):
                word_text = self.word_lexicon[vocab_id]
                self.records.append((img_rel_path, vocab_id, word_text))
                
        if max_samples and max_samples > 0:
            self.records = self.records[:max_samples]
            
        # Zipfile handles for lazy loading (instantiated per process/thread)
        self._main_zip = None
        self._nested_zip = None

    def _get_zip_handles(self):
        """Lazy thread-safe initialization of zip reader."""
        if self._main_zip is None and self.zip_path:
            self._main_zip = zipfile.ZipFile(self.zip_path, 'r')
            nested_name = f"{self.split}.zip"
            nested_bytes = io.BytesIO(self._main_zip.read(nested_name))
            self._nested_zip = zipfile.ZipFile(nested_bytes, 'r')
        return self._nested_zip

    def __len__(self) -> int:
        return len(self.records)

    def _load_pil_image(self, img_rel_path: str) -> Image.Image:
        """Loads image from disk or from nested zip archive."""
        # Check extracted directory on disk first (fastest for Kaggle)
        if self.extracted_dir:
            full_path = os.path.join(self.extracted_dir, img_rel_path)
            if not os.path.exists(full_path):
                # Sometimes extracted as train/1.jpg or just 1.jpg
                full_path = os.path.join(self.extracted_dir, os.path.basename(img_rel_path))
            if os.path.exists(full_path):
                return Image.open(full_path).convert('L')
                
        # Fallback to reading directly from nested zip
        if self.zip_path:
            nested_z = self._get_zip_handles()
            with nested_z.open(img_rel_path) as f:
                return Image.open(f).convert('L')
                
        raise FileNotFoundError(f"Image {img_rel_path} not found in disk or zip storage.")

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        img_rel_path, vocab_id, word_text = self.records[idx]
        
        # 1. Load image as Grayscale PIL
        gray = self._load_pil_image(img_rel_path)
        orig_w, orig_h = gray.size
        
        # 2. Aspect-ratio preserving width scaling to target height (64px)
        # Ensure minimum width of 16px to guarantee valid downsampled CTC sequence length
        new_w = max(16, min(self.max_width, int(round(orig_w * self.target_height / max(1, orig_h)))))
        resized = gray.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
        
        # 3. Invert and normalize to [0.0, 1.0] (handwritten stroke = 1.0, background = 0.0)
        img_np = (255.0 - np.array(resized, dtype=np.float32)) / 255.0
        tensor_img = torch.from_numpy(img_np).unsqueeze(0) # (1, H, W)
        
        # 4. Character Token Encoding
        target_indices = self.vocab.encode(word_text)
        
        return {
            'image': tensor_img,
            'text': word_text,
            'target': torch.tensor(target_indices, dtype=torch.long),
            'target_length': len(target_indices),
            'width': new_w,
            'img_path': img_rel_path
        }

def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-width word images.
    Pads each batch to the maximum width present in that specific batch (capped at multiple of 4).
    Calculates exact CTC input sequence lengths (T = padded_width // 4).
    """
    batch_size = len(batch)
    max_w = max(item['width'] for item in batch)
    
    # Pad to nearest multiple of 4 (since CNN downsamples width by factor of 4)
    if max_w % 4 != 0:
        max_w = ((max_w // 4) + 1) * 4
        
    target_height = batch[0]['image'].shape[1] # 64
    
    # Initialize padded batch with zeros (background after inversion is 0.0)
    padded_images = torch.zeros(batch_size, 1, target_height, max_w, dtype=torch.float32)
    
    targets = []
    target_lengths = []
    texts = []
    img_paths = []
    
    for i, item in enumerate(batch):
        w = item['width']
        padded_images[i, :, :, :w] = item['image']
        targets.extend(item['target'].tolist())
        target_lengths.append(item['target_length'])
        texts.append(item['text'])
        img_paths.append(item['img_path'])
        
    input_length = max_w // 4
    input_lengths = torch.full((batch_size,), fill_value=input_length, dtype=torch.long)
    
    return {
        'images': padded_images, # (B, 1, 64, max_w)
        'targets': torch.tensor(targets, dtype=torch.long), # 1D flattened targets
        'target_lengths': torch.tensor(target_lengths, dtype=torch.long),
        'input_lengths': input_lengths,
        'texts': texts,
        'img_paths': img_paths
    }
