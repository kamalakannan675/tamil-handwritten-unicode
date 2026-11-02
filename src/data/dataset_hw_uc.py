"""
dataset_hw_uc.py - Dedicated PyTorch Dataset and Collation for IIIT-Indic-HW-UC Tamil Word Dataset.

Features:
1. Direct disk loading from extracted IIIT-Indic-HW-UC directory structure:
   - datasets/iiit_indic_hw_uc/tamil/1_New_Annoatation_Google/01_tamil_word_images/...
   - datasets/iiit_indic_hw_uc/tamil/2_New_Manually_Annotated/...
2. Reads tab-separated ground truth files (e.g. train_gt.txt, val_gt.txt, or custom document-split files):
   Format: '<relative_image_path>\\t<authentic_unicode_text>'
3. Strict Unicode normalization (NFC) preserving authentic Tamil text without destructive loss.
4. Clean handling of stray ASCII punctuation / digits with detailed reporting of unsupported samples.
5. Aspect-ratio preserving resizing to target_height (default: 64) and variable-width batch collation.
6. Full compatibility with TamilCRNN and CTC loss in src/models/crnn.py and src/training/train_crnn.py.
"""

import os
import io
import unicodedata
from typing import List, Tuple, Dict, Optional, Union, Any, Set
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.vocab import TamilCTCCharacterVocabulary
from src.data.dataset import pad_collate_fn

class IIITIndicHWUCDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: Optional[str] = 'train',
                 gt_path: Optional[str] = None,
                 vocab: Optional[TamilCTCCharacterVocabulary] = None,
                 vocab_path: Optional[str] = None,
                 target_height: int = 64,
                 max_width: int = 512,
                 filter_oov: bool = True,
                 clean_punctuation: bool = True,
                 cache_in_memory: bool = False,
                 max_samples: Optional[int] = None):
        """
        Args:
            root_dir: Path to extracted dataset root (e.g. datasets/iiit_indic_hw_uc/tamil)
            split: 'train', 'val', or 'test'. Used to locate file/<split>_gt.txt if gt_path is not specified.
            gt_path: Optional explicit path to a tab-separated ground truth file.
            vocab: Pre-loaded TamilCTCCharacterVocabulary instance.
            vocab_path: Path to tamil_ctc_char_vocab.json if vocab instance is not supplied.
            target_height: Fixed image height (default: 64, matching TamilCRNN).
            max_width: Maximum image width after aspect-ratio scaling (default: 512).
            filter_oov: If True, skips samples containing characters not in vocab.
            clean_punctuation: If True, strips leading/trailing non-Tamil punctuation marks before vocab validation.
            cache_in_memory: If True, caches preprocessed uint8 image arrays in RAM for ultra-fast subsequent epochs.
            max_samples: Optional limit on samples for quick testing or debugging.
        """
        self.root_dir = os.path.abspath(root_dir)
        self.split = split.lower() if split else 'custom'
        self.target_height = target_height
        self.max_width = max_width
        self.filter_oov = filter_oov
        self.clean_punctuation = clean_punctuation
        self.cache_in_memory = cache_in_memory
        self._cache: Dict[int, Tuple[np.ndarray, int]] = {}
        self.max_samples = max_samples

        # 1. Load Vocabulary
        if vocab is not None:
            self.vocab = vocab
        elif vocab_path and os.path.exists(vocab_path):
            self.vocab = TamilCTCCharacterVocabulary(vocab_file=vocab_path)
        else:
            default_vocab = os.path.abspath('checkpoints/crnn/tamil_ctc_char_vocab.json')
            if os.path.exists(default_vocab):
                self.vocab = TamilCTCCharacterVocabulary(vocab_file=default_vocab)
            else:
                raise ValueError("A valid vocabulary instance or vocab_path must be provided.")

        self.allowed_chars: Set[str] = set(self.vocab.char2idx.keys())

        # 2. Resolve Ground Truth File
        if gt_path and os.path.exists(gt_path):
            self.gt_path = os.path.abspath(gt_path)
        else:
            candidate_gt = os.path.join(self.root_dir, 'file', f'{self.split}_gt.txt')
            if os.path.exists(candidate_gt):
                self.gt_path = candidate_gt
            else:
                raise FileNotFoundError(f"Ground truth file not found at '{candidate_gt}' or specified gt_path.")

        # 3. Index and Validate Records
        self.records: List[Tuple[str, str, List[int]]] = []
        self.dropped_records: List[Tuple[str, str, str]] = []  # (path, original_text, reason)
        self.total_lines_read: int = 0

        self._index_dataset()

    def _clean_text(self, text: str) -> str:
        """Applies canonical composition NFC and optional edge-punctuation stripping."""
        norm_text = unicodedata.normalize('NFC', text.strip())
        if self.clean_punctuation:
            # Strip non-Tamil leading and trailing characters (e.g. quotes, periods, commas, brackets)
            def is_edge_punct(c: str) -> bool:
                if 0x0B80 <= ord(c) <= 0x0BFF:
                    return False
                return c in '.,:;\'"?!()[]{}*&/\\|-·”’“‘'

            start = 0
            while start < len(norm_text) and is_edge_punct(norm_text[start]):
                start += 1
            end = len(norm_text)
            while end > start and is_edge_punct(norm_text[end - 1]):
                end -= 1
            norm_text = norm_text[start:end]
        return norm_text

    def _index_dataset(self):
        """Parses the tab-separated ground truth file and indexes valid samples."""
        with open(self.gt_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                self.total_lines_read += 1

                parts = line.split('\t')
                if len(parts) < 2:
                    self.dropped_records.append((line, "", "Malformed line (missing tab separator)"))
                    continue

                rel_img_path = parts[0].strip()
                raw_text = parts[1].strip()

                cleaned_text = self._clean_text(raw_text)
                if not cleaned_text:
                    self.dropped_records.append((rel_img_path, raw_text, "Empty transcription after cleaning"))
                    continue

                # Check character vocabulary compatibility
                oov_chars = [c for c in cleaned_text if c not in self.allowed_chars]
                if oov_chars:
                    if self.filter_oov:
                        self.dropped_records.append(
                            (rel_img_path, cleaned_text, f"Contains OOV characters: {oov_chars}")
                        )
                        continue

                # Encode CTC targets
                try:
                    target_indices = self.vocab.encode(cleaned_text)
                    if len(target_indices) == 0:
                        self.dropped_records.append((rel_img_path, cleaned_text, "Zero tokens encoded"))
                        continue
                except Exception as e:
                    self.dropped_records.append((rel_img_path, cleaned_text, f"Encoding error: {e}"))
                    continue

                self.records.append((rel_img_path, cleaned_text, target_indices))

                if self.max_samples and len(self.records) >= self.max_samples:
                    break

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics of the indexed dataset."""
        return {
            'total_lines_read': self.total_lines_read,
            'indexed_samples': len(self.records),
            'dropped_samples': len(self.dropped_records),
            'retention_rate': len(self.records) / max(1, self.total_lines_read),
            'split': self.split,
            'gt_path': self.gt_path
        }

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_full_path(self, rel_path: str) -> str:
        """Resolves relative image path with cross-platform separator normalization."""
        norm_rel = os.path.normpath(rel_path)
        full_path = os.path.join(self.root_dir, norm_rel)
        if not os.path.exists(full_path):
            # Check if root_dir already includes or excludes top-level 'tamil'
            alt_path = os.path.join(os.path.dirname(self.root_dir), norm_rel)
            if os.path.exists(alt_path):
                return alt_path
            raise FileNotFoundError(f"Image not found at '{full_path}' or '{alt_path}'.")
        return full_path

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        rel_img_path, text, target_indices = self.records[idx]

        if self.cache_in_memory and idx in self._cache:
            img_uint8, new_w = self._cache[idx]
        else:
            # 1. Load image and convert to Grayscale
            full_path = self._resolve_full_path(rel_img_path)
            with Image.open(full_path) as im:
                gray = im.convert('L')

            orig_w, orig_h = gray.size

            # 2. Aspect-ratio preserving width scaling to target_height (default: 64)
            new_w = max(16, min(self.max_width, int(round(orig_w * self.target_height / max(1, orig_h)))))
            resized = gray.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
            img_uint8 = np.array(resized, dtype=np.uint8)
            if self.cache_in_memory:
                self._cache[idx] = (img_uint8, new_w)

        # 3. Invert and normalize to [0.0, 1.0] (handwritten stroke = 1.0, background = 0.0)
        img_np = (255.0 - img_uint8.astype(np.float32)) / 255.0
        tensor_img = torch.from_numpy(img_np).unsqueeze(0)  # (1, 64, W)

        return {
            'image': tensor_img,
            'text': text,
            'target': torch.tensor(target_indices, dtype=torch.long),
            'target_length': len(target_indices),
            'width': new_w,
            'img_path': rel_img_path
        }
