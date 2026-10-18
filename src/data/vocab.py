"""
vocab.py - Character Tokenizer and Vocabulary for Tamil CTC OCR.

Maps the 72 unique Unicode characters found in the IIIT-INDIC-HW Tamil dataset
to integer indices with CTC <blank> token reserved at index 0.

Provides:
- Token-to-ID and ID-to-Token bidirectional encoding/decoding.
- Greedy CTC decoding with repeated character collapsing and blank removal.
- Standardized character dictionary serialization to JSON.
"""

import os
import json
from typing import List, Dict

CTC_BLANK_TOKEN = "<blank>"
CTC_BLANK_ID = 0

class TamilCTCCharacterVocabulary:
    def __init__(self, vocab_file: str = None):
        self.blank_token = CTC_BLANK_TOKEN
        self.blank_id = CTC_BLANK_ID
        self.blank_idx = CTC_BLANK_ID
        
        self.char2idx: Dict[str, int] = {self.blank_token: self.blank_id}
        self.idx2char: Dict[int, str] = {self.blank_id: self.blank_token}
        
        if vocab_file and os.path.exists(vocab_file):
            self.load(vocab_file)

    def load_from_json(self, json_path: str):
        return self.load(json_path)

    def build_from_words(self, words: List[str]):
        """Extracts unique characters from word list and builds vocabulary."""
        unique_chars = sorted(list(set("".join(words))))
        self.char2idx = {self.blank_token: self.blank_id}
        self.idx2char = {self.blank_id: self.blank_token}
        
        for idx, char in enumerate(unique_chars, start=1):
            self.char2idx[char] = idx
            self.idx2char[idx] = char

    def __len__(self) -> int:
        return len(self.char2idx)

    def encode(self, text: str) -> List[int]:
        """Encodes string to list of character integer IDs."""
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices: List[int]) -> str:
        """Decodes raw index sequence without CTC collapsing."""
        return "".join([self.idx2char.get(i, "") for i in indices if i != self.blank_id])

    def ctc_decode(self, indices: List[int]) -> str:
        """
        Greedy CTC decoding:
        1. Collapse consecutive identical indices.
        2. Filter out CTC blank tokens.
        """
        collapsed = []
        prev = None
        for idx in indices:
            if idx != prev:
                collapsed.append(idx)
                prev = idx
        
        # Remove blank tokens and convert to characters
        chars = [self.idx2char.get(i, "") for i in collapsed if i != self.blank_id]
        return "".join(chars)

    def save(self, json_path: str):
        data = {
            'blank_token': self.blank_token,
            'blank_id': self.blank_id,
            'num_tokens': len(self),
            'char2idx': self.char2idx,
            'idx2char': {str(k): v for k, v in self.idx2char.items()}
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.blank_token = data['blank_token']
        self.blank_id = data['blank_id']
        self.char2idx = data['char2idx']
        self.idx2char = {int(k): v for k, v in data['idx2char'].items()}
