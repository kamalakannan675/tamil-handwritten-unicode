"""
vocab.py - Character Tokenizer and Vocabulary for Tamil CTC OCR.

Maps the 72 unique Unicode characters found in the IIIT-INDIC-HW Tamil dataset
to integer indices with CTC <blank> token reserved at index 0.

Provides:
- Token-to-ID and ID-to-Token bidirectional encoding/decoding.
- Greedy CTC decoding with repeated character collapsing and blank removal.
- Consistent Unicode NFC normalization preserving Tamil vowel signs and pulli marks.
- Standardized character dictionary serialization to JSON.
"""

import os
import json
import unicodedata
from typing import List, Dict

CTC_BLANK_TOKEN = "<blank>"
CTC_BLANK_ID = 0

# Tamil Combining Marks (cannot stand alone at the beginning of a word)
TAMIL_COMBINING_MARKS = {
    '\u0bcd',  # Pulli (Virama) ்
    '\u0bbe',  # Vowel sign AA ா
    '\u0bbf',  # Vowel sign I ி
    '\u0bc0',  # Vowel sign II ீ
    '\u0bc1',  # Vowel sign U ு
    '\u0bc2',  # Vowel sign UU ூ
    '\u0bc6',  # Vowel sign E ெ
    '\u0bc7',  # Vowel sign EE ே
    '\u0bc8',  # Vowel sign AI ை
    '\u0bca',  # Vowel sign O ொ
    '\u0bcb',  # Vowel sign OO ோ
    '\u0bcc',  # Vowel sign AU ௌ
    '\u0bd7',  # Length mark ௗ
}

# Tamil Orthography Constants for Post-Processing Validation
TAMIL_VIRAMA = '\u0bcd'  # Pulli ்

# Tamil consonants: U+0B95 to U+0BB9 (all base consonants in Unicode Tamil block)
TAMIL_CONSONANTS = {chr(c) for c in range(0x0b95, 0x0bb9 + 1)}

# Tamil vowel signs (combining vowel marks)
TAMIL_VOWEL_SIGNS = {
    '\u0bbe',  # Vowel sign AA ா
    '\u0bbf',  # Vowel sign I ி
    '\u0bc0',  # Vowel sign II ீ
    '\u0bc1',  # Vowel sign U ு
    '\u0bc2',  # Vowel sign UU ூ
    '\u0bc6',  # Vowel sign E ெ
    '\u0bc7',  # Vowel sign EE ே
    '\u0bc8',  # Vowel sign AI ை
    '\u0bca',  # Vowel sign O ொ
    '\u0bcb',  # Vowel sign OO ோ
    '\u0bcc',  # Vowel sign AU ௌ
    '\u0bd7',  # Length mark ௗ
}

def repair_geminate_virama(text: str) -> str:
    """
    Repairs misplaced virama sequences where CTC decoder emits:
    [VOWEL_SIGN] + [VIRAMA] + [CONSONANT C] + [VOWEL_SIGN]
    reconstructing the intended geminate:
    [CONSONANT C] + [VIRAMA] + [CONSONANT C] + [VOWEL_SIGN]

    In standard Tamil orthography, a virama (\u0bcd) immediately following
    a vowel sign is strictly illegal Unicode. In handwriting with geminates
    (e.g., 'க்கு'), CTC decoders often detect the upper pulli mark before
    the base consonant, omitting the first consonant stroke.

    Example:
        'திரு்குறள்' -> 'திருக்குறள்'
    """
    if not text or TAMIL_VIRAMA not in text:
        return text

    chars = list(text)
    n = len(chars)
    repaired = []
    i = 0

    while i < n:
        if (repaired and repaired[-1] in TAMIL_VOWEL_SIGNS and
            chars[i] == TAMIL_VIRAMA and
            i + 2 < n and
            chars[i + 1] in TAMIL_CONSONANTS and
            chars[i + 2] in TAMIL_VOWEL_SIGNS):

            cons = chars[i + 1]
            v_sign = chars[i + 2]
            repaired.extend([cons, TAMIL_VIRAMA, cons, v_sign])
            i += 3
        else:
            repaired.append(chars[i])
            i += 1

    return unicodedata.normalize('NFC', "".join(repaired))


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
        norm_text = unicodedata.normalize('NFC', text)
        # Note: if normalized decomposes differently than vocab keys, try fallback
        encoded = []
        for c in norm_text:
            if c in self.char2idx:
                encoded.append(self.char2idx[c])
            else:
                # Try NFD decomposed fallback
                decomposed = unicodedata.normalize('NFD', c)
                for dc in decomposed:
                    if dc in self.char2idx:
                        encoded.append(self.char2idx[dc])
        return encoded

    def decode(self, indices: List[int]) -> str:
        """Decodes raw index sequence without CTC collapsing, returned as canonical NFC string."""
        raw_str = "".join([self.idx2char.get(i, "") for i in indices if i != self.blank_id])
        return unicodedata.normalize('NFC', raw_str)

    def ctc_decode(self, indices: List[int]) -> str:
        """
        Greedy CTC decoding:
        1. Collapse consecutive identical indices.
        2. Filter out CTC blank tokens.
        3. Normalize to canonical Unicode NFC.
        """
        collapsed = []
        prev = None
        for idx in indices:
            if idx != prev:
                collapsed.append(idx)
                prev = idx
        
        # Remove blank tokens and convert to characters
        chars = [self.idx2char.get(i, "") for i in collapsed if i != self.blank_id]
        raw_str = "".join(chars)
        return unicodedata.normalize('NFC', raw_str)

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
