"""
full_page_ocr.py - End-to-End Tamil Handwritten Full-Page OCR Engine.

Orchestrates:
1. Document loading (from file path, byte stream, or PIL Image).
2. Zero-dependency Page & Word Segmentation (PageSegmenter).
3. Preprocessing word crops to aspect-ratio scaled grayscale tensors (H=64, W proportional),
   with adaptive background illumination normalization matching the IIIT-INDIC-HW training distribution.
4. CRNN + CTC batch inference.
5. Tamil Grapheme-Aware Unicode reconstruction:
   - Preserves Tamil base characters and combining marks (pulli, vowel signs).
   - Prevents artificial intra-word spaces.
   - Enforces Unicode NFC normalization.
6. Clean line and paragraph assembly preserving natural reading order.
7. Optional developer diagnostics (`debug=True`).
"""

import os
import unicodedata
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import torch

from src.data.vocab import TamilCTCCharacterVocabulary, TAMIL_COMBINING_MARKS, repair_geminate_virama
from src.models.crnn import TamilCRNN
from src.inference.ctc_decoder import ctc_greedy_decode
from src.ocr.page_segmentation import PageSegmenter

# Tamil Unicode Codepoint Sets for Non-Destructive Advisory Validation
TAMIL_INDEPENDENT_VOWELS = set(range(0x0B85, 0x0B95))  # அ to ஔ
TAMIL_CONSONANTS = set(range(0x0B95, 0x0BA0)) | set(range(0x0BA3, 0x0BA5)) | set(range(0x0BA8, 0x0BAF)) | set(range(0x0BB0, 0x0BBA))
TAMIL_VOWEL_SIGNS = set(range(0x0BBE, 0x0BCD)) | {0x0BD7}  # ா to ௌ, plus length mark ௗ
TAMIL_VIRAMA = 0x0BCD  # ்

def validate_tamil_unicode_sequence(text: str) -> List[Dict[str, Any]]:
    """
    Validates Tamil Unicode text and returns a list of structural/orthographic anomalies
    WITHOUT modifying or deleting characters (non-destructive advisory validator).
    
    Checks:
    1. LEADING_COMBINING_MARK: Combining mark or virama appearing without a preceding base character.
    2. INVALID_VOWEL_COMBINATION: Independent vowel (அ-ஔ) followed by combining mark or virama (e.g. 'உி').
    3. CONSECUTIVE_VIRAMA: Multiple consecutive virama marks (e.g. 'க்்').
    4. MULTIPLE_VOWEL_SIGNS: Multiple consecutive dependent vowel signs on a base character.
    """
    if not text:
        return []
        
    text_nfc = unicodedata.normalize('NFC', text)
    anomalies = []
    
    # Check 1: Leading combining mark at token starts
    tokens = text_nfc.split()
    token_offset = 0
    for token in tokens:
        idx = text_nfc.find(token, token_offset)
        if idx >= 0:
            token_offset = idx + len(token)
            if ord(token[0]) in (TAMIL_VOWEL_SIGNS | {TAMIL_VIRAMA}):
                anomalies.append({
                    'type': 'LEADING_COMBINING_MARK',
                    'pos': idx,
                    'sequence': token[0],
                    'codepoints': f"U+{ord(token[0]):04X}",
                    'description': f"Token begins with an orphaned combining mark '{token[0]}' without a base character"
                })
                
    # Check 2: Independent vowel immediately followed by combining mark or virama
    for i in range(len(text_nfc) - 1):
        c1, c2 = ord(text_nfc[i]), ord(text_nfc[i + 1])
        if c1 in TAMIL_INDEPENDENT_VOWELS and (c2 in TAMIL_VOWEL_SIGNS or c2 == TAMIL_VIRAMA):
            anomalies.append({
                'type': 'INVALID_VOWEL_COMBINATION',
                'pos': i,
                'sequence': text_nfc[i:i + 2],
                'codepoints': f"U+{c1:04X} + U+{c2:04X}",
                'description': f"Independent vowel '{chr(c1)}' is illegally followed by combining mark '{chr(c2)}'"
            })
            
    # Check 3: Consecutive viramas
    for i in range(len(text_nfc) - 1):
        c1, c2 = ord(text_nfc[i]), ord(text_nfc[i + 1])
        if c1 == TAMIL_VIRAMA and c2 == TAMIL_VIRAMA:
            anomalies.append({
                'type': 'CONSECUTIVE_VIRAMA',
                'pos': i,
                'sequence': text_nfc[i:i + 2],
                'codepoints': f"U+{c1:04X} + U+{c2:04X}",
                'description': 'Consecutive virama (pulli) marks detected'
            })
            
    # Check 4: Multiple consecutive vowel signs
    for i in range(len(text_nfc) - 1):
        c1, c2 = ord(text_nfc[i]), ord(text_nfc[i + 1])
        if c1 in TAMIL_VOWEL_SIGNS and c2 in TAMIL_VOWEL_SIGNS:
            anomalies.append({
                'type': 'MULTIPLE_VOWEL_SIGNS',
                'pos': i,
                'sequence': text_nfc[i:i + 2],
                'codepoints': f"U+{c1:04X} + U+{c2:04X}",
                'description': f"Consecutive dependent vowel signs '{chr(c1)}{chr(c2)}' detected"
            })
            
    return anomalies

class TamilFullPageOCR:
    def __init__(self,
                 checkpoint_path: str = "checkpoints/crnn/best_model.pt",
                 vocab_path: str = "reports/tamil_ctc_char_vocab.json",
                 device: str = "auto",
                 target_height: int = 64,
                 max_width: int = 512):
        """
        Initializes the Full-Page OCR pipeline.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.target_height = target_height
        self.max_width = max_width
        self.segmenter = PageSegmenter()
        
        # Load Vocabulary
        self.vocab = TamilCTCCharacterVocabulary()
        if os.path.exists(vocab_path):
            self.vocab.load_from_json(vocab_path)
        else:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
            
        self.num_classes = len(self.vocab)
        
        # Load Model Checkpoint if available
        self.model = None
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            hidden_size = ckpt.get('hidden_size', 256)
            self.model = TamilCRNN(num_classes=self.num_classes, in_channels=1, hidden_size=hidden_size).to(self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            print(f"[TamilFullPageOCR] Loaded weights from {checkpoint_path} onto {self.device}.")
        else:
            print(f"[TamilFullPageOCR] Warning: Checkpoint {checkpoint_path} not found. Running in mock/untrained mode.")
            self.model = TamilCRNN(num_classes=self.num_classes, in_channels=1, hidden_size=256).to(self.device)
            self.model.eval()

    def preprocess_word_crop(self, pil_crop: Image.Image) -> torch.Tensor:
        """
        Converts a PIL word crop into normalized (1, 1, 64, W) tensor where W is a multiple of 4.
        Aligns input with the IIIT-INDIC-HW dataset training distribution:
        - Grayscale conversion
        - Adaptive background normalization (maps paper background to ~255 and ink to ~0)
        - Aspect-ratio preserving bilinear resize to fixed height H=64
        - Width clamped to [16, max_width] and padded to multiple of 4
        - Inverted and normalized to [0.0, 1.0] (ink stroke = 1.0, background = 0.0)
        """
        gray = pil_crop.convert('L')
        orig_w, orig_h = gray.size
        
        arr = np.array(gray, dtype=np.float32)
        # Candidate D: ink-density-gated adaptive percentile stretching.
        ink_density = float(np.mean(arr < 128.0))
        p_lo_pct, p_hi_pct = (2, 98) if ink_density < 0.05 else (5, 95)
        p_low, p_high = np.percentile(arr, (p_lo_pct, p_hi_pct))
        if (p_high - p_low) > 15:
            arr = np.clip((arr - p_low) / (p_high - p_low) * 255.0, 0, 255)
        gray = Image.fromarray(arr.astype(np.uint8))
        
        new_w = max(16, min(self.max_width, int(round(orig_w * self.target_height / max(1, orig_h)))))
        # Pad width to nearest multiple of 4 (CNN downsamples W by 4)
        if new_w % 4 != 0:
            new_w = ((new_w // 4) + 1) * 4
            
        resized = gray.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
        img_np = (255.0 - np.array(resized, dtype=np.float32)) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0) # (1, 1, 64, W)
        return tensor

    def recognize_word_crop(self, pil_crop: Image.Image) -> str:
        """
        Performs CRNN forward pass and CTC decoding for a single word crop.
        Returns canonical Unicode NFC string.
        """
        tensor = self.preprocess_word_crop(pil_crop).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor) # (T, 1, num_classes)
            decoded = ctc_greedy_decode(logits, self.vocab, blank_idx=0)
        raw_word = decoded[0] if decoded else ""
        repaired_word = repair_geminate_virama(raw_word.strip())
        return unicodedata.normalize('NFC', repaired_word)

    def assess_image_quality(self, input_source: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Lightweight pre-flight image quality assessment.
        Estimates detected text line heights without running expensive neural inference.
        """
        if isinstance(input_source, str):
            page_img = Image.open(input_source).convert('RGB')
        else:
            page_img = input_source.convert('RGB')
        return self.segmenter.assess_image_quality(page_img)

    @staticmethod
    def assemble_line_words(words_list: List[str]) -> str:
        """
        Intelligently reconstructs a text line from recognized word tokens:
        1. Preserves true word boundaries with a single space.
        2. Merges orphaned combining marks (pulli ், vowel signs ா, ி, ீ, etc.)
           into the preceding word without inserting spaces.
        3. Normalizes final line to canonical Unicode NFC.
        """
        if not words_list:
            return ""
            
        assembled_parts = []
        for word in words_list:
            clean_word = repair_geminate_virama(word.strip())
            if not clean_word:
                continue
                
            # Check if this word token starts with a combining mark
            # In Tamil, combining marks cannot stand alone; attach to preceding token
            first_char = clean_word[0]
            if assembled_parts and first_char in TAMIL_COMBINING_MARKS:
                assembled_parts[-1] = assembled_parts[-1] + clean_word
            else:
                assembled_parts.append(clean_word)
                
        line_str = " ".join(assembled_parts)
        return unicodedata.normalize('NFC', line_str)

    def process_document(self, input_source: Union[str, Image.Image], debug: bool = False) -> Dict[str, Any]:
        """
        Performs full-page layout analysis, line/word segmentation, recognition,
        and natural reading-order Unicode assembly.
        
        Args:
            input_source: File path string or PIL.Image instance.
            debug: If True, include raw tokens and per-word diagnostic details.
            
        Returns:
            Dict containing:
            - 'full_text': Formatted multiline Tamil Unicode text
            - 'total_lines': Number of segmented lines
            - 'total_words': Number of segmented words
            - 'lines': Detailed structured hierarchy with bounding boxes
            - 'image_size': [width, height]
        """
        if isinstance(input_source, str):
            page_img = Image.open(input_source).convert('RGB')
        else:
            page_img = input_source.convert('RGB')
            
        # 1. Page Layout and Line/Word Segmentation
        segmented_lines = self.segmenter.process_page(page_img)
        
        full_text_lines = []
        structured_lines = []
        total_words_count = 0
        
        # 2. Sequential Word Recognition
        for line in segmented_lines:
            line_words = []
            line_raw_tokens = []
            
            for word_item in line['words']:
                crop = word_item['crop']
                recognized_word = self.recognize_word_crop(crop)
                
                word_info = {
                    'word_idx': word_item['word_idx'],
                    'bbox': word_item['bbox'],
                    'text': recognized_word
                }
                if debug:
                    word_info['crop_size'] = list(crop.size)
                    
                line_words.append(word_info)
                if recognized_word:
                    line_raw_tokens.append(recognized_word)
                total_words_count += 1
                
            # Assemble line with combining-mark safety and NFC normalization
            line_str = self.assemble_line_words(line_raw_tokens)
            if line_str:
                full_text_lines.append(line_str)
                
            line_anomalies = validate_tamil_unicode_sequence(line_str)
            structured_lines.append({
                'line_idx': line['line_idx'],
                'bbox': line['bbox'],
                'line_text': line_str,
                'words': line_words,
                'anomalies': line_anomalies
            })
            
        assembled_full_text = "\n".join(full_text_lines)
        
        # Attach lightweight pre-flight quality assessment
        preflight_assessment = self.segmenter.assess_image_quality(page_img)

        # Aggregate document-level anomalies
        doc_anomalies = []
        for s_line in structured_lines:
            for anom in s_line['anomalies']:
                doc_anomalies.append({
                    'line_idx': s_line['line_idx'],
                    **anom
                })
        
        result = {
            'full_text': assembled_full_text,
            'total_lines': len(structured_lines),
            'total_words': total_words_count,
            'lines': structured_lines,
            'image_size': [page_img.width, page_img.height],
            'preflight': preflight_assessment,
            'anomalies': doc_anomalies
        }
        
        return result
