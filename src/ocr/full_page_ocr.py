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
from typing import List, Dict, Any, Union, Tuple, Optional
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

# Whitelist of legitimate single-syllable / short independent Tamil words
# Used as a safeguard to ensure valid separate words (e.g. 'நீ தான்') are NEVER fused.
INDEPENDENT_SHORT_WORDS = {
    'நீ', 'தீ', 'வா', 'போ', 'கை', 'பூ', 'தை', 'மை', 'நா', 'தா',
    'என்', 'உன்', 'தன்', 'முன்', 'பின்', 'நாள்', 'பால்', 'தான்', 'ஆம்', 'ஒரு', 'அது', 'ஓடி'
}

def is_word_refusion_enabled(explicit_setting: Optional[bool] = None) -> bool:
    """
    Resolves the feature flag for word re-fusion:
    - If explicit_setting is provided (not None), returns its boolean value.
    - Otherwise, checks environment variable ENABLE_WORD_REFUSION.
    - Default is strictly False: only affirmative values ('true', '1', 'yes') enable the feature.
    - Missing, empty, or unrecognized values safely resolve to False.
    """
    if explicit_setting is not None:
        return bool(explicit_setting)
    val = os.environ.get('ENABLE_WORD_REFUSION', '').strip().lower()
    return val in {'true', '1', 'yes'}

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

    def refuse_adjacent_word_segments(self,
                                      page_img: Image.Image,
                                      line_bbox: List[int],
                                      base_words: List[Dict[str, Any]],
                                      max_gap_ratio: float = 0.15,
                                      max_gap_px: int = 8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Experimental prototype: Evaluates adjacent word crops separated by an intra-word
        pen lift and re-fuses them if geometric, orthographic, and acoustic evidence
        demonstrates an over-segmentation.

        Decision Rules:
        1. Geometric proximity: gap <= max_gap_px and gap / line_h <= max_gap_ratio (or gap <= 4px).
        2. Non-independence: Never fuses if both tokens are legitimate independent words in INDEPENDENT_SHORT_WORDS.
        3. Fragment trigger: Initiated if W1 is a bare consonant (e.g. 'ப'), narrow fragment, or gap <= 3px.
        4. Acoustic verification: Evaluates combined crop CRNN inference.
        5. Acceptance: Accepted only if combined prediction resolves the fragment without Unicode anomalies.
        """
        line_h = max(1, line_bbox[3])
        i = 0
        new_words = []
        fusion_events = []

        while i < len(base_words):
            if i == len(base_words) - 1:
                new_words.append(base_words[i])
                break

            w1 = base_words[i]
            w2 = base_words[i + 1]

            x1_right = w1['bbox'][0] + w1['bbox'][2]
            x2_left = w2['bbox'][0]
            gap_px = x2_left - x1_right
            gap_ratio = gap_px / line_h

            is_close = (gap_px <= max_gap_px and gap_ratio <= max_gap_ratio) or (gap_px <= 4)
            t1 = w1['text']
            t2 = w2['text']

            both_independent = (t1 in INDEPENDENT_SHORT_WORDS and t2 in INDEPENDENT_SHORT_WORDS)
            is_bare_consonant = (len(t1) == 1 and ord(t1[0]) in TAMIL_CONSONANTS)
            is_narrow = (w1['bbox'][2] < 0.45 * line_h)

            should_test_refusion = is_close and not both_independent and (is_bare_consonant or is_narrow or gap_px <= 3)

            if should_test_refusion:
                ux1 = min(w1['bbox'][0], w2['bbox'][0])
                uy1 = min(w1['bbox'][1], w2['bbox'][1])
                ux2 = max(w1['bbox'][0] + w1['bbox'][2], w2['bbox'][0] + w2['bbox'][2])
                uy2 = max(w1['bbox'][1] + w1['bbox'][3], w2['bbox'][1] + w2['bbox'][3])

                try:
                    comb_crop = page_img.crop((ux1, uy1, ux2, uy2))
                    comb_pred = self.recognize_word_crop(comb_crop)
                except Exception as e:
                    # Defensive fallback: union crop failure must never terminate entire OCR
                    fusion_events.append({
                        'pair': (t1, t2),
                        'gap_px': gap_px,
                        'gap_ratio': round(gap_ratio, 3),
                        'combined_pred': '',
                        'accepted': False,
                        'reason': f"Exception during union crop inference: {type(e).__name__}"
                    })
                    new_words.append(w1)
                    i += 1
                    continue

                has_anomaly = bool(validate_tamil_unicode_sequence(comb_pred))
                is_valid_expansion = (
                    len(comb_pred) >= max(len(t1), len(t2)) and
                    not has_anomaly and
                    comb_pred not in (t1, t2)
                )

                accept = False
                reason = ""
                if is_bare_consonant and comb_pred.startswith(t1) and not has_anomaly:
                    accept = True
                    reason = f"Bare consonant '{t1}' fused into '{comb_pred}'"
                elif is_close and is_valid_expansion and (t1 not in INDEPENDENT_SHORT_WORDS or t2 not in INDEPENDENT_SHORT_WORDS):
                    accept = True
                    reason = f"Narrow fragment '{t1}' + '{t2}' fused into '{comb_pred}'"
                else:
                    accept = False
                    reason = f"Rejected: is_valid_expansion={is_valid_expansion}, anomaly={has_anomaly}"

                fusion_events.append({
                    'pair': (t1, t2),
                    'gap_px': gap_px,
                    'gap_ratio': round(gap_ratio, 3),
                    'combined_pred': comb_pred,
                    'accepted': accept,
                    'reason': reason
                })

                if accept:
                    new_w = {
                        'word_idx': len(new_words),
                        'bbox': [ux1, uy1, ux2 - ux1, uy2 - uy1],
                        'text': comb_pred,
                        'crop': comb_crop
                    }
                    new_words.append(new_w)
                    i += 2
                    continue

            new_words.append(w1)
            i += 1

        for idx, w in enumerate(new_words):
            w['word_idx'] = idx

        return new_words, fusion_events

    def process_document(self, input_source: Union[str, Image.Image], debug: bool = False, enable_word_refusion: Optional[bool] = None) -> Dict[str, Any]:
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
                
            # Experimental Word Re-Fusion Pass (Disabled by default unless explicitly enabled)
            refusion_active = is_word_refusion_enabled(enable_word_refusion)
            line_fusion_events = []
            if refusion_active and len(line_words) > 1:
                line_words, line_fusion_events = self.refuse_adjacent_word_segments(
                    page_img=page_img,
                    line_bbox=line['bbox'],
                    base_words=line_words
                )
                line_raw_tokens = [w['text'] for w in line_words if w.get('text')]

            # Assemble line with combining-mark safety and NFC normalization
            line_str = self.assemble_line_words(line_raw_tokens)
            if line_str:
                full_text_lines.append(line_str)
                
            line_anomalies = validate_tamil_unicode_sequence(line_str)
            line_data = {
                'line_idx': line['line_idx'],
                'bbox': line['bbox'],
                'line_text': line_str,
                'words': line_words,
                'anomalies': line_anomalies
            }
            if line_fusion_events:
                line_data['fusion_events'] = line_fusion_events
            structured_lines.append(line_data)
            
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
            'total_words': sum(len(sl['words']) for sl in structured_lines),
            'lines': structured_lines,
            'image_size': [page_img.width, page_img.height],
            'preflight': preflight_assessment,
            'anomalies': doc_anomalies
        }
        
        return result
