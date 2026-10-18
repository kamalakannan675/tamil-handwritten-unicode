"""
full_page_ocr.py - End-to-End Tamil Handwritten Full-Page OCR Engine.

Orchestrates:
1. Document loading (from file path, byte stream, or PIL Image).
2. Zero-dependency Page & Word Segmentation (PageSegmenter).
3. Preprocessing word crops to aspect-ratio scaled grayscale tensors (H=64, W proportional).
4. CRNN + CTC batch inference.
5. Natural reading-order Unicode reconstruction (words -> lines -> full text).
6. Bounding box coordinates for Generative UI / visual overlay.
"""

import os
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.data.vocab import TamilCTCCharacterVocabulary
from src.models.crnn import TamilCRNN
from src.inference.ctc_decoder import ctc_greedy_decode
from src.ocr.page_segmentation import PageSegmenter

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
        Applies adaptive contrast stretching so low-contrast mobile camera photos match
        the training distribution (white background ~240-255, dark ink ~0-30).
        """
        gray = pil_crop.convert('L')
        orig_w, orig_h = gray.size
        
        # Adaptive contrast normalization for real-world photos & lighting variations
        arr = np.array(gray, dtype=np.float32)
        p_low, p_high = np.percentile(arr, (2, 98))
        if (p_high - p_low) > 10:
            arr = np.clip((arr - p_low) / (p_high - p_low) * 255.0, 0, 255)
        gray = Image.fromarray(arr.astype(np.uint8))
        
        # Add background-aware whitespace margin (matches dataset training distribution)
        pad_y = max(2, int(orig_h * 0.08))
        pad_x = max(2, int(orig_h * 0.08))
        from PIL import ImageOps
        gray = ImageOps.expand(gray, border=(pad_x, pad_y, pad_x, pad_y), fill=255)
        pad_w, pad_h = gray.size
        
        new_w = max(16, min(self.max_width, int(round(pad_w * self.target_height / max(1, pad_h)))))
        # Pad width to nearest multiple of 4
        if new_w % 4 != 0:
            new_w = ((new_w // 4) + 1) * 4
            
        resized = gray.resize((new_w, self.target_height), Image.Resampling.BILINEAR)
        img_np = (255.0 - np.array(resized, dtype=np.float32)) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0) # (1, 1, 64, W)
        return tensor

    def recognize_word_crop(self, pil_crop: Image.Image) -> str:
        """
        Performs CRNN forward pass and CTC decoding for a single word crop.
        """
        tensor = self.preprocess_word_crop(pil_crop).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor) # (T, 1, num_classes)
            decoded = ctc_greedy_decode(logits, self.vocab, blank_idx=0)
        return decoded[0] if decoded else ""

    def process_document(self, input_source: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Performs full-page layout analysis, line/word segmentation, recognition,
        and natural reading-order Unicode assembly.
        
        Args:
            input_source: File path string or PIL.Image instance.
            
        Returns:
            Dict containing:
            - 'full_text': Formatted multiline Tamil text
            - 'total_lines': Number of segmented lines
            - 'total_words': Number of segmented words
            - 'lines': Detailed structured hierarchy with bounding boxes
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
            line_text_tokens = []
            
            for word_item in line['words']:
                crop = word_item['crop']
                recognized_word = self.recognize_word_crop(crop)
                
                line_words.append({
                    'word_idx': word_item['word_idx'],
                    'bbox': word_item['bbox'],
                    'text': recognized_word
                })
                line_text_tokens.append(recognized_word)
                total_words_count += 1
                
            line_str = " ".join(line_text_tokens)
            full_text_lines.append(line_str)
            
            structured_lines.append({
                'line_idx': line['line_idx'],
                'bbox': line['bbox'],
                'line_text': line_str,
                'words': line_words
            })
            
        assembled_full_text = "\n".join(full_text_lines)
        
        return {
            'full_text': assembled_full_text,
            'total_lines': len(structured_lines),
            'total_words': total_words_count,
            'lines': structured_lines,
            'image_size': [page_img.width, page_img.height]
        }
