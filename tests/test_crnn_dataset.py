"""
test_crnn_dataset.py - Unit and Integration Tests for Tamil CRNN Pipeline.

Tests:
1. TamilCTCCharacterVocabulary integrity and token round-tripping.
2. TamilCRNN forward pass shape, time-step computation, and gradient flow.
3. pad_collate_fn dynamic width padding (multiples of 4) and CTC sequence lengths.
4. PyTorch CTCLoss compatibility on CRNN outputs.
5. ctc_greedy_decode correctness with blank collapse and deduplication.
6. Levenshtein edit distance, CER, WER, and exact match metrics.
7. Real dataset loader sample loading from raw ZIP (when available).
"""

import os
import sys
# Bootstrap: ensure the project root is on sys.path when run as a plain script
# (e.g. `python tests/test_crnn_dataset.py` from /kaggle/working/tamil-handwritten-unicode)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import torch.nn as nn
from PIL import Image

from src.data.vocab import TamilCTCCharacterVocabulary
from src.data.dataset import TamilWordDataset, pad_collate_fn
from src.models.crnn import TamilCRNN
from src.inference.ctc_decoder import (
    ctc_greedy_decode,
    levenshtein_distance,
    compute_cer,
    compute_wer,
    compute_exact_accuracy
)

class TestTamilCRNNPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a sample vocabulary
        cls.sample_words = ["அம்மா", "அப்பா", "தமிழ்", "வணக்கம்", "பள்ளி"]
        cls.vocab = TamilCTCCharacterVocabulary()
        cls.vocab.build_from_words(cls.sample_words)
        
        # Real dataset zip path
        cls.zip_path = os.path.join("datasets", "raw", "tamil_word_dataset.zip")

    def test_vocab_properties(self):
        """Verify CTC blank at index 0 and bi-directional encoding/decoding."""
        self.assertEqual(self.vocab.blank_idx, 0)
        self.assertEqual(self.vocab.blank_token, "<blank>")
        self.assertEqual(self.vocab.char2idx["<blank>"], 0)
        self.assertEqual(self.vocab.idx2char[0], "<blank>")
        
        test_word = "தமிழ்"
        encoded = self.vocab.encode(test_word)
        self.assertTrue(all(isinstance(idx, int) and idx > 0 for idx in encoded))
        decoded = self.vocab.decode(encoded)
        self.assertEqual(decoded, test_word)

    def test_crnn_forward_pass_and_shape(self):
        """Verify model output shape (T, B, C) where T = W / 4."""
        num_classes = 73
        model = TamilCRNN(num_classes=num_classes, in_channels=1, hidden_size=128)
        model.eval()
        
        # Batch of 2 images with H=64, W=128 -> Expected T = 128 / 4 = 32
        dummy_input = torch.randn(2, 1, 64, 128)
        with torch.no_grad():
            output = model(dummy_input)
            
        self.assertEqual(output.shape, (32, 2, num_classes))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_pad_collate_fn_synthetic(self):
        """Verify pad_collate_fn pads width to multiple of 4 and flattens targets."""
        item1 = {
            'image': torch.ones(1, 64, 45), # W=45 -> padded should be 48
            'text': "அம்மா",
            'target': torch.tensor([1, 2, 3]),
            'target_length': 3,
            'width': 45,
            'img_path': "val/1.jpg"
        }
        item2 = {
            'image': torch.ones(1, 64, 60), # W=60 -> already multiple of 4, max_w=60
            'text': "தமிழ்",
            'target': torch.tensor([4, 5, 6, 7]),
            'target_length': 4,
            'width': 60,
            'img_path': "val/2.jpg"
        }
        batch = [item1, item2]
        collated = pad_collate_fn(batch)
        
        # Max width is 60 (multiple of 4)
        self.assertEqual(collated['images'].shape, (2, 1, 64, 60))
        self.assertEqual(collated['input_lengths'].tolist(), [15, 15]) # 60 // 4 = 15
        self.assertEqual(collated['target_lengths'].tolist(), [3, 4])
        self.assertEqual(collated['targets'].tolist(), [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(collated['texts'], ["அம்மா", "தமிழ்"])

    def test_ctc_loss_backward_step(self):
        """Verify CTCLoss computes valid scalar loss and backpropagates gradients."""
        model = TamilCRNN(num_classes=73, in_channels=1, hidden_size=64)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # Batch: B=2, W=64 -> T = 16
        inputs = torch.randn(2, 1, 64, 64)
        targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
        target_lengths = torch.tensor([2, 3], dtype=torch.long)
        input_lengths = torch.tensor([16, 16], dtype=torch.long)
        
        logits = model(inputs) # (16, 2, 73)
        log_probs = logits.log_softmax(2)
        
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Verify gradients exist for all model weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_ctc_greedy_decoder(self):
        """Verify greedy decoder correctly collapses repetitions and drops blanks."""
        # Assume vocab has: 0: blank, 1: charA, 2: charB
        # Frame tokens: [blank, 1, 1, blank, 2, 2, 2, blank, 1]
        # Should collapse to: [1, 2, 1]
        raw_token_seq = [0, 1, 1, 0, 2, 2, 2, 0, 1]
        T = len(raw_token_seq)
        B = 1
        num_classes = len(self.vocab)
        
        logits = torch.zeros(T, B, num_classes)
        for t, tok in enumerate(raw_token_seq):
            logits[t, 0, tok] = 10.0 # High confidence for this token
            
        decoded = ctc_greedy_decode(logits, self.vocab, blank_idx=0)
        expected_chars = [self.vocab.idx2char[1], self.vocab.idx2char[2], self.vocab.idx2char[1]]
        expected_str = "".join(expected_chars)
        self.assertEqual(decoded, [expected_str])

    def test_evaluation_metrics(self):
        """Verify Levenshtein distance, CER, WER, and exact accuracy."""
        # 1. Edit distance
        self.assertEqual(levenshtein_distance(list("kitten"), list("sitting")), 3)
        self.assertEqual(levenshtein_distance(["அ", "ம்", "மா"], ["அ", "ப்", "பா"]), 2)
        
        # 2. CER
        preds = ["அம்மா", "தமிள்"]
        targets = ["அம்மா", "தமிழ்"]
        # targets total chars: 4 + 4 = 8 (assuming 4 unicode codepoints each)
        # distance: 0 + 1 = 1
        cer = compute_cer(preds, targets)
        self.assertAlmostEqual(cer, 1.0 / (len(list("அம்மா")) + len(list("தமிழ்"))))
        
        # 3. WER & Exact Match
        wer = compute_wer(preds, targets)
        self.assertEqual(wer, 0.5) # 1 incorrect out of 2 words
        acc = compute_exact_accuracy(preds, targets)
        self.assertEqual(acc, 0.5)

    def test_real_dataset_loader(self):
        """Verify loading real samples from tamil_word_dataset.zip if enabled."""
        if os.environ.get("RUN_SLOW_ZIP_TEST") != "1":
            self.skipTest("Skipping slow 5.1GB zip test (set RUN_SLOW_ZIP_TEST=1 to execute).")
            
        if not os.path.exists(self.zip_path):
            self.skipTest(f"Dataset archive {self.zip_path} not found.")
            
        # Initialize validation split with max_samples=3 for fast test
        dataset = TamilWordDataset(
            split='val',
            zip_path=self.zip_path,
            max_samples=3
        )
        self.assertEqual(len(dataset), 3)
        
        item = dataset[0]
        self.assertIn('image', item)
        self.assertIn('text', item)
        self.assertIn('target', item)
        self.assertIn('target_length', item)
        self.assertIn('width', item)
        
        tensor = item['image']
        self.assertEqual(tensor.shape[0], 1)
        self.assertEqual(tensor.shape[1], 64) # Height 64
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)
        self.assertGreater(item['target_length'], 0)
        
        # Test collate on real samples
        batch = [dataset[0], dataset[1]]
        collated = pad_collate_fn(batch)
        self.assertEqual(collated['images'].shape[0], 2)
        self.assertEqual(collated['images'].shape[1], 1)
        self.assertEqual(collated['images'].shape[2], 64)
        self.assertEqual(collated['images'].shape[3] % 4, 0)

if __name__ == '__main__':
    unittest.main()
