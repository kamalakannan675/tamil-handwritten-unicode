"""
test_model.py - Unit tests for CNN model architecture and inference.
"""

import unittest
import torch
import torch.nn as nn

from model.model import uTHCDNet

class TestModel(unittest.TestCase):
    def test_model_initialization(self):
        """Verify model layer topology and output dimension."""
        model = uTHCDNet(num_classes=156)
        self.assertEqual(model.num_classes, 156)
        self.assertIsInstance(model.conv1, nn.Conv2d)
        self.assertIsInstance(model.fc3, nn.Linear)
        self.assertEqual(model.fc3.out_features, 156)

    def test_forward_pass_shape(self):
        """Verify output logits shape on forward pass."""
        model = uTHCDNet(num_classes=156)
        dummy_input = torch.randn(4, 1, 64, 64)
        logits = model(dummy_input)

        self.assertEqual(logits.shape, (4, 156))

    def test_predict_proba_and_top_k(self):
        """Verify probability range, sum, and top-k output."""
        model = uTHCDNet(num_classes=156)
        dummy_input = torch.randn(2, 1, 64, 64)

        probs = model.predict_proba(dummy_input)
        self.assertEqual(probs.shape, (2, 156))
        # Sum of probabilities across classes should equal 1.0
        row_sums = probs.sum(dim=1).detach().cpu().numpy()
        for s in row_sums:
            self.assertAlmostEqual(s, 1.0, places=4)

        top_idx, top_prob = model.predict_top_k(dummy_input, k=3)
        self.assertEqual(top_idx.shape, (2, 3))
        self.assertEqual(top_prob.shape, (2, 3))

    def test_cpu_loss_backward(self):
        """Verify loss computation and backward pass on CPU."""
        model = uTHCDNet(num_classes=156)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        dummy_input = torch.randn(2, 1, 64, 64)
        dummy_target = torch.tensor([1, 15], dtype=torch.long)

        optimizer.zero_grad()
        logits = model(dummy_input)
        loss = criterion(logits, dummy_target)
        loss.backward()
        optimizer.step()

        self.assertGreater(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
