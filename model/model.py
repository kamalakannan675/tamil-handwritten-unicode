"""
model.py - CNN Architecture for Tamil Handwritten Character Recognition.

Implements the baseline CNN architecture described in the IEEE Access paper:
"uTHCD: A New Benchmarking for Tamil Handwritten OCR"
Authors: Noushath Shaffi and Faizal Hajamohideen (IEEE Access, Vol. 9, 2021)
Figure 13 & Table 4:
- Input: (1, 64, 64)
- Conv1: 32 filters, 3x3 kernel, valid padding -> ReLU -> MaxPool2d(2, 2)
- Conv2: 64 filters, 3x3 kernel, valid padding -> ReLU -> MaxPool2d(2, 2)
- Flatten: 14 * 14 * 64 = 12,544
- Dense1: 1024 units -> ReLU
- Dense2: 512 units -> ReLU
- Classifier: 156 classes (logits output)

With optimal regularizing dropouts from paper Table 4 & Figure 20:
- Conv1 dropout: 0.10
- Conv2 dropout: 0.05
- FC dropouts: 0.50
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class uTHCDNet(nn.Module):
    """
    uTHCD Baseline CNN model matching Figure 13 of the IEEE paper.
    """
    def __init__(self,
                 num_classes: int = 156,
                 dropout_conv1: float = 0.10,
                 dropout_conv2: float = 0.05,
                 dropout_fc: float = 0.50):
        super(uTHCDNet, self).__init__()

        self.num_classes = num_classes

        # Conv Block 1: Input (B, 1, 64, 64) -> Conv(3x3 valid) -> (B, 32, 62, 62) -> MaxPool(2x2) -> (B, 32, 31, 31)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0)
        self.drop_conv1 = nn.Dropout2d(p=dropout_conv1) if dropout_conv1 > 0 else nn.Identity()

        # Conv Block 2: Input (B, 32, 31, 31) -> Conv(3x3 valid) -> (B, 64, 29, 29) -> MaxPool(2x2) -> (B, 64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0)
        self.drop_conv2 = nn.Dropout2d(p=dropout_conv2) if dropout_conv2 > 0 else nn.Identity()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dense Layers: 64 * 14 * 14 = 12,544
        self.flatten_dim = 64 * 14 * 14
        self.fc1 = nn.Linear(self.flatten_dim, 1024)
        self.drop_fc1 = nn.Dropout(p=dropout_fc)

        self.fc2 = nn.Linear(1024, 512)
        self.drop_fc2 = nn.Dropout(p=dropout_fc)

        # Classification Logits
        self.fc3 = nn.Linear(512, num_classes)

        # Initialize weights with Xavier uniform (Glorot) as specified on page 13 of the paper
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        # Conv Block 1
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop_conv1(x)

        # Conv Block 2
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop_conv2(x)

        # Flatten & FC
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x)

        x = F.relu(self.fc2(x))
        x = self.drop_fc2(x)

        logits = self.fc3(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities for input tensor x."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_top_k(self, x: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns top-k class indices and confidence scores.
        Output: (top_k_indices, top_k_confidences)
        """
        probs = self.predict_proba(x)
        top_probs, top_indices = torch.topk(probs, k=k, dim=1)
        return top_indices, top_probs
