"""
crnn.py - Convolutional Recurrent Neural Network with CTC for Tamil Word OCR.

Architecture:
Input Image: (B, 1, 64, W)
  -> 4-stage VGG-style CNN backbone with MaxPool (downsamples H: 64->1, W: W->W/4)
  -> Linear Projection to BiLSTM hidden size
  -> 2-layer Bidirectional LSTM (hidden_size=256, dropout=0.2)
  -> Linear Classifier (num_classes=73, including CTC blank at index 0)
Output Logits: (T, B, 73) ready for torch.nn.CTCLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TamilCRNN(nn.Module):
    def __init__(self, num_classes: int = 73, in_channels: int = 1, hidden_size: int = 256):
        super(TamilCRNN, self).__init__()
        
        # 1. Convolutional Feature Extractor
        # Downsamples height from 64 to 1 while preserving horizontal time-steps (W/4)
        self.cnn = nn.Sequential(
            # Block 1: 64xW -> 32x(W/2)
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32x(W/2) -> 16x(W/4)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 16x(W/4) -> 4x(W/4)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),

            # Block 4: 4x(W/4) -> 1x(W/4)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1))
        )
        
        # 2. Sequential Projection & Recurrent Sequence Modeling
        self.proj = nn.Linear(512, hidden_size)
        
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # 3. CTC Linear Classification Head
        # Bidirectional produces 2 * hidden_size outputs per time-step
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, 1, 64, W)
        Returns:
            Logits of shape (T, B, num_classes) where T = W / 4
        """
        # Feature maps: (B, 512, 1, W/4)
        features = self.cnn(x)
        features = features.squeeze(2) # (B, 512, W/4)
        features = features.permute(0, 2, 1) # (B, W/4, 512)
        
        projected = self.proj(features) # (B, W/4, hidden_size)
        recurrent_out, _ = self.rnn(projected) # (B, W/4, hidden_size * 2)
        logits = self.fc(recurrent_out) # (B, W/4, num_classes)
        
        # CTC loss requires (T, B, C)
        logits = logits.permute(1, 0, 2)
        return logits
