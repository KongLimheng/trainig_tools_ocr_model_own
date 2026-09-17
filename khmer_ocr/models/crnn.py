"""CRNN (Convolutional Recurrent Neural Network) for Khmer OCR."""

import torch
import torch.nn as nn
from .backbones.resnet import ResNetOCR
from .backbones.mobilenet import MobileNetOCR


class KhmerCRNN(nn.Module):
    """CRNN Architecture combining CNN Backbone + BiLSTM + Linear CTC Head."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        backbone_type: str = "resnet34",
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_type = backbone_type

        # Select backbone
        if backbone_type == "mobilenet":
            self.backbone = MobileNetOCR(in_channels=in_channels)
            feature_dim = self.backbone.out_channels
        else:
            self.backbone = ResNetOCR(in_channels=in_channels, base_channels=32)
            feature_dim = self.backbone.out_channels

        # Sequence modeling with BiLSTM
        self.rnn = nn.LSTM(
            input_size=feature_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Output projection to vocabulary classes (including blank token at index 0)
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        Args:
            x: Tensor of shape [B, C, H, W]
        Returns:
            Log-probabilities of shape [T, B, num_classes] (standard for PyTorch CTCLoss)
        """
        # Feature extraction: [B, C, H, W] -> [B, Feat, 1, W']
        features = self.backbone(x)

        # Squeeze height dimension: [B, Feat, W']
        features = features.squeeze(2)

        # Permute to [B, W', Feat] for RNN
        features = features.permute(0, 2, 1)

        # BiLSTM: [B, W', rnn_hidden * 2]
        rnn_out, _ = self.rnn(features)

        # Projection: [B, W', num_classes]
        logits = self.fc(rnn_out)

        # PyTorch CTCLoss expects input shape: [T, B, num_classes]
        # with log_softmax applied
        log_probs = logits.log_softmax(dim=2).permute(1, 0, 2)
        return log_probs

    def decode_greedy(self, log_probs: torch.Tensor) -> list[list[int]]:
        """Greedy argmax CTC decoding for a batch of predictions.
        Args:
            log_probs: Tensor of shape [T, B, num_classes]
        Returns:
            List of decoded token sequences (one per batch item).
        """
        # [T, B, C] -> [B, T, C]
        probs = log_probs.permute(1, 0, 2)
        argmax = torch.argmax(probs, dim=2)  # [B, T]

        batch_sequences = []
        for seq in argmax.cpu().numpy():
            collapsed = []
            prev_idx = -1
            for idx in seq:
                if idx != prev_idx:
                    if idx != 0:  # 0 is CTC Blank token
                        collapsed.append(int(idx))
                    prev_idx = idx
            batch_sequences.append(collapsed)

        return batch_sequences

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freezes or unfreezes CNN backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def is_backbone_frozen(self) -> bool:
        """Returns True if any backbone parameter has requires_grad=False."""
        return any(not p.requires_grad for p in self.backbone.parameters())
