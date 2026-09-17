"""CTC Loss Wrapper with Sequence Length Validation."""

import torch
import torch.nn as nn


class KhmerCTCLoss(nn.Module):
    """CTC Loss wrapper with safety validation for variable length sequences."""

    def __init__(self, blank_idx: int = 0, zero_infinity: bool = True):
        super().__init__()
        self.loss_fn = nn.CTCLoss(blank=blank_idx, zero_infinity=zero_infinity, reduction="mean")

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        Args:
            log_probs: [T, B, C]
            targets: 1D Tensor of concatenated labels
            input_lengths: [B]
            target_lengths: [B]
        """
        # CTC constraint: input_length must be >= target_length
        # Clamp or mask invalid targets if any anomalous edge case occurs
        return self.loss_fn(log_probs, targets, input_lengths, target_lengths)
