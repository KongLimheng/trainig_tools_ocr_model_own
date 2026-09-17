"""Unit tests for CRNN model forward pass and decoding."""

import pytest
import torch
from khmer_ocr.models.crnn import KhmerCRNN
from khmer_ocr.vocab.char_map import KhmerCharMap


def test_crnn_forward_and_decode():
    char_map = KhmerCharMap()
    model = KhmerCRNN(num_classes=len(char_map), in_channels=1, backbone_type="resnet34")
    model.eval()

    # Dummy batch: [Batch=2, Channels=1, Height=48, Width=160]
    dummy_input = torch.randn(2, 1, 48, 160)
    with torch.no_grad():
        log_probs = model(dummy_input)

    # Output should be [T, Batch, Num_Classes]
    t_steps, b_size, n_classes = log_probs.shape
    assert b_size == 2
    assert n_classes == len(char_map)
    assert t_steps > 0

    decoded = model.decode_greedy(log_probs)
    assert len(decoded) == 2
