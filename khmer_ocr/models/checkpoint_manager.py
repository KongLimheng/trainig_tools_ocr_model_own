"""Checkpoint Inspection and Vocabulary Adaptation Manager for Khmer OCR Models."""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from .crnn import KhmerCRNN
from ..vocab.char_map import KhmerCharMap, DEFAULT_KHMER_CHARACTERS


def inspect_checkpoint(checkpoint_path: str | Path) -> Dict[str, Any]:
    """Inspects a model checkpoint and returns metadata without GPU allocation.

    Args:
        checkpoint_path: Path to .pth checkpoint file.

    Returns:
        Dictionary containing epoch, backbone, char_count, loss, val_metrics, and status.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load on CPU with safe settings
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    char_list = ckpt.get("char_list", DEFAULT_KHMER_CHARACTERS)
    epoch = ckpt.get("epoch", 0)
    backbone = ckpt.get("backbone", "resnet34")
    train_loss = ckpt.get("train_loss", None)
    val_metrics = ckpt.get("val_metrics", {})
    has_optimizer = "optimizer_state_dict" in ckpt

    return {
        "path": str(ckpt_path),
        "epoch": epoch,
        "backbone": backbone,
        "char_count": len(char_list),
        "char_list": char_list,
        "train_loss": train_loss,
        "val_metrics": val_metrics,
        "norm_cer": val_metrics.get("norm_cer", None),
        "has_optimizer": has_optimizer,
    }


def load_and_adapt_checkpoint(
    checkpoint_path: str | Path,
    target_char_map: Optional[KhmerCharMap] = None,
    target_backbone: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> Tuple[KhmerCRNN, Dict[str, Any]]:
    """Loads a checkpoint and adapts weights to the target character map.

    Handles vocabulary changes smoothly:
    - Shared characters: Copies learned weights and biases directly.
    - New characters: Initializes with Gaussian weights and zero bias.
    - Preserves 100% of convolutional and recurrent feature extraction layers.

    Args:
        checkpoint_path: Path to checkpoint file.
        target_char_map: Target KhmerCharMap (if None, creates one from checkpoint char_list).
        target_backbone: Target backbone ('resnet34' or 'mobilenet', defaults to checkpoint's).
        device: Torch device to place the adapted model on.

    Returns:
        Tuple of (adapted_model, adaptation_metadata)
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_char_list = ckpt.get("char_list", DEFAULT_KHMER_CHARACTERS)
    ckpt_backbone = ckpt.get("backbone", "resnet34")
    backbone_type = target_backbone or ckpt_backbone

    if target_char_map is None:
        target_char_map = KhmerCharMap(characters=ckpt_char_list)

    model = KhmerCRNN(
        num_classes=len(target_char_map),
        backbone_type=backbone_type,
    )

    state_dict = ckpt["model_state_dict"]

    # Check for identical vocabulary and architecture
    if ckpt_char_list == target_char_map.char_list and ckpt_backbone == backbone_type:
        model.load_state_dict(state_dict)
        adaptation_info = {
            "mode": "exact_match",
            "transferred_chars": len(target_char_map),
            "new_chars": 0,
            "new_char_list": [],
            "epoch": ckpt.get("epoch", 0),
            "raw_checkpoint": ckpt,
        }
        return model.to(device), adaptation_info

    # 1. Adapt Backbone and RNN layers (shapes are identical if same backbone)
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for k, v in state_dict.items():
        if k.startswith("fc."):
            continue  # Linear head handled separately
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k].copy_(v)
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    # 2. Adapt Linear Classification Head (fc.weight and fc.bias)
    old_weight = state_dict.get("fc.weight")  # Shape: [old_num_classes, rnn_hidden*2]
    old_bias = state_dict.get("fc.bias")      # Shape: [old_num_classes]

    transferred = 0
    new_chars = []

    if old_weight is not None:
        new_weight = model_state["fc.weight"].clone()
        new_bias = model_state["fc.bias"].clone()

        # Map character to old index
        old_char_to_idx = {char: idx for idx, char in enumerate(ckpt_char_list)}

        for char, new_idx in target_char_map.char_to_idx.items():
            if char in old_char_to_idx:
                old_idx = old_char_to_idx[char]
                if old_idx < old_weight.size(0):
                    new_weight[new_idx].copy_(old_weight[old_idx])
                    if old_bias is not None and old_idx < old_bias.size(0):
                        new_bias[new_idx].copy_(old_bias[old_idx])
                    transferred += 1
            else:
                new_chars.append(char)
                # Healthy initialization for brand new glyphs:
                # Small Gaussian perturbation around zero
                nn.init.normal_(new_weight[new_idx], mean=0.0, std=0.02)
                if new_bias is not None:
                    new_bias[new_idx].zero_()

        model_state["fc.weight"].copy_(new_weight)
        if old_bias is not None:
            model_state["fc.bias"].copy_(new_bias)

    model.load_state_dict(model_state)

    adaptation_info = {
        "mode": "adapted",
        "transferred_chars": transferred,
        "new_chars": len(new_chars),
        "new_char_list": new_chars,
        "epoch": ckpt.get("epoch", 0),
        "loaded_keys_count": len(loaded_keys),
        "skipped_keys": skipped_keys,
        "raw_checkpoint": ckpt,
    }

    return model.to(device), adaptation_info
