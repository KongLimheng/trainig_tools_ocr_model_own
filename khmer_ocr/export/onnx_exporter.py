"""ONNX Dynamic-Width Exporter & Runtime Validator for Khmer OCR Models."""

import json
from pathlib import Path
import torch
import torch.nn as nn
from ..models.crnn import KhmerCRNN
from ..vocab.char_map import KhmerCharMap


class ONNXWrapper(nn.Module):
    """Wraps CRNN to output raw softmax probabilities or argmax for standard ONNX runtimes."""

    def __init__(self, model: KhmerCRNN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward through CRNN: returns log_probs of shape [T, B, C]
        log_probs = self.model(x)
        # Convert to [B, T, C]
        probs = log_probs.permute(1, 0, 2).exp()
        return probs


def export_to_onnx(
    checkpoint_path: str | Path,
    output_onnx_path: str | Path,
    img_height: int = 48,
    dummy_width: int = 256,
    opset_version: int = 14,
) -> Path:
    """Exports a trained PyTorch model checkpoint to ONNX with dynamic image width."""
    checkpoint_path = Path(checkpoint_path)
    output_onnx_path = Path(output_onnx_path)
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    char_list = ckpt.get("char_list", [])
    backbone = ckpt.get("backbone", "resnet34")

    char_map = KhmerCharMap(characters=char_list, include_specials=False)
    num_classes = len(char_map)

    # Initialize model
    model = KhmerCRNN(num_classes=num_classes, backbone_type=backbone)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    wrapper = ONNXWrapper(model)
    wrapper.eval()

    # Create dummy input: [Batch=1, Channels=1, Height, Width]
    dummy_input = torch.randn(1, 1, img_height, dummy_width, requires_grad=False)

    dynamic_axes = {
        "input": {0: "batch_size", 3: "width"},
        "output": {0: "batch_size", 1: "sequence_length"},
    }

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )

    # Also save metadata JSON alongside ONNX
    meta_path = output_onnx_path.with_suffix(".json")
    metadata = {
        "format": "KhmerOCR-ONNX",
        "img_height": img_height,
        "backbone": backbone,
        "num_classes": num_classes,
        "characters": char_map.char_list,
        "blank_idx": char_map.blank_idx,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return output_onnx_path
