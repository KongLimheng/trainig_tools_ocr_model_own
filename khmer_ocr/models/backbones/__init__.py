"""Backbones Package for Khmer OCR."""

from .resnet import ResNetOCR
from .mobilenet import MobileNetOCR

__all__ = ["ResNetOCR", "MobileNetOCR"]
