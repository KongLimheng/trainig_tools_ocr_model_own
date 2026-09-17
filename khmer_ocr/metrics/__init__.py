"""Khmer OCR Evaluation Metrics Package."""

from .cer import levenshtein_distance, calculate_cer, calculate_exact_match
from .normalized_cer import calculate_normalized_cer
from .syllable_cer import calculate_syllable_cer
from .confusion import KhmerConfusionTracker, align_sequences

__all__ = [
    "levenshtein_distance",
    "calculate_cer",
    "calculate_exact_match",
    "calculate_normalized_cer",
    "calculate_syllable_cer",
    "KhmerConfusionTracker",
    "align_sequences",
]
