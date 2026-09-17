"""Khmer Normalized Character Error Rate (NCER).

Normalizes both ground-truth and hypothesis before computing Levenshtein distance:
1. Strips invisible zero-width spaces (ZWSP, ZWNJ, BOM)
2. Normalizes compound vowels and deduplicates coeng signs
3. Applies canonical syllable ordering
"""

from .cer import levenshtein_distance
from ..normalizer import normalize_khmer_text


def calculate_normalized_cer(targets: list[str], predictions: list[str]) -> float:
    """Calculates Normalized CER where both targets and predictions are normalized canonically."""
    total_dist = 0
    total_len = 0

    for target, pred in zip(targets, predictions):
        norm_t = normalize_khmer_text(target, strip_zwsp=True)
        norm_p = normalize_khmer_text(pred, strip_zwsp=True)

        dist = levenshtein_distance(norm_t, norm_p)
        total_dist += dist
        total_len += len(norm_t)

    return total_dist / max(1, total_len)
