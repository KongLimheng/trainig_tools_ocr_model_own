"""Khmer Syllable Cluster Error Rate (SCER).

Evaluates OCR performance on the level of orthographic syllable clusters.
Helps pinpoint dropped subscripts, diacritics, and vowel clusters.
"""

from .cer import levenshtein_distance
from ..normalizer.syllable_parser import split_into_syllables
from ..normalizer import normalize_khmer_text


def calculate_syllable_cer(targets: list[str], predictions: list[str]) -> float:
    """Calculates Error Rate on orthographic syllable clusters."""
    total_dist = 0
    total_len = 0

    for target, pred in zip(targets, predictions):
        norm_t = normalize_khmer_text(target, strip_zwsp=True)
        norm_p = normalize_khmer_text(pred, strip_zwsp=True)

        syl_t = split_into_syllables(norm_t)
        syl_p = split_into_syllables(norm_p)

        dist = levenshtein_distance(syl_t, syl_p)
        total_dist += dist
        total_len += len(syl_t)

    return total_dist / max(1, total_len)
