"""Unit tests for Khmer-specific OCR metrics."""

import pytest
from khmer_ocr.metrics import (
    levenshtein_distance,
    calculate_cer,
    calculate_normalized_cer,
    calculate_syllable_cer,
    calculate_exact_match,
    KhmerConfusionTracker,
)
from khmer_ocr.normalizer import ZWSP


def test_levenshtein_distance():
    assert levenshtein_distance("កម្ពុជា", "កម្ពុជា") == 0
    assert levenshtein_distance("កម្ពុជា", "កម្ពុជ") == 1
    assert levenshtein_distance("abc", "abd") == 1


def test_normalized_cer_vs_standard_cer():
    # If target has ZWSP and prediction does not, raw CER counts errors, but Normalized CER is 0.0!
    target = f"សួ{ZWSP}ស្តី"
    pred = "សួស្តី"

    raw_cer = calculate_cer([target], [pred])
    assert raw_cer > 0.0  # Raw CER fails because of invisible ZWSP

    norm_cer = calculate_normalized_cer([target], [pred])
    assert norm_cer == 0.0  # Normalized CER correctly ignores invisible ZWSP!


def test_syllable_cer():
    # Target: [ក, ម្ពុ, ជា] (3 syllables)
    # Pred:   [ក, ម្ពុ]     (1 deletion)
    target = "កម្ពុជា"
    pred = "កម្ពុ"

    scer = calculate_syllable_cer([target], [pred])
    assert pytest.approx(scer, 0.01) == 1 / 3


def test_confusion_tracker():
    tracker = KhmerConfusionTracker()
    # Simulate a typical OCR confusion: គ misidentified as ត
    tracker.update(target="គណៈកម្មការ", prediction="តណៈកម្មការ")
    top = tracker.top_confusions(5)
    assert len(top) > 0
    assert top[0]["ground_truth"] == "គ"
    assert top[0]["predicted"] == "ត"
    assert top[0]["count"] == 1
