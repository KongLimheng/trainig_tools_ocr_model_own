"""Unit tests for Khmer Unicode Canonical Normalization & Segmentation."""

import pytest
from khmer_ocr.normalizer import (
    normalize_khmer_text,
    normalize_khmer_canonical,
    clean_duplicate_coeng,
    strip_invisible_chars,
    has_invisible_chars,
    split_into_syllables,
    count_khmer_syllables,
    ZWSP,
)


def test_strip_invisible_chars():
    # Text with invisible Zero-Width Space
    dirty_text = f"សួ{ZWSP}ស្តី{ZWSP}កម្ពុជា"
    assert has_invisible_chars(dirty_text) is True

    clean = strip_invisible_chars(dirty_text)
    assert has_invisible_chars(clean) is False
    assert clean == "សួស្តីកម្ពុជា"


def test_clean_duplicate_coeng():
    # Double coeng typing error: ក + ្ + ្ + ក
    bad_coeng = "ក\u17D2\u17D2ក"
    cleaned = clean_duplicate_coeng(bad_coeng)
    assert cleaned == "ក\u17D2ក"


def test_canonical_reordering():
    # Consonant: ក (0x1780)
    # Coeng Subscript: ្ក (\u17D2\u1780)
    # Dependent Vowel: ា (\u17B6)
    # Diacritic: ់ (\u17CB)
    # Standard: ក + ្ + ក + ា + ់ (ក្កាត់)
    standard = "ក\u17D2កា់"
    normalized = normalize_khmer_canonical(standard)
    assert normalized == standard


def test_syllable_splitting():
    text = "កម្ពុជា"
    syllables = split_into_syllables(text)
    # Expected clusters: [ក, ម្ពុ, ជា]
    assert len(syllables) == 3
    assert count_khmer_syllables(text) == 3


def test_mixed_text_normalization():
    text = f"សាលា A1 {ZWSP} តម្លៃ: ៥០,០០០៛"
    norm = normalize_khmer_text(text, strip_zwsp=True)
    assert ZWSP not in norm
    assert "A1" in norm
    assert "៥០,០០០៛" in norm
