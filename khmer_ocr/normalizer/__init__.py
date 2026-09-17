"""Khmer Normalization & Text Processing Package."""

from .unicode_rules import (
    normalize_khmer_canonical,
    clean_duplicate_coeng,
    normalize_compound_vowels,
    is_khmer_char,
    KHMER_CONSONANTS,
    KHMER_INDEP_VOWELS,
    KHMER_DEP_VOWELS,
    KHMER_COENG,
    KHMER_SHIFTERS,
    KHMER_DIACRITICS,
    KHMER_DIGITS,
    KHMER_SYMBOLS,
)
from .zwsp_cleaner import (
    strip_invisible_chars,
    has_invisible_chars,
    count_invisible_chars,
    ZWSP,
    ZWNJ,
)
from .syllable_parser import (
    split_into_syllables,
    count_khmer_syllables,
)


def normalize_khmer_text(text: str, strip_zwsp: bool = True) -> str:
    """Convenience master normalizer:
    1. Removes invisible zero-width spaces (if strip_zwsp is True)
    2. Enforces canonical Unicode ordering
    """
    if strip_zwsp:
        text = strip_invisible_chars(text)
    return normalize_khmer_canonical(text)


__all__ = [
    "normalize_khmer_text",
    "normalize_khmer_canonical",
    "clean_duplicate_coeng",
    "normalize_compound_vowels",
    "strip_invisible_chars",
    "has_invisible_chars",
    "count_invisible_chars",
    "split_into_syllables",
    "count_khmer_syllables",
    "is_khmer_char",
    "ZWSP",
    "ZWNJ",
]
