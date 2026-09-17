"""Khmer OCR Post-Processing Package."""

from .word_segmenter import KhmerWordSegmenter, DEFAULT_KHMER_DICTIONARY
from .spell_corrector import KhmerSpellCorrector

__all__ = [
    "KhmerWordSegmenter",
    "KhmerSpellCorrector",
    "DEFAULT_KHMER_DICTIONARY",
]
