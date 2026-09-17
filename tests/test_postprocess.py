"""Unit tests for Khmer Word Segmentation and Spell Correction."""

import pytest
from khmer_ocr.postprocess.word_segmenter import KhmerWordSegmenter
from khmer_ocr.postprocess.spell_corrector import KhmerSpellCorrector


def test_word_segmenter_basic():
    segmenter = KhmerWordSegmenter()
    # Test continuous nation slogan
    text = "ព្រះរាជាណាចក្រកម្ពុជា"
    words = segmenter.segment(text)
    assert len(words) >= 2
    assert "ព្រះរាជាណាចក្រ" in words
    assert "កម្ពុជា" in words

    segmented_str = segmenter.segment_to_string(text, delimiter=" ")
    assert segmented_str == "ព្រះរាជាណាចក្រ កម្ពុជា"


def test_word_segmenter_sentence():
    segmenter = KhmerWordSegmenter()
    sentence = "រាជធានីភ្នំពេញជាបេះដូងនៃព្រះរាជាណាចក្រកម្ពុជា"
    words = segmenter.segment(sentence)
    assert "រាជធានី" in words
    assert "ភ្នំពេញ" in words
    assert "កម្ពុជា" in words


def test_word_segmenter_fallback():
    segmenter = KhmerWordSegmenter()
    # An unknown word or syllable cluster should still be preserved
    text = "កខគ"
    words = segmenter.segment(text)
    assert "".join(words) == "កខគ"


def test_spell_corrector():
    corrector = KhmerSpellCorrector()
    # "គម្ពុជា" is an OCR error (គ confused with ក or vice versa; in this case គ vs ត or common pairs)
    # Let's test a known confusion pair: គ vs ត -> "ព្រះរាជាណាចក្រគម្ពុជា" or "សន្តិភាព"
    # In COMMON_CONFUSION_PAIRS: ("ត", "គ") and ("គ", "ត")
    # If text is "សន្គិភាព" (with គ instead of ត) -> should correct to "សន្តិភាព"
    # Or "ខេគ្ត" (with គ instead of ត) -> "ខេត្ត"
    
    # Add "ខេត្ត" to vocab if needed, but it's already in DEFAULT_KHMER_DICTIONARY
    corrected, changed = corrector.correct_word("ខេគ្ត")
    assert changed is True
    assert corrected == "ខេត្ត"

    # Test sentence correction
    sentence = "ខ្ញុំរស់នៅ ខេគ្ត សៀមរាប"
    corr_sent, count = corrector.correct_sentence(sentence)
    assert "ខេត្ត" in corr_sent
    assert count >= 1
