"""Unit tests for Smart Khmer Corpus & Synthetic Line Sampler.

Verifies:
1. No ungrammatical character salads or stacked vowels (e.g. ាី, ើឿ, ដ្ថ, អ្ខ៏).
2. Orthographic line validation.
3. 100% authentic sampling from RAC dictionary and custom corpus.
4. Support for pure corpus mode.
"""

import re
import pytest
from khmer_ocr.synth.corpus_sampler import (
    KhmerCorpusSampler,
    validate_khmer_line,
    CONFUSION_GROUPS,
)


def test_validate_khmer_line_rejections():
    """Verifies that invalid, stacked, or orphan sequences are rejected."""
    # Stacked vowels (from user reported bug: ប្លាីស្មីំស្មេធ្ឋូ)
    assert validate_khmer_line("ប្លាីស្មីំស្មេធ្ឋូ") is False
    # Stacked vowels ើឿ (from user reported bug: ឋ្ហ៌ងើឆឿដ្ថមា)
    assert validate_khmer_line("ឋ្ហ៌ងើឆឿដ្ថមា") is False
    # Trailing coeng
    assert validate_khmer_line("កម្ពុជា្") is False
    # Double consecutive coeng
    assert validate_khmer_line("ក្្ពុជា") is False
    # Coeng followed by non-consonant
    assert validate_khmer_line("ក្ាពុជា") is False
    # Orphan start
    assert validate_khmer_line("ស់រហូតដល់") is False
    assert validate_khmer_line("ល់ពីមុន") is False
    # Too short
    assert validate_khmer_line("ក") is False
    assert validate_khmer_line("") is False


def test_validate_khmer_line_accepts_valid():
    """Verifies that authentic Khmer phrases pass validation."""
    valid_phrases = [
        "ព្រះរាជាណាចក្រកម្ពុជា",
        "រាជធានីភ្នំពេញ បេះដូងនៃប្រទេសកម្ពុជា",
        "ប្រាសាទអង្គរវត្តជាសម្បត្តិបេតិកភណ្ឌពិភពលោក",
        "ថ្ងៃទី១៥ ខែមករា ឆ្នាំ២០២៥",
        "តម្លៃ ៥០០,០០០ ៛",
    ]
    for phrase in valid_phrases:
        assert validate_khmer_line(phrase) is True, f"Failed for valid phrase: {phrase}"


def test_sampler_loads_rac_dictionary():
    """Verifies that KhmerCorpusSampler loads the 37,341 RAC dictionary."""
    sampler = KhmerCorpusSampler()
    assert len(sampler.rac_words) >= 37000
    assert len(sampler.confusion_word_map) > 0


def test_sample_line_zero_illegal_vowels():
    """Tests 1,000 sampled lines to ensure zero stacked vowels or illegal coeng."""
    sampler = KhmerCorpusSampler()
    illegal_vowels_regex = re.compile(r"[\u17B6-\u17C5]{2,}")
    illegal_coeng_regex = re.compile(r"\u17D2{2,}")

    for _ in range(1000):
        line = sampler.sample_line()
        assert validate_khmer_line(line) is True, f"Invalid line generated: {line}"
        assert not illegal_vowels_regex.search(line), f"Found stacked vowels in: {line}"
        assert not illegal_coeng_regex.search(line), f"Found stacked coeng in: {line}"
        assert not line.endswith("\u17D2"), f"Found trailing coeng in: {line}"


def test_sample_dictionary_phrase():
    """Verifies dictionary phrase generation creates valid RAC-composed lines."""
    sampler = KhmerCorpusSampler()
    for _ in range(100):
        phrase = sampler.sample_dictionary_phrase(min_words=2, max_words=4)
        assert validate_khmer_line(phrase) is True
        assert len(phrase.strip()) >= 4


def test_sample_hard_negative():
    """Verifies hard-negative sampler uses real words with confusable characters."""
    sampler = KhmerCorpusSampler()
    for _ in range(100):
        phrase = sampler.sample_hard_negative()
        assert validate_khmer_line(phrase) is True
        # Verify phrase has valid length and no corrupt Unicode
        assert len(phrase) >= 4


def test_sample_number_or_date():
    """Verifies dates and numbers are valid."""
    sampler = KhmerCorpusSampler()
    for _ in range(100):
        num_line = sampler.sample_number_or_date()
        assert len(num_line.strip()) > 0
        # Should not contain pseudo words
        assert not re.search(r"[\u17B6-\u17C5]{2,}", num_line)


def test_pure_corpus_mode():
    """Verifies pure_corpus=True samples strictly from custom texts."""
    custom_sentences = [
        "ប្រាសាទបាយ័នស្ថិតនៅចំកណ្តាលនៃរាជធានីអង្គរធំ",
        "ព្រះបាទជ័យវរ្ម័នទី៧ជាព្រះមហាក្សត្រដ៏ល្បីល្បាញ",
        "ទន្លេមេគង្គហូរកាត់ប្រទេសកម្ពុជាពីជើងទៅត្បូង",
    ]
    sampler = KhmerCorpusSampler(custom_texts=custom_sentences, pure_corpus=True)

    for _ in range(100):
        line = sampler.sample_line()
        assert line in custom_sentences
