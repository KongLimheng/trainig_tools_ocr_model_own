"""Unit tests for word-aware chunker, dictionary segmentation, and font anti-tofu safeguards."""

from pathlib import Path
import pytest
from khmer_ocr.synth.fonts import is_khmer_font, KhmerFontManager
from khmer_ocr.postprocess.word_segmenter import (
    KhmerWordSegmenter,
    get_khmer_word_segmenter,
    chunk_text_by_words,
)


def test_is_khmer_font_validation():
    """Verifies that is_khmer_font validates fonts and rejects non-Khmer fonts."""
    # Find a real Khmer font in fonts/
    fonts_dir = Path("fonts")
    if fonts_dir.exists():
        khmer_ttfs = list(fonts_dir.glob("*.ttf"))
        if khmer_ttfs:
            assert is_khmer_font(khmer_ttfs[0], min_consonants=30) is True

    # Test system Thai font rejection (e.g. NotoLoopedThai)
    thai_font = Path("/usr/share/fonts/truetype/noto/NotoLoopedThai-Regular.ttf")
    if thai_font.exists():
        assert is_khmer_font(thai_font, min_consonants=30) is False


def test_word_segmenter_rac_dictionary_loaded():
    """Verifies that the 37,341 RAC dictionary is automatically loaded."""
    seg = get_khmer_word_segmenter()
    assert seg.words_count >= 37000
    # Check key words in RAC dictionary are segmented into valid words without breaking
    assert seg.segment("ទេសចរណ៍") == ["ទេសចរណ៍"]
    assert seg.segment("ចាស់") == ["ចាស់"]
    assert seg.segment("រស់") == ["រស់"]
    assert seg.segment("របស់") == ["របស់"]
    assert seg.segment("សិវលិង្គ") == ["សិវលិង្គ"]
    # Compound word like ទេវបដិមា decomposes cleanly into ទេវ and បដិមា
    assert seg.segment("ទេវបដិមា") == ["ទេវ", "បដិមា"]


def test_word_segmenter_no_orphan_clusters():
    """Verifies DP segmenter avoids orphan clusters like ស់ in compound contexts."""
    seg = get_khmer_word_segmenter()

    # 'មករស់' should segment into ['មក', 'រស់'], never leaving orphan 'ស់'
    tokens = seg.segment("មករស់")
    assert tokens == ["មក", "រស់"]
    assert "ស់" not in tokens

    # 'គោលរចរបស់' should segment with 'របស់' intact
    tokens = seg.segment("គោលរចរបស់")
    assert "របស់" in tokens
    assert "ស់" not in tokens

    # 'ដល់ចាស់រហូតដល់មន្ត្រីធំៗ'
    tokens = seg.segment("ដល់ចាស់រហូតដល់មន្ត្រីធំៗ នេះមិនទាន់គិតទៅលើទេសចរណ៍")
    assert "ចាស់" in tokens
    assert "ទេសចរណ៍" in tokens
    assert "ស់" not in tokens


def test_chunk_text_by_words_preserves_boundaries():
    """Verifies chunk_text_by_words breaks strictly on whole-word boundaries."""
    text = (
        "រាប់រយលាននាក់ ហើយគេលេងគ្រប់វ័យពីក្មេងដល់ចាស់"
        "រហូតដល់មន្ត្រីធំៗ នេះមិនទាន់គិតទៅលើទេសចរណ៍"
        "បរទេសចំរុះជាតិសាសន៍ដែលគេមកលេងផង"
    )
    chunks = chunk_text_by_words(text, min_chars=24, max_chars=45)
    assert len(chunks) >= 2

    # None of the chunks should start or end with an orphan cluster
    for c in chunks:
        assert not c.startswith("ស់")
        assert not c.startswith("ល់")
        assert not c.endswith("ចា")
        assert not c.endswith("ទេសចរ")  # Should be ទេសចរណ៍

    # Check that 'ចាស់' and 'ទេសចរណ៍' are preserved whole
    full_rejoined = "".join(chunks)
    assert "ចាស់" in full_rejoined
    assert "ទេសចរណ៍" in full_rejoined


def test_chunk_text_by_words_empty_or_short():
    """Verifies handling of empty or already short strings."""
    assert chunk_text_by_words("") == []
    assert chunk_text_by_words("   ") == []

    short_text = "ព្រះរាជាណាចក្រកម្ពុជា"
    res = chunk_text_by_words(short_text, min_chars=24, max_chars=45)
    assert len(res) == 1
    assert res[0] == short_text
