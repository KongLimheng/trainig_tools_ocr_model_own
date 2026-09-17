"""Unit tests for Khmer Wikipedia Data Harvester & Corpus Pipeline."""

from pathlib import Path
from khmer_ocr.dataset.wikipedia_loader import KhmerWikipediaLoader, CURATED_TOPICS


def test_clean_and_segment_extract():
    loader = KhmerWikipediaLoader()
    sample_wikitext = """
    == សេចក្តីផ្តើម ==
    ប្រទេសកម្ពុជា [kɑmˈpuˈciə] មានរាជធានីឈ្មោះភ្នំពេញ។ [1]
    == ប្រវត្តិសាស្ត្រ ==
    ប្រាសាទអង្គរវត្តត្រូវបានសាងសង់ឡើងនៅសតវត្សរ៍ទី១២។ [២] គេចាត់ទុកថាជាអច្ឆរិយវត្ថុមួយក្នុងលោក។ https://km.wikipedia.org
    """
    sentences = loader.clean_and_segment_extract(sample_wikitext)
    assert len(sentences) >= 2

    for s in sentences:
        assert "[1]" not in s
        assert "[២]" not in s
        assert "[kɑmˈpuˈciə]" not in s
        assert "http" not in s
        assert "== " not in s
        # Ensure characters are predominantly Khmer
        khmer_count = sum(1 for c in s if 0x1780 <= ord(c) <= 0x17FF)
        assert khmer_count >= 10


def test_save_corpus(tmp_path: Path):
    loader = KhmerWikipediaLoader()
    sentences = [
        "ព្រះរាជាណាចក្រកម្ពុជា ជាប្រទេសមួយស្ថិតនៅអាស៊ីអាគ្នេយ៍",
        "រាជធានីភ្នំពេញ គឺជាបេះដូងនៃប្រទេសកម្ពុជា",
    ]
    out_file = tmp_path / "wiki_corpus.txt"
    count = loader.save_corpus(sentences, out_file)
    assert count == 2
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 2
    assert lines[0] == sentences[0]


def test_curated_topics_definitions():
    assert "History & Angkor" in CURATED_TOPICS
    assert "Geography & Nature" in CURATED_TOPICS
    assert len(CURATED_TOPICS["History & Angkor"]) > 0
