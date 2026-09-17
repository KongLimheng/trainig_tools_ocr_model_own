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


def test_foreign_language_and_script_stripping():
    loader = KhmerWikipediaLoader()
    wikitext = """
    រានទេវតា ( ចិន: 土地神屋 អង់គ្លេស: Spirit houses ថៃ: ศាលพระภูมิ ) គឺជា កន្លែងដាក់របស់របរថ្វាយទេវតាឬវិញ្ញាណរបស់សាសនាខ្មោចនិងសាសនាស្រុកក្នុងអាស៊ីអាគ្នេយ៍។
    ប្រទេសចិន (ចិនសម័យ៖ 中国; ចិនបុរាណ៖ 中國; ភិងអ៊ិង៖ Zhōngguó) ដោយមានឈ្មោះជាផ្លូវការថា សាធារណរដ្ឋប្រជាមានិតចិន គឺជាប្រទេសមួយស្ថិតនៅភូមិភាគអាស៊ីបូព៌ា។
    កុំព្យូទ័រ ដែលក្លាយមកពី computer ។
    """
    # Pure Khmer mode (default)
    sentences = loader.clean_and_segment_extract(wikitext, pure_khmer=True, allow_latin=False)
    assert len(sentences) >= 2

    for s in sentences:
        # Must not contain any Latin letters
        assert not any("a" <= c.lower() <= "z" for c in s), f"Found Latin in: {s}"
        # Must not contain any CJK/Chinese characters
        assert not any(0x4E00 <= ord(c) <= 0x9FFF for c in s), f"Found Chinese in: {s}"
        # Must not contain any Thai characters
        assert not any(0x0E00 <= ord(c) <= 0x0E7F for c in s), f"Found Thai in: {s}"

    # Verify that the sentence containing lone English 'computer' was filtered out
    assert not any("computer" in s for s in sentences)
    # Verify that 'រានទេវតា', 'សាធារណរដ្ឋ', and 'ប្រជាមានិតចិន' remain intact
    full_text = " ".join(sentences)
    assert "រានទេវតា" in full_text
    assert "សាធារណរដ្ឋ" in full_text
    assert "ប្រជាមានិតចិន" in full_text


def test_ignored_namespaces_exclusion():
    loader = KhmerWikipediaLoader()
    # Mock search response containing meta pages
    raw_mock_items = [
        {"title": "វិគីភិឌា:WikiProject History/Outreach"},
        {"title": "ជំនួយ:ការកែសម្រួល"},
        {"title": "ទំព័រគំរូ:Infobox Country"},
        {"title": "ប្រវត្តិសាស្ត្រខ្មែរ"},
        {"title": "អង្គរវត្ត"},
    ]
    from khmer_ocr.dataset.wikipedia_loader import IGNORED_NAMESPACES
    filtered = [
        it["title"] for it in raw_mock_items
        if not any(it["title"].startswith(p) for p in IGNORED_NAMESPACES)
    ]
    assert filtered == ["ប្រវត្តិសាស្ត្រខ្មែរ", "អង្គរវត្ត"]
