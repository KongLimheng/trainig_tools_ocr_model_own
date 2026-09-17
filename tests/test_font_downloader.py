import sys
import json
import zipfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_app import build_parser
from khmer_ocr.synth.font_downloader import KhmerFontDownloader, compute_file_sha256
from khmer_ocr.synth.fonts import _load_font_cache, _save_font_cache, CACHE_FILENAME


def test_download_fonts_cli_argument_parsing():
    """Verifies that download-fonts argument parser parses all options and defaults."""
    parser = build_parser()

    # Test default arguments
    args = parser.parse_args(["download-fonts"])
    assert args.command == "download-fonts"
    assert args.out == "fonts"
    assert args.source == "all"
    assert args.zip_file is None
    assert args.custom_url is None
    assert args.clean is False

    # Test custom arguments
    args2 = parser.parse_args([
        "download-fonts",
        "--out", "custom_fonts",
        "--source", "7piseth",
        "--zip", "my_fonts.zip",
        "--url", "https://example.com/fonts.zip",
        "--clean",
    ])
    assert args2.out == "custom_fonts"
    assert args2.source == "7piseth"
    assert args2.zip_file == "my_fonts.zip"
    assert args2.custom_url == "https://example.com/fonts.zip"
    assert args2.clean is True


def test_file_sha256_computation(tmp_path):
    """Verifies SHA-256 fingerprinting for byte-exact duplicate detection."""
    f1 = tmp_path / "font1.ttf"
    f2 = tmp_path / "font2.ttf"
    f3 = tmp_path / "font3.ttf"

    f1.write_bytes(b"TEST_FONT_DATA_A")
    f2.write_bytes(b"TEST_FONT_DATA_A")  # Identical content
    f3.write_bytes(b"TEST_FONT_DATA_B")  # Different content

    h1 = compute_file_sha256(f1)
    h2 = compute_file_sha256(f2)
    h3 = compute_file_sha256(f3)

    assert h1 == h2
    assert h1 != h3


def test_ingest_zip_extraction(tmp_path):
    """Verifies that font files are extracted from a zip archive while ignoring non-fonts."""
    zip_path = tmp_path / "test_fonts.zip"
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()

    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("KhmerFont_Regular.ttf", b"DUMMY_TTF")
        z.writestr("KhmerFont_Bold.otf", b"DUMMY_OTF")
        z.writestr("notes.txt", b"Should be ignored")
        z.writestr("subfolder/Display.ttf", b"DUMMY_SUB_TTF")

    downloader = KhmerFontDownloader(output_dir=tmp_path / "fonts_out")
    extracted = downloader.ingest_zip(zip_path, extract_dir)

    assert len(extracted) == 3
    extracted_names = {p.name for p in extracted}
    assert "KhmerFont_Regular.ttf" in extracted_names
    assert "KhmerFont_Bold.otf" in extracted_names
    assert "Display.ttf" in extracted_names


def test_font_cache_serialization_and_fast_loading(tmp_path):
    """Verifies persistent font cache saving and loading."""
    cache_file = tmp_path / CACHE_FILENAME
    cache_data = {
        "/fake/path/KhmerFont.ttf": {
            "mtime": 123456.0,
            "size": 10240,
            "is_khmer": True,
            "style": "Standard Print (Khatt / ខាត់)"
        }
    }

    _save_font_cache(cache_file, cache_data)
    assert cache_file.exists()

    loaded = _load_font_cache(cache_file)
    assert "/fake/path/KhmerFont.ttf" in loaded
    assert loaded["/fake/path/KhmerFont.ttf"]["is_khmer"] is True


def test_generate_catalog(tmp_path):
    """Verifies generation of fonts_catalog.json."""
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    (fonts_dir / "KhmerMoul_Test.ttf").write_bytes(b"DUMMY_MOUL")
    (fonts_dir / "KhmerChrieng_Test.ttf").write_bytes(b"DUMMY_CHRIENG")

    downloader = KhmerFontDownloader(output_dir=fonts_dir)
    cat_path = downloader.generate_catalog()

    assert cat_path.exists()
    with open(cat_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_fonts"] == 2
    assert "styles_breakdown" in data
