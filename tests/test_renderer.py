"""Unit tests for Khmer synthetic text renderer."""

import pytest
from khmer_ocr.synth.renderer import KhmerTextRenderer
from khmer_ocr.synth.fonts import KhmerFontManager


def test_font_discovery():
    mgr = KhmerFontManager()
    names = mgr.get_font_names()
    assert len(names) > 0, "Expected at least one Khmer font on the system!"


def test_renderer_outputs():
    renderer = KhmerTextRenderer()
    text = "សួស្តីកម្ពុជា"
    img, label = renderer.render_line(text, target_height=48, augment=False)

    assert img is not None
    assert img.size[1] == 48
    assert img.size[0] > 48
    assert label == "សួស្តីកម្ពុជា"


def test_renderer_with_subscripts():
    renderer = KhmerTextRenderer()
    text = "សង្គ្រាម និង វិទ្យាសាស្ត្រ"
    img, label = renderer.render_line(text, target_height=48, augment=True)

    assert img is not None
    assert img.size[1] == 48


def test_is_khmer_font_validation():
    from pathlib import Path
    from khmer_ocr.synth.fonts import is_khmer_font

    mgr = KhmerFontManager()
    font_names = mgr.get_font_names()
    assert len(font_names) > 0

    first_path = mgr.get_font_path(font_names[0])
    assert first_path is not None
    assert is_khmer_font(first_path) is True
