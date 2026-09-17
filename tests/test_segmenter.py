"""Unit tests for Khmer document line segmenter."""

import pytest
from PIL import Image, ImageDraw, ImageFont
from khmer_ocr.pipeline.segmenter import KhmerDocumentLineSegmenter
from khmer_ocr.synth.fonts import KhmerFontManager


def test_document_line_segmentation():
    mgr = KhmerFontManager()
    font = mgr.load_font(mgr.get_font_names()[0], size=32)

    # Create a 3-line document image
    img = Image.new("RGB", (600, 300), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 30), "ព្រះរាជាណាចក្រកម្ពុជា", font=font, fill=(0, 0, 0))
    draw.text((30, 110), "ជាតិ សាសនា ព្រះមហាក្សត្រ", font=font, fill=(0, 0, 0))
    draw.text((30, 190), "សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ", font=font, fill=(0, 0, 0))

    segmenter = KhmerDocumentLineSegmenter(min_line_height=15, min_line_width=40)
    bboxes, crops, annotated = segmenter.segment(img)

    assert len(bboxes) == 3, f"Expected 3 lines, got {len(bboxes)}"
    assert len(crops) == 3
    assert annotated.size == (600, 300)

    for (x, y, w, h), crop in zip(bboxes, crops):
        assert w > 40
        assert h > 15
        assert crop.size[0] == w
        assert crop.size[1] == h
