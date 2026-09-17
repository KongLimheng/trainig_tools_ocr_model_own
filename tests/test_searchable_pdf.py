"""Unit tests for Searchable PDF (Sandwich PDF) Exporter."""

from pathlib import Path
import zlib
import re
from PIL import Image, ImageDraw
from khmer_ocr.export.searchable_pdf import SearchablePDFExporter


def test_export_searchable_pdf(tmp_path: Path):
    exporter = SearchablePDFExporter()
    out_pdf = tmp_path / "test_doc.pdf"

    # Create dummy image
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Line 1 Test", fill=(0, 0, 0))
    draw.text((50, 150), "Line 2 Test", fill=(0, 0, 0))

    bboxes = [(50, 50, 200, 30), (50, 150, 220, 30)]
    transcriptions = ["ព្រះរាជាណាចក្រកម្ពុជា", "រាជធានីភ្នំពេញ"]

    result_path = exporter.export(
        image=img,
        bboxes=bboxes,
        transcriptions=transcriptions,
        output_pdf_path=out_pdf,
    )

    assert result_path.exists()
    assert result_path.stat().st_size > 1000

    pdf_bytes = result_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"%%EOF" in pdf_bytes
    assert b"/Type /Pages" in pdf_bytes
    assert b"/Type /Font" in pdf_bytes
    assert b"/Type /XObject" in pdf_bytes
