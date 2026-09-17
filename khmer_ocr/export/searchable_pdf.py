"""Searchable PDF (Sandwich PDF) Exporter for Khmer Documents."""

import os
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from ..synth.fonts import KhmerFontManager


class SearchablePDFExporter:
    """Generates a dual-layer Searchable PDF (original scan image + invisible text layer)."""

    def __init__(self, font_path: str | None = None):
        self.font_name = "KhmerSearchableFont"
        self._register_font(font_path)

    def _register_font(self, font_path: str | None = None) -> None:
        """Registers a TrueType font for PDF text generation."""
        if font_path is None:
            mgr = KhmerFontManager()
            names = mgr.get_font_names()
            p = mgr.get_font_path(names[0]) if names else None
            font_path = str(p) if p else None

        if font_path and Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont(self.font_name, font_path))
                self.has_font = True
            except Exception:
                self.has_font = False
        else:
            self.has_font = False

    def export(
        self,
        image: Image.Image,
        bboxes: List[Tuple[int, int, int, int]],
        transcriptions: List[str],
        output_pdf_path: str | Path,
    ) -> Path:
        """Exports a single document page as a Searchable PDF.

        Args:
            image: Original PIL Image.
            bboxes: List of (x, y, w, h) bounding boxes in image pixel coordinates.
            transcriptions: List of recognized text strings corresponding to bboxes.
            output_pdf_path: Destination PDF file path.

        Returns:
            Path to the generated PDF.
        """
        out_path = Path(output_pdf_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img_w, img_h = image.size
        # PDF dimensions match image dimensions (in points)
        pdf = canvas.Canvas(str(out_path), pagesize=(img_w, img_h))

        # 1. Draw original image as the visual background layer
        reader = ImageReader(image)
        pdf.drawImage(reader, 0, 0, width=img_w, height=img_h)

        # 2. Draw invisible text layer over the exact bounding box coordinates
        # Text render mode 3 = invisible text (neither fill nor stroke)
        pdf._code.append("3 Tr")

        font_to_use = self.font_name if self.has_font else "Helvetica"

        for (x, y, w, h), text in zip(bboxes, transcriptions):
            if not text.strip():
                continue

            # Convert image coordinate (top-left origin) to PDF coordinate (bottom-left origin)
            pdf_x = x
            pdf_y = img_h - (y + h)

            # Estimate font size to fit bounding box height
            font_size = max(8, int(h * 0.75))
            try:
                pdf.setFont(font_to_use, font_size)
                pdf.drawString(pdf_x, pdf_y + int(h * 0.15), text)
            except Exception:
                # Fallback if character outside standard encoding
                try:
                    pdf.setFont("Helvetica", font_size)
                    pdf.drawString(pdf_x, pdf_y + int(h * 0.15), text)
                except Exception:
                    pass

        # Reset text render mode to normal (0 = fill)
        pdf._code.append("0 Tr")

        pdf.showPage()
        pdf.save()
        return out_path
