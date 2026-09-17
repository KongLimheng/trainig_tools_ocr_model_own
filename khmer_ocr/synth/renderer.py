"""Khmer Text Line Renderer using Pillow with Raqm/HarfBuzz Shaping."""

import random
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont
from .fonts import KhmerFontManager
from .augmentations import (
    create_paper_background,
    apply_random_augmentations,
)
from ..normalizer.unicode_rules import normalize_khmer_canonical
from ..normalizer.zwsp_cleaner import strip_invisible_chars


class KhmerTextRenderer:
    """Renders high-quality Khmer text images with proper subscript shaping and bounding box."""

    def __init__(self, font_manager: KhmerFontManager | None = None):
        self.font_mgr = font_manager or KhmerFontManager()

    def render_line(
        self,
        text: str,
        font_name: str | None = None,
        font_size: int = 36,
        target_height: int = 48,
        text_color: Tuple[int, int, int] | None = None,
        bg_style: str = "clean",
        augment: bool = False,
        padding: int = 12,
    ) -> Tuple[Image.Image, str]:
        """Renders a single line of Khmer text into an image.
        Returns:
            (PIL Image, canonical normalized label)
        """
        # Step 1: Clean and canonicalize text
        clean_text = strip_invisible_chars(text)
        canonical_label = normalize_khmer_canonical(clean_text)
        if not canonical_label:
            canonical_label = "ក"

        # Step 2: Select font
        if font_name is None:
            avail = self.font_mgr.get_font_names()
            font_name = random.choice(avail) if avail else "NotoSansKhmer-Regular"

        font = self.font_mgr.load_font(font_name, size=font_size)

        # Step 3: Measure text bounds using Raqm
        # Create a dummy image to measure precise text bbox
        dummy_img = Image.new("RGB", (10, 10), (255, 255, 255))
        draw = ImageDraw.Draw(dummy_img)

        try:
            bbox = draw.textbbox((0, 0), canonical_label, font=font, direction="ltr")
        except TypeError:
            bbox = draw.textbbox((0, 0), canonical_label, font=font)

        # bbox is (left, top, right, bottom)
        left, top, right, bottom = bbox
        text_w = max(1, right - left)
        text_h = max(1, bottom - top)

        # In clean_doc mode, use tighter padding matching document line segmenter
        actual_padding = random.choice([6, 8, 10]) if bg_style == "clean_doc" and padding == 12 else padding

        # Canvas width and height with padding for subscripts and upper diacritics
        canvas_w = text_w + actual_padding * 2
        canvas_h = text_h + actual_padding * 2

        # Step 4: Create background
        bg_image = create_paper_background(canvas_w, canvas_h, style=bg_style)
        draw = ImageDraw.Draw(bg_image)

        # Step 5: Choose ink color
        if text_color is None:
            if bg_style == "clean_doc":
                text_color = random.choice([(0, 0, 0), (10, 10, 10), (20, 20, 20)])
            else:
                c_val = random.randint(10, 45)
                text_color = (c_val, c_val, c_val)

        # Draw text compensating for bounding box offset
        draw_x = actual_padding - left
        draw_y = actual_padding - top

        try:
            draw.text((draw_x, draw_y), canonical_label, font=font, fill=text_color, direction="ltr")
        except TypeError:
            draw.text((draw_x, draw_y), canonical_label, font=font, fill=text_color)

        # Step 6: Resize to standard target height if requested (preserving aspect ratio)
        if target_height is not None and canvas_h != target_height:
            aspect_ratio = canvas_w / canvas_h
            new_w = max(16, int(target_height * aspect_ratio))
            # High-quality Lanczos resampling
            bg_image = bg_image.resize((new_w, target_height), Image.Resampling.LANCZOS)

        # Step 7: Apply optional augmentations
        if augment and bg_style != "clean_doc":
            bg_image = apply_random_augmentations(bg_image)

        return bg_image, canonical_label
