"""Khmer Font Discovery, Categorization & Management."""

import os
from pathlib import Path
from PIL import ImageFont


# Known system paths for Khmer fonts on Ubuntu/Linux
SYSTEM_FONT_DIRS = [
    Path("fonts"),  # Local project fonts folder
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path("~/.fonts").expanduser(),
    Path("~/.local/share/fonts").expanduser(),
]

STYLE_ALL = "All Styles"
STYLE_KHATT = "Standard Print (Khatt / ខាត់)"
STYLE_MOUL = "Ornate / Title (Moul / មូល)"
STYLE_CHRIENG = "Slanted / Cursive (Chrieng / ជ្រៀង)"


class KhmerFontManager:
    """Discovers, catalogs, and loads Khmer TrueType and OpenType fonts."""

    def __init__(self, custom_dirs: list[str | Path] | None = None):
        self.font_dirs = [Path(d) for d in (custom_dirs or [])] + SYSTEM_FONT_DIRS
        self.fonts: dict[str, Path] = {}
        self.font_styles: dict[str, str] = {}
        self.scan_fonts()

    def scan_fonts(self) -> dict[str, Path]:
        """Scans directories for Khmer-compatible fonts and categorizes them by style."""
        self.fonts.clear()
        self.font_styles.clear()
        extensions = {".ttf", ".otf", ".ttc"}

        for font_dir in self.font_dirs:
            if not font_dir.exists():
                continue
            for ext in extensions:
                for path in font_dir.rglob(f"*{ext}"):
                    name_lower = path.stem.lower()
                    if any(k in name_lower for k in [
                        "khmer", "moul", "battambang", "bayon", "bokor",
                        "fasthand", "koulen", "metal", "siemreap", "dangrek", "suwannaphum"
                    ]):
                        self.fonts[path.stem] = path

                        # Classify font style
                        if "moul" in name_lower or "muol" in name_lower:
                            self.font_styles[path.stem] = STYLE_MOUL
                        elif "chrieng" in name_lower or "slant" in name_lower or "italic" in name_lower or "fasthand" in name_lower:
                            self.font_styles[path.stem] = STYLE_CHRIENG
                        else:
                            self.font_styles[path.stem] = STYLE_KHATT

        return self.fonts

    def add_custom_font(self, font_path: str | Path) -> str:
        """Adds a custom font file directly into the manager."""
        p = Path(font_path)
        if p.exists() and p.suffix.lower() in {".ttf", ".otf", ".ttc"}:
            name = p.stem
            self.fonts[name] = p
            name_lower = name.lower()
            if "moul" in name_lower or "muol" in name_lower:
                self.font_styles[name] = STYLE_MOUL
            elif "chrieng" in name_lower or "slant" in name_lower:
                self.font_styles[name] = STYLE_CHRIENG
            else:
                self.font_styles[name] = STYLE_KHATT
            return name
        raise ValueError(f"Invalid font file: {font_path}")

    def get_font_names(self, style: str = STYLE_ALL) -> list[str]:
        """Returns sorted list of font names, optionally filtered by style category."""
        if style == STYLE_ALL:
            return sorted(list(self.fonts.keys()))
        return sorted([name for name, st in self.font_styles.items() if st == style])

    def get_font_path(self, font_name: str) -> Path | None:
        """Returns Path object for the specified font name."""
        return self.fonts.get(font_name)

    def load_font(self, font_name: str, size: int = 32) -> ImageFont.FreeTypeFont:
        """Loads a font by name or falls back to first available font."""
        path = self.fonts.get(font_name)
        if path is None or not path.exists():
            if not self.fonts:
                raise FileNotFoundError("No Khmer fonts found on the system!")
            path = next(iter(self.fonts.values()))

        # layout_engine=ImageFont.LAYOUT_RAQM is used where supported
        try:
            return ImageFont.truetype(str(path), size=size, layout_engine=ImageFont.LAYOUT_RAQM)
        except Exception:
            return ImageFont.truetype(str(path), size=size)
