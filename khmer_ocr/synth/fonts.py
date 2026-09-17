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

KHMER_NAME_KEYWORDS = [
    "khmer", "moul", "battambang", "bayon", "bokor",
    "fasthand", "koulen", "metal", "siemreap", "dangrek", "suwannaphum",
    "chenla", "angkor", "kantumruy", "preahvihear", "hanuman", "taprom",
    "content", "nokora", "muol"
]


def is_khmer_font(font_path: Path, min_consonants: int = 30) -> bool:
    """Checks if a font file truly supports Khmer characters via Unicode cmap.

    Requires at least min_consonants (default: 30 of 35) in U+1780 - U+17A2 to
    prevent Thai/Lao/Latin fonts with lone symbols (e.g. NotoLoopedThai) from
    rendering tofu boxes.
    """
    name_lower = font_path.stem.lower()
    # Reject known non-Khmer font families by name immediately
    if any(non in name_lower for non in [
        "thai", "lao", "myanmar", "burmese", "tibetan", "devanagari",
        "bengali", "gurmukhi", "gujarati", "oriya", "tamil", "telugu",
        "kannada", "malayalam", "sinhala", "arabic", "hebrew"
    ]):
        return False

    # Inspect cmap tables using fontTools
    try:
        from fontTools.ttLib import TTFont
        tt = TTFont(str(font_path), fontNumber=0)
        best_cmap = tt.getBestCmap()
        if best_cmap:
            matching = sum(1 for cp in range(0x1780, 0x17A3) if cp in best_cmap)
            return matching >= min_consonants

        cmap = tt.get("cmap")
        if cmap:
            for table in cmap.tables:
                matching = sum(1 for cp in range(0x1780, 0x17A3) if cp in table.cmap.keys())
                if matching >= min_consonants:
                    return True
    except Exception:
        pass

    return False


def classify_font_style(font_name: str) -> str:
    """Classifies a font into Khatt, Moul, or Chrieng style."""
    name_lower = font_name.lower()
    if "moul" in name_lower or "muol" in name_lower or "ornate" in name_lower:
        return STYLE_MOUL
    elif any(s in name_lower for s in ["chrieng", "slant", "italic", "fasthand"]):
        return STYLE_CHRIENG
    else:
        return STYLE_KHATT


CACHE_FILENAME = ".font_cache.json"


def _load_font_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        try:
            import json
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_font_cache(cache_file: Path, cache_data: dict) -> None:
    try:
        import json
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)
    except Exception:
        pass


class KhmerFontManager:
    """Discovers, catalogs, and loads Khmer TrueType and OpenType fonts with persistent caching."""

    def __init__(self, custom_dirs: list[str | Path] | None = None, use_cache: bool = True):
        user_dirs = [Path(d) for d in (custom_dirs or []) if d is not None]
        if user_dirs:
            # Isolate to explicitly requested directories (e.g. fonts/)
            self.font_dirs = user_dirs
        else:
            self.font_dirs = SYSTEM_FONT_DIRS
        self.use_cache = use_cache
        self.fonts: dict[str, Path] = {}
        self.font_styles: dict[str, str] = {}
        self.scan_fonts()

    def scan_fonts(self) -> dict[str, Path]:
        """Scans directories for Khmer-compatible fonts and categorizes them by style using cache."""
        self.fonts.clear()
        self.font_styles.clear()
        extensions = {".ttf", ".otf", ".ttc"}

        cache_dir = Path("fonts")
        cache_file = cache_dir / CACHE_FILENAME
        cache_data = _load_font_cache(cache_file) if self.use_cache else {}
        cache_dirty = False

        for font_dir in self.font_dirs:
            if not font_dir.exists():
                continue
            for ext in extensions:
                for path in font_dir.rglob(f"*{ext}"):
                    if path.stem in self.fonts:
                        continue

                    key = str(path.resolve())
                    try:
                        st = path.stat()
                        mtime = st.st_mtime
                        size = st.st_size
                    except Exception:
                        continue

                    cached = cache_data.get(key)
                    if cached and cached.get("mtime") == mtime and cached.get("size") == size:
                        is_kh = cached.get("is_khmer", False)
                        style = cached.get("style", STYLE_KHATT)
                    else:
                        is_kh = is_khmer_font(path)
                        style = classify_font_style(path.stem)
                        cache_data[key] = {
                            "mtime": mtime,
                            "size": size,
                            "is_khmer": is_kh,
                            "style": style
                        }
                        cache_dirty = True

                    if is_kh:
                        self.fonts[path.stem] = path
                        self.font_styles[path.stem] = style

        if cache_dirty and self.use_cache:
            _save_font_cache(cache_file, cache_data)

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

        # Resolve Raqm layout engine across Pillow versions (ImageFont.Layout.RAQM in Pillow 10+)
        layout_engine = getattr(getattr(ImageFont, "Layout", None), "RAQM", getattr(ImageFont, "LAYOUT_RAQM", None))
        try:
            if layout_engine is not None:
                return ImageFont.truetype(str(path), size=size, layout_engine=layout_engine)
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            return ImageFont.truetype(str(path), size=size)
