"""Zero-Width Space & Invisible Character Cleaning for Khmer Text."""

import re

# Invisible and Zero-Width Unicode codepoints
ZWSP = "\u200B"       # Zero-Width Space
ZWNJ = "\u200C"       # Zero-Width Non-Joiner
ZWJ = "\u200D"        # Zero-Width Joiner
BOM = "\uFEFF"        # Byte Order Mark / Zero Width No-Break Space
NBSP = "\u00A0"       # Non-Breaking Space
SOFT_HYPHEN = "\u00AD"# Soft Hyphen

INVISIBLE_CHARS = {ZWSP, ZWNJ, ZWJ, BOM, SOFT_HYPHEN}

RE_CONSECUTIVE_SPACES = re.compile(r"[ \t]+")
RE_CONSECUTIVE_NEWLINES = re.compile(r"\n\s*\n+")


def strip_invisible_chars(text: str, keep_zwnj: bool = False) -> str:
    """Removes invisible characters such as ZWSP, BOM, and ZWNJ.
    In OCR visual ground-truth labels, invisible spaces do not correspond to any visual ink,
    so removing them ensures visual correspondence and prevents penalizing OCR models.
    """
    if not text:
        return ""

    chars_to_remove = set(INVISIBLE_CHARS)
    if keep_zwnj:
        chars_to_remove.discard(ZWNJ)

    filtered = [ch for ch in text if ch not in chars_to_remove]
    result = "".join(filtered)
    # Convert NBSP to normal space
    result = result.replace(NBSP, " ")
    # Clean multiple spaces
    result = RE_CONSECUTIVE_SPACES.sub(" ", result)
    return result.strip()


def has_invisible_chars(text: str) -> bool:
    """Checks if text contains any zero-width or invisible characters."""
    return any(ch in INVISIBLE_CHARS for ch in text)


def count_invisible_chars(text: str) -> dict[str, int]:
    """Counts occurrences of each invisible character in text."""
    counts = {
        "ZWSP": text.count(ZWSP),
        "ZWNJ": text.count(ZWNJ),
        "ZWJ": text.count(ZWJ),
        "BOM": text.count(BOM),
        "NBSP": text.count(NBSP),
    }
    return {k: v for k, v in counts.items() if v > 0}
