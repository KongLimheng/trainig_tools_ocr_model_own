"""Khmer Unicode Normalization & Canonical Ordering Rules.

Enforces canonical order according to Unicode Khmer block specifications:
Base Consonant -> Consonant Shifter -> Subscripts (Coeng + Consonant) -> Dependent Vowel -> Diacritics.
Cleans duplicate Coeng signs, handles compound vowels, and normalizes legacy sequences.
"""

import re
import unicodedata

# Khmer Unicode Block Constants
KHMER_CONSONANTS = set(chr(cp) for cp in range(0x1780, 0x17A3))  # ក - អ (33)
KHMER_INDEP_VOWELS = set(chr(cp) for cp in range(0x17A3, 0x17B4))  # ឦ - ឳ
KHMER_DEP_VOWELS = set(chr(cp) for cp in range(0x17B6, 0x17C6))  # ា - ៅ
KHMER_COENG = "\u17D2"  # ្ Coeng
KHMER_SHIFTERS = {"\u17C9", "\u17CA"}  # ៉ (Muusikatoan), ៊ (Triisap)
KHMER_DIACRITICS = {
    "\u17C6",  # ំ (Nikahit)
    "\u17C7",  # ះ (Reahmuk)
    "\u17C8",  # ៈ (Camnuc Pii Kuuh)
    "\u17CB",  # ់ (Bantoc)
    "\u17CC",  # ៌ (Robat)
    "\u17CD",  # ៍ (Toandakhiat)
    "\u17CE",  # ៎ (Kakabat)
    "\u17CF",  # ៏ (Ahsda)
    "\u17D0",  # ័ (Samyok Sannya)
    "\u17D1",  # ៑ (Viriam)
    "\u17D3",  # ៓ (Bathamasat)
}
KHMER_DIGITS = set(chr(cp) for cp in range(0x17E0, 0x17EA))  # ០ - ៩
KHMER_SYMBOLS = {"\u17D4", "\u17D5", "\u17D6", "\u17D7", "\u17D8", "\u17D9", "\u17DA", "\u17DB"}

# Compound Vowel Normalization Map (Decomposed -> Composed standard)
COMPOUND_VOWEL_MAP = {
    "\u17C1\u17B8": "\u17BE",  # ើ
    "\u17C1\u17B6": "\u17C4",  # ោ
    "\u17C1\u17B6\u17C6": "\u17C4\u17C6",  # ោម
    "\u17C1\u17C5": "\u17C5",  # ៅ
    "\u17C1\u17B9": "\u17BF",  # ឿ
    "\u17C1\u17BA": "\u17C0",  # ៀ
}

# Regex patterns
RE_MULTIPLE_COENG = re.compile(r"\u17D2{2,}")
RE_ORPHAN_COENG_END = re.compile(r"\u17D2+$")


def clean_duplicate_coeng(text: str) -> str:
    """Replaces duplicate Coeng signs with a single Coeng."""
    return RE_MULTIPLE_COENG.sub(KHMER_COENG, text)


def normalize_compound_vowels(text: str) -> str:
    """Normalizes compound vowel keystrokes into canonical single codepoints."""
    for decomposed, composed in COMPOUND_VOWEL_MAP.items():
        text = text.replace(decomposed, composed)
    return text


def is_khmer_char(char: str) -> bool:
    """Checks if a character belongs to the Khmer Unicode block."""
    cp = ord(char)
    return (0x1780 <= cp <= 0x17FF) or (0x19E0 <= cp <= 0x19FF)


class KhmerSyllable:
    """Represents a structured Khmer orthographic cluster."""

    def __init__(self):
        self.base: str = ""
        self.shifter: str = ""
        self.subscripts: list[str] = []  # List of (COENG + consonant)
        self.vowels: list[str] = []
        self.diacritics: list[str] = []
        self.others: list[str] = []

    def canonical_string(self) -> str:
        """Returns the syllable characters reordered strictly in canonical sequence:
        Base -> Shifter -> Subscripts (in sorted order if multiple) -> Vowels -> Diacritics.
        """
        parts = []
        if self.base:
            parts.append(self.base)
        if self.shifter:
            parts.append(self.shifter)
        parts.extend(self.subscripts)
        parts.extend(self.vowels)
        parts.extend(self.diacritics)
        parts.extend(self.others)
        return "".join(parts)


def reorder_cluster(cluster_chars: list[str]) -> str:
    """Reorders a single Khmer cluster into canonical form."""
    if not cluster_chars:
        return ""

    syllable = KhmerSyllable()
    idx = 0
    n = len(cluster_chars)

    # 1. Base consonant or independent vowel
    if idx < n and (cluster_chars[idx] in KHMER_CONSONANTS or cluster_chars[idx] in KHMER_INDEP_VOWELS):
        syllable.base = cluster_chars[idx]
        idx += 1

    # Loop through remaining characters in the cluster
    while idx < n:
        ch = cluster_chars[idx]
        if ch in KHMER_SHIFTERS:
            if not syllable.shifter:
                syllable.shifter = ch
            else:
                syllable.diacritics.append(ch)
            idx += 1
        elif ch == KHMER_COENG:
            # Subscript sequence: Coeng + following char
            if idx + 1 < n and (cluster_chars[idx + 1] in KHMER_CONSONANTS or cluster_chars[idx + 1] in KHMER_INDEP_VOWELS):
                syllable.subscripts.append(KHMER_COENG + cluster_chars[idx + 1])
                idx += 2
            else:
                idx += 1  # Skip orphan coeng
        elif ch in KHMER_DEP_VOWELS:
            syllable.vowels.append(ch)
            idx += 1
        elif ch in KHMER_DIACRITICS:
            syllable.diacritics.append(ch)
            idx += 1
        else:
            syllable.others.append(ch)
            idx += 1

    return syllable.canonical_string()


def normalize_khmer_canonical(text: str) -> str:
    """Performs full canonical normalization of Khmer text.
    1. Unicode NFC normalization.
    2. Deduplication of Coeng characters.
    3. Compound vowel normalization.
    4. Syllable-level reordering of base, shifter, subscript, vowel, diacritics.
    """
    if not text:
        return ""

    # First NFC normalize
    text = unicodedata.normalize("NFC", text)
    text = clean_duplicate_coeng(text)
    text = normalize_compound_vowels(text)

    # Segment into clusters and reorder
    result = []
    current_cluster: list[str] = []

    for char in text:
        if not is_khmer_char(char):
            if current_cluster:
                result.append(reorder_cluster(current_cluster))
                current_cluster = []
            result.append(char)
            continue

        # Check if this starts a new base consonant cluster
        if char in KHMER_CONSONANTS or char in KHMER_INDEP_VOWELS or char in KHMER_DIGITS or char in KHMER_SYMBOLS:
            # If preceding char was Coeng, this char is part of the subscript!
            if current_cluster and current_cluster[-1] == KHMER_COENG:
                current_cluster.append(char)
            else:
                if current_cluster:
                    result.append(reorder_cluster(current_cluster))
                    current_cluster = []
                current_cluster.append(char)
        else:
            current_cluster.append(char)

    if current_cluster:
        result.append(reorder_cluster(current_cluster))

    return "".join(result)
