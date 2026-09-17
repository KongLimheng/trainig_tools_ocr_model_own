"""Khmer Orthographic Syllable / Cluster Segmenter.

Splits continuous Khmer text into indivisible orthographic clusters.
Each cluster consists of:
[Base Consonant / Independent Vowel] + [Shifter]? + [Coeng + Subscript]* + [Dependent Vowels]* + [Diacritics]*
"""

from .unicode_rules import (
    KHMER_CONSONANTS,
    KHMER_INDEP_VOWELS,
    KHMER_DEP_VOWELS,
    KHMER_COENG,
    KHMER_SHIFTERS,
    KHMER_DIACRITICS,
    KHMER_DIGITS,
    KHMER_SYMBOLS,
    is_khmer_char,
)


def split_into_syllables(text: str) -> list[str]:
    """Splits a Khmer string into a list of orthographic syllable clusters.
    Non-Khmer characters (Latin, punctuation, whitespace) are returned as individual tokens.
    """
    if not text:
        return []

    syllables: list[str] = []
    current_cluster: list[str] = []

    for i, char in enumerate(text):
        if not is_khmer_char(char):
            if current_cluster:
                syllables.append("".join(current_cluster))
                current_cluster = []
            syllables.append(char)
            continue

        # Khmer digit or punctuation symbol forms its own atomic token
        if char in KHMER_DIGITS or char in KHMER_SYMBOLS:
            if current_cluster:
                syllables.append("".join(current_cluster))
                current_cluster = []
            syllables.append(char)
            continue

        # Check if this starts a new base consonant cluster
        is_base_starter = (char in KHMER_CONSONANTS or char in KHMER_INDEP_VOWELS)
        is_following_coeng = (current_cluster and current_cluster[-1] == KHMER_COENG)

        if is_base_starter and not is_following_coeng:
            if current_cluster:
                syllables.append("".join(current_cluster))
                current_cluster = []
            current_cluster.append(char)
        else:
            current_cluster.append(char)

    if current_cluster:
        syllables.append("".join(current_cluster))

    return syllables


def count_khmer_syllables(text: str) -> int:
    """Counts the number of Khmer orthographic clusters in the text."""
    return len([s for s in split_into_syllables(text) if s and any(is_khmer_char(c) for c in s)])
