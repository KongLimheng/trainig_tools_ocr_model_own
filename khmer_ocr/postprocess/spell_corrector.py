"""Khmer OCR Spell Corrector using Confusion Candidates & Dictionary Matching."""

from typing import List, Tuple
from .word_segmenter import KhmerWordSegmenter, DEFAULT_KHMER_DICTIONARY
from ..normalizer import normalize_khmer_text

# High-frequency OCR confusion pairs in Khmer script
COMMON_CONFUSION_PAIRS = [
    ("គ", "ត"), ("ត", "គ"),
    ("ញ", "ឈ"), ("ឈ", "ញ"),
    ("ឋ", "យ"), ("យ", "ឋ"),
    ("ដ", "ឌ"), ("ឌ", "ដ"),
    ("ផ", "ធ"), ("ធ", "ផ"),
    ("ព", "ឃ"), ("ឃ", "ព"),
    ("ប", "ម"), ("ម", "ប"),
    ("់", ""), ("", "់"),       # Missing or false Bantoc
    ("្ញ", "្ឋ"), ("្ឋ", "្ញ"),
    ("្ម", "្ន"), ("្ន", "្ម"),
]


class KhmerSpellCorrector:
    """Detects and rectifies common OCR substitution errors in Khmer words."""

    def __init__(self, segmenter: KhmerWordSegmenter | None = None):
        self.segmenter = segmenter or KhmerWordSegmenter()
        self.valid_words = set(DEFAULT_KHMER_DICTIONARY)

    def add_vocabulary(self, words: List[str]):
        """Adds additional words to the valid vocabulary set."""
        for w in words:
            self.valid_words.add(normalize_khmer_text(w, strip_zwsp=True))

    def correct_word(self, word: str) -> Tuple[str, bool]:
        """Checks if a single word has an OCR error and rectifies it.
        Returns: (corrected_word, was_corrected)
        """
        word = normalize_khmer_text(word, strip_zwsp=True)
        if not word or word in self.valid_words:
            return word, False

        # Try 1-edit substitution based on Khmer OCR confusion pairs
        for src, dst in COMMON_CONFUSION_PAIRS:
            if src and src in word:
                # Replace occurrence of confusable glyph
                candidate = word.replace(src, dst, 1)
                candidate_norm = normalize_khmer_text(candidate, strip_zwsp=True)
                if candidate_norm in self.valid_words:
                    return candidate_norm, True

        return word, False

    def correct_sentence(self, sentence: str) -> Tuple[str, int]:
        """Segments sentence into words, applies spell correction, and rejoins.
        Returns: (corrected_sentence, number_of_corrections)
        """
        raw_tokens = sentence.split()
        if not raw_tokens:
            return "", 0

        corrected_tokens = []
        corrections_count = 0

        for token in raw_tokens:
            # First check if the token itself is a single misspelled dictionary word
            corr_t, changed = self.correct_word(token)
            if changed:
                corrected_tokens.append(corr_t)
                corrections_count += 1
                continue

            # If token is not directly corrected, segment it into words / syllables
            sub_words = self.segmenter.segment(token)
            i = 0
            while i < len(sub_words):
                # Try window of 2 syllables if misspelled
                if i + 1 < len(sub_words):
                    combined_2 = sub_words[i] + sub_words[i + 1]
                    corr_w, chg = self.correct_word(combined_2)
                    if chg:
                        corrected_tokens.append(corr_w)
                        corrections_count += 1
                        i += 2
                        continue

                # Single word or syllable
                corr_w, chg = self.correct_word(sub_words[i])
                corrected_tokens.append(corr_w)
                if chg:
                    corrections_count += 1
                i += 1

        return " ".join([w for w in corrected_tokens if w.strip()]), corrections_count

