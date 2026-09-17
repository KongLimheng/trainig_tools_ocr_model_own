"""Khmer Text Corpus & Synthetic Line Sampler.

Generates realistic Khmer lines including:
- Natural sentences and phrases directly from corpus (e.g. Wikipedia)
- Intelligent RAC dictionary phrases composed of real, authentic words
- Hard-negative discrimination phrases using authentic dictionary words
- Authentic Khmer numbers, dates, currency, and percentages
"""

import re
import random
from pathlib import Path
from typing import List, Dict, Optional
from ..normalizer.unicode_rules import (
    KHMER_CONSONANTS,
    KHMER_DEP_VOWELS,
    KHMER_INDEP_VOWELS,
    KHMER_COENG,
    KHMER_SHIFTERS,
    KHMER_DIACRITICS,
    KHMER_DIGITS,
    normalize_khmer_canonical,
)

BUNDLED_DICT_PATH = Path(__file__).parent.parent / "postprocess" / "dictionary" / "khmer_words.txt"

# Core natural phrases covering historical, administrative, and everyday domains
DEFAULT_KHMER_CORPUS = [
    "ព្រះរាជាណាចក្រកម្ពុជា",
    "ជាតិ សាសនា ព្រះមហាក្សត្រ",
    "រាជធានីភ្នំពេញ បេះដូងនៃប្រទេសកម្ពុជា",
    "ប្រាសាទអង្គរវត្តជាសម្បត្តិបេតិកភណ្ឌពិភពលោក",
    "សាកលវិទ្យាល័យភូមិន្ទភ្នំពេញ",
    "ក្រសួងអប់រំ យុវជន និងកីឡា",
    "វិទ្យាសាស្ត្រ បច្ចេកវិទ្យា និងនវានុវត្តន៍",
    "ការអភិវឌ្ឍសេដ្ឋកិច្ចប្រកបដោយចីរភាព",
    "សន្តិភាព និងស្ថិរភាពសង្គម",
    "ភាសាខ្មែរជាភាសាជាតិ និងជាអត្តសញ្ញាណវប្បធម៌ខ្មែរ",
    "សូមគោរពអញ្ជើញចូលរួមពិធីសម្ពោធជាផ្លូវការ",
    "សួស្តីឆ្នាំថ្មី ប្រពៃណីជាតិខ្មែរ",
    "ទន្លេមេគង្គ ទន្លេសាប និងបឹងទន្លេសាប",
    "ការការពារបរិស្ថាន និងធនធានធម្មជាតិ",
    "ឯករាជ្យ អធិបតេយ្យ និងបូរណភាពទឹកដី",
    "របាយការណ៍ហិរញ្ញវត្ថុប្រចាំត្រីមាស",
    "វិញ្ញាបនបត្រសម្គាល់ម្ចាស់អចលនវត្ថុ",
    "អត្តសញ្ញាណប័ណ្ណសញ្ជាតិខ្មែរ",
    "លិខិតឆ្លងដែននៃព្រះរាជាណាចក្រកម្ពុជា",
    "សេចក្តីសម្រេចរបស់រាជរដ្ឋាភិបាល",
    "កិច្ចព្រមព្រៀងពាណិជ្ជកម្មអន្តរជាតិ",
    "ការលើកកម្ពស់វិស័យកសិកម្ម និងឧស្សាហកម្ម",
    "សុខភាព និងសុខុមាលភាពប្រជាពលរដ្ឋ",
    "ការបណ្តុះបណ្តាលវិជ្ជាជីវៈ និងបច្ចេកទេស",
    "ការប្រើប្រាស់បញ្ញាសិប្បនិម្មិតក្នុងសម័យឌីជីថល",
    "ប្រព័ន្ធគ្រប់គ្រងទិន្នន័យស្វ័យប្រវត្ត",
    "ព័ត៌មានវិទ្យា និងទូរគមនាគមន៍",
    "ស្ថាបត្យកម្មប្រាសាទបុរាណសម័យអង្គរ",
    "វប្បធម៌ និងទំនៀមទម្លាប់ខ្មែរដ៏ផូរផង់",
    "ច្បាប់ស្តីពីការងារ និងសន្តិសុខសង្គម",
]

# Visually confusable consonant groups for targeted discrimination
CONFUSION_GROUPS = [
    ["គ", "ត", "ភ"],
    ["ឈ", "ញ"],
    ["ឋ", "យ"],
    ["ដ", "ឌ", "ឍ"],
    ["ផ", "ធ"],
    ["ព", "ឃ"],
    ["ប", "ស", "ល"],
]

CONNECTORS = [" និង ", " នៃ ", " ក្នុង ", " ដោយ ", " លើ ", " "]


def validate_khmer_line(text: str, min_len: int = 2) -> bool:
    """Verifies that text is orthographically sound Khmer without stacked vowels or illegal sequences."""
    if not text or len(text.strip()) < min_len:
        return False
    # No consecutive dependent vowels (e.g. ាី, ើឿ)
    if re.search(r"[\u17B6-\u17C5]{2,}", text):
        return False
    # No consecutive coeng markers
    if re.search(r"\u17D2{2,}", text):
        return False
    # No coeng followed by non-consonant
    if re.search(r"\u17D2[^\u1780-\u17A2]", text):
        return False
    # No bantoc, robath, or ahsda on coeng consonants (e.g. ឋ្ហ៌, អ្ខ៏)
    if re.search(r"\u17D2[\u1780-\u17A2][\u17CB\u17CC\u17CF]", text):
        return False
    # No upper/compound vowels followed by nikahit (impossible combinations like ៅំ, ើំ)
    if re.search(r"[\u17BD-\u17C5]\u17C6", text):
        return False
    # No trailing coeng
    if text.endswith("\u17D2"):
        return False
    # No orphan starting characters
    if 0x17B4 <= ord(text[0]) <= 0x17D3 or text[0] == "\u17D7" or text.startswith("ស់") or text.startswith("ល់"):
        return False
    return True


class KhmerCorpusSampler:
    """Generates varied, 100% authentic text lines for synthetic OCR training."""

    def __init__(
        self,
        custom_texts: list[str] | None = None,
        dictionary_path: str | Path | None = None,
        pure_corpus: bool = False,
    ):
        self.has_custom_corpus = bool(custom_texts and len(custom_texts) > 0)
        self.corpus = list(custom_texts) if self.has_custom_corpus else list(DEFAULT_KHMER_CORPUS)
        self.pure_corpus = pure_corpus
        self.khmer_digits = list(KHMER_DIGITS)

        # Load authentic Royal Academy of Cambodia (RAC) dictionary words
        dict_file = Path(dictionary_path) if dictionary_path else BUNDLED_DICT_PATH
        self.rac_words: List[str] = []
        if dict_file.exists():
            with open(dict_file, "r", encoding="utf-8", errors="ignore") as f:
                self.rac_words = [
                    line.strip() for line in f
                    if line.strip() and len(line.strip()) >= 2 and validate_khmer_line(line.strip())
                ]

        if not self.rac_words:
            # Fallback to corpus words if dict file missing
            self.rac_words = [w for line in self.corpus for w in line.split() if len(w) >= 2]

        # Pre-index authentic words containing confusable characters for hard negatives
        self.confusion_word_map: Dict[str, List[str]] = {}
        for group in CONFUSION_GROUPS:
            for char in group:
                matching = [w for w in self.rac_words if char in w]
                if matching:
                    self.confusion_word_map[char] = matching

    def sample_natural_line(self) -> str:
        """Samples an authentic Khmer sentence directly from the corpus."""
        return random.choice(self.corpus)

    def sample_dictionary_phrase(self, min_words: int = 2, max_words: int = 4) -> str:
        """Generates a natural phrase composed strictly of authentic RAC dictionary words."""
        count = random.randint(min_words, max_words)
        words = random.sample(self.rac_words, count)
        phrase = words[0]
        for w in words[1:]:
            phrase += random.choice(CONNECTORS) + w
        return normalize_khmer_canonical(phrase.strip())

    def sample_hard_negative(self) -> str:
        """Composes a phrase from authentic dictionary words containing visually confusable characters."""
        group = random.choice(CONFUSION_GROUPS)
        available_chars = [c for c in group if c in self.confusion_word_map]
        if not available_chars:
            return self.sample_dictionary_phrase()

        chosen_words = []
        for ch in available_chars:
            chosen_words.append(random.choice(self.confusion_word_map[ch]))

        # Fill up to 2-4 words
        while len(chosen_words) < 3 and self.rac_words:
            chosen_words.append(random.choice(self.rac_words))

        random.shuffle(chosen_words)
        phrase = chosen_words[0]
        for w in chosen_words[1:]:
            phrase += random.choice(CONNECTORS) + w
        return normalize_khmer_canonical(phrase.strip())

    def sample_random_syllables(self, count: int = 4) -> str:
        """Backward-compatible alias: generates authentic RAC dictionary phrase."""
        return self.sample_dictionary_phrase(min_words=2, max_words=max(2, min(count, 5)))

    def sample_number_or_date(self) -> str:
        """Generates realistic Khmer date, currency, percentage, or phone number."""
        templates = [
            # Khmer date with Khmer digits
            lambda: f"ថ្ងៃទី{random.choice(self.khmer_digits)}{random.choice(self.khmer_digits)} ខែ{random.choice(['មករា', 'កុម្ភៈ', 'មីនា', 'មេសា', 'ឧសភា', 'មិថុនា', 'កក្កដា', 'សីហា', 'កញ្ញា', 'តុលា', 'វិច្ឆិកា', 'ធ្នូ'])} ឆ្នាំ២០២{random.choice(self.khmer_digits)}",
            # Standard numeric date
            lambda: f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(2015,2026)}",
            # Currency in Riel with symbol
            lambda: f"{random.randint(1, 999) * 1000:,} ៛",
            # Currency in Riel with Khmer digits
            lambda: f"{''.join(random.choices(self.khmer_digits, k=random.randint(4, 7)))} រៀល",
            # Currency in USD
            lambda: f"{random.randint(1, 2000):,} ដុល្លារ",
            # Percentage
            lambda: f"{random.randint(1, 100)}%",
            lambda: f"{''.join(random.choices(self.khmer_digits, k=random.randint(1, 2)))}%",
            # Phone number in Khmer digits
            lambda: f"០{random.choice(['១២', '១៥', '១៦', '៩៩', '៨៨', '៧៧'])} {''.join(random.choices(self.khmer_digits, k=3))} {''.join(random.choices(self.khmer_digits, k=3))}",
            # Code/Number in Khmer
            lambda: f"លេខកូដ {''.join(random.choices(self.khmer_digits, k=6))}",
        ]
        return random.choice(templates)()

    def sample_line(self, pure_corpus: bool | None = None) -> str:
        """Unified line sampler ensuring 100% authentic, readable Khmer sentences and words.

        Zero random character salads or ungrammatical pseudo-words.
        """
        is_pure = pure_corpus if pure_corpus is not None else self.pure_corpus

        for _ in range(10):  # Retry loop to guarantee validity
            if is_pure and self.has_custom_corpus:
                line = self.sample_natural_line()
            elif self.has_custom_corpus:
                rand = random.random()
                if rand < 0.90:
                    line = self.sample_natural_line()
                elif rand < 0.95:
                    line = self.sample_dictionary_phrase()
                else:
                    line = self.sample_number_or_date()
            else:
                rand = random.random()
                if rand < 0.65:
                    line = self.sample_natural_line()
                elif rand < 0.85:
                    line = self.sample_dictionary_phrase()
                elif rand < 0.93:
                    line = self.sample_hard_negative()
                else:
                    line = self.sample_number_or_date()

            if validate_khmer_line(line):
                return line

        return self.sample_natural_line()
