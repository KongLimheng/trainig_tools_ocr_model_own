"""Khmer Text Corpus & Synthetic Line Sampler.

Generates realistic Khmer lines including:
- Natural sentences and phrases
- Khmer and Arabic numbers, dates, currency, phone numbers
- Hard-negative confusion clusters for fine discrimination
- Orthographic pseudo-words for zero-shot generalization
"""

import random
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

# Visually confusable glyph groups for hard-negative training
CONFUSION_CONSONANT_GROUPS = [
    ["គ", "ត", "ភ"],
    ["ឈ", "ញ", "ញ្ញ"],
    ["ឋ", "យ"],
    ["ដ", "ឌ", "ឍ"],
    ["ផ", "ធ"],
    ["ព", "ឃ"],
    ["ប", "បា", "បៅ"],
    ["ល", "ស"],
    ["អ", "អា"],
]

CONFUSION_SUBSCRIPTS = [
    [f"{KHMER_COENG}ញ", f"{KHMER_COENG}្ឋ"],
    [f"{KHMER_COENG}ម", f"{KHMER_COENG}ន"],
    [f"{KHMER_COENG}ល", f"{KHMER_COENG}ខ"],
    [f"{KHMER_COENG}ជ", f"{KHMER_COENG}ញ"],
    [f"{KHMER_COENG}ត", f"{KHMER_COENG}គ"],
    [f"{KHMER_COENG}ដ", f"{KHMER_COENG}ឋ"],
]

CONFUSION_DIACRITICS = [
    ["់", "៏", "៍", "៌", "័"],
    ["ំ", "ះ"],
    ["៉", "៊"],
]


class KhmerCorpusSampler:
    """Generates varied text lines for synthetic OCR training."""

    def __init__(self, custom_texts: list[str] | None = None):
        self.corpus = list(custom_texts) if custom_texts else list(DEFAULT_KHMER_CORPUS)
        self.consonants = list(KHMER_CONSONANTS)
        self.dep_vowels = list(KHMER_DEP_VOWELS)
        self.indep_vowels = list(KHMER_INDEP_VOWELS)
        self.khmer_digits = list(KHMER_DIGITS)
        self.diacritics = list(KHMER_DIACRITICS)

    def sample_natural_line(self, min_words: int = 2, max_words: int = 8) -> str:
        """Samples a natural Khmer phrase or composite phrase."""
        line = random.choice(self.corpus)
        if random.random() < 0.4:
            line2 = random.choice(self.corpus)
            line = f"{line} {line2}" if random.random() < 0.5 else f"{line} និង{line2}"
        words = line.split()
        if len(words) > max_words:
            start = random.randint(0, len(words) - max_words)
            words = words[start : start + random.randint(min_words, max_words)]
        return " ".join(words)

    def sample_number_or_date(self) -> str:
        """Generates realistic Khmer date, currency, or ID number."""
        templates = [
            # Khmer date
            lambda: f"ថ្ងៃទី{random.choice(self.khmer_digits)}{random.choice(self.khmer_digits)} ខែ{random.choice(['មករា', 'កុម្ភៈ', 'មីនា', 'មេសា', 'ឧសភា', 'មិថុនា', 'កក្កដា', 'សីហា', 'កញ្ញា', 'តុលា', 'វិច្ឆិកា', 'ធ្នូ'])} ឆ្នាំ២០២{random.choice(self.khmer_digits)}",
            # Standard numeric date
            lambda: f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(2015,2026)}",
            # Currency in Riel
            lambda: f"{random.randint(1, 999) * 1000:,} ៛",
            # Currency in Khmer digits
            lambda: f"{''.join(random.choices(self.khmer_digits, k=random.randint(4, 7)))} រៀល",
            # Currency in USD
            lambda: f"${random.randint(1, 5000):,}.{random.randint(0, 99):02d}",
            # Phone number
            lambda: f"០{random.choice(['១២', '១៥', '១៦', '៩៩', '៨៨', '៧៧'])} {random.randint(100,999)} {random.randint(100,999)}",
            # ID number
            lambda: f"ID: {''.join(random.choices('0123456789', k=9))}",
        ]
        return random.choice(templates)()

    def sample_hard_negative(self, length: int = 4) -> str:
        """Constructs a line emphasizing visually confusable Khmer characters."""
        cluster_parts = []
        for _ in range(length):
            group = random.choice(CONFUSION_CONSONANT_GROUPS)
            base = random.choice(group)
            sub = ""
            if random.random() < 0.5:
                sub_pair = random.choice(CONFUSION_SUBSCRIPTS)
                sub = random.choice(sub_pair)
            vowel = random.choice(self.dep_vowels) if random.random() < 0.6 else ""
            diacritic = random.choice(random.choice(CONFUSION_DIACRITICS)) if random.random() < 0.4 else ""
            cluster = normalize_khmer_canonical(f"{base}{sub}{vowel}{diacritic}")
            cluster_parts.append(cluster)
        return "".join(cluster_parts)

    def sample_random_syllables(self, count: int = 5) -> str:
        """Generates random valid Khmer syllables for vocabulary coverage."""
        syllables = []
        for _ in range(count):
            base = random.choice(self.consonants)
            sub = f"{KHMER_COENG}{random.choice(self.consonants)}" if random.random() < 0.35 else ""
            vowel = random.choice(self.dep_vowels) if random.random() < 0.7 else ""
            diacritic = random.choice(self.diacritics) if random.random() < 0.25 else ""
            syllable = normalize_khmer_canonical(f"{base}{sub}{vowel}{diacritic}")
            syllables.append(syllable)
        return "".join(syllables)

    def sample_line(self) -> str:
        """Unified line sampler combining natural, numerical, and hard-negative lines."""
        rand = random.random()
        if rand < 0.65:
            return self.sample_natural_line()
        elif rand < 0.80:
            return self.sample_number_or_date()
        elif rand < 0.92:
            return self.sample_hard_negative()
        else:
            return self.sample_random_syllables()
