"""Dataset Vocabulary Coverage Booster for Rare Khmer Glyphs."""

import random
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional
from ..vocab.char_map import KhmerCharMap
from ..synth.renderer import KhmerTextRenderer
from ..normalizer.unicode_rules import (
    KHMER_COENG,
    KHMER_CONSONANTS,
    KHMER_DEP_VOWELS,
    KHMER_DIACRITICS,
    normalize_khmer_canonical,
)

# Rare Khmer characters that often suffer from 0% or low training representation
RARE_TARGET_CHARACTERS = [
    # Rare independent vowels
    "ឦ", "ឧ", "ឩ", "ឪ", "ឫ", "ឬ", "ឭ", "ឮ", "ឯ", "ឰ", "ឱ", "ឳ",
    # Rare diacritics & shifters
    "៑", "៏", "៓", "៎", "៌", "័", "៉", "៊",
    # Rare consonants
    "ឍ", "ឌ", "ឡ", "ឋ", "ឈ", "ណ",
    # Symbols & digits
    "៘", "៙", "៚", "៛", "០", "៨", "៩",
]


class KhmerCoverageBooster:
    """Detects underrepresented Khmer glyphs in datasets and synthesizes targeted samples."""

    def __init__(self, renderer: KhmerTextRenderer | None = None):
        self.renderer = renderer or KhmerTextRenderer()
        self.char_map = KhmerCharMap()

    def audit_coverage(self, labels_file: str | Path) -> Dict[str, int]:
        """Counts character frequencies in a dataset labels file."""
        char_counts = Counter()
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    for ch in parts[1]:
                        char_counts[ch] += 1
        return dict(char_counts)

    def generate_targeted_phrases(self, target_char: str, count: int = 5) -> List[str]:
        """Generates realistic synthetic phrases intentionally containing the target character."""
        phrases = []
        consonants = list(KHMER_CONSONANTS)
        vowels = list(KHMER_DEP_VOWELS)

        for _ in range(count):
            if target_char in RARE_TARGET_CHARACTERS[:12]:
                # Target is an independent vowel (e.g. ឪពុក, ឬស, ឫស្សី, ឱកាស, ឯកឧត្តម)
                sample_templates = [
                    f"ឯកឧត្តម និង លោកជំទាវ {target_char}",
                    f"ឱកាស នៃ {target_char}កាសម្ព័ន្ធ",
                    f"ឪពុក ម្តាយ {target_char}សី",
                    f"សេចក្តីសម្រេច ស្ដីពី {target_char}ទ្វារ",
                    f"សាកលវិទ្យាល័យ {target_char}រម្យ",
                ]
                phrases.append(random.choice(sample_templates))
            else:
                # Syllable containing target
                base = target_char if target_char in consonants else random.choice(consonants)
                sub = f"{KHMER_COENG}{random.choice(consonants)}" if random.random() < 0.4 else ""
                vowel = target_char if target_char in vowels else random.choice(vowels)
                dia = target_char if target_char in KHMER_DIACRITICS else ""
                cluster = normalize_khmer_canonical(f"{base}{sub}{vowel}{dia}")
                phrase = f"ការប្រើប្រាស់ {cluster} ក្នុងឯកសាររដ្ឋបាល"
                phrases.append(phrase)

        return phrases

    def boost_dataset(
        self,
        dataset_dir: str | Path,
        min_threshold: int = 50,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, int]:
        """Audits dataset, identifies underrepresented glyphs, and appends targeted samples."""
        ds_path = Path(dataset_dir)
        labels_file = ds_path / "labels.txt"
        images_dir = ds_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        if not labels_file.exists():
            raise FileNotFoundError(f"labels.txt not found in {dataset_dir}")

        char_counts = self.audit_coverage(labels_file)

        underrepresented = [
            ch for ch in RARE_TARGET_CHARACTERS
            if char_counts.get(ch, 0) < min_threshold
        ]

        if not underrepresented:
            return {"boosted_count": 0, "samples_added": 0}

        new_records = []
        start_idx = 900000 + random.randint(100, 9000)
        total_targets = len(underrepresented)

        for t_idx, target_ch in enumerate(underrepresented):
            deficit = min_threshold - char_counts.get(target_ch, 0)
            samples_to_add = min(deficit, 15)  # Add up to 15 targeted lines per rare char

            phrases = self.generate_targeted_phrases(target_ch, count=samples_to_add)
            for phrase in phrases:
                start_idx += 1
                img, label = self.renderer.render_line(phrase, augment=True)
                img_name = f"boost_{start_idx:07d}.jpg"
                img.save(images_dir / img_name, "JPEG", quality=90)
                new_records.append(f"{img_name}\t{label}\n")

            if progress_callback:
                progress_callback(t_idx + 1, total_targets, f"Boosted '{target_ch}' (+{samples_to_add} samples)")

        # Append new samples to labels.txt
        with open(labels_file, "a", encoding="utf-8") as f:
            f.writelines(new_records)

        return {
            "boosted_characters": len(underrepresented),
            "samples_added": len(new_records),
        }
