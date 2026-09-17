"""Dataset Auditing & Quality Control for Khmer OCR."""

from collections import Counter
from pathlib import Path
from PIL import Image
from ..normalizer.zwsp_cleaner import has_invisible_chars, count_invisible_chars
from ..normalizer.unicode_rules import normalize_khmer_canonical
from ..vocab.char_map import KhmerCharMap


def audit_dataset(
    data_dir: str | Path,
    labels_filename: str = "labels.txt",
    char_map: KhmerCharMap | None = None,
) -> dict:
    """Performs a comprehensive audit of an OCR dataset:
    - Total samples
    - OOV (Out-of-Vocabulary) characters
    - Invisible character counts (ZWSP, ZWNJ, BOM)
    - Canonical normalization discrepancies
    - Aspect ratio distribution
    - Missing image files
    """
    data_path = Path(data_dir)
    labels_path = data_path / labels_filename
    images_dir = data_path / "images"

    if not labels_path.exists():
        # Try direct file if data_dir was pointing directly to labels.txt
        if data_path.is_file():
            labels_path = data_path
            images_dir = data_path.parent / "images"
            if not images_dir.exists():
                images_dir = data_path.parent
        else:
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

    vocab = char_map or KhmerCharMap()
    known_chars = set(vocab.char_list)

    total_samples = 0
    missing_images = 0
    invisible_char_samples = 0
    non_canonical_samples = 0
    char_freq: Counter[str] = Counter()
    oov_chars: Counter[str] = Counter()
    invisible_counts: Counter[str] = Counter()
    aspect_ratios = []

    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue

            parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
            if len(parts) < 2:
                continue

            img_name, label = parts[0], parts[1]
            total_samples += 1

            # Check image existence
            img_file = images_dir / img_name
            if not img_file.exists():
                img_file = data_path / img_name

            if img_file.exists():
                try:
                    with Image.open(img_file) as im:
                        w, h = im.size
                        aspect_ratios.append(w / max(1, h))
                except Exception:
                    pass
            else:
                missing_images += 1

            # Check invisible characters
            if has_invisible_chars(label):
                invisible_char_samples += 1
                for k, v in count_invisible_chars(label).items():
                    invisible_counts[k] += v

            # Check canonical ordering
            canonical = normalize_khmer_canonical(label)
            if canonical != label:
                non_canonical_samples += 1

            # Check vocabulary
            for ch in label:
                char_freq[ch] += 1
                if ch not in known_chars:
                    oov_chars[ch] += 1

    return {
        "total_samples": total_samples,
        "missing_images": missing_images,
        "invisible_char_samples": invisible_char_samples,
        "invisible_counts": dict(invisible_counts),
        "non_canonical_samples": non_canonical_samples,
        "unique_characters": len(char_freq),
        "oov_character_count": sum(oov_chars.values()),
        "top_oov_characters": oov_chars.most_common(10),
        "avg_aspect_ratio": sum(aspect_ratios) / max(1, len(aspect_ratios)),
        "char_frequencies": char_freq.most_common(30),
    }
