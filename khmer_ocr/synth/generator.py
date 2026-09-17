"""Batch Synthetic Khmer Dataset Generator."""

import os
import random
from pathlib import Path
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor
from .fonts import KhmerFontManager
from .renderer import KhmerTextRenderer
from .corpus_sampler import KhmerCorpusSampler, validate_khmer_line


class KhmerDatasetGenerator:
    """Generates synthetic OCR datasets with multi-threading and progress reporting."""

    def __init__(
        self,
        font_manager: KhmerFontManager | None = None,
        corpus_sampler: KhmerCorpusSampler | None = None,
        target_height: int = 48,
        pure_corpus: bool = False,
    ):
        self.font_mgr = font_manager or KhmerFontManager()
        self.sampler = corpus_sampler or KhmerCorpusSampler(pure_corpus=pure_corpus)
        self.renderer = KhmerTextRenderer(font_manager=self.font_mgr)
        self.target_height = target_height
        self._stop_requested = False

    def stop(self) -> None:
        """Signals to cancel ongoing generation."""
        self._stop_requested = True

    def generate_batch(
        self,
        output_dir: str | Path,
        num_samples: int = 1000,
        selected_fonts: list[str] | None = None,
        augment: bool = True,
        bg_style: str = "random",
        val_ratio: float = 0.0,
        clean_ratio: float = 0.40,
        pure_corpus: bool | None = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Path:
        """Generates a dataset of synthetic Khmer line images and ground truth labels.

        Args:
            output_dir: Path where images and labels.txt will be saved.
            num_samples: Number of images to generate.
            selected_fonts: List of font names to choose from (None for all).
            augment: Whether to apply image degradations.
            bg_style: Paper background texture ('clean', 'clean_doc', 'parchment', 'aged', 'random').
            val_ratio: Ratio of samples reserved for validation (e.g. 0.1 for 10% val split).
            clean_ratio: Ratio of samples rendered with high-contrast clean document print.
            progress_callback: Function called with (current_index, total_count, message).

        Returns:
            Path to the primary labels file (train/labels.txt if split, else labels.txt).
        """
        self._stop_requested = False
        output_path = Path(output_dir)

        is_split = val_ratio > 0.0
        if is_split:
            train_dir = output_path / "train"
            val_dir = output_path / "val"
            train_img_dir = train_dir / "images"
            val_img_dir = val_dir / "images"
            train_img_dir.mkdir(parents=True, exist_ok=True)
            val_img_dir.mkdir(parents=True, exist_ok=True)
            train_labels_file = train_dir / "labels.txt"
            val_labels_file = val_dir / "labels.txt"
        else:
            images_dir = output_path / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_file = output_path / "labels.txt"

        available_fonts = selected_fonts or self.font_mgr.get_font_names()
        if not available_fonts:
            raise RuntimeError("No Khmer fonts available to generate data!")

        train_records = []
        val_records = []

        for idx in range(num_samples):
            if self._stop_requested:
                break

            text = self.sampler.sample_line(pure_corpus=pure_corpus)
            if not validate_khmer_line(text):
                for _ in range(5):
                    text = self.sampler.sample_line(pure_corpus=pure_corpus)
                    if validate_khmer_line(text):
                        break
            font_name = random.choice(available_fonts)
            font_size = random.randint(28, 44)

            # High-contrast clean document vs augmented styling
            if clean_ratio > 0.0 and random.random() < clean_ratio:
                sample_bg = "clean_doc"
                sample_aug = False
            else:
                sample_bg = bg_style
                sample_aug = augment

            # Determine train vs validation assignment
            is_val_sample = is_split and (random.random() < val_ratio)
            target_img_dir = val_img_dir if is_val_sample else (
                train_img_dir if is_split else images_dir)

            try:
                img, label = self.renderer.render_line(
                    text=text,
                    font_name=font_name,
                    font_size=font_size,
                    target_height=self.target_height,
                    bg_style=sample_bg,
                    augment=sample_aug,
                )

                img_filename = f"sample_{idx:07d}.jpg"
                img_path = target_img_dir / img_filename
                img.save(img_path, "JPEG", quality=random.randint(88, 98))

                record_line = f"{img_filename}\t{label}\n"
                if is_val_sample:
                    val_records.append(record_line)
                else:
                    train_records.append(record_line)

            except Exception:
                continue

            if progress_callback and (idx % 20 == 0 or idx == num_samples - 1):
                progress_callback(idx + 1, num_samples,
                                  f"Generated {idx + 1}/{num_samples} images...")

        if is_split:
            with open(train_labels_file, "w", encoding="utf-8") as f:
                f.writelines(train_records)
            with open(val_labels_file, "w", encoding="utf-8") as f:
                f.writelines(val_records)
            primary_labels = train_labels_file
            summary_msg = f"Finished! {len(train_records)} train + {len(val_records)} val samples saved to {output_path}"
        else:
            with open(labels_file, "w", encoding="utf-8") as f:
                f.writelines(train_records)
            primary_labels = labels_file
            summary_msg = f"Finished! {len(train_records)} samples saved to {output_path}"

        if progress_callback:
            progress_callback(len(train_records) +
                              len(val_records), num_samples, summary_msg)

        return primary_labels
