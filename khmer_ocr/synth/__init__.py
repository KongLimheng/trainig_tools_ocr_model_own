"""Khmer Synthetic Data Generation Package."""

from .fonts import KhmerFontManager
from .corpus_sampler import KhmerCorpusSampler
from .augmentations import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_blur,
    apply_ink_bleed,
    add_shadow_gradient,
    create_paper_background,
    apply_random_augmentations,
)
from .renderer import KhmerTextRenderer
from .generator import KhmerDatasetGenerator

__all__ = [
    "KhmerFontManager",
    "KhmerCorpusSampler",
    "KhmerTextRenderer",
    "KhmerDatasetGenerator",
    "create_paper_background",
    "apply_random_augmentations",
]
