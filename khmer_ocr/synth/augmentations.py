"""Document Image Augmentations & Degradations for OCR Robustness."""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


def add_gaussian_noise(image: Image.Image, mean: float = 0.0, sigma: float = 12.0) -> Image.Image:
    """Adds subtle Gaussian sensor noise to image."""
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(mean, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def add_salt_and_pepper_noise(image: Image.Image, amount: float = 0.005) -> Image.Image:
    """Adds salt and pepper noise simulating dust or scanner artifacts."""
    arr = np.array(image)
    mask = np.random.random(arr.shape[:2])
    # Salt (white pixels)
    arr[mask < (amount / 2)] = 255
    # Pepper (black pixels)
    arr[(mask >= (amount / 2)) & (mask < amount)] = 0
    return Image.fromarray(arr)


def apply_blur(image: Image.Image, radius: float | None = None) -> Image.Image:
    """Applies slight Gaussian blur simulating out-of-focus camera or scanner."""
    if radius is None:
        radius = random.uniform(0.3, 1.2)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_ink_bleed(image: Image.Image) -> Image.Image:
    """Simulates ink bleeding into cheap paper (MinFilter / MaxFilter)."""
    if random.random() < 0.5:
        # Ink spread (darker)
        return image.filter(ImageFilter.MinFilter(size=3))
    else:
        # Faded print (lighter)
        return image.filter(ImageFilter.MaxFilter(size=3))


def add_shadow_gradient(image: Image.Image) -> Image.Image:
    """Adds uneven lighting or shadow gradient across the image."""
    arr = np.array(image, dtype=np.float32)
    h, w = arr.shape[:2]
    # Linear gradient from one side to another
    start_intensity = random.uniform(0.6, 0.9)
    end_intensity = 1.0
    if random.random() < 0.5:
        start_intensity, end_intensity = end_intensity, start_intensity

    gradient = np.linspace(start_intensity, end_intensity, w).reshape(1, w)
    if len(arr.shape) == 3:
        gradient = np.repeat(gradient[:, :, np.newaxis], arr.shape[2], axis=2)

    shaded = np.clip(arr * gradient, 0, 255).astype(np.uint8)
    return Image.fromarray(shaded)


def create_paper_background(width: int, height: int, style: str = "random") -> Image.Image:
    """Generates authentic paper textures: clean, aged parchment, or newspaper."""
    if style == "random":
        style = random.choice(["clean", "parchment", "fiber", "aged"])

    if style == "clean_doc":
        return Image.new("RGB", (width, height), (255, 255, 255))

    if style == "clean":
        val = random.randint(245, 255)
        return Image.new("RGB", (width, height), (val, val, val))
    elif style == "parchment":
        # Warm yellowish/beige
        r = random.randint(235, 250)
        g = random.randint(225, 240)
        b = random.randint(205, 220)
        base = Image.new("RGB", (width, height), (r, g, b))
        return add_gaussian_noise(base, sigma=4.0)
    elif style == "aged":
        # Darker aged document
        r = random.randint(220, 240)
        g = random.randint(215, 235)
        b = random.randint(190, 210)
        base = Image.new("RGB", (width, height), (r, g, b))
        return add_shadow_gradient(add_gaussian_noise(base, sigma=6.0))
    else:
        # White paper with slight scanner grain
        base = Image.new("RGB", (width, height), (250, 250, 250))
        return add_gaussian_noise(base, sigma=5.0)


def apply_random_augmentations(
    image: Image.Image,
    p_blur: float = 0.35,
    p_noise: float = 0.40,
    p_shadow: float = 0.30,
    p_contrast: float = 0.40,
) -> Image.Image:
    """Applies a randomized pipeline of document augmentations."""
    if random.random() < p_blur:
        image = apply_blur(image)

    if random.random() < p_noise:
        if random.random() < 0.5:
            image = add_gaussian_noise(image, sigma=random.uniform(4.0, 10.0))
        else:
            image = add_salt_and_pepper_noise(image, amount=random.uniform(0.001, 0.005))

    if random.random() < p_shadow:
        image = add_shadow_gradient(image)

    if random.random() < p_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(random.uniform(0.85, 1.25))

    return image
