"""Unit tests for Document Preprocessing: Deskew, Shadow Removal & Enhancement."""

import numpy as np
from PIL import Image, ImageDraw
from khmer_ocr.pipeline.preprocessor import KhmerDocumentPreprocessor


def _create_skewed_test_image(angle: float = 5.0) -> Image.Image:
    """Creates a synthetic image with horizontal black stripes rotated by `angle`."""
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw several horizontal text-like black bars
    for y in range(40, 260, 35):
        draw.rectangle([40, y, 360, y + 14], fill=(0, 0, 0))

    # Rotate by angle
    rotated = img.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return rotated


def test_deskew_detection():
    preprocessor = KhmerDocumentPreprocessor(max_skew_angle=15.0, angle_step=0.5)
    true_angle = 5.0
    skewed = _create_skewed_test_image(angle=true_angle)

    gray = np.array(skewed.convert("L"))
    detected_angle = preprocessor.detect_skew_angle(gray)

    # Detected angle should be very close to the true angle (within 1.0 degree)
    assert abs(detected_angle - true_angle) <= 1.0

    # Test deskew method
    deskewed, returned_angle = preprocessor.deskew(skewed)
    assert deskewed.size == skewed.size
    assert abs(returned_angle - true_angle) <= 1.0


def test_shadow_removal():
    preprocessor = KhmerDocumentPreprocessor()
    # Create image with artificial gradient lighting (shadow)
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    arr = np.array(img, dtype=np.float32)
    # Add dark gradient across width
    gradient = np.linspace(0.3, 1.0, 300).reshape(1, 300, 1)
    shadowed_arr = (arr * gradient).astype(np.uint8)
    shadowed_img = Image.fromarray(shadowed_arr)

    cleaned = preprocessor.remove_shadows(shadowed_img)
    assert cleaned.size == shadowed_img.size
    # Shadowed left region should now be substantially brighter
    cleaned_arr = np.array(cleaned)
    assert cleaned_arr[:, :50, :].mean() > shadowed_arr[:, :50, :].mean()


def test_preprocess_document_pipeline():
    preprocessor = KhmerDocumentPreprocessor()
    test_img = _create_skewed_test_image(angle=4.0)
    result, metadata = preprocessor.preprocess_document(
        test_img, auto_deskew=True, remove_shadow=True, enhance=True
    )
    assert result.size == test_img.size
    assert "skew_angle_deg" in metadata
    assert metadata["was_shadow_removed"] is True
    assert metadata["was_enhanced"] is True
