"""Document Preprocessing: Auto-Deskew, Shadow Removal & Contrast Enhancement."""

from typing import Tuple
import numpy as np
import cv2
from PIL import Image


class KhmerDocumentPreprocessor:
    """Preprocesses raw document photos & scans for optimal OCR transcription."""

    def __init__(self, max_skew_angle: float = 20.0, angle_step: float = 0.5):
        self.max_skew_angle = max_skew_angle
        self.angle_step = angle_step

    def detect_skew_angle(self, gray: np.ndarray) -> float:
        """Finds tilt angle using Horizontal Projection Profile variance."""
        h, w = gray.shape
        # Downsample for fast angle calculation if image is large
        if w > 800:
            scale = 800.0 / w
            resized = cv2.resize(gray, (800, int(h * scale)))
        else:
            resized = gray

        # Binarize
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        best_score = -1.0
        best_angle = 0.0

        angles = np.arange(-self.max_skew_angle, self.max_skew_angle + self.angle_step, self.angle_step)
        center = (resized.shape[1] // 2, resized.shape[0] // 2)

        for angle in angles:
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                binary, rot_mat, (resized.shape[1], resized.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )

            # Horizontal projection (sum along width)
            proj = np.sum(rotated, axis=1)
            # Variance of projection: peaks are highest when text lines are horizontal
            score = np.var(proj)

            if score > best_score:
                best_score = score
                best_angle = angle

        return float(best_angle)

    def deskew(self, image: Image.Image, angle: float | None = None) -> Tuple[Image.Image, float]:
        """Straightens a rotated document image."""
        cv_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

        if angle is None:
            angle = self.detect_skew_angle(gray)

        if abs(angle) < 0.2:
            return image, 0.0

        h, w = gray.shape
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate with white border background
        rotated = cv2.warpAffine(
            cv_img, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )

        return Image.fromarray(rotated), angle

    def remove_shadows(self, image: Image.Image) -> Image.Image:
        """Removes uneven shadows and smartphone lighting gradients."""
        cv_img = np.array(image.convert("RGB"))

        # Process each color channel to preserve colored letterheads / stamps
        planes = cv2.split(cv_img)
        result_planes = []

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))

        for plane in planes:
            # Estimate background illumination with morphological opening
            bg = cv2.morphologyEx(plane, cv2.MORPH_OPEN, kernel)
            # Divide image by background
            diff = cv2.absdiff(plane, bg)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result = 255 - norm
            result_planes.append(result)

        cleaned = cv2.merge(result_planes)
        return Image.fromarray(cleaned)

    def enhance_contrast(self, image: Image.Image) -> Image.Image:
        """Applies adaptive contrast enhancement for faded prints."""
        gray = np.array(image.convert("L"))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return Image.fromarray(enhanced)

    def preprocess_document(
        self,
        image: Image.Image,
        auto_deskew: bool = True,
        remove_shadow: bool = True,
        enhance: bool = False,
    ) -> Tuple[Image.Image, dict]:
        """Complete preprocessing pipeline."""
        detected_angle = 0.0
        result = image

        if auto_deskew:
            result, detected_angle = self.deskew(result)

        if remove_shadow:
            result = self.remove_shadows(result)

        if enhance:
            result = self.enhance_contrast(result)

        metadata = {
            "skew_angle_deg": detected_angle,
            "was_shadow_removed": remove_shadow,
            "was_enhanced": enhance,
        }
        return result, metadata
