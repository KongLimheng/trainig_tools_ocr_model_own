"""Khmer Document Line Segmenter with Subscript-Safe Morphological Slicing."""

from typing import Tuple, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def merge_co_linear_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Merges bounding boxes that lie on the same horizontal text line."""
    if not boxes:
        return []

    # Sort boxes primarily by vertical position y
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    merged_lines: List[List[Tuple[int, int, int, int]]] = []

    for box in sorted_boxes:
        bx, by, bw, bh = box
        matched = False

        for line in merged_lines:
            # Check vertical overlap with line
            line_y_min = min(b[1] for b in line)
            line_y_max = max(b[1] + b[3] for b in line)
            line_h = line_y_max - line_y_min

            overlap_y = min(by + bh, line_y_max) - max(by, line_y_min)
            if overlap_y > 0.4 * min(bh, line_h):
                line.append(box)
                matched = True
                break

        if not matched:
            merged_lines.append([box])

    # For each line group, calculate bounding box spanning all words in the line
    result_boxes = []
    for line in merged_lines:
        min_x = min(b[0] for b in line)
        min_y = min(b[1] for b in line)
        max_x = max(b[0] + b[2] for b in line)
        max_y = max(b[1] + b[3] for b in line)
        result_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

    # Sort lines from top to bottom
    result_boxes.sort(key=lambda b: b[1])
    return result_boxes


class KhmerDocumentLineSegmenter:
    """Detects and extracts text lines from full-page document images."""

    def __init__(
        self,
        min_line_height: int = 15,
        min_line_width: int = 40,
        subscript_pad_bottom: int = 6,
        subscript_pad_top: int = 4,
        horizontal_kernel_width: int = 25,
    ):
        self.min_line_height = min_line_height
        self.min_line_width = min_line_width
        self.subscript_pad_bottom = subscript_pad_bottom
        self.subscript_pad_top = subscript_pad_top
        self.horizontal_kernel_width = horizontal_kernel_width

    def segment(
        self,
        image: Image.Image,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Image.Image], Image.Image]:
        """Segments a document image into ordered text lines.

        Returns:
            bboxes: List of (x, y, w, h) for each line.
            line_crops: List of cropped line PIL Images.
            annotated_image: PIL Image with colored bounding boxes and line numbers.
        """
        # Convert PIL to OpenCV format (grayscale)
        cv_img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape

        # 1. Binarize (invert so text is white, background is black)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 2. Morphological horizontal closing: connect characters along the line
        kernel_h = max(15, min(img_w // 30, self.horizontal_kernel_width))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_h, 3))
        dilated = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 3. Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w >= self.min_line_width and h >= self.min_line_height:
                raw_boxes.append((x, y, w, h))

        # 4. Fallback if no contours
        if not raw_boxes:
            raw_boxes = [(0, 0, img_w, img_h)]

        # 5. Merge horizontally co-linear boxes on the same line
        line_boxes = merge_co_linear_boxes(raw_boxes)

        # 6. Apply multi-column reading order analysis
        from .layout_analyzer import KhmerLayoutAnalyzer
        layout_analyzer = KhmerLayoutAnalyzer()
        columns = layout_analyzer.detect_columns(thresh)
        sorted_indices = layout_analyzer.sort_reading_order(line_boxes, img_w, columns)
        line_boxes = [line_boxes[i] for i in sorted_indices]

        # 7. Apply Khmer subscript & ascender padding with boundary clamping
        safe_boxes = []
        line_crops = []
        annotated = image.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)

        for idx, (x, y, w, h) in enumerate(line_boxes):
            # Pad top for diacritics and bottom for subscripts
            y1 = max(0, y - self.subscript_pad_top)
            y2 = min(img_h, y + h + self.subscript_pad_bottom)
            x1 = max(0, x - 4)
            x2 = min(img_w, x + w + 4)

            safe_boxes.append((x1, y1, x2 - x1, y2 - y1))

            # Crop line from original image
            crop = image.crop((x1, y1, x2, y2))
            line_crops.append(crop)

            # Draw visual bounding box and line index
            draw.rectangle([(x1, y1), (x2, y2)], outline=(59, 130, 246), width=2)
            draw.rectangle([(x1, y1 - 16), (x1 + 24, y1)], fill=(59, 130, 246))
            draw.text((x1 + 4, y1 - 16), str(idx + 1), fill=(255, 255, 255))

        return safe_boxes, line_crops, annotated
