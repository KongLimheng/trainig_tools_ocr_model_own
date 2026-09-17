"""Multi-Column & Reading-Order Document Layout Analysis (DLA)."""

from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image


class KhmerLayoutAnalyzer:
    """Detects multi-column reading order and document column boundaries."""

    def __init__(self, min_gutter_width: int = 18, min_col_width: int = 80):
        self.min_gutter_width = min_gutter_width
        self.min_col_width = min_col_width

    def detect_columns(self, binary_img: np.ndarray) -> List[Tuple[int, int]]:
        """Detects column boundaries (x_start, x_end) using vertical projection profile."""
        h, w = binary_img.shape
        # Vertical projection: sum of text pixels along columns
        v_proj = np.sum(binary_img > 0, axis=0)

        # Smooth projection to avoid single-pixel noise
        kernel = np.ones(5) / 5.0
        smoothed = np.convolve(v_proj, kernel, mode="same")

        # Find continuous zero or near-zero regions (gutters)
        threshold = np.max(smoothed) * 0.05
        is_empty = smoothed < threshold

        gutters = []
        in_gutter = False
        start = 0

        for x in range(w):
            if is_empty[x] and not in_gutter:
                in_gutter = True
                start = x
            elif not is_empty[x] and in_gutter:
                in_gutter = False
                width = x - start
                # Only consider gutters that don't touch image outer borders
                if width >= self.min_gutter_width and start > 20 and x < (w - 20):
                    gutters.append((start, x))

        if not gutters:
            # Single column document
            return [(0, w)]

        # Convert gutters into column intervals
        columns = []
        col_start = 0
        for g_start, g_end in gutters:
            gutter_mid = (g_start + g_end) // 2
            if (gutter_mid - col_start) >= self.min_col_width:
                columns.append((col_start, gutter_mid))
                col_start = gutter_mid

        if (w - col_start) >= self.min_col_width:
            columns.append((col_start, w))

        return columns if columns else [(0, w)]

    def sort_reading_order(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        image_width: int,
        columns: List[Tuple[int, int]] | None = None,
    ) -> List[int]:
        """Returns sorted indices of bounding boxes in multi-column reading order.
        All lines in Column 1 (top to bottom), then Column 2 (top to bottom), etc.
        """
        if not bboxes:
            return []

        if columns is None or len(columns) <= 1:
            # Single column: sort by y
            indexed = list(enumerate(bboxes))
            indexed.sort(key=lambda item: item[1][1])
            return [idx for idx, _ in indexed]

        # Assign each box to the column containing its center x
        col_buckets: List[List[Tuple[int, Tuple[int, int, int, int]]]] = [[] for _ in columns]

        for orig_idx, (x, y, w, h) in enumerate(bboxes):
            center_x = x + w / 2
            assigned = False
            for col_idx, (c_start, c_end) in enumerate(columns):
                if c_start <= center_x < c_end:
                    col_buckets[col_idx].append((orig_idx, (x, y, w, h)))
                    assigned = True
                    break
            if not assigned:
                # Assign to closest column
                col_buckets[0].append((orig_idx, (x, y, w, h)))

        # Sort each column top to bottom
        sorted_indices = []
        for bucket in col_buckets:
            bucket.sort(key=lambda item: item[1][1])
            for orig_idx, _ in bucket:
                sorted_indices.append(orig_idx)

        return sorted_indices
