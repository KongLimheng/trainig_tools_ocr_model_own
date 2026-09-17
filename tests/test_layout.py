"""Unit tests for Multi-Column & Reading-Order Document Layout Analysis."""

import numpy as np
from khmer_ocr.pipeline.layout_analyzer import KhmerLayoutAnalyzer


def test_single_column_layout():
    analyzer = KhmerLayoutAnalyzer()
    # Continuous text across full width
    img = np.zeros((400, 600), dtype=np.uint8)
    img[50:80, 50:550] = 255
    img[120:150, 50:550] = 255

    cols = analyzer.detect_columns(img)
    assert len(cols) == 1
    assert cols[0] == (0, 600)

    # Reading order top to bottom
    bboxes = [(50, 120, 500, 30), (50, 50, 500, 30)]
    order = analyzer.sort_reading_order(bboxes, 600, cols)
    # Box at y=50 should come before box at y=120
    assert order == [1, 0]


def test_multi_column_layout():
    analyzer = KhmerLayoutAnalyzer(min_gutter_width=30, min_col_width=80)
    # Two-column page with distinct empty vertical gutter between x=280 and x=320
    img = np.zeros((500, 600), dtype=np.uint8)
    # Column 1
    img[50:100, 40:270] = 255
    img[120:180, 40:270] = 255
    # Gutter is empty between 270 and 330 (width 60)
    # Column 2
    img[50:100, 330:560] = 255
    img[120:180, 330:560] = 255

    cols = analyzer.detect_columns(img)
    assert len(cols) == 2

    # Boxes:
    # 0: Col 2, line 1 (x=330, y=50)
    # 1: Col 1, line 1 (x=40, y=50)
    # 2: Col 1, line 2 (x=40, y=120)
    # 3: Col 2, line 2 (x=330, y=120)
    bboxes = [
        (330, 50, 200, 30),
        (40, 50, 200, 30),
        (40, 120, 200, 30),
        (330, 120, 200, 30),
    ]

    order = analyzer.sort_reading_order(bboxes, 600, cols)
    # Reading order must read all lines of Col 1 first (indices 1, 2), then Col 2 (indices 0, 3)
    assert order == [1, 2, 0, 3]
