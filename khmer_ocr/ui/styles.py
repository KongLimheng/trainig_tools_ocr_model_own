"""Modern Dark Slate Theme (QSS) for Khmer OCR Studio."""

DARK_THEME_QSS = """
/* Global Window and Base Styles */
QMainWindow, QDialog, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans Khmer", sans-serif;
    font-size: 13px;
}

/* Header & Toolbars */
QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    padding: 6px;
    spacing: 8px;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #313244;
    background-color: #181825;
    border-radius: 8px;
    top: -1px;
}

QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    padding: 8px 14px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
    font-weight: 600;
    font-size: 12.5px;
    border: 1px solid #313244;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #252538;
    color: #89b4fa;
    border-top: 3px solid #89b4fa;
}

QTabBar::tab:hover:!selected {
    background-color: #202030;
    color: #cdd6f4;
}

/* Scroll Areas */
QScrollArea {
    background-color: transparent;
    border: none;
}

/* Group Boxes & Cards */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: 600;
    color: #89b4fa;
    background-color: #252538;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #252538;
}

/* Buttons */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #585b70;
}

QPushButton:pressed {
    background-color: #585b70;
}

QPushButton:disabled {
    background-color: #181825;
    color: #6c7086;
    border-color: #313244;
}

/* Primary Action Button */
QPushButton#btn_primary {
    background-color: #89b4fa;
    color: #11111b;
    border: none;
}

QPushButton#btn_primary:hover {
    background-color: #b4befe;
}

QPushButton#btn_primary:pressed {
    background-color: #74c7ec;
}

/* Success Button */
QPushButton#btn_success {
    background-color: #a6e3a1;
    color: #11111b;
    border: none;
}

QPushButton#btn_success:hover {
    background-color: #94e2d5;
}

/* Danger Button */
QPushButton#btn_danger {
    background-color: #f38ba8;
    color: #11111b;
    border: none;
}

QPushButton#btn_danger:hover {
    background-color: #eba0ac;
}

/* Inputs & Form Controls */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: #45475a;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #89b4fa;
}

/* Tables & Trees */
QTableWidget, QTreeWidget {
    background-color: #181825;
    alternate-background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 6px;
    gridline-color: #313244;
}

QHeaderView::section {
    background-color: #252538;
    color: #a6adc8;
    padding: 6px;
    border: 1px solid #313244;
    font-weight: 600;
}

/* Sliders */
QSlider::groove:horizontal {
    height: 6px;
    background: #313244;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #89b4fa;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #cdd6f4;
    border: 2px solid #89b4fa;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

/* Progress Bars */
QProgressBar {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    height: 16px;
    text-align: center;
    color: #cdd6f4;
    font-weight: 600;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    border-radius: 5px;
}

/* Status Bar */
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
}

/* Scrollbars */
QScrollBar:vertical {
    background: #181825;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #313244;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #45475a;
}

QScrollBar:horizontal {
    background: #181825;
    height: 8px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #313244;
    min-width: 20px;
    border-radius: 4px;
}
"""
