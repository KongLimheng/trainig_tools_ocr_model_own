"""Main Window Shell for KhmerOCR Studio with Vector Icons."""

import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTabWidget, QStatusBar, QApplication
)
import qtawesome as qta
import torch
from .styles import DARK_THEME_QSS
from .tabs.tab_synth import TabSynthStudio
from .tabs.tab_audit import TabDatasetAudit
from .tabs.tab_train import TabTrainingStudio
from .tabs.tab_eval import TabEvalStudio
from .tabs.tab_infer import TabInferAndExport
from .widgets.gpu_monitor import LiveTelemetryWidget


class KhmerOCRStudioMainWindow(QMainWindow):
    """Main Application Window for KhmerOCR Studio."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("KhmerOCR Studio - AI OCR Training & Tooling Suite")
        self.resize(1280, 850)
        self.setMinimumSize(1024, 680)

        # Apply Dark Theme
        self.setStyleSheet(DARK_THEME_QSS)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(14, 10, 14, 10)
        main_layout.setSpacing(8)

        # App Header Banner
        header = self._create_header()
        main_layout.addWidget(header)

        # Main Tab Widget
        self.tab_widget = QTabWidget()
        self.tab_synth = TabSynthStudio(self)
        self.tab_audit = TabDatasetAudit(self)
        self.tab_train = TabTrainingStudio(self)
        self.tab_eval = TabEvalStudio(self)
        self.tab_infer = TabInferAndExport(self)

        # Tabs with crisp FontAwesome vector icons (zero tofu boxes!)
        icon_synth = qta.icon("fa5s.paint-brush", color="#89b4fa")
        icon_audit = qta.icon("fa5s.database", color="#a6e3a1")
        icon_train = qta.icon("fa5s.brain", color="#fab387")
        icon_eval = qta.icon("fa5s.chart-bar", color="#f38ba8")
        icon_infer = qta.icon("fa5s.bolt", color="#cba6f7")

        self.tab_widget.addTab(self.tab_synth, icon_synth, "Synthetic Studio")
        self.tab_widget.addTab(self.tab_audit, icon_audit, "Dataset Auditor")
        self.tab_widget.addTab(self.tab_train, icon_train, "Model Training")
        self.tab_widget.addTab(self.tab_eval, icon_eval, "Evaluation && Diagnostics")
        self.tab_widget.addTab(self.tab_infer, icon_infer, "Inference && Export")

        main_layout.addWidget(self.tab_widget)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._setup_status_bar()

    def _create_header(self) -> QWidget:
        header = QWidget()
        header.setStyleSheet("background-color: #181825; border: 1px solid #313244; border-radius: 8px; padding: 4px 10px;")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        # Vector Brand Icon
        brand_icon_lbl = QLabel()
        brand_icon_lbl.setPixmap(qta.icon("fa5s.layer-group", color="#89b4fa").pixmap(28, 28))
        layout.addWidget(brand_icon_lbl)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title_lbl = QLabel("KhmerOCR Studio")
        title_lbl.setStyleSheet("font-size: 17px; font-weight: bold; color: #89b4fa;")
        sub_lbl = QLabel("Professional AI OCR Training, Linguistic Auditing & Document Tooling for Khmer Script")
        sub_lbl.setStyleSheet("font-size: 11px; color: #a6adc8;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        layout.addLayout(title_box)

        layout.addStretch()

        # Live Hardware & Process Telemetry Center
        self.telemetry_widget = LiveTelemetryWidget(self)
        layout.addWidget(self.telemetry_widget)

        return header

    def set_process_status(self, text: str, state: str = "idle"):
        """Propagates task state to header telemetry."""
        if hasattr(self, "telemetry_widget"):
            self.telemetry_widget.set_process_status(text, state)

    def _setup_status_bar(self):
        self.status_bar.showMessage("KhmerOCR Studio ready. Powered by uv & PyTorch.")


def run_studio_app():
    """Entry point to launch the PyQt5 GUI."""
    from PyQt5.QtCore import qInstallMessageHandler

    def _qt_message_filter(msg_type, context, message):
        # Cleanly suppress benign platform/thread socket notifier warnings
        if "QSocketNotifier" in message:
            return
        if "Ignoring XDG_SESSION_TYPE=wayland" in message:
            return
        sys.stderr.write(f"{message}\n")

    qInstallMessageHandler(_qt_message_filter)

    app = QApplication(sys.argv)
    window = KhmerOCRStudioMainWindow()
    window.show()
    sys.exit(app.exec_())
