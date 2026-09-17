"""Tab 4: Evaluation & Khmer Diagnostics Studio with Vector Icons."""

from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QFileDialog, QProgressBar, QMessageBox,
    QTableWidget, QTableWidgetItem, QSplitter, QHeaderView, QCheckBox
)
import qtawesome as qta
from ..widgets.image_viewer import KhmerImageViewer
from ..workers.eval_worker import EvaluationWorker


class TabEvalStudio(QWidget):
    """Evaluation Studio Tab with Khmer-specific metrics and error diagnostics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: EvaluationWorker | None = None
        self.all_sample_details: list[dict] = []
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Top Bar: Checkpoint and Dataset
        top_group = QGroupBox("Model Evaluation Setup")
        top_layout = QHBoxLayout(top_group)

        top_layout.addWidget(QLabel("Checkpoint (.pth):"))
        self.ckpt_edit = QLineEdit(str(Path.cwd() / "checkpoints" / "best_model.pth"))
        top_layout.addWidget(self.ckpt_edit)
        btn_browse_ckpt = QPushButton()
        btn_browse_ckpt.setIcon(qta.icon("fa5s.folder-open", color="#cdd6f4"))
        btn_browse_ckpt.clicked.connect(self._browse_ckpt)
        top_layout.addWidget(btn_browse_ckpt)

        top_layout.addWidget(QLabel("Test Dataset Dir:"))
        self.test_dir_edit = QLineEdit(str(Path.cwd() / "data" / "synthetic_train"))
        top_layout.addWidget(self.test_dir_edit)
        btn_browse_test = QPushButton()
        btn_browse_test.setIcon(qta.icon("fa5s.folder-open", color="#cdd6f4"))
        btn_browse_test.clicked.connect(self._browse_test)
        top_layout.addWidget(btn_browse_test)

        self.btn_run_eval = QPushButton("Run Evaluation")
        self.btn_run_eval.setObjectName("btn_primary")
        self.btn_run_eval.setIcon(qta.icon("fa5s.chart-line", color="#11111b"))
        self.btn_run_eval.clicked.connect(self._run_eval)
        top_layout.addWidget(self.btn_run_eval)

        main_layout.addWidget(top_group)

        # Metric Summary Cards
        cards_group = QGroupBox("Khmer OCR Performance Metrics")
        cards_layout = QHBoxLayout(cards_group)

        self.card_norm_cer = self._create_card("Normalized CER", "0.0%", "#a6e3a1")
        self.card_std_cer = self._create_card("Standard CER", "0.0%", "#89b4fa")
        self.card_scer = self._create_card("Syllable Cluster ER", "0.0%", "#fab387")
        self.card_exact = self._create_card("Exact Match", "0.0%", "#cba6f7")
        self.card_total = self._create_card("Total Tested", "0", "#cdd6f4")

        cards_layout.addWidget(self.card_norm_cer)
        cards_layout.addWidget(self.card_std_cer)
        cards_layout.addWidget(self.card_scer)
        cards_layout.addWidget(self.card_exact)
        cards_layout.addWidget(self.card_total)
        main_layout.addWidget(cards_group)

        # Splitter: Detailed Error Samples & Glyph Confusion Table
        splitter = QSplitter(Qt.Horizontal)

        # Left: Sample Inspection Table
        samples_group = QGroupBox("Sample Transcription Predictions")
        samples_layout = QVBoxLayout(samples_group)

        filter_layout = QHBoxLayout()
        self.chk_only_errors = QCheckBox("Show Mismatches Only")
        self.chk_only_errors.stateChanged.connect(self._filter_samples)
        filter_layout.addWidget(self.chk_only_errors)
        filter_layout.addStretch()
        samples_layout.addLayout(filter_layout)

        self.samples_table = QTableWidget()
        self.samples_table.setColumnCount(4)
        self.samples_table.setHorizontalHeaderLabels(["Image", "Ground Truth", "Prediction", "Result"])
        self.samples_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.samples_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.samples_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.samples_table.itemSelectionChanged.connect(self._on_sample_selected)
        samples_layout.addWidget(self.samples_table)
        splitter.addWidget(samples_group)

        # Right: Image Preview & Glyph Confusion Matrix
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Selected Sample Inspection")
        preview_layout = QVBoxLayout(preview_group)
        self.image_viewer = KhmerImageViewer()
        preview_layout.addWidget(self.image_viewer)
        right_layout.addWidget(preview_group, stretch=1)

        conf_group = QGroupBox("Top Confused Khmer Glyphs")
        conf_layout = QVBoxLayout(conf_group)
        self.confusion_table = QTableWidget()
        self.confusion_table.setColumnCount(3)
        self.confusion_table.setHorizontalHeaderLabels(["Ground Truth", "Mispredicted As", "Count"])
        self.confusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        conf_layout.addWidget(self.confusion_table)
        right_layout.addWidget(conf_group, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([580, 420])
        main_layout.addWidget(splitter)

        # Bottom Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

    def _create_card(self, title: str, val: str, color: str) -> QWidget:
        card = QWidget()
        card.setStyleSheet("background-color: #181825; border: 1px solid #313244; border-radius: 6px; padding: 6px;")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 4, 8, 4)
        t_lbl = QLabel(title)
        t_lbl.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: 600;")
        layout.addWidget(t_lbl)
        v_lbl = QLabel(val)
        v_lbl.setObjectName("val_label")
        v_lbl.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        layout.addWidget(v_lbl)
        return card

    def _update_card_val(self, card: QWidget, val: str):
        lbl = card.findChild(QLabel, "val_label")
        if lbl:
            lbl.setText(val)

    def _browse_ckpt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "PyTorch Models (*.pth *.pt)")
        if f:
            self.ckpt_edit.setText(f)

    def _browse_test(self):
        d = QFileDialog.getExistingDirectory(self, "Select Test Dataset Directory")
        if d:
            self.test_dir_edit.setText(d)

    def _run_eval(self):
        ckpt_path = Path(self.ckpt_edit.text().strip())
        test_path = Path(self.test_dir_edit.text().strip())

        if not ckpt_path.exists():
            QMessageBox.warning(self, "Warning", f"Checkpoint not found at: {ckpt_path}")
            return
        if not (test_path / "labels.txt").exists():
            QMessageBox.warning(self, "Warning", f"labels.txt not found in {test_path}!")
            return

        self.worker = EvaluationWorker(checkpoint_path=ckpt_path, dataset_dir=test_path)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.eval_finished.connect(self._on_eval_finished)
        self.worker.eval_error.connect(self._on_eval_error)

        self.btn_run_eval.setEnabled(False)
        self.progress_bar.setValue(0)
        self.worker.start()

    def _on_progress(self, current: int, total: int):
        pct = int((current / max(1, total)) * 100)
        self.progress_bar.setValue(pct)

    def _on_eval_finished(self, metrics: dict, sample_details: list, top_confusions: list):
        self.btn_run_eval.setEnabled(True)
        self.progress_bar.setValue(100)
        self.all_sample_details = sample_details

        self._update_card_val(self.card_norm_cer, f"{metrics['normalized_cer'] * 100:.2f}%")
        self._update_card_val(self.card_std_cer, f"{metrics['cer'] * 100:.2f}%")
        self._update_card_val(self.card_scer, f"{metrics['syllable_cer'] * 100:.2f}%")
        self._update_card_val(self.card_exact, f"{metrics['exact_match'] * 100:.2f}%")
        self._update_card_val(self.card_total, str(metrics['total_samples']))

        self._filter_samples()

        self.confusion_table.setRowCount(len(top_confusions))
        for idx, conf in enumerate(top_confusions):
            self.confusion_table.setItem(idx, 0, QTableWidgetItem(conf["ground_truth"]))
            self.confusion_table.setItem(idx, 1, QTableWidgetItem(conf["predicted"]))
            self.confusion_table.setItem(idx, 2, QTableWidgetItem(str(conf["count"])))

    def _filter_samples(self):
        only_errors = self.chk_only_errors.isChecked()
        filtered = [s for s in self.all_sample_details if (not only_errors or not s["match"])]

        display_cnt = min(500, len(filtered))
        self.samples_table.setRowCount(display_cnt)

        for idx in range(display_cnt):
            item = filtered[idx]
            fname = Path(item["path"]).name
            self.samples_table.setItem(idx, 0, QTableWidgetItem(fname))
            self.samples_table.setItem(idx, 1, QTableWidgetItem(item["target"]))
            self.samples_table.setItem(idx, 2, QTableWidgetItem(item["pred"]))
            status_item = QTableWidgetItem("Exact" if item["match"] else "Error")
            status_item.setForeground(Qt.green if item["match"] else Qt.red)
            self.samples_table.setItem(idx, 3, status_item)

    def _on_sample_selected(self):
        row = self.samples_table.currentRow()
        if row < 0 or row >= len(self.all_sample_details):
            return

        only_errors = self.chk_only_errors.isChecked()
        filtered = [s for s in self.all_sample_details if (not only_errors or not s["match"])]
        if row < len(filtered):
            path = filtered[row]["path"]
            if Path(path).exists():
                self.image_viewer.set_image_path(path)

    def _on_eval_error(self, err_msg: str):
        self.btn_run_eval.setEnabled(True)
        QMessageBox.critical(self, "Evaluation Error", f"Failed to run evaluation:\n{err_msg}")
