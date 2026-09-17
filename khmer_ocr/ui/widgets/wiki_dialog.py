"""Interactive Khmer Wikipedia Data Harvester Dialog (km.wikipedia.org)."""

from pathlib import Path
from typing import List
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSpinBox, QPushButton, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QRadioButton, QButtonGroup, QFileDialog, QApplication
)
import qtawesome as qta
from ...dataset.wikipedia_loader import KhmerWikipediaLoader, CURATED_TOPICS


class WikiHarvestWorker(QThread):
    """Background worker thread for fetching Wikipedia articles without freezing GUI."""

    progress_updated = pyqtSignal(int, int, str, int)  # (current, total, current_title, total_sentences)
    harvest_finished = pyqtSignal(list)                # list of sentences
    harvest_error = pyqtSignal(str)

    def __init__(self, mode: str, param: str, max_articles: int):
        super().__init__()
        self.mode = mode
        self.param = param
        self.max_articles = max_articles
        self.loader = KhmerWikipediaLoader()
        self._is_cancelled = False

    def stop(self):
        self._is_cancelled = True

    def run(self):
        try:
            sentences: List[str] = []

            def on_progress(cur, tot, msg):
                if self._is_cancelled:
                    raise InterruptedError("Cancelled by user")
                self.progress_updated.emit(cur, tot, msg, len(sentences))

            if self.mode == "curated":
                cats = None if self.param == "All Domains" else [self.param]
                sentences = self.loader.harvest_from_topics(
                    categories=cats,
                    max_articles=self.max_articles,
                    progress_callback=on_progress,
                )
            elif self.mode == "random":
                sentences = self.loader.harvest_from_random(
                    count=self.max_articles,
                    progress_callback=on_progress,
                )
            elif self.mode == "search":
                sentences = self.loader.harvest_from_search(
                    query=self.param,
                    max_articles=self.max_articles,
                    progress_callback=on_progress,
                )

            self.harvest_finished.emit(sentences)
        except InterruptedError:
            self.harvest_error.emit("Harvesting cancelled by user.")
        except Exception as e:
            self.harvest_error.emit(str(e))


class KhmerWikipediaDialog(QDialog):
    """Dialog for streaming authentic Khmer Wikipedia text directly into synthetic OCR generation."""

    corpus_ready = pyqtSignal(list)  # Emits list of authentic Khmer sentences

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Khmer Wikipedia Massive Data Harvester (km.wikipedia.org)")
        self.resize(880, 620)
        self.setMinimumSize(740, 520)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")

        self.harvested_sentences: List[str] = []
        self.worker: WikiHarvestWorker | None = None

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # 1. Top Setup: Mode & Options
        setup_group = QGroupBox("Wikipedia Harvest Source")
        setup_layout = QVBoxLayout(setup_group)

        # Radio Mode Selector
        mode_row = QHBoxLayout()
        self.radio_curated = QRadioButton("Curated Topics")
        self.radio_curated.setChecked(True)
        self.radio_curated.toggled.connect(self._on_mode_toggled)

        self.radio_random = QRadioButton("Random Articles Explorer")
        self.radio_random.toggled.connect(self._on_mode_toggled)

        self.radio_search = QRadioButton("Custom Keyword Search")
        self.radio_search.toggled.connect(self._on_mode_toggled)

        mode_row.addWidget(self.radio_curated)
        mode_row.addWidget(self.radio_random)
        mode_row.addWidget(self.radio_search)
        mode_row.addStretch()
        setup_layout.addLayout(mode_row)

        # Controls Row
        ctrl_row = QHBoxLayout()

        # Category dropdown for curated mode
        self.cat_label = QLabel("Domain Topic:")
        self.cat_combo = QComboBox()
        self.cat_combo.addItem("All Domains")
        for cat in CURATED_TOPICS.keys():
            self.cat_combo.addItem(cat)
        ctrl_row.addWidget(self.cat_label)
        ctrl_row.addWidget(self.cat_combo)

        # Search box for custom search mode
        self.search_label = QLabel("Khmer Keyword:")
        self.search_label.setVisible(False)
        self.search_edit = QLineEdit("ប្រវត្តិសាស្ត្រខ្មែរ")
        self.search_edit.setVisible(False)
        ctrl_row.addWidget(self.search_label)
        ctrl_row.addWidget(self.search_edit)

        ctrl_row.addSpacing(16)
        ctrl_row.addWidget(QLabel("Max Articles:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(2, 60)
        self.count_spin.setValue(12)
        ctrl_row.addWidget(self.count_spin)

        self.btn_harvest = QPushButton("Harvest Articles")
        self.btn_harvest.setObjectName("btn_primary")
        self.btn_harvest.setIcon(qta.icon("fa5b.wikipedia-w", color="#11111b"))
        self.btn_harvest.clicked.connect(self._start_harvest)
        ctrl_row.addWidget(self.btn_harvest)

        self.btn_cancel = QPushButton("Stop")
        self.btn_cancel.setObjectName("btn_danger")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._stop_harvest)
        ctrl_row.addWidget(self.btn_cancel)

        setup_layout.addLayout(ctrl_row)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        setup_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to harvest authentic text from km.wikipedia.org")
        self.status_label.setStyleSheet("color: #a6adc8; font-size: 11.5px;")
        setup_layout.addWidget(self.status_label)

        main_layout.addWidget(setup_group)

        # 2. Middle: Live Harvested Sentences Table
        table_group = QGroupBox("Harvested Khmer Sentence Samples")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["#", "Authentic Khmer Sentence", "Chars"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(2, 65)
        self.table.setStyleSheet(
            "font-family: 'Noto Sans Khmer', sans-serif; font-size: 13px; background-color: #181825;"
        )
        table_layout.addWidget(self.table)

        self.summary_label = QLabel("0 sentences harvested")
        self.summary_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
        table_layout.addWidget(self.summary_label)

        main_layout.addWidget(table_group)

        # 3. Bottom Actions
        bottom_bar = QHBoxLayout()

        self.btn_save_corpus = QPushButton("Save to .txt File...")
        self.btn_save_corpus.setIcon(qta.icon("fa5s.save", color="#cdd6f4"))
        self.btn_save_corpus.clicked.connect(self._save_corpus_to_file)
        bottom_bar.addWidget(self.btn_save_corpus)

        bottom_bar.addStretch()

        self.btn_use_corpus = QPushButton("Use as Active Synthetic Corpus")
        self.btn_use_corpus.setObjectName("btn_success")
        self.btn_use_corpus.setIcon(qta.icon("fa5s.check-circle", color="#11111b"))
        self.btn_use_corpus.clicked.connect(self._apply_active_corpus)
        bottom_bar.addWidget(self.btn_use_corpus)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        bottom_bar.addWidget(btn_close)

        main_layout.addLayout(bottom_bar)

    def _on_mode_toggled(self):
        is_curated = self.radio_curated.isChecked()
        is_search = self.radio_search.isChecked()

        self.cat_label.setVisible(is_curated)
        self.cat_combo.setVisible(is_curated)

        self.search_label.setVisible(is_search)
        self.search_edit.setVisible(is_search)

    def _start_harvest(self):
        if self.radio_curated.isChecked():
            mode = "curated"
            param = self.cat_combo.currentText()
        elif self.radio_random.isChecked():
            mode = "random"
            param = ""
        else:
            mode = "search"
            param = self.search_edit.text().strip()
            if not param:
                QMessageBox.warning(self, "Warning", "Please enter a search query!")
                return

        max_articles = self.count_spin.value()

        self.btn_harvest.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(5)
        self.status_label.setText("Connecting to km.wikipedia.org API...")

        self.worker = WikiHarvestWorker(mode=mode, param=param, max_articles=max_articles)
        self.worker.progress_updated.connect(self._on_progress_updated)
        self.worker.harvest_finished.connect(self._on_harvest_finished)
        self.worker.harvest_error.connect(self._on_harvest_error)
        self.worker.start()

    def _stop_harvest(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Cancelling request...")

    def _on_progress_updated(self, cur: int, tot: int, msg: str, sentence_count: int):
        pct = int((cur / max(1, tot)) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"[{cur}/{tot}] {msg} (Collected {sentence_count} sentences)")

    def _on_harvest_finished(self, sentences: List[str]):
        self.harvested_sentences = sentences
        self.btn_harvest.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100)

        total = len(sentences)
        self.status_label.setText(f"Harvest complete! Successfully extracted {total} authentic Khmer sentences.")
        self.summary_label.setText(f"Total: {total:,} sentences ready for dataset synthesis")

        # Populate preview table with up to 300 rows
        preview_count = min(300, total)
        self.table.setRowCount(preview_count)
        for i in range(preview_count):
            text = sentences[i]
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(text))
            self.table.setItem(i, 2, QTableWidgetItem(str(len(text))))

        if total > 0:
            QMessageBox.information(
                self, "Harvest Complete",
                f"Successfully harvested {total:,} Khmer sentences from Wikipedia!\n"
                f"You can now set this as the active corpus for synthetic data generation or save to file."
            )
        else:
            QMessageBox.warning(
                self, "No Sentences",
                "Could not extract clean sentences from the specified articles. Please try another category or keyword."
            )

    def _on_harvest_error(self, err: str):
        self.btn_harvest.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.status_label.setText(f"Error: {err}")
        QMessageBox.critical(self, "Harvest Error", f"Failed to harvest from Wikipedia:\n{err}")

    def _apply_active_corpus(self):
        if not self.harvested_sentences:
            QMessageBox.warning(self, "Warning", "Please harvest sentences first before applying!")
            return
        self.corpus_ready.emit(self.harvested_sentences)
        self.accept()

    def _save_corpus_to_file(self):
        if not self.harvested_sentences:
            QMessageBox.warning(self, "Warning", "No harvested sentences to save!")
            return

        f, _ = QFileDialog.getSaveFileName(
            self, "Save Wikipedia Corpus", "wikipedia_khmer_corpus.txt", "Text Files (*.txt)"
        )
        if f:
            loader = KhmerWikipediaLoader()
            count = loader.save_corpus(self.harvested_sentences, f)
            QMessageBox.information(
                self, "Saved", f"Successfully saved {count:,} sentences to:\n{f}"
            )
