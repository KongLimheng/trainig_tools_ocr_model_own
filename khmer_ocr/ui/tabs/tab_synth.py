"""Tab 1: Synthetic Data Studio with Font Categorization & Corpus Importer."""

from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QPushButton, QFileDialog, QProgressBar, QMessageBox, QSplitter,
    QScrollArea, QApplication
)
import qtawesome as qta
from ..widgets.image_viewer import KhmerImageViewer
from ..workers.synth_worker import SyntheticGenWorker
from ...synth.fonts import (
    KhmerFontManager, STYLE_ALL, STYLE_KHATT, STYLE_MOUL, STYLE_CHRIENG
)
from ...synth.renderer import KhmerTextRenderer
from ...synth.corpus_sampler import KhmerCorpusSampler
from ...synth.generator import KhmerDatasetGenerator
from ...dataset.coverage_booster import KhmerCoverageBooster
from ...normalizer.syllable_parser import count_khmer_syllables
from ...normalizer import normalize_khmer_text
from ..widgets.wiki_dialog import KhmerWikipediaDialog


class TabSynthStudio(QWidget):
    """Synthetic Data Studio Tab with live HarfBuzz rendering and batch generation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.font_mgr = KhmerFontManager()
        self.sampler = KhmerCorpusSampler()
        self.renderer = KhmerTextRenderer(font_manager=self.font_mgr)
        self.worker: SyntheticGenWorker | None = None

        self._init_ui()
        self._update_preview()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # Left Scroll Area: Settings & Controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(4, 4, 8, 4)
        left_layout.setSpacing(10)

        # 1. Text Selection Group (Use && to prevent Qt accelerator underscore)
        text_group = QGroupBox("Text Source && Content")
        text_layout = QVBoxLayout(text_group)

        self.source_combo = QComboBox()
        self.source_combo.addItems([
            "Corpus Natural Line",
            "Khmer Wikipedia (km.wikipedia.org)",
            "Date / Number / Currency",
            "Hard-Negative Confusion Pairs",
            "Random Khmer Syllables",
            "Custom Khmer Text",
        ])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        text_layout.addWidget(QLabel("Generator Mode:"))
        text_layout.addWidget(self.source_combo)

        self.custom_text_input = QLineEdit("ព្រះរាជាណាចក្រកម្ពុជា")
        self.custom_text_input.setStyleSheet("font-size: 13.5px;")
        self.custom_text_input.textChanged.connect(self._update_preview)
        text_layout.addWidget(QLabel("Text Content:"))
        text_layout.addWidget(self.custom_text_input)

        btn_row1 = QHBoxLayout()
        self.btn_sample_next = QPushButton("Sample Text")
        self.btn_sample_next.setIcon(qta.icon("fa5s.random", color="#cdd6f4"))
        self.btn_sample_next.clicked.connect(self._sample_next_text)
        btn_row1.addWidget(self.btn_sample_next)

        self.btn_import_corpus = QPushButton("Import Corpus...")
        self.btn_import_corpus.setIcon(qta.icon("fa5s.file-upload", color="#cdd6f4"))
        self.btn_import_corpus.setToolTip("Import a custom .txt file with Khmer sentences")
        self.btn_import_corpus.clicked.connect(self._import_corpus_file)
        btn_row1.addWidget(self.btn_import_corpus)
        text_layout.addLayout(btn_row1)

        self.btn_wiki = QPushButton("Harvest Khmer Wikipedia (km.wikipedia.org)...")
        self.btn_wiki.setIcon(qta.icon("fa5b.wikipedia-w", color="#89b4fa"))
        self.btn_wiki.setToolTip("Harvest massive authentic Khmer text corpus from km.wikipedia.org")
        self.btn_wiki.clicked.connect(self._open_wikipedia_dialog)
        text_layout.addWidget(self.btn_wiki)

        left_layout.addWidget(text_group)

        # 2. Typography & Font Shaping Group (Use &&)
        font_group = QGroupBox("Typography && Font Shaping")
        font_layout = QVBoxLayout(font_group)

        font_filter_layout = QHBoxLayout()
        font_filter_layout.addWidget(QLabel("Font Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems([STYLE_ALL, STYLE_KHATT, STYLE_MOUL, STYLE_CHRIENG])
        self.style_combo.currentIndexChanged.connect(self._on_style_filter_changed)
        font_filter_layout.addWidget(self.style_combo)
        font_layout.addLayout(font_filter_layout)

        self.font_combo = QComboBox()
        self._populate_fonts(STYLE_ALL)
        self.font_combo.currentIndexChanged.connect(self._update_preview)
        font_layout.addWidget(QLabel("Selected Font:"))
        font_layout.addWidget(self.font_combo)

        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(16, 72)
        self.font_size_spin.setValue(36)
        self.font_size_spin.valueChanged.connect(self._update_preview)
        font_size_layout.addWidget(self.font_size_spin)

        font_size_layout.addWidget(QLabel("Height (px):"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(24, 96)
        self.height_spin.setValue(48)
        self.height_spin.valueChanged.connect(self._update_preview)
        font_size_layout.addWidget(self.height_spin)
        font_layout.addLayout(font_size_layout)

        left_layout.addWidget(font_group)

        # 3. Document Degradations Group (Use &&)
        aug_group = QGroupBox("Style && Document Degradations")
        aug_layout = QVBoxLayout(aug_group)

        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Paper Texture:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["clean", "parchment", "aged", "fiber"])
        self.bg_combo.currentIndexChanged.connect(self._update_preview)
        bg_row.addWidget(self.bg_combo)
        aug_layout.addLayout(bg_row)

        self.chk_augment = QCheckBox("Apply Noise, Blur && Sensor Grain")
        self.chk_augment.stateChanged.connect(self._update_preview)
        aug_layout.addWidget(self.chk_augment)

        left_layout.addWidget(aug_group)

        # 4. Batch Synthetic Dataset Generator (Use &&)
        batch_group = QGroupBox("Batch Synthetic Dataset Generator")
        batch_layout = QVBoxLayout(batch_group)

        batch_layout.addWidget(QLabel("Output Directory:"))
        dir_row = QHBoxLayout()
        self.out_dir_edit = QLineEdit(str(Path.cwd() / "data" / "synthetic_train"))
        dir_row.addWidget(self.out_dir_edit)
        self.btn_browse = QPushButton()
        self.btn_browse.setIcon(qta.icon("fa5s.folder-open", color="#cdd6f4"))
        self.btn_browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(self.btn_browse)
        batch_layout.addLayout(dir_row)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Sample Count:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(10, 100000)
        self.count_spin.setValue(1000)
        self.count_spin.setSingleStep(500)
        count_row.addWidget(self.count_spin)
        batch_layout.addLayout(count_row)

        btn_action_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Dataset")
        self.btn_generate.setObjectName("btn_primary")
        self.btn_generate.setIcon(qta.icon("fa5s.cogs", color="#11111b"))
        self.btn_generate.clicked.connect(self._start_batch_generation)
        btn_action_row.addWidget(self.btn_generate)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setIcon(qta.icon("fa5s.stop-circle", color="#11111b"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_generation)
        btn_action_row.addWidget(self.btn_stop)
        batch_layout.addLayout(btn_action_row)

        btn_boost_row = QHBoxLayout()
        self.btn_boost = QPushButton("Boost Rare Glyphs Coverage")
        self.btn_boost.setIcon(qta.icon("fa5s.shield-alt", color="#fab387"))
        self.btn_boost.setToolTip("Audits dataset for underrepresented Khmer vowels/subscripts and adds targeted samples")
        self.btn_boost.clicked.connect(self._boost_rare_glyphs)
        btn_boost_row.addWidget(self.btn_boost)
        batch_layout.addLayout(btn_boost_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        batch_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #a6adc8; font-size: 11.5px;")
        batch_layout.addWidget(self.status_label)

        left_layout.addWidget(batch_group)
        left_layout.addStretch()

        left_scroll.setWidget(left_content)
        splitter.addWidget(left_scroll)

        # Right Panel: Interactive Canvas Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        preview_group = QGroupBox("Live HarfBuzz Complex Text Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(8, 8, 8, 8)

        self.image_viewer = KhmerImageViewer()
        preview_layout.addWidget(self.image_viewer)

        self.metrics_label = QLabel("Label: None")
        self.metrics_label.setStyleSheet(
            "color: #89b4fa; font-size: 12.5px; padding: 6px; background-color: #181825; border-radius: 4px;"
        )
        self.metrics_label.setWordWrap(True)
        preview_layout.addWidget(self.metrics_label)

        right_layout.addWidget(preview_group)
        splitter.addWidget(right_panel)

        splitter.setSizes([380, 620])
        main_layout.addWidget(splitter)

    def _populate_fonts(self, style: str):
        self.font_combo.blockSignals(True)
        self.font_combo.clear()
        font_names = self.font_mgr.get_font_names(style)
        if not font_names:
            font_names = self.font_mgr.get_font_names(STYLE_ALL)
        self.font_combo.addItems(font_names)
        self.font_combo.blockSignals(False)

    def _on_style_filter_changed(self, idx: int):
        style = self.style_combo.currentText()
        self._populate_fonts(style)
        self._update_preview()

    def _on_source_changed(self, idx: int):
        if idx != 5:  # Not custom text
            self._sample_next_text()
        self._update_preview()

    def _sample_next_text(self):
        mode = self.source_combo.currentIndex()
        if mode in (0, 1):  # Corpus Natural Line or Wikipedia
            text = self.sampler.sample_natural_line()
        elif mode == 2:
            text = self.sampler.sample_number_or_date()
        elif mode == 3:
            text = self.sampler.sample_hard_negative()
        elif mode == 4:
            text = self.sampler.sample_random_syllables()
        else:
            text = self.custom_text_input.text()

        self.custom_text_input.setText(text)

    def _open_wikipedia_dialog(self):
        dialog = KhmerWikipediaDialog(self)
        dialog.corpus_ready.connect(self._on_wikipedia_corpus_loaded)
        dialog.exec_()

    def _on_wikipedia_corpus_loaded(self, sentences: list[str]):
        if not sentences:
            return
        self.sampler = KhmerCorpusSampler(custom_texts=sentences)
        self.renderer = KhmerTextRenderer(font_manager=self.font_mgr)
        self.source_combo.setCurrentIndex(1)  # Khmer Wikipedia
        self._sample_next_text()

    def _notify_app_status(self, text: str, state: str = "idle"):
        top = self.window()
        if hasattr(top, "set_process_status"):
            top.set_process_status(text, state)

    def _import_corpus_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Text Corpus File", "", "Text Files (*.txt *.csv *.tsv);;All Files (*)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                QMessageBox.warning(self, "Warning", "Selected file contains no readable text!")
                return

            clean_lines = [normalize_khmer_text(l, strip_zwsp=True) for l in lines]
            self.sampler = KhmerCorpusSampler(custom_texts=clean_lines)
            self.renderer = KhmerTextRenderer(font_manager=self.font_mgr)
            self._sample_next_text()
            QMessageBox.information(
                self, "Corpus Imported",
                f"Successfully imported {len(clean_lines)} lines into the generator!"
            )
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import corpus file:\n{e}")

    def _update_preview(self):
        text = self.custom_text_input.text()
        if not text:
            return

        font_name = self.font_combo.currentText()
        font_size = self.font_size_spin.value()
        target_h = self.height_spin.value()
        bg_style = self.bg_combo.currentText()
        augment = self.chk_augment.isChecked()

        try:
            img, label = self.renderer.render_line(
                text=text,
                font_name=font_name,
                font_size=font_size,
                target_height=target_h,
                bg_style=bg_style,
                augment=augment,
            )
            self.image_viewer.set_pil_image(img)
            syllables_cnt = count_khmer_syllables(label)
            w, h = img.size
            self.metrics_label.setText(
                f"Label: {label}\nChars: {len(label)} | Syllables: {syllables_cnt} | Dimensions: {w}x{h} px"
            )
        except Exception as e:
            self.metrics_label.setText(f"Rendering error: {e}")

    def _browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.out_dir_edit.setText(dir_path)

    def _start_batch_generation(self):
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Warning", "Please specify an output directory!")
            return

        num_samples = self.count_spin.value()
        bg_style = self.bg_combo.currentText()
        augment = self.chk_augment.isChecked()

        # Respect current font style filter if not 'All Styles'
        selected_style = self.style_combo.currentText()
        font_subset = self.font_mgr.get_font_names(selected_style) if selected_style != STYLE_ALL else None

        generator = KhmerDatasetGenerator(
            font_manager=self.font_mgr,
            corpus_sampler=self.sampler,
            target_height=self.height_spin.value(),
        )

        self.worker = SyntheticGenWorker(
            generator=generator,
            output_dir=out_dir,
            num_samples=num_samples,
            selected_fonts=font_subset,
            augment=augment,
            bg_style=bg_style,
        )

        self.worker.progress_updated.connect(self._on_gen_progress)
        self.worker.generation_finished.connect(self._on_gen_finished)
        self.worker.generation_error.connect(self._on_gen_error)

        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Generating dataset...")
        self._notify_app_status("Generating Synthetic Dataset...", "synth")
        self.worker.start()

    def _stop_generation(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Cancellation requested...")
            self._notify_app_status("Stopping Generation...", "synth")

    def _on_gen_progress(self, current: int, total: int, message: str):
        pct = int((current / max(1, total)) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)
        self._notify_app_status(f"Generating Synth ({current}/{total})", "synth")

    def _on_gen_finished(self, labels_path: str):
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Dataset generated successfully at: {labels_path}")
        self._notify_app_status("Idle", "idle")
        QMessageBox.information(self, "Success", f"Dataset successfully generated!\nLabels saved to:\n{labels_path}")

    def _on_gen_error(self, err_msg: str):
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText(f"Generation error: {err_msg}")
        self._notify_app_status("Idle", "idle")
        QMessageBox.critical(self, "Error", f"Generation failed:\n{err_msg}")

    def _boost_rare_glyphs(self):
        out_dir = self.out_dir_edit.text().strip()
        labels_file = Path(out_dir) / "labels.txt"
        if not labels_file.exists():
            QMessageBox.warning(
                self, "Warning",
                f"No existing dataset found in:\n{out_dir}\nPlease generate or select a dataset first!"
            )
            return

        self.status_label.setText("Auditing and boosting rare glyphs...")
        self._notify_app_status("Boosting Rare Glyphs...", "synth")
        self.progress_bar.setValue(20)
        QApplication.processEvents()

        try:
            booster = KhmerCoverageBooster(renderer=self.renderer)

            def on_progress(cur, total, msg):
                pct = 20 + int((cur / max(1, total)) * 75)
                self.progress_bar.setValue(pct)
                self.status_label.setText(msg)
                self._notify_app_status(f"Boosting Glyphs ({cur}/{total})", "synth")
                QApplication.processEvents()

            result = booster.boost_dataset(
                dataset_dir=out_dir,
                min_threshold=30,
                progress_callback=on_progress,
            )
            self.progress_bar.setValue(100)
            self._notify_app_status("Idle", "idle")
            boosted = result.get("boosted_characters", 0)
            added = result.get("samples_added", 0)

            if boosted == 0:
                self.status_label.setText("Coverage Audit Passed: All rare glyphs meet threshold!")
                QMessageBox.information(
                    self, "Coverage Excellent",
                    "All rare Khmer glyphs and independent vowels meet the target threshold (>= 30 samples)!"
                )
            else:
                self.status_label.setText(f"Boost Complete: +{added} samples added for {boosted} rare glyphs")
                QMessageBox.information(
                    self, "Coverage Boost Complete",
                    f"Successfully boosted dataset coverage!\n\n"
                    f"Rare Glyphs Reinforced: {boosted}\n"
                    f"Targeted Samples Added: {added}\n"
                    f"Appended to: {labels_file}"
                )
        except Exception as e:
            self.status_label.setText(f"Boost failed: {e}")
            QMessageBox.critical(self, "Error", f"Failed to boost rare glyphs:\n{e}")

