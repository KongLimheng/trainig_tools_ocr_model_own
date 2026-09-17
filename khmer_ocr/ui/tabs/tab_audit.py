"""Tab 2: Dataset Auditor, Normalizer & Format Exporter."""

import shutil
import csv
from pathlib import Path
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QSplitter, QHeaderView, QStackedWidget
)
import qtawesome as qta
from ..widgets.image_viewer import KhmerImageViewer
from ..widgets.annotator_canvas import DocumentAnnotatorWidget
from ...dataset.data_audit import audit_dataset
from ...normalizer import normalize_khmer_text


class TabDatasetAudit(QWidget):
    """Dataset Auditor & Unicode Normalizer Tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_labels_path: Path | None = None
        self.loaded_records: list[tuple[str, str]] = []
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Top Bar: Directory Selector & Action Buttons
        top_group = QGroupBox("Select OCR Dataset")
        top_layout = QHBoxLayout(top_group)

        top_layout.addWidget(QLabel("Dataset Directory:"))
        self.data_dir_edit = QLineEdit(str(Path.cwd() / "data" / "synthetic_train"))
        top_layout.addWidget(self.data_dir_edit)

        btn_browse = QPushButton("Browse...")
        btn_browse.setIcon(qta.icon("fa5s.folder-open", color="#cdd6f4"))
        btn_browse.clicked.connect(self._browse_dir)
        top_layout.addWidget(btn_browse)

        btn_audit = QPushButton("Run Audit")
        btn_audit.setObjectName("btn_primary")
        btn_audit.setIcon(qta.icon("fa5s.search", color="#11111b"))
        btn_audit.clicked.connect(self._run_audit)
        top_layout.addWidget(btn_audit)

        btn_fix = QPushButton("One-Click Fix")
        btn_fix.setObjectName("btn_success")
        btn_fix.setIcon(qta.icon("fa5s.magic", color="#11111b"))
        btn_fix.setToolTip("Sanitize invisible ZWSP and canonicalize Khmer sequences")
        btn_fix.clicked.connect(self._fix_dataset)
        top_layout.addWidget(btn_fix)

        main_layout.addWidget(top_group)

        # View Mode Selector Toolbar
        mode_toolbar = QWidget()
        mode_layout = QHBoxLayout(mode_toolbar)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.addWidget(QLabel("Audit Studio View:"))

        self.btn_mode_audit = QPushButton("Dataset Records && Audit Table")
        self.btn_mode_audit.setIcon(qta.icon("fa5s.table", color="#cdd6f4"))
        self.btn_mode_audit.setCheckable(True)
        self.btn_mode_audit.setChecked(True)
        self.btn_mode_audit.clicked.connect(lambda: self._switch_mode(0))
        mode_layout.addWidget(self.btn_mode_audit)

        self.btn_mode_annotator = QPushButton("Interactive Document Annotator")
        self.btn_mode_annotator.setIcon(qta.icon("fa5s.vector-square", color="#cdd6f4"))
        self.btn_mode_annotator.setCheckable(True)
        self.btn_mode_annotator.setChecked(False)
        self.btn_mode_annotator.clicked.connect(lambda: self._switch_mode(1))
        mode_layout.addWidget(self.btn_mode_annotator)

        mode_layout.addStretch()
        main_layout.addWidget(mode_toolbar)

        # Stacked view container
        self.stack = QStackedWidget()

        # Page 0: Audit & Explorer
        audit_page = QWidget()
        audit_page_layout = QVBoxLayout(audit_page)
        audit_page_layout.setContentsMargins(0, 0, 0, 0)
        audit_page_layout.setSpacing(8)

        # Summary Metric Cards
        self.summary_group = QGroupBox("Audit Diagnostics Summary")
        summary_layout = QHBoxLayout(self.summary_group)

        self.card_samples = self._create_card("Total Samples", "0", "#89b4fa")
        self.card_invisible = self._create_card("Invisible ZWSP", "0", "#fab387")
        self.card_canonical = self._create_card("Non-Canonical", "0", "#f38ba8")
        self.card_oov = self._create_card("OOV Characters", "0", "#f38ba8")
        self.card_unique = self._create_card("Unique Chars", "0", "#a6e3a1")

        summary_layout.addWidget(self.card_samples)
        summary_layout.addWidget(self.card_invisible)
        summary_layout.addWidget(self.card_canonical)
        summary_layout.addWidget(self.card_oov)
        summary_layout.addWidget(self.card_unique)
        audit_page_layout.addWidget(self.summary_group)

        # Splitter: Table Explorer and Image Preview
        splitter = QSplitter(Qt.Horizontal)

        # Left: Table Explorer
        table_group = QGroupBox("Dataset Records Explorer")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Filename", "Ground Truth Label", "Length"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        table_layout.addWidget(self.table)

        # Export Format Buttons Bar
        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("Export Dataset:"))

        btn_export_paddle = QPushButton("Export PaddleOCR (rec_gt.txt)")
        btn_export_paddle.setIcon(qta.icon("fa5s.file-export", color="#cdd6f4"))
        btn_export_paddle.clicked.connect(self._export_paddleocr)
        export_row.addWidget(btn_export_paddle)

        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.setIcon(qta.icon("fa5s.file-csv", color="#cdd6f4"))
        btn_export_csv.clicked.connect(self._export_csv)
        export_row.addWidget(btn_export_csv)

        export_row.addStretch()
        table_layout.addLayout(export_row)
        splitter.addWidget(table_group)

        # Right: Image Preview & Editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Sample Inspection && Editor")
        preview_layout = QVBoxLayout(preview_group)

        self.image_viewer = KhmerImageViewer()
        preview_layout.addWidget(self.image_viewer)

        self.edit_label = QLineEdit()
        self.edit_label.setStyleSheet("font-size: 14px; padding: 6px;")
        preview_layout.addWidget(QLabel("Editable Ground Truth:"))
        preview_layout.addWidget(self.edit_label)

        btn_save_row = QPushButton("Save Edited Label")
        btn_save_row.setIcon(qta.icon("fa5s.save", color="#cdd6f4"))
        btn_save_row.clicked.connect(self._save_edited_row)
        preview_layout.addWidget(btn_save_row)

        right_layout.addWidget(preview_group)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        audit_page_layout.addWidget(splitter)

        self.stack.addWidget(audit_page)

        # Page 1: Interactive Document Annotator
        annotator_page = QWidget()
        ann_layout = QVBoxLayout(annotator_page)
        ann_layout.setContentsMargins(0, 0, 0, 0)
        ann_layout.setSpacing(6)

        ann_top_bar = QWidget()
        ann_tb_layout = QHBoxLayout(ann_top_bar)
        ann_tb_layout.setContentsMargins(0, 0, 0, 0)

        ann_tb_layout.addWidget(QLabel("Document Scan Image:"))
        self.ann_img_edit = QLineEdit()
        self.ann_img_edit.setPlaceholderText("Select scanned page, phone photo, or receipt to annotate...")
        ann_tb_layout.addWidget(self.ann_img_edit)

        btn_browse_doc = QPushButton("Browse Document...")
        btn_browse_doc.setIcon(qta.icon("fa5s.image", color="#cdd6f4"))
        btn_browse_doc.clicked.connect(self._browse_annotator_image)
        ann_tb_layout.addWidget(btn_browse_doc)

        btn_save_ann_dataset = QPushButton("Save Crops to Dataset")
        btn_save_ann_dataset.setObjectName("btn_success")
        btn_save_ann_dataset.setIcon(qta.icon("fa5s.save", color="#11111b"))
        btn_save_ann_dataset.clicked.connect(self._save_annotator_crops_to_dataset)
        ann_tb_layout.addWidget(btn_save_ann_dataset)

        ann_layout.addWidget(ann_top_bar)

        self.annotator_canvas = DocumentAnnotatorWidget()
        ann_layout.addWidget(self.annotator_canvas)
        self.stack.addWidget(annotator_page)

        main_layout.addWidget(self.stack)

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

    def _browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if dir_path:
            self.data_dir_edit.setText(dir_path)

    def _run_audit(self):
        path = Path(self.data_dir_edit.text().strip())
        try:
            res = audit_dataset(path)
            self._update_card_val(self.card_samples, str(res["total_samples"]))
            self._update_card_val(self.card_invisible, str(res["invisible_char_samples"]))
            self._update_card_val(self.card_canonical, str(res["non_canonical_samples"]))
            self._update_card_val(self.card_oov, str(res["oov_character_count"]))
            self._update_card_val(self.card_unique, str(res["unique_characters"]))

            self._populate_table(path)
            QMessageBox.information(
                self, "Audit Complete",
                f"Audit finished successfully!\n"
                f"Total Samples: {res['total_samples']}\n"
                f"Samples with invisible ZWSP: {res['invisible_char_samples']}\n"
                f"Non-canonical sequences: {res['non_canonical_samples']}\n"
                f"OOV Characters: {res['oov_character_count']}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Audit Error", f"Failed to audit dataset:\n{e}")

    def _populate_table(self, data_path: Path):
        labels_file = data_path / "labels.txt"
        if not labels_file.exists():
            return

        self.current_labels_path = labels_file
        self.loaded_records.clear()
        self.table.setRowCount(0)

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
                if len(parts) >= 2:
                    self.loaded_records.append((parts[0], parts[1]))

        display_count = min(500, len(self.loaded_records))
        self.table.setRowCount(display_count)

        for i in range(display_count):
            fname, text = self.loaded_records[i]
            self.table.setItem(i, 0, QTableWidgetItem(fname))
            self.table.setItem(i, 1, QTableWidgetItem(text))
            self.table.setItem(i, 2, QTableWidgetItem(str(len(text))))

    def _on_row_selected(self):
        selected = self.table.currentRow()
        if selected < 0 or selected >= len(self.loaded_records):
            return

        fname, text = self.loaded_records[selected]
        self.edit_label.setText(text)

        data_dir = Path(self.data_dir_edit.text().strip())
        img_path = data_dir / "images" / fname
        if not img_path.exists():
            img_path = data_dir / fname

        if img_path.exists():
            self.image_viewer.set_image_path(str(img_path))
        else:
            self.image_viewer.clear()

    def _save_edited_row(self):
        selected = self.table.currentRow()
        if selected < 0 or not self.current_labels_path:
            return

        new_text = self.edit_label.text().strip()
        fname, _ = self.loaded_records[selected]
        self.loaded_records[selected] = (fname, new_text)
        self.table.setItem(selected, 1, QTableWidgetItem(new_text))
        self.table.setItem(selected, 2, QTableWidgetItem(str(len(new_text))))

        with open(self.current_labels_path, "w", encoding="utf-8") as f:
            for fn, tx in self.loaded_records:
                f.write(f"{fn}\t{tx}\n")

        QMessageBox.information(self, "Saved", "Label updated successfully!")

    def _fix_dataset(self):
        path = Path(self.data_dir_edit.text().strip())
        labels_file = path / "labels.txt"
        if not labels_file.exists():
            QMessageBox.warning(self, "Warning", "labels.txt not found in selected directory!")
            return

        reply = QMessageBox.question(
            self, "Confirm Normalization",
            "This will strip invisible characters (ZWSP) and normalize all labels into canonical Khmer order.\n"
            "A backup (labels.txt.bak) will be created. Proceed?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        bak_file = path / "labels.txt.bak"
        shutil.copy2(labels_file, bak_file)

        fixed_records = []
        fixed_count = 0

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t") if "\t" in line else line.split(maxsplit=1)
                if len(parts) >= 2:
                    fname, text = parts[0], parts[1]
                    norm_text = normalize_khmer_text(text, strip_zwsp=True)
                    if norm_text != text:
                        fixed_count += 1
                    fixed_records.append(f"{fname}\t{norm_text}\n")

        with open(labels_file, "w", encoding="utf-8") as f:
            f.writelines(fixed_records)

        self._run_audit()
        QMessageBox.information(
            self, "Dataset Fixed",
            f"Successfully normalized dataset!\n{fixed_count} labels modified and saved.\nBackup saved to {bak_file.name}."
        )

    def _export_paddleocr(self):
        data_dir = Path(self.data_dir_edit.text().strip())
        if not self.loaded_records:
            QMessageBox.warning(self, "Warning", "Please run audit or load a dataset first!")
            return

        out_path = data_dir / "rec_gt.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for fname, label in self.loaded_records:
                rel_path = f"images/{fname}" if (data_dir / "images" / fname).exists() else fname
                f.write(f"{rel_path}\t{label}\n")

        QMessageBox.information(self, "Exported", f"Exported PaddleOCR annotations to:\n{out_path}")

    def _export_csv(self):
        data_dir = Path(self.data_dir_edit.text().strip())
        if not self.loaded_records:
            QMessageBox.warning(self, "Warning", "Please run audit or load a dataset first!")
            return

        out_path = data_dir / "dataset.csv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "text", "length"])
            for fname, label in self.loaded_records:
                writer.writerow([fname, label, len(label)])

        QMessageBox.information(self, "Exported", f"Exported CSV dataset to:\n{out_path}")

    def _switch_mode(self, index: int):
        self.stack.setCurrentIndex(index)
        self.btn_mode_audit.setChecked(index == 0)
        self.btn_mode_annotator.setChecked(index == 1)

    def _browse_annotator_image(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Document Scan", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if f:
            self.ann_img_edit.setText(f)
            try:
                img = Image.open(f)
                self.annotator_canvas.load_document_image(img, auto_detect=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def _save_annotator_crops_to_dataset(self):
        records = self.annotator_canvas.get_annotated_records()
        if not records:
            QMessageBox.warning(self, "Warning", "No annotated lines found! Draw or auto-detect bounding boxes first.")
            return

        data_dir = Path(self.data_dir_edit.text().strip())
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_file = data_dir / "labels.txt"

        import time
        ts = int(time.time())
        saved_count = 0
        new_lines = []

        for idx, (crop_img, label_text) in enumerate(records):
            if not label_text.strip():
                continue
            norm_label = normalize_khmer_text(label_text, strip_zwsp=True)
            filename = f"ann_{ts}_{idx + 1:04d}.jpg"
            crop_img.convert("RGB").save(images_dir / filename, "JPEG", quality=95)
            new_lines.append(f"{filename}\t{norm_label}\n")
            saved_count += 1

        if not new_lines:
            QMessageBox.warning(self, "Warning", "None of the bounding boxes have text labels assigned!")
            return

        with open(labels_file, "a", encoding="utf-8") as f:
            f.writelines(new_lines)

        QMessageBox.information(
            self, "Saved to Dataset",
            f"Successfully saved {saved_count} annotated lines into:\n{images_dir}\nand appended to labels.txt!"
        )
        self._populate_table(data_dir)

