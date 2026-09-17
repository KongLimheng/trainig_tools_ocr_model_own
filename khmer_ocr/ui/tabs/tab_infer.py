"""Tab 5: End-to-End Document OCR Pipeline & ONNX Exporter."""

import time
from pathlib import Path
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QFileDialog, QMessageBox, QTextEdit,
    QApplication, QSplitter, QComboBox, QRadioButton, QButtonGroup,
    QCheckBox
)
import qtawesome as qta
import torch
import torchvision.transforms as T
from ..widgets.image_viewer import KhmerImageViewer
from ...models.crnn import KhmerCRNN
from ...vocab.char_map import KhmerCharMap
from ...export.onnx_exporter import export_to_onnx
from ...export.searchable_pdf import SearchablePDFExporter
from ...pipeline.segmenter import KhmerDocumentLineSegmenter
from ...pipeline.preprocessor import KhmerDocumentPreprocessor
from ...postprocess.word_segmenter import KhmerWordSegmenter
from ...postprocess.spell_corrector import KhmerSpellCorrector


class TabInferAndExport(QWidget):
    """Interactive Inference & ONNX Export Studio with Full-Document Page Segmentation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model: KhmerCRNN | None = None
        self.char_map: KhmerCharMap | None = None
        self.current_ckpt_path: str = ""
        self.current_img_path: str = ""
        self.segmenter = KhmerDocumentLineSegmenter()
        self.preprocessor = KhmerDocumentPreprocessor()
        self.word_segmenter = KhmerWordSegmenter()
        self.spell_corrector = KhmerSpellCorrector(segmenter=self.word_segmenter)
        self.pdf_exporter = SearchablePDFExporter()
        self.last_bboxes: list[tuple[int, int, int, int]] = []
        self.last_transcriptions: list[str] = []
        self.last_doc_image: Image.Image | None = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # 1. Top Setup: Checkpoint, Image, and Mode
        top_group = QGroupBox("OCR Playground Setup")
        top_layout = QVBoxLayout(top_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Model Checkpoint (.pth):"))
        self.ckpt_edit = QLineEdit(str(Path.cwd() / "checkpoints" / "best_model.pth"))
        row1.addWidget(self.ckpt_edit)
        btn_browse_ckpt = QPushButton()
        btn_browse_ckpt.setIcon(qta.icon("fa5s.folder-open", color="#cdd6f4"))
        btn_browse_ckpt.clicked.connect(self._browse_ckpt)
        row1.addWidget(btn_browse_ckpt)
        top_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Input Image:"))
        self.img_edit = QLineEdit()
        row2.addWidget(self.img_edit)
        btn_browse_img = QPushButton("Browse Image...")
        btn_browse_img.setIcon(qta.icon("fa5s.image", color="#cdd6f4"))
        btn_browse_img.clicked.connect(self._browse_img)
        row2.addWidget(btn_browse_img)

        # Mode Selection: Single Line vs Full Page Document
        row2.addWidget(QLabel("OCR Mode:"))
        self.radio_doc = QRadioButton("Full Page / Document")
        self.radio_doc.setChecked(True)
        self.radio_line = QRadioButton("Single Text Line")
        row2.addWidget(self.radio_doc)
        row2.addWidget(self.radio_line)

        self.btn_run_ocr = QPushButton("Recognize Text")
        self.btn_run_ocr.setObjectName("btn_primary")
        self.btn_run_ocr.setIcon(qta.icon("fa5s.bolt", color="#11111b"))
        self.btn_run_ocr.clicked.connect(self._run_inference)
        row2.addWidget(self.btn_run_ocr)
        top_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Pipeline Enhancements:"))
        self.chk_deskew = QCheckBox("Auto-Deskew && Flatten Shadows")
        self.chk_deskew.setChecked(True)
        row3.addWidget(self.chk_deskew)

        self.chk_wordseg = QCheckBox("Word Segmentation (Add Spaces)")
        self.chk_wordseg.setChecked(True)
        row3.addWidget(self.chk_wordseg)

        self.chk_spell = QCheckBox("Auto Spell Correction")
        self.chk_spell.setChecked(True)
        row3.addWidget(self.chk_spell)

        row3.addStretch()
        top_layout.addLayout(row3)

        main_layout.addWidget(top_group)

        # 2. Splitter: Visual Image & Multi-line Transcription
        splitter = QSplitter(Qt.Horizontal)

        left_preview = QGroupBox("Document Inspection && Detected Lines")
        preview_layout = QVBoxLayout(left_preview)
        self.image_viewer = KhmerImageViewer()
        preview_layout.addWidget(self.image_viewer)
        splitter.addWidget(left_preview)

        right_result = QGroupBox("Recognized Khmer Transcription")
        result_layout = QVBoxLayout(right_result)

        self.text_output = QTextEdit()
        self.text_output.setStyleSheet(
            "font-size: 16px; line-height: 1.6; color: #cdd6f4; background-color: #181825; border: 1px solid #313244; font-family: 'Noto Sans Khmer', sans-serif;"
        )
        result_layout.addWidget(self.text_output)

        self.meta_label = QLabel("Status: Ready")
        self.meta_label.setStyleSheet("color: #89b4fa; font-size: 12px; font-weight: 500;")
        result_layout.addWidget(self.meta_label)

        action_row = QHBoxLayout()
        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.setIcon(qta.icon("fa5s.copy", color="#cdd6f4"))
        btn_copy.clicked.connect(self._copy_clipboard)
        action_row.addWidget(btn_copy)

        btn_save_txt = QPushButton("Export to .txt File")
        btn_save_txt.setIcon(qta.icon("fa5s.save", color="#cdd6f4"))
        btn_save_txt.clicked.connect(self._save_txt_file)
        action_row.addWidget(btn_save_txt)

        btn_save_pdf = QPushButton("Export Searchable PDF")
        btn_save_pdf.setIcon(qta.icon("fa5s.file-pdf", color="#f38ba8"))
        btn_save_pdf.clicked.connect(self._export_searchable_pdf)
        action_row.addWidget(btn_save_pdf)

        action_row.addStretch()
        result_layout.addLayout(action_row)

        splitter.addWidget(right_result)
        splitter.setSizes([520, 480])
        main_layout.addWidget(splitter, stretch=2)

        # 3. Bottom Group: ONNX Dynamic Exporter (Use &&)
        export_group = QGroupBox("Export Model to Production Dynamic ONNX")
        export_layout = QVBoxLayout(export_group)

        export_row = QHBoxLayout()
        export_row.addWidget(QLabel("Output ONNX Path:"))
        self.onnx_out_edit = QLineEdit(str(Path.cwd() / "exports" / "khmer_ocr.onnx"))
        export_row.addWidget(self.onnx_out_edit)

        btn_export = QPushButton("Export to Dynamic ONNX")
        btn_export.setObjectName("btn_success")
        btn_export.setIcon(qta.icon("fa5s.file-export", color="#11111b"))
        btn_export.clicked.connect(self._export_onnx)
        export_row.addWidget(btn_export)
        export_layout.addLayout(export_row)

        main_layout.addWidget(export_group, stretch=1)

    def _browse_ckpt(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Checkpoint", "", "PyTorch Models (*.pth *.pt)")
        if f:
            self.ckpt_edit.setText(f)

    def _browse_img(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if f:
            self.img_edit.setText(f)
            self.current_img_path = f
            self.image_viewer.set_image_path(f)

    def _load_model_if_needed(self):
        ckpt_path = self.ckpt_edit.text().strip()
        if self.model is not None and self.current_ckpt_path == ckpt_path:
            return

        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        char_list = ckpt.get("char_list", [])
        backbone = ckpt.get("backbone", "resnet34")

        self.char_map = KhmerCharMap(characters=char_list, include_specials=False)
        self.model = KhmerCRNN(num_classes=len(self.char_map), backbone_type=backbone)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self.current_ckpt_path = ckpt_path

    def _preprocess_crop(self, crop: Image.Image, device: torch.device) -> torch.Tensor:
        im = crop.convert("L")
        w, h = im.size
        new_w = max(16, int(48 * (w / max(1, h))))
        im = im.resize((new_w, 48), Image.Resampling.BILINEAR)
        tensor = T.functional.to_tensor(im)
        return ((tensor - 0.5) / 0.5).unsqueeze(0).to(device)

    def _notify_app_status(self, text: str, state: str = "idle"):
        top = self.window()
        if hasattr(top, "set_process_status"):
            top.set_process_status(text, state)

    def _run_inference(self):
        img_path = self.img_edit.text().strip()
        if not img_path or not Path(img_path).exists():
            QMessageBox.warning(self, "Warning", "Please select a valid image file!")
            return

        self._notify_app_status("Processing Document OCR...", "ocr")
        try:
            self._load_model_if_needed()
            device = next(self.model.parameters()).device

            with Image.open(img_path) as raw_img:
                start_t = time.perf_counter()

                # Document preprocessing: auto-deskew & shadow removal
                deskew_info = ""
                if self.chk_deskew.isChecked():
                    full_img, meta = self.preprocessor.preprocess_document(
                        raw_img, auto_deskew=True, remove_shadow=True
                    )
                    if abs(meta.get("skew_angle_deg", 0.0)) > 0.2:
                        deskew_info = f" | Deskew: {meta['skew_angle_deg']:.1f}°"
                else:
                    full_img = raw_img.copy()

                self.last_doc_image = full_img.copy()

                if self.radio_doc.isChecked():
                    # Full Page Document Mode: Segment lines first!
                    bboxes, line_crops, annotated_img = self.segmenter.segment(full_img)
                    self.image_viewer.set_pil_image(annotated_img)

                    transcriptions = []
                    with torch.no_grad():
                        for crop in line_crops:
                            tensor = self._preprocess_crop(crop, device)
                            log_probs = self.model(tensor)
                            decoded = self.model.decode_greedy(log_probs)
                            text = self.char_map.decode(decoded[0]) if decoded else ""
                            if text.strip():
                                if self.chk_spell.isChecked():
                                    text, _ = self.spell_corrector.correct_sentence(text)
                                if self.chk_wordseg.isChecked():
                                    text = self.word_segmenter.segment_to_string(text, delimiter=" ")
                                transcriptions.append(text)

                    self.last_bboxes = bboxes
                    self.last_transcriptions = transcriptions

                    elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                    full_text = "\n".join(transcriptions)
                    self.text_output.setPlainText(full_text)
                    self.meta_label.setText(
                        f"Detected {len(bboxes)} text lines | Latency: {elapsed_ms:.1f} ms{deskew_info} | Device: {device}"
                    )

                else:
                    # Single Line Mode
                    # Safeguard: Check if the image appears to be a multi-line document
                    aspect_ratio = full_img.width / max(1, full_img.height)
                    if full_img.height > 100 and aspect_ratio < 4.0:
                        ans = QMessageBox.question(
                            self,
                            "Multi-line Document Detected",
                            "This image appears to contain multiple text lines or a full page.\n\n"
                            "Would you like to switch to 'Document (Multi-line) Mode' to automatically segment each line for optimal accuracy?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if ans == QMessageBox.Yes:
                            self.radio_doc.setChecked(True)
                            return self._run_inference()

                    self.image_viewer.set_pil_image(full_img)
                    tensor = self._preprocess_crop(full_img, device)
                    with torch.no_grad():
                        log_probs = self.model(tensor)
                        decoded = self.model.decode_greedy(log_probs)
                    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

                    text = self.char_map.decode(decoded[0]) if decoded else ""
                    if text.strip():
                        if self.chk_spell.isChecked():
                            text, _ = self.spell_corrector.correct_sentence(text)
                        if self.chk_wordseg.isChecked():
                            text = self.word_segmenter.segment_to_string(text, delimiter=" ")

                    self.last_bboxes = [(0, 0, full_img.width, full_img.height)]
                    self.last_transcriptions = [text]

                    self.text_output.setPlainText(text)
                    self.meta_label.setText(f"Single Line Latency: {elapsed_ms:.1f} ms{deskew_info} | Device: {device}")

                self._notify_app_status("Idle", "idle")

        except Exception as e:
            self._notify_app_status("Idle", "idle")
            QMessageBox.critical(self, "Inference Error", f"Inference failed:\n{e}")

    def _export_searchable_pdf(self):
        if self.last_doc_image is None or not self.last_transcriptions:
            QMessageBox.warning(self, "Warning", "Please run text recognition on an image before exporting PDF!")
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save Searchable PDF", "document_searchable.pdf", "PDF Documents (*.pdf)")
        if not f:
            return
        try:
            out_p = self.pdf_exporter.export(
                image=self.last_doc_image,
                bboxes=self.last_bboxes,
                transcriptions=self.last_transcriptions,
                output_pdf_path=f,
            )
            QMessageBox.information(self, "Success", f"Searchable PDF exported successfully!\nSaved to:\n{out_p}")
        except Exception as e:
            QMessageBox.critical(self, "PDF Export Error", f"Failed to export PDF:\n{e}")

    def _copy_clipboard(self):
        text = self.text_output.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            QMessageBox.information(self, "Copied", "Recognized text copied to clipboard!")

    def _save_txt_file(self):
        text = self.text_output.toPlainText()
        if not text:
            return
        f, _ = QFileDialog.getSaveFileName(self, "Save Document Text", "transcription.txt", "Text Files (*.txt)")
        if f:
            with open(f, "w", encoding="utf-8") as out:
                out.write(text)
            QMessageBox.information(self, "Saved", f"Transcription saved to:\n{f}")

    def _export_onnx(self):
        ckpt_path = self.ckpt_edit.text().strip()
        onnx_path = self.onnx_out_edit.text().strip()

        if not Path(ckpt_path).exists():
            QMessageBox.warning(self, "Warning", f"Checkpoint not found: {ckpt_path}")
            return

        try:
            out_file = export_to_onnx(ckpt_path, onnx_path, img_height=48)
            QMessageBox.information(
                self, "Export Successful",
                f"Model successfully exported to dynamic ONNX!\nSaved to:\n{out_file}\nMetadata: {out_file.with_suffix('.json')}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export to ONNX:\n{e}")
