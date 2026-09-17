"""Tab 3: Model Training Studio with Scrollable Controls & Vector Icons."""

from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QLabel,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QFileDialog, QProgressBar, QMessageBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea
)
import qtawesome as qta
import torch
from torch.utils.data import DataLoader
from ..widgets.plot_widget import RealtimePlotWidget
from ..workers.train_worker import TrainingWorker
from ...models.crnn import KhmerCRNN
from ...models.checkpoint_manager import load_and_adapt_checkpoint, inspect_checkpoint
from ...vocab.char_map import KhmerCharMap
from ...dataset.folder_dataset import KhmerOCRDataset
from ...dataset.collate import DynamicAspectCollate
from ...training.trainer import KhmerOCRTrainer


class TabTrainingStudio(QWidget):
    """Model Training Studio Tab with real-time loss/CER visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: TrainingWorker | None = None
        self.epochs_list: list[int] = []
        self.losses_list: list[float] = []
        self.cers_list: list[float] = []
        self.exact_list: list[float] = []
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Scroll Area for Controls
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(4, 4, 8, 4)
        left_layout.setSpacing(10)

        # Datasets Group (Use &&)
        data_group = QGroupBox("Training && Validation Datasets")
        data_layout = QVBoxLayout(data_group)

        data_layout.addWidget(QLabel("Train Dataset Dir:"))
        train_dir_row = QHBoxLayout()
        self.train_dir_edit = QLineEdit(
            str(Path.cwd() / "data" / "synthetic_train"))
        train_dir_row.addWidget(self.train_dir_edit)
        btn_train_browse = QPushButton()
        btn_train_browse.setIcon(qta.icon("fa5s.folder-open", color="#cdd6f4"))
        btn_train_browse.clicked.connect(self._browse_train_dir)
        train_dir_row.addWidget(btn_train_browse)
        data_layout.addLayout(train_dir_row)

        data_layout.addWidget(QLabel("Val Dataset Dir (Optional):"))
        val_dir_row = QHBoxLayout()
        self.val_dir_edit = QLineEdit(
            str(Path.cwd() / "data" / "synthetic_train"))
        val_dir_row.addWidget(self.val_dir_edit)
        btn_val_browse = QPushButton()
        btn_val_browse.setIcon(qta.icon("fa5s.folder-open", color="#cdd6f4"))
        btn_val_browse.clicked.connect(self._browse_val_dir)
        val_dir_row.addWidget(btn_val_browse)
        data_layout.addLayout(val_dir_row)

        left_layout.addWidget(data_group)

        # Model Architecture Group
        model_group = QGroupBox("Model Architecture")
        model_layout = QVBoxLayout(model_group)

        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(
            ["resnet34 (Recommended)", "mobilenet (Ultra-Lightweight)"])
        model_layout.addWidget(QLabel("Backbone:"))
        model_layout.addWidget(self.backbone_combo)

        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Input Height (px):"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 64)
        self.height_spin.setValue(48)
        height_layout.addWidget(self.height_spin)
        model_layout.addLayout(height_layout)
        left_layout.addWidget(model_group)

        # Checkpoint & Continual Learning Group (Use &&)
        resume_group = QGroupBox("Checkpoint && Continual Learning")
        resume_layout = QVBoxLayout(resume_group)

        self.chk_continue = QCheckBox(
            "Continue Training from Existing Checkpoint")
        self.chk_continue.setStyleSheet("font-weight: bold; color: #89b4fa;")
        self.chk_continue.toggled.connect(self._on_continue_toggled)
        resume_layout.addWidget(self.chk_continue)

        self.resume_container = QWidget()
        resume_box_layout = QVBoxLayout(self.resume_container)
        resume_box_layout.setContentsMargins(0, 4, 0, 0)
        resume_box_layout.setSpacing(6)

        resume_box_layout.addWidget(QLabel("Pre-trained Checkpoint (.pth):"))
        ckpt_path_row = QHBoxLayout()
        self.resume_ckpt_edit = QLineEdit(
            str(Path.cwd() / "checkpoints" / "best_model.pth"))
        self.resume_ckpt_edit.textChanged.connect(
            self._inspect_selected_checkpoint)
        ckpt_path_row.addWidget(self.resume_ckpt_edit)

        btn_resume_browse = QPushButton()
        btn_resume_browse.setIcon(
            qta.icon("fa5s.folder-open", color="#cdd6f4"))
        btn_resume_browse.clicked.connect(self._browse_resume_ckpt)
        ckpt_path_row.addWidget(btn_resume_browse)
        resume_box_layout.addLayout(ckpt_path_row)

        # Inspection badge
        self.lbl_ckpt_badge = QLabel(
            "Select checkpoint to inspect compatibility")
        self.lbl_ckpt_badge.setWordWrap(True)
        self.lbl_ckpt_badge.setStyleSheet(
            "background-color: #181825; color: #a6adc8; padding: 4px 8px; border-radius: 4px; font-size: 11px;"
        )
        resume_box_layout.addWidget(self.lbl_ckpt_badge)

        resume_box_layout.addWidget(QLabel("Continual Training Mode:"))
        self.combo_resume_mode = QComboBox()
        self.combo_resume_mode.addItems([
            "Fine-Tune on New Data (Transfers weights, fresh LR & optimizer)",
            "Resume Interrupted Run (Restores exact optimizer, epoch & loss)",
        ])
        self.combo_resume_mode.currentIndexChanged.connect(
            self._on_resume_mode_changed)
        resume_box_layout.addWidget(self.combo_resume_mode)

        self.chk_freeze_backbone = QCheckBox(
            "Freeze CNN Backbone (Train RNN Head Only)")
        self.chk_freeze_backbone.setToolTip(
            "Freezes feature extraction layers to prevent catastrophic forgetting during fine-tuning")
        resume_box_layout.addWidget(self.chk_freeze_backbone)

        self.resume_container.setEnabled(False)
        resume_layout.addWidget(self.resume_container)
        left_layout.addWidget(resume_group)

        # Training Parameters Group (Use &&)
        param_group = QGroupBox("Training Parameters")
        param_layout = QVBoxLayout(param_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 256)
        self.batch_spin.setValue(16)
        row1.addWidget(self.batch_spin)

        row1.addWidget(QLabel("Epochs:"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 10000)
        self.epoch_spin.setValue(100)
        row1.addWidget(self.epoch_spin)
        param_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setValue(0.0002)
        row2.addWidget(self.lr_spin)
        param_layout.addLayout(row2)

        self.chk_lr_scheduler = QCheckBox(
            "Adaptive LR Scheduler (ReduceLROnPlateau)")
        self.chk_lr_scheduler.setChecked(True)
        self.chk_lr_scheduler.setToolTip(
            "Halves learning rate when validation loss plateaus to reach higher accuracy")
        param_layout.addWidget(self.chk_lr_scheduler)

        self.chk_amp = QCheckBox("Enable AMP (FP16 Mixed Precision)")
        self.chk_amp.setChecked(True)
        param_layout.addWidget(self.chk_amp)

        param_layout.addWidget(QLabel("Checkpoint Dir:"))
        self.ckpt_dir_edit = QLineEdit(str(Path.cwd() / "checkpoints"))
        param_layout.addWidget(self.ckpt_dir_edit)

        left_layout.addWidget(param_group)

        # Action Buttons
        btn_action_row = QHBoxLayout()
        self.btn_train = QPushButton("Start Training")
        self.btn_train.setObjectName("btn_primary")
        self.btn_train.setIcon(qta.icon("fa5s.play", color="#11111b"))
        self.btn_train.clicked.connect(self._start_training)
        btn_action_row.addWidget(self.btn_train)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("btn_danger")
        self.btn_stop.setIcon(qta.icon("fa5s.stop-circle", color="#11111b"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_training)
        btn_action_row.addWidget(self.btn_stop)
        left_layout.addLayout(btn_action_row)

        left_layout.addStretch()
        left_scroll.setWidget(left_content)
        splitter.addWidget(left_scroll)

        # Right Panel: Real-time Plot & Validation Table
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Matplotlib Plot Canvas
        self.plot_widget = RealtimePlotWidget()
        right_layout.addWidget(self.plot_widget, stretch=3)

        # Validation Preview Table
        preview_group = QGroupBox("Live Validation Predictions")
        preview_layout = QVBoxLayout(preview_group)

        self.val_table = QTableWidget()
        self.val_table.setColumnCount(3)
        self.val_table.setHorizontalHeaderLabels(
            ["Target Ground Truth", "Model Prediction", "Status"])
        self.val_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.val_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        preview_layout.addWidget(self.val_table)
        right_layout.addWidget(preview_group, stretch=2)

        splitter.addWidget(right_panel)
        splitter.setSizes([360, 640])
        main_layout.addWidget(splitter)

        # Bottom: Progress Bars & Status
        bottom_box = QWidget()
        bottom_layout = QVBoxLayout(bottom_box)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self.step_bar = QProgressBar()
        self.step_bar.setValue(0)
        bottom_layout.addWidget(self.step_bar)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("color: #a6adc8; font-size: 11.5px;")
        bottom_layout.addWidget(self.status_label)

        main_layout.addWidget(bottom_box)

    def _browse_train_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Train Dataset Directory")
        if d:
            self.train_dir_edit.setText(d)

    def _browse_val_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Validation Dataset Directory")
        if d:
            self.val_dir_edit.setText(d)

    def _browse_resume_ckpt(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint (.pth)",
            str(Path.cwd() / "checkpoints"),
            "PyTorch Checkpoints (*.pth)"
        )
        if f:
            self.resume_ckpt_edit.setText(f)

    def _on_continue_toggled(self, checked: bool):
        self.resume_container.setEnabled(checked)
        if checked:
            self._inspect_selected_checkpoint()

    def _on_resume_mode_changed(self, index: int):
        if index == 0:  # Fine-tune mode
            if self.lr_spin.value() >= 0.0005:
                self.lr_spin.setValue(0.0002)
        self._inspect_selected_checkpoint()

    def _inspect_selected_checkpoint(self):
        p = Path(self.resume_ckpt_edit.text().strip())
        if not p.exists():
            self.lbl_ckpt_badge.setText("⚠️ Checkpoint file does not exist")
            self.lbl_ckpt_badge.setStyleSheet(
                "background-color: #313244; color: #f38ba8; padding: 4px 8px; border-radius: 4px; font-size: 11px;"
            )
            return
        try:
            info = inspect_checkpoint(p)
            epoch = info["epoch"]
            chars = info["char_count"]
            backbone = info["backbone"]
            has_opt = "Optimizer ✔" if info["has_optimizer"] else "Optimizer ✖"
            cer_str = f" | Norm CER: {info['norm_cer']:.2%}" if info["norm_cer"] is not None else ""
            self.lbl_ckpt_badge.setText(
                f"✔ {backbone.upper()} | Trained Epoch: {epoch} | Vocab: {chars} chars{cer_str} | {has_opt}"
            )
            self.lbl_ckpt_badge.setStyleSheet(
                "background-color: #1e1e2e; color: #a6e3a1; padding: 4px 8px; border-radius: 4px; font-size: 11px;"
            )
        except Exception as e:
            self.lbl_ckpt_badge.setText(f"⚠️ Error reading checkpoint: {e}")
            self.lbl_ckpt_badge.setStyleSheet(
                "background-color: #313244; color: #f38ba8; padding: 4px 8px; border-radius: 4px; font-size: 11px;"
            )

    def _start_training(self):
        train_path = Path(self.train_dir_edit.text().strip())
        val_path = Path(self.val_dir_edit.text().strip())
        ckpt_dir = Path(self.ckpt_dir_edit.text().strip())

        if not (train_path / "labels.txt").exists():
            QMessageBox.warning(
                self, "Warning", f"labels.txt not found in {train_path}!")
            return

        char_map = KhmerCharMap()
        img_h = self.height_spin.value()
        train_dataset = KhmerOCRDataset(
            train_path, char_map=char_map, img_height=img_h)

        val_dataset = None
        if val_path.exists() and (val_path / "labels.txt").exists():
            val_dataset = KhmerOCRDataset(
                val_path, char_map=char_map, img_height=img_h)

        batch_size = self.batch_spin.value()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=DynamicAspectCollate(),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=DynamicAspectCollate(),
            )

        backbone_key = "mobilenet" if "mobilenet" in self.backbone_combo.currentText() else "resnet34"

        # Continual training / checkpoint logic
        use_continue = self.chk_continue.isChecked()
        ckpt_path = Path(self.resume_ckpt_edit.text().strip()
                         ) if use_continue else None
        is_resume_exact = use_continue and (
            "Resume Interrupted" in self.combo_resume_mode.currentText())

        start_epoch = 0
        best_val_cer = float("inf")
        resume_optimizer_state = None

        if use_continue and ckpt_path and ckpt_path.exists():
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            model, adapt_info = load_and_adapt_checkpoint(
                checkpoint_path=ckpt_path,
                target_char_map=char_map,
                target_backbone=backbone_key,
                device=device_str,
            )

            raw_ckpt = adapt_info.get("raw_checkpoint", {})
            if is_resume_exact:
                start_epoch = raw_ckpt.get("epoch", 0)
                best_val_cer = raw_ckpt.get("val_metrics", {}).get(
                    "norm_cer", float("inf"))
                resume_optimizer_state = raw_ckpt.get(
                    "optimizer_state_dict", None)
            else:
                # Fine-tuning mode starts fresh epoch count with adapted weights
                start_epoch = 0
                best_val_cer = float("inf")
                resume_optimizer_state = None
        else:
            model = KhmerCRNN(num_classes=len(char_map),
                              backbone_type=backbone_key)

        trainer = KhmerOCRTrainer(
            model=model,
            char_map=char_map,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=self.lr_spin.value(),
            use_amp=self.chk_amp.isChecked(),
            output_dir=ckpt_dir,
            start_epoch=start_epoch,
            best_val_cer=best_val_cer,
            resume_optimizer_state=resume_optimizer_state,
            lr_scheduler_type="plateau" if self.chk_lr_scheduler.isChecked() else None,
            freeze_backbone=self.chk_freeze_backbone.isChecked() if use_continue else False,
            override_lr=True,
        )

        self.worker = TrainingWorker(
            trainer=trainer, epochs=self.epoch_spin.value())
        self.worker.step_progress.connect(self._on_step_progress)
        self.worker.epoch_finished.connect(self._on_epoch_finished)
        self.worker.training_completed.connect(self._on_training_completed)
        self.worker.training_error.connect(self._on_training_error)

        if not is_resume_exact:
            self.epochs_list.clear()
            self.losses_list.clear()
            self.cers_list.clear()
            self.exact_list.clear()

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_label.setText("Training started...")
        self._notify_app_status(
            f"Training ({start_epoch}/{start_epoch + self.epoch_spin.value()})", "training")
        self.worker.start()

    def _notify_app_status(self, text: str, state: str = "idle"):
        top = self.window()
        if hasattr(top, "set_process_status"):
            top.set_process_status(text, state)

    def _stop_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping training...")
            self._notify_app_status("Stopping Training...", "training")

    def _on_step_progress(self, step: int, total_steps: int, loss: float, lr: float = 0.0):
        pct = int((step / max(1, total_steps)) * 100)
        self.step_bar.setValue(pct)
        lr_str = f" | LR: {lr:.6f}" if lr > 0 else ""
        self.status_label.setText(
            f"Batch {step}/{total_steps} | Loss: {loss:.4f}{lr_str}")

    def _on_epoch_finished(self, epoch: int, total_epochs: int, train_loss: float, val_metrics: dict):
        self._notify_app_status(
            f"Training (Epoch {epoch}/{total_epochs})", "training")
        self.epochs_list.append(epoch)
        self.losses_list.append(train_loss)
        val_cer = val_metrics.get("norm_cer", 0.0)
        exact_match = val_metrics.get("exact_match", 0.0)
        self.cers_list.append(val_cer)
        self.exact_list.append(exact_match)

        self.plot_widget.update_metrics(
            self.epochs_list, self.losses_list, self.cers_list, self.exact_list)

        samples = val_metrics.get("samples", [])
        self.val_table.setRowCount(len(samples))
        for idx, item in enumerate(samples):
            self.val_table.setItem(idx, 0, QTableWidgetItem(item["target"]))
            self.val_table.setItem(idx, 1, QTableWidgetItem(item["pred"]))
            is_match = item["target"] == item["pred"]
            status_item = QTableWidgetItem(
                "Exact Match" if is_match else "Diff")
            status_item.setForeground(Qt.green if is_match else Qt.red)
            self.val_table.setItem(idx, 2, status_item)

        self.status_label.setText(
            f"Epoch {epoch}/{total_epochs} Complete | Train Loss: {train_loss:.4f} | Val Norm CER: {val_cer:.4f} | Exact: {exact_match:.1%}"
        )

    def _on_training_completed(self, best_model_path: str):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.step_bar.setValue(100)
        self.status_label.setText("Training completed successfully!")
        self._notify_app_status("Idle", "idle")
        QMessageBox.information(
            self, "Training Complete",
            f"Model training finished!\nBest checkpoint saved to:\n{best_model_path}"
        )

    def _on_training_error(self, err_msg: str):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText(f"Error: {err_msg}")
        self._notify_app_status("Idle", "idle")
        QMessageBox.critical(self, "Training Error",
                             f"Training failed:\n{err_msg}")
