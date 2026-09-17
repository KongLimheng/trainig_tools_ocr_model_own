"""Trainer Engine for Khmer OCR Models."""

import os
from pathlib import Path
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ..models.crnn import KhmerCRNN
from ..vocab.char_map import KhmerCharMap
from ..metrics.normalized_cer import calculate_normalized_cer
from ..metrics.cer import calculate_cer, calculate_exact_match
from .loss import KhmerCTCLoss


class KhmerOCRTrainer:
    """Manages model training, validation, checkpointing, and progress reporting."""

    def __init__(
        self,
        model: KhmerCRNN,
        char_map: KhmerCharMap,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: Optional[float] = 2e-4,
        weight_decay: float = 1e-4,
        device: str | None = None,
        use_amp: bool = True,
        output_dir: str | Path = "checkpoints",
        start_epoch: int = 0,
        best_val_cer: float = float("inf"),
        resume_optimizer_state: Optional[dict] = None,
        lr_scheduler_type: Optional[str] = "plateau",
        freeze_backbone: bool = False,
        unfreeze_after_epochs: int = 0,
        override_lr: bool = True,
        early_stopping_patience: Optional[int] = None,
    ):
        self.device = torch.device(device or (
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.char_map = char_map
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.best_val_cer = best_val_cer
        self.best_train_loss = float("inf")
        self.unfreeze_after_epochs = unfreeze_after_epochs
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0

        # Apply backbone freezing if requested
        if freeze_backbone:
            self.model.freeze_backbone(True)

        init_lr = learning_rate if learning_rate is not None else 2e-4
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=init_lr, weight_decay=weight_decay
        )
        if resume_optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(resume_optimizer_state)
                if override_lr and learning_rate is not None:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = learning_rate
                    print(f"Resumed optimizer state and set learning rate to: {learning_rate:.6f}")
                else:
                    saved_lr = self.get_current_lr()
                    print(f"Resumed optimizer state with preserved learning rate: {saved_lr:.6f}")
            except Exception as e:
                print(
                    f"Warning: Could not restore optimizer state: {e}. Starting with fresh optimizer.")

        # Adaptive Learning Rate Scheduler
        self.lr_scheduler_type = lr_scheduler_type
        if lr_scheduler_type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
            )
        elif lr_scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=15, eta_min=1e-6
            )
        else:
            self.scheduler = None

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.criterion = KhmerCTCLoss(blank_idx=char_map.blank_idx)

        self._stop_requested = False

    def get_current_lr(self) -> float:
        """Returns the current learning rate from the first parameter group."""
        if self.optimizer and self.optimizer.param_groups:
            return float(self.optimizer.param_groups[0]["lr"])
        return 0.0

    def request_stop(self) -> None:
        """Requests early termination of training loop."""
        self._stop_requested = True

    def train_epoch(
        self,
        step_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> float:
        """Trains for one epoch.
        Returns average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            if self._stop_requested:
                break

            images = batch["images"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)
            b_size = images.size(0)

            self.optimizer.zero_grad()

            autocast_ctx = torch.autocast("cuda", enabled=self.use_amp) if hasattr(
                torch, "amp") else torch.cuda.amp.autocast(enabled=self.use_amp)
            with autocast_ctx:
                # Forward: [T, B, C]
                log_probs = self.model(images)
                t_steps = log_probs.size(0)
                input_lengths = torch.full(
                    (b_size,), t_steps, dtype=torch.long, device=self.device)
                loss = self.criterion(log_probs, targets,
                                      input_lengths, target_lengths)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()
            total_loss += loss_val

            if step_callback and (batch_idx % 5 == 0 or batch_idx == num_batches - 1):
                try:
                    step_callback(batch_idx + 1, num_batches,
                                  loss_val, self.get_current_lr())
                except TypeError:
                    step_callback(batch_idx + 1, num_batches, loss_val)

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluates on validation dataset.
        Returns dictionary of metrics and sample prediction pairs.
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return {"cer": 0.0, "norm_cer": 0.0, "exact_match": 0.0, "samples": [], "has_val": False}

        self.model.eval()
        all_targets = []
        all_preds = []
        preview_samples = []

        for batch in self.val_loader:
            if self._stop_requested:
                break

            images = batch["images"].to(self.device)
            raw_texts = batch["texts"]

            log_probs = self.model(images)
            decoded_indices = self.model.decode_greedy(log_probs)

            for idx, indices in enumerate(decoded_indices):
                pred_text = self.char_map.decode(indices)
                target_text = raw_texts[idx]

                all_targets.append(target_text)
                all_preds.append(pred_text)

                if len(preview_samples) < 10:
                    preview_samples.append({
                        "target": target_text,
                        "pred": pred_text,
                        "path": batch["paths"][idx],
                    })

        standard_cer = calculate_cer(all_targets, all_preds)
        normalized_cer = calculate_normalized_cer(all_targets, all_preds)
        exact_match = calculate_exact_match(all_targets, all_preds)

        return {
            "cer": standard_cer,
            "norm_cer": normalized_cer,
            "exact_match": exact_match,
            "samples": preview_samples,
            "has_val": True,
        }

    def train(
        self,
        epochs: int = 20,
        epoch_callback: Optional[Callable[[
            int, int, float, dict], None]] = None,
        step_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Path:
        """Main training loop across epochs."""
        self._stop_requested = False
        best_checkpoint_path = self.output_dir / "best_model.pth"

        start_ep = self.start_epoch + 1
        end_ep = self.start_epoch + epochs
        total_epochs = end_ep

        try:
            for epoch in range(start_ep, end_ep + 1):
                if self._stop_requested:
                    break

                self.current_epoch = epoch

                # Check if backbone unfreezing is scheduled
                if self.unfreeze_after_epochs > 0 and (epoch - self.start_epoch) > self.unfreeze_after_epochs:
                    if self.model.is_backbone_frozen():
                        self.model.freeze_backbone(False)

                train_loss = self.train_epoch(step_callback=step_callback)
                val_metrics = self.evaluate()

                # Step adaptive LR scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        metric_to_track = (
                            val_metrics["norm_cer"]
                            if (self.val_loader and len(self.val_loader) > 0)
                            else train_loss
                        )
                        self.scheduler.step(metric_to_track)
                    else:
                        self.scheduler.step()

                # Record current LR in val_metrics
                val_metrics["lr"] = self.get_current_lr()

                # Save latest checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "char_list": self.char_map.char_list,
                    "backbone": self.model.backbone_type,
                    "train_loss": train_loss,
                    "val_metrics": val_metrics,
                }, self.output_dir / "latest_model.pth")

                # Check if this is the best model (by Normalized CER if val set available, else by train_loss)
                has_val = val_metrics.get("has_val", False)
                if has_val:
                    is_best = val_metrics["norm_cer"] < self.best_val_cer
                    if is_best:
                        self.best_val_cer = val_metrics["norm_cer"]
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1
                else:
                    is_best = train_loss < self.best_train_loss
                    if is_best:
                        self.best_train_loss = train_loss
                        self.epochs_without_improvement = 0
                    else:
                        self.epochs_without_improvement += 1

                if is_best:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "char_list": self.char_map.char_list,
                        "backbone": self.model.backbone_type,
                        "val_metrics": val_metrics,
                        "train_loss": train_loss,
                    }, best_checkpoint_path)

                if epoch_callback:
                    epoch_callback(epoch, total_epochs, train_loss, val_metrics)

                if (
                    self.early_stopping_patience
                    and self.epochs_without_improvement >= self.early_stopping_patience
                ):
                    print(
                        f"\n[Early Stopping] No improvement for {self.epochs_without_improvement} consecutive epochs. Stopping early at epoch {epoch}."
                    )
                    break

        except KeyboardInterrupt:
            interrupted_path = self.output_dir / "interrupted_checkpoint.pth"
            torch.save({
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "char_list": self.char_map.char_list,
                "backbone": self.model.backbone_type,
            }, interrupted_path)
            print("\n" + "=" * 60)
            print(" [Training Interrupted by User]")
            print(f" Progress safely saved to: {interrupted_path}")
            print(" To resume seamlessly, run:")
            print(f"   uv run --no-sync python run_app.py train --train-dir {getattr(self.train_loader.dataset, 'data_dir', '<dataset_dir>')} --resume {interrupted_path}")
            print("=" * 60 + "\n")
            return interrupted_path

        return best_checkpoint_path if best_checkpoint_path.exists() else (self.output_dir / "latest_model.pth")
