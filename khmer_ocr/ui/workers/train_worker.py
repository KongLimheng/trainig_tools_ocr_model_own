"""Background QThread Worker for PyTorch Model Training."""

from PyQt5.QtCore import QThread, pyqtSignal
from ...training.trainer import KhmerOCRTrainer


class TrainingWorker(QThread):
    """Background worker for non-blocking model training."""

    step_progress = pyqtSignal(int, int, float, float)
    epoch_finished = pyqtSignal(int, int, float, dict)
    training_completed = pyqtSignal(str)
    training_error = pyqtSignal(str)

    def __init__(self, trainer: KhmerOCRTrainer, epochs: int = 20, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.epochs = epochs

    def stop(self):
        """Requests training stop."""
        self.trainer.request_stop()

    def run(self):
        try:
            best_model_path = self.trainer.train(
                epochs=self.epochs,
                epoch_callback=self._on_epoch,
                step_callback=self._on_step,
            )
            self.training_completed.emit(str(best_model_path))
        except Exception as e:
            self.training_error.emit(str(e))

    def _on_step(self, step: int, total_steps: int, loss: float, lr: float = 0.0):
        self.step_progress.emit(step, total_steps, loss, lr)

    def _on_epoch(self, epoch: int, total_epochs: int, train_loss: float, val_metrics: dict):
        self.epoch_finished.emit(epoch, total_epochs, train_loss, val_metrics)
