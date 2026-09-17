"""Background QThread Worker for Batch Synthetic Data Generation."""

from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from ...synth.generator import KhmerDatasetGenerator


class SyntheticGenWorker(QThread):
    """Background worker for non-blocking synthetic dataset generation."""

    progress_updated = pyqtSignal(int, int, str)
    generation_finished = pyqtSignal(str)
    generation_error = pyqtSignal(str)

    def __init__(
        self,
        generator: KhmerDatasetGenerator,
        output_dir: str | Path,
        num_samples: int,
        selected_fonts: list[str] | None = None,
        augment: bool = True,
        bg_style: str = "random",
        parent=None,
    ):
        super().__init__(parent)
        self.generator = generator
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.selected_fonts = selected_fonts
        self.augment = augment
        self.bg_style = bg_style

    def stop(self):
        """Requests cancellation."""
        self.generator.stop()

    def run(self):
        try:
            labels_path = self.generator.generate_batch(
                output_dir=self.output_dir,
                num_samples=self.num_samples,
                selected_fonts=self.selected_fonts,
                augment=self.augment,
                bg_style=self.bg_style,
                progress_callback=self._on_progress,
            )
            self.generation_finished.emit(str(labels_path))
        except Exception as e:
            self.generation_error.emit(str(e))

    def _on_progress(self, current: int, total: int, message: str):
        self.progress_updated.emit(current, total, message)
