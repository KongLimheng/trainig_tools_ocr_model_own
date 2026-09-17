"""Embedded Real-Time Matplotlib Training Curves Widget."""

from PyQt5.QtWidgets import QWidget, QVBoxLayout
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RealtimePlotWidget(QWidget):
    """Matplotlib canvas embedded in PyQt5 for live training loss and validation CER."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create Dark Theme Matplotlib Figure
        self.figure = Figure(figsize=(6, 4), facecolor="#181825")
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(2, 1, 1)
        self.ax2 = self.figure.add_subplot(2, 1, 2)
        self._format_axes()

    def _format_axes(self):
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor("#252538")
            ax.tick_params(colors="#a6adc8", labelsize=9)
            ax.spines["bottom"].set_color("#313244")
            ax.spines["top"].set_color("#313244")
            ax.spines["left"].set_color("#313244")
            ax.spines["right"].set_color("#313244")
            ax.grid(True, linestyle="--", alpha=0.3, color="#45475a")

        self.ax1.set_title("Training Loss", color="#89b4fa", fontsize=11, fontweight="bold", pad=8)
        self.ax2.set_title("Validation Metrics (Normalized CER & Exact Match)", color="#a6e3a1", fontsize=11, fontweight="bold", pad=8)
        self.figure.tight_layout()

    def update_metrics(
        self,
        epochs: list[int],
        losses: list[float],
        cers: list[float],
        exact_matches: list[float] | None = None,
    ):
        """Redraws the loss and accuracy curves."""
        self.ax1.clear()
        self.ax2.clear()
        self._format_axes()

        if epochs and losses:
            self.ax1.plot(epochs, losses, color="#89b4fa", linewidth=2, marker="o", markersize=4, label="Train Loss")
            self.ax1.legend(facecolor="#181825", edgecolor="#313244", labelcolor="#cdd6f4", fontsize=8)

        if epochs and cers:
            self.ax2.plot(epochs, cers, color="#f38ba8", linewidth=2, marker="s", markersize=4, label="Norm CER (Lower=Better)")
            if exact_matches:
                self.ax2.plot(epochs, exact_matches, color="#a6e3a1", linewidth=2, marker="^", markersize=4, label="Exact Match (Higher=Better)")
            self.ax2.legend(facecolor="#181825", edgecolor="#313244", labelcolor="#cdd6f4", fontsize=8)

        self.figure.tight_layout()
        self.canvas.draw()
