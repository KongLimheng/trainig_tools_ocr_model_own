"""Live Process & GPU Telemetry Monitor Widget for Application Header."""

import os
import subprocess
from typing import Tuple, Dict, Any
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QFrame, QSizePolicy
)
import qtawesome as qta
import torch


def get_gpu_telemetry() -> Dict[str, Any]:
    """Queries GPU metrics using nvidia-smi with fast execution."""
    if not torch.cuda.is_available():
        return {"available": False}

    try:
        # Fast query of GPU compute utilization, memory, and temperature
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=0.4,
            check=True,
            close_fds=True,
            stdin=subprocess.DEVNULL,
        )
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpu_util = int(parts[0])
            mem_used_mb = int(parts[1])
            mem_total_mb = int(parts[2])
            temp_c = int(parts[3])
            gpu_name = parts[4]

            # Shorten common names (e.g. NVIDIA GeForce GTX 1060 -> GTX 1060)
            short_name = gpu_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")

            return {
                "available": True,
                "gpu_util": gpu_util,
                "mem_used_gb": mem_used_mb / 1024.0,
                "mem_total_gb": mem_total_mb / 1024.0,
                "mem_pct": int((mem_used_mb / max(1, mem_total_mb)) * 100),
                "temp_c": temp_c,
                "name": short_name,
            }
    except Exception:
        pass

    # Fallback to PyTorch CUDA queries if nvidia-smi is unavailable
    try:
        dev_idx = 0
        allocated = torch.cuda.memory_allocated(dev_idx) / (1024.0 ** 3)
        reserved = torch.cuda.memory_reserved(dev_idx) / (1024.0 ** 3)
        dev_name = torch.cuda.get_device_name(dev_idx).replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        return {
            "available": True,
            "gpu_util": 0,
            "mem_used_gb": allocated,
            "mem_total_gb": reserved if reserved > 0 else 6.0,
            "mem_pct": int((allocated / max(0.1, reserved)) * 100) if reserved > 0 else 0,
            "temp_c": 0,
            "name": dev_name,
        }
    except Exception:
        return {"available": False}


def get_cpu_ram_telemetry() -> Dict[str, Any]:
    """Fallback telemetry for host CPU & System RAM."""
    mem_total_gb = 16.0
    mem_used_gb = 4.0
    cpu_pct = 0

    try:
        # Read /proc/meminfo on Linux
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", "r") as f:
                lines = f.readlines()
            info = {}
            for l in lines:
                parts = l.split(":")
                if len(parts) == 2:
                    info[parts[0].strip()] = int(parts[1].split()[0])
            total_kb = info.get("MemTotal", 16000000)
            avail_kb = info.get("MemAvailable", total_kb // 2)
            used_kb = total_kb - avail_kb
            mem_total_gb = total_kb / (1024.0 * 1024.0)
            mem_used_gb = used_kb / (1024.0 * 1024.0)

        # 1-minute load average
        load1, _, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 4
        cpu_pct = min(100, int((load1 / cpu_count) * 100))
    except Exception:
        pass

    return {
        "cpu_pct": cpu_pct,
        "mem_used_gb": mem_used_gb,
        "mem_total_gb": mem_total_gb,
        "mem_pct": int((mem_used_gb / max(0.1, mem_total_gb)) * 100),
    }


class LiveTelemetryWidget(QWidget):
    """Real-time Header Monitor displaying GPU / CPU compute, VRAM, temp, and process state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_state = "idle"
        self.current_task_text = "Idle"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 1. Process Task Status Badge
        self.task_badge = QWidget()
        self.task_badge.setStyleSheet(
            "background-color: #181825; border: 1px solid #313244; border-radius: 6px; padding: 3px 8px;"
        )
        task_layout = QHBoxLayout(self.task_badge)
        task_layout.setContentsMargins(6, 2, 6, 2)
        task_layout.setSpacing(6)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #89b4fa; font-size: 11px;")
        task_layout.addWidget(self.status_dot)

        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #cdd6f4; font-size: 11.5px; font-weight: 600;")
        task_layout.addWidget(self.status_label)

        layout.addWidget(self.task_badge)

        # 2. Hardware Telemetry Card
        self.telemetry_card = QWidget()
        self.telemetry_card.setStyleSheet(
            "background-color: #1e1e2e; border: 1px solid #313244; border-radius: 6px; padding: 3px 10px;"
        )
        card_layout = QHBoxLayout(self.telemetry_card)
        card_layout.setContentsMargins(8, 2, 8, 2)
        card_layout.setSpacing(10)

        # Device Icon
        self.dev_icon = QLabel()
        self.dev_icon.setPixmap(qta.icon("fa5s.microchip", color="#a6e3a1").pixmap(16, 16))
        card_layout.addWidget(self.dev_icon)

        # GPU Name / Device Label
        self.name_label = QLabel("GPU")
        self.name_label.setStyleSheet("color: #cdd6f4; font-size: 12px; font-weight: bold;")
        card_layout.addWidget(self.name_label)

        # GPU Compute Usage Pill
        self.util_pill = QLabel("0%")
        self.util_pill.setStyleSheet(
            "background-color: #313244; color: #a6e3a1; font-size: 11.5px; font-weight: bold; "
            "padding: 2px 7px; border-radius: 4px;"
        )
        card_layout.addWidget(self.util_pill)

        # VRAM / Memory Badge
        self.mem_label = QLabel("VRAM: 0.0 / 0.0 GB")
        self.mem_label.setStyleSheet("color: #89b4fa; font-size: 11.5px; font-weight: 500;")
        card_layout.addWidget(self.mem_label)

        # Thermal Badge
        self.temp_label = QLabel("0°C")
        self.temp_label.setStyleSheet("color: #f9e2af; font-size: 11.5px; font-weight: 600;")
        card_layout.addWidget(self.temp_label)

        layout.addWidget(self.telemetry_card)

        # Setup refresh timer (1500 ms)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_telemetry)
        self.timer.start(1500)

        # Initial immediate update
        self.update_telemetry()

    def closeEvent(self, event):
        if hasattr(self, "timer") and self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)

    def set_process_status(self, text: str, state: str = "idle"):
        """Updates the process task indicator.
        States:
        - 'idle': Blue dot (#89b4fa)
        - 'training': Amber dot (#fab387)
        - 'synth': Green dot (#a6e3a1)
        - 'ocr': Purple dot (#cba6f7)
        """
        self.current_state = state
        self.current_task_text = text
        self.status_label.setText(text)

        colors = {
            "idle": "#89b4fa",
            "training": "#fab387",
            "synth": "#a6e3a1",
            "ocr": "#cba6f7",
        }
        color = colors.get(state, "#89b4fa")
        self.status_dot.setStyleSheet(f"color: {color}; font-size: 11px;")

    def update_telemetry(self):
        """Manual update helper (for direct polling / testing)."""
        gpu_data = get_gpu_telemetry()
        if not gpu_data.get("available", False):
            cpu_data = get_cpu_ram_telemetry()
            cpu_data["available"] = False
            self._apply_telemetry_data(cpu_data)
        else:
            self._apply_telemetry_data(gpu_data)

    def _apply_telemetry_data(self, data: Dict[str, Any]):
        """Updates UI labels from telemetry dictionary on the Qt main thread."""
        if data.get("available", False):
            # GPU metrics available
            gpu_util = data["gpu_util"]
            mem_used = data["mem_used_gb"]
            mem_total = data["mem_total_gb"]
            mem_pct = data["mem_pct"]
            temp_c = data["temp_c"]
            name = data["name"]

            self.name_label.setText(name)
            self.util_pill.setText(f"GPU: {gpu_util}%")

            # Dynamic color coding for utilization
            if gpu_util < 50:
                util_color = "#a6e3a1"  # Light green
                bg_color = "#182b24"
            elif gpu_util < 80:
                util_color = "#fab387"  # Peach / amber
                bg_color = "#33261a"
            else:
                util_color = "#f38ba8"  # Red
                bg_color = "#381920"

            self.util_pill.setStyleSheet(
                f"background-color: {bg_color}; color: {util_color}; font-size: 11.5px; font-weight: bold; "
                f"padding: 2px 7px; border-radius: 4px; border: 1px solid {util_color}44;"
            )

            self.mem_label.setText(f"VRAM: {mem_used:.1f} / {mem_total:.1f} GB ({mem_pct}%)")

            if temp_c > 0:
                self.temp_label.setVisible(True)
                temp_color = "#a6e3a1" if temp_c < 65 else ("#fab387" if temp_c < 80 else "#f38ba8")
                self.temp_label.setStyleSheet(f"color: {temp_color}; font-size: 11.5px; font-weight: 600;")
                self.temp_label.setText(f"{temp_c}°C")
            else:
                self.temp_label.setVisible(False)

            self.setToolTip(
                f"<b>GPU Device:</b> {name}<br>"
                f"<b>Compute Utilization:</b> {gpu_util}%<br>"
                f"<b>VRAM Usage:</b> {mem_used:.2f} GB / {mem_total:.2f} GB ({mem_pct}%)<br>"
                f"<b>Temperature:</b> {temp_c}°C<br>"
                f"<b>Process Status:</b> {self.current_task_text}"
            )
        else:
            # Fallback to CPU / Host RAM telemetry
            cpu_pct = data.get("cpu_pct", 0)
            mem_used = data.get("mem_used_gb", 0.0)
            mem_total = data.get("mem_total_gb", 16.0)
            mem_pct = data.get("mem_pct", 0)

            self.dev_icon.setPixmap(qta.icon("fa5s.microchip", color="#89b4fa").pixmap(16, 16))
            self.name_label.setText("CPU Host")
            self.util_pill.setText(f"CPU: {cpu_pct}%")
            self.util_pill.setStyleSheet(
                "background-color: #313244; color: #89b4fa; font-size: 11.5px; font-weight: bold; "
                "padding: 2px 7px; border-radius: 4px;"
            )
            self.mem_label.setText(f"RAM: {mem_used:.1f} / {mem_total:.1f} GB ({mem_pct}%)")
            self.temp_label.setVisible(False)

            self.setToolTip(
                f"<b>Device:</b> CPU Host<br>"
                f"<b>Load:</b> {cpu_pct}%<br>"
                f"<b>RAM:</b> {mem_used:.1f} GB / {mem_total:.1f} GB ({mem_pct}%)<br>"
                f"<b>Process Status:</b> {self.current_task_text}"
            )
