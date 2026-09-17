"""Unit tests for GPU & Process Telemetry Monitor."""

import pytest
from khmer_ocr.ui.widgets.gpu_monitor import get_gpu_telemetry, get_cpu_ram_telemetry, LiveTelemetryWidget
from PyQt5.QtWidgets import QApplication
import sys


def test_cpu_ram_telemetry():
    data = get_cpu_ram_telemetry()
    assert "cpu_pct" in data
    assert "mem_used_gb" in data
    assert "mem_total_gb" in data
    assert "mem_pct" in data
    assert data["mem_total_gb"] > 0
    assert 0 <= data["mem_pct"] <= 100


def test_gpu_telemetry():
    data = get_gpu_telemetry()
    assert "available" in data
    if data["available"]:
        assert "gpu_util" in data
        assert "mem_used_gb" in data
        assert "mem_total_gb" in data
        assert "name" in data
        assert data["mem_total_gb"] > 0


def test_live_telemetry_widget_init():
    app = QApplication.instance() or QApplication(sys.argv)
    widget = LiveTelemetryWidget()
    assert widget.status_label.text() == "Idle"

    # Test status update
    widget.set_process_status("Training (Epoch 1/10)", "training")
    assert widget.status_label.text() == "Training (Epoch 1/10)"
    assert widget.current_state == "training"

    widget.set_process_status("Idle", "idle")
    assert widget.status_label.text() == "Idle"
    assert widget.current_state == "idle"
