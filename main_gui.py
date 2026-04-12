"""PyQt6 main GUI tying together the optical model, controller, and view.

Important environment note for this Windows ao311 setup:
`feedback_controller` must be imported before NumPy/Matplotlib so that Torch
loads its DLLs cleanly.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_mplconfig_dir = Path(tempfile.gettempdir()) / "adaptive_optics_mplconfig"
_mplconfig_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfig_dir))

from PyQt6.QtCore import (
    QMargins,
    QObject,
    QPointF,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from feedback_controller import FeedbackController
from optical_model_rayoptics import MechanicalLimitWarning, RayOpticsPhysicsEngine

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


SLIDER_SCALE = 1000
MANUAL_LIMIT_MM = 1.0
DEFAULT_SHWFS_NUM_MODES = 3


def runtime_path(*parts: str) -> Path:
    """Resolve a project/bundle-relative path for source and PyInstaller runs."""

    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    else:
        base_dir = Path(__file__).resolve().parent
    return base_dir.joinpath(*parts)


DEFAULT_CNN_CHECKPOINT_PATH = runtime_path(
    "artifacts",
    "2p_ao_phase_to_comp74_lightweight_cnn_1000samples.pt",
)


@dataclass(frozen=True)
class SnapshotData:
    psf_image: np.ndarray
    wavefront_opd: np.ndarray
    position_mm: np.ndarray
    sharpness: float


class LensSchematicWidget(QWidget):
    """Simple interactive lens schematic with a draggable compensator group."""

    compensatorDragged = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._position_mm = np.zeros(3, dtype=np.float64)
        self._drag_active = False
        self._drag_start_pos = QPointF()
        self._drag_start_xy = np.zeros(2, dtype=np.float64)
        self._warning_active = False
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._clear_warning_flash)

        self._group_boxes = [
            ("Front Group", 0.16, 0.08),
            ("Relay", 0.34, 0.08),
            ("Comp 74-76", 0.50, 0.10),
            ("Rear Group", 0.68, 0.08),
            ("Sample", 0.85, 0.05),
        ]

    def set_position(self, position_mm: np.ndarray | list[float] | tuple[float, ...]) -> None:
        xyz = np.asarray(position_mm, dtype=np.float64).reshape(3)
        self._position_mm = xyz
        self.update()

    def flash_warning(self) -> None:
        self._warning_active = True
        self._flash_timer.start(700)
        self.update()

    def _clear_warning_flash(self) -> None:
        self._warning_active = False
        self.update()

    def _compensator_rect(self) -> tuple[float, float, float, float]:
        width = float(self.width())
        height = float(self.height())
        base_center_x = width * 0.50
        base_center_y = height * 0.50

        dx_px = self._position_mm[0] / MANUAL_LIMIT_MM * width * 0.15
        dy_px = self._position_mm[1] / MANUAL_LIMIT_MM * height * 0.18

        rect_w = width * 0.10
        rect_h = height * 0.34
        left = base_center_x + dx_px - rect_w / 2.0
        top = base_center_y - dy_px - rect_h / 2.0
        return left, top, rect_w, rect_h

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg = QColor("#1c1f24") if self._warning_active else QColor("#f7f5ef")
        fg = QColor("#f4eee2") if self._warning_active else QColor("#1d222b")
        axis = QColor("#ffd0d0") if self._warning_active else QColor("#7c6f64")
        group_color = QColor("#e26d5a")
        passive_color = QColor("#b6c0c9")

        painter.fillRect(self.rect(), bg)

        axis_pen = QPen(axis, 3)
        painter.setPen(axis_pen)
        mid_y = self.height() * 0.52
        painter.drawLine(20, int(mid_y), self.width() - 20, int(mid_y))

        painter.setFont(QFont("Segoe UI", 10))
        for label, cx_norm, w_norm in self._group_boxes:
            cx = self.width() * cx_norm
            box_w = self.width() * w_norm
            box_h = self.height() * 0.24
            top = mid_y - box_h / 2.0

            if "Comp" in label:
                left, top, box_w, box_h = self._compensator_rect()
                color = QColor("#ff5a5f") if self._warning_active else group_color
                painter.setBrush(color)
            else:
                left = cx - box_w / 2.0
                painter.setBrush(passive_color)

            painter.setPen(QPen(fg, 2))
            painter.drawRoundedRect(int(left), int(top), int(box_w), int(box_h), 8, 8)
            painter.drawText(
                int(left),
                int(top - 10),
                int(box_w),
                20,
                Qt.AlignmentFlag.AlignCenter,
                label,
            )

        painter.setPen(QPen(fg, 1))
        readout = (
            f"Compensator Position: dx={self._position_mm[0]:+.3f} mm, "
            f"dy={self._position_mm[1]:+.3f} mm, dz={self._position_mm[2]:+.3f} mm"
        )
        painter.drawText(
            16,
            self.height() - 18,
            self.width() - 32,
            18,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            readout,
        )

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        left, top, width, height = self._compensator_rect()
        if left <= event.position().x() <= left + width and top <= event.position().y() <= top + height:
            self._drag_active = True
            self._drag_start_pos = QPointF(event.position())
            self._drag_start_xy = self._position_mm[:2].copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if not self._drag_active:
            return
        delta = event.position() - self._drag_start_pos
        dx_mm = self._drag_start_xy[0] + (delta.x() / max(1.0, self.width() * 0.15)) * MANUAL_LIMIT_MM
        dy_mm = self._drag_start_xy[1] - (delta.y() / max(1.0, self.height() * 0.18)) * MANUAL_LIMIT_MM
        dx_mm = float(np.clip(dx_mm, -MANUAL_LIMIT_MM, MANUAL_LIMIT_MM))
        dy_mm = float(np.clip(dy_mm, -MANUAL_LIMIT_MM, MANUAL_LIMIT_MM))
        self.compensatorDragged.emit(dx_mm, dy_mm)

    def mouseReleaseEvent(self, event) -> None:
        del event
        if self._drag_active:
            self._drag_active = False
            self.unsetCursor()


class ComparisonCanvas(FigureCanvas):
    """Two-image Matplotlib canvas updated with set_data/draw_idle only."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(5.2, 4.6), tight_layout=True)
        self.figure = figure
        super().__init__(figure)
        self.setParent(parent)

        self.axes_psf = figure.add_subplot(2, 1, 1)
        self.axes_wf = figure.add_subplot(2, 1, 2)
        self.axes_psf.set_title(f"{title}: PSF")
        self.axes_wf.set_title(f"{title}: Wavefront OPD")

        placeholder = np.zeros((32, 32), dtype=np.float32)
        self.psf_image = self.axes_psf.imshow(placeholder, cmap="inferno", origin="lower")
        self.wf_image = self.axes_wf.imshow(placeholder, cmap="RdBu_r", origin="lower")
        self.axes_psf.set_xticks([])
        self.axes_psf.set_yticks([])
        self.axes_wf.set_xticks([])
        self.axes_wf.set_yticks([])

    def update_images(self, psf_image: np.ndarray, wavefront_opd: np.ndarray) -> None:
        psf = np.asarray(psf_image, dtype=np.float32)
        wavefront = np.asarray(wavefront_opd, dtype=np.float32)

        self.psf_image.set_data(psf)
        self.psf_image.set_clim(float(np.min(psf)), float(np.max(psf)) if np.max(psf) > np.min(psf) else float(np.min(psf)) + 1.0)

        wf_min = float(np.min(wavefront))
        wf_max = float(np.max(wavefront))
        if np.isclose(wf_min, wf_max):
            wf_min -= 1.0
            wf_max += 1.0
        wf_abs = max(abs(wf_min), abs(wf_max))
        self.wf_image.set_data(wavefront)
        self.wf_image.set_clim(-wf_abs, wf_abs)

        self.draw_idle()


class SnapshotPanel(QGroupBox):
    """Panel containing a comparison canvas and compact snapshot metadata."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.canvas = ComparisonCanvas(title)
        self.position_label = QLabel("Position: dx=+0.000, dy=+0.000, dz=+0.000 mm")
        self.sharpness_label = QLabel("Sharpness: 0.000000")

        layout.addWidget(self.canvas)
        layout.addWidget(self.position_label)
        layout.addWidget(self.sharpness_label)

    def update_snapshot(self, snapshot: SnapshotData) -> None:
        self.canvas.update_images(snapshot.psf_image, snapshot.wavefront_opd)
        dx, dy, dz = snapshot.position_mm
        self.position_label.setText(
            f"Position: dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f} mm"
        )
        self.sharpness_label.setText(f"Sharpness: {snapshot.sharpness:.6f}")


class OptimizationWorker(QObject):
    """Worker object owning the optical model and control logic in its own thread."""

    initialized = pyqtSignal()
    beforeSnapshotReady = pyqtSignal(object)
    afterSnapshotReady = pyqtSignal(object)
    positionChanged = pyqtSignal(float, float, float)
    statusMessage = pyqtSignal(str)
    warningRaised = pyqtSignal(str)
    busyChanged = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.optical_model: RayOpticsPhysicsEngine | None = None
        self.controller: FeedbackController | None = None
        self._busy = False

    def _ensure_ready(self) -> None:
        if self.optical_model is None or self.controller is None:
            raise RuntimeError("Worker has not been initialized yet.")

    def _set_busy(self, value: bool) -> None:
        if self._busy != value:
            self._busy = value
            self.busyChanged.emit(value)

    def _emit_limit_if_needed(
        self, target_position: np.ndarray, applied_position: np.ndarray
    ) -> None:
        if not np.allclose(target_position, applied_position, atol=1e-9, rtol=0.0):
            self.warningRaised.emit(
                "Mechanical limit reached while moving compensator group."
            )

    def _snapshot(self) -> SnapshotData:
        self._ensure_ready()
        psf_image, wavefront_opd, sharpness = self.controller.get_current_snapshot()
        return SnapshotData(
            psf_image=np.asarray(psf_image, dtype=np.float32),
            wavefront_opd=np.asarray(wavefront_opd, dtype=np.float64),
            position_mm=self.controller.current_position_mm.copy(),
            sharpness=float(sharpness),
        )

    def _emit_before_snapshot(self) -> None:
        self.beforeSnapshotReady.emit(self._snapshot())

    def _emit_after_snapshot(self) -> None:
        snapshot = self._snapshot()
        self.afterSnapshotReady.emit(snapshot)
        dx, dy, dz = snapshot.position_mm
        self.positionChanged.emit(float(dx), float(dy), float(dz))

    def _run_with_warning_capture(self, operation) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", MechanicalLimitWarning)
            result = operation()
        for warning in caught:
            if issubclass(warning.category, MechanicalLimitWarning):
                self.warningRaised.emit(str(warning.message))
        return result

    def _optimization_update(
        self,
        target_position: np.ndarray,
        applied_position: np.ndarray,
        psf_image: np.ndarray,
        wavefront_opd: np.ndarray,
        sharpness: float,
    ) -> None:
        self._emit_limit_if_needed(target_position, applied_position)
        snapshot = SnapshotData(
            psf_image=np.asarray(psf_image, dtype=np.float32),
            wavefront_opd=np.asarray(wavefront_opd, dtype=np.float64),
            position_mm=np.asarray(applied_position, dtype=np.float64),
            sharpness=float(sharpness),
        )
        self.afterSnapshotReady.emit(snapshot)
        dx, dy, dz = snapshot.position_mm
        self.positionChanged.emit(float(dx), float(dy), float(dz))

    def _estimate_jacobian(
        self,
        *,
        num_modes: int = DEFAULT_SHWFS_NUM_MODES,
        step_mm: float = 0.05,
    ) -> np.ndarray:
        self._ensure_ready()
        base_position = self.controller.current_position_mm.copy()
        base_coeffs = self.controller.extract_w_bessel_coeffs(num_modes=num_modes)
        jacobian = np.zeros((num_modes, 3), dtype=np.float64)

        for axis_index in range(3):
            offset_position = base_position.copy()
            offset_position[axis_index] += step_mm
            self._run_with_warning_capture(
                lambda pos=offset_position: self.controller.set_compensator_position(pos)
            )
            coeffs_offset = self.controller.extract_w_bessel_coeffs(num_modes=num_modes)
            jacobian[:, axis_index] = (coeffs_offset - base_coeffs) / step_mm

        self._run_with_warning_capture(
            lambda pos=base_position: self.controller.set_compensator_position(pos)
        )
        self._emit_after_snapshot()

        if np.linalg.norm(jacobian) <= 1.0e-12:
            jacobian = 2.0e4 * np.eye(num_modes, 3)

        return jacobian

    @pyqtSlot()
    def initialize(self) -> None:
        self.statusMessage.emit("Initializing optical model and controller...")
        self.optical_model = RayOpticsPhysicsEngine()
        self.controller = FeedbackController(self.optical_model)
        if DEFAULT_CNN_CHECKPOINT_PATH.exists():
            try:
                self.controller.load_checkpoint(DEFAULT_CNN_CHECKPOINT_PATH)
                self.statusMessage.emit(
                    f"Loaded CNN checkpoint: {DEFAULT_CNN_CHECKPOINT_PATH.name}"
                )
            except Exception as exc:
                self.statusMessage.emit(
                    "Checkpoint load failed; continuing with the untrained CNN. "
                    f"Reason: {exc}"
                )
        self._emit_before_snapshot()
        self._emit_after_snapshot()
        self.statusMessage.emit("System ready.")
        self.initialized.emit()

    @pyqtSlot(float, float, float)
    def set_manual_position(self, dx_mm: float, dy_mm: float, dz_mm: float) -> None:
        if self._busy:
            return
        self._ensure_ready()
        target = np.asarray([dx_mm, dy_mm, dz_mm], dtype=np.float64)
        self.statusMessage.emit("Applying manual compensator position...")
        applied = self._run_with_warning_capture(
            lambda: self.controller.set_compensator_position(target)
        )
        self._emit_limit_if_needed(target, applied)
        self._emit_after_snapshot()

    @pyqtSlot(str)
    def start_auto_correction(self, mode_name: str) -> None:
        if self._busy:
            return

        self._ensure_ready()
        self._set_busy(True)
        try:
            self.statusMessage.emit("Capturing pre-optimization snapshot...")
            self._emit_before_snapshot()

            if mode_name == "CNN Only":
                self.statusMessage.emit("Running CNN-only coarse correction...")
                psf_image = self.optical_model.get_psf_image()
                inference = self._run_with_warning_capture(
                    lambda: self.controller.infer_and_apply(psf_image)
                )
                self._emit_limit_if_needed(
                    inference.target_position_mm,
                    inference.applied_position_mm,
                )
                self._emit_after_snapshot()
                self.statusMessage.emit("CNN-only correction finished.")

            elif mode_name == "Sensorless CNN + Scipy":
                self.statusMessage.emit("Running CNN coarse alignment...")
                psf_image = self.optical_model.get_psf_image()
                inference = self._run_with_warning_capture(
                    lambda: self.controller.infer_and_apply(psf_image)
                )
                self._emit_limit_if_needed(
                    inference.target_position_mm,
                    inference.applied_position_mm,
                )
                self._emit_after_snapshot()

                self.statusMessage.emit("Running sensorless fine alignment...")
                fine_result = self._run_with_warning_capture(
                    lambda: self.controller.run_sensorless_fine_alignment(
                        x0=inference.applied_position_mm,
                        options={"maxiter": 4, "maxfun": 20},
                        evaluation_callback=self._optimization_update,
                    )
                )
                self._emit_after_snapshot()
                self.statusMessage.emit(
                    "Sensorless optimization finished. "
                    f"Best sharpness={fine_result.best_sharpness:.6f}"
                )

            elif mode_name == "SHWFS Matrix":
                self.statusMessage.emit("Estimating SHWFS Jacobian matrix...")
                jacobian = self._estimate_jacobian(num_modes=DEFAULT_SHWFS_NUM_MODES)
                self.statusMessage.emit("Running SHWFS matrix control...")
                shwfs_result = self._run_with_warning_capture(
                    lambda: self.controller.run_shwfs_matrix_alignment(
                        jacobian_matrix=jacobian,
                        max_iterations=8,
                        tolerance=1.0e-3,
                        num_modes=DEFAULT_SHWFS_NUM_MODES,
                        iteration_callback=self._optimization_update,
                    )
                )
                self._emit_after_snapshot()
                self.statusMessage.emit(
                    "SHWFS matrix control finished. "
                    f"Converged={shwfs_result.converged}"
                )
            else:
                raise ValueError(f"Unsupported mode: {mode_name}")

        except Exception as exc:
            self.statusMessage.emit(f"Auto-correction failed: {exc}")
        finally:
            self._set_busy(False)


class MainWindow(QMainWindow):
    """Main application window."""

    manualPositionRequested = pyqtSignal(float, float, float)
    autoCorrectionRequested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Adaptive Optics Alignment GUI")
        self.resize(1520, 980)

        self._selftest_mode = (os.getenv("AO_GUI_SELFTEST_MODE", "") or "").strip() or None
        self._selftest_timeout_ms = int(os.getenv("AO_GUI_SELFTEST_TIMEOUT_MS", "0") or "0")
        self._selftest_started = False
        self._syncing_controls = False
        self._worker_busy = False
        self._warning_timer = QTimer(self)
        self._warning_timer.setSingleShot(True)
        self._warning_timer.timeout.connect(self._clear_warning_label)

        self._manual_timer = QTimer(self)
        self._manual_timer.setSingleShot(True)
        self._manual_timer.setInterval(70)
        self._manual_timer.timeout.connect(self._emit_manual_position_request)

        self._build_ui()
        self._setup_worker_thread()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        top_group = QGroupBox("Lens System / Compensator")
        top_layout = QVBoxLayout(top_group)
        top_layout.setContentsMargins(12, 12, 12, 12)

        self.lens_schematic = LensSchematicWidget()
        self.lens_schematic.compensatorDragged.connect(self._on_schematic_dragged)
        top_layout.addWidget(self.lens_schematic)

        slider_frame = QFrame()
        slider_layout = QGridLayout(slider_frame)
        slider_layout.setContentsMargins(QMargins(0, 0, 0, 0))
        slider_layout.setHorizontalSpacing(10)
        slider_layout.setVerticalSpacing(8)

        self.dx_slider = self._create_axis_slider("dx", 0, slider_layout)
        self.dy_slider = self._create_axis_slider("dy", 1, slider_layout)
        self.dz_slider = self._create_axis_slider("dz", 2, slider_layout)

        top_layout.addWidget(slider_frame)

        self.warning_label = QLabel("No mechanical warnings.")
        self.warning_label.setStyleSheet("color: #5f6b76;")
        self.current_position_label = QLabel("Current Position: dx=+0.000, dy=+0.000, dz=+0.000 mm")
        top_layout.addWidget(self.current_position_label)
        top_layout.addWidget(self.warning_label)

        middle_container = QWidget()
        middle_layout = QHBoxLayout(middle_container)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(12)

        self.before_panel = SnapshotPanel("Before Optimization")
        self.after_panel = SnapshotPanel("After Optimization")
        middle_layout.addWidget(self.before_panel, stretch=1)
        middle_layout.addWidget(self.after_panel, stretch=1)

        bottom_group = QGroupBox("Control Panel")
        bottom_layout = QFormLayout(bottom_group)
        bottom_layout.setContentsMargins(12, 12, 12, 12)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["CNN Only", "SHWFS Matrix", "Sensorless CNN + Scipy"])
        self.start_button = QPushButton("Start Auto-Correction")
        self.start_button.clicked.connect(self._start_auto_correction)
        self.reset_button = QPushButton("Reset Compensator To 0")
        self.reset_button.clicked.connect(self._reset_compensator)
        self.status_label = QLabel("Launching worker...")
        self.status_label.setWordWrap(True)

        button_row = QWidget()
        button_row_layout = QHBoxLayout(button_row)
        button_row_layout.setContentsMargins(0, 0, 0, 0)
        button_row_layout.addWidget(self.start_button)
        button_row_layout.addWidget(self.reset_button)
        button_row_layout.addStretch(1)

        bottom_layout.addRow("Mode", self.mode_combo)
        bottom_layout.addRow("Actions", button_row)
        bottom_layout.addRow("Status", self.status_label)

        root_layout.addWidget(top_group)
        root_layout.addWidget(middle_container, stretch=1)
        root_layout.addWidget(bottom_group)

    def _create_axis_slider(
        self,
        axis_name: str,
        row: int,
        layout: QGridLayout,
    ) -> QSlider:
        label = QLabel(axis_name.upper())
        value_label = QLabel("+0.000 mm")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-int(MANUAL_LIMIT_MM * SLIDER_SCALE), int(MANUAL_LIMIT_MM * SLIDER_SCALE))
        slider.setSingleStep(1)
        slider.setPageStep(25)
        slider.valueChanged.connect(self._on_slider_changed)
        slider.sliderReleased.connect(self._emit_manual_position_request)

        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(value_label, row, 2)

        setattr(self, f"{axis_name}_value_label", value_label)
        return slider

    def _setup_worker_thread(self) -> None:
        self.worker_thread = QThread(self)
        self.worker = OptimizationWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.initialize)
        self.manualPositionRequested.connect(self.worker.set_manual_position)
        self.autoCorrectionRequested.connect(self.worker.start_auto_correction)

        self.worker.beforeSnapshotReady.connect(self._on_before_snapshot)
        self.worker.afterSnapshotReady.connect(self._on_after_snapshot)
        self.worker.positionChanged.connect(self._on_worker_position_changed)
        self.worker.statusMessage.connect(self._on_status_message)
        self.worker.warningRaised.connect(self._on_warning_message)
        self.worker.busyChanged.connect(self._on_busy_changed)
        self.worker.initialized.connect(lambda: self._on_status_message("GUI ready."))
        if self._selftest_mode is not None:
            self.worker.initialized.connect(self._start_selftest_mode)

        self.worker_thread.start()

    @pyqtSlot()
    def _start_selftest_mode(self) -> None:
        if self._selftest_started or self._selftest_mode is None:
            return
        self._selftest_started = True
        mode_index = self.mode_combo.findText(self._selftest_mode)
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        if self._selftest_timeout_ms > 0:
            QTimer.singleShot(self._selftest_timeout_ms, self.close)
        QTimer.singleShot(400, self._start_auto_correction)

    def _slider_values_to_mm(self) -> np.ndarray:
        return np.asarray(
            [
                self.dx_slider.value() / SLIDER_SCALE,
                self.dy_slider.value() / SLIDER_SCALE,
                self.dz_slider.value() / SLIDER_SCALE,
            ],
            dtype=np.float64,
        )

    def _set_slider_values_from_position(self, position_mm: np.ndarray) -> None:
        xyz = np.asarray(position_mm, dtype=np.float64).reshape(3)
        self._syncing_controls = True
        try:
            self.dx_slider.setValue(int(round(xyz[0] * SLIDER_SCALE)))
            self.dy_slider.setValue(int(round(xyz[1] * SLIDER_SCALE)))
            self.dz_slider.setValue(int(round(xyz[2] * SLIDER_SCALE)))
            self._update_slider_labels()
        finally:
            self._syncing_controls = False

    def _update_slider_labels(self) -> None:
        self.dx_value_label.setText(f"{self.dx_slider.value() / SLIDER_SCALE:+.3f} mm")
        self.dy_value_label.setText(f"{self.dy_slider.value() / SLIDER_SCALE:+.3f} mm")
        self.dz_value_label.setText(f"{self.dz_slider.value() / SLIDER_SCALE:+.3f} mm")

    def _on_slider_changed(self) -> None:
        self._update_slider_labels()
        if not self._syncing_controls and not self._worker_busy:
            self._manual_timer.start()

    def _on_schematic_dragged(self, dx_mm: float, dy_mm: float) -> None:
        if self._worker_busy:
            return
        current = self._slider_values_to_mm()
        target = np.asarray([dx_mm, dy_mm, current[2]], dtype=np.float64)
        self._set_slider_values_from_position(target)
        self._manual_timer.start()

    def _emit_manual_position_request(self) -> None:
        if self._syncing_controls or self._worker_busy:
            return
        dx_mm, dy_mm, dz_mm = self._slider_values_to_mm()
        self.manualPositionRequested.emit(float(dx_mm), float(dy_mm), float(dz_mm))

    def _start_auto_correction(self) -> None:
        self.autoCorrectionRequested.emit(self.mode_combo.currentText())

    def _reset_compensator(self) -> None:
        self._set_slider_values_from_position(np.zeros(3, dtype=np.float64))
        self._manual_timer.start()

    @pyqtSlot(object)
    def _on_before_snapshot(self, snapshot: SnapshotData) -> None:
        self.before_panel.update_snapshot(snapshot)

    @pyqtSlot(object)
    def _on_after_snapshot(self, snapshot: SnapshotData) -> None:
        self.after_panel.update_snapshot(snapshot)

    @pyqtSlot(float, float, float)
    def _on_worker_position_changed(self, dx_mm: float, dy_mm: float, dz_mm: float) -> None:
        position = np.asarray([dx_mm, dy_mm, dz_mm], dtype=np.float64)
        self._set_slider_values_from_position(position)
        self.lens_schematic.set_position(position)
        self.current_position_label.setText(
            f"Current Position: dx={dx_mm:+.3f}, dy={dy_mm:+.3f}, dz={dz_mm:+.3f} mm"
        )

    @pyqtSlot(str)
    def _on_status_message(self, message: str) -> None:
        self.status_label.setText(message)
        if self._selftest_mode is not None:
            print(f"[GUI] {message}", flush=True)

    @pyqtSlot(str)
    def _on_warning_message(self, message: str) -> None:
        self.warning_label.setText(message)
        self.warning_label.setStyleSheet("color: #b42318; font-weight: 600;")
        self.lens_schematic.flash_warning()
        self._warning_timer.start(2400)

    def _clear_warning_label(self) -> None:
        self.warning_label.setText("No mechanical warnings.")
        self.warning_label.setStyleSheet("color: #5f6b76;")

    @pyqtSlot(bool)
    def _on_busy_changed(self, busy: bool) -> None:
        self._worker_busy = busy
        self.start_button.setEnabled(not busy)
        self.reset_button.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.dx_slider.setEnabled(not busy)
        self.dy_slider.setEnabled(not busy)
        self.dz_slider.setEnabled(not busy)
        if self._selftest_mode is not None and self._selftest_started and not busy:
            QTimer.singleShot(500, self.close)

    def closeEvent(self, event) -> None:
        self.worker_thread.quit()
        self.worker_thread.wait(5000)
        super().closeEvent(event)


def main() -> int:
    app = QApplication([])
    app.setApplicationName("Adaptive Optics Alignment GUI")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
