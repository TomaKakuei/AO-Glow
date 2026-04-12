"""Closed-loop feedback control bound directly to the optical model actuator."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

# The local ao311 environment can load duplicate OpenMP runtimes when torch,
# scipy, and optics libraries share a process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from optical_model_rayoptics import DEFAULT_WAVELENGTH_NM, SHIFT_LIMIT_MM
from w_bessel_core import DEFAULT_NUM_MODES, generate_w_bessel_basis, project_phase_map


FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
ArchitectureName = Literal["lightweight_cnn", "resnet18"]
PredictionMode = Literal["absolute", "delta"]
COMPENSATOR_SURFACE_INDEX = 74
OptimizationUpdateCallback = Callable[
    [FloatArray, FloatArray, Float32Array, FloatArray, float],
    None,
]


class OpticalModelProtocol(Protocol):
    """Subset of the optical engine API required by the controller."""

    wavelength_nm: float

    def perturb_lens(
        self,
        surface_index: int,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ) -> Any:
        ...

    def get_psf_image(self, **kwargs: Any) -> Float32Array:
        ...

    def get_wavefront_opd(self, **kwargs: Any) -> FloatArray:
        ...

    def set_surface_perturbations(self, perturbations: dict[int, Any]) -> dict[int, Any]:
        ...


@dataclass(frozen=True)
class InferenceResult:
    predicted_position_mm: FloatArray
    target_position_mm: FloatArray
    applied_position_mm: FloatArray


@dataclass(frozen=True)
class FineAlignmentResult:
    scipy_result: OptimizeResult
    best_position_mm: FloatArray
    best_sharpness: float
    evaluation_count: int


@dataclass(frozen=True)
class ShwfsControlResult:
    delta_position_mm: FloatArray
    applied_position_mm: FloatArray
    residual_signal: FloatArray


@dataclass(frozen=True)
class ShwfsLoopResult:
    final_position_mm: FloatArray
    final_coeffs: FloatArray
    converged: bool
    iteration_count: int


@dataclass
class _ObjectiveTracker:
    best_position_mm: FloatArray
    best_cost: float = np.inf
    best_sharpness: float = -np.inf
    evaluation_count: int = 0
    penalties: int = 0
    history: list[tuple[FloatArray, float]] = field(default_factory=list)


class LightweightPSFCNN(nn.Module):
    """Minimal 5-stage PSF regressor: Conv -> ReLU -> Pool -> Flatten -> Linear."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.head = nn.LazyLinear(output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


def build_lightweight_cnn(*, output_dim: int = 3) -> nn.Module:
    """Create the lightweight 5-layer CNN regressor."""

    return LightweightPSFCNN(output_dim=output_dim)


def build_resnet18_regressor(
    *,
    weights: ResNet18_Weights | None = None,
    output_dim: int = 3,
) -> nn.Module:
    """Create a single-channel ResNet18 regressor with a 3-value output head."""

    model = resnet18(weights=weights)

    conv1 = nn.Conv2d(
        1,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    if weights is not None:
        with torch.no_grad():
            conv1.weight.copy_(model.conv1.weight.mean(dim=1, keepdim=True))
    model.conv1 = conv1
    model.fc = nn.Linear(512, output_dim)
    return model


def build_alignment_model(
    architecture: ArchitectureName = "lightweight_cnn",
    *,
    weights: ResNet18_Weights | None = None,
    output_dim: int = 3,
) -> nn.Module:
    """Build either the lightweight CNN or the modified ResNet18 regressor."""

    if architecture == "lightweight_cnn":
        return build_lightweight_cnn(output_dim=output_dim)
    if architecture == "resnet18":
        return build_resnet18_regressor(weights=weights, output_dim=output_dim)
    raise ValueError(f"Unsupported architecture: {architecture!r}")


def _strip_module_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _normalize_surface_indices(surface_indices: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(surface_indices, (int, np.integer)):
        normalized = (int(surface_indices),)
    else:
        normalized = tuple(int(surface_index) for surface_index in surface_indices)

    if not normalized:
        raise ValueError("At least one actuator surface index is required.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Actuator surface indices must be unique.")
    return normalized


class FeedbackController:
    """Closed-loop controller bound to one or more optical actuators."""

    def __init__(
        self,
        optical_model: OpticalModelProtocol,
        *,
        model: nn.Module | None = None,
        architecture: ArchitectureName = "lightweight_cnn",
        checkpoint_path: str | Path | None = None,
        device: str | torch.device | None = None,
        compensator_surface_index: int = COMPENSATOR_SURFACE_INDEX,
        actuator_surface_indices: int | Sequence[int] | None = None,
        prediction_mode: PredictionMode = "absolute",
    ) -> None:
        self.optical_model = optical_model
        default_surface_indices = (
            compensator_surface_index
            if actuator_surface_indices is None
            else actuator_surface_indices
        )
        self.actuator_surface_indices = _normalize_surface_indices(default_surface_indices)
        self.compensator_surface_index = self.actuator_surface_indices[0]
        self.num_actuators = len(self.actuator_surface_indices)
        self.output_dim = 3 * self.num_actuators
        if prediction_mode not in {"absolute", "delta"}:
            raise ValueError("prediction_mode must be 'absolute' or 'delta'.")
        self.prediction_mode = prediction_mode
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = (
            model
            if model is not None
            else build_alignment_model(architecture, output_dim=self.output_dim)
        ).to(self.device)
        self.model_architecture = architecture
        self.current_position_mm = np.zeros(self.output_dim, dtype=np.float64)
        self.last_inference_position_mm = self.current_position_mm.copy()
        self.last_best_position_mm = self.current_position_mm.copy()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a model state dict from disk."""

        checkpoint = torch.load(
            Path(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            checkpoint_output_dim = checkpoint.get("output_dim")
            if checkpoint_output_dim is not None and int(checkpoint_output_dim) != self.output_dim:
                raise ValueError(
                    "Checkpoint output_dim does not match the configured actuator count."
                )

            checkpoint_surface_indices = checkpoint.get("actuator_surface_indices")
            if checkpoint_surface_indices is not None:
                normalized_surface_indices = _normalize_surface_indices(
                    checkpoint_surface_indices
                )
                if normalized_surface_indices != self.actuator_surface_indices:
                    raise ValueError(
                        "Checkpoint actuator_surface_indices do not match the controller."
                    )

            checkpoint_prediction_mode = checkpoint.get("prediction_mode")
            if checkpoint_prediction_mode in {"absolute", "delta"}:
                self.prediction_mode = checkpoint_prediction_mode

            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif (
                "model_state_dict" in checkpoint
                and isinstance(checkpoint["model_state_dict"], dict)
            ):
                state_dict = checkpoint["model_state_dict"]

        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a valid state_dict.")

        self.model.load_state_dict(_strip_module_prefix(state_dict))
        self.model.eval()

    @staticmethod
    def compute_image_sharpness(psf_image: Float32Array | FloatArray) -> float:
        """Return the sensorless sharpness metric used by the optimizer."""

        psf = np.asarray(psf_image, dtype=np.float64)
        return float(np.sum(psf * psf, dtype=np.float64))

    def _coerce_xyz(self, values: Sequence[float] | FloatArray) -> FloatArray:
        xyz = np.asarray(values, dtype=np.float64).reshape(-1)
        if xyz.size != self.output_dim:
            raise ValueError(
                f"Expected exactly {self.output_dim} actuator values "
                f"({self.num_actuators} surfaces x [dx, dy, dz])."
            )
        return xyz

    def _reshape_xyz(self, values: Sequence[float] | FloatArray) -> FloatArray:
        return self._coerce_xyz(values).reshape(self.num_actuators, 3)

    def _prepare_psf_tensor(self, psf_image: Float32Array | FloatArray) -> Tensor:
        psf = np.asarray(psf_image, dtype=np.float32)
        if psf.ndim != 2:
            raise ValueError("PSF input must be a 2D image.")
        if not np.all(np.isfinite(psf)):
            raise ValueError("PSF input contains non-finite values.")

        # Explicitly reshape to (1, 1, H, W) for single-image inference.
        psf = np.ascontiguousarray(psf[None, None, :, :], dtype=np.float32)
        return torch.from_numpy(psf).to(self.device)

    def _extract_applied_position(
        self,
        perturbation_result: Any,
        fallback_position: FloatArray,
    ) -> FloatArray:
        if perturbation_result is None:
            return fallback_position.astype(np.float64, copy=True)

        components = []
        for attr, fallback_value in zip(
            ("dx_mm", "dy_mm", "dz_mm"),
            fallback_position,
            strict=True,
        ):
            components.append(float(getattr(perturbation_result, attr, fallback_value)))
        return np.asarray(components, dtype=np.float64)

    def _apply_absolute_compensator_position(
        self, absolute_position_mm: Sequence[float] | FloatArray
    ) -> FloatArray:
        target_position = self._coerce_xyz(absolute_position_mm)
        target_blocks = self._reshape_xyz(target_position)

        if hasattr(self.optical_model, "set_surface_perturbations"):
            perturbation_requests = {
                surface_index: block
                for surface_index, block in zip(
                    self.actuator_surface_indices, target_blocks, strict=True
                )
            }
            perturbation_results = self.optical_model.set_surface_perturbations(
                perturbation_requests
            )
            applied_blocks = [
                self._extract_applied_position(
                    perturbation_results.get(surface_index),
                    block,
                )
                for surface_index, block in zip(
                    self.actuator_surface_indices, target_blocks, strict=True
                )
            ]
            applied_position = np.concatenate(applied_blocks, dtype=np.float64)
        else:
            applied_blocks = []
            for surface_index, block in zip(
                self.actuator_surface_indices, target_blocks, strict=True
            ):
                perturbation_result = self.optical_model.perturb_lens(
                    surface_index=surface_index,
                    dx=float(block[0]),
                    dy=float(block[1]),
                    dz=float(block[2]),
                )
                applied_blocks.append(
                    self._extract_applied_position(perturbation_result, block)
                )
            applied_position = np.concatenate(applied_blocks, dtype=np.float64)

        self.current_position_mm = applied_position
        return applied_position

    def set_actuator_positions(
        self, absolute_position_mm: Sequence[float] | FloatArray
    ) -> FloatArray:
        """Set the absolute actuator vector in millimeters."""

        return self._apply_absolute_compensator_position(absolute_position_mm)

    def set_compensator_position(
        self, absolute_position_mm: Sequence[float] | FloatArray
    ) -> FloatArray:
        """Backward-compatible single-actuator setter for surface 74-style control."""

        return self.set_actuator_positions(absolute_position_mm)

    def get_current_snapshot(self) -> tuple[Float32Array, FloatArray, float]:
        """Return the current PSF, wavefront OPD, and sharpness metric."""

        psf_image = np.asarray(self.optical_model.get_psf_image(), dtype=np.float32)
        wavefront_opd = np.asarray(self.optical_model.get_wavefront_opd(), dtype=np.float64)
        sharpness = self.compute_image_sharpness(psf_image)
        return psf_image, wavefront_opd, sharpness

    def infer(self, psf_image: Float32Array | FloatArray) -> FloatArray:
        """Run model inference and return the actuator command vector."""

        psf_tensor = self._prepare_psf_tensor(psf_image)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(psf_tensor).detach().cpu().numpy().reshape(self.output_dim)
        return np.asarray(prediction, dtype=np.float64)

    def infer_and_apply(
        self, psf_image: Float32Array | FloatArray
    ) -> InferenceResult:
        """Predict actuator commands from the PSF and immediately drive the optics."""

        predicted_position = self.infer(psf_image)
        if self.prediction_mode == "absolute":
            target_position = predicted_position
        else:
            target_position = self.current_position_mm + predicted_position

        applied_position = self._apply_absolute_compensator_position(target_position)
        self.last_inference_position_mm = applied_position.copy()
        self.last_best_position_mm = applied_position.copy()
        return InferenceResult(
            predicted_position_mm=predicted_position,
            target_position_mm=target_position,
            applied_position_mm=applied_position,
        )

    def objective_function(
        self,
        xyz_mm: Sequence[float] | FloatArray,
        tracker: _ObjectiveTracker | None = None,
        evaluation_callback: OptimizationUpdateCallback | None = None,
    ) -> float:
        """Physical hardware-loop objective for sensorless fine alignment."""

        candidate_position = self._coerce_xyz(xyz_mm)
        applied_position = self._apply_absolute_compensator_position(candidate_position)

        try:
            psf_image, wavefront_opd, sharpness = self.get_current_snapshot()
            cost = -sharpness
        except (RuntimeError, ValueError):
            psf_image = np.zeros((1, 1), dtype=np.float32)
            wavefront_opd = np.zeros((1, 1), dtype=np.float64)
            sharpness = -np.inf
            cost = float(np.finfo(np.float64).max / 4.0)
            if tracker is not None:
                tracker.penalties += 1

        if tracker is not None:
            tracker.evaluation_count += 1
            tracker.history.append((applied_position.copy(), cost))
            if cost < tracker.best_cost:
                tracker.best_cost = cost
                tracker.best_sharpness = sharpness
                tracker.best_position_mm = applied_position.copy()

        if evaluation_callback is not None:
            evaluation_callback(
                candidate_position.copy(),
                applied_position.copy(),
                psf_image,
                wavefront_opd,
                sharpness,
            )

        return cost

    def run_sensorless_fine_alignment(
        self,
        x0: Sequence[float] | FloatArray | None = None,
        *,
        method: Literal["L-BFGS-B", "Nelder-Mead"] = "L-BFGS-B",
        options: dict[str, Any] | None = None,
        evaluation_callback: OptimizationUpdateCallback | None = None,
    ) -> FineAlignmentResult:
        """Refine the compensator position with SciPy minimize.

        If `x0` is omitted, the controller uses the current actuator state,
        which is typically the coarse CNN output after `infer_and_apply()`.
        """

        initial_guess = (
            self.current_position_mm.copy()
            if x0 is None
            else self._coerce_xyz(x0)
        )
        tracker = _ObjectiveTracker(best_position_mm=initial_guess.copy())

        # Seed the tracker so the best-known position is always valid even if
        # the optimizer exits early or fails.
        self.objective_function(
            initial_guess,
            tracker=tracker,
            evaluation_callback=evaluation_callback,
        )

        bounds = (
            [(-SHIFT_LIMIT_MM, SHIFT_LIMIT_MM)] * self.output_dim
            if method == "L-BFGS-B"
            else None
        )
        solver_options = {"maxiter": 50}
        if options is not None:
            solver_options.update(options)

        try:
            scipy_result = minimize(
                fun=lambda x: self.objective_function(
                    x,
                    tracker=tracker,
                    evaluation_callback=evaluation_callback,
                ),
                x0=initial_guess,
                method=method,
                bounds=bounds,
                options=solver_options,
            )
        finally:
            best_position = tracker.best_position_mm.copy()
            self._apply_absolute_compensator_position(best_position)
            self.last_best_position_mm = best_position.copy()

        return FineAlignmentResult(
            scipy_result=scipy_result,
            best_position_mm=tracker.best_position_mm.copy(),
            best_sharpness=tracker.best_sharpness,
            evaluation_count=tracker.evaluation_count,
        )

    def extract_w_bessel_coeffs(
        self,
        *,
        beam_fill_ratio: float = 1.0,
        num_modes: int = DEFAULT_NUM_MODES,
        wavelength_nm: float | None = None,
    ) -> FloatArray:
        """Project the current optical OPD map onto the weighted-Bessel basis."""

        opd_map = np.asarray(self.optical_model.get_wavefront_opd(), dtype=np.float64)
        if opd_map.ndim != 2 or opd_map.shape[0] != opd_map.shape[1]:
            raise ValueError("Wavefront OPD must be a square 2D array.")

        if wavelength_nm is None:
            wavelength_nm = float(
                getattr(self.optical_model, "wavelength_nm", DEFAULT_WAVELENGTH_NM)
            )

        if hasattr(self.optical_model, "opm") and hasattr(self.optical_model.opm, "nm_to_sys_units"):
            wavelength_sys_units = float(self.optical_model.opm.nm_to_sys_units(wavelength_nm))
        else:
            wavelength_sys_units = 1.0

        wavefront_in_waves = opd_map / wavelength_sys_units
        basis_result = generate_w_bessel_basis(
            grid_size=opd_map.shape[0],
            beam_fill_ratio=beam_fill_ratio,
            num_modes=num_modes,
        )
        return project_phase_map(
            wavefront_in_waves,
            basis_result.basis_cube,
            basis_result.W_xy,
            basis_result.pupil_mask,
        )

    def apply_shwfs_feedback(
        self,
        current_w_bessel_coeffs: Sequence[float] | FloatArray,
        jacobian_matrix: Sequence[Sequence[float]] | FloatArray,
    ) -> ShwfsControlResult:
        """Apply the pseudo-inverse Jacobian control law to the configured actuators.

        The Jacobian maps the actuator vector to the sensed coefficient vector.
        Because the optical engine stores absolute surface positions, the delta
        command is integrated into the current actuator state before the hardware
        call.
        """

        residual_signal = np.asarray(current_w_bessel_coeffs, dtype=np.float64).reshape(-1)
        jacobian = np.asarray(jacobian_matrix, dtype=np.float64)
        if jacobian.ndim != 2 or jacobian.shape[1] != self.output_dim:
            raise ValueError(
                "jacobian_matrix must have shape "
                f"(num_measurements, {self.output_dim})."
            )
        if jacobian.shape[0] != residual_signal.size:
            raise ValueError(
                "jacobian_matrix row count must match the number of residual coefficients."
            )

        delta_position = -np.linalg.pinv(jacobian) @ residual_signal
        absolute_target = self.current_position_mm + delta_position
        applied_position = self._apply_absolute_compensator_position(absolute_target)
        self.last_best_position_mm = applied_position.copy()
        return ShwfsControlResult(
            delta_position_mm=np.asarray(delta_position, dtype=np.float64),
            applied_position_mm=applied_position,
            residual_signal=residual_signal,
        )

    def run_shwfs_matrix_alignment(
        self,
        jacobian_matrix: Sequence[Sequence[float]] | FloatArray,
        *,
        max_iterations: int = 10,
        tolerance: float = 1e-6,
        beam_fill_ratio: float = 1.0,
        num_modes: int = DEFAULT_NUM_MODES,
        wavelength_nm: float | None = None,
        iteration_callback: OptimizationUpdateCallback | None = None,
    ) -> ShwfsLoopResult:
        """Iteratively apply the pseudo-inverse SHWFS matrix controller."""

        final_coeffs = np.zeros(num_modes, dtype=np.float64)
        converged = False
        iteration_count = 0

        for iteration_index in range(max_iterations):
            iteration_count = iteration_index + 1
            current_coeffs = self.extract_w_bessel_coeffs(
                beam_fill_ratio=beam_fill_ratio,
                num_modes=num_modes,
                wavelength_nm=wavelength_nm,
            )
            final_coeffs = current_coeffs
            if np.linalg.norm(current_coeffs) <= tolerance:
                converged = True
                break

            control_result = self.apply_shwfs_feedback(current_coeffs, jacobian_matrix)
            psf_image, wavefront_opd, sharpness = self.get_current_snapshot()
            if iteration_callback is not None:
                target_position = control_result.applied_position_mm.copy()
                iteration_callback(
                    target_position,
                    control_result.applied_position_mm.copy(),
                    psf_image,
                    wavefront_opd,
                    sharpness,
                )

        return ShwfsLoopResult(
            final_position_mm=self.current_position_mm.copy(),
            final_coeffs=np.asarray(final_coeffs, dtype=np.float64),
            converged=converged,
            iteration_count=iteration_count,
        )

    def measure_and_apply_shwfs_feedback(
        self,
        jacobian_matrix: Sequence[Sequence[float]] | FloatArray,
        *,
        beam_fill_ratio: float = 1.0,
        num_modes: int = DEFAULT_NUM_MODES,
        wavelength_nm: float | None = None,
    ) -> ShwfsControlResult:
        """Measure current w-Bessel coefficients from the optical model and close the loop."""

        current_coeffs = self.extract_w_bessel_coeffs(
            beam_fill_ratio=beam_fill_ratio,
            num_modes=num_modes,
            wavelength_nm=wavelength_nm,
        )
        return self.apply_shwfs_feedback(current_coeffs, jacobian_matrix)


__all__ = [
    "ArchitectureName",
    "COMPENSATOR_SURFACE_INDEX",
    "FeedbackController",
    "FineAlignmentResult",
    "InferenceResult",
    "LightweightPSFCNN",
    "OpticalModelProtocol",
    "OptimizationUpdateCallback",
    "PredictionMode",
    "ShwfsControlResult",
    "ShwfsLoopResult",
    "build_alignment_model",
    "build_lightweight_cnn",
    "build_resnet18_regressor",
]
