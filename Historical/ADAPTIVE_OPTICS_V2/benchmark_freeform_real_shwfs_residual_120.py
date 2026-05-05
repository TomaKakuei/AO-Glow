from __future__ import annotations

import copy
import csv
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.optimize import minimize

from optical_model_rayoptics import MechanicalLimitWarning, RayOpticsPhysicsEngine, SurfacePerturbation
from w_bessel_core import generate_w_bessel_basis, project_phase_map


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
SOURCE_CASE_PATH = ARTIFACTS_DIR / "wb_all_lenses_012_sharpness.json"

BEAM_FILL_RATIO = 1.15
NUM_MODES = 8
CASE_LEVEL_MM = 0.12
SOURCE_LEVEL_MM = 0.012
SCALE_FACTOR = CASE_LEVEL_MM / SOURCE_LEVEL_MM
LOCAL_RADIUS_MM = 0.10
LENSLET_COUNT = 8
SHWFS_MAXITER = 8
SHWFS_MAXFUN = 120
SHWFS_RCOND = 1.0e-3

OUTPUT_JSON_PATH = ARTIFACTS_DIR / "wb_freeform_real_shwfs_residual_120_compare.json"
OUTPUT_CSV_PATH = ARTIFACTS_DIR / "wb_freeform_real_shwfs_residual_120_compare.csv"
OUTPUT_NPZ_PATH = ARTIFACTS_DIR / "wb_freeform_real_shwfs_residual_120_compare_psfs.npz"
OUTPUT_PNG_PATH = ARTIFACTS_DIR / "wb_freeform_real_shwfs_residual_120_compare.png"

warnings.filterwarnings("ignore", category=MechanicalLimitWarning)


@dataclass
class Tracker:
    best_delta_mm: np.ndarray
    best_any_delta_mm: np.ndarray
    best_cost: float = np.inf
    best_any_cost: float = np.inf
    best_sharpness: float = -np.inf
    best_any_sharpness: float = -np.inf
    best_rmse: float = np.inf
    best_any_rmse: float = np.inf
    feasible_count: int = 0
    evaluation_count: int = 0


@dataclass
class ShwfsMeasurementModel:
    lenslet_count: int
    pupil_x: np.ndarray
    pupil_y: np.ndarray
    valid_mask: np.ndarray
    basis_matrix: np.ndarray
    response_matrix: np.ndarray
    reference_coeffs: np.ndarray
    focus_shift_mm: float = 0.0

    def measure_slopes(self, phase_waves: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase_waves, dtype=np.float64)
        measurements: list[float] = []
        edges = np.linspace(-1.0, 1.0, self.lenslet_count + 1)
        for row in range(self.lenslet_count):
            y_lo = edges[row]
            y_hi = edges[row + 1]
            row_mask = (self.pupil_y >= y_lo) & (self.pupil_y < y_hi)
            for col in range(self.lenslet_count):
                x_lo = edges[col]
                x_hi = edges[col + 1]
                cell_mask = self.valid_mask & row_mask & (self.pupil_x >= x_lo) & (self.pupil_x < x_hi)
                if int(np.count_nonzero(cell_mask)) < 4:
                    continue
                x = self.pupil_x[cell_mask].reshape(-1)
                y = self.pupil_y[cell_mask].reshape(-1)
                z = phase[cell_mask].reshape(-1)
                design = np.stack([x, y, np.ones_like(x)], axis=1)
                coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
                measurements.extend([float(coeffs[0]), float(coeffs[1])])
        return np.asarray(measurements, dtype=np.float64)

    def estimate_coeffs(self, optical_model: RayOpticsPhysicsEngine) -> np.ndarray:
        phase_waves = self._current_phase_waves(optical_model)
        slopes = self.measure_slopes(phase_waves)
        coeffs = np.linalg.pinv(self.response_matrix, rcond=SHWFS_RCOND) @ slopes
        return np.asarray(coeffs, dtype=np.float64)

    def _current_phase_waves(self, optical_model: RayOpticsPhysicsEngine) -> np.ndarray:
        _, _, opd_sys_units, _ = optical_model._sample_wavefront(
            num_rays=int(optical_model.pupil_samples),
            field_index=0,
            wavelength_nm=float(optical_model.wavelength_nm),
            focus=float(self.focus_shift_mm),
        )
        wavelength_sys_units = float(optical_model.opm.nm_to_sys_units(float(optical_model.wavelength_nm)))
        return np.asarray(opd_sys_units, dtype=np.float64) / max(wavelength_sys_units, 1.0e-15)


def _load_source_case() -> dict[str, Any]:
    with SOURCE_CASE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["case"]


def _build_system() -> RayOpticsPhysicsEngine:
    return RayOpticsPhysicsEngine()


def _ideal_diffraction_limited_psf(optical_model: RayOpticsPhysicsEngine) -> np.ndarray:
    _, _, _, valid_mask = optical_model._sample_wavefront(
        num_rays=optical_model.pupil_samples,
        field_index=0,
        wavelength_nm=optical_model.wavelength_nm,
        focus=0.0,
    )
    complex_pupil = valid_mask.astype(np.float64)
    pad_total = optical_model.fft_samples - complex_pupil.shape[0]
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded_pupil = np.pad(
        complex_pupil,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=0.0,
    )
    intensity = np.abs(fftshift(fft2(padded_pupil))) ** 2
    intensity = np.asarray(intensity, dtype=np.float64)
    intensity_min = float(np.min(intensity))
    intensity_max = float(np.max(intensity))
    if np.isclose(intensity_max, intensity_min):
        raise RuntimeError("Ideal diffraction PSF is degenerate.")
    return ((intensity - intensity_min) / (intensity_max - intensity_min)).astype(np.float32)


def _sharpness(psf: np.ndarray) -> float:
    psf64 = np.asarray(psf, dtype=np.float64)
    return float(np.sum(psf64 * psf64, dtype=np.float64))


def _wavefront_metrics(optical_model: RayOpticsPhysicsEngine) -> dict[str, float]:
    pupil_samples = int(optical_model.pupil_samples)
    fft_samples = int(optical_model.fft_samples)
    wavelength_nm = float(optical_model.wavelength_nm)
    wavelength_sys_units = float(optical_model.opm.nm_to_sys_units(wavelength_nm))

    best_focus = optical_model.find_best_focus(
        num_rays=pupil_samples,
        field_index=0,
        wavelength_nm=wavelength_nm,
    )
    focus_shift_mm = float(best_focus["best_focus_shift_mm"])

    pupil_x, pupil_y, opd_sys_units, valid_mask = optical_model._sample_wavefront(
        num_rays=pupil_samples,
        field_index=0,
        wavelength_nm=wavelength_nm,
        focus=focus_shift_mm,
    )
    if not np.any(valid_mask):
        raise RuntimeError("Wavefront sampling returned no valid pupil points.")

    opd_centered = np.asarray(opd_sys_units, dtype=np.float64).copy()
    pupil_values = opd_centered[valid_mask]
    pupil_values = pupil_values - float(np.mean(pupil_values))
    opd_centered[valid_mask] = pupil_values

    rms_sys_units = float(np.sqrt(np.mean(pupil_values * pupil_values, dtype=np.float64)))
    rms_waves = float(rms_sys_units / max(wavelength_sys_units, 1.0e-15))
    rms_nm = float(rms_waves * wavelength_nm)

    design_matrix = np.stack(
        [
            np.asarray(pupil_x, dtype=np.float64)[valid_mask],
            np.asarray(pupil_y, dtype=np.float64)[valid_mask],
            np.ones_like(pupil_values, dtype=np.float64),
        ],
        axis=1,
    )
    plane_coeffs, *_ = np.linalg.lstsq(design_matrix, pupil_values, rcond=None)
    high_order_residual = pupil_values - design_matrix @ plane_coeffs
    high_order_rms_sys_units = float(
        np.sqrt(np.mean(high_order_residual * high_order_residual, dtype=np.float64))
    )
    high_order_rms_waves = float(high_order_rms_sys_units / max(wavelength_sys_units, 1.0e-15))
    high_order_rms_nm = float(high_order_rms_waves * wavelength_nm)

    radial = np.asarray(pupil_x, dtype=np.float64)[valid_mask] ** 2 + np.asarray(
        pupil_y, dtype=np.float64
    )[valid_mask] ** 2
    low_order_basis = np.stack(
        [
            np.ones_like(pupil_values, dtype=np.float64),
            np.asarray(pupil_x, dtype=np.float64)[valid_mask],
            np.asarray(pupil_y, dtype=np.float64)[valid_mask],
            radial,
        ],
        axis=1,
    )
    low_order_coeffs, *_ = np.linalg.lstsq(low_order_basis, pupil_values, rcond=None)
    low_order_residual = pupil_values - low_order_basis @ low_order_coeffs
    low_order_rms_sys_units = float(
        np.sqrt(np.mean(low_order_residual * low_order_residual, dtype=np.float64))
    )
    low_order_rms_waves = float(low_order_rms_sys_units / max(wavelength_sys_units, 1.0e-15))
    low_order_rms_nm = float(low_order_rms_waves * wavelength_nm)

    amplitude_mask = valid_mask.astype(np.float64)
    pupil_aberrated = amplitude_mask * np.exp(1j * 2.0 * np.pi * opd_centered / wavelength_sys_units)
    pupil_ideal = amplitude_mask.astype(np.float64)

    pad_total = fft_samples - pupil_aberrated.shape[0]
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded_aberrated = np.pad(
        pupil_aberrated,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=0.0,
    )
    padded_ideal = np.pad(
        pupil_ideal,
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode="constant",
        constant_values=0.0,
    )

    intensity_aberrated = np.abs(fftshift(fft2(padded_aberrated))) ** 2
    intensity_ideal = np.abs(fftshift(fft2(padded_ideal))) ** 2
    strehl_ratio = float(
        np.max(np.asarray(intensity_aberrated, dtype=np.float64))
        / max(float(np.max(np.asarray(intensity_ideal, dtype=np.float64))), 1.0e-15)
    )

    return {
        "best_focus_shift_mm": float(focus_shift_mm),
        "wavefront_rms_sys_units": float(rms_sys_units),
        "wavefront_rms_waves": float(rms_waves),
        "wavefront_rms_nm": float(rms_nm),
        "wavefront_rms_tilt_removed_sys_units": float(high_order_rms_sys_units),
        "wavefront_rms_tilt_removed_waves": float(high_order_rms_waves),
        "wavefront_rms_tilt_removed_nm": float(high_order_rms_nm),
        "wavefront_rms_low_order_removed_sys_units": float(low_order_rms_sys_units),
        "wavefront_rms_low_order_removed_waves": float(low_order_rms_waves),
        "wavefront_rms_low_order_removed_nm": float(low_order_rms_nm),
        "strehl_ratio": float(strehl_ratio),
        "pupil_valid_fraction": float(np.mean(valid_mask.astype(np.float64))),
    }


def _project_w_bessel_coeffs_from_opd(
    optical_model: RayOpticsPhysicsEngine,
    opd_map: np.ndarray,
    *,
    beam_fill_ratio: float = BEAM_FILL_RATIO,
    num_modes: int = NUM_MODES,
) -> np.ndarray:
    opd_map = np.asarray(opd_map, dtype=np.float64)
    wavelength_nm = float(optical_model.wavelength_nm)
    wavelength_sys_units = float(optical_model.opm.nm_to_sys_units(wavelength_nm))
    wavefront_in_waves = opd_map / max(wavelength_sys_units, 1.0e-15)
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


def _true_residual_norm(
    optical_model: RayOpticsPhysicsEngine,
    reference_coeffs: np.ndarray,
    *,
    focus_shift_mm: float = 0.0,
) -> float:
    coeffs = _project_w_bessel_coeffs_from_opd(
        optical_model,
        optical_model.get_wavefront_opd(focus=float(focus_shift_mm)),
        beam_fill_ratio=BEAM_FILL_RATIO,
        num_modes=min(NUM_MODES, int(reference_coeffs.size)),
    )
    reference = np.asarray(reference_coeffs, dtype=np.float64).reshape(-1)
    mode_count = min(coeffs.size, reference.size)
    if mode_count <= 0:
        raise ValueError("Residual comparison requires at least one mode.")
    residual = coeffs[:mode_count] - reference[:mode_count]
    return float(np.linalg.norm(residual))


def _state_metrics(
    optical_model: RayOpticsPhysicsEngine,
    shwfs: ShwfsMeasurementModel,
    psf: np.ndarray,
    *,
    ideal_psf: np.ndarray,
    nominal_psf: np.ndarray,
    true_reference_coeffs: np.ndarray,
    wavefront_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    if wavefront_metrics is None:
        wavefront_metrics = _wavefront_metrics(optical_model)
    estimated_coeffs = shwfs.estimate_coeffs(optical_model)
    estimated_residual = estimated_coeffs - shwfs.reference_coeffs
    return {
        "rmse_to_ideal_perfect": float(
            np.sqrt(
                np.mean(
                    (np.asarray(psf, dtype=np.float64) - np.asarray(ideal_psf, dtype=np.float64)) ** 2,
                    dtype=np.float64,
                )
            )
        ),
        "rmse_to_nominal": float(
            np.sqrt(
                np.mean(
                    (np.asarray(psf, dtype=np.float64) - np.asarray(nominal_psf, dtype=np.float64)) ** 2,
                    dtype=np.float64,
                )
            )
        ),
        "sharpness": _sharpness(psf),
        "residual_norm": float(np.linalg.norm(estimated_residual)),
        "true_residual_norm": _true_residual_norm(
            optical_model,
            true_reference_coeffs,
            focus_shift_mm=float(shwfs.focus_shift_mm),
        ),
        **wavefront_metrics,
    }


def _apply_perturbations(
    optical_model: RayOpticsPhysicsEngine,
    perturbations: list[dict[str, Any]],
    *,
    scale_factor: float,
) -> None:
    for item in perturbations:
        anchor = int(item["anchor_surface_id"])
        dx = float(item["perturbation_dx_mm"]) * scale_factor
        dy = float(item["perturbation_dy_mm"]) * scale_factor
        dz = float(item["perturbation_dz_mm"]) * scale_factor
        optical_model.perturb_lens(anchor, dx, dy, dz)


def _freeform_actuator_ids(optical_model: RayOpticsPhysicsEngine) -> list[int]:
    return optical_model.get_group_anchor_surface_ids(
        include_coverglass=True,
        include_sample_media=False,
    )


def _delta_bounds(
    optical_model: RayOpticsPhysicsEngine,
    actuator_ids: list[int],
    base_state: dict[int, SurfacePerturbation],
    *,
    local_radius_mm: float,
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for surface_id in actuator_ids:
        envelope = optical_model.get_group_mechanical_envelope(surface_id)
        base = base_state.get(surface_id, SurfacePerturbation())
        for lower_abs, upper_abs, base_value in (
            (-envelope.lateral_limit_mm, envelope.lateral_limit_mm, base.dx_mm),
            (-envelope.lateral_limit_mm, envelope.lateral_limit_mm, base.dy_mm),
            (envelope.axial_min_mm, envelope.axial_max_mm, base.dz_mm),
        ):
            lower = max(float(lower_abs) - float(base_value), -local_radius_mm)
            upper = min(float(upper_abs) - float(base_value), local_radius_mm)
            if upper < lower:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            bounds.append((float(lower), float(upper)))
    return bounds


def _delta_state(
    actuator_ids: list[int],
    base_state: dict[int, SurfacePerturbation],
    delta_vector: np.ndarray,
) -> dict[int, SurfacePerturbation]:
    deltas = np.asarray(delta_vector, dtype=np.float64).reshape(len(actuator_ids), 3)
    requested: dict[int, SurfacePerturbation] = {}
    for row_index, surface_id in enumerate(actuator_ids):
        base = base_state.get(surface_id, SurfacePerturbation())
        requested[surface_id] = SurfacePerturbation(
            dx_mm=float(base.dx_mm + deltas[row_index, 0]),
            dy_mm=float(base.dy_mm + deltas[row_index, 1]),
            dz_mm=float(base.dz_mm + deltas[row_index, 2]),
            tilt_x_deg=float(base.tilt_x_deg),
            tilt_y_deg=float(base.tilt_y_deg),
        )
    return requested


def _flatten_state(actuator_ids: list[int], state: dict[int, SurfacePerturbation]) -> list[float]:
    values: list[float] = []
    for surface_id in actuator_ids:
        perturbation = state.get(surface_id, SurfacePerturbation())
        values.extend(
            [
                float(perturbation.dx_mm),
                float(perturbation.dy_mm),
                float(perturbation.dz_mm),
            ]
        )
    return values


def _build_shwfs_measurement_model(
    optical_model: RayOpticsPhysicsEngine,
    *,
    focus_shift_mm: float,
) -> ShwfsMeasurementModel:
    pupil_x, pupil_y, opd_sys_units, valid_mask = optical_model._sample_wavefront(
        num_rays=int(optical_model.pupil_samples),
        field_index=0,
        wavelength_nm=float(optical_model.wavelength_nm),
        focus=float(focus_shift_mm),
    )
    wavelength_sys_units = float(optical_model.opm.nm_to_sys_units(float(optical_model.wavelength_nm)))
    phase_waves = np.asarray(opd_sys_units, dtype=np.float64) / max(wavelength_sys_units, 1.0e-15)

    basis_result = generate_w_bessel_basis(
        grid_size=phase_waves.shape[0],
        beam_fill_ratio=BEAM_FILL_RATIO,
        num_modes=NUM_MODES,
    )

    shwfs = ShwfsMeasurementModel(
        lenslet_count=LENSLET_COUNT,
        pupil_x=np.asarray(pupil_x, dtype=np.float64),
        pupil_y=np.asarray(pupil_y, dtype=np.float64),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        basis_matrix=np.asarray(basis_result.basis_cube, dtype=np.float64),
        response_matrix=np.empty((0, 0), dtype=np.float64),
        reference_coeffs=np.empty(NUM_MODES, dtype=np.float64),
        focus_shift_mm=float(focus_shift_mm),
    )

    response_columns: list[np.ndarray] = []
    for mode_index in range(NUM_MODES):
        slopes = shwfs.measure_slopes(np.asarray(basis_result.basis_cube[mode_index], dtype=np.float64))
        response_columns.append(slopes)
    if not response_columns:
        raise RuntimeError("Failed to build SHWFS response matrix.")
    response_matrix = np.stack(response_columns, axis=1)
    reference_coeffs = np.linalg.pinv(response_matrix, rcond=SHWFS_RCOND) @ shwfs.measure_slopes(phase_waves)

    shwfs.response_matrix = response_matrix
    shwfs.reference_coeffs = np.asarray(reference_coeffs, dtype=np.float64)
    return shwfs


def _run_freeform_case(
    perturbations: list[dict[str, Any]],
    *,
    ideal_psf: np.ndarray,
) -> dict[str, Any]:
    optical_model = _build_system()
    actuator_ids = _freeform_actuator_ids(optical_model)
    nominal_wavefront_metrics = _wavefront_metrics(optical_model)
    nominal_focus_shift_mm = float(nominal_wavefront_metrics["best_focus_shift_mm"])
    shwfs = _build_shwfs_measurement_model(optical_model, focus_shift_mm=nominal_focus_shift_mm)

    nominal_psf = optical_model.get_psf_image(focus=nominal_focus_shift_mm)
    nominal_true_reference_coeffs = _project_w_bessel_coeffs_from_opd(
        optical_model,
        optical_model.get_wavefront_opd(focus=nominal_focus_shift_mm),
        beam_fill_ratio=BEAM_FILL_RATIO,
        num_modes=NUM_MODES,
    )
    nominal_metrics = _state_metrics(
        optical_model,
        shwfs,
        nominal_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=nominal_true_reference_coeffs,
        wavefront_metrics=nominal_wavefront_metrics,
    )

    _apply_perturbations(optical_model, perturbations, scale_factor=SCALE_FACTOR)
    moved_wavefront_metrics = _wavefront_metrics(optical_model)
    moved_focus_shift_mm = float(moved_wavefront_metrics["best_focus_shift_mm"])
    moved_psf = optical_model.get_psf_image(focus=moved_focus_shift_mm)
    moved_metrics = _state_metrics(
        optical_model,
        shwfs,
        moved_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=nominal_true_reference_coeffs,
        wavefront_metrics=moved_wavefront_metrics,
    )

    base_state = optical_model.get_surface_perturbations()
    x0 = np.zeros(3 * len(actuator_ids), dtype=np.float64)
    bounds = _delta_bounds(
        optical_model,
        actuator_ids,
        base_state,
        local_radius_mm=LOCAL_RADIUS_MM,
    )
    tracker = Tracker(best_delta_mm=x0.copy(), best_any_delta_mm=x0.copy())

    def objective(delta_vector: np.ndarray) -> float:
        delta_vector = np.asarray(delta_vector, dtype=np.float64).reshape(-1)
        if delta_vector.size != x0.size:
            raise ValueError("Delta vector length does not match the actuator set.")
        candidate_state = _delta_state(actuator_ids, base_state, delta_vector)
        optical_model.set_surface_perturbations(candidate_state)
        psf_image = optical_model.get_psf_image()
        current_coeffs = shwfs.estimate_coeffs(optical_model)
        residual = current_coeffs - shwfs.reference_coeffs
        residual_norm = float(np.linalg.norm(residual))
        sharpness = _sharpness(psf_image)
        cost = residual_norm
        tracker.evaluation_count += 1
        if residual_norm < tracker.best_cost:
            tracker.best_cost = residual_norm
            tracker.best_rmse = float(
                np.sqrt(
                    np.mean(
                        (np.asarray(psf_image, dtype=np.float64) - np.asarray(ideal_psf, dtype=np.float64)) ** 2,
                        dtype=np.float64,
                    )
                )
            )
            tracker.best_sharpness = sharpness
            tracker.best_delta_mm = delta_vector.copy()
            tracker.feasible_count += 1
        if cost < tracker.best_any_cost:
            tracker.best_any_cost = cost
            tracker.best_any_rmse = float(
                np.sqrt(
                    np.mean(
                        (np.asarray(psf_image, dtype=np.float64) - np.asarray(ideal_psf, dtype=np.float64)) ** 2,
                        dtype=np.float64,
                    )
                )
            )
            tracker.best_any_sharpness = sharpness
            tracker.best_any_delta_mm = delta_vector.copy()
        return cost

    started_at = time.perf_counter()
    warnings_list: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        objective(x0)
        minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": SHWFS_MAXITER, "maxfun": SHWFS_MAXFUN, "ftol": 1.0e-6},
        )
        warnings_list = [str(record.message) for record in warning_records]
    runtime_s = time.perf_counter() - started_at

    best_delta = tracker.best_delta_mm.copy()
    if not np.isfinite(tracker.best_cost):
        best_delta = tracker.best_any_delta_mm.copy()

    repaired_state = _delta_state(actuator_ids, base_state, best_delta)
    optical_model.set_surface_perturbations(repaired_state)
    repaired_wavefront_metrics = _wavefront_metrics(optical_model)
    repaired_focus_shift_mm = float(repaired_wavefront_metrics["best_focus_shift_mm"])
    repaired_psf = optical_model.get_psf_image(focus=repaired_focus_shift_mm)
    repaired_metrics = _state_metrics(
        optical_model,
        shwfs,
        repaired_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=nominal_true_reference_coeffs,
        wavefront_metrics=repaired_wavefront_metrics,
    )

    return {
        "algorithm": "freeform_real_shwfs_residual",
        "settings": {
            "actuator_surface_ids": actuator_ids,
            "beam_fill_ratio": BEAM_FILL_RATIO,
            "num_modes": NUM_MODES,
            "lenslet_count": LENSLET_COUNT,
            "maxiter": SHWFS_MAXITER,
            "maxfun": SHWFS_MAXFUN,
            "local_radius_mm": LOCAL_RADIUS_MM,
            "method": "L-BFGS-B",
            "delta_only": True,
            "measurement_model": "simulated_lenslet_centroid_slopes",
        },
        "nominal": nominal_metrics,
        "moved": moved_metrics,
        "repaired": repaired_metrics,
        "rmse_improvement_vs_ideal_pct": float(
            100.0
            * (moved_metrics["rmse_to_ideal_perfect"] - repaired_metrics["rmse_to_ideal_perfect"])
            / max(moved_metrics["rmse_to_ideal_perfect"], 1.0e-12)
        ),
        "sharpness_change_pct": float(
            100.0 * (repaired_metrics["sharpness"] - moved_metrics["sharpness"])
            / max(abs(moved_metrics["sharpness"]), 1.0e-12)
        ),
        "wavefront_rms_change_pct": float(
            100.0
            * (repaired_metrics["wavefront_rms_waves"] - moved_metrics["wavefront_rms_waves"])
            / max(moved_metrics["wavefront_rms_waves"], 1.0e-12)
        ),
        "wavefront_rms_tilt_removed_change_pct": float(
            100.0
            * (
                repaired_metrics["wavefront_rms_tilt_removed_waves"]
                - moved_metrics["wavefront_rms_tilt_removed_waves"]
            )
            / max(moved_metrics["wavefront_rms_tilt_removed_waves"], 1.0e-12)
        ),
        "strehl_change_pct": float(
            100.0
            * (repaired_metrics["strehl_ratio"] - moved_metrics["strehl_ratio"])
            / max(moved_metrics["strehl_ratio"], 1.0e-12)
        ),
        "estimated_residual_change_pct": float(
            100.0 * (moved_metrics["residual_norm"] - repaired_metrics["residual_norm"])
            / max(moved_metrics["residual_norm"], 1.0e-12)
        ),
        "true_residual_change_pct": float(
            100.0 * (moved_metrics["true_residual_norm"] - repaired_metrics["true_residual_norm"])
            / max(moved_metrics["true_residual_norm"], 1.0e-12)
        ),
        "runtime_s": float(runtime_s),
        "iteration_count": int(tracker.evaluation_count),
        "best_delta_mm": [float(v) for v in best_delta],
        "best_delta_by_anchor": {
            str(surface_id): {
                "dx_mm": float(best_delta[3 * index + 0]),
                "dy_mm": float(best_delta[3 * index + 1]),
                "dz_mm": float(best_delta[3 * index + 2]),
            }
            for index, surface_id in enumerate(actuator_ids)
        },
        "final_position_by_anchor": {
            str(surface_id): {
                "dx_mm": float(repaired_state.get(surface_id, SurfacePerturbation()).dx_mm),
                "dy_mm": float(repaired_state.get(surface_id, SurfacePerturbation()).dy_mm),
                "dz_mm": float(repaired_state.get(surface_id, SurfacePerturbation()).dz_mm),
            }
            for surface_id in actuator_ids
        },
        "best_sharpness": float(tracker.best_any_sharpness),
        "warnings_count": len(warnings_list),
        "warnings": warnings_list[:20],
        "z_bound_flags": {
            str(surface_id): {
                "final_x_within_bounds": bool(
                    -2.0
                    <= float(repaired_state.get(surface_id, SurfacePerturbation()).dx_mm)
                    <= 2.0
                ),
                "final_y_within_bounds": bool(
                    -2.0
                    <= float(repaired_state.get(surface_id, SurfacePerturbation()).dy_mm)
                    <= 2.0
                ),
                "final_z_within_bounds": bool(
                    -0.97
                    <= float(repaired_state.get(surface_id, SurfacePerturbation()).dz_mm)
                    <= 37.43
                ),
            }
            for surface_id in actuator_ids
        },
        "psfs": {
            "nominal": np.asarray(nominal_psf, dtype=np.float32),
            "moved": np.asarray(moved_psf, dtype=np.float32),
            "repaired": np.asarray(repaired_psf, dtype=np.float32),
        },
    }


def _write_json(summary: dict[str, Any]) -> None:
    json_ready = copy.deepcopy(summary)
    for run in json_ready.get("runs", {}).values():
        run.pop("psfs", None)
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(json_ready, handle, indent=2)


def _write_csv(summary: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for algorithm_name, run in summary["runs"].items():
        for state_name in ("nominal", "moved", "repaired"):
            metrics = run[state_name]
            rows.append(
                {
                    "algorithm": algorithm_name,
                    "state": state_name,
                    "rmse_to_ideal_perfect": metrics["rmse_to_ideal_perfect"],
                    "rmse_to_nominal": metrics["rmse_to_nominal"],
                    "sharpness": metrics["sharpness"],
                    "residual_norm": metrics["residual_norm"],
                    "true_residual_norm": metrics["true_residual_norm"],
                    "wavefront_rms_sys_units": metrics["wavefront_rms_sys_units"],
                    "wavefront_rms_waves": metrics["wavefront_rms_waves"],
                    "wavefront_rms_nm": metrics["wavefront_rms_nm"],
                    "wavefront_rms_tilt_removed_sys_units": metrics["wavefront_rms_tilt_removed_sys_units"],
                    "wavefront_rms_tilt_removed_waves": metrics["wavefront_rms_tilt_removed_waves"],
                    "wavefront_rms_tilt_removed_nm": metrics["wavefront_rms_tilt_removed_nm"],
                    "wavefront_rms_low_order_removed_sys_units": metrics["wavefront_rms_low_order_removed_sys_units"],
                    "wavefront_rms_low_order_removed_waves": metrics["wavefront_rms_low_order_removed_waves"],
                    "wavefront_rms_low_order_removed_nm": metrics["wavefront_rms_low_order_removed_nm"],
                    "strehl_ratio": metrics["strehl_ratio"],
                    "pupil_valid_fraction": metrics["pupil_valid_fraction"],
                    "runtime_s": run["runtime_s"] if state_name == "repaired" else "",
                    "iteration_count": run["iteration_count"] if state_name == "repaired" else "",
                    "best_delta_mm": (
                        ";".join(f"{value:.9f}" for value in run["best_delta_mm"])
                        if state_name == "repaired"
                        else ""
                    ),
                }
            )
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(summary: dict[str, Any], ideal_psf: np.ndarray) -> None:
    arrays: dict[str, np.ndarray] = {"ideal_psf": np.asarray(ideal_psf, dtype=np.float32)}
    for algorithm_name, run in summary["runs"].items():
        for state_name, psf in run["psfs"].items():
            arrays[f"{algorithm_name}_{state_name}_psf"] = np.asarray(psf, dtype=np.float32)
    np.savez_compressed(OUTPUT_NPZ_PATH, **arrays)


def _render_montage(summary: dict[str, Any]) -> None:
    run = summary["runs"]["freeform_real_shwfs_residual"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    for col_index, state_name in enumerate(("nominal", "moved", "repaired")):
        ax = axes[0, col_index]
        psf = np.asarray(run["psfs"][state_name], dtype=np.float64)
        im = ax.imshow(psf, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(state_name.capitalize(), fontsize=10)
        metric = run[state_name]
        ax.text(
            0.02,
            0.02,
            f"RMSE={metric['rmse_to_ideal_perfect']:.4f}\n"
            f"S={metric['sharpness']:.1f}\n"
            f"WRMS={metric['wavefront_rms_waves']:.1f}\n"
            f"SR={metric['strehl_ratio']:.4g}",
            transform=ax.transAxes,
            fontsize=7,
            color="white",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.35),
        )
    fig.colorbar(im, ax=axes[0, :], fraction=0.03, pad=0.02)
    fig.suptitle(
        "0.12 mm all-lens perturbation | simulated real SHWFS lenslet-slope w-Bessel residual target",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    source_case = _load_source_case()
    perturbations = list(source_case["perturbations"])
    source_level_mm = float(source_case["level_mm"])
    if not np.isclose(source_level_mm, SOURCE_LEVEL_MM):
        raise RuntimeError(
            f"Source perturbation level mismatch: expected {SOURCE_LEVEL_MM}, got {source_level_mm}"
        )

    baseline_optical_model = _build_system()
    baseline_wavefront_metrics = _wavefront_metrics(baseline_optical_model)
    nominal_focus_shift_mm = float(baseline_wavefront_metrics["best_focus_shift_mm"])
    nominal_psf = baseline_optical_model.get_psf_image(focus=nominal_focus_shift_mm)
    ideal_psf = _ideal_diffraction_limited_psf(baseline_optical_model)
    nominal_true_reference_coeffs = _project_w_bessel_coeffs_from_opd(
        baseline_optical_model,
        baseline_optical_model.get_wavefront_opd(focus=nominal_focus_shift_mm),
        beam_fill_ratio=BEAM_FILL_RATIO,
        num_modes=NUM_MODES,
    )
    nominal_shwfs = _build_shwfs_measurement_model(
        baseline_optical_model,
        focus_shift_mm=nominal_focus_shift_mm,
    )
    nominal_metrics = _state_metrics(
        baseline_optical_model,
        nominal_shwfs,
        nominal_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=nominal_true_reference_coeffs,
        wavefront_metrics=baseline_wavefront_metrics,
    )

    run = _run_freeform_case(perturbations, ideal_psf=ideal_psf)
    summary = {
        "status": "complete",
        "case_level_mm": CASE_LEVEL_MM,
        "source_level_mm": SOURCE_LEVEL_MM,
        "scale_factor": SCALE_FACTOR,
        "nominal": nominal_metrics,
        "ideal_perfect_reference": {
            "rmse_to_ideal_perfect": 0.0,
            "rmse_to_nominal": float(
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(ideal_psf, dtype=np.float64)
                            - np.asarray(nominal_psf, dtype=np.float64)
                        )
                        ** 2,
                        dtype=np.float64,
                    )
                )
            ),
        },
        "runs": {"freeform_real_shwfs_residual": run},
    }

    _write_json(summary)
    _write_csv(summary)
    _write_npz(summary, ideal_psf)
    _render_montage(summary)

    print(f"Wrote {OUTPUT_JSON_PATH}")
    print(f"Wrote {OUTPUT_CSV_PATH}")
    print(f"Wrote {OUTPUT_NPZ_PATH}")
    print(f"Wrote {OUTPUT_PNG_PATH}")


if __name__ == "__main__":
    main()
