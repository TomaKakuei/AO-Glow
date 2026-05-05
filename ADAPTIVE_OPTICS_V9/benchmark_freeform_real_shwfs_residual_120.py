from __future__ import annotations

import copy
import csv
import json
import os
import time
import warnings
from dataclasses import dataclass, field
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    return int(str(raw).strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    return float(str(raw).strip())


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    text = str(raw).strip()
    return text if text else str(default)


def _env_profile_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    return float(str(raw).strip())

BEAM_FILL_RATIO = 1.15
NUM_MODES = 8
CASE_LEVEL_MM = 0.12
SOURCE_LEVEL_MM = 0.012
SCALE_FACTOR = CASE_LEVEL_MM / SOURCE_LEVEL_MM
LOCAL_RADIUS_MM = 0.10
LENSLET_COUNT = _env_int("AO_V5_SHWFS_LENSLET_COUNT", _env_int("AO_V3_SHWFS_LENSLET_COUNT", 13))
SHWFS_MAXITER = 8
SHWFS_MAXFUN = 120
SHWFS_RCOND = 1.0e-3
SHWFS_SLOPE_LIMIT = _env_int("AO_V5_SHWFS_SLOPE_LIMIT", _env_int("AO_V3_SHWFS_SLOPE_LIMIT", 256))
MECHANICAL_RESPONSE_STEP_MM = 5.0e-3
MECHANICAL_RESPONSE_STEP_SCHEDULE_MM = (5.0e-3,)
MECHANICAL_RESPONSE_RCOND = 1.0e-6
SHWFS_NOISE_SEED = _env_int("AO_V5_SHWFS_NOISE_SEED", _env_int("AO_V3_SHWFS_NOISE_SEED", 20260430))
SHWFS_NOISE_PROFILE = _env_str("AO_V5_SHWFS_NOISE_PROFILE", "realistic").strip().lower()
SHWFS_CALIBRATION_NOISE_PROFILE = _env_str("AO_V5_SHWFS_CALIBRATION_NOISE_PROFILE", SHWFS_NOISE_PROFILE).strip().lower()
SHWFS_LENSLET_PIXEL_COUNT = _env_int("AO_V5_SHWFS_LENSLET_PIXEL_COUNT", 12)
SHWFS_FORWARD_AVERAGES = _env_int("AO_V5_SHWFS_FORWARD_AVERAGES", 1)
SHWFS_SIGNATURE_NOISE_SAMPLES = _env_int("AO_V9_SHWFS_SIGNATURE_NOISE_SAMPLES", 8)
SHWFS_CANONICAL_ACTIVE_LENSLETS = _env_int("AO_V9_CANONICAL_ACTIVE_LENSLETS", 128)


@dataclass(frozen=True)
class ShwfsNoiseProfile:
    name: str
    description: str
    bit_depth: int
    adu_max: int
    peak_e: float
    analog_gain_adu_per_e: float
    black_level_adu: float
    read_noise_rms_e: float
    prnu_std: float
    dsnu_mean_e: float
    dsnu_std_e: float
    full_well_e: float = 10000.0
    spot_sigma_px: float = 1.35
    slope_to_pixel_gain: float = 2.4
    max_centroid_shift_px: float = 3.0


NOISE_PROFILES: dict[str, ShwfsNoiseProfile] = {
    "none": ShwfsNoiseProfile(
        name="none",
        description="Idealized slope readout with no sensor-domain noise.",
        bit_depth=16,
        adu_max=65535,
        peak_e=4000.0,
        analog_gain_adu_per_e=2.0,
        black_level_adu=100.0,
        read_noise_rms_e=0.0,
        prnu_std=0.0,
        dsnu_mean_e=0.0,
        dsnu_std_e=0.0,
        full_well_e=10000.0,
    ),
    "minimal": ShwfsNoiseProfile(
        name="minimal",
        description="16-bit top-end sCMOS baseline with near-ideal geometric fitting conditions.",
        bit_depth=16,
        adu_max=65535,
        peak_e=_env_profile_float("AO_V7_MINIMAL_PEAK_E", 4000.0),
        analog_gain_adu_per_e=_env_profile_float("AO_V7_MINIMAL_ANALOG_GAIN_ADU_PER_E", 4.0),
        black_level_adu=_env_profile_float("AO_V7_MINIMAL_BLACK_LEVEL_ADU", 100.0),
        read_noise_rms_e=_env_profile_float("AO_V7_MINIMAL_READ_NOISE_RMS_E", 1.2),
        prnu_std=_env_profile_float("AO_V7_MINIMAL_PRNU_STD", 0.005),
        dsnu_mean_e=_env_profile_float("AO_V7_MINIMAL_DSNU_MEAN_E", 2.0),
        dsnu_std_e=_env_profile_float("AO_V7_MINIMAL_DSNU_STD_E", 0.5),
        full_well_e=_env_profile_float("AO_V7_MINIMAL_FULL_WELL_E", 10000.0),
    ),
    "realistic": ShwfsNoiseProfile(
        name="realistic",
        description="Thorlabs-class 8-bit best-effort operating point with tuned gain and black level.",
        bit_depth=8,
        adu_max=255,
        peak_e=_env_profile_float("AO_V7_REALISTIC_PEAK_E", 800.0),
        analog_gain_adu_per_e=_env_profile_float("AO_V7_REALISTIC_ANALOG_GAIN_ADU_PER_E", 0.305),
        black_level_adu=_env_profile_float("AO_V7_REALISTIC_BLACK_LEVEL_ADU", 10.0),
        read_noise_rms_e=_env_profile_float("AO_V7_REALISTIC_READ_NOISE_RMS_E", 4.0),
        prnu_std=_env_profile_float("AO_V7_REALISTIC_PRNU_STD", 0.015),
        dsnu_mean_e=_env_profile_float("AO_V7_REALISTIC_DSNU_MEAN_E", 15.0),
        dsnu_std_e=_env_profile_float("AO_V7_REALISTIC_DSNU_STD_E", 3.0),
        full_well_e=_env_profile_float("AO_V7_REALISTIC_FULL_WELL_E", 10000.0),
    ),
    "hard": ShwfsNoiseProfile(
        name="hard",
        description="Thorlabs-class 8-bit factory-default low-gain regime with severe quantization loss.",
        bit_depth=8,
        adu_max=255,
        peak_e=300.0,
        analog_gain_adu_per_e=0.03,
        black_level_adu=0.0,
        read_noise_rms_e=6.0,
        prnu_std=0.03,
        dsnu_mean_e=30.0,
        dsnu_std_e=8.0,
        full_well_e=10000.0,
    ),
}


def configure_shwfs_runtime(
    *,
    lenslet_count: int | None = None,
    slope_limit: int | None = None,
    noise_profile: str | None = None,
    noise_seed: int | None = None,
    lenslet_pixel_count: int | None = None,
    forward_averages: int | None = None,
) -> None:
    global LENSLET_COUNT, SHWFS_SLOPE_LIMIT, SHWFS_NOISE_PROFILE, SHWFS_CALIBRATION_NOISE_PROFILE, SHWFS_NOISE_SEED, SHWFS_LENSLET_PIXEL_COUNT, SHWFS_FORWARD_AVERAGES
    if lenslet_count is not None:
        LENSLET_COUNT = max(1, int(lenslet_count))
    if slope_limit is not None:
        SHWFS_SLOPE_LIMIT = max(2, int(slope_limit))
    if noise_profile is not None:
        profile_name = str(noise_profile).strip().lower()
        if profile_name not in NOISE_PROFILES:
            raise ValueError(
                f"Unsupported SHWFS noise profile '{noise_profile}'. Valid profiles: {sorted(NOISE_PROFILES)}"
            )
        SHWFS_NOISE_PROFILE = profile_name
    calib_profile_name = _env_str("AO_V5_SHWFS_CALIBRATION_NOISE_PROFILE", SHWFS_NOISE_PROFILE).strip().lower()
    if calib_profile_name not in NOISE_PROFILES:
        raise ValueError(
            f"Unsupported SHWFS calibration noise profile '{calib_profile_name}'. Valid profiles: {sorted(NOISE_PROFILES)}"
        )
    SHWFS_CALIBRATION_NOISE_PROFILE = calib_profile_name
    if noise_seed is not None:
        SHWFS_NOISE_SEED = int(noise_seed)
    if lenslet_pixel_count is not None:
        SHWFS_LENSLET_PIXEL_COUNT = max(6, int(lenslet_pixel_count))
    if forward_averages is not None:
        SHWFS_FORWARD_AVERAGES = max(1, int(forward_averages))

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
    active_cell_indices: tuple[np.ndarray, ...]
    basis_matrix: np.ndarray
    response_matrix: np.ndarray
    reference_coeffs: np.ndarray
    reference_slopes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    mechanical_response_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    mechanical_response_pinv: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    actuator_ids: tuple[int, ...] = ()
    mechanical_dof_names: tuple[str, ...] = ("dx_mm", "dy_mm", "dz_mm")
    mechanical_step_mm: float = MECHANICAL_RESPONSE_STEP_MM
    mechanical_steps_mm: tuple[float, ...] = MECHANICAL_RESPONSE_STEP_SCHEDULE_MM
    tilt_step_deg: float = 0.0
    focus_shift_mm: float = 0.0
    noise_profile_name: str = SHWFS_NOISE_PROFILE
    noise_seed: int = SHWFS_NOISE_SEED
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(SHWFS_NOISE_SEED))
    lenslet_pixel_count: int = SHWFS_LENSLET_PIXEL_COUNT
    forward_average_count: int = SHWFS_FORWARD_AVERAGES
    noise_profile: ShwfsNoiseProfile | None = None
    pixel_x_grid: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    pixel_y_grid: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    gain_maps: tuple[np.ndarray, ...] = ()
    offset_maps_e: tuple[np.ndarray, ...] = ()
    coeff_noise_std: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __post_init__(self) -> None:
        profile_name = str(self.noise_profile_name).strip().lower()
        if profile_name not in NOISE_PROFILES:
            raise ValueError(f"Unsupported SHWFS noise profile '{self.noise_profile_name}'.")
        self.noise_profile_name = profile_name
        self.noise_profile = NOISE_PROFILES[profile_name]
        self.forward_average_count = max(1, int(self.forward_average_count))
        pixel_count = max(6, int(self.lenslet_pixel_count))
        pixel_axis = np.arange(pixel_count, dtype=np.float64)
        self.pixel_x_grid, self.pixel_y_grid = np.meshgrid(pixel_axis, pixel_axis)
        fixed_rng = np.random.default_rng(int(self.noise_seed))
        self.rng = np.random.default_rng(int(self.noise_seed) + 1)
        gain_maps: list[np.ndarray] = []
        offset_maps: list[np.ndarray] = []
        for _ in range(len(self.active_cell_indices)):
            if self.noise_profile.prnu_std > 0.0:
                gain_map = 1.0 + self.noise_profile.prnu_std * fixed_rng.standard_normal(
                    (pixel_count, pixel_count)
                )
                gain_map = np.clip(gain_map, 0.5, 1.5)
            else:
                gain_map = np.ones((pixel_count, pixel_count), dtype=np.float64)
            if self.noise_profile.dsnu_std_e > 0.0 or self.noise_profile.dsnu_mean_e > 0.0:
                offset_map = self.noise_profile.dsnu_mean_e + self.noise_profile.dsnu_std_e * fixed_rng.standard_normal(
                    (pixel_count, pixel_count)
                )
                offset_map = np.clip(offset_map, 0.0, None)
            else:
                offset_map = np.zeros((pixel_count, pixel_count), dtype=np.float64)
            gain_maps.append(np.asarray(gain_map, dtype=np.float64))
            offset_maps.append(np.asarray(offset_map, dtype=np.float64))
        self.gain_maps = tuple(gain_maps)
        self.offset_maps_e = tuple(offset_maps)

    def measure_slopes(self, phase_waves: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase_waves, dtype=np.float64)
        measurements: list[float] = []
        pupil_x = np.asarray(self.pupil_x, dtype=np.float64).reshape(-1)
        pupil_y = np.asarray(self.pupil_y, dtype=np.float64).reshape(-1)
        phase_flat = phase.reshape(-1)
        for lenslet_index, flat_indices in enumerate(self.active_cell_indices):
            x = pupil_x[flat_indices]
            y = pupil_y[flat_indices]
            z = phase_flat[flat_indices]
            design = np.stack([x, y, np.ones_like(x)], axis=1)
            coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
            slope_xy = self._measure_lenslet_centroid_slopes(np.asarray(coeffs[:2], dtype=np.float64), lenslet_index)
            measurements.extend([float(slope_xy[0]), float(slope_xy[1])])
        return np.asarray(measurements, dtype=np.float64)

    def measure_model_slopes(self, optical_model: RayOpticsPhysicsEngine) -> np.ndarray:
        phase = self._current_phase_waves(optical_model)
        if self.forward_average_count <= 1:
            return self.measure_slopes(phase)
        samples = [self.measure_slopes(phase) for _ in range(int(self.forward_average_count))]
        return np.mean(np.stack(samples, axis=0), axis=0, dtype=np.float64)

    def estimate_coeffs(self, optical_model: RayOpticsPhysicsEngine) -> np.ndarray:
        slopes = self.measure_model_slopes(optical_model)
        coeffs = np.linalg.pinv(self.response_matrix, rcond=SHWFS_RCOND) @ slopes
        return np.asarray(coeffs, dtype=np.float64)

    def estimate_from_model(self, optical_model: RayOpticsPhysicsEngine) -> tuple[np.ndarray, np.ndarray]:
        slopes = self.measure_model_slopes(optical_model)
        coeffs = np.linalg.pinv(self.response_matrix, rcond=SHWFS_RCOND) @ slopes
        return np.asarray(slopes, dtype=np.float64), np.asarray(coeffs, dtype=np.float64)

    def estimate_mechanical_delta(self, slope_residual: np.ndarray) -> np.ndarray:
        residual = np.asarray(slope_residual, dtype=np.float64).reshape(-1)
        if self.mechanical_response_pinv.size == 0:
            raise RuntimeError("Mechanical SHWFS response pseudo-inverse has not been initialized.")
        if self.mechanical_response_pinv.shape[1] != residual.size:
            raise ValueError("Slope residual size does not match the mechanical response model.")
        return np.asarray(self.mechanical_response_pinv @ residual, dtype=np.float64)

    def estimate_nominal_correction_seed(self, optical_model: RayOpticsPhysicsEngine) -> np.ndarray:
        current_slopes = self.measure_model_slopes(optical_model)
        if self.reference_slopes.size != current_slopes.size:
            raise RuntimeError("Reference SHWFS slopes are unavailable for blind seed estimation.")
        return -self.estimate_mechanical_delta(current_slopes - self.reference_slopes)

    def _current_phase_waves(self, optical_model: RayOpticsPhysicsEngine) -> np.ndarray:
        _, _, opd_sys_units, _ = optical_model._sample_wavefront(
            num_rays=int(optical_model.pupil_samples),
            field_index=0,
            wavelength_nm=float(optical_model.wavelength_nm),
            focus=float(self.focus_shift_mm),
        )
        wavelength_sys_units = float(optical_model.opm.nm_to_sys_units(float(optical_model.wavelength_nm)))
        return np.asarray(opd_sys_units, dtype=np.float64) / max(wavelength_sys_units, 1.0e-15)

    def _measure_lenslet_centroid_slopes(self, ideal_slope_xy: np.ndarray, lenslet_index: int) -> np.ndarray:
        slope_xy = np.asarray(ideal_slope_xy, dtype=np.float64).reshape(2)
        if self.noise_profile is None or self.noise_profile.name == "none":
            return slope_xy
        center_px = 0.5 * float(self.lenslet_pixel_count - 1)
        shift_xy_px = np.clip(
            slope_xy * float(self.noise_profile.slope_to_pixel_gain),
            -float(self.noise_profile.max_centroid_shift_px),
            float(self.noise_profile.max_centroid_shift_px),
        )
        radius_sq = (
            (self.pixel_x_grid - (center_px + shift_xy_px[0])) ** 2
            + (self.pixel_y_grid - (center_px + shift_xy_px[1])) ** 2
        )
        ideal_spot_normalized = np.exp(
            -radius_sq / max(2.0 * float(self.noise_profile.spot_sigma_px) ** 2, 1.0e-12)
        )
        ideal_electrons = ideal_spot_normalized * float(self.noise_profile.peak_e)
        gained_electrons = ideal_electrons * self.gain_maps[int(lenslet_index)]
        base_electrons = np.clip(gained_electrons + self.offset_maps_e[int(lenslet_index)], 0.0, None)
        shot_electrons = self.rng.poisson(base_electrons).astype(np.float64)
        silicon_electrons = np.clip(shot_electrons, 0.0, float(self.noise_profile.full_well_e))
        read_noise_e = self.noise_profile.read_noise_rms_e * self.rng.standard_normal(silicon_electrons.shape)
        adc_input_electrons = np.clip(silicon_electrons + read_noise_e, 0.0, None)
        sensor_adu = np.clip(
            np.rint(
                adc_input_electrons * max(float(self.noise_profile.analog_gain_adu_per_e), 0.0)
                + float(self.noise_profile.black_level_adu)
            ),
            0.0,
            float(self.noise_profile.adu_max),
        )
        total_signal = float(np.sum(sensor_adu, dtype=np.float64))
        if total_signal <= 1.0e-12:
            return np.zeros(2, dtype=np.float64)
        centroid_x = float(np.sum(self.pixel_x_grid * sensor_adu, dtype=np.float64) / total_signal)
        centroid_y = float(np.sum(self.pixel_y_grid * sensor_adu, dtype=np.float64) / total_signal)
        measured_shift_xy = np.asarray([centroid_x - center_px, centroid_y - center_px], dtype=np.float64)
        return measured_shift_xy / max(float(self.noise_profile.slope_to_pixel_gain), 1.0e-12)


def _load_source_case() -> dict[str, Any]:
    with SOURCE_CASE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)["case"]


def _build_system() -> RayOpticsPhysicsEngine:
    return RayOpticsPhysicsEngine()


def _build_active_lenslet_cells(
    pupil_x: np.ndarray,
    pupil_y: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lenslet_count: int,
    slope_limit: int,
) -> tuple[np.ndarray, ...]:
    edges = np.linspace(-1.0, 1.0, int(lenslet_count) + 1)
    candidates: list[tuple[float, int, int, np.ndarray]] = []
    for row in range(int(lenslet_count)):
        y_lo = edges[row]
        y_hi = edges[row + 1]
        row_mask = (pupil_y >= y_lo) & (pupil_y < y_hi)
        for col in range(int(lenslet_count)):
            x_lo = edges[col]
            x_hi = edges[col + 1]
            cell_mask = valid_mask & row_mask & (pupil_x >= x_lo) & (pupil_x < x_hi)
            flat_indices = np.flatnonzero(cell_mask.reshape(-1))
            if int(flat_indices.size) < 4:
                continue
            cell_center_x = 0.5 * (x_lo + x_hi)
            cell_center_y = 0.5 * (y_lo + y_hi)
            radius_sq = float(cell_center_x * cell_center_x + cell_center_y * cell_center_y)
            candidates.append((radius_sq, -int(flat_indices.size), row * int(lenslet_count) + col, flat_indices))

    if not candidates:
        return ()

    max_lenslets = max(1, int(slope_limit) // 2)
    if SHWFS_CANONICAL_ACTIVE_LENSLETS > 0:
        max_lenslets = min(max_lenslets, int(SHWFS_CANONICAL_ACTIVE_LENSLETS))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = candidates[:max_lenslets]
    selected.sort(key=lambda item: item[2])
    return tuple(np.asarray(item[3], dtype=np.int64) for item in selected)


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
    flat = np.asarray(delta_vector, dtype=np.float64).reshape(-1)
    if len(actuator_ids) == 0:
        return {}
    dof_count, remainder = divmod(int(flat.size), int(len(actuator_ids)))
    if remainder != 0 or dof_count not in {3, 5}:
        raise ValueError("Delta vector must contain either 3 or 5 values per actuator.")
    deltas = flat.reshape(len(actuator_ids), dof_count)
    requested: dict[int, SurfacePerturbation] = {}
    for row_index, surface_id in enumerate(actuator_ids):
        base = base_state.get(surface_id, SurfacePerturbation())
        requested[surface_id] = SurfacePerturbation(
            dx_mm=float(base.dx_mm + deltas[row_index, 0]),
            dy_mm=float(base.dy_mm + deltas[row_index, 1]),
            dz_mm=float(base.dz_mm + deltas[row_index, 2]),
            tilt_x_deg=float(base.tilt_x_deg + (deltas[row_index, 3] if dof_count >= 5 else 0.0)),
            tilt_y_deg=float(base.tilt_y_deg + (deltas[row_index, 4] if dof_count >= 5 else 0.0)),
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


def _single_axis_perturbation(axis_name: str, amplitude: float) -> SurfacePerturbation:
    values = {"dx_mm": 0.0, "dy_mm": 0.0, "dz_mm": 0.0, "tilt_x_deg": 0.0, "tilt_y_deg": 0.0}
    values[str(axis_name)] = float(amplitude)
    return SurfacePerturbation(**values)


def _build_mechanical_response_matrix(
    optical_model: RayOpticsPhysicsEngine,
    shwfs: ShwfsMeasurementModel,
    actuator_ids: list[int],
    *,
    step_schedule_mm: tuple[float, ...],
) -> np.ndarray:
    cleaned_steps = tuple(
        sorted(
            {
                float(abs(step_mm))
                for step_mm in step_schedule_mm
                if np.isfinite(float(step_mm)) and float(abs(step_mm)) > 0.0
            }
        )
    )
    if not cleaned_steps:
        raise ValueError("Mechanical response step schedule must contain at least one positive step.")

    baseline_state = optical_model.get_surface_perturbations()
    baseline_slopes = np.asarray(shwfs.reference_slopes, dtype=np.float64).reshape(-1)
    columns: list[np.ndarray] = []

    try:
        for anchor_surface_id in actuator_ids:
            for axis_name in shwfs.mechanical_dof_names:
                derivative_estimates: list[np.ndarray] = []
                if axis_name in {"tilt_x_deg", "tilt_y_deg"}:
                    step_schedule = (float(shwfs.tilt_step_deg),)
                else:
                    step_schedule = cleaned_steps
                for step_mm in step_schedule:
                    optical_model.clear_perturbation()
                    if baseline_state:
                        optical_model.set_surface_perturbations(baseline_state)
                    optical_model.set_surface_perturbations(
                        {int(anchor_surface_id): _single_axis_perturbation(axis_name, step_mm)}
                    )
                    plus_slopes = shwfs.measure_model_slopes(optical_model) - baseline_slopes

                    optical_model.clear_perturbation()
                    if baseline_state:
                        optical_model.set_surface_perturbations(baseline_state)
                    optical_model.set_surface_perturbations(
                        {int(anchor_surface_id): _single_axis_perturbation(axis_name, -step_mm)}
                    )
                    minus_slopes = shwfs.measure_model_slopes(optical_model) - baseline_slopes
                    derivative_estimates.append((plus_slopes - minus_slopes) / (2.0 * float(step_mm)))
                columns.append(np.mean(np.stack(derivative_estimates, axis=0), axis=0))
    finally:
        optical_model.clear_perturbation()
        if baseline_state:
            optical_model.set_surface_perturbations(baseline_state)

    if not columns:
        return np.empty((baseline_slopes.size, 0), dtype=np.float64)
    return np.stack(columns, axis=1)


def _build_shwfs_measurement_model(
    optical_model: RayOpticsPhysicsEngine,
    *,
    focus_shift_mm: float,
    actuator_ids: list[int] | None = None,
    mechanical_step_mm: float = MECHANICAL_RESPONSE_STEP_MM,
    mechanical_step_schedule_mm: tuple[float, ...] = MECHANICAL_RESPONSE_STEP_SCHEDULE_MM,
    mechanical_dof_names: tuple[str, ...] = ("dx_mm", "dy_mm", "dz_mm"),
    tilt_step_deg: float = 0.0,
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
        active_cell_indices=_build_active_lenslet_cells(
            np.asarray(pupil_x, dtype=np.float64),
            np.asarray(pupil_y, dtype=np.float64),
            np.asarray(valid_mask, dtype=bool),
            lenslet_count=LENSLET_COUNT,
            slope_limit=SHWFS_SLOPE_LIMIT,
        ),
        basis_matrix=np.asarray(basis_result.basis_cube, dtype=np.float64),
        response_matrix=np.empty((0, 0), dtype=np.float64),
        reference_coeffs=np.empty(NUM_MODES, dtype=np.float64),
        actuator_ids=tuple(int(v) for v in (actuator_ids or [])),
        mechanical_dof_names=tuple(str(v) for v in mechanical_dof_names),
        mechanical_step_mm=float(mechanical_step_mm),
        mechanical_steps_mm=tuple(float(v) for v in mechanical_step_schedule_mm),
        tilt_step_deg=float(tilt_step_deg),
        focus_shift_mm=float(focus_shift_mm),
        noise_profile_name=str(SHWFS_CALIBRATION_NOISE_PROFILE),
        noise_seed=int(SHWFS_NOISE_SEED),
        rng=np.random.default_rng(int(SHWFS_NOISE_SEED)),
        lenslet_pixel_count=int(SHWFS_LENSLET_PIXEL_COUNT),
        forward_average_count=int(SHWFS_FORWARD_AVERAGES),
    )

    response_columns: list[np.ndarray] = []
    for mode_index in range(NUM_MODES):
        slopes = shwfs.measure_slopes(np.asarray(basis_result.basis_cube[mode_index], dtype=np.float64))
        response_columns.append(slopes)
    if not response_columns:
        raise RuntimeError("Failed to build SHWFS response matrix.")
    response_matrix = np.stack(response_columns, axis=1)
    reference_slopes = shwfs.measure_slopes(phase_waves)
    if int(reference_slopes.size) > SHWFS_SLOPE_LIMIT:
        raise RuntimeError(
            f"SHWFS slope count {int(reference_slopes.size)} exceeds the configured limit of {SHWFS_SLOPE_LIMIT}."
        )
    reference_coeffs = np.linalg.pinv(response_matrix, rcond=SHWFS_RCOND) @ reference_slopes
    shwfs.reference_slopes = np.asarray(reference_slopes, dtype=np.float64)
    mechanical_response_matrix = _build_mechanical_response_matrix(
        optical_model,
        shwfs,
        list(actuator_ids or []),
        step_schedule_mm=tuple(float(v) for v in mechanical_step_schedule_mm),
    )

    shwfs.response_matrix = response_matrix
    shwfs.reference_coeffs = np.asarray(reference_coeffs, dtype=np.float64)
    shwfs.mechanical_response_matrix = np.asarray(mechanical_response_matrix, dtype=np.float64)
    if shwfs.mechanical_response_matrix.size > 0:
        shwfs.mechanical_response_pinv = np.linalg.pinv(
            shwfs.mechanical_response_matrix,
            rcond=MECHANICAL_RESPONSE_RCOND,
        )
    shwfs.noise_profile_name = str(SHWFS_NOISE_PROFILE)
    shwfs.noise_profile = NOISE_PROFILES[str(SHWFS_NOISE_PROFILE)]
    if SHWFS_SIGNATURE_NOISE_SAMPLES > 1:
        coeff_samples = []
        for _ in range(int(SHWFS_SIGNATURE_NOISE_SAMPLES)):
            noisy_reference_slopes = shwfs.measure_slopes(phase_waves)
            noisy_reference_coeffs = np.linalg.pinv(response_matrix, rcond=SHWFS_RCOND) @ noisy_reference_slopes
            coeff_samples.append(np.asarray(noisy_reference_coeffs, dtype=np.float64))
        coeff_sample_array = np.stack(coeff_samples, axis=0)
        coeff_noise_std = np.std(coeff_sample_array, axis=0, dtype=np.float64, ddof=0)
        positive = coeff_noise_std[coeff_noise_std > 1.0e-12]
        floor = float(np.median(positive)) if positive.size else 1.0
        shwfs.coeff_noise_std = np.maximum(np.asarray(coeff_noise_std, dtype=np.float64), floor * 0.5)
    else:
        shwfs.coeff_noise_std = np.ones(NUM_MODES, dtype=np.float64)
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
            "lenslet_pixel_count": SHWFS_LENSLET_PIXEL_COUNT,
            "maxiter": SHWFS_MAXITER,
            "maxfun": SHWFS_MAXFUN,
            "local_radius_mm": LOCAL_RADIUS_MM,
            "method": "L-BFGS-B",
            "delta_only": True,
            "measurement_model": "simulated_lenslet_centroid_slopes",
            "noise_profile": str(SHWFS_NOISE_PROFILE),
            "noise_profile_parameters": {
                "bit_depth": int(shwfs.noise_profile.bit_depth if shwfs.noise_profile else 0),
                "peak_e": float(shwfs.noise_profile.peak_e if shwfs.noise_profile else 0.0),
                "analog_gain_adu_per_e": float(
                    shwfs.noise_profile.analog_gain_adu_per_e if shwfs.noise_profile else 0.0
                ),
                "black_level_adu": float(shwfs.noise_profile.black_level_adu if shwfs.noise_profile else 0.0),
                "read_noise_rms_e": float(shwfs.noise_profile.read_noise_rms_e if shwfs.noise_profile else 0.0),
                "prnu_std": float(shwfs.noise_profile.prnu_std if shwfs.noise_profile else 0.0),
                "dsnu_mean_e": float(shwfs.noise_profile.dsnu_mean_e if shwfs.noise_profile else 0.0),
                "dsnu_std_e": float(shwfs.noise_profile.dsnu_std_e if shwfs.noise_profile else 0.0),
                "full_well_e": float(shwfs.noise_profile.full_well_e if shwfs.noise_profile else 0.0),
                "adu_max": int(shwfs.noise_profile.adu_max if shwfs.noise_profile else 0),
            },
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
