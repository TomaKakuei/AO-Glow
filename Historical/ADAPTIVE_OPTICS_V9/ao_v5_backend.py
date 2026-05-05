from __future__ import annotations

import copy
import csv
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import benchmark_1mm_real_shwfs_converging_920nm as bench
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal
from optical_model_rayoptics import MechanicalLimitWarning, RayOpticsPhysicsEngine, SurfacePerturbation
from w_bessel_core import nm_sequence


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
BASE_GEOMETRY_PATH = ARTIFACTS_DIR / "geometry_inspection_1000_after_coupled_gap.json"
ACTUATOR_IDS = [68, 70, 72, 74, 77, 79, 81, 84]
SEARCH_BOX_SCALE = 1.5
SEARCH_BOX_MIN_MM = 0.02
BLIND_SEED_DZ_WEIGHT = 0.25
TIPTILT_TEST_LIMIT_ARCMIN = float(os.environ.get("AO_V5_TIPTILT_TEST_LIMIT_ARCMIN", os.environ.get("AO_V4_1_TIPTILT_TEST_LIMIT_ARCMIN", "6.0")))
BLOCK_STAGE_MAXITER = int(os.environ.get("AO_V5_BLOCK_STAGE_MAXITER", os.environ.get("AO_V4_BLOCK_STAGE_MAXITER", "2")))
ENABLE_TIPTILT = os.environ.get("AO_V3_ENABLE_TIPTILT", "").strip().lower() in {"1", "true", "yes", "on"}
TIPTILT_LOCAL_LIMIT_ARCMIN = float(os.environ.get("AO_V3_TIPTILT_LIMIT_ARCMIN", "10.0"))
TIPTILT_LOCAL_LIMIT_DEG = float(TIPTILT_LOCAL_LIMIT_ARCMIN / 60.0)
TIPTILT_RESPONSE_STEP_ARCMIN = float(os.environ.get("AO_V3_TIPTILT_RESPONSE_STEP_ARCMIN", "1.0"))
TIPTILT_RESPONSE_STEP_DEG = float(TIPTILT_RESPONSE_STEP_ARCMIN / 60.0)
ENABLE_BLIND_SEED = os.environ.get("AO_V3_DISABLE_BLIND_SEED", "").strip().lower() not in {"1", "true", "yes", "on"}
TT_DOMINANT_ANGULAR_RATIO = float(os.environ.get("AO_V5_TT_DOMINANT_ANGULAR_RATIO", os.environ.get("AO_V4_TT_DOMINANT_ANGULAR_RATIO", "2.0")))
TT_DOMINANT_XYZ_BIAS_WEIGHT = float(os.environ.get("AO_V5_TT_DOMINANT_XYZ_BIAS_WEIGHT", os.environ.get("AO_V4_1_TT_DOMINANT_XYZ_BIAS_WEIGHT", "0.03")))
TT_LOCAL_VARIANCE_WEIGHT = float(os.environ.get("AO_V5_TT_LOCAL_VARIANCE_WEIGHT", os.environ.get("AO_V4_1_TT_LOCAL_VARIANCE_WEIGHT", "0.02")))
TT_GLOBAL_PROJECTION_REG = float(os.environ.get("AO_V7_1_TT_GLOBAL_PROJECTION_REG", "1.0e-3"))
TT_GLOBAL_PROJECTED_RESIDUAL_WEIGHT = float(os.environ.get("AO_V7_1_TT_GLOBAL_PROJECTED_RESIDUAL_WEIGHT", "1.0"))
TT_GLOBAL_FULL_RESIDUAL_WEIGHT = float(os.environ.get("AO_V7_1_TT_GLOBAL_FULL_RESIDUAL_WEIGHT", "0.25"))
TT_LOCAL_RESIDUAL_LIMIT_ARCMIN = float(os.environ.get("AO_V7_2_TT_LOCAL_RESIDUAL_LIMIT_ARCMIN", "2.0"))
TT_LOCAL_L2_WEIGHT = float(os.environ.get("AO_V7_2_TT_LOCAL_L2_WEIGHT", "0.01"))
DEFAULT_SHWFS_NOISE_PROFILE = os.environ.get("AO_V5_SHWFS_NOISE_PROFILE", "realistic").strip().lower() or "realistic"
DEFAULT_SHWFS_LENSLET_COUNT = int(os.environ.get("AO_V5_SHWFS_LENSLET_COUNT", os.environ.get("AO_V3_SHWFS_LENSLET_COUNT", "13")))
DEFAULT_SHWFS_SLOPE_LIMIT = int(os.environ.get("AO_V5_SHWFS_SLOPE_LIMIT", os.environ.get("AO_V3_SHWFS_SLOPE_LIMIT", "256")))
DEFAULT_SHWFS_FORWARD_AVERAGES = int(os.environ.get("AO_V5_SHWFS_FORWARD_AVERAGES", "1"))
SEARCH_BOX_XY_OVERRIDE_MM = float(os.environ.get("AO_V5_SEARCH_BOX_XY_MM", "0.0") or "0.0")
SEARCH_BOX_Z_OVERRIDE_MM = float(os.environ.get("AO_V5_SEARCH_BOX_Z_MM", "0.0") or "0.0")

warnings.filterwarnings("ignore", category=MechanicalLimitWarning)


LogCallback = Callable[[str], None]


@dataclass
class EnvironmentStatus:
    ok: bool
    python_executable: str
    torch_version: str
    torch_cuda_available: bool
    torch_error: str


@dataclass
class RunConfig:
    overall_variation_mm: float = 1.0
    per_anchor_overrides: dict[int, dict[str, float]] | None = None
    output_name: str = "ui_case"
    optimizer_maxiter_limit: int | None = None
    max_eval_limit: int | None = None
    shwfs_noise_profile: str = DEFAULT_SHWFS_NOISE_PROFILE
    shwfs_forward_averages: int = DEFAULT_SHWFS_FORWARD_AVERAGES


@dataclass
class RelativeRepairPlant:
    """Hidden plant for repair-time evaluation.

    Important boundary:
    - The plant owns the absolute moved/start state and applies clamp/limit logic.
    - The repair/controller side must interact only through zero-centered relative
      commands and SHWFS-visible outputs.
    - Absolute anchor positions must not be exposed back into the repair logic.
    """

    optical_model: RayOpticsPhysicsEngine
    actuator_ids: list[int]
    absolute_start_state: dict[int, SurfacePerturbation]

    def reset_to_start(self) -> None:
        self.optical_model.set_surface_perturbations(self.absolute_start_state)

    def evaluate_relative_delta(
        self,
        relative_delta_vector: np.ndarray,
        *,
        shwfs: Any,
    ) -> dict[str, Any]:
        requested_absolute_state = refined_nominal._delta_state(
            self.actuator_ids,
            self.absolute_start_state,
            relative_delta_vector,
        )
        self.optical_model.set_surface_perturbations(requested_absolute_state)
        applied_absolute_state = bench._anchor_state_subset(
            self.optical_model.get_surface_perturbations(),
            self.actuator_ids,
        )
        applied_relative_state = _relative_state_from_reference(
            applied_absolute_state,
            self.absolute_start_state,
            self.actuator_ids,
        )
        requested_relative_state = _relative_state_from_delta_vector(self.actuator_ids, relative_delta_vector)
        violation = _relative_constraint_violation(requested_relative_state, applied_relative_state, self.actuator_ids)
        current_coeffs = shwfs.estimate_coeffs(self.optical_model)
        return {
            "requested_relative_state": requested_relative_state,
            "requested_absolute_state": requested_absolute_state,
            "applied_absolute_state": applied_absolute_state,
            "applied_relative_state": applied_relative_state,
            "violation": violation,
            "current_coeffs": current_coeffs,
        }

    def estimate_blind_nominal_seed(self, *, shwfs: Any) -> np.ndarray:
        self.reset_to_start()
        return np.asarray(shwfs.estimate_nominal_correction_seed(self.optical_model), dtype=np.float64)


def _default_log(message: str) -> None:
    print(message)


def initialize_environment() -> EnvironmentStatus:
    import sys

    torch_version = ""
    torch_cuda_available = False
    torch_error = ""
    ok = True
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - env dependent
        ok = False
        torch_error = f"{type(exc).__name__}: {exc}"
    return EnvironmentStatus(
        ok=ok,
        python_executable=str(sys.executable),
        torch_version=torch_version,
        torch_cuda_available=torch_cuda_available,
        torch_error=torch_error,
    )


def _timestamp_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _make_run_dir(output_name: str) -> Path:
    safe_name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in output_name).strip("_")
    if not safe_name:
        safe_name = "ui_case"
    run_dir = RUNS_DIR / f"{_timestamp_slug()}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _base_geometry_payload() -> dict[str, Any]:
    with BASE_GEOMETRY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _base_anchor_state_scaled(overall_variation_mm: float) -> dict[int, SurfacePerturbation]:
    payload = _base_geometry_payload()
    final_positions = payload.get("final_anchor_positions", {})
    effective_level_mm = float(payload.get("effective_level_mm", 1.0))
    if abs(effective_level_mm) < 1.0e-12:
        scale = 0.0
    else:
        scale = float(overall_variation_mm) / effective_level_mm
    state: dict[int, SurfacePerturbation] = {}
    for anchor_surface_id in ACTUATOR_IDS:
        row = final_positions.get(str(anchor_surface_id), {})
        state[anchor_surface_id] = SurfacePerturbation(
            dx_mm=float(row.get("dx_mm", 0.0)) * scale,
            dy_mm=float(row.get("dy_mm", 0.0)) * scale,
            dz_mm=float(row.get("dz_mm", 0.0)) * scale,
        )
    return state


def _apply_overrides(
    base_state: dict[int, SurfacePerturbation],
    overrides: dict[int, dict[str, float]] | None,
) -> dict[int, SurfacePerturbation]:
    state = copy.deepcopy(base_state)
    if not overrides:
        return state
    for anchor_surface_id, values in overrides.items():
        current = state.get(int(anchor_surface_id), SurfacePerturbation())
        state[int(anchor_surface_id)] = SurfacePerturbation(
            dx_mm=float(values.get("dx_mm", current.dx_mm)),
            dy_mm=float(values.get("dy_mm", current.dy_mm)),
            dz_mm=float(values.get("dz_mm", current.dz_mm)),
            tilt_x_deg=float(values.get("tilt_x_deg", current.tilt_x_deg)),
            tilt_y_deg=float(values.get("tilt_y_deg", current.tilt_y_deg)),
        )
    return state


def _build_requested_state(config: RunConfig) -> dict[int, SurfacePerturbation]:
    scaled = _base_anchor_state_scaled(config.overall_variation_mm)
    return _apply_overrides(scaled, config.per_anchor_overrides)


def _repair_search_box_radius_mm(config: RunConfig) -> float:
    estimated_scale_mm = max(float(config.overall_variation_mm), 0.0)
    return max(float(SEARCH_BOX_MIN_MM), float(SEARCH_BOX_SCALE) * estimated_scale_mm)


def _repair_search_box_xy_radius_mm(config: RunConfig) -> float:
    base_radius = _repair_search_box_radius_mm(config)
    if float(SEARCH_BOX_XY_OVERRIDE_MM) > 0.0:
        return float(SEARCH_BOX_XY_OVERRIDE_MM)
    return float(base_radius)


def _repair_search_box_z_radius_mm(config: RunConfig) -> float:
    base_radius = _repair_search_box_radius_mm(config)
    if float(SEARCH_BOX_Z_OVERRIDE_MM) > 0.0:
        return float(SEARCH_BOX_Z_OVERRIDE_MM)
    return float(base_radius)


def _optimizer_maxiter_limit(config: RunConfig) -> int:
    if config.optimizer_maxiter_limit is None:
        return int(bench.OPT_MAXITER)
    return max(1, int(config.optimizer_maxiter_limit))


def _max_eval_limit(config: RunConfig) -> int | None:
    if config.max_eval_limit is None:
        return None
    return max(1, int(config.max_eval_limit))


def _configure_v5_shwfs_runtime(config: RunConfig) -> str:
    profile_name = str(config.shwfs_noise_profile or DEFAULT_SHWFS_NOISE_PROFILE).strip().lower()
    real_shwfs.configure_shwfs_runtime(
        lenslet_count=int(DEFAULT_SHWFS_LENSLET_COUNT),
        slope_limit=int(DEFAULT_SHWFS_SLOPE_LIMIT),
        noise_profile=profile_name,
        noise_seed=int(real_shwfs.SHWFS_NOISE_SEED),
        forward_averages=max(1, int(config.shwfs_forward_averages)),
    )
    return profile_name


def _v3_delta_bounds(
    optical_model: RayOpticsPhysicsEngine,
    actuator_ids: list[int],
    base_state: dict[int, SurfacePerturbation],
    *,
    search_box_radius_mm: float,
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for surface_id in actuator_ids:
        envelope = optical_model.get_group_mechanical_envelope(surface_id)
        base = base_state.get(surface_id, SurfacePerturbation())
        axis_specs = (
            (-envelope.lateral_limit_mm, envelope.lateral_limit_mm, float(base.dx_mm), search_box_radius_mm),
            (-envelope.lateral_limit_mm, envelope.lateral_limit_mm, float(base.dy_mm), search_box_radius_mm),
            (envelope.axial_min_mm, envelope.axial_max_mm, float(base.dz_mm), search_box_radius_mm),
        )
        if ENABLE_TIPTILT:
            axis_specs += (
                (-envelope.tilt_limit_deg, envelope.tilt_limit_deg, float(base.tilt_x_deg), TIPTILT_LOCAL_LIMIT_DEG),
                (-envelope.tilt_limit_deg, envelope.tilt_limit_deg, float(base.tilt_y_deg), TIPTILT_LOCAL_LIMIT_DEG),
            )
        for lower_abs, upper_abs, base_value, local_radius in axis_specs:
            lower = max(lower_abs - base_value, -float(local_radius))
            upper = min(upper_abs - base_value, float(local_radius))
            if upper < lower:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            bounds.append((float(lower), float(upper)))
    return bounds


def _zero_centered_search_bounds(
    actuator_ids: list[int],
    *,
    search_box_xy_radius_mm: float,
    search_box_z_radius_mm: float,
) -> list[tuple[float, float]]:
    """Return repair/controller-visible bounds only.

    These bounds are centered at the pre-repair position (relative zero). They are
    not recentered around the hidden absolute moved state, so the repair group does
    not gain direct access to absolute anchor positions.
    """

    bounds: list[tuple[float, float]] = []
    for _surface_id in actuator_ids:
        bounds.extend(
            [
                (-float(search_box_xy_radius_mm), float(search_box_xy_radius_mm)),
                (-float(search_box_xy_radius_mm), float(search_box_xy_radius_mm)),
                (-float(search_box_z_radius_mm), float(search_box_z_radius_mm)),
            ]
        )
        if ENABLE_TIPTILT:
            bounds.extend(
                [
                    (-float(TIPTILT_LOCAL_LIMIT_DEG), float(TIPTILT_LOCAL_LIMIT_DEG)),
                    (-float(TIPTILT_LOCAL_LIMIT_DEG), float(TIPTILT_LOCAL_LIMIT_DEG)),
                ]
            )
    return bounds


def _relative_state_from_reference(
    absolute_state: dict[int, SurfacePerturbation],
    reference_state: dict[int, SurfacePerturbation],
    actuator_ids: list[int],
) -> dict[int, SurfacePerturbation]:
    relative: dict[int, SurfacePerturbation] = {}
    for surface_id in actuator_ids:
        absolute = absolute_state.get(surface_id, SurfacePerturbation())
        reference = reference_state.get(surface_id, SurfacePerturbation())
        relative[surface_id] = SurfacePerturbation(
            dx_mm=float(absolute.dx_mm - reference.dx_mm),
            dy_mm=float(absolute.dy_mm - reference.dy_mm),
            dz_mm=float(absolute.dz_mm - reference.dz_mm),
            tilt_x_deg=float(absolute.tilt_x_deg - reference.tilt_x_deg),
            tilt_y_deg=float(absolute.tilt_y_deg - reference.tilt_y_deg),
        )
    return relative


def _relative_state_from_delta_vector(
    actuator_ids: list[int],
    delta_vector: np.ndarray,
) -> dict[int, SurfacePerturbation]:
    zeros = {surface_id: SurfacePerturbation() for surface_id in actuator_ids}
    return refined_nominal._delta_state(actuator_ids, zeros, delta_vector)


def _relative_constraint_violation(
    requested_relative_state: dict[int, SurfacePerturbation],
    applied_relative_state: dict[int, SurfacePerturbation],
    actuator_ids: list[int],
) -> dict[str, float]:
    requested = []
    applied = []
    for surface_id in actuator_ids:
        requested_perturb = requested_relative_state.get(surface_id, SurfacePerturbation())
        applied_perturb = applied_relative_state.get(surface_id, SurfacePerturbation())
        requested.extend([float(requested_perturb.dx_mm), float(requested_perturb.dy_mm), float(requested_perturb.dz_mm)])
        applied.extend([float(applied_perturb.dx_mm), float(applied_perturb.dy_mm), float(applied_perturb.dz_mm)])
        if ENABLE_TIPTILT:
            requested.extend([float(requested_perturb.tilt_x_deg), float(requested_perturb.tilt_y_deg)])
            applied.extend([float(applied_perturb.tilt_x_deg), float(applied_perturb.tilt_y_deg)])
    requested_arr = np.asarray(requested, dtype=np.float64)
    applied_arr = np.asarray(applied, dtype=np.float64)
    delta = applied_arr - requested_arr
    return {
        "l2_mm": float(np.linalg.norm(delta)),
        "max_abs_mm": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _downweight_blind_seed_dz(delta_vector: np.ndarray, actuator_ids: list[int]) -> np.ndarray:
    dof_count = 5 if ENABLE_TIPTILT else 3
    weighted = np.asarray(delta_vector, dtype=np.float64).reshape(len(actuator_ids), dof_count).copy()
    weighted[:, 2] *= float(BLIND_SEED_DZ_WEIGHT)
    return weighted.reshape(-1)


def _residual_block_signature(residual_vector: np.ndarray) -> dict[str, float]:
    residual = np.asarray(residual_vector, dtype=np.float64).reshape(-1)
    if residual.size <= 0:
        return {
            "radial": 0.0,
            "angular_u": 0.0,
            "angular_v": 0.0,
            "angular": 0.0,
            "higher": 0.0,
            "angular_over_radial": 0.0,
            "higher_over_angular": 0.0,
        }

    mode_pairs = nm_sequence(int(residual.size))
    pair_to_index = {pair: idx for idx, pair in enumerate(mode_pairs)}

    def abs_mode(pair: tuple[int, int]) -> float:
        idx = pair_to_index.get(pair)
        if idx is None:
            return 0.0
        return float(abs(residual[idx]))

    radial = abs_mode((0, 0)) + abs_mode((2, 0))
    if radial <= 0.0:
        radial = float(np.sum(np.abs(residual)))

    # TT-aware split:
    # - angular_u tracks the (-1) family, which is strongest for global tilt_y
    # - angular_v tracks the (+1) family, which is strongest for global tilt_x
    angular_u = abs_mode((1, -1)) + abs_mode((3, -1))
    angular_v = abs_mode((1, 1)) + abs_mode((3, 1))

    higher = 0.0
    for idx, pair in enumerate(mode_pairs):
        if pair in {(0, 0), (2, 0), (1, -1), (3, -1), (1, 1), (3, 1)}:
            continue
        higher += float(abs(residual[idx]))
    angular = angular_u + angular_v
    return {
        "radial": radial,
        "angular_u": angular_u,
        "angular_v": angular_v,
        "angular": angular,
        "higher": higher,
        "angular_over_radial": float(angular / max(radial, 1.0e-12)),
        "higher_over_angular": float(higher / max(angular, 1.0e-12)),
    }


def _active_block_indices(actuator_ids: list[int]) -> dict[str, np.ndarray]:
    dof_count = 5 if ENABLE_TIPTILT else 3
    xy_indices: list[int] = []
    z_indices: list[int] = []
    tt_indices: list[int] = []
    tt_x_indices: list[int] = []
    tt_y_indices: list[int] = []
    for anchor_index in range(len(actuator_ids)):
        base = anchor_index * dof_count
        xy_indices.extend([base + 0, base + 1])
        z_indices.append(base + 2)
        if dof_count >= 5:
            tt_x_indices.append(base + 3)
            tt_y_indices.append(base + 4)
            tt_indices.extend([base + 3, base + 4])
    return {
        "xy": np.asarray(xy_indices, dtype=np.int64),
        "z": np.asarray(z_indices, dtype=np.int64),
        "tt": np.asarray(tt_indices, dtype=np.int64),
        "tt_x": np.asarray(tt_x_indices, dtype=np.int64),
        "tt_y": np.asarray(tt_y_indices, dtype=np.int64),
    }


def _choose_primary_block_order(signature: dict[str, float]) -> list[str]:
    if float(signature["angular_over_radial"]) < 0.1:
        return ["z", "xy"]
    return ["xy", "z"]


def _tt_dominant_case(signature: dict[str, float]) -> bool:
    return bool(ENABLE_TIPTILT and float(signature["angular_over_radial"]) >= float(TT_DOMINANT_ANGULAR_RATIO))


def _clamped_state(requested_state: dict[int, SurfacePerturbation]) -> tuple[dict[int, SurfacePerturbation], dict[str, Any]]:
    model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
    model.set_surface_perturbations(requested_state)
    clamped = bench._anchor_state_subset(model.get_surface_perturbations(), ACTUATOR_IDS)
    mech = bench._mechanical_summary(model)
    return clamped, mech


def _summary_text_block(summary: dict[str, Any]) -> str:
    nominal = summary["nominal"]
    moved = summary["moved"]
    repaired = summary.get("repaired", moved)
    lines = [
        f"Nominal  | Strehl {nominal['strehl_ratio']:.6f} | WRMS {nominal['wavefront_rms_waves']:.6f} | Residual {nominal['residual_norm']:.6f}",
        f"Before   | Strehl {moved['strehl_ratio']:.6f} | WRMS {moved['wavefront_rms_waves']:.6f} | Residual {moved['residual_norm']:.6f}",
        f"Repaired | Strehl {repaired['strehl_ratio']:.6f} | WRMS {repaired['wavefront_rms_waves']:.6f} | Residual {repaired['residual_norm']:.6f}",
    ]
    return "\n".join(lines)


def _set_benchmark_output_prefix(output_prefix: Path) -> dict[str, Any]:
    old_values = {
        "OUTPUT_PREFIX": bench.OUTPUT_PREFIX,
        "OUTPUT_JSON_PATH": bench.OUTPUT_JSON_PATH,
        "OUTPUT_CSV_PATH": bench.OUTPUT_CSV_PATH,
        "OUTPUT_FOV_CSV_PATH": bench.OUTPUT_FOV_CSV_PATH,
        "OUTPUT_NPZ_PATH": bench.OUTPUT_NPZ_PATH,
        "OUTPUT_LIGHTPATH_PNG_PATH": bench.OUTPUT_LIGHTPATH_PNG_PATH,
        "OUTPUT_METRICS_PNG_PATH": bench.OUTPUT_METRICS_PNG_PATH,
        "OUTPUT_FOV_PNG_PATH": bench.OUTPUT_FOV_PNG_PATH,
        "OUTPUT_LOG_PATH": bench.OUTPUT_LOG_PATH,
    }
    bench.OUTPUT_PREFIX = output_prefix
    bench.OUTPUT_JSON_PATH = output_prefix.with_name(output_prefix.name + "_compare.json")
    bench.OUTPUT_CSV_PATH = output_prefix.with_name(output_prefix.name + "_compare.csv")
    bench.OUTPUT_FOV_CSV_PATH = output_prefix.with_name(output_prefix.name + "_fov_rms.csv")
    bench.OUTPUT_NPZ_PATH = output_prefix.with_name(output_prefix.name + "_psfs.npz")
    bench.OUTPUT_LIGHTPATH_PNG_PATH = output_prefix.with_name(output_prefix.name + "_lightpath_psf_wavefront.png")
    bench.OUTPUT_METRICS_PNG_PATH = output_prefix.with_name(output_prefix.name + "_metrics.png")
    bench.OUTPUT_FOV_PNG_PATH = output_prefix.with_name(output_prefix.name + "_fov_rms.png")
    bench.OUTPUT_LOG_PATH = output_prefix.with_name(output_prefix.name + "_run.log")
    return old_values


def _restore_benchmark_output_prefix(old_values: dict[str, Any]) -> None:
    for key, value in old_values.items():
        setattr(bench, key, value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_before_only(
    run_dir: Path,
    nominal_model: RayOpticsPhysicsEngine,
    moved_model: RayOpticsPhysicsEngine,
    nominal_metrics: dict[str, float],
    moved_metrics: dict[str, float],
    nominal_psf: np.ndarray,
    moved_psf: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    shared_limits = bench._shared_lightpath_limits([nominal_model, moved_model])
    shared_vmax = bench._shared_wavefront_vmax([nominal_model, moved_model])
    panels = [
        ("Nominal", nominal_model, nominal_metrics, nominal_psf),
        ("Before Calibration", moved_model, moved_metrics, moved_psf),
    ]
    for col, (name, model, metrics, psf) in enumerate(panels):
        bench._draw_optical_panel(axes[0, col], model, state_name=name, metrics=metrics, limits=shared_limits)
        bench._draw_wavefront(axes[1, col], model, title=f"{name} Wavefront (waves)", vmax=shared_vmax)
        bench._draw_psf(axes[2, col], psf, title=f"{name} PSF (normalized)")
    fig.suptitle("Before Calibration Summary", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path = run_dir / "before_calibration_wavefront_psf_lightpath.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _evaluate_nominal_and_moved(
    moved_state: dict[int, SurfacePerturbation],
    run_dir: Path,
    log: LogCallback,
) -> tuple[dict[str, Any], dict[str, Any]]:
    log("Building nominal and before-calibration models...")
    nominal_model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
    shwfs = real_shwfs._build_shwfs_measurement_model(
        nominal_model,
        focus_shift_mm=bench.REFERENCE_FOCUS_MM,
        actuator_ids=ACTUATOR_IDS,
        mechanical_dof_names=("dx_mm", "dy_mm", "dz_mm", "tilt_x_deg", "tilt_y_deg")
        if ENABLE_TIPTILT
        else ("dx_mm", "dy_mm", "dz_mm"),
        tilt_step_deg=float(TIPTILT_RESPONSE_STEP_DEG if ENABLE_TIPTILT else 0.0),
    )
    ideal_psf = refined_nominal._ideal_diffraction_limited_psf(nominal_model, focus=bench.REFERENCE_FOCUS_MM)
    nominal_psf = nominal_model.get_psf_image(focus=bench.REFERENCE_FOCUS_MM)
    true_reference_coeffs = refined_nominal._project_w_bessel_coeffs_from_opd(
        nominal_model,
        nominal_model.get_wavefront_opd(focus=bench.REFERENCE_FOCUS_MM),
        beam_fill_ratio=bench.BEAM_FILL_RATIO,
        num_modes=bench.NUM_MODES,
    )
    nominal_metrics = bench._state_metrics(
        nominal_model,
        shwfs,
        nominal_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=true_reference_coeffs,
    )

    moved_model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
    moved_model.set_surface_perturbations(moved_state)
    moved_state_clamped = bench._anchor_state_subset(moved_model.get_surface_perturbations(), ACTUATOR_IDS)
    moved_psf = moved_model.get_psf_image(focus=bench.REFERENCE_FOCUS_MM)
    moved_metrics = bench._state_metrics(
        moved_model,
        shwfs,
        moved_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=true_reference_coeffs,
    )
    moved_mech = bench._mechanical_summary(moved_model)

    before_summary = {
        "status": "before_complete",
        "configuration": {
            "wavelength_nm": float(bench.WAVELENGTH_NM),
            "object_space_mm": float(bench.OBJECT_SPACE_MM),
            "reference_focus_mm": float(bench.REFERENCE_FOCUS_MM),
            "fast_pupil_samples": int(bench.FAST_PUPIL_SAMPLES),
            "fast_fft_samples": int(bench.FAST_FFT_SAMPLES),
        },
        "nominal": nominal_metrics,
        "moved": moved_metrics,
        "moved_mechanical_summary": moved_mech,
        "moved_position_by_anchor": {
            str(surface_id): {
                "dx_mm": float(moved_state_clamped.get(surface_id, SurfacePerturbation()).dx_mm),
                "dy_mm": float(moved_state_clamped.get(surface_id, SurfacePerturbation()).dy_mm),
                "dz_mm": float(moved_state_clamped.get(surface_id, SurfacePerturbation()).dz_mm),
            }
            for surface_id in ACTUATOR_IDS
        },
    }
    figure_path = _render_before_only(
        run_dir,
        nominal_model,
        moved_model,
        nominal_metrics,
        moved_metrics,
        nominal_psf,
        moved_psf,
    )
    _write_json(run_dir / "before_calibration_summary.json", before_summary)
    before_rows = [
        {
            "state": "nominal",
            "strehl_ratio": nominal_metrics["strehl_ratio"],
            "wavefront_rms_waves": nominal_metrics["wavefront_rms_waves"],
            "residual_norm": nominal_metrics["residual_norm"],
            "rmse_to_ideal_perfect": nominal_metrics["rmse_to_ideal_perfect"],
        },
        {
            "state": "before_calibration",
            "strehl_ratio": moved_metrics["strehl_ratio"],
            "wavefront_rms_waves": moved_metrics["wavefront_rms_waves"],
            "residual_norm": moved_metrics["residual_norm"],
            "rmse_to_ideal_perfect": moved_metrics["rmse_to_ideal_perfect"],
        },
    ]
    _write_csv(run_dir / "before_calibration_metrics.csv", before_rows)
    log("Before-calibration outputs generated.")
    runtime_context = {
        "nominal_model": nominal_model,
        "moved_model": moved_model,
        "moved_state": moved_state_clamped,
        "nominal_psf": nominal_psf,
        "ideal_psf": ideal_psf,
        "shwfs": shwfs,
        "true_reference_coeffs": true_reference_coeffs,
        "nominal_metrics": nominal_metrics,
        "moved_metrics": moved_metrics,
        "figure_path": figure_path,
    }
    return before_summary, runtime_context


def generate_before_calibration(config: RunConfig, log: LogCallback | None = None) -> dict[str, Any]:
    logger = log or _default_log
    run_dir = _make_run_dir(config.output_name)
    logger(f"Created run directory: {run_dir}")
    noise_profile_name = _configure_v5_shwfs_runtime(config)
    logger(
        "Configured V5 SHWFS runtime: "
        f"lenslet_grid={DEFAULT_SHWFS_LENSLET_COUNT}x{DEFAULT_SHWFS_LENSLET_COUNT} "
        f"slope_limit={DEFAULT_SHWFS_SLOPE_LIMIT} "
        f"noise_profile={noise_profile_name} "
        f"forward_averages={max(1, int(config.shwfs_forward_averages))}"
    )
    requested_state = _build_requested_state(config)
    clamped_state, clamp_summary = _clamped_state(requested_state)
    before_summary, _ = _evaluate_nominal_and_moved(clamped_state, run_dir, logger)
    before_summary["requested_position_by_anchor"] = {
        str(surface_id): {
            "dx_mm": float(requested_state.get(surface_id, SurfacePerturbation()).dx_mm),
            "dy_mm": float(requested_state.get(surface_id, SurfacePerturbation()).dy_mm),
            "dz_mm": float(requested_state.get(surface_id, SurfacePerturbation()).dz_mm),
            "tilt_x_deg": float(requested_state.get(surface_id, SurfacePerturbation()).tilt_x_deg),
            "tilt_y_deg": float(requested_state.get(surface_id, SurfacePerturbation()).tilt_y_deg),
        }
        for surface_id in ACTUATOR_IDS
    }
    before_summary["clamp_summary"] = clamp_summary
    _write_json(run_dir / "before_calibration_summary.json", before_summary)
    return {
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "before_calibration_summary.json"),
        "figure_path": str(run_dir / "before_calibration_wavefront_psf_lightpath.png"),
        "summary_text": _summary_text_block({"nominal": before_summary["nominal"], "moved": before_summary["moved"]}),
    }


def run_repair(config: RunConfig, log: LogCallback | None = None) -> dict[str, Any]:
    logger = log or _default_log
    run_dir = _make_run_dir(config.output_name)
    logger(f"Created run directory: {run_dir}")
    noise_profile_name = _configure_v5_shwfs_runtime(config)
    logger(
        "Configured V5 SHWFS runtime: "
        f"lenslet_grid={DEFAULT_SHWFS_LENSLET_COUNT}x{DEFAULT_SHWFS_LENSLET_COUNT} "
        f"slope_limit={DEFAULT_SHWFS_SLOPE_LIMIT} "
        f"noise_profile={noise_profile_name} "
        f"forward_averages={max(1, int(config.shwfs_forward_averages))}"
    )
    requested_state = _build_requested_state(config)
    moved_state, clamp_summary = _clamped_state(requested_state)
    before_summary, context = _evaluate_nominal_and_moved(moved_state, run_dir, logger)
    nominal_model = context["nominal_model"]
    moved_model = context["moved_model"]
    nominal_psf = context["nominal_psf"]
    ideal_psf = context["ideal_psf"]
    shwfs = context["shwfs"]
    true_reference_coeffs = context["true_reference_coeffs"]
    nominal_metrics = context["nominal_metrics"]
    moved_metrics = context["moved_metrics"]

    # Hidden plant state boundary:
    # - This absolute moved/start state may be used by the clamp/plant layer.
    # - The repair/controller layer must not receive or recenter around it.
    hidden_start_state = bench._anchor_state_subset(moved_model.get_surface_perturbations(), ACTUATOR_IDS)
    plant = RelativeRepairPlant(
        optical_model=moved_model,
        actuator_ids=ACTUATOR_IDS,
        absolute_start_state=hidden_start_state,
    )
    hold_seed = np.zeros((5 if ENABLE_TIPTILT else 3) * len(ACTUATOR_IDS), dtype=np.float64)
    search_box_radius_mm = _repair_search_box_radius_mm(config)
    search_box_xy_radius_mm = _repair_search_box_xy_radius_mm(config)
    search_box_z_radius_mm = _repair_search_box_z_radius_mm(config)
    optimizer_maxiter_limit = _optimizer_maxiter_limit(config)
    max_eval_limit = _max_eval_limit(config)
    optimizer_maxfun_limit = max(int(bench.OPT_MAXFUN), 50 * int(optimizer_maxiter_limit))
    # Repair-visible search box is centered at relative zero (the pre-repair position),
    # not at the hidden absolute moved state.
    bounds = _zero_centered_search_bounds(
        ACTUATOR_IDS,
        search_box_xy_radius_mm=search_box_xy_radius_mm,
        search_box_z_radius_mm=search_box_z_radius_mm,
    )
    blind_nominal_seed = hold_seed.copy()
    if ENABLE_BLIND_SEED:
        blind_nominal_seed = bench._project_delta_to_bounds(
            _downweight_blind_seed_dz(
                plant.estimate_blind_nominal_seed(shwfs=shwfs),
                ACTUATOR_IDS,
            ),
            bounds,
        )
    timing_totals = {"set_state_s": 0.0, "psf_s": 0.0, "shwfs_s": 0.0, "constraint_s": 0.0}
    tracker = bench.Tracker(best_delta_mm=hold_seed.copy(), best_any_delta_mm=hold_seed.copy())
    optimizer_state = {
        "iteration_count": 0,
        "success": None,
        "status": None,
        "message": "",
        "nfev": 0,
    }
    new_best_events: list[dict[str, float | int]] = []
    stage_records: list[dict[str, Any]] = []

    output_prefix = run_dir / "repair_result"
    old_outputs = _set_benchmark_output_prefix(output_prefix)
    bench.OUTPUT_LOG_PATH.write_text("", encoding="utf-8")

    def local_log(message: str) -> None:
        logger(message)
        bench._log(message)

    def _evaluate_candidate(delta_vector: np.ndarray, *, count_eval: bool) -> dict[str, Any]:
        delta_vector = np.asarray(delta_vector, dtype=np.float64).reshape(-1)

        t0 = time.perf_counter()
        plant_result = plant.evaluate_relative_delta(delta_vector, shwfs=shwfs)
        plant_elapsed = time.perf_counter() - t0
        timing_totals["set_state_s"] += plant_elapsed
        timing_totals["shwfs_s"] += plant_elapsed

        rmse = float("nan")
        sharpness = float("nan")
        current_coeffs = np.asarray(plant_result["current_coeffs"], dtype=np.float64)
        residual = current_coeffs - shwfs.reference_coeffs
        residual_norm = float(np.linalg.norm(residual))
        violation = plant_result["violation"]
        timing_totals["constraint_s"] += 0.0

        cost = residual_norm + bench.CONSTRAINT_PENALTY_WEIGHT * float(violation["l2_mm"])
        result = {
            "cost": float(cost),
            "requested_relative_state": plant_result["requested_relative_state"],
            "applied_relative_state": plant_result["applied_relative_state"],
            "applied_absolute_state": plant_result["applied_absolute_state"],
            "violation": violation,
            "current_coeffs": current_coeffs,
            "residual": residual,
            "residual_norm": residual_norm,
            "rmse": rmse,
            "sharpness": sharpness,
            "delta_vector": delta_vector.copy(),
        }
        if count_eval:
            tracker.evaluation_count += 1
            if float(violation["max_abs_mm"]) <= bench.CONSTRAINT_TOL_MM and residual_norm < tracker.best_cost:
                tracker.best_cost = residual_norm
                tracker.best_rmse = rmse
                tracker.best_sharpness = sharpness
                tracker.best_delta_mm = delta_vector.copy()
                tracker.feasible_count += 1
                tracker.last_best_eval = tracker.evaluation_count
                new_best_events.append(
                    {
                        "eval": int(tracker.evaluation_count),
                        "optimizer_iteration": int(optimizer_state["iteration_count"]),
                        "residual_norm": float(residual_norm),
                        "clamp_l2_mm": float(violation["l2_mm"]),
                    }
                )
            if cost < tracker.best_any_cost:
                tracker.best_any_cost = cost
                tracker.best_any_rmse = rmse
                tracker.best_any_sharpness = sharpness
                tracker.best_any_delta_mm = delta_vector.copy()
            result["evaluation_count"] = int(tracker.evaluation_count)
        return result

    def objective(delta_vector: np.ndarray) -> float:
        eval_started = time.perf_counter()
        result = _evaluate_candidate(delta_vector, count_eval=True)
        residual_norm = float(result["residual_norm"])
        violation = result["violation"]
        if tracker.evaluation_count == 1 or tracker.evaluation_count % 5 == 0:
            local_log(
                "eval="
                f"{tracker.evaluation_count} residual={residual_norm:.6f} "
                f"norm_resid={residual_norm / max(moved_metrics['residual_norm'], 1.0e-12):.6f} "
                f"best={tracker.best_cost:.6f} "
                f"best_norm={tracker.best_cost / max(moved_metrics['residual_norm'], 1.0e-12):.6f} "
                f"clamp_l2={violation['l2_mm']:.3e} "
                f"elapsed={time.perf_counter() - eval_started:.2f}s"
            )
        if max_eval_limit is not None and tracker.evaluation_count >= max_eval_limit:
            raise bench.EarlyStopOptimization(f"Reached max evaluation limit of {max_eval_limit}.")
        if tracker.feasible_count > 0 and tracker.evaluation_count - tracker.last_best_eval >= bench.BEST_PATIENCE_EVALS:
            raise bench.EarlyStopOptimization(
                f"No best-cost improvement for {bench.BEST_PATIENCE_EVALS} evaluations."
            )
        return float(result["cost"])

    def optimizer_callback(_xk: np.ndarray) -> None:
        optimizer_state["iteration_count"] = int(optimizer_state["iteration_count"]) + 1

    def inspect_signature(delta_vector: np.ndarray) -> dict[str, Any]:
        result = _evaluate_candidate(delta_vector, count_eval=False)
        signature = _residual_block_signature(np.asarray(result["residual"], dtype=np.float64))
        signature["residual_norm"] = float(result["residual_norm"])
        signature["clamp_l2_mm"] = float(result["violation"]["l2_mm"])
        return signature

    def optimize_block(
        stage_name: str,
        start_vector: np.ndarray,
        active_indices: np.ndarray,
        *,
        stage_tt_limit_deg: float | None = None,
        xyz_bias_weight: float = 0.0,
        tt_variance_weight: float = 0.0,
    ) -> np.ndarray:
        if active_indices.size == 0:
            return np.asarray(start_vector, dtype=np.float64).copy()
        x0_full = np.asarray(start_vector, dtype=np.float64).reshape(-1).copy()
        x0_reduced = x0_full[active_indices].copy()
        reduced_bounds: list[tuple[float, float]] = []
        for local_index, global_index in enumerate(active_indices.tolist()):
            lower, upper = bounds[int(global_index)]
            if stage_tt_limit_deg is not None:
                current_value = float(x0_full[int(global_index)])
                lower = max(lower, -float(stage_tt_limit_deg) + current_value)
                upper = min(upper, float(stage_tt_limit_deg) + current_value)
            if upper < lower:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            x0_reduced[local_index] = float(np.clip(x0_reduced[local_index], lower, upper))
            reduced_bounds.append((float(lower), float(upper)))

        stage_started_at = time.perf_counter()
        stage_eval_before = int(tracker.evaluation_count)
        stage_iter_before = int(optimizer_state["iteration_count"])
        stage_best_before = float(tracker.best_cost)
        stage_signature_before = inspect_signature(x0_full)
        stage_maxiter_limit = max(1, min(int(optimizer_maxiter_limit), int(BLOCK_STAGE_MAXITER)))
        local_log(
            f"Stage {stage_name}: start residual={stage_signature_before['residual_norm']:.6f} "
            f"angular_over_radial={stage_signature_before['angular_over_radial']:.3f} "
            f"higher_over_angular={stage_signature_before['higher_over_angular']:.3f} "
            f"active_dims={int(active_indices.size)} "
            f"stage_maxiter={stage_maxiter_limit} "
            f"xyz_bias_weight={float(xyz_bias_weight):.4f} "
            f"tt_variance_weight={float(tt_variance_weight):.4f}"
        )

        def reduced_objective(active_delta: np.ndarray) -> float:
            full_vector = x0_full.copy()
            full_vector[active_indices] = np.asarray(active_delta, dtype=np.float64).reshape(-1)
            cost = float(objective(full_vector))
            if xyz_bias_weight > 0.0:
                normalized = np.asarray(active_delta, dtype=np.float64).reshape(-1) / max(float(search_box_radius_mm), 1.0e-12)
                cost += float(xyz_bias_weight) * float(np.mean(normalized * normalized))
            if tt_variance_weight > 0.0 and block_indices["tt_x"].size > 0 and block_indices["tt_y"].size > 0:
                tt_scale = max(
                    float(stage_tt_limit_deg) if stage_tt_limit_deg is not None else float(TIPTILT_TEST_LIMIT_ARCMIN / 60.0),
                    1.0e-12,
                )
                tx = full_vector[block_indices["tt_x"]]
                ty = full_vector[block_indices["tt_y"]]
                variance_penalty = float(np.var(tx) + np.var(ty)) / float(tt_scale * tt_scale)
                cost += float(tt_variance_weight) * variance_penalty
            return cost

        stage_status = 0
        stage_success = False
        stage_message = ""
        stage_nfev = 0
        x_final = x0_full.copy()
        stage_exception: bench.EarlyStopOptimization | None = None
        stage_continues_after_patience = False
        try:
            stage_result = minimize(
                reduced_objective,
                x0=x0_reduced,
                method="L-BFGS-B",
                bounds=reduced_bounds,
                callback=optimizer_callback,
                options={
                    "maxiter": stage_maxiter_limit,
                    "maxfun": max(int(bench.OPT_MAXFUN), 12 * int(active_indices.size)),
                    "ftol": 1.0e-6,
                    "eps": bench.OPT_DIFF_STEP_MM,
                },
            )
            x_final[active_indices] = np.asarray(stage_result.x, dtype=np.float64).reshape(-1)
            stage_status = int(stage_result.status)
            stage_success = bool(stage_result.success)
            stage_message = str(stage_result.message)
            stage_nfev = int(stage_result.nfev)
        except bench.EarlyStopOptimization as exc:
            stage_exception = exc
            if "Reached max evaluation limit" in str(exc):
                raise
            best_stage_vector = tracker.best_delta_mm if np.isfinite(tracker.best_cost) else tracker.best_any_delta_mm
            x_final = np.asarray(best_stage_vector, dtype=np.float64).reshape(-1).copy()
            stage_status = -1
            stage_success = False
            stage_message = f"EarlyStopOptimization: {exc}"
            stage_nfev = int(tracker.evaluation_count - stage_eval_before)
            stage_continues_after_patience = True
        stage_signature_after = inspect_signature(x_final)
        stage_records.append(
            {
                "stage_name": stage_name,
                "active_block": stage_name.split("_", 1)[-1],
                "active_dims": int(active_indices.size),
                "xyz_bias_weight": float(xyz_bias_weight),
                "tt_variance_weight": float(tt_variance_weight),
                "evals_used": int(tracker.evaluation_count - stage_eval_before),
                "optimizer_iterations_used": int(optimizer_state["iteration_count"] - stage_iter_before),
                "stage_runtime_s": float(time.perf_counter() - stage_started_at),
                "residual_before": float(stage_signature_before["residual_norm"]),
                "residual_after": float(stage_signature_after["residual_norm"]),
                "best_residual_before": float(stage_best_before),
                "best_residual_after": float(tracker.best_cost),
                "signature_before": stage_signature_before,
                "signature_after": stage_signature_after,
                "optimizer_status": int(stage_status),
                "optimizer_success": bool(stage_success),
                "optimizer_message": str(stage_message),
                "function_evaluations": int(stage_nfev),
                "continued_after_patience_stop": bool(stage_continues_after_patience),
            }
        )
        local_log(
            f"Stage {stage_name}: finish residual={stage_signature_after['residual_norm']:.6f} "
            f"best={tracker.best_cost:.6f} evals_used={tracker.evaluation_count - stage_eval_before}"
        )
        optimizer_state["success"] = bool(stage_success)
        optimizer_state["status"] = int(stage_status)
        optimizer_state["message"] = str(stage_message)
        optimizer_state["nfev"] = int(optimizer_state["nfev"]) + int(stage_nfev)
        # Reset block-level patience so the next block starts from the current
        # best state rather than inheriting a stale "no improvement" counter.
        tracker.last_best_eval = int(tracker.evaluation_count)
        if stage_continues_after_patience:
            local_log(f"Stage {stage_name}: continuing to next block from current best state")
        return x_final

    def optimize_global_tt_block(
        stage_name: str,
        start_vector: np.ndarray,
        tt_x_indices: np.ndarray,
        tt_y_indices: np.ndarray,
        *,
        stage_tt_limit_deg: float,
    ) -> np.ndarray:
        if tt_x_indices.size == 0 or tt_y_indices.size == 0:
            return np.asarray(start_vector, dtype=np.float64).copy()
        x0_full = np.asarray(start_vector, dtype=np.float64).reshape(-1).copy()
        x0_reduced = np.asarray(
            [
                float(np.mean(x0_full[tt_x_indices])),
                float(np.mean(x0_full[tt_y_indices])),
            ],
            dtype=np.float64,
        )
        lower = -float(stage_tt_limit_deg)
        upper = float(stage_tt_limit_deg)
        reduced_bounds = [(lower, upper), (lower, upper)]
        x0_reduced = np.clip(x0_reduced, lower, upper)

        stage_started_at = time.perf_counter()
        stage_eval_before = int(tracker.evaluation_count)
        stage_iter_before = int(optimizer_state["iteration_count"])
        stage_best_before = float(tracker.best_cost)
        stage_signature_before = inspect_signature(x0_full)
        stage_maxiter_limit = max(1, min(int(optimizer_maxiter_limit), int(BLOCK_STAGE_MAXITER)))
        local_log(
            f"Stage {stage_name}: start residual={stage_signature_before['residual_norm']:.6f} "
            f"angular_over_radial={stage_signature_before['angular_over_radial']:.3f} "
            f"higher_over_angular={stage_signature_before['higher_over_angular']:.3f} "
            f"active_dims=2 stage_maxiter={stage_maxiter_limit}"
        )

        def _compose_global_tt_vector(base_vector: np.ndarray, tx_deg: float, ty_deg: float) -> np.ndarray:
            full_vector = np.asarray(base_vector, dtype=np.float64).reshape(-1).copy()
            full_vector[tt_x_indices] = float(tx_deg)
            full_vector[tt_y_indices] = float(ty_deg)
            return full_vector

        def _residual_and_violation(full_vector: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
            result = _evaluate_candidate(full_vector, count_eval=False)
            return np.asarray(result["residual"], dtype=np.float64), result["violation"]

        def _global_tt_sensitivity_matrix(base_vector: np.ndarray, center_reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            base_residual, _ = _residual_and_violation(base_vector)
            sensitivity_columns: list[np.ndarray] = []
            for axis_index in range(2):
                current_value = float(center_reduced[axis_index])
                axis_lower, axis_upper = reduced_bounds[axis_index]
                plus_value = min(current_value + float(stage_tt_limit_deg) * 0.25, axis_upper)
                minus_value = max(current_value - float(stage_tt_limit_deg) * 0.25, axis_lower)
                if plus_value > current_value and minus_value < current_value:
                    plus_vector = _compose_global_tt_vector(
                        x0_full,
                        plus_value if axis_index == 0 else center_reduced[0],
                        plus_value if axis_index == 1 else center_reduced[1],
                    )
                    minus_vector = _compose_global_tt_vector(
                        x0_full,
                        minus_value if axis_index == 0 else center_reduced[0],
                        minus_value if axis_index == 1 else center_reduced[1],
                    )
                    plus_residual, _ = _residual_and_violation(plus_vector)
                    minus_residual, _ = _residual_and_violation(minus_vector)
                    derivative = (plus_residual - minus_residual) / max(plus_value - minus_value, 1.0e-12)
                elif plus_value > current_value:
                    plus_vector = _compose_global_tt_vector(
                        x0_full,
                        plus_value if axis_index == 0 else center_reduced[0],
                        plus_value if axis_index == 1 else center_reduced[1],
                    )
                    plus_residual, _ = _residual_and_violation(plus_vector)
                    derivative = (plus_residual - base_residual) / max(plus_value - current_value, 1.0e-12)
                elif minus_value < current_value:
                    minus_vector = _compose_global_tt_vector(
                        x0_full,
                        minus_value if axis_index == 0 else center_reduced[0],
                        minus_value if axis_index == 1 else center_reduced[1],
                    )
                    minus_residual, _ = _residual_and_violation(minus_vector)
                    derivative = (base_residual - minus_residual) / max(current_value - minus_value, 1.0e-12)
                else:
                    derivative = np.zeros_like(base_residual)
                sensitivity_columns.append(np.asarray(derivative, dtype=np.float64))
            return base_residual, np.stack(sensitivity_columns, axis=1)

        base_residual, tt_sensitivity = _global_tt_sensitivity_matrix(x0_full, x0_reduced)
        projected_seed = x0_reduced.copy()
        try:
            lhs = tt_sensitivity.T @ tt_sensitivity + float(TT_GLOBAL_PROJECTION_REG) * np.eye(2, dtype=np.float64)
            rhs = -(tt_sensitivity.T @ base_residual)
            projected_step = np.linalg.solve(lhs, rhs)
            projected_seed = np.clip(
                x0_reduced + np.asarray(projected_step, dtype=np.float64).reshape(-1),
                [bound[0] for bound in reduced_bounds],
                [bound[1] for bound in reduced_bounds],
            )
        except np.linalg.LinAlgError:
            projected_seed = x0_reduced.copy()

        projected_seed_vector = _compose_global_tt_vector(x0_full, projected_seed[0], projected_seed[1])
        projected_seed_signature = inspect_signature(projected_seed_vector)
        stage_start_signature = inspect_signature(x0_full)
        if projected_seed_signature["residual_norm"] < stage_start_signature["residual_norm"]:
            local_log(
                f"Stage {stage_name}: accepted TT projected seed "
                f"residual={projected_seed_signature['residual_norm']:.6f} "
                f"(start={stage_start_signature['residual_norm']:.6f})"
            )
            x0_reduced = projected_seed.copy()
            x0_full = projected_seed_vector.copy()
        else:
            local_log(
                f"Stage {stage_name}: rejected TT projected seed "
                f"residual={projected_seed_signature['residual_norm']:.6f} "
                f"(start={stage_start_signature['residual_norm']:.6f})"
            )

        def reduced_objective(active_delta: np.ndarray) -> float:
            full_vector = x0_full.copy()
            active_delta = np.asarray(active_delta, dtype=np.float64).reshape(-1)
            full_vector[tt_x_indices] = float(active_delta[0])
            full_vector[tt_y_indices] = float(active_delta[1])
            result = _evaluate_candidate(full_vector, count_eval=True)
            residual = np.asarray(result["residual"], dtype=np.float64)
            projected = tt_sensitivity @ (np.linalg.pinv(tt_sensitivity, rcond=1.0e-6) @ residual)
            residual_norm = float(np.linalg.norm(residual))
            projected_norm = float(np.linalg.norm(projected))
            violation = result["violation"]
            cost = (
                float(TT_GLOBAL_PROJECTED_RESIDUAL_WEIGHT) * projected_norm
                + float(TT_GLOBAL_FULL_RESIDUAL_WEIGHT) * residual_norm
                + bench.CONSTRAINT_PENALTY_WEIGHT * float(violation["l2_mm"])
            )
            if max_eval_limit is not None and tracker.evaluation_count >= max_eval_limit:
                raise bench.EarlyStopOptimization(f"Reached max evaluation limit of {max_eval_limit}.")
            if tracker.feasible_count > 0 and tracker.evaluation_count - tracker.last_best_eval >= bench.BEST_PATIENCE_EVALS:
                raise bench.EarlyStopOptimization(
                    f"No best-cost improvement for {bench.BEST_PATIENCE_EVALS} evaluations."
                )
            return float(cost)

        stage_status = 0
        stage_success = False
        stage_message = ""
        stage_nfev = 0
        x_final = x0_full.copy()
        stage_continues_after_patience = False
        try:
            stage_result = minimize(
                reduced_objective,
                x0=x0_reduced,
                method="L-BFGS-B",
                bounds=reduced_bounds,
                callback=optimizer_callback,
                options={
                    "maxiter": stage_maxiter_limit,
                    "maxfun": max(int(bench.OPT_MAXFUN), 24),
                    "ftol": 1.0e-6,
                    "eps": bench.OPT_DIFF_STEP_MM,
                },
            )
            reduced_final = np.asarray(stage_result.x, dtype=np.float64).reshape(-1)
            x_final[tt_x_indices] = float(reduced_final[0])
            x_final[tt_y_indices] = float(reduced_final[1])
            stage_status = int(stage_result.status)
            stage_success = bool(stage_result.success)
            stage_message = str(stage_result.message)
            stage_nfev = int(stage_result.nfev)
        except bench.EarlyStopOptimization as exc:
            if "Reached max evaluation limit" in str(exc):
                raise
            best_stage_vector = tracker.best_delta_mm if np.isfinite(tracker.best_cost) else tracker.best_any_delta_mm
            x_final = np.asarray(best_stage_vector, dtype=np.float64).reshape(-1).copy()
            stage_status = -1
            stage_success = False
            stage_message = f"EarlyStopOptimization: {exc}"
            stage_nfev = int(tracker.evaluation_count - stage_eval_before)
            stage_continues_after_patience = True
        stage_signature_after = inspect_signature(x_final)
        if stage_signature_after["residual_norm"] > stage_signature_before["residual_norm"]:
            x_final = np.asarray(start_vector, dtype=np.float64).reshape(-1).copy()
            stage_signature_after = inspect_signature(x_final)
            stage_message = str(stage_message) + " | reverted_to_stage_start"
        stage_records.append(
            {
                "stage_name": stage_name,
                "active_block": "tt_global",
                "active_dims": 2,
                "xyz_bias_weight": 0.0,
                "tt_variance_weight": 0.0,
                "evals_used": int(tracker.evaluation_count - stage_eval_before),
                "optimizer_iterations_used": int(optimizer_state["iteration_count"] - stage_iter_before),
                "stage_runtime_s": float(time.perf_counter() - stage_started_at),
                "residual_before": float(stage_signature_before["residual_norm"]),
                "residual_after": float(stage_signature_after["residual_norm"]),
                "best_residual_before": float(stage_best_before),
                "best_residual_after": float(tracker.best_cost),
                "signature_before": stage_signature_before,
                "signature_after": stage_signature_after,
                "optimizer_status": int(stage_status),
                "optimizer_success": bool(stage_success),
                "optimizer_message": str(stage_message),
                "function_evaluations": int(stage_nfev),
                "continued_after_patience_stop": bool(stage_continues_after_patience),
            }
        )
        local_log(
            f"Stage {stage_name}: finish residual={stage_signature_after['residual_norm']:.6f} "
            f"best={tracker.best_cost:.6f} evals_used={tracker.evaluation_count - stage_eval_before}"
        )
        optimizer_state["success"] = bool(stage_success)
        optimizer_state["status"] = int(stage_status)
        optimizer_state["message"] = str(stage_message)
        optimizer_state["nfev"] = int(optimizer_state["nfev"]) + int(stage_nfev)
        tracker.last_best_eval = int(tracker.evaluation_count)
        if stage_continues_after_patience:
            local_log(f"Stage {stage_name}: continuing to next block from current best state")
        return x_final

    def optimize_local_tt_block(
        stage_name: str,
        start_vector: np.ndarray,
        tt_x_indices: np.ndarray,
        tt_y_indices: np.ndarray,
        *,
        stage_tt_limit_deg: float,
        tt_variance_weight: float = 0.0,
    ) -> np.ndarray:
        if tt_x_indices.size == 0 or tt_y_indices.size == 0:
            return np.asarray(start_vector, dtype=np.float64).copy()

        x0_full = np.asarray(start_vector, dtype=np.float64).reshape(-1).copy()
        base_mean_tx = float(np.mean(x0_full[tt_x_indices]))
        base_mean_ty = float(np.mean(x0_full[tt_y_indices]))
        local_limit_deg = min(float(stage_tt_limit_deg), float(TT_LOCAL_RESIDUAL_LIMIT_ARCMIN) / 60.0)
        local_limit_deg = max(local_limit_deg, 1.0e-6)
        reduced_size = int(tt_x_indices.size + tt_y_indices.size)
        x0_reduced = np.zeros(reduced_size, dtype=np.float64)
        reduced_bounds = [(-local_limit_deg, local_limit_deg)] * reduced_size

        stage_started_at = time.perf_counter()
        stage_eval_before = int(tracker.evaluation_count)
        stage_iter_before = int(optimizer_state["iteration_count"])
        stage_best_before = float(tracker.best_cost)
        stage_signature_before = inspect_signature(x0_full)
        stage_maxiter_limit = max(1, min(int(optimizer_maxiter_limit), int(BLOCK_STAGE_MAXITER)))
        local_log(
            f"Stage {stage_name}: start residual={stage_signature_before['residual_norm']:.6f} "
            f"angular_over_radial={stage_signature_before['angular_over_radial']:.3f} "
            f"higher_over_angular={stage_signature_before['higher_over_angular']:.3f} "
            f"active_dims={reduced_size} "
            f"stage_maxiter={stage_maxiter_limit} "
            f"tt_variance_weight={float(tt_variance_weight):.4f} "
            f"tt_local_limit_arcmin={float(local_limit_deg) * 60.0:.3f}"
        )

        def compose_local_tt_vector(active_delta: np.ndarray) -> np.ndarray:
            active_delta = np.asarray(active_delta, dtype=np.float64).reshape(-1)
            dx = active_delta[: int(tt_x_indices.size)].copy()
            dy = active_delta[int(tt_x_indices.size) :].copy()
            dx -= float(np.mean(dx))
            dy -= float(np.mean(dy))
            full_vector = x0_full.copy()
            full_vector[tt_x_indices] = base_mean_tx + dx
            full_vector[tt_y_indices] = base_mean_ty + dy
            return full_vector

        def reduced_objective(active_delta: np.ndarray) -> float:
            full_vector = compose_local_tt_vector(active_delta)
            cost = float(objective(full_vector))
            normalized = np.asarray(active_delta, dtype=np.float64).reshape(-1) / max(local_limit_deg, 1.0e-12)
            cost += float(TT_LOCAL_L2_WEIGHT) * float(np.mean(normalized * normalized))
            if tt_variance_weight > 0.0:
                tx = full_vector[tt_x_indices]
                ty = full_vector[tt_y_indices]
                variance_penalty = float(np.var(tx) + np.var(ty)) / float(local_limit_deg * local_limit_deg)
                cost += float(tt_variance_weight) * variance_penalty
            return cost

        stage_status = 0
        stage_success = False
        stage_message = ""
        stage_nfev = 0
        x_final = x0_full.copy()
        stage_continues_after_patience = False
        try:
            stage_result = minimize(
                reduced_objective,
                x0=x0_reduced,
                method="L-BFGS-B",
                bounds=reduced_bounds,
                callback=optimizer_callback,
                options={
                    "maxiter": stage_maxiter_limit,
                    "maxfun": max(int(bench.OPT_MAXFUN), 12 * reduced_size),
                    "ftol": 1.0e-6,
                    "eps": bench.OPT_DIFF_STEP_MM,
                },
            )
            x_final = compose_local_tt_vector(stage_result.x)
            stage_status = int(stage_result.status)
            stage_success = bool(stage_result.success)
            stage_message = str(stage_result.message)
            stage_nfev = int(stage_result.nfev)
        except bench.EarlyStopOptimization as exc:
            if "Reached max evaluation limit" in str(exc):
                raise
            best_stage_vector = tracker.best_delta_mm if np.isfinite(tracker.best_cost) else tracker.best_any_delta_mm
            x_final = np.asarray(best_stage_vector, dtype=np.float64).reshape(-1).copy()
            stage_status = -1
            stage_success = False
            stage_message = f"EarlyStopOptimization: {exc}"
            stage_nfev = int(tracker.evaluation_count - stage_eval_before)
            stage_continues_after_patience = True

        stage_signature_after = inspect_signature(x_final)
        if stage_signature_after["residual_norm"] > stage_signature_before["residual_norm"]:
            x_final = np.asarray(start_vector, dtype=np.float64).reshape(-1).copy()
            stage_signature_after = inspect_signature(x_final)
            stage_message = str(stage_message) + " | reverted_to_stage_start"

        stage_records.append(
            {
                "stage_name": stage_name,
                "active_block": "tt_local_zero_mean",
                "active_dims": int(reduced_size),
                "xyz_bias_weight": 0.0,
                "tt_variance_weight": float(tt_variance_weight),
                "evals_used": int(tracker.evaluation_count - stage_eval_before),
                "optimizer_iterations_used": int(optimizer_state["iteration_count"] - stage_iter_before),
                "stage_runtime_s": float(time.perf_counter() - stage_started_at),
                "residual_before": float(stage_signature_before["residual_norm"]),
                "residual_after": float(stage_signature_after["residual_norm"]),
                "best_residual_before": float(stage_best_before),
                "best_residual_after": float(tracker.best_cost),
                "signature_before": stage_signature_before,
                "signature_after": stage_signature_after,
                "optimizer_status": int(stage_status),
                "optimizer_success": bool(stage_success),
                "optimizer_message": str(stage_message),
                "function_evaluations": int(stage_nfev),
                "continued_after_patience_stop": bool(stage_continues_after_patience),
            }
        )
        local_log(
            f"Stage {stage_name}: finish residual={stage_signature_after['residual_norm']:.6f} "
            f"best={tracker.best_cost:.6f} evals_used={tracker.evaluation_count - stage_eval_before}"
        )
        optimizer_state["success"] = bool(stage_success)
        optimizer_state["status"] = int(stage_status)
        optimizer_state["message"] = str(stage_message)
        optimizer_state["nfev"] = int(optimizer_state["nfev"]) + int(stage_nfev)
        tracker.last_best_eval = int(tracker.evaluation_count)
        if stage_continues_after_patience:
            local_log(f"Stage {stage_name}: continuing to next block from current best state")
        return x_final

    started_at = time.perf_counter()
    try:
        local_log("Starting repair optimization")
        local_log(
            "Search configuration: "
            f"search_box_radius_mm={search_box_radius_mm:.6f} "
            f"search_box_xy_radius_mm={search_box_xy_radius_mm:.6f} "
            f"search_box_z_radius_mm={search_box_z_radius_mm:.6f} "
            f"search_box_scale={SEARCH_BOX_SCALE:.3f} "
            f"finite_diff_step_mm={bench.OPT_DIFF_STEP_MM:.6f} "
            f"optimizer_maxiter={optimizer_maxiter_limit} "
            f"max_eval_limit={'none' if max_eval_limit is None else max_eval_limit}"
        )
        local_log(
            "Seed strategy: "
            f"{'hold + blind_nominal_reset' if ENABLE_BLIND_SEED else 'hold_only'} "
            f"(mechanical_steps_mm={[float(v) for v in shwfs.mechanical_steps_mm]}, "
            f"response_shape={tuple(int(v) for v in shwfs.mechanical_response_matrix.shape)}, "
            f"slope_channels={int(shwfs.reference_slopes.size)}, "
            f"lenslet_count={int(shwfs.lenslet_count)}, "
            f"lenslet_pixels={int(shwfs.lenslet_pixel_count)}, "
            f"tiptilt_enabled={ENABLE_TIPTILT}, "
            f"tiptilt_limit_arcmin={TIPTILT_LOCAL_LIMIT_ARCMIN:.3f}, "
            f"noise_profile={str(shwfs.noise_profile_name)}, "
            f"bit_depth={int(shwfs.noise_profile.bit_depth if shwfs.noise_profile else 0)}, "
            f"peak_e={float(shwfs.noise_profile.peak_e if shwfs.noise_profile else 0.0):.1f}, "
            f"analog_gain_adu_per_e={float(shwfs.noise_profile.analog_gain_adu_per_e if shwfs.noise_profile else 0.0):.3f}, "
            f"black_level_adu={float(shwfs.noise_profile.black_level_adu if shwfs.noise_profile else 0.0):.1f}, "
            f"full_well_e={float(shwfs.noise_profile.full_well_e if shwfs.noise_profile else 0.0):.1f}, "
            f"read_noise_rms_e={float(shwfs.noise_profile.read_noise_rms_e if shwfs.noise_profile else 0.0):.2f}, "
            f"blind_seed_dz_weight={BLIND_SEED_DZ_WEIGHT:.3f})"
        )
        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            seed_candidates: list[tuple[str, np.ndarray]] = [("hold", hold_seed)]
            if ENABLE_BLIND_SEED and not np.allclose(blind_nominal_seed, hold_seed, atol=1.0e-12, rtol=0.0):
                seed_candidates.append(("blind_nominal_reset", blind_nominal_seed))
            chosen_seed_name = "hold"
            chosen_seed = hold_seed.copy()
            chosen_cost = np.inf
            for seed_name, seed_vector in seed_candidates:
                seed_cost = float(objective(seed_vector))
                local_log(f"seed={seed_name} objective={seed_cost:.6f}")
                if seed_cost < chosen_cost:
                    chosen_seed_name = seed_name
                    chosen_seed = seed_vector.copy()
                    chosen_cost = seed_cost
            local_log(f"Selected initial seed: {chosen_seed_name}")
            try:
                block_indices = _active_block_indices(ACTUATOR_IDS)
                current_vector = chosen_seed.copy()
                initial_signature = inspect_signature(current_vector)
                primary_order = _choose_primary_block_order(initial_signature)
                tt_dominant_search = _tt_dominant_case(initial_signature)
                initial_xyz_bias_weight = float(TT_DOMINANT_XYZ_BIAS_WEIGHT if tt_dominant_search else 0.0)
                local_log(
                    "Residual-guided block schedule: "
                    f"initial angular_over_radial={initial_signature['angular_over_radial']:.3f} "
                    f"higher_over_angular={initial_signature['higher_over_angular']:.3f} "
                    f"primary_order={primary_order} "
                    f"tt_dominant_search={tt_dominant_search} "
                    f"initial_xyz_bias_weight={initial_xyz_bias_weight:.4f}"
                )
                current_vector = optimize_block(
                    f"stage1_{primary_order[0]}",
                    current_vector,
                    block_indices[primary_order[0]],
                    xyz_bias_weight=initial_xyz_bias_weight if primary_order[0] in {"xy", "z"} else 0.0,
                )
                current_vector = optimize_block(
                    f"stage2_{primary_order[1]}",
                    current_vector,
                    block_indices[primary_order[1]],
                    xyz_bias_weight=initial_xyz_bias_weight if primary_order[1] in {"xy", "z"} else 0.0,
                )
                if ENABLE_TIPTILT and block_indices["tt"].size > 0:
                    tt_limit_deg = min(float(TIPTILT_LOCAL_LIMIT_DEG), float(TIPTILT_TEST_LIMIT_ARCMIN / 60.0))
                    current_vector = optimize_global_tt_block(
                        "stage3_tt_global",
                        current_vector,
                        block_indices["tt_x"],
                        block_indices["tt_y"],
                        stage_tt_limit_deg=tt_limit_deg,
                    )
                    current_vector = optimize_local_tt_block(
                        "stage4_tt_local",
                        current_vector,
                        block_indices["tt_x"],
                        block_indices["tt_y"],
                        stage_tt_limit_deg=tt_limit_deg,
                        tt_variance_weight=float(TT_LOCAL_VARIANCE_WEIGHT),
                    )
                final_primary_order = _choose_primary_block_order(inspect_signature(current_vector))
                current_vector = optimize_block(
                    f"stage5_{final_primary_order[0]}_polish",
                    current_vector,
                    block_indices[final_primary_order[0]],
                )
            except bench.EarlyStopOptimization as exc:
                optimizer_state["success"] = False
                optimizer_state["status"] = -1
                optimizer_state["message"] = f"EarlyStopOptimization: {exc}"
                local_log(f"Early stop: {exc}")
            optimization_warnings = [str(record.message) for record in warning_records]

        optimization_runtime_s = time.perf_counter() - started_at
        local_log(f"Optimization finished in {optimization_runtime_s:.2f}s with {tracker.evaluation_count} evaluations")
        ray_tracing_runtime_s = float(timing_totals["psf_s"] + timing_totals["shwfs_s"])
        local_log(
            "Timing summary: "
            f"optimizer_wall={optimization_runtime_s:.2f}s "
            f"ray_tracing={ray_tracing_runtime_s:.2f}s "
            f"set_state={timing_totals['set_state_s']:.2f}s "
            f"constraint={timing_totals['constraint_s']:.2f}s"
        )
        best_delta = tracker.best_delta_mm.copy()
        if not np.isfinite(tracker.best_cost):
            best_delta = tracker.best_any_delta_mm.copy()
            local_log("No strictly feasible best state found; falling back to best_any_delta")

        repaired_model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
        repaired_model.set_surface_perturbations(hidden_start_state)
        repaired_state = refined_nominal._delta_state(ACTUATOR_IDS, hidden_start_state, best_delta)
        repaired_model.set_surface_perturbations(repaired_state)
        repaired_applied_state = bench._anchor_state_subset(repaired_model.get_surface_perturbations(), ACTUATOR_IDS)
        repaired_violation = bench._constraint_violation(repaired_state, repaired_applied_state, ACTUATOR_IDS)
        repaired_psf = repaired_model.get_psf_image(focus=bench.REFERENCE_FOCUS_MM)
        repaired_metrics = bench._state_metrics(
            repaired_model,
            shwfs,
            repaired_psf,
            ideal_psf=ideal_psf,
            nominal_psf=nominal_psf,
            true_reference_coeffs=true_reference_coeffs,
        )
        repaired_mech = bench._mechanical_summary(repaired_model)
        local_log(
            "Repaired metrics: "
            f"strehl={repaired_metrics['strehl_ratio']:.6f}, "
            f"wrms={repaired_metrics['wavefront_rms_waves']:.6f}, "
            f"residual={repaired_metrics['residual_norm']:.6f}, "
            f"clamp_max={repaired_violation['max_abs_mm']:.3e} mm, "
            f"min_gap={repaired_mech['minimum_gap_mm']:.6f} mm"
        )

        fov_rows: list[dict[str, float]] = []
        local_log("Evaluating FOV RMS curves")
        fov_rows.extend(bench._evaluate_fov_curve({}, label="nominal"))
        fov_rows.extend(bench._evaluate_fov_curve(hidden_start_state, label="moved"))
        fov_rows.extend(bench._evaluate_fov_curve(repaired_applied_state, label="repaired"))

        moved_residual = max(float(moved_metrics["residual_norm"]), 1.0e-12)
        moved_rmse = max(float(moved_metrics["rmse_to_ideal_perfect"]), 1.0e-12)
        normalized_metrics = {
            "nominal": {
                "w_bessel_residual_vs_moved": float(nominal_metrics["residual_norm"]) / moved_residual,
                "psf_rmse_to_ideal_vs_moved": float(nominal_metrics["rmse_to_ideal_perfect"]) / moved_rmse,
            },
            "moved": {
                "w_bessel_residual_vs_moved": 1.0,
                "psf_rmse_to_ideal_vs_moved": 1.0,
            },
            "repaired": {
                "w_bessel_residual_vs_moved": float(repaired_metrics["residual_norm"]) / moved_residual,
                "psf_rmse_to_ideal_vs_moved": float(repaired_metrics["rmse_to_ideal_perfect"]) / moved_rmse,
            },
        }

        summary = {
            "status": "complete",
            "algorithm": "adaptive_optics_v5_global_tt_first_noisy_shwfs_block_search",
            "configuration": {
                "wavelength_nm": float(bench.WAVELENGTH_NM),
                "object_space_mm": float(bench.OBJECT_SPACE_MM),
                "reference_focus_mm": float(bench.REFERENCE_FOCUS_MM),
                "fast_pupil_samples": int(bench.FAST_PUPIL_SAMPLES),
                "fast_fft_samples": int(bench.FAST_FFT_SAMPLES),
                "overall_variation_mm": float(config.overall_variation_mm),
                "per_anchor_overrides": config.per_anchor_overrides or {},
                "initial_seed_strategy": "hold_plus_shwfs_blind_linearized_reset" if ENABLE_BLIND_SEED else "hold_only",
                "optimizer_maxiter_limit": int(optimizer_maxiter_limit),
                "max_eval_limit": None if max_eval_limit is None else int(max_eval_limit),
                "optimizer_maxfun_limit": int(optimizer_maxfun_limit),
                "mechanical_response_step_mm": float(shwfs.mechanical_step_mm),
                "mechanical_response_steps_mm": [float(v) for v in shwfs.mechanical_steps_mm],
                "mechanical_response_rcond": float(real_shwfs.MECHANICAL_RESPONSE_RCOND),
                "shwfs_lenslet_count": int(shwfs.lenslet_count),
                "shwfs_lenslet_pixel_count": int(shwfs.lenslet_pixel_count),
                "shwfs_slope_channels": int(shwfs.reference_slopes.size),
                "shwfs_slope_limit": int(real_shwfs.SHWFS_SLOPE_LIMIT),
                "tiptilt_enabled": bool(ENABLE_TIPTILT),
                "tiptilt_limit_arcmin": float(TIPTILT_LOCAL_LIMIT_ARCMIN),
                "tiptilt_limit_deg": float(TIPTILT_LOCAL_LIMIT_DEG),
                "tiptilt_response_step_arcmin": float(TIPTILT_RESPONSE_STEP_ARCMIN),
                "tiptilt_response_step_deg": float(TIPTILT_RESPONSE_STEP_DEG),
                "shwfs_noise_profile": str(shwfs.noise_profile_name),
                "shwfs_noise_seed": int(shwfs.noise_seed),
                "shwfs_forward_averages": int(shwfs.forward_average_count),
                "shwfs_noise_parameters": {
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
                    "spot_sigma_px": float(shwfs.noise_profile.spot_sigma_px if shwfs.noise_profile else 0.0),
                    "slope_to_pixel_gain": float(
                        shwfs.noise_profile.slope_to_pixel_gain if shwfs.noise_profile else 0.0
                    ),
                },
                "shwfs_resolution": {
                    "lenslet_grid": [int(shwfs.lenslet_count), int(shwfs.lenslet_count)],
                    "slope_channels": int(shwfs.reference_slopes.size),
                    "slope_limit": int(real_shwfs.SHWFS_SLOPE_LIMIT),
                    "lenslet_pixels": int(shwfs.lenslet_pixel_count),
                    "pupil_samples": int(nominal_model.pupil_samples),
                },
                "search_box_radius_mm": float(search_box_radius_mm),
                "search_box_xy_radius_mm": float(search_box_xy_radius_mm),
                "search_box_z_radius_mm": float(search_box_z_radius_mm),
                "search_box_scale": float(SEARCH_BOX_SCALE),
                "search_box_reference_frame": "relative_zero_at_pre_repair_position",
                "blind_seed_dz_weight": float(BLIND_SEED_DZ_WEIGHT),
                "blind_seed_enabled": bool(ENABLE_BLIND_SEED),
                "tiptilt_test_limit_arcmin": float(TIPTILT_TEST_LIMIT_ARCMIN),
                "tt_dominant_angular_ratio": float(TT_DOMINANT_ANGULAR_RATIO),
                "tt_dominant_xyz_bias_weight": float(TT_DOMINANT_XYZ_BIAS_WEIGHT),
                "tt_local_variance_weight": float(TT_LOCAL_VARIANCE_WEIGHT),
                "block_stage_maxiter": int(BLOCK_STAGE_MAXITER),
            },
            "closed_loop_truth_policy": {
                "allowed_during_repair": [
                    "current_shwfs_measurement",
                    "nominal_shwfs_measurement_reference",
                    "precomputed_mechanical_to_shwfs_response_matrix",
                    "mechanical_bounds_and_gap_constraints",
                    "relative_zero_centered_search_box",
                ],
                "forbidden_during_repair": [
                    "ground_truth_moved_delta",
                    "hidden_absolute_start_anchor_state",
                    "ground_truth_nominal_reset_delta",
                    "true_residual_norm",
                    "ideal_psf_error_terms",
                ],
                "evaluation_only": [
                    "ideal_psf",
                    "true_reference_coeffs",
                    "true_residual_norm",
                    "nominal_psf",
                ],
            },
            "nominal": nominal_metrics,
            "moved": moved_metrics,
            "repaired": repaired_metrics,
            "normalized_metrics": normalized_metrics,
            "requested_position_by_anchor": {
                str(surface_id): {
                    "dx_mm": float(requested_state.get(surface_id, SurfacePerturbation()).dx_mm),
                    "dy_mm": float(requested_state.get(surface_id, SurfacePerturbation()).dy_mm),
                    "dz_mm": float(requested_state.get(surface_id, SurfacePerturbation()).dz_mm),
                    "tilt_x_deg": float(requested_state.get(surface_id, SurfacePerturbation()).tilt_x_deg),
                    "tilt_y_deg": float(requested_state.get(surface_id, SurfacePerturbation()).tilt_y_deg),
                }
                for surface_id in ACTUATOR_IDS
            },
            "moved_position_by_anchor": {
                str(surface_id): {
                    "dx_mm": float(hidden_start_state.get(surface_id, SurfacePerturbation()).dx_mm),
                    "dy_mm": float(hidden_start_state.get(surface_id, SurfacePerturbation()).dy_mm),
                    "dz_mm": float(hidden_start_state.get(surface_id, SurfacePerturbation()).dz_mm),
                    "tilt_x_deg": float(hidden_start_state.get(surface_id, SurfacePerturbation()).tilt_x_deg),
                    "tilt_y_deg": float(hidden_start_state.get(surface_id, SurfacePerturbation()).tilt_y_deg),
                }
                for surface_id in ACTUATOR_IDS
            },
            "final_position_by_anchor": {
                str(surface_id): {
                    "dx_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dx_mm),
                    "dy_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dy_mm),
                    "dz_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dz_mm),
                    "tilt_x_deg": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).tilt_x_deg),
                    "tilt_y_deg": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).tilt_y_deg),
                }
                for surface_id in ACTUATOR_IDS
            },
            "delta_to_nominal_by_anchor": {
                str(surface_id): {
                    "dx_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dx_mm),
                    "dy_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dy_mm),
                    "dz_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dz_mm),
                    "tilt_x_deg": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).tilt_x_deg),
                    "tilt_y_deg": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).tilt_y_deg),
                }
                for surface_id in ACTUATOR_IDS
            },
            "clamp_summary": clamp_summary,
            "repaired_constraint_violation_mm": repaired_violation,
            "moved_mechanical_summary": bench._mechanical_summary(moved_model),
            "repaired_mechanical_summary": repaired_mech,
            "iteration_count": int(tracker.evaluation_count),
            "optimizer_iteration_count": int(optimizer_state["iteration_count"]),
            "optimizer_status": optimizer_state["status"],
            "optimizer_success": optimizer_state["success"],
            "optimizer_message": str(optimizer_state["message"]),
            "optimizer_function_evaluations": int(optimizer_state["nfev"]),
            "block_search_stages": stage_records,
            "block_search_stage_count": int(len(stage_records)),
            "new_best_events": new_best_events,
            "new_best_event_count": int(len(new_best_events)),
            "runtime_s": float(time.perf_counter() - started_at),
            "optimization_runtime_s": float(optimization_runtime_s),
            "ray_tracing_runtime_s": float(ray_tracing_runtime_s),
            "timing_summary_s": {
                "optimizer_wall_s": float(optimization_runtime_s),
                "ray_tracing_s": float(ray_tracing_runtime_s),
                "objective_set_state_s": float(timing_totals["set_state_s"]),
                "objective_constraint_s": float(timing_totals["constraint_s"]),
                "objective_psf_s": float(timing_totals["psf_s"]),
                "objective_shwfs_s": float(timing_totals["shwfs_s"]),
            },
            "resource_breakdown_s": timing_totals,
            "warnings": optimization_warnings,
            "psfs": {
                "nominal": np.asarray(nominal_psf, dtype=np.float32),
                "moved": np.asarray(context["moved_model"].get_psf_image(focus=bench.REFERENCE_FOCUS_MM), dtype=np.float32),
                "repaired": np.asarray(repaired_psf, dtype=np.float32),
            },
            "fov_rms_rows": fov_rows,
        }

        bench._write_csv(summary)
        bench._write_fov_csv(fov_rows)
        bench._write_npz(summary)
        json_ready = copy.deepcopy(summary)
        json_ready.pop("psfs", None)
        _write_json(bench.OUTPUT_JSON_PATH, json_ready)
        context["moved_model"].set_surface_perturbations(hidden_start_state)
        bench._render_lightpath_wavefront_psf(nominal_model, context["moved_model"], repaired_model, summary)
        bench._render_metrics(summary)
        bench._render_fov_rms(fov_rows)

        ui_result = {
            "run_dir": str(run_dir),
            "before_summary_path": str(run_dir / "before_calibration_summary.json"),
            "repair_summary_path": str(bench.OUTPUT_JSON_PATH),
            "before_figure_path": str(run_dir / "before_calibration_wavefront_psf_lightpath.png"),
            "repair_compare_figure_path": str(bench.OUTPUT_LIGHTPATH_PNG_PATH),
            "repair_metrics_figure_path": str(bench.OUTPUT_METRICS_PNG_PATH),
            "repair_fov_figure_path": str(bench.OUTPUT_FOV_PNG_PATH),
            "summary_text": _summary_text_block(summary),
        }
        return ui_result
    finally:
        _restore_benchmark_output_prefix(old_outputs)
