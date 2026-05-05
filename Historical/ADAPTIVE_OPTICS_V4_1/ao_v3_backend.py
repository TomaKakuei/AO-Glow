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


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
BASE_GEOMETRY_PATH = ARTIFACTS_DIR / "geometry_inspection_1000_after_coupled_gap.json"
ACTUATOR_IDS = [68, 70, 72, 74, 77, 79, 81, 84]
SEARCH_BOX_SCALE = 1.5
SEARCH_BOX_MIN_MM = 0.02
BLIND_SEED_DZ_WEIGHT = 0.25
ENABLE_TIPTILT = os.environ.get("AO_V3_ENABLE_TIPTILT", "").strip().lower() in {"1", "true", "yes", "on"}
TIPTILT_LOCAL_LIMIT_ARCMIN = float(os.environ.get("AO_V3_TIPTILT_LIMIT_ARCMIN", "10.0"))
TIPTILT_LOCAL_LIMIT_DEG = float(TIPTILT_LOCAL_LIMIT_ARCMIN / 60.0)
TIPTILT_RESPONSE_STEP_ARCMIN = float(os.environ.get("AO_V3_TIPTILT_RESPONSE_STEP_ARCMIN", "1.0"))
TIPTILT_RESPONSE_STEP_DEG = float(TIPTILT_RESPONSE_STEP_ARCMIN / 60.0)
ENABLE_BLIND_SEED = os.environ.get("AO_V3_DISABLE_BLIND_SEED", "").strip().lower() not in {"1", "true", "yes", "on"}

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
            tilt_x_deg=float(current.tilt_x_deg),
            tilt_y_deg=float(current.tilt_y_deg),
        )
    return state


def _build_requested_state(config: RunConfig) -> dict[int, SurfacePerturbation]:
    scaled = _base_anchor_state_scaled(config.overall_variation_mm)
    return _apply_overrides(scaled, config.per_anchor_overrides)


def _repair_search_box_radius_mm(config: RunConfig) -> float:
    estimated_scale_mm = max(float(config.overall_variation_mm), 0.0)
    return max(float(SEARCH_BOX_MIN_MM), float(SEARCH_BOX_SCALE) * estimated_scale_mm)


def _optimizer_maxiter_limit(config: RunConfig) -> int:
    if config.optimizer_maxiter_limit is None:
        return int(bench.OPT_MAXITER)
    return max(1, int(config.optimizer_maxiter_limit))


def _max_eval_limit(config: RunConfig) -> int | None:
    if config.max_eval_limit is None:
        return None
    return max(1, int(config.max_eval_limit))


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


def _downweight_blind_seed_dz(delta_vector: np.ndarray, actuator_ids: list[int]) -> np.ndarray:
    dof_count = 5 if ENABLE_TIPTILT else 3
    weighted = np.asarray(delta_vector, dtype=np.float64).reshape(len(actuator_ids), dof_count).copy()
    weighted[:, 2] *= float(BLIND_SEED_DZ_WEIGHT)
    return weighted.reshape(-1)


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

    base_state = bench._anchor_state_subset(moved_model.get_surface_perturbations(), ACTUATOR_IDS)
    hold_seed = np.zeros((5 if ENABLE_TIPTILT else 3) * len(ACTUATOR_IDS), dtype=np.float64)
    search_box_radius_mm = _repair_search_box_radius_mm(config)
    optimizer_maxiter_limit = _optimizer_maxiter_limit(config)
    max_eval_limit = _max_eval_limit(config)
    optimizer_maxfun_limit = max(int(bench.OPT_MAXFUN), 50 * int(optimizer_maxiter_limit))
    bounds = _v3_delta_bounds(
        moved_model,
        ACTUATOR_IDS,
        base_state,
        search_box_radius_mm=search_box_radius_mm,
    )
    blind_nominal_seed = hold_seed.copy()
    if ENABLE_BLIND_SEED:
        blind_nominal_seed = bench._project_delta_to_bounds(
            _downweight_blind_seed_dz(
                shwfs.estimate_nominal_correction_seed(moved_model),
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

    output_prefix = run_dir / "repair_result"
    old_outputs = _set_benchmark_output_prefix(output_prefix)
    bench.OUTPUT_LOG_PATH.write_text("", encoding="utf-8")

    def local_log(message: str) -> None:
        logger(message)
        bench._log(message)

    def objective(delta_vector: np.ndarray) -> float:
        eval_started = time.perf_counter()
        delta_vector = np.asarray(delta_vector, dtype=np.float64).reshape(-1)
        candidate_state = refined_nominal._delta_state(ACTUATOR_IDS, base_state, delta_vector)

        t0 = time.perf_counter()
        moved_model.set_surface_perturbations(candidate_state)
        applied_state = bench._anchor_state_subset(moved_model.get_surface_perturbations(), ACTUATOR_IDS)
        timing_totals["set_state_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        violation = bench._constraint_violation(candidate_state, applied_state, ACTUATOR_IDS)
        timing_totals["constraint_s"] += time.perf_counter() - t0

        rmse = float("nan")
        sharpness = float("nan")

        t0 = time.perf_counter()
        current_coeffs = shwfs.estimate_coeffs(moved_model)
        residual = current_coeffs - shwfs.reference_coeffs
        residual_norm = float(np.linalg.norm(residual))
        timing_totals["shwfs_s"] += time.perf_counter() - t0

        cost = residual_norm + bench.CONSTRAINT_PENALTY_WEIGHT * float(violation["l2_mm"])
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
        return cost

    def optimizer_callback(_xk: np.ndarray) -> None:
        optimizer_state["iteration_count"] = int(optimizer_state["iteration_count"]) + 1

    started_at = time.perf_counter()
    try:
        local_log("Starting repair optimization")
        local_log(
            "Search configuration: "
            f"search_box_radius_mm={search_box_radius_mm:.6f} "
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
            f"tiptilt_enabled={ENABLE_TIPTILT}, "
            f"tiptilt_limit_arcmin={TIPTILT_LOCAL_LIMIT_ARCMIN:.3f}, "
            f"noise_range=({float(shwfs.noise_min_fraction):.4f},{float(shwfs.noise_max_fraction):.4f}), "
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
                optimizer_result = minimize(
                    objective,
                    x0=chosen_seed,
                    method="L-BFGS-B",
                    bounds=bounds,
                    callback=optimizer_callback,
                    options={
                        "maxiter": optimizer_maxiter_limit,
                        "maxfun": optimizer_maxfun_limit,
                        "ftol": 1.0e-6,
                        "eps": bench.OPT_DIFF_STEP_MM,
                    },
                )
                optimizer_state["success"] = bool(optimizer_result.success)
                optimizer_state["status"] = int(optimizer_result.status)
                optimizer_state["message"] = str(optimizer_result.message)
                optimizer_state["iteration_count"] = int(optimizer_result.nit)
                optimizer_state["nfev"] = int(optimizer_result.nfev)
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
        repaired_model.set_surface_perturbations(base_state)
        repaired_state = refined_nominal._delta_state(ACTUATOR_IDS, base_state, best_delta)
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
        fov_rows.extend(bench._evaluate_fov_curve(base_state, label="moved"))
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
            "algorithm": "adaptive_optics_v3_repair_blind_seed",
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
                "shwfs_slope_channels": int(shwfs.reference_slopes.size),
                "shwfs_slope_limit": int(real_shwfs.SHWFS_SLOPE_LIMIT),
                "tiptilt_enabled": bool(ENABLE_TIPTILT),
                "tiptilt_limit_arcmin": float(TIPTILT_LOCAL_LIMIT_ARCMIN),
                "tiptilt_limit_deg": float(TIPTILT_LOCAL_LIMIT_DEG),
                "tiptilt_response_step_arcmin": float(TIPTILT_RESPONSE_STEP_ARCMIN),
                "tiptilt_response_step_deg": float(TIPTILT_RESPONSE_STEP_DEG),
                "shwfs_noise_min_fraction": float(shwfs.noise_min_fraction),
                "shwfs_noise_max_fraction": float(shwfs.noise_max_fraction),
                "shwfs_noise_seed": int(shwfs.noise_seed),
                "shwfs_resolution": {
                    "lenslet_grid": [int(shwfs.lenslet_count), int(shwfs.lenslet_count)],
                    "slope_channels": int(shwfs.reference_slopes.size),
                    "slope_limit": int(real_shwfs.SHWFS_SLOPE_LIMIT),
                    "pupil_samples": int(nominal_model.pupil_samples),
                },
                "search_box_radius_mm": float(search_box_radius_mm),
                "search_box_scale": float(SEARCH_BOX_SCALE),
                "blind_seed_dz_weight": float(BLIND_SEED_DZ_WEIGHT),
                "blind_seed_enabled": bool(ENABLE_BLIND_SEED),
            },
            "closed_loop_truth_policy": {
                "allowed_during_repair": [
                    "current_shwfs_measurement",
                    "nominal_shwfs_measurement_reference",
                    "precomputed_mechanical_to_shwfs_response_matrix",
                    "mechanical_bounds_and_gap_constraints",
                ],
                "forbidden_during_repair": [
                    "ground_truth_moved_delta",
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
                }
                for surface_id in ACTUATOR_IDS
            },
            "moved_position_by_anchor": {
                str(surface_id): {
                    "dx_mm": float(base_state.get(surface_id, SurfacePerturbation()).dx_mm),
                    "dy_mm": float(base_state.get(surface_id, SurfacePerturbation()).dy_mm),
                    "dz_mm": float(base_state.get(surface_id, SurfacePerturbation()).dz_mm),
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
        context["moved_model"].set_surface_perturbations(base_state)
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
