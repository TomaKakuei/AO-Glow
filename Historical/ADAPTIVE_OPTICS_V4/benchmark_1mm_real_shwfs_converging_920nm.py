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
from matplotlib.patches import Polygon as MplPolygon
from rayoptics.raytr.analyses import trace_ray_list
from scipy.optimize import minimize

import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal
from optical_model_rayoptics import MechanicalLimitWarning, RayOpticsPhysicsEngine, SurfacePerturbation


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
INPUT_GEOMETRY_JSON = ARTIFACTS_DIR / "geometry_inspection_1000_after_coupled_gap.json"

OUTPUT_PREFIX = ARTIFACTS_DIR / "wb_1mm_real_shwfs_converging_920nm"
OUTPUT_JSON_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_compare.json")
OUTPUT_CSV_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_compare.csv")
OUTPUT_FOV_CSV_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_fov_rms.csv")
OUTPUT_NPZ_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_psfs.npz")
OUTPUT_LIGHTPATH_PNG_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_lightpath_psf_wavefront.png")
OUTPUT_METRICS_PNG_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_metrics.png")
OUTPUT_FOV_PNG_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_fov_rms.png")
OUTPUT_LOG_PATH = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_run.log")

WAVELENGTH_NM = 920.0
OBJECT_SPACE_MM = -45.31
REFERENCE_FOCUS_MM = 4.64725296651765
FAST_PUPIL_SAMPLES = 64
FAST_FFT_SAMPLES = 256
BEAM_FILL_RATIO = 1.15
NUM_MODES = 8
LENSLET_COUNT = 8
LOCAL_RADIUS_MM = None
OPT_MAXITER = 4
OPT_MAXFUN = 60
OPT_DIFF_STEP_MM = 5.0e-3
BEST_PATIENCE_EVALS = 40
FOV_ANGLES_DEG = tuple(float(v) for v in range(0, 18, 2))
CONSTRAINT_TOL_MM = 1.0e-6
CONSTRAINT_PENALTY_WEIGHT = 1.0e5

warnings.filterwarnings("ignore", category=MechanicalLimitWarning)


class EarlyStopOptimization(RuntimeError):
    pass


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
    last_best_eval: int = 0


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with OUTPUT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _build_model(*, pupil_samples: int, fft_samples: int) -> RayOpticsPhysicsEngine:
    return RayOpticsPhysicsEngine(
        design_name="2P_AO",
        propagation_direction="forward",
        include_tube_lens=False,
        pupil_samples=int(pupil_samples),
        fft_samples=int(fft_samples),
        wavelength_nm=float(WAVELENGTH_NM),
        object_space_mm=float(OBJECT_SPACE_MM),
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_moved_state(
    geometry_json_path: Path,
    actuator_ids: list[int],
) -> tuple[dict[int, SurfacePerturbation], dict[str, Any]]:
    payload = _load_json(geometry_json_path)
    final_positions = payload.get("final_anchor_positions", {})
    moved_state: dict[int, SurfacePerturbation] = {}
    for surface_id in actuator_ids:
        item = final_positions.get(str(surface_id))
        if item is None:
            moved_state[int(surface_id)] = SurfacePerturbation()
            continue
        moved_state[int(surface_id)] = SurfacePerturbation(
            dx_mm=float(item.get("dx_mm", 0.0)),
            dy_mm=float(item.get("dy_mm", 0.0)),
            dz_mm=float(item.get("dz_mm", 0.0)),
        )
    return moved_state, payload


def _anchor_state_subset(
    state: dict[int, SurfacePerturbation],
    actuator_ids: list[int],
) -> dict[int, SurfacePerturbation]:
    return {
        int(surface_id): copy.deepcopy(state.get(surface_id, SurfacePerturbation()))
        for surface_id in actuator_ids
    }


def _flatten_anchor_state(
    state: dict[int, SurfacePerturbation],
    actuator_ids: list[int],
) -> np.ndarray:
    values: list[float] = []
    for surface_id in actuator_ids:
        perturbation = state.get(surface_id, SurfacePerturbation())
        values.extend([float(perturbation.dx_mm), float(perturbation.dy_mm), float(perturbation.dz_mm)])
    return np.asarray(values, dtype=np.float64)


def _constraint_violation(
    requested_state: dict[int, SurfacePerturbation],
    applied_state: dict[int, SurfacePerturbation],
    actuator_ids: list[int],
) -> dict[str, float]:
    requested = _flatten_anchor_state(requested_state, actuator_ids)
    applied = _flatten_anchor_state(applied_state, actuator_ids)
    delta = applied - requested
    return {
        "l2_mm": float(np.linalg.norm(delta)),
        "max_abs_mm": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _project_delta_to_bounds(
    delta_vector: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    projected = np.asarray(delta_vector, dtype=np.float64).reshape(-1).copy()
    if projected.size != len(bounds):
        raise ValueError("Delta vector size does not match bounds size.")
    for index, (lower, upper) in enumerate(bounds):
        projected[index] = float(np.clip(projected[index], lower, upper))
    return projected


def _nominal_reset_seed(
    actuator_ids: list[int],
    base_state: dict[int, SurfacePerturbation],
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    delta_values: list[float] = []
    for surface_id in actuator_ids:
        base = base_state.get(surface_id, SurfacePerturbation())
        delta_values.extend(
            [
                -float(base.dx_mm),
                -float(base.dy_mm),
                -float(base.dz_mm),
            ]
        )
    return _project_delta_to_bounds(np.asarray(delta_values, dtype=np.float64), bounds)


def _catalog_map(optical_model: RayOpticsPhysicsEngine) -> dict[int, dict[str, Any]]:
    catalog = optical_model.get_surface_catalog(
        include_object=False,
        include_image=False,
        include_sample_media=True,
    )
    return {int(entry["surface_id"]): entry for entry in catalog}


def _mechanical_summary(optical_model: RayOpticsPhysicsEngine) -> dict[str, Any]:
    perturbations = optical_model.get_surface_perturbations()
    anchors = sorted(optical_model.group_anchor_to_members)
    position_rows: list[dict[str, Any]] = []
    catalog_map = _catalog_map(optical_model)
    gap_rows: list[dict[str, Any]] = []

    for anchor_surface_id in anchors:
        envelope = optical_model.group_mechanical_envelopes[int(anchor_surface_id)]
        perturbation = perturbations.get(int(anchor_surface_id), SurfacePerturbation())
        position_rows.append(
            {
                "anchor_surface_id": int(anchor_surface_id),
                "member_surface_ids": list(envelope.member_surface_ids),
                "dx_mm": float(perturbation.dx_mm),
                "dy_mm": float(perturbation.dy_mm),
                "dz_mm": float(perturbation.dz_mm),
                "axial_min_mm": float(envelope.axial_min_mm),
                "axial_max_mm": float(envelope.axial_max_mm),
                "lateral_limit_mm": float(envelope.lateral_limit_mm),
                "within_x_limit": abs(float(perturbation.dx_mm)) <= float(envelope.lateral_limit_mm) + 1.0e-12,
                "within_y_limit": abs(float(perturbation.dy_mm)) <= float(envelope.lateral_limit_mm) + 1.0e-12,
                "within_z_limit": (
                    float(envelope.axial_min_mm) - 1.0e-12
                    <= float(perturbation.dz_mm)
                    <= float(envelope.axial_max_mm) + 1.0e-12
                ),
            }
        )

    for index in range(len(anchors) - 1):
        left_anchor = int(anchors[index])
        right_anchor = int(anchors[index + 1])
        left_members = optical_model.group_anchor_to_members[left_anchor]
        right_members = optical_model.group_anchor_to_members[right_anchor]
        left_last_surface = int(left_members[-1])
        right_first_surface = int(right_members[0])
        nominal_gap_mm = float(catalog_map[right_first_surface]["nominal_z_mm"]) - float(
            catalog_map[left_last_surface]["nominal_z_mm"]
        )
        left_dz = float(perturbations.get(left_anchor, SurfacePerturbation()).dz_mm)
        right_dz = float(perturbations.get(right_anchor, SurfacePerturbation()).dz_mm)
        actual_gap_mm = nominal_gap_mm + right_dz - left_dz
        gap_rows.append(
            {
                "left_anchor_surface_id": left_anchor,
                "right_anchor_surface_id": right_anchor,
                "nominal_gap_mm": nominal_gap_mm,
                "actual_gap_mm": float(actual_gap_mm),
                "overlap": bool(actual_gap_mm < 0.0),
            }
        )

    return {
        "group_positions": position_rows,
        "group_gap_report": gap_rows,
        "all_position_limits_satisfied": bool(
            all(
                row["within_x_limit"] and row["within_y_limit"] and row["within_z_limit"]
                for row in position_rows
            )
        ),
        "any_gap_overlap": bool(any(row["overlap"] for row in gap_rows)),
        "minimum_gap_mm": float(min(row["actual_gap_mm"] for row in gap_rows)) if gap_rows else float("nan"),
    }


def _state_metrics(
    optical_model: RayOpticsPhysicsEngine,
    shwfs: real_shwfs.ShwfsMeasurementModel,
    psf: np.ndarray,
    *,
    ideal_psf: np.ndarray,
    nominal_psf: np.ndarray,
    true_reference_coeffs: np.ndarray,
) -> dict[str, float]:
    wavefront_metrics = refined_nominal._wavefront_metrics(optical_model, focus=REFERENCE_FOCUS_MM)
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
        "sharpness": refined_nominal._sharpness(psf),
        "residual_norm": float(np.linalg.norm(estimated_residual)),
        "true_residual_norm": refined_nominal._residual_norm(
            optical_model,
            true_reference_coeffs,
            focus=REFERENCE_FOCUS_MM,
        ),
        "best_focus_shift_mm": float(REFERENCE_FOCUS_MM),
        **wavefront_metrics,
    }


def _evaluate_fov_curve(
    state_by_anchor: dict[int, SurfacePerturbation],
    *,
    label: str,
) -> list[dict[str, float]]:
    model = _build_model(pupil_samples=FAST_PUPIL_SAMPLES, fft_samples=FAST_FFT_SAMPLES)
    model.set_surface_perturbations(state_by_anchor)
    field = model.optical_spec["fov"].fields[0]
    rows: list[dict[str, float]] = []
    for angle_deg in FOV_ANGLES_DEG:
        field.x = float(angle_deg)
        model.opm.update_model()
        metrics = refined_nominal._wavefront_metrics(model, focus=REFERENCE_FOCUS_MM)
        rows.append(
            {
                "state": label,
                "field_angle_deg": float(angle_deg),
                "wavefront_rms_waves": float(metrics["wavefront_rms_waves"]),
                "wavefront_rms_low_order_removed_waves": float(metrics["wavefront_rms_low_order_removed_waves"]),
                "wavefront_rms_nm": float(metrics["wavefront_rms_nm"]),
                "pupil_valid_fraction": float(metrics["pupil_valid_fraction"]),
            }
        )
    field.x = 0.0
    model.opm.update_model()
    return rows


def _render_polygon_projected(
    catalog: list[dict[str, Any]],
    positions: dict[int, SurfacePerturbation],
) -> tuple[list[dict[str, Any]], tuple[float, float, float, float]]:
    projected: list[dict[str, Any]] = []
    z_min = np.inf
    z_max = -np.inf
    r_min = np.inf
    r_max = -np.inf
    for entry in catalog:
        surface_id = int(entry["surface_id"])
        perturbation = positions.get(surface_id, SurfacePerturbation())
        axial_mm = float(perturbation.dz_mm)
        lateral_mm = float(perturbation.dy_mm)
        nominal_z = float(entry.get("nominal_z_mm", 0.0))
        polygons = entry.get("render_polygon_mm") or []
        projected_polys: list[np.ndarray] = []
        for polygon in polygons:
            poly = np.asarray(polygon, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
                continue
            poly_plot = poly.copy()
            poly_plot[:, 0] += nominal_z + axial_mm
            poly_plot[:, 1] += lateral_mm
            projected_polys.append(poly_plot)
            z_min = min(z_min, float(np.min(poly_plot[:, 0])))
            z_max = max(z_max, float(np.max(poly_plot[:, 0])))
            r_min = min(r_min, float(np.min(poly_plot[:, 1])))
            r_max = max(r_max, float(np.max(poly_plot[:, 1])))
        projected.append(
            {
                "surface_id": surface_id,
                "is_compensator": bool(entry.get("is_compensator", False)),
                "polygons": projected_polys,
            }
        )
    if not np.isfinite(z_min):
        z_min, z_max, r_min, r_max = 0.0, 1.0, -1.0, 1.0
    return projected, (float(z_min), float(z_max), float(r_min), float(r_max))


def _trace_meridian(
    model: RayOpticsPhysicsEngine,
    *,
    pupil_coord: tuple[float, float],
) -> np.ndarray:
    fld = model.optical_spec["fov"].fields[0]
    rays = trace_ray_list(
        model.opm,
        [pupil_coord],
        fld,
        model.wavelength_nm,
        float(REFERENCE_FOCUS_MM),
        append_if_none=True,
        output_filter="last",
        rayerr_filter="summary",
        check_apertures=False,
    )
    if not rays or rays[0][2] is None:
        return np.zeros((0, 3), dtype=np.float64)
    segments = rays[0][2][0]
    points: list[np.ndarray] = []
    for seg in segments:
        point = np.asarray(seg[0], dtype=np.float64).reshape(-1)
        if point.size >= 3:
            points.append(point[:3])
    return np.vstack(points) if points else np.zeros((0, 3), dtype=np.float64)


def _wavefront_waves(model: RayOpticsPhysicsEngine, *, focus_mm: float) -> np.ndarray:
    opd = np.asarray(model.get_wavefront_opd(focus=float(focus_mm)), dtype=np.float64)
    wavelength_sys_units = float(model.opm.nm_to_sys_units(float(model.wavelength_nm)))
    return opd / max(wavelength_sys_units, 1.0e-15)


def _shared_lightpath_limits(models: list[RayOpticsPhysicsEngine]) -> tuple[float, float, float, float]:
    z_min = np.inf
    z_max = -np.inf
    r_min = np.inf
    r_max = -np.inf
    for model in models:
        catalog = model.get_surface_catalog(include_object=False, include_image=True, include_sample_media=True)
        _, limits = _render_polygon_projected(catalog, model.get_surface_perturbations())
        model_z_min, model_z_max, model_r_min, model_r_max = limits
        z_min = min(z_min, float(model_z_min))
        z_max = max(z_max, float(model_z_max))
        r_min = min(r_min, float(model_r_min))
        r_max = max(r_max, float(model_r_max))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or not np.isfinite(r_min) or not np.isfinite(r_max):
        return (-1.0, 1.0, -1.0, 1.0)
    return (z_min, z_max, r_min, r_max)


def _shared_wavefront_vmax(models: list[RayOpticsPhysicsEngine]) -> float:
    vmax = 0.0
    for model in models:
        wavefront = _wavefront_waves(model, focus_mm=REFERENCE_FOCUS_MM)
        if wavefront.size:
            vmax = max(vmax, float(np.nanmax(np.abs(wavefront))))
    return max(vmax, 1.0e-6)


def _draw_optical_panel(
    ax: plt.Axes,
    model: RayOpticsPhysicsEngine,
    *,
    state_name: str,
    metrics: dict[str, float],
    limits: tuple[float, float, float, float],
) -> None:
    catalog = model.get_surface_catalog(include_object=False, include_image=True, include_sample_media=True)
    projected, _ = _render_polygon_projected(catalog, model.get_surface_perturbations())
    for item in projected:
        fill_color = "#e26d5a" if item["is_compensator"] else "#9fb7c7"
        edge_color = "#7a2219" if item["is_compensator"] else "#32414b"
        for poly in item["polygons"]:
            ax.add_patch(
                MplPolygon(
                    poly,
                    closed=True,
                    facecolor=fill_color,
                    edgecolor=edge_color,
                    linewidth=0.7,
                    alpha=0.45 if item["is_compensator"] else 0.3,
                )
            )
    y_ray = _trace_meridian(model, pupil_coord=(0.0, 0.85))
    x_ray = _trace_meridian(model, pupil_coord=(0.85, 0.0))
    if y_ray.size:
        ax.plot(y_ray[:, 2], y_ray[:, 1], "-", color="#d1495b", linewidth=1.5, label="y meridian")
    if x_ray.size:
        ax.plot(x_ray[:, 2], x_ray[:, 0], "--", color="#7a7a7a", linewidth=1.2, label="x meridian")
    z_min, z_max, r_min, r_max = limits
    ax.axvline(float(REFERENCE_FOCUS_MM), color="#1f6feb", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.set_xlim(z_min - 4.0, z_max + 10.0)
    ax.set_ylim(r_min - 2.0, r_max + 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{state_name} Light Path", fontsize=10)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("y / x (mm)")
    ax.grid(True, alpha=0.15, linestyle=":")
    ax.text(
        0.02,
        0.02,
        (
            f"Strehl={metrics['strehl_ratio']:.4f}\n"
            f"WRMS={metrics['wavefront_rms_waves']:.4f}\n"
            f"Residual={metrics['residual_norm']:.4f}"
        ),
        transform=ax.transAxes,
        fontsize=7.5,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="#cccccc"),
    )


def _draw_wavefront(ax: plt.Axes, model: RayOpticsPhysicsEngine, *, title: str, vmax: float) -> None:
    wavefront_waves = _wavefront_waves(model, focus_mm=REFERENCE_FOCUS_MM)
    im = ax.imshow(wavefront_waves, cmap="coolwarm", origin="lower", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def _draw_psf(ax: plt.Axes, psf: np.ndarray, *, title: str) -> None:
    im = ax.imshow(np.asarray(psf, dtype=np.float64), cmap="inferno", origin="lower", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def _draw_normalized_metrics(ax: plt.Axes, summary: dict[str, Any]) -> None:
    state_names = ["nominal", "moved", "repaired"]
    x = np.arange(len(state_names), dtype=np.float64)
    width = 0.36
    residual_vals = [float(summary["normalized_metrics"][state]["w_bessel_residual_vs_moved"]) for state in state_names]
    psf_vals = [float(summary["normalized_metrics"][state]["psf_rmse_to_ideal_vs_moved"]) for state in state_names]

    ax.bar(x - width / 2.0, residual_vals, width=width, color="#4c78a8", label="W-Bessel residual")
    ax.bar(x + width / 2.0, psf_vals, width=width, color="#f58518", label="PSF RMSE to ideal")
    for idx, value in enumerate(residual_vals):
        ax.text(idx - width / 2.0, value, f"{value:.3g}", fontsize=8, ha="center", va="bottom")
    for idx, value in enumerate(psf_vals):
        ax.text(idx + width / 2.0, value, f"{value:.3g}", fontsize=8, ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in state_names])
    ax.set_ylabel("Normalized vs moved")
    ax.set_title("Normalized Metrics")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper center", ncol=2, frameon=False)


def _render_lightpath_wavefront_psf(
    nominal_model: RayOpticsPhysicsEngine,
    moved_model: RayOpticsPhysicsEngine,
    repaired_model: RayOpticsPhysicsEngine,
    summary: dict[str, Any],
) -> None:
    shared_limits = _shared_lightpath_limits([nominal_model, moved_model, repaired_model])
    shared_vmax = _shared_wavefront_vmax([nominal_model, moved_model, repaired_model])
    fig, axes = plt.subplots(4, 3, figsize=(16, 15), gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.75]})
    states = [
        ("Nominal", nominal_model, summary["nominal"]),
        ("Moved", moved_model, summary["moved"]),
        ("Repaired", repaired_model, summary["repaired"]),
    ]
    for col, (name, model, metrics) in enumerate(states):
        _draw_optical_panel(axes[0, col], model, state_name=name, metrics=metrics, limits=shared_limits)
        _draw_wavefront(axes[1, col], model, title=f"{name} Wavefront (waves)", vmax=shared_vmax)
        _draw_psf(axes[2, col], summary["psfs"][name.lower()], title=f"{name} PSF (normalized)")
    _draw_normalized_metrics(axes[3, 0], summary)
    axes[3, 1].axis("off")
    axes[3, 2].axis("off")
    fig.suptitle(
        "1 mm moved case | 920 nm | object_space=-45 mm | real SHWFS repair",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(OUTPUT_LIGHTPATH_PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_metrics(summary: dict[str, Any]) -> None:
    state_names = ["nominal", "moved", "repaired"]
    x = np.arange(len(state_names), dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        ("Strehl Ratio", [float(summary[state]["strehl_ratio"]) for state in state_names], "#4c78a8", False),
        (
            "W-Bessel Residual / Moved",
            [float(summary["normalized_metrics"][state]["w_bessel_residual_vs_moved"]) for state in state_names],
            "#4c78a8",
            True,
        ),
        (
            "PSF RMSE / Moved",
            [float(summary["normalized_metrics"][state]["psf_rmse_to_ideal_vs_moved"]) for state in state_names],
            "#f58518",
            True,
        ),
        ("Wavefront RMS (waves)", [float(summary[state]["wavefront_rms_waves"]) for state in state_names], "#54a24b", False),
    ]
    for ax, (title, values, color, is_bar) in zip(axes.flat, panels):
        if is_bar:
            ax.bar(x, values, color=color, width=0.55)
        else:
            ax.plot(x, values, marker="o", linewidth=2.0, color=color)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.4g}", fontsize=8, ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([name.capitalize() for name in state_names])
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Key Metrics")
    fig.tight_layout()
    fig.savefig(OUTPUT_METRICS_PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _render_fov_rms(rows: list[dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for state_name, color in (("nominal", "#4c78a8"), ("moved", "#f58518"), ("repaired", "#54a24b")):
        subset = [row for row in rows if row["state"] == state_name]
        angles = [row["field_angle_deg"] for row in subset]
        wrms = [row["wavefront_rms_waves"] for row in subset]
        low = [row["wavefront_rms_low_order_removed_waves"] for row in subset]
        axes[0].plot(angles, wrms, marker="o", label=state_name.capitalize(), color=color)
        axes[1].plot(angles, low, marker="o", label=state_name.capitalize(), color=color)
    axes[0].set_title("FOV 0-16 deg RMS")
    axes[1].set_title("FOV 0-16 deg Low-Order-Removed RMS")
    for ax in axes:
        ax.set_xlabel("Field angle (deg)")
        ax.grid(True, alpha=0.2)
        ax.legend()
    axes[0].set_ylabel("Waves")
    axes[1].set_ylabel("Waves")
    fig.tight_layout()
    fig.savefig(OUTPUT_FOV_PNG_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(summary: dict[str, Any]) -> None:
    rows = []
    for state_name in ("nominal", "moved", "repaired"):
        metrics = summary[state_name]
        normalized = summary["normalized_metrics"][state_name]
        rows.append(
            {
                "state": state_name,
                "strehl_ratio": metrics["strehl_ratio"],
                "wavefront_rms_waves": metrics["wavefront_rms_waves"],
                "wavefront_rms_low_order_removed_waves": metrics["wavefront_rms_low_order_removed_waves"],
                "residual_norm": metrics["residual_norm"],
                "true_residual_norm": metrics["true_residual_norm"],
                "rmse_to_ideal_perfect": metrics["rmse_to_ideal_perfect"],
                "psf_rmse_to_ideal_vs_moved": normalized["psf_rmse_to_ideal_vs_moved"],
                "w_bessel_residual_vs_moved": normalized["w_bessel_residual_vs_moved"],
                "runtime_s": summary["runtime_s"] if state_name == "repaired" else "",
                "iteration_count": summary["iteration_count"] if state_name == "repaired" else "",
            }
        )
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_fov_csv(rows: list[dict[str, float]]) -> None:
    with OUTPUT_FOV_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_npz(summary: dict[str, Any]) -> None:
    np.savez_compressed(
        OUTPUT_NPZ_PATH,
        nominal=np.asarray(summary["psfs"]["nominal"], dtype=np.float32),
        moved=np.asarray(summary["psfs"]["moved"], dtype=np.float32),
        repaired=np.asarray(summary["psfs"]["repaired"], dtype=np.float32),
    )


def main() -> None:
    OUTPUT_LOG_PATH.write_text("", encoding="utf-8")
    started_at = time.perf_counter()
    _log("Starting benchmark_1mm_real_shwfs_converging_920nm")
    _log(f"Using wavelength_nm={WAVELENGTH_NM}, object_space_mm={OBJECT_SPACE_MM}, reference_focus_mm={REFERENCE_FOCUS_MM}")
    _log(f"Fast model pupil/fft samples = {FAST_PUPIL_SAMPLES}/{FAST_FFT_SAMPLES}")

    nominal_model = _build_model(pupil_samples=FAST_PUPIL_SAMPLES, fft_samples=FAST_FFT_SAMPLES)
    actuator_ids = nominal_model.get_group_anchor_surface_ids(include_coverglass=True, include_sample_media=False)
    moved_state, geometry_payload = _load_moved_state(INPUT_GEOMETRY_JSON, actuator_ids)
    _log(f"Loaded moved state from {INPUT_GEOMETRY_JSON.name}; actuator ids = {actuator_ids}")

    stage_started = time.perf_counter()
    shwfs = real_shwfs._build_shwfs_measurement_model(nominal_model, focus_shift_mm=REFERENCE_FOCUS_MM)
    ideal_psf = refined_nominal._ideal_diffraction_limited_psf(nominal_model, focus=REFERENCE_FOCUS_MM)
    nominal_psf = nominal_model.get_psf_image(focus=REFERENCE_FOCUS_MM)
    true_reference_coeffs = refined_nominal._project_w_bessel_coeffs_from_opd(
        nominal_model,
        nominal_model.get_wavefront_opd(focus=REFERENCE_FOCUS_MM),
        beam_fill_ratio=BEAM_FILL_RATIO,
        num_modes=NUM_MODES,
    )
    nominal_metrics = _state_metrics(
        nominal_model,
        shwfs,
        nominal_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=true_reference_coeffs,
    )
    _log(f"Built nominal SHWFS model in {time.perf_counter() - stage_started:.2f}s")
    _log(
        "Nominal metrics: "
        f"strehl={nominal_metrics['strehl_ratio']:.6f}, "
        f"wrms={nominal_metrics['wavefront_rms_waves']:.6f}, "
        f"residual={nominal_metrics['residual_norm']:.6f}"
    )

    moved_model = _build_model(pupil_samples=FAST_PUPIL_SAMPLES, fft_samples=FAST_FFT_SAMPLES)
    moved_model.set_surface_perturbations(moved_state)
    moved_psf = moved_model.get_psf_image(focus=REFERENCE_FOCUS_MM)
    moved_metrics = _state_metrics(
        moved_model,
        shwfs,
        moved_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=true_reference_coeffs,
    )
    moved_mech = _mechanical_summary(moved_model)
    _log(
        "Moved metrics: "
        f"strehl={moved_metrics['strehl_ratio']:.6f}, "
        f"wrms={moved_metrics['wavefront_rms_waves']:.6f}, "
        f"residual={moved_metrics['residual_norm']:.6f}, "
        f"min_gap={moved_mech['minimum_gap_mm']:.6f} mm"
    )

    base_state = _anchor_state_subset(moved_model.get_surface_perturbations(), actuator_ids)
    hold_seed = np.zeros(3 * len(actuator_ids), dtype=np.float64)
    bounds = refined_nominal._delta_bounds(
        moved_model,
        actuator_ids,
        base_state,
        local_radius_mm=LOCAL_RADIUS_MM,
    )
    nominal_seed = _nominal_reset_seed(actuator_ids, base_state, bounds)
    timing_totals = {"set_state_s": 0.0, "psf_s": 0.0, "shwfs_s": 0.0, "constraint_s": 0.0}
    tracker = Tracker(best_delta_mm=hold_seed.copy(), best_any_delta_mm=hold_seed.copy())

    def objective(delta_vector: np.ndarray) -> float:
        eval_started = time.perf_counter()
        delta_vector = np.asarray(delta_vector, dtype=np.float64).reshape(-1)
        candidate_state = refined_nominal._delta_state(actuator_ids, base_state, delta_vector)

        t0 = time.perf_counter()
        moved_model.set_surface_perturbations(candidate_state)
        applied_state = _anchor_state_subset(moved_model.get_surface_perturbations(), actuator_ids)
        timing_totals["set_state_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        violation = _constraint_violation(candidate_state, applied_state, actuator_ids)
        timing_totals["constraint_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        psf_image = moved_model.get_psf_image(focus=REFERENCE_FOCUS_MM)
        rmse = float(
            np.sqrt(
                np.mean(
                    (np.asarray(psf_image, dtype=np.float64) - np.asarray(ideal_psf, dtype=np.float64)) ** 2,
                    dtype=np.float64,
                )
            )
        )
        sharpness = refined_nominal._sharpness(psf_image)
        timing_totals["psf_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        current_coeffs = shwfs.estimate_coeffs(moved_model)
        residual = current_coeffs - shwfs.reference_coeffs
        residual_norm = float(np.linalg.norm(residual))
        timing_totals["shwfs_s"] += time.perf_counter() - t0

        cost = residual_norm + CONSTRAINT_PENALTY_WEIGHT * float(violation["l2_mm"])
        tracker.evaluation_count += 1
        if float(violation["max_abs_mm"]) <= CONSTRAINT_TOL_MM and residual_norm < tracker.best_cost:
            tracker.best_cost = residual_norm
            tracker.best_rmse = rmse
            tracker.best_sharpness = sharpness
            tracker.best_delta_mm = delta_vector.copy()
            tracker.feasible_count += 1
            tracker.last_best_eval = tracker.evaluation_count
        if cost < tracker.best_any_cost:
            tracker.best_any_cost = cost
            tracker.best_any_rmse = rmse
            tracker.best_any_sharpness = sharpness
            tracker.best_any_delta_mm = delta_vector.copy()

        if tracker.evaluation_count == 1 or tracker.evaluation_count % 5 == 0:
            _log(
                "eval="
                f"{tracker.evaluation_count} residual={residual_norm:.6f} "
                f"norm_resid={residual_norm / max(moved_metrics['residual_norm'], 1.0e-12):.6f} "
                f"best={tracker.best_cost:.6f} "
                f"best_norm={tracker.best_cost / max(moved_metrics['residual_norm'], 1.0e-12):.6f} "
                f"rmse={rmse:.6f} clamp_l2={violation['l2_mm']:.3e} "
                f"elapsed={time.perf_counter() - eval_started:.2f}s"
            )
        if (
            tracker.feasible_count > 0
            and tracker.evaluation_count - tracker.last_best_eval >= BEST_PATIENCE_EVALS
        ):
            raise EarlyStopOptimization(
                f"No best-cost improvement for {BEST_PATIENCE_EVALS} evaluations."
            )
        return cost

    stage_started = time.perf_counter()
    _log("Starting optimization")
    _log(
        "Search configuration: "
        f"local_radius_mm={'full_mechanical' if LOCAL_RADIUS_MM is None else f'{LOCAL_RADIUS_MM:.6f}'} "
        f"finite_diff_step_mm={OPT_DIFF_STEP_MM:.6f}"
    )
    _log(
        "Seed norms: "
        f"hold={float(np.linalg.norm(hold_seed)):.6f} "
        f"nominal_reset={float(np.linalg.norm(nominal_seed)):.6f}"
    )
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        seed_candidates: list[tuple[str, np.ndarray]] = [("hold", hold_seed)]
        if not np.allclose(nominal_seed, hold_seed, atol=1.0e-12, rtol=0.0):
            seed_candidates.append(("nominal_reset", nominal_seed))
        chosen_seed_name = "hold"
        chosen_seed = hold_seed.copy()
        chosen_cost = np.inf
        for seed_name, seed_vector in seed_candidates:
            seed_cost = float(objective(seed_vector))
            _log(f"seed={seed_name} objective={seed_cost:.6f}")
            if seed_cost < chosen_cost:
                chosen_seed_name = seed_name
                chosen_seed = seed_vector.copy()
                chosen_cost = seed_cost
        _log(f"Selected initial seed: {chosen_seed_name}")
        try:
            minimize(
                objective,
                x0=chosen_seed,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "maxiter": OPT_MAXITER,
                    "maxfun": OPT_MAXFUN,
                    "ftol": 1.0e-6,
                    "eps": OPT_DIFF_STEP_MM,
                },
            )
        except EarlyStopOptimization as exc:
            _log(f"Early stop: {exc}")
        optimization_warnings = [str(record.message) for record in warning_records]
    optimization_runtime_s = time.perf_counter() - stage_started
    _log(f"Optimization finished in {optimization_runtime_s:.2f}s with {tracker.evaluation_count} evaluations")
    _log(
        "Timing totals: "
        f"set_state={timing_totals['set_state_s']:.2f}s, "
        f"constraint={timing_totals['constraint_s']:.2f}s, "
        f"psf={timing_totals['psf_s']:.2f}s, "
        f"shwfs={timing_totals['shwfs_s']:.2f}s"
    )

    best_delta = tracker.best_delta_mm.copy()
    if not np.isfinite(tracker.best_cost):
        best_delta = tracker.best_any_delta_mm.copy()
        _log("No strictly feasible best state found; falling back to best_any_delta")

    repaired_model = _build_model(pupil_samples=FAST_PUPIL_SAMPLES, fft_samples=FAST_FFT_SAMPLES)
    repaired_model.set_surface_perturbations(base_state)
    repaired_state = refined_nominal._delta_state(actuator_ids, base_state, best_delta)
    repaired_model.set_surface_perturbations(repaired_state)
    repaired_applied_state = _anchor_state_subset(repaired_model.get_surface_perturbations(), actuator_ids)
    repaired_violation = _constraint_violation(repaired_state, repaired_applied_state, actuator_ids)
    repaired_psf = repaired_model.get_psf_image(focus=REFERENCE_FOCUS_MM)
    repaired_metrics = _state_metrics(
        repaired_model,
        shwfs,
        repaired_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        true_reference_coeffs=true_reference_coeffs,
    )
    repaired_mech = _mechanical_summary(repaired_model)
    _log(
        "Repaired metrics: "
        f"strehl={repaired_metrics['strehl_ratio']:.6f}, "
        f"wrms={repaired_metrics['wavefront_rms_waves']:.6f}, "
        f"residual={repaired_metrics['residual_norm']:.6f}, "
        f"clamp_max={repaired_violation['max_abs_mm']:.3e} mm, "
        f"min_gap={repaired_mech['minimum_gap_mm']:.6f} mm"
    )

    fov_rows: list[dict[str, float]] = []
    _log("Evaluating FOV RMS curves")
    fov_rows.extend(_evaluate_fov_curve({}, label="nominal"))
    fov_rows.extend(_evaluate_fov_curve(base_state, label="moved"))
    fov_rows.extend(_evaluate_fov_curve(repaired_applied_state, label="repaired"))

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
        "algorithm": "1mm_real_shwfs_converging_920nm_fast",
        "configuration": {
            "wavelength_nm": float(WAVELENGTH_NM),
            "object_space_mm": float(OBJECT_SPACE_MM),
            "reference_focus_mm": float(REFERENCE_FOCUS_MM),
            "fast_pupil_samples": int(FAST_PUPIL_SAMPLES),
            "fast_fft_samples": int(FAST_FFT_SAMPLES),
            "measurement_model": "simulated_real_shwfs_lenslet_slopes",
            "w_bessel_source": "estimated_from_simulated_shwfs_not_direct_truth",
            "delta_only": True,
            "local_radius_mm": None if LOCAL_RADIUS_MM is None else float(LOCAL_RADIUS_MM),
            "optimizer": {
                "method": "L-BFGS-B",
                "maxiter": int(OPT_MAXITER),
                "maxfun": int(OPT_MAXFUN),
                "finite_diff_step_mm": float(OPT_DIFF_STEP_MM),
            },
            "new_display_geometry_version": True,
        },
        "case_metadata": {
            "effective_level_mm": float(geometry_payload.get("effective_level_mm", 1.0)),
            "scale_factor": float(geometry_payload.get("scale_factor", np.nan)),
            "source_case": geometry_payload.get("source_case"),
            "warning_count": int(geometry_payload.get("warning_count", 0)),
            "warnings": list(geometry_payload.get("warnings", [])),
        },
        "nominal": nominal_metrics,
        "moved": moved_metrics,
        "repaired": repaired_metrics,
        "normalized_metrics": normalized_metrics,
        "moved_mechanical_summary": moved_mech,
        "repaired_mechanical_summary": repaired_mech,
        "repaired_constraint_violation_mm": repaired_violation,
        "runtime_s": float(time.perf_counter() - started_at),
        "optimization_runtime_s": float(optimization_runtime_s),
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
        "moved_position_by_anchor": {
            str(surface_id): {
                "dx_mm": float(base_state.get(surface_id, SurfacePerturbation()).dx_mm),
                "dy_mm": float(base_state.get(surface_id, SurfacePerturbation()).dy_mm),
                "dz_mm": float(base_state.get(surface_id, SurfacePerturbation()).dz_mm),
            }
            for surface_id in actuator_ids
        },
        "final_position_by_anchor": {
            str(surface_id): {
                "dx_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dx_mm),
                "dy_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dy_mm),
                "dz_mm": float(repaired_applied_state.get(surface_id, SurfacePerturbation()).dz_mm),
            }
            for surface_id in actuator_ids
        },
        "resource_breakdown_s": timing_totals,
        "warnings": optimization_warnings,
        "psfs": {
            "nominal": np.asarray(nominal_psf, dtype=np.float32),
            "moved": np.asarray(moved_psf, dtype=np.float32),
            "repaired": np.asarray(repaired_psf, dtype=np.float32),
        },
        "fov_rms_rows": fov_rows,
    }

    _write_csv(summary)
    _write_fov_csv(fov_rows)
    _write_npz(summary)

    json_ready = copy.deepcopy(summary)
    json_ready.pop("psfs", None)
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump(json_ready, handle, indent=2)

    moved_model.set_surface_perturbations(base_state)
    _render_lightpath_wavefront_psf(nominal_model, moved_model, repaired_model, summary)
    _render_metrics(summary)
    _render_fov_rms(fov_rows)

    _log(f"Wrote {OUTPUT_JSON_PATH.name}")
    _log(f"Wrote {OUTPUT_CSV_PATH.name}")
    _log(f"Wrote {OUTPUT_FOV_CSV_PATH.name}")
    _log(f"Wrote {OUTPUT_NPZ_PATH.name}")
    _log(f"Wrote {OUTPUT_LIGHTPATH_PNG_PATH.name}")
    _log(f"Wrote {OUTPUT_METRICS_PNG_PATH.name}")
    _log(f"Wrote {OUTPUT_FOV_PNG_PATH.name}")


if __name__ == "__main__":
    main()
