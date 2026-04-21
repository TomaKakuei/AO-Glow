from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

from optical_model_rayoptics import MechanicalLimitWarning, RayOpticsPhysicsEngine, SurfacePerturbation
from w_bessel_core import generate_w_bessel_basis, project_phase_map


warnings.filterwarnings("ignore", category=MechanicalLimitWarning)

DEFAULT_NUM_MODES = 8
DEFAULT_BEAM_FILL_RATIO = 1.15
DEFAULT_LOCAL_RADIUS_MM = 0.10
DEFAULT_MAXITER = 8
DEFAULT_MAXFUN = 120
DEFAULT_FTOL = 1.0e-6


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


def _build_system() -> tuple[RayOpticsPhysicsEngine, list[int]]:
    optical_model = RayOpticsPhysicsEngine()
    actuator_ids = optical_model.get_group_anchor_surface_ids(
        include_coverglass=True,
        include_sample_media=False,
    )
    return optical_model, actuator_ids


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


def _wavefront_metrics(optical_model: RayOpticsPhysicsEngine) -> dict[str, float]:
    pupil_samples = int(optical_model.pupil_samples)
    fft_samples = int(optical_model.fft_samples)
    wavelength_nm = float(optical_model.wavelength_nm)
    wavelength_sys_units = float(optical_model.opm.nm_to_sys_units(wavelength_nm))

    pupil_x, pupil_y, opd_sys_units, valid_mask = optical_model._sample_wavefront(
        num_rays=pupil_samples,
        field_index=0,
        wavelength_nm=wavelength_nm,
        focus=0.0,
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
        "wavefront_rms_sys_units": float(rms_sys_units),
        "wavefront_rms_waves": float(rms_waves),
        "wavefront_rms_nm": float(rms_nm),
        "wavefront_rms_tilt_removed_sys_units": float(high_order_rms_sys_units),
        "wavefront_rms_tilt_removed_waves": float(high_order_rms_waves),
        "wavefront_rms_tilt_removed_nm": float(high_order_rms_nm),
        "strehl_ratio": float(strehl_ratio),
        "pupil_valid_fraction": float(np.mean(valid_mask.astype(np.float64))),
    }


def _sharpness(psf: np.ndarray) -> float:
    psf64 = np.asarray(psf, dtype=np.float64)
    return float(np.sum(psf64 * psf64, dtype=np.float64))


def _extract_w_bessel_coeffs(
    optical_model: RayOpticsPhysicsEngine,
    *,
    beam_fill_ratio: float,
    num_modes: int,
) -> np.ndarray:
    opd_map = np.asarray(optical_model.get_wavefront_opd(), dtype=np.float64)
    wavelength_nm = float(optical_model.wavelength_nm)
    wavelength_sys_units = float(optical_model.opm.nm_to_sys_units(wavelength_nm))
    wavefront_in_waves = opd_map / max(wavelength_sys_units, 1.0e-15)
    basis_result = generate_w_bessel_basis(
        grid_size=opd_map.shape[0],
        beam_fill_ratio=float(beam_fill_ratio),
        num_modes=int(num_modes),
    )
    coeffs = project_phase_map(
        wavefront_in_waves,
        basis_result.basis_cube,
        basis_result.W_xy,
        basis_result.pupil_mask,
    )
    return np.asarray(coeffs, dtype=np.float64).reshape(-1)


def _residual_norm(
    optical_model: RayOpticsPhysicsEngine,
    reference_coeffs: np.ndarray,
    *,
    beam_fill_ratio: float,
    num_modes: int,
) -> float:
    coeffs = _extract_w_bessel_coeffs(
        optical_model,
        beam_fill_ratio=beam_fill_ratio,
        num_modes=num_modes,
    )
    reference = np.asarray(reference_coeffs, dtype=np.float64).reshape(-1)
    mode_count = min(coeffs.size, reference.size)
    if mode_count <= 0:
        raise ValueError("Residual comparison requires at least one mode.")
    residual = coeffs[:mode_count] - reference[:mode_count]
    return float(np.linalg.norm(residual))


def _state_metrics(
    optical_model: RayOpticsPhysicsEngine,
    psf: np.ndarray,
    *,
    ideal_psf: np.ndarray,
    nominal_psf: np.ndarray,
    reference_coeffs: np.ndarray,
    beam_fill_ratio: float,
    num_modes: int,
) -> dict[str, float]:
    wavefront_metrics = _wavefront_metrics(optical_model)
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
        "residual_norm": _residual_norm(
            optical_model,
            reference_coeffs,
            beam_fill_ratio=beam_fill_ratio,
            num_modes=num_modes,
        ),
        **wavefront_metrics,
    }


def _normalize_perturbations(payload: dict[str, Any]) -> list[dict[str, float]]:
    items = payload.get("perturbations", [])
    normalized: list[dict[str, float]] = []
    for item in items:
        surface_id = int(item.get("surface_index", item.get("anchor_surface_id")))
        dx_mm = float(item.get("dx_mm", item.get("perturbation_dx_mm", 0.0)))
        dy_mm = float(item.get("dy_mm", item.get("perturbation_dy_mm", 0.0)))
        dz_mm = float(item.get("dz_mm", item.get("perturbation_dz_mm", 0.0)))
        normalized.append(
            {
                "anchor_surface_id": surface_id,
                "perturbation_dx_mm": dx_mm,
                "perturbation_dy_mm": dy_mm,
                "perturbation_dz_mm": dz_mm,
            }
        )
    return normalized


def _apply_perturbations(optical_model: RayOpticsPhysicsEngine, perturbations: list[dict[str, float]]) -> None:
    for item in perturbations:
        optical_model.perturb_lens(
            int(item["anchor_surface_id"]),
            float(item["perturbation_dx_mm"]),
            float(item["perturbation_dy_mm"]),
            float(item["perturbation_dz_mm"]),
        )


def _delta_state(
    actuator_ids: list[int],
    base_state: dict[int, SurfacePerturbation],
    delta_vector: np.ndarray,
) -> dict[int, SurfacePerturbation]:
    deltas = np.asarray(delta_vector, dtype=np.float64).reshape(len(actuator_ids), 3)
    candidate = copy.deepcopy(base_state)
    for idx, surface_id in enumerate(actuator_ids):
        base = candidate.get(surface_id, SurfacePerturbation())
        candidate[surface_id] = SurfacePerturbation(
            dx_mm=float(base.dx_mm + deltas[idx, 0]),
            dy_mm=float(base.dy_mm + deltas[idx, 1]),
            dz_mm=float(base.dz_mm + deltas[idx, 2]),
            tilt_x_deg=float(base.tilt_x_deg),
            tilt_y_deg=float(base.tilt_y_deg),
        )
    return candidate


def _apply_actuator_state(
    optical_model: RayOpticsPhysicsEngine,
    actuator_ids: list[int],
    state: dict[int, SurfacePerturbation],
) -> None:
    for surface_id in actuator_ids:
        perturbation = state.get(surface_id, SurfacePerturbation())
        optical_model.perturb_lens(
            surface_id,
            float(perturbation.dx_mm),
            float(perturbation.dy_mm),
            float(perturbation.dz_mm),
            float(perturbation.tilt_x_deg),
            float(perturbation.tilt_y_deg),
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
        axis_defs = (
            (-envelope.lateral_limit_mm, envelope.lateral_limit_mm, float(base.dx_mm)),
            (-envelope.lateral_limit_mm, envelope.lateral_limit_mm, float(base.dy_mm)),
            (envelope.axial_min_mm, envelope.axial_max_mm, float(base.dz_mm)),
        )
        for lower_abs, upper_abs, base_value in axis_defs:
            lower = max(lower_abs - base_value, -local_radius_mm)
            upper = min(upper_abs - base_value, local_radius_mm)
            if upper < lower:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            bounds.append((float(lower), float(upper)))
    return bounds


def run_chain(
    perturbations: list[dict[str, float]],
    *,
    beam_fill_ratio: float,
    num_modes: int,
    local_radius_mm: float,
    maxiter: int,
    maxfun: int,
    ftol: float,
) -> dict[str, Any]:
    optical_model, actuator_ids = _build_system()

    nominal_psf = optical_model.get_psf_image()
    ideal_psf = _ideal_diffraction_limited_psf(optical_model)
    nominal_reference_coeffs = _extract_w_bessel_coeffs(
        optical_model,
        beam_fill_ratio=beam_fill_ratio,
        num_modes=num_modes,
    )
    nominal_metrics = _state_metrics(
        optical_model,
        nominal_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        reference_coeffs=nominal_reference_coeffs,
        beam_fill_ratio=beam_fill_ratio,
        num_modes=num_modes,
    )

    _apply_perturbations(optical_model, perturbations)

    moved_psf = optical_model.get_psf_image()
    moved_metrics = _state_metrics(
        optical_model,
        moved_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        reference_coeffs=nominal_reference_coeffs,
        beam_fill_ratio=beam_fill_ratio,
        num_modes=num_modes,
    )

    base_state = optical_model.get_surface_perturbations()
    x0 = np.zeros(3 * len(actuator_ids), dtype=np.float64)
    bounds = _delta_bounds(
        optical_model,
        actuator_ids,
        base_state,
        local_radius_mm=local_radius_mm,
    )
    tracker = Tracker(best_delta_mm=x0.copy(), best_any_delta_mm=x0.copy())

    def objective(delta_vector: np.ndarray) -> float:
        delta_vector = np.asarray(delta_vector, dtype=np.float64).reshape(-1)
        candidate_state = _delta_state(actuator_ids, base_state, delta_vector)
        _apply_actuator_state(optical_model, actuator_ids, candidate_state)
        psf_image = optical_model.get_psf_image()
        current_coeffs = _extract_w_bessel_coeffs(
            optical_model,
            beam_fill_ratio=beam_fill_ratio,
            num_modes=num_modes,
        )
        mode_count = min(current_coeffs.size, nominal_reference_coeffs.size)
        residual = current_coeffs[:mode_count] - nominal_reference_coeffs[:mode_count]
        residual_norm = float(np.linalg.norm(residual))
        sharpness = _sharpness(psf_image)
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
        if residual_norm < tracker.best_any_cost:
            tracker.best_any_cost = residual_norm
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
        return residual_norm

    started_at = time.perf_counter()
    warnings_list: list[str] = []
    skip_optimization = int(maxiter) <= 0 or int(maxfun) <= 0
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        if not skip_optimization:
            objective(x0)
            minimize(
                objective,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": int(maxiter), "maxfun": int(maxfun), "ftol": float(ftol)},
            )
        warnings_list = [str(record.message) for record in warning_records]
    runtime_s = time.perf_counter() - started_at

    best_delta = tracker.best_delta_mm.copy()
    if skip_optimization:
        best_delta = np.zeros_like(x0)
    elif not np.isfinite(tracker.best_cost):
        best_delta = tracker.best_any_delta_mm.copy()

    repaired_state = _delta_state(actuator_ids, base_state, best_delta)
    _apply_actuator_state(optical_model, actuator_ids, repaired_state)
    repaired_psf = optical_model.get_psf_image()
    repaired_metrics = _state_metrics(
        optical_model,
        repaired_psf,
        ideal_psf=ideal_psf,
        nominal_psf=nominal_psf,
        reference_coeffs=nominal_reference_coeffs,
        beam_fill_ratio=beam_fill_ratio,
        num_modes=num_modes,
    )

    return {
        "status": "complete",
        "algorithm": "freeform_shwfs_proxy_residual",
        "settings": {
            "actuator_surface_ids": actuator_ids,
            "beam_fill_ratio": float(beam_fill_ratio),
            "num_modes": int(num_modes),
            "local_radius_mm": float(local_radius_mm),
            "optimizer": {"method": "L-BFGS-B", "maxiter": int(maxiter), "maxfun": int(maxfun), "ftol": float(ftol)},
            "skip_optimization": bool(skip_optimization),
            "delta_only": True,
            "measurement_proxy": "w_bessel_core.project_phase_map",
        },
        "nominal": nominal_metrics,
        "moved": moved_metrics,
        "repaired": repaired_metrics,
        "improvements": {
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
            "residual_change_pct": float(
                100.0 * (moved_metrics["residual_norm"] - repaired_metrics["residual_norm"])
                / max(moved_metrics["residual_norm"], 1.0e-12)
            ),
        },
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
        "warnings_count": len(warnings_list),
        "warnings": warnings_list[:20],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shortest non-UI SHWFS-residual recovery chain with a JSON perturbation input."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(__file__).resolve().parent / "displacement_input_example.json",
        help="JSON path containing lens perturbations.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "short_chain_result.json",
        help="Where to write the output summary JSON.",
    )
    parser.add_argument("--num-modes", type=int, default=DEFAULT_NUM_MODES)
    parser.add_argument("--beam-fill-ratio", type=float, default=DEFAULT_BEAM_FILL_RATIO)
    parser.add_argument("--local-radius-mm", type=float, default=DEFAULT_LOCAL_RADIUS_MM)
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--maxfun", type=int, default=DEFAULT_MAXFUN)
    parser.add_argument("--ftol", type=float, default=DEFAULT_FTOL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with args.input_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    perturbations = _normalize_perturbations(payload)
    if not perturbations:
        raise ValueError("Input JSON contains no perturbations.")

    result = run_chain(
        perturbations,
        beam_fill_ratio=float(args.beam_fill_ratio),
        num_modes=int(args.num_modes),
        local_radius_mm=float(args.local_radius_mm),
        maxiter=int(args.maxiter),
        maxfun=int(args.maxfun),
        ftol=float(args.ftol),
    )

    output = {
        "input_json": str(args.input_json),
        "input_perturbation_count": len(perturbations),
        "input_perturbations": perturbations,
        "generated_at_epoch_s": time.time(),
        "result": result,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Wrote {args.output_json}")
    print(
        "RMS waves (nominal/moved/repaired): "
        f"{result['nominal']['wavefront_rms_waves']:.6f} / "
        f"{result['moved']['wavefront_rms_waves']:.6f} / "
        f"{result['repaired']['wavefront_rms_waves']:.6f}"
    )


if __name__ == "__main__":
    main()
