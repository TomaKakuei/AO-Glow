from __future__ import annotations

import csv
import json
import math
import os
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PYTHON_EXE = Path(sys.executable)
OUTPUT_DIR = ROOT / "artifacts" / "v10_2_quantized12_mode5cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACTUATOR_IDS = [68, 70, 72, 74, 77, 79, 81, 84]
PROFILE = "realistic"
RNG_SEED = 20260504


def _state_from_positions(position_by_anchor: dict[str, dict[str, float]]):
    from optical_model_rayoptics import SurfacePerturbation

    state: dict[int, SurfacePerturbation] = {}
    for anchor_surface_id in ACTUATOR_IDS:
        row = position_by_anchor.get(str(anchor_surface_id), {})
        state[anchor_surface_id] = SurfacePerturbation(
            dx_mm=float(row.get("dx_mm", 0.0)),
            dy_mm=float(row.get("dy_mm", 0.0)),
            dz_mm=float(row.get("dz_mm", 0.0)),
            tilt_x_deg=float(row.get("tilt_x_deg", 0.0)),
            tilt_y_deg=float(row.get("tilt_y_deg", 0.0)),
        )
    return state


def _case_specs() -> list[dict[str, object]]:
    rng = np.random.default_rng(RNG_SEED)
    cases: list[dict[str, object]] = []
    for case_index in range(1, 6):
        overall_variation_mm = float(rng.uniform(0.05, 0.15))
        tt_arcmin = float(rng.uniform(1.0, 5.0))
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        tilt_x_deg = float((tt_arcmin * math.cos(theta)) / 60.0)
        tilt_y_deg = float((tt_arcmin * math.sin(theta)) / 60.0)
        cases.append(
            {
                "case_index": case_index,
                "overall_variation_mm": overall_variation_mm,
                "tt_arcmin": tt_arcmin,
                "tt_theta_deg": float(math.degrees(theta)),
                "tilt_x_deg": tilt_x_deg,
                "tilt_y_deg": tilt_y_deg,
                "output_name": f"v10_2_{PROFILE}_quant12_case{case_index:02d}",
            }
        )
    return cases


def _run_case(case_spec: dict[str, object]) -> dict[str, object]:
    started = time.perf_counter()
    os.environ["AO_V3_ENABLE_TIPTILT"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"

    from ao_v10_2_backend import RunConfig, run_repair

    cfg = RunConfig(
        overall_variation_mm=float(case_spec["overall_variation_mm"]),
        per_anchor_overrides={
            surface_id: {
                "tilt_x_deg": float(case_spec["tilt_x_deg"]),
                "tilt_y_deg": float(case_spec["tilt_y_deg"]),
            }
            for surface_id in ACTUATOR_IDS
        },
        output_name=str(case_spec["output_name"]),
        shwfs_noise_profile=PROFILE,
    )
    result = run_repair(cfg, log=lambda message: None)
    elapsed_s = float(time.perf_counter() - started)
    summary_path = Path(str(result.get("repair_summary_path", "")))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "case_index": int(case_spec["case_index"]),
        "overall_variation_mm": float(case_spec["overall_variation_mm"]),
        "tt_arcmin": float(case_spec["tt_arcmin"]),
        "tt_theta_deg": float(case_spec["tt_theta_deg"]),
        "tilt_x_deg": float(case_spec["tilt_x_deg"]),
        "tilt_y_deg": float(case_spec["tilt_y_deg"]),
        "elapsed_s": elapsed_s,
        "run_dir": str(result.get("run_dir", "")),
        "repair_summary_path": str(summary_path),
        "summary": summary,
    }


def _collect_coeff_rows(case_result: dict[str, object]) -> list[dict[str, object]]:
    from ao_v10_2_backend import PROFILE_SENSOR_PRESETS
    import benchmark_freeform_real_shwfs_residual_120 as real
    import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal
    from optical_model_rayoptics import SurfacePerturbation
    from w_bessel_core import nm_sequence

    summary = case_result["summary"]
    assert isinstance(summary, dict)

    optical_model = real._build_system()
    nominal_wavefront_metrics = real._wavefront_metrics(optical_model)
    nominal_focus_shift_mm = float(nominal_wavefront_metrics["best_focus_shift_mm"])
    shwfs = real._build_shwfs_measurement_model(optical_model, focus_shift_mm=nominal_focus_shift_mm)
    actuator_ids = real._freeform_actuator_ids(optical_model)
    nominal_true_reference_coeffs = real._project_w_bessel_coeffs_from_opd(
        optical_model,
        optical_model.get_wavefront_opd(focus=nominal_focus_shift_mm),
        beam_fill_ratio=real.BEAM_FILL_RATIO,
        num_modes=real.NUM_MODES,
    )
    reference_coeffs = np.asarray(shwfs.reference_coeffs, dtype=np.float64).reshape(-1)
    mode_pairs = nm_sequence(int(real.NUM_MODES))

    def _state_coeffs(position_by_anchor: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
        state_map = _state_from_positions(position_by_anchor)
        optical_model.set_surface_perturbations(state_map)
        estimated_coeffs = np.asarray(shwfs.estimate_coeffs(optical_model), dtype=np.float64).reshape(-1)
        true_coeffs = np.asarray(
            refined_nominal._project_w_bessel_coeffs_from_opd(
                optical_model,
                optical_model.get_wavefront_opd(focus=nominal_focus_shift_mm),
                beam_fill_ratio=real.BEAM_FILL_RATIO,
                num_modes=real.NUM_MODES,
            ),
            dtype=np.float64,
        ).reshape(-1)
        return {
            "estimated_coeffs": estimated_coeffs,
            "true_coeffs": true_coeffs,
        }

    nominal_coeffs = _state_coeffs(
        summary["moved_position_by_anchor"] if False else {
            str(surface_id): {
                "dx_mm": 0.0,
                "dy_mm": 0.0,
                "dz_mm": 0.0,
                "tilt_x_deg": 0.0,
                "tilt_y_deg": 0.0,
            }
            for surface_id in actuator_ids
        }
    )
    moved_coeffs = _state_coeffs(summary["moved_position_by_anchor"])
    repaired_coeffs = _state_coeffs(summary["final_position_by_anchor"])

    rows: list[dict[str, object]] = []
    for mode_index, (n, m) in enumerate(mode_pairs, start=1):
        nominal_true = float(nominal_true_reference_coeffs[mode_index - 1])
        rows.append(
            {
                "case_index": int(case_result["case_index"]),
                "profile": PROFILE,
                "num_modes": int(real.NUM_MODES),
                "mode_index": mode_index,
                "n": int(n),
                "m": int(m),
                "moved_estimated_coeff": float(moved_coeffs["estimated_coeffs"][mode_index - 1]),
                "repaired_estimated_coeff": float(repaired_coeffs["estimated_coeffs"][mode_index - 1]),
                "reference_coeff": float(reference_coeffs[mode_index - 1]),
                "nominal_true_coeff": float(nominal_true),
                "moved_true_coeff": float(moved_coeffs["true_coeffs"][mode_index - 1]),
                "repaired_true_coeff": float(repaired_coeffs["true_coeffs"][mode_index - 1]),
                "moved_estimated_residual": float(moved_coeffs["estimated_coeffs"][mode_index - 1] - reference_coeffs[mode_index - 1]),
                "repaired_estimated_residual": float(
                    repaired_coeffs["estimated_coeffs"][mode_index - 1] - reference_coeffs[mode_index - 1]
                ),
                "moved_true_residual": float(moved_coeffs["true_coeffs"][mode_index - 1] - nominal_true),
                "repaired_true_residual": float(repaired_coeffs["true_coeffs"][mode_index - 1] - nominal_true),
            }
        )
    return rows


def main() -> int:
    case_specs = _case_specs()
    results: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for case_spec in case_specs:
        print(
            f"Running case {int(case_spec['case_index'])}: "
            f"overall={float(case_spec['overall_variation_mm']):.4f} mm, "
            f"tt={float(case_spec['tt_arcmin']):.3f} arcmin",
            flush=True,
        )
        case_result = _run_case(case_spec)
        results.append(
            {
                "case_index": int(case_result["case_index"]),
                "overall_variation_mm": float(case_result["overall_variation_mm"]),
                "tt_arcmin": float(case_result["tt_arcmin"]),
                "tt_theta_deg": float(case_result["tt_theta_deg"]),
                "tilt_x_deg": float(case_result["tilt_x_deg"]),
                "tilt_y_deg": float(case_result["tilt_y_deg"]),
                "elapsed_s": float(case_result["elapsed_s"]),
                "repaired_residual": float(case_result["summary"]["repaired"]["residual_norm"]),
                "repaired_true_residual": float(case_result["summary"]["repaired"]["true_residual_norm"]),
                "repaired_wrms": float(case_result["summary"]["repaired"]["wavefront_rms_waves"]),
                "repaired_strehl": float(case_result["summary"]["repaired"]["strehl_ratio"]),
                "nominal_wrms": float(case_result["summary"]["nominal"]["wavefront_rms_waves"]),
                "slope_channels": int(case_result["summary"]["configuration"]["shwfs_slope_channels"]),
                "lenslet_count": int(case_result["summary"]["configuration"]["shwfs_lenslet_count"]),
                "run_dir": str(case_result["run_dir"]),
                "repair_summary_path": str(case_result["repair_summary_path"]),
            }
        )
        long_rows.extend(_collect_coeff_rows(case_result))
        (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        with (OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        with (OUTPUT_DIR / "mode_components_long.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_rows[0].keys()))
            writer.writeheader()
            writer.writerows(long_rows)

    summary_rows: list[dict[str, object]] = []
    for mode_index in range(1, 13):
        subset = [row for row in long_rows if int(row["mode_index"]) == mode_index]
        summary_rows.append(
            {
                "mode_index": mode_index,
                "n": int(subset[0]["n"]),
                "m": int(subset[0]["m"]),
                "mean_abs_repaired_true_residual": float(np.mean([abs(float(row["repaired_true_residual"])) for row in subset])),
                "rms_repaired_true_residual": float(np.sqrt(np.mean([float(row["repaired_true_residual"]) ** 2 for row in subset]))),
                "mean_abs_repaired_estimated_residual": float(np.mean([abs(float(row["repaired_estimated_residual"])) for row in subset])),
                "rms_repaired_estimated_residual": float(np.sqrt(np.mean([float(row["repaired_estimated_residual"]) ** 2 for row in subset]))),
                "mean_abs_moved_true_residual": float(np.mean([abs(float(row["moved_true_residual"])) for row in subset])),
                "rms_moved_true_residual": float(np.sqrt(np.mean([float(row["moved_true_residual"]) ** 2 for row in subset]))),
            }
        )
    with (OUTPUT_DIR / "mode_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (OUTPUT_DIR / "mode_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
