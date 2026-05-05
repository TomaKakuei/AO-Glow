from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ao_v6_backend
import benchmark_1mm_real_shwfs_converging_920nm as bench
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
from optical_model_rayoptics import SurfacePerturbation
from w_bessel_core import reconstruct_phase_map


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
PILOT_DIR = ARTIFACTS_DIR / "cnn_wb12_pilot"
ACTUATOR_IDS = tuple(int(v) for v in ao_v6_backend.ACTUATOR_IDS)
NUM_MODES = 12
LENSLET_COUNT = 13
SLOPE_LIMIT = 256
NOISE_PROFILE = "realistic"
FORWARD_AVERAGES = 1
XYZ_LIMIT_MM = 1.0
TT_LIMIT_ARCMIN = 20.0
Z_PRIOR_LIMIT_MM = 0.15
MECHANICAL_EPS_MM = 1.0e-6

_WORKER_CACHE: dict[str, Any] = {}


def _state_to_serializable(state: dict[int, SurfacePerturbation]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for anchor_id in ACTUATOR_IDS:
        perturb = state.get(int(anchor_id), SurfacePerturbation())
        payload[str(anchor_id)] = {
            "dx_mm": float(perturb.dx_mm),
            "dy_mm": float(perturb.dy_mm),
            "dz_mm": float(perturb.dz_mm),
            "tilt_x_deg": float(perturb.tilt_x_deg),
            "tilt_y_deg": float(perturb.tilt_y_deg),
        }
    return payload


def _serializable_to_state(payload: dict[str, dict[str, float]]) -> dict[int, SurfacePerturbation]:
    state: dict[int, SurfacePerturbation] = {}
    for anchor_id in ACTUATOR_IDS:
        row = payload[str(anchor_id)]
        state[int(anchor_id)] = SurfacePerturbation(
            dx_mm=float(row["dx_mm"]),
            dy_mm=float(row["dy_mm"]),
            dz_mm=float(row["dz_mm"]),
            tilt_x_deg=float(row["tilt_x_deg"]),
            tilt_y_deg=float(row["tilt_y_deg"]),
        )
    return state


def _mech_summary_ok(mech: dict[str, Any]) -> bool:
    if not bool(mech.get("all_position_limits_satisfied", False)):
        return False
    gaps = mech.get("group_gap_report", [])
    for row in gaps:
        if bool(row.get("overlap", False)):
            return False
        if float(row.get("actual_gap_mm", 0.0)) < 0.0:
            return False
    return True


def _same_state(
    requested: dict[int, SurfacePerturbation],
    applied: dict[int, SurfacePerturbation],
    *,
    atol: float = MECHANICAL_EPS_MM,
) -> bool:
    for anchor_id in ACTUATOR_IDS:
        lhs = requested.get(int(anchor_id), SurfacePerturbation())
        rhs = applied.get(int(anchor_id), SurfacePerturbation())
        values = (
            (float(lhs.dx_mm), float(rhs.dx_mm)),
            (float(lhs.dy_mm), float(rhs.dy_mm)),
            (float(lhs.dz_mm), float(rhs.dz_mm)),
            (float(lhs.tilt_x_deg), float(rhs.tilt_x_deg)),
            (float(lhs.tilt_y_deg), float(rhs.tilt_y_deg)),
        )
        for requested_value, applied_value in values:
            if not math.isclose(requested_value, applied_value, abs_tol=atol, rel_tol=0.0):
                return False
    return True


def _sample_xy_z_vector(rng: np.random.Generator) -> tuple[float, float, float]:
    z_mm = float(rng.uniform(-Z_PRIOR_LIMIT_MM, Z_PRIOR_LIMIT_MM))
    radial_limit_sq = max(float(XYZ_LIMIT_MM * XYZ_LIMIT_MM - z_mm * z_mm), 0.0)
    radial_limit = math.sqrt(radial_limit_sq)
    radial_mm = radial_limit * math.sqrt(float(rng.uniform(0.0, 1.0)))
    angle_rad = float(rng.uniform(0.0, 2.0 * math.pi))
    dx_mm = radial_mm * math.cos(angle_rad)
    dy_mm = radial_mm * math.sin(angle_rad)
    return dx_mm, dy_mm, z_mm


def _sample_tilt_vector_deg(rng: np.random.Generator) -> tuple[float, float]:
    radial_limit_deg = float(TT_LIMIT_ARCMIN) / 60.0
    radial_deg = radial_limit_deg * math.sqrt(float(rng.uniform(0.0, 1.0)))
    angle_rad = float(rng.uniform(0.0, 2.0 * math.pi))
    tilt_x_deg = radial_deg * math.cos(angle_rad)
    tilt_y_deg = radial_deg * math.sin(angle_rad)
    return tilt_x_deg, tilt_y_deg


def _sample_requested_state(rng: np.random.Generator) -> dict[int, SurfacePerturbation]:
    state: dict[int, SurfacePerturbation] = {}
    for anchor_id in ACTUATOR_IDS:
        dx_mm, dy_mm, dz_mm = _sample_xy_z_vector(rng)
        tilt_x_deg, tilt_y_deg = _sample_tilt_vector_deg(rng)
        state[int(anchor_id)] = SurfacePerturbation(
            dx_mm=float(dx_mm),
            dy_mm=float(dy_mm),
            dz_mm=float(dz_mm),
            tilt_x_deg=float(tilt_x_deg),
            tilt_y_deg=float(tilt_y_deg),
        )
    return state


def _accepted_case_specs(num_cases: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    accepted: list[dict[str, Any]] = []
    attempts = 0
    rejections = 0
    while len(accepted) < int(num_cases):
        attempts += 1
        requested_state = _sample_requested_state(rng)
        applied_state, mech = ao_v6_backend._clamped_state(requested_state)
        if not _same_state(requested_state, applied_state):
            rejections += 1
            continue
        if not _mech_summary_ok(mech):
            rejections += 1
            continue
        accepted.append(
            {
                "case_index": int(len(accepted) + 1),
                "requested_state": _state_to_serializable(requested_state),
                "mechanical_summary": mech,
            }
        )
    stats = {
        "num_cases": int(num_cases),
        "sampling_seed": int(seed),
        "xyz_vector_limit_mm": float(XYZ_LIMIT_MM),
        "tt_vector_limit_arcmin": float(TT_LIMIT_ARCMIN),
        "z_prior_limit_mm": float(Z_PRIOR_LIMIT_MM),
        "attempts": int(attempts),
        "rejections": int(rejections),
        "acceptance_rate": float(num_cases / max(attempts, 1)),
    }
    return accepted, stats


def _init_worker() -> None:
    real_shwfs.NUM_MODES = int(NUM_MODES)
    real_shwfs.SHWFS_ACTIVE_MODES = int(NUM_MODES)
    real_shwfs.configure_shwfs_runtime(
        lenslet_count=LENSLET_COUNT,
        slope_limit=SLOPE_LIMIT,
        noise_profile=NOISE_PROFILE,
        forward_averages=FORWARD_AVERAGES,
    )
    nominal_model = bench._build_model(
        pupil_samples=bench.FAST_PUPIL_SAMPLES,
        fft_samples=bench.FAST_FFT_SAMPLES,
    )
    shwfs = real_shwfs._build_shwfs_measurement_model(
        nominal_model,
        focus_shift_mm=bench.REFERENCE_FOCUS_MM,
        actuator_ids=list(ACTUATOR_IDS),
        mechanical_dof_names=("dx_mm", "dy_mm", "dz_mm", "tilt_x_deg", "tilt_y_deg"),
        tilt_step_deg=float(1.0 / 60.0),
        build_mechanical_response=False,
    )
    _WORKER_CACHE["shwfs"] = shwfs


def _simulate_case(case_spec: dict[str, Any]) -> dict[str, Any]:
    if "shwfs" not in _WORKER_CACHE:
        _init_worker()
    shwfs = _WORKER_CACHE["shwfs"]
    shwfs.rng = np.random.default_rng(int(real_shwfs.SHWFS_NOISE_SEED) + int(case_spec["case_index"]))
    moved_model = bench._build_model(
        pupil_samples=bench.FAST_PUPIL_SAMPLES,
        fft_samples=bench.FAST_FFT_SAMPLES,
    )
    requested_state = _serializable_to_state(case_spec["requested_state"])
    moved_model.set_surface_perturbations(requested_state)
    applied_state = bench._anchor_state_subset(moved_model.get_surface_perturbations(), list(ACTUATOR_IDS))
    coeffs_est, slopes, sensor_image = shwfs.estimate_coeffs_with_sensor_image(moved_model)
    opd_map = moved_model.get_wavefront_opd(focus=float(shwfs.focus_shift_mm))
    coeffs_true = real_shwfs._project_w_bessel_coeffs_from_opd(
        moved_model,
        opd_map,
        beam_fill_ratio=real_shwfs.BEAM_FILL_RATIO,
        num_modes=NUM_MODES,
    )
    fit_phase_waves = reconstruct_phase_map(coeffs_est[:NUM_MODES], shwfs.basis_matrix[:NUM_MODES])
    return {
        "case_index": int(case_spec["case_index"]),
        "requested_state": case_spec["requested_state"],
        "applied_state": _state_to_serializable(applied_state),
        "mechanical_summary": case_spec["mechanical_summary"],
        "estimated_coeffs_wb12": np.asarray(coeffs_est[:NUM_MODES], dtype=np.float32),
        "true_coeffs_wb12": np.asarray(coeffs_true[:NUM_MODES], dtype=np.float32),
        "slopes": np.asarray(slopes, dtype=np.float32),
        "sensor_image_adu": np.asarray(sensor_image, dtype=np.float32),
        "estimated_fit_phase_waves": np.asarray(fit_phase_waves, dtype=np.float32),
    }


def _run_serial_cases(case_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _init_worker()
    results = [_simulate_case(case_spec) for case_spec in case_specs]
    results.sort(key=lambda item: int(item["case_index"]))
    return results


def _run_parallel(run_dir: Path, case_specs: list[dict[str, Any]], workers: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _run_parallel_labeled(run_dir, case_specs, workers, label=f"workers_{workers}")


def _run_parallel_labeled(
    run_dir: Path,
    case_specs: list[dict[str, Any]],
    workers: int,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_count = max(1, min(int(workers), len(case_specs)))
    chunks = [case_specs[index::chunk_count] for index in range(chunk_count)]
    chunk_dir = run_dir / f"{label}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[str]] = []
    output_paths: list[Path] = []
    started = time.perf_counter()
    for chunk_index, chunk in enumerate(chunks, start=1):
        specs_path = chunk_dir / f"chunk_{chunk_index:02d}_specs.json"
        output_path = chunk_dir / f"chunk_{chunk_index:02d}_output.json"
        specs_path.write_text(json.dumps(chunk, indent=2), encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--serial-worker-specs",
            str(specs_path),
            "--serial-worker-output",
            str(output_path),
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        processes.append(
            subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        )
        output_paths.append(output_path)

    stderr_lines: list[str] = []
    for process in processes:
        stdout_text, stderr_text = process.communicate()
        if stdout_text.strip():
            stderr_lines.append(stdout_text.strip())
        if stderr_text.strip():
            stderr_lines.append(stderr_text.strip())
        if process.returncode != 0:
            raise RuntimeError("\n".join(stderr_lines) if stderr_lines else f"worker exited {process.returncode}")

    wall_s = time.perf_counter() - started
    results: list[dict[str, Any]] = []
    for output_path in output_paths:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        results.extend(payload["cases"])
    results.sort(key=lambda item: int(item["case_index"]))
    timing = {
        "workers": int(workers),
        "wall_s": float(wall_s),
        "seconds_per_case": float(wall_s / max(len(results), 1)),
        "num_cases": int(len(results)),
    }
    return results, timing


def _save_dataset(run_dir: Path, label: str, results: list[dict[str, Any]], timing: dict[str, Any]) -> None:
    sensor_images = np.stack([np.asarray(row["sensor_image_adu"], dtype=np.float32) for row in results], axis=0)
    estimated_coeffs = np.stack([np.asarray(row["estimated_coeffs_wb12"], dtype=np.float32) for row in results], axis=0)
    true_coeffs = np.stack([np.asarray(row["true_coeffs_wb12"], dtype=np.float32) for row in results], axis=0)
    slopes = np.stack([np.asarray(row["slopes"], dtype=np.float32) for row in results], axis=0)
    fit_phase = np.stack([np.asarray(row["estimated_fit_phase_waves"], dtype=np.float32) for row in results], axis=0)
    np.savez_compressed(
        run_dir / f"{label}_dataset.npz",
        sensor_images_adu=sensor_images,
        estimated_coeffs_wb12=estimated_coeffs,
        true_coeffs_wb12=true_coeffs,
        slopes=slopes,
        estimated_fit_phase_waves=fit_phase,
    )
    json_payload = {
        "timing": timing,
        "cases": [
            {
                "case_index": int(row["case_index"]),
                "requested_state": row["requested_state"],
                "applied_state": row["applied_state"],
                "mechanical_summary": row["mechanical_summary"],
                "estimated_coeffs_wb12": [float(v) for v in row["estimated_coeffs_wb12"]],
                "true_coeffs_wb12": [float(v) for v in row["true_coeffs_wb12"]],
            }
            for row in results
        ],
    }
    (run_dir / f"{label}_summary.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


def _result_to_jsonable(result: dict[str, Any]) -> dict[str, Any]:
    payload = dict(result)
    payload["estimated_coeffs_wb12"] = np.asarray(result["estimated_coeffs_wb12"], dtype=np.float32).tolist()
    payload["true_coeffs_wb12"] = np.asarray(result["true_coeffs_wb12"], dtype=np.float32).tolist()
    payload["slopes"] = np.asarray(result["slopes"], dtype=np.float32).tolist()
    payload["sensor_image_adu"] = np.asarray(result["sensor_image_adu"], dtype=np.float32).tolist()
    payload["estimated_fit_phase_waves"] = np.asarray(result["estimated_fit_phase_waves"], dtype=np.float32).tolist()
    return payload


def _batch_slices(total_count: int, batch_size: int) -> list[tuple[int, int]]:
    slices: list[tuple[int, int]] = []
    start = 0
    while start < int(total_count):
        end = min(int(total_count), start + int(batch_size))
        slices.append((start, end))
        start = end
    return slices


def _update_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument("--sampling-seed", type=int, default=20260502)
    parser.add_argument("--workers", type=int, nargs="*", default=[4, 8])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--serial-worker-specs", type=str, default="")
    parser.add_argument("--serial-worker-output", type=str, default="")
    args = parser.parse_args()

    if args.serial_worker_specs:
        specs = json.loads(Path(args.serial_worker_specs).read_text(encoding="utf-8-sig"))
        results = _run_serial_cases(specs)
        Path(args.serial_worker_output).write_text(
            json.dumps({"cases": [_result_to_jsonable(row) for row in results]}, indent=2),
            encoding="utf-8",
        )
        return 0

    if str(args.run_name).strip():
        run_dir = PILOT_DIR / str(args.run_name).strip()
    else:
        run_dir = PILOT_DIR / time.strftime("%Y%m%d_%H%M%S_wb12_shwfs_pilot")
    run_dir.mkdir(parents=True, exist_ok=False)

    case_specs, sampling_stats = _accepted_case_specs(int(args.num_cases), int(args.sampling_seed))
    (run_dir / "pilot_case_specs.json").write_text(
        json.dumps({"sampling": sampling_stats, "cases": case_specs}, indent=2),
        encoding="utf-8",
    )
    launch_config = {
        "num_cases": int(args.num_cases),
        "sampling_seed": int(args.sampling_seed),
        "workers": [int(v) for v in args.workers],
        "batch_size": int(args.batch_size),
        "run_name": str(args.run_name),
    }
    (run_dir / "launch_config.json").write_text(json.dumps(launch_config, indent=2), encoding="utf-8")

    cpu_count = os.cpu_count() or 1
    manifest = {
        "run_dir": str(run_dir),
        "sampling": sampling_stats,
        "launch_config": launch_config,
        "workers": {},
    }
    _update_manifest(run_dir, manifest)
    timing_rows: list[dict[str, Any]] = []
    for workers in [int(v) for v in args.workers]:
        worker_key = f"workers_{workers}"
        worker_manifest = {
            "workers": int(workers),
            "cpu_count": int(cpu_count),
            "completed_batches": [],
            "total_wall_s": 0.0,
            "total_cases_completed": 0,
        }
        manifest["workers"][worker_key] = worker_manifest
        _update_manifest(run_dir, manifest)
        batch_timings: list[dict[str, Any]] = []
        for batch_index, (start, end) in enumerate(_batch_slices(len(case_specs), int(args.batch_size)), start=1):
            batch_label = f"{worker_key}_batch_{batch_index:04d}"
            batch_specs = case_specs[start:end]
            results, timing = _run_parallel_labeled(run_dir, batch_specs, workers, label=batch_label)
            timing["cpu_count"] = int(cpu_count)
            timing["batch_index"] = int(batch_index)
            timing["case_index_start"] = int(start + 1)
            timing["case_index_end"] = int(end)
            _save_dataset(run_dir, batch_label, results, timing)
            batch_timings.append(timing)
            worker_manifest["completed_batches"].append(
                {
                    "batch_index": int(batch_index),
                    "label": batch_label,
                    "num_cases": int(len(results)),
                    "wall_s": float(timing["wall_s"]),
                    "case_index_start": int(start + 1),
                    "case_index_end": int(end),
                }
            )
            worker_manifest["total_wall_s"] = float(worker_manifest["total_wall_s"]) + float(timing["wall_s"])
            worker_manifest["total_cases_completed"] = int(worker_manifest["total_cases_completed"]) + int(len(results))
            _update_manifest(run_dir, manifest)
            print(
                json.dumps(
                    {
                        "progress": batch_label,
                        "completed_cases": worker_manifest["total_cases_completed"],
                        "total_cases": len(case_specs),
                        "batch_wall_s": timing["wall_s"],
                    }
                )
            )
        timing_rows.extend(batch_timings)

    (run_dir / "timing_summary.json").write_text(json.dumps(timing_rows, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "sampling": sampling_stats, "timing": timing_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
