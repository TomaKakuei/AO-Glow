from __future__ import annotations

import json
import os
from pathlib import Path

from runtime_bootstrap import bootstrap_runtime

bootstrap_runtime()

import numpy as np

import benchmark_1mm_real_shwfs_converging_920nm as bench
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal


ROOT = Path(__file__).resolve().parent
OUTPUT_JSON = ROOT / "artifacts" / "shwfs_estimated_nominal_wrms.json"
PROGRESS_LOG = ROOT / "artifacts" / "shwfs_estimated_nominal_wrms_progress.log"


def _rms_waves_from_phase(phase_waves: np.ndarray, pupil_mask: np.ndarray) -> float:
    mask = np.asarray(pupil_mask, dtype=bool)
    values = np.asarray(phase_waves, dtype=np.float64)[mask]
    centered = values - float(np.mean(values))
    return float(np.sqrt(np.mean(centered * centered, dtype=np.float64)))


def _progress(message: str) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> None:
    PROGRESS_LOG.write_text("", encoding="utf-8")
    noise_profile = str(os.environ.get("AO_V5_SHWFS_NOISE_PROFILE", "realistic")).strip().lower()
    forward_averages = int(os.environ.get("AO_V5_SHWFS_FORWARD_AVERAGES", "1"))
    _progress(f"start profile={noise_profile} avg={forward_averages}")

    real_shwfs.configure_shwfs_runtime(
        lenslet_count=int(bench.LENSLET_COUNT),
        slope_limit=256,
        noise_profile=noise_profile,
        noise_seed=real_shwfs.SHWFS_NOISE_SEED,
        forward_averages=forward_averages,
    )
    _progress("configured shwfs runtime")

    nominal_model = bench._build_model(
        pupil_samples=int(bench.FAST_PUPIL_SAMPLES),
        fft_samples=int(bench.FAST_FFT_SAMPLES),
    )
    _progress("built nominal model")
    shwfs = real_shwfs._build_shwfs_measurement_model(
        nominal_model,
        focus_shift_mm=float(bench.REFERENCE_FOCUS_MM),
    )
    _progress("built shwfs model")
    _, _, _, valid_mask = nominal_model._sample_wavefront(
        num_rays=int(nominal_model.pupil_samples),
        field_index=0,
        wavelength_nm=float(nominal_model.wavelength_nm),
        focus=float(bench.REFERENCE_FOCUS_MM),
    )
    _progress("sampled nominal valid mask")

    compare_target_phase = np.tensordot(shwfs.reference_coeffs, shwfs.basis_matrix, axes=(0, 0))
    compare_target_wrms_waves = _rms_waves_from_phase(compare_target_phase, valid_mask)
    _progress(f"compare target wrms={compare_target_wrms_waves:.9f}")

    fresh_nominal_coeffs = shwfs.estimate_coeffs(nominal_model)
    fresh_nominal_phase = np.tensordot(fresh_nominal_coeffs, shwfs.basis_matrix, axes=(0, 0))
    fresh_nominal_wrms_waves = _rms_waves_from_phase(fresh_nominal_phase, valid_mask)
    _progress(f"fresh nominal wrms={fresh_nominal_wrms_waves:.9f}")

    actual_nominal_metrics = refined_nominal._wavefront_metrics(
        nominal_model,
        focus=float(bench.REFERENCE_FOCUS_MM),
    )
    _progress("computed actual nominal metrics")

    result = {
        "noise_profile": noise_profile,
        "forward_averages": forward_averages,
        "reference_focus_mm": float(bench.REFERENCE_FOCUS_MM),
        "object_space_mm": float(bench.OBJECT_SPACE_MM),
        "compare_target_wrms_waves": float(compare_target_wrms_waves),
        "compare_target_wrms_nm": float(compare_target_wrms_waves * bench.WAVELENGTH_NM),
        "fresh_nominal_estimated_wrms_waves": float(fresh_nominal_wrms_waves),
        "fresh_nominal_estimated_wrms_nm": float(fresh_nominal_wrms_waves * bench.WAVELENGTH_NM),
        "fresh_nominal_residual_norm": float(np.linalg.norm(fresh_nominal_coeffs - shwfs.reference_coeffs)),
        "actual_nominal_wrms_waves": float(actual_nominal_metrics["wavefront_rms_waves"]),
        "actual_nominal_wrms_nm": float(actual_nominal_metrics["wavefront_rms_nm"]),
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _progress("wrote output json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
