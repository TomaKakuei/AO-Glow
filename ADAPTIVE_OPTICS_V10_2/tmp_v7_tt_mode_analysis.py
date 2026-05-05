from __future__ import annotations

import json
from pathlib import Path
import traceback

import numpy as np

import benchmark_1mm_real_shwfs_converging_920nm as bench
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
import ao_v5_backend as backend
from optical_model_rayoptics import SurfacePerturbation
from w_bessel_core import nm_sequence


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "artifacts" / "v7_tt_mode_analysis.json"
ACTUATOR_IDS = backend.ACTUATOR_IDS


def _make_global_tt_state(*, tilt_x_deg: float = 0.0, tilt_y_deg: float = 0.0) -> dict[int, SurfacePerturbation]:
    return {
        surface_id: SurfacePerturbation(tilt_x_deg=float(tilt_x_deg), tilt_y_deg=float(tilt_y_deg))
        for surface_id in ACTUATOR_IDS
    }


def _analyze_case(
    optical_model,
    shwfs,
    reference_coeffs: np.ndarray,
    *,
    name: str,
    state: dict[int, SurfacePerturbation],
) -> dict[str, object]:
    optical_model.set_surface_perturbations(state)
    coeffs = np.asarray(shwfs.estimate_coeffs(optical_model), dtype=np.float64)
    residual = coeffs - reference_coeffs
    signature = backend._residual_block_signature(residual)
    return {
        "name": name,
        "coeffs": coeffs.tolist(),
        "residual": residual.tolist(),
        "residual_norm": float(np.linalg.norm(residual)),
        "signature": {key: float(value) for key, value in signature.items()},
    }


def main() -> None:
    print("configuring runtime...", flush=True)
    real_shwfs.configure_shwfs_runtime(
        lenslet_count=13,
        slope_limit=256,
        noise_profile="realistic",
        forward_averages=1,
    )

    print("building nominal model...", flush=True)
    nominal_model = bench._build_model(
        pupil_samples=bench.FAST_PUPIL_SAMPLES,
        fft_samples=bench.FAST_FFT_SAMPLES,
    )
    print("computing nominal wavefront metrics...", flush=True)
    nominal_wavefront_metrics = real_shwfs._wavefront_metrics(nominal_model)
    focus_shift_mm = float(nominal_wavefront_metrics["best_focus_shift_mm"])
    print("building shwfs model...", flush=True)
    shwfs = real_shwfs._build_shwfs_measurement_model(
        nominal_model,
        focus_shift_mm=focus_shift_mm,
        actuator_ids=ACTUATOR_IDS,
        tilt_step_deg=float(backend.TIPTILT_RESPONSE_STEP_DEG),
        mechanical_dof_names=("dx_mm", "dy_mm", "dz_mm", "tilt_x_deg", "tilt_y_deg"),
    )

    print("analyzing cases...", flush=True)
    reference_coeffs = np.asarray(shwfs.reference_coeffs, dtype=np.float64)
    labels = [{"index": int(i), "nm": list(pair)} for i, pair in enumerate(nm_sequence(real_shwfs.NUM_MODES))]

    step_deg = 1.0 / 60.0
    cases = [
        _analyze_case(nominal_model, shwfs, reference_coeffs, name="global_ttx_1arcmin", state=_make_global_tt_state(tilt_x_deg=step_deg)),
        _analyze_case(nominal_model, shwfs, reference_coeffs, name="global_tty_1arcmin", state=_make_global_tt_state(tilt_y_deg=step_deg)),
        _analyze_case(
            nominal_model,
            shwfs,
            reference_coeffs,
            name="global_ttx_tty_3arcmin",
            state=_make_global_tt_state(tilt_x_deg=3.0 / 60.0, tilt_y_deg=3.0 / 60.0),
        ),
    ]

    sensitivity_matrix = np.stack(
        [
            np.asarray(cases[0]["residual"], dtype=np.float64) / step_deg,
            np.asarray(cases[1]["residual"], dtype=np.float64) / step_deg,
        ],
        axis=1,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "noise_profile": "realistic",
                "num_modes": int(real_shwfs.NUM_MODES),
                "mode_labels": labels,
                "tt_global_sensitivity_matrix_per_deg": sensitivity_matrix.tolist(),
                "cases": cases,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(str(OUT_PATH))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
