from pathlib import Path

log = Path("tmp_debug_ascii.log")
log.write_text("", encoding="ascii")


def w(msg: str) -> None:
    with log.open("a", encoding="ascii") as handle:
        handle.write(msg + "\n")


w("start")

import numpy as np

w("import numpy")

import benchmark_1mm_real_shwfs_converging_920nm as bench

w("import bench")

import benchmark_freeform_real_shwfs_residual_120 as real_shwfs

w("import real_shwfs")

import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal

w("import refined_nominal")

from optical_model_rayoptics import SurfacePerturbation

w("import SurfacePerturbation")

bench.NUM_MODES = 12
real_shwfs.NUM_MODES = 12
refined_nominal.NUM_MODES = 12
ACTUATOR_IDS = [68, 70, 72, 74, 77, 79, 81, 84]

w("before nominal model")
model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
w("after nominal model")
shwfs = real_shwfs._build_shwfs_measurement_model(model, focus_shift_mm=bench.REFERENCE_FOCUS_MM, actuator_ids=ACTUATOR_IDS)
w(f"after shwfs {shwfs.response_matrix.shape}")

rng = np.random.default_rng(123)
state = {}
for anchor in ACTUATOR_IDS:
    state[int(anchor)] = SurfacePerturbation(
        dx_mm=float(rng.uniform(-0.01, 0.01)),
        dy_mm=float(rng.uniform(-0.01, 0.01)),
        dz_mm=float(rng.uniform(-0.01, 0.01)),
    )
w("after state")

model2 = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
w("after model2")
model2.set_surface_perturbations(state)
w("after set_surface_perturbations")
coeffs = shwfs.estimate_coeffs(model2)
w("after estimate_coeffs")
residual = float(np.linalg.norm(coeffs - shwfs.reference_coeffs))
w(f"residual {residual}")
