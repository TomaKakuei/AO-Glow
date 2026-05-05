import numpy as np
import benchmark_1mm_real_shwfs_converging_920nm as bench
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal
from optical_model_rayoptics import SurfacePerturbation
print('imports ok', flush=True)
bench.NUM_MODES = 12
real_shwfs.NUM_MODES = 12
refined_nominal.NUM_MODES = 12
ACTUATOR_IDS = [68,70,72,74,77,79,81,84]
nominal_model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
print('model ok', flush=True)
shwfs = real_shwfs._build_shwfs_measurement_model(nominal_model, focus_shift_mm=bench.REFERENCE_FOCUS_MM, actuator_ids=ACTUATOR_IDS)
print('shwfs ok', shwfs.response_matrix.shape, flush=True)
def residual_for_state(state):
    model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
    model.set_surface_perturbations(state)
    coeffs = shwfs.estimate_coeffs(model)
    return float(np.linalg.norm(coeffs - shwfs.reference_coeffs))
rng = np.random.default_rng(123)
state = {a: SurfacePerturbation(dx_mm=float(rng.uniform(-0.01,0.01)), dy_mm=float(rng.uniform(-0.01,0.01)), dz_mm=float(rng.uniform(-0.01,0.01))) for a in ACTUATOR_IDS}
print('dxyz10um', residual_for_state(state), flush=True)
