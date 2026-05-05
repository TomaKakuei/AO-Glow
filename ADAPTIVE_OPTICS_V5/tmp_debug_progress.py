from pathlib import Path
log = Path('tmp_debug_progress.log')
def w(msg):
    with log.open('a', encoding='utf-8') as f:
        f.write(msg+'\n')
w('start')
import numpy as np
w('import numpy')
import benchmark_1mm_real_shwfs_converging_920nm as bench
w('import bench')
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
w('import real_shwfs')
import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal
w('import refined')
from optical_model_rayoptics import SurfacePerturbation
w('import perturbation')
bench.NUM_MODES = 12
real_shwfs.NUM_MODES = 12
refined_nominal.NUM_MODES = 12
ACTUATOR_IDS = [68,70,72,74,77,79,81,84]
nominal_model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
w('built nominal model')
shwfs = real_shwfs._build_shwfs_measurement_model(nominal_model, focus_shift_mm=bench.REFERENCE_FOCUS_MM, actuator_ids=ACTUATOR_IDS)
w(f'built shwfs {shwfs.response_matrix.shape}')
model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
w('built test model')
state = {a: SurfacePerturbation(dx_mm=0.001, dy_mm=0.0, dz_mm=0.0) for a in ACTUATOR_IDS}
model.set_surface_perturbations(state)
w('set perturbations')
coeffs = shwfs.estimate_coeffs(model)
w('estimated coeffs')
import numpy.linalg as LA
res = float(LA.norm(coeffs - shwfs.reference_coeffs))
w(f'residual {res}')
