import json
from pathlib import Path
import numpy as np
import benchmark_1mm_real_shwfs_converging_920nm as bench
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal
from optical_model_rayoptics import SurfacePerturbation

bench.NUM_MODES = 12
real_shwfs.NUM_MODES = 12
refined_nominal.NUM_MODES = 12
ACTUATOR_IDS = [68, 70, 72, 74, 77, 79, 81, 84]

nominal_model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
shwfs = real_shwfs._build_shwfs_measurement_model(
    nominal_model,
    focus_shift_mm=bench.REFERENCE_FOCUS_MM,
    actuator_ids=ACTUATOR_IDS,
)

rng_master = np.random.default_rng(20260501)
case_seeds = rng_master.integers(0, 2**31 - 1, size=10, dtype=np.int64)


def residual_for_state(state):
    model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
    model.set_surface_perturbations(state)
    coeffs = shwfs.estimate_coeffs(model)
    return float(np.linalg.norm(coeffs - shwfs.reference_coeffs))


def random_dxyz_case(amplitude_mm, seed):
    rng = np.random.default_rng(int(seed))
    state = {}
    for anchor in ACTUATOR_IDS:
        state[int(anchor)] = SurfacePerturbation(
            dx_mm=float(rng.uniform(-amplitude_mm, amplitude_mm)),
            dy_mm=float(rng.uniform(-amplitude_mm, amplitude_mm)),
            dz_mm=float(rng.uniform(-amplitude_mm, amplitude_mm)),
        )
    return state


def random_tilt_case(amplitude_arcmin, seed):
    amplitude_deg = float(amplitude_arcmin) / 60.0
    rng = np.random.default_rng(int(seed))
    state = {}
    for anchor in ACTUATOR_IDS:
        state[int(anchor)] = SurfacePerturbation(
            tilt_x_deg=float(rng.uniform(-amplitude_deg, amplitude_deg)),
            tilt_y_deg=float(rng.uniform(-amplitude_deg, amplitude_deg)),
        )
    return state


def summarize_group(group_name, level_value, builder):
    rows = []
    for idx, seed in enumerate(case_seeds, start=1):
        residual_norm = residual_for_state(builder(level_value, int(seed)))
        rows.append({'case_index': int(idx), 'seed': int(seed), 'residual_norm': float(residual_norm)})
    residuals = np.asarray([row['residual_norm'] for row in rows], dtype=np.float64)
    return {
        'group': group_name,
        'level': level_value,
        'cases': rows,
        'summary': {
            'mean_residual': float(np.mean(residuals)),
            'median_residual': float(np.median(residuals)),
            'min_residual': float(np.min(residuals)),
            'max_residual': float(np.max(residuals)),
            'std_residual': float(np.std(residuals)),
        },
    }

analysis = {
    'configuration': {
        'object_space_mm': float(bench.OBJECT_SPACE_MM),
        'wavelength_nm': float(bench.WAVELENGTH_NM),
        'reference_focus_mm': float(bench.REFERENCE_FOCUS_MM),
        'shwfs_lenslet_count': int(shwfs.lenslet_count),
        'shwfs_slope_channels': int(shwfs.reference_slopes.size),
        'num_modes': 12,
        'actuator_ids': [int(v) for v in ACTUATOR_IDS],
        'case_seed_vector': [int(v) for v in case_seeds],
    },
    'formula': {
        'phase_model': 'phi(rho,theta) ~= sum_{k=1}^{12} a_k B_k(rho,theta)',
        'shwfs_inverse': 'c_hat = pinv(R) @ s',
        'reference': 'c_ref = pinv(R) @ s_ref',
        'residual_vector': 'r = c_hat - c_ref',
        'residual_norm': '||r||_2 = sqrt(sum_{k=1}^{12} (c_hat_k - c_ref_k)^2)',
    },
    'dxyz_groups': [
        summarize_group('dxyz_mm', 0.010, random_dxyz_case),
        summarize_group('dxyz_mm', 0.100, random_dxyz_case),
        summarize_group('dxyz_mm', 0.500, random_dxyz_case),
    ],
    'tilt_groups': [
        summarize_group('tilt_arcmin', 2.0, random_tilt_case),
        summarize_group('tilt_arcmin', 5.0, random_tilt_case),
        summarize_group('tilt_arcmin', 10.0, random_tilt_case),
    ],
}

out_path = Path('artifacts') / 'wb12_residual_sensitivity_analysis.json'
out_path.write_text(json.dumps(analysis, ensure_ascii=True, indent=2), encoding='utf-8')
print(json.dumps({'output_path': str(out_path), 'analysis': analysis}, ensure_ascii=True))
