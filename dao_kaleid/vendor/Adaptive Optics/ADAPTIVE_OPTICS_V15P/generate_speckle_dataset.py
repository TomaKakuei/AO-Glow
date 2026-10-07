import os
import time
import math
import numpy as np
import scipy.fft as sfft
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from runtime_bootstrap import bootstrap_runtime
    bootstrap_runtime()
except ImportError:
    pass

import argparse
from optical_model_rayoptics import RayOpticsPhysicsEngine, SurfacePerturbation
from benchmark_freeform_real_shwfs_residual_120 import _build_shwfs_measurement_model

ROOT = Path(__file__).resolve().parent / "artifacts"
TOTAL_CASES = 5000
BATCH_SIZE = 100
NUM_BATCHES = TOTAL_CASES // BATCH_SIZE

REFERENCE_FOCUS_MM = 4.64725296651765

POV_AXIS_X_DEG = (-2.0, -1.0, 0.0, 1.0, 2.0)
POV_AXIS_Y_DEG = (-1.8, -1.0, 0.0, 1.0, 2.2)

def _pov_points() -> tuple[tuple[float, float], ...]:
    rows = [(float(x_deg), 0.0) for x_deg in POV_AXIS_X_DEG if x_deg != 0.0]
    rows.extend((0.0, float(y_deg)) for y_deg in POV_AXIS_Y_DEG)
    return tuple(rows)

def _set_field_point(model: RayOpticsPhysicsEngine, x_deg: float, y_deg: float) -> None:
    field = model.optical_spec["fov"].fields[0]
    field.x = float(x_deg)
    field.y = float(y_deg)
    model.opm.update_model()

def generate_random_phase_mask(size: int, grit_size: float = 2.0, max_phase: float = 2*np.pi) -> np.ndarray:
    noise = np.random.randn(size, size)
    f_noise = sfft.fft2(noise)
    f_noise = sfft.fftshift(f_noise)
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    mask = np.exp(-(x**2 + y**2) / (2 * grit_size**2))
    filtered_f_noise = f_noise * mask
    smooth_noise = np.real(sfft.ifft2(sfft.ifftshift(filtered_f_noise)))
    smooth_noise = smooth_noise - np.min(smooth_noise)
    smooth_noise = (smooth_noise / np.max(smooth_noise)) * max_phase
    return smooth_noise

MASK_PATH = ROOT / "diffuser_mask.npy"
if not MASK_PATH.exists():
    print("Generating Fixed Diffuser Mask...")
    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixed_mask = generate_random_phase_mask(128, grit_size=3.0, max_phase=4*np.pi)
    np.save(MASK_PATH, fixed_mask)
else:
    fixed_mask = np.load(MASK_PATH)

def fresnel_propagate(U_in: np.ndarray, dx: float, z: float, wavelength: float) -> np.ndarray:
    N = U_in.shape[0]
    df = 1.0 / (N * dx)
    y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
    fx = x * df
    fy = y * df
    k = 2 * np.pi / wavelength
    term = 1.0 - (wavelength * fx)**2 - (wavelength * fy)**2
    term = np.maximum(term, 0.0)
    H = np.exp(1j * k * z * np.sqrt(term))
    H = sfft.ifftshift(H)
    return sfft.ifft2(sfft.fft2(U_in) * H)

def apply_minimal_scmos_noise(image: np.ndarray) -> np.ndarray:
    peak_I = float(np.max(image))
    if peak_I <= 0.0:
        return np.zeros_like(image, dtype=np.float32)
    fixed_gain_e_per_raw = 4000.0 / peak_I
    expected_e = np.clip(image * fixed_gain_e_per_raw, 0.0, 10000.0)
    
    prnu = np.random.normal(1.0, 0.005, size=image.shape)
    expected_e = expected_e * prnu
    
    shot_e = np.random.poisson(expected_e).astype(np.float64)
    read_e = np.random.normal(0.0, 1.2, size=image.shape)
    dsnu = np.random.normal(2.0, 0.5, size=image.shape)
    
    sensed_e = np.clip(shot_e + read_e + dsnu, 0.0, 10000.0)
    adu = np.clip(np.rint(sensed_e * 4.0 + 100.0), 0.0, 65535.0)
    return adu.astype(np.float32)

def _build_model() -> RayOpticsPhysicsEngine:
    return RayOpticsPhysicsEngine(
        design_name="2P_AO",
        propagation_direction="forward",
        include_tube_lens=False,
        pupil_samples=64,
        fft_samples=128
    )

def _sample_state(rng, anchors, group: str) -> dict:
    xyz_limit = 0.040
    tt_limit_deg = 0.020
    mode = int(rng.integers(0, 5))

    state = {}
    
    front_anchors = [68, 70, 72, 74]
    rear_anchors = [77, 79, 81, 84]
    active_anchors = front_anchors if group == "front" else rear_anchors

    dominant_anchor = int(rng.choice(active_anchors))
    dominant_axis = int(rng.integers(0, 5))

    for anchor_id in anchors:
        if anchor_id not in active_anchors:
            state[anchor_id] = SurfacePerturbation(0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        if mode == 0:
            scale_xyz = 1.0
            scale_tt = 1.0
        elif mode == 1:
            scale_xyz = 1.0 if int(anchor_id) == dominant_anchor else 0.25
            scale_tt = 1.0 if int(anchor_id) == dominant_anchor else 0.25
        elif mode == 2:
            scale_xyz = 1.0
            scale_tt = 1.0
        elif mode == 3:
            scale_xyz = 0.20
            scale_tt = 0.20
        else:
            scale_xyz = 0.90
            scale_tt = 0.90

        dx = rng.uniform(-xyz_limit, xyz_limit) * scale_xyz
        dy = rng.uniform(-xyz_limit, xyz_limit) * scale_xyz
        dz = rng.uniform(-xyz_limit, xyz_limit) * scale_xyz
        tx = rng.uniform(-tt_limit_deg, tt_limit_deg) * scale_tt
        ty = rng.uniform(-tt_limit_deg, tt_limit_deg) * scale_tt

        if mode == 2:
            if dominant_axis == 0:
                dy *= 0.15
                dz *= 0.15
                tx *= 0.20
                ty *= 0.20
            elif dominant_axis == 1:
                dx *= 0.15
                dz *= 0.15
                tx *= 0.20
                ty *= 0.20
            elif dominant_axis == 2:
                dx *= 0.20
                dy *= 0.20
                tx *= 0.20
                ty *= 0.20
            elif dominant_axis == 3:
                dx *= 0.20
                dy *= 0.20
                dz *= 0.20
                ty *= 0.35
            else:
                dx *= 0.20
                dy *= 0.20
                dz *= 0.20
                tx *= 0.35
        elif mode == 3:
            dx *= 0.20
            dy *= 0.20
            dz *= 0.20
            tx *= 0.20
            ty *= 0.20
        elif mode == 4:
            dx = math.copysign(rng.uniform(0.8 * xyz_limit, xyz_limit), rng.uniform(-1.0, 1.0))
            dy = math.copysign(rng.uniform(0.8 * xyz_limit, xyz_limit), rng.uniform(-1.0, 1.0))
            dz = math.copysign(rng.uniform(0.8 * xyz_limit, xyz_limit), rng.uniform(-1.0, 1.0))
            tx = math.copysign(rng.uniform(0.8 * tt_limit_deg, tt_limit_deg), rng.uniform(-1.0, 1.0))
            ty = math.copysign(rng.uniform(0.8 * tt_limit_deg, tt_limit_deg), rng.uniform(-1.0, 1.0))

        state[anchor_id] = SurfacePerturbation(
            dx_mm=float(dx),
            dy_mm=float(dy),
            dz_mm=float(dz),
            tilt_x_deg=float(tx),
            tilt_y_deg=float(ty),
        )
    return state

def worker_process_batch(args):
    batch_idx, group, dataset_dir = args
    rng = np.random.default_rng(42 + batch_idx)
    
    print(f"Worker {batch_idx}: Initializing 9 models...")
    engines = []
    shwfs_models = []
    for x_deg, y_deg in _pov_points():
        engine = _build_model()
        _set_field_point(engine, float(x_deg), float(y_deg))
        _ = engine.get_wavefront_opd(focus=REFERENCE_FOCUS_MM)
        engines.append(engine)
        shwfs = _build_shwfs_measurement_model(engine, focus_shift_mm=0.0)
        shwfs_models.append(shwfs)
    print(f"Worker {batch_idx}: Models ready.")
        
    wavelength_mm = float(engines[0].wavelength_nm) * 1e-6
    dx = float(engines[0].pupil_diameter_mm) / 64.0
    anchors = sorted(list(engines[0].group_anchor_to_members.keys()))
    
    batch_speckles = []
    batch_shwfs = []
    batch_opds = []
    batch_mechs = []
    batch_wrms = []
    
    mask = np.load(MASK_PATH)
    
    skipped = 0
    i = 0
    while i < BATCH_SIZE:
        perturbations = _sample_state(rng, anchors, group)
        mech_vec = []
        
        active_anchors = [68, 70, 72, 74] if group == "front" else [77, 79, 81, 84]
        for anchor in active_anchors:
            p = perturbations[anchor]
            mech_vec.extend([p.dx_mm, p.dy_mm, p.dz_mm, p.tilt_x_deg, p.tilt_y_deg])
            
        try:
            speckle_stack = []
            shwfs_stack = []
            center_opd = None
            wrms = 0.0
            
            for fov_idx, (x_deg, y_deg) in enumerate(_pov_points()):
                engine = engines[fov_idx]
                shwfs = shwfs_models[fov_idx]
                engine.set_surface_perturbations(perturbations, clear_others=True)
                
                _, coeffs = shwfs.estimate_from_model(engine)
                shwfs_stack.append(np.asarray(coeffs, dtype=np.float32))
                
                cam_opd = np.asarray(engine.get_wavefront_opd(focus=REFERENCE_FOCUS_MM), dtype=np.float64)
                
                if x_deg == 0.0 and y_deg == 0.0:
                    valid_opd = cam_opd[cam_opd != 0.0]
                    wrms = np.std(valid_opd) / wavelength_mm if valid_opd.size > 0 else 0.0
                    center_opd = cam_opd.copy()
                
                A = (cam_opd != 0).astype(float)
                phase = (2 * np.pi / wavelength_mm) * cam_opd
                U_pupil = A * np.exp(1j * phase)
                
                U_pupil_128 = np.zeros((128, 128), dtype=np.complex128)
                U_pupil_128[32:96, 32:96] = U_pupil
                
                U_modulated = U_pupil_128 * np.exp(1j * mask)
                U_sensor = fresnel_propagate(U_modulated, dx=dx, z=2.0, wavelength=wavelength_mm)
                speckle_I = np.abs(U_sensor)**2
                speckle_noisy = apply_minimal_scmos_noise(speckle_I)
                speckle_stack.append(speckle_noisy)
            
        except Exception as e:
            skipped += 1
            continue
            
        batch_speckles.append(np.stack(speckle_stack, axis=0))
        batch_shwfs.append(np.stack(shwfs_stack, axis=0))
        batch_opds.append(center_opd.astype(np.float32))
        batch_mechs.append(np.array(mech_vec, dtype=np.float32))
        batch_wrms.append(wrms)
        i += 1
        print(f"Batch {batch_idx:04d}: Case {i}/{BATCH_SIZE} done", flush=True)
        
    out_path = Path(dataset_dir) / f"batch_{batch_idx:04d}.npz"
    np.savez_compressed(
        out_path,
        speckles=np.stack(batch_speckles),
        shwfs_coeffs=np.stack(batch_shwfs),
        opds=np.stack(batch_opds),
        mechs=np.stack(batch_mechs),
        wrms=np.array(batch_wrms, dtype=np.float32)
    )
    print(f"[{time.strftime('%H:%M:%S')}] Saved {out_path} (skipped {skipped} unphysical cases)")

def run_generation(group_name: str):
    dataset_dir = ROOT / f"{group_name}_speckle_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting generation of {TOTAL_CASES} cases for group '{group_name}' in {NUM_BATCHES} batches...")
    t0 = time.time()
    
    worker_args = [(i, group_name, str(dataset_dir)) for i in range(NUM_BATCHES)]
    
    with mp.Pool(4) as p:
        p.map(worker_process_batch, worker_args)
    t1 = time.time()
    print(f"Finished '{group_name}' dataset generation in {(t1-t0)/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=str, required=True, choices=["front", "rear", "all"], help="Which group to generate perturbations for, or 'all' for sequential generation.")
    args = parser.parse_args()
    
    if args.group == "all":
        run_generation("front")
        run_generation("rear")
    else:
        run_generation(args.group)
