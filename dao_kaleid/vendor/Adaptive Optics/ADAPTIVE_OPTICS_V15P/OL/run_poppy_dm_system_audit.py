"""KLA case definitions only; unrelated DM/Poppy experiments omitted."""
import math
import numpy as np
from kla_full_fov_engine import KLAFullFOVEngine

KLA_CASE_DEFINITIONS = ({'seed': 156, 'shift_mm': 0.08, 'tilt_arcmin': 2.5}, {'seed': 128, 'shift_mm': 0.08, 'tilt_arcmin': 2.5}, {'seed': 121, 'shift_mm': 0.08, 'tilt_arcmin': 2.5}, {'seed': 135, 'shift_mm': 0.08, 'tilt_arcmin': 2.5}, {'seed': 142, 'shift_mm': 0.08, 'tilt_arcmin': 2.5}, {'seed': 256, 'shift_mm': 0.04, 'tilt_arcmin': 1.2}, {'seed': 480, 'shift_mm': 0.04, 'tilt_arcmin': 1.2}, {'seed': 270, 'shift_mm': 0.04, 'tilt_arcmin': 1.2}, {'seed': 221, 'shift_mm': 0.04, 'tilt_arcmin': 1.2}, {'seed': 389, 'shift_mm': 0.04, 'tilt_arcmin': 1.2}, {'seed': 200, 'shift_mm': 0.04, 'tilt_arcmin': 1.2})

def make_kla_perturbations(engine: KLAFullFOVEngine, definition: dict) -> dict:
    seed = int(definition['seed'])
    shift = float(definition['shift_mm'])
    tilt_rad = float(definition['tilt_arcmin']) / 60.0 * math.pi / 180.0
    np.random.seed(seed)
    perturbations = {}
    for module_id in (2, 3, 4):
        for lens_name in engine.module_lenses[module_id]:
            dx, dy = np.random.uniform(-shift, shift, 2)
            dz = np.random.uniform(-shift, shift)
            tx, ty = np.random.uniform(-tilt_rad, tilt_rad, 2)
            perturbations[lens_name] = {'dx': float(dx), 'dy': float(dy), 'dz': float(dz), 'tx': float(tx), 'ty': float(ty)}
    return perturbations
