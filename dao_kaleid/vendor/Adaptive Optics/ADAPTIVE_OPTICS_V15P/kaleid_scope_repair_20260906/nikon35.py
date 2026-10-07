"""Complete seven-body Nikon objective. Separate from all archived 30-DOF runs."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
from scipy.interpolate import griddata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'eval_1p4na_realistic_v4'
for directory in (ROOT, SOURCE, ROOT / 'canonical_three_phase_universal'):
    sys.path.insert(0, str(directory))
import run_repaired_small_11case_audit as base

base.absolute.disable_file_logging()
base.configure_small_scales()
_bias30, _, _ = base.load_bias()
BIAS = np.r_[np.zeros(5), _bias30]
GROUPS = {'L1': (3, 5), 'L2': (6, 7), 'L3': (8, 10),
          'L4': (11, 13), 'L5': (14, 16), 'L6': (17, 20), 'L7p': (21, 24)}
MODULES = {'front': (0, 1, 2), 'middle': (3, 4, 5), 'rear': (6,)}
SCALES = np.tile([.020, .020, .020, .040, .040], 7)
base.corrected.GROUPS = GROUPS.copy()
base.corrected.SCALES = SCALES.copy()
base.SMALL_SCALES = SCALES.copy()
NODES = base.corrected.CONTROL_PUPIL_NODES
AUDIT_NODES = base.corrected.AUDIT_PUPIL_NODES
GX, GY = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
GRID_MATRIX = (griddata(NODES, np.eye(len(NODES)), (GX, GY),
                        method='linear', fill_value=0) * (GX * GX + GY * GY <= 1)[..., None]).astype(np.float32)
TEST_SEEDS = (4, 7, 13, 14, 21, 25, 27, 31, 32, 41, 42)
DATA_OLD = SOURCE / 'results_repaired_multigroup_30case_nn_linear_local_audit/training/capture_dataset_3000.npz'


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sample(seed):
    """Two to four complete five-axis lens bodies; no outcome-dependent choices."""
    rng = np.random.RandomState(int(seed))
    count = 2 + int(seed) % 3
    first = int(rng.choice([0, 1, 2]))
    second = int(rng.choice([3, 4, 5, 6]))
    rest = [i for i in range(7) if i not in (first, second)]
    groups = sorted([first, second] + list(rng.choice(rest, size=count-2, replace=False)))
    q = np.zeros(35)
    for group in groups:
        q[group*5:(group+1)*5] = rng.uniform(-1, 1, 5)
    return q, [list(GROUPS)[i] for i in groups]


def grid(raw):
    return np.einsum('...fk,hwk->...fhw', np.asarray(raw).reshape(-1, 3, 49), GRID_MATRIX).astype(np.float32)


class Plant:
    """Controller interface accepts commands and returns measured fields only."""
    def __init__(self, hidden=None, noise_seed=0, noise_std=.015, score_center=False):
        self._hidden = np.zeros(35) if hidden is None else np.asarray(hidden).copy()
        self._plant = base.BiasedHighNAPlant(BIAS, self._hidden)
        self._absolute, _ = base.build_validated_forward_model()
        self.rng = np.random.RandomState(int(noise_seed))
        self.noise_std = noise_std
        self.score_center = score_center
        self.readings = 0
        self.records = []

    def measure(self, command):
        command = np.asarray(command, float)
        self.readings += 1
        try:
            value = self._plant.evaluate(command)
            raw = np.asarray(value.residual_waves, float)
            noise = self.rng.normal(0., self.noise_std, raw.shape)
            result = raw + noise
            self.records.append({'reading': self.readings, 'command': command.tolist(),
                                 'score': float(np.sqrt(np.mean(result**2))), 'valid': True})
            if self.score_center:
                center=self.terminal(command)[0]['residual_waves']
                center=center+self.rng.normal(0., self.noise_std, center.shape)
                self.records[-1]['absolute_center_score']=float(np.sqrt(np.mean(center**2)))
                return {'network':result,'center':center}
            return result
        except Exception as exc:
            self.records.append({'reading': self.readings, 'command': command.tolist(),
                                 'valid': False, 'error': str(exc)})
            return None

    def terminal(self, command, fields=(0.,)):
        output = []
        for field in fields:
            output.append(base.absolute.reference_sphere_rms(
                self._absolute, BIAS + self._hidden + command, field, AUDIT_NODES))
        return output


def inspect():
    out = HERE / 'results'
    out.mkdir(parents=True,exist_ok=True)
    t0 = time.perf_counter()
    plant = Plant(noise_std=0.)
    zero = np.zeros(35)
    nominal = plant.terminal(zero)[0]['absolute_pttd_rms_waves']
    reference = plant.measure(zero)
    response = []
    step = .001
    for index in range(35):
        dq = np.zeros(35); dq[index] = step
        plus, minus = plant.measure(dq), plant.measure(-dq)
        if plus is None or minus is None:
            raise RuntimeError(f'Coordinate {index} failed local trace')
        response.append(((plus-minus)/(2*step)).ravel())
    jac = np.column_stack(response)
    np.savez_compressed(out / 'calibration35.npz', jacobian=jac, bias=BIAS, scales=SCALES, grid_matrix=GRID_MATRIX)
    cases = []
    for seed in TEST_SEEDS:
        hidden, groups = sample(seed)
        candidate = Plant(hidden, noise_seed=seed, noise_std=0.)
        measured = candidate.measure(zero)
        try:
            initial = candidate.terminal(zero)[0]['absolute_pttd_rms_waves']
            error = None
        except Exception as exc:
            initial, error = None, str(exc)
        cases.append({'seed': seed, 'active_groups': groups, 'hidden': hidden.tolist(),
                      'initial_wrms': initial, 'complete_control_trace': measured is not None, 'error': error})
        print(f'INITIAL seed={seed} groups={groups} WRMS={initial}', flush=True)
    singular = np.linalg.svd(jac, compute_uv=False)
    report = {'groups': GROUPS, 'modules': MODULES, 'coordinates': ['dx','dy','dz','tx','ty'],
              'dofs': 35, 'nominal_absolute_center_wrms': float(nominal),
              'jacobian_shape': list(jac.shape), 'column_norms': np.linalg.norm(jac, axis=0).tolist(),
              'singular_values': singular.tolist(), 'cases': cases,
              'elapsed_seconds': time.perf_counter()-t0, 'old_training_states_reusable': 3000}
    save_json(out / 'physical_check.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'column_norms', 'singular_values')}, indent=2), flush=True)


if __name__ == '__main__':
    inspect()
