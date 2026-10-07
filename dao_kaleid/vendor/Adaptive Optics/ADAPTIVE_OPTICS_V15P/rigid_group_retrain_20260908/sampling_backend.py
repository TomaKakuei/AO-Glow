"""Opt-in sampling acceleration, gated by the paired validation report."""
import json
from contextlib import nullcontext
from common import HERE, RESULTS, sha256


def enabled(system):
    path = HERE/'sampling_backend.json'
    if not path.exists():
        return False
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg.get(system) != 'batch_current_rayoptics_path':
        return False
    if system == 'trepan' and cfg.get('trepan_validation') == 'speckle_mainline_50':
        report = json.loads((RESULTS/'fast_trial50/summary.json').read_text(encoding='utf-8'))
        evidence = report['trepan']
        if any(evidence[k] != 50 for k in ('count','optical_and_noiseless_camera_pass_count','noisy_camera_pass_count')):
            raise RuntimeError('Trepan mainline requires all 50 optical and speckle comparisons')
        if report['backend_sha256'] != cfg['backend_sha256']:
            raise RuntimeError('Trepan validation/backend hash mismatch')
    else:
        report = json.loads((RESULTS/'fast_trial'/f'{system}_report.json').read_text(encoding='utf-8'))
        if not report['passed'] or report['count'] != 10:
            raise RuntimeError(f'{system}: batch backend requires ten passing paired samples')
    if cfg['backend_sha256'] != sha256(HERE/'fast_raytrace.py'):
        raise RuntimeError('Batch backend changed since activation')
    return True


def trepan_context():
    if enabled('trepan'):
        from fast_raytrace import trepan_backend
        return trepan_backend()
    return nullcontext()


def nikon_sampler():
    if enabled('nikon'):
        from fast_raytrace import NikonSampler
        return NikonSampler()
    return None
