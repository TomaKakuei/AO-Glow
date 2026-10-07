"""Resolve the curated source tree on Windows or Linux without home-directory paths."""
from pathlib import Path
import hashlib
import importlib
import json
import sys

PACKAGE = Path(__file__).resolve().parent
VENDOR = PACKAGE / 'vendor'
DAO = VENDOR / 'Adaptive Optics/ADAPTIVE_OPTICS_V15P'
RESTORE = VENDOR / 'RAYOPTICS_RESTORE_20260929'
MOD = RESTORE / 'dao_mod03/sampling_front3_rear4'
BASE = MOD / 'modular_sensorless_1k_r1'
RETRAIN = DAO / 'rigid_group_retrain_20260908'
_SYSTEM = None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def relocate(value):
    """Rebase archived Windows metadata, preserving hashes and numerical values."""
    if isinstance(value, dict):
        return {relocate(k): relocate(v) for k,v in value.items()}
    if isinstance(value, list):
        return [relocate(v) for v in value]
    if isinstance(value, str):
        normalized=value.replace('\\','/')
        for marker in ('Adaptive Optics/','RAYOPTICS_RESTORE_20260929/'):
            if marker in normalized and (':/' in normalized or normalized.startswith('/')):
                return str(VENDOR / (marker+normalized.split(marker,1)[1]))
    return value

def activate(system):
    global _SYSTEM
    if _SYSTEM is not None and _SYSTEM != system:
        raise RuntimeError('Legacy source modules share names. Use a separate process for each optical system.')
    _SYSTEM=system
    directories=[DAO.parent,DAO,DAO/'canonical_three_phase_universal',
        DAO/'minimal_ablation_completion',DAO/'eval_1p4na_realistic_v4',
        DAO/'eval_1p4na_hybrid/model_repair_scratch',DAO/'eval_1p4na_hybrid/v3_corrected',
        DAO/'kaleid_scope_repair_20260906',RETRAIN,
        DAO/'kaleid_modular_repair_20260906',DAO/'OL']
    if system=='mod3':
        directories.extend([RESTORE,RESTORE/'patent_comparison',RESTORE/'dao_mod03',
            MOD,BASE,BASE/'controller_repair01',BASE/'nonlinear_prior01'])
    # Explicit order avoids depending on whatever folder launched Python.
    for directory in directories:
        text=str(directory)
        if text in sys.path:sys.path.remove(text)
        sys.path.insert(0,text)
    if system!='mod3':
        # 'common' in original active-model bridge is the retraining common.
        sys.path.remove(str(RETRAIN));sys.path.insert(0,str(RETRAIN))
    return directories

def legacy_bridge(device=None):
    bridge=importlib.import_module('control_bridge')
    source=json.loads((RETRAIN/'active_manifest.json').read_text(encoding='utf-8'))
    manifest=relocate(source)
    bridge.manifest=lambda:manifest
    if device is not None:
        import torch
        bridge.DEVICE=torch.device(device)
    return bridge

def registry():
    return json.loads((PACKAGE/'model_registry.json').read_text(encoding='utf-8'))

def checkpoint(system, profile, key):
    entry=registry()['systems'][system][profile]['checkpoints'][key]
    path=PACKAGE/entry['path']
    if sha(path)!=entry['sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch: {system}/{profile}/{key}')
    return path
