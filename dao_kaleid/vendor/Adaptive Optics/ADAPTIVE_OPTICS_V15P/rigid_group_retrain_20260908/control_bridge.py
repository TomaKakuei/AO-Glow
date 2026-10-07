from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import ACTIVE, ROOT

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE = {}


def manifest():
    if not ACTIVE.exists():
        raise RuntimeError(f"Rigid-body-corrected models are not active yet: {ACTIVE}")
    value = json.loads(ACTIVE.read_text(encoding="utf-8"))
    if value.get("state") != "ready":
        raise RuntimeError(f"Rigid-body-corrected model state is {value.get('state')!r}, not ready")
    return value


def _model(path, in_fields, group_count, architecture='arch_a'):
    from model_factory import build
    path=Path(path)
    cache_key=(str(path.resolve()),path.stat().st_mtime_ns,architecture,in_fields,group_count)
    if cache_key in _MODEL_CACHE:return _MODEL_CACHE[cache_key]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get('architecture','arch_a') != architecture:
        raise RuntimeError(f'Checkpoint architecture mismatch: {path}')
    model = build(architecture, in_fields, group_count)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    value=model, checkpoint, Path(path)
    _MODEL_CACHE[cache_key]=value
    return value


def load_trepan_branch(branch, architecture='arch_a'):
    value = manifest()
    value = value['trepan2p'] if architecture=='arch_a' else value['architectures'][architecture]['trepan2p']
    path = value[f"{branch}_checkpoint"]
    return _model(path, 9, 4, architecture)


def load_nikon_pair(architecture='arch_a'):
    value = manifest()
    value = value['nikon'] if architecture=='arch_a' else value['architectures'][architecture]['nikon']
    if 'group_checkpoints' in value:
        paths=value['group_checkpoints']
        if set(paths) != {'G1','G2','G3'}:
            raise RuntimeError('All three Nikon group checkpoints are required')
        return tuple(_model(paths[g],3,n,architecture) for g,n in [('G1',3),('G2',3),('G3',1)])
    front = _model(value["front_checkpoint"], 3, 3)
    rear = _model(value["rear_checkpoint"], 3, 4)
    return front, rear


def _normal_image(image):
    return np.asarray(image, dtype=np.float32)


def trepan_predict(branch, observation, architecture='arch_a'):
    model, checkpoint, _ = load_trepan_branch(branch,architecture)
    x = torch.from_numpy(_normal_image(observation)).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        out = model(x)[0].cpu().numpy()
    return out * np.asarray(checkpoint["target_std"]) + np.asarray(checkpoint["target_mean"])


def nikon_predict(raw, architecture='arch_a'):
    from nikon35 import grid
    if isinstance(raw, dict):
        raw = raw["network"]
    image = _normal_image(grid(raw)[0])
    x = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    values = []
    with torch.inference_mode():
        for model, checkpoint, _ in load_nikon_pair(architecture):
            out = model(x)[0].cpu().numpy()
            values.append(out * np.asarray(checkpoint["target_std"]) + np.asarray(checkpoint["target_mean"]))
    return np.concatenate(values)


def nikon_calibration():
    value = manifest()["nikon"]
    center = np.load(value["center_calibration"])["jacobian"]
    capture = np.load(value["capture_calibration"])["jacobian"]
    return center, capture


def smoke():
    value = manifest()
    tf = sorted((Path(ROOT) / "rigid_group_retrain_20260908" / "data" / "trepan2p_front").glob("batch_*.npz"))[0]
    tr = sorted((Path(ROOT) / "rigid_group_retrain_20260908" / "data" / "trepan2p_rear").glob("batch_*.npz"))[0]
    if 'smoke_dataset' in value['nikon']:
        nf=Path(value['nikon']['smoke_dataset'])
    else:
        nf = sorted((Path(ROOT) / "rigid_group_retrain_20260908" / "data" / "nikon_paired").glob("batch_*.npz"))[0]
    with np.load(tf) as data: front = trepan_predict("front", data["speckles"][0])
    with np.load(tr) as data: rear = trepan_predict("rear", data["speckles"][0])
    with np.load(nf) as data: nikon = nikon_predict(data["raw"][0])
    center, capture = nikon_calibration()
    checks = {"manifest": value["version"], "trepan_front_shape": list(front.shape),
              "trepan_rear_shape": list(rear.shape), "nikon_shape": list(nikon.shape),
              "nikon_center_jacobian": list(center.shape), "nikon_capture_jacobian": list(capture.shape),
              "finite": bool(np.isfinite(np.r_[front, rear, nikon]).all())}
    if not checks["finite"] or checks["trepan_front_shape"] != [20] or checks["trepan_rear_shape"] != [20] or checks["nikon_shape"] != [35]:
        raise RuntimeError(checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    if args.smoke: smoke()
