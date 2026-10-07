"""Existing-array baselines for the ablation_and_new_results controller.

No optical engine or optical-data generator is imported here.
Slor architecture: five 2048-wide hidden layers, ReLU, skips every two layers.
The optical observation input is adapted to this project's multi-field maps.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import griddata
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANON = ROOT / "canonical_three_phase_universal"
OUT = HERE / "results"
CHECKPOINTS = HERE / "checkpoints"
sys.path.insert(0, str(CANON))
from models import build_universal_model

SYSTEMS = {"nikon": (3, [5] * 6), "kla": (9, [5] * 3), "trepan2p": (9, [5] * 4)}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pool_maps(x):
    return F.adaptive_avg_pool2d(x, (16, 16)).flatten(1)


class SlorMLP(nn.Module):
    def __init__(self, fields, outputs):
        super().__init__()
        dim = fields * 16 * 16
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_std", torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, 2048)] + [nn.Linear(2048, 2048) for _ in range(4)])
        self.output = nn.Linear(2048, outputs)

    def forward_features(self, x):
        x = (x - self.feature_mean) / self.feature_std
        x = F.relu(self.layers[0](x))
        for first in (1, 3):
            skip = x
            x = F.relu(self.layers[first](x))
            x = F.relu(self.layers[first + 1](x) + skip)
        return self.output(x)

    def forward(self, x):
        return self.forward_features(pool_maps(x))


class SensitivitySVD(nn.Module):
    """Calibrated forward sensitivity and damped singular-value inverse."""
    def __init__(self, fields, outputs):
        super().__init__()
        dim = fields * 16 * 16
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_std", torch.ones(dim))
        self.register_buffer("sensitivity", torch.zeros(dim, outputs))
        self.register_buffer("inverse", torch.zeros(outputs, dim))
        self.register_buffer("intercept", torch.zeros(dim))

    def forward_features(self, x):
        x = (x - self.feature_mean) / self.feature_std
        return (x - self.intercept) @ self.inverse.T

    def forward(self, x):
        return self.forward_features(pool_maps(x))


def checkpoint_path(system, method):
    if method in ("slor_mlp", "sensitivity_svd"):
        return CHECKPOINTS / f"{system}_{method}.pt"
    return CANON / "checkpoints" / f"{system}_{method}.pt"


def build_proposal(system, method, checkpoint=None):
    fields, groups = SYSTEMS[system]
    if method == "slor_mlp":
        model = SlorMLP(fields, sum(groups))
    elif method == "sensitivity_svd":
        model = SensitivitySVD(fields, sum(groups))
    else:
        model = build_universal_model(method, fields, groups)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["state_dict"])
    return model


def load_pooled_arrays(system):
    """Read archived arrays in batches; cache only their spatial averages."""
    cache = HERE / "cache" / f"{system}_pooled16.npz"
    if cache.exists():
        with np.load(cache) as d:
            return d["features"], d["targets"], json.loads(str(d["source_json"]))
    t0 = time.perf_counter()
    xs, ys, sources = [], [], []
    if system == "trepan2p":
        paths = sorted((ROOT / "artifacts/front_speckle_dataset").glob("batch_*.npz"))[:50]
        if len(paths) != 50:
            raise ValueError("Expected existing 50 Trepan2p batches")
        for p in paths:
            with np.load(p) as d:
                x = torch.from_numpy(d["speckles"].astype(np.float32))
                xs.append(pool_maps(x).numpy())
                ys.append(d["mechs"].astype(np.float32))
            sources.append(str(p.relative_to(ROOT)))
    elif system == "kla":
        p = ROOT.parent / "artifacts/dataset_module2_5dof.npz"
        if not p.exists():
            p = ROOT / "artifacts/dataset_module2_5dof.npz"
        with np.load(p) as d:
            maps = d["opd_maps"].astype(np.float32)
            ys.append(d["labels"].astype(np.float32))
        for start in range(0, len(maps), 100):
            xs.append(pool_maps(torch.from_numpy(maps[start:start + 100])).numpy())
        sources.append(str(p))
    else:
        p = ROOT / "eval_1p4na_realistic_v4/results_repaired_multigroup_30case_nn_linear_local_audit/training/capture_dataset_3000.npz"
        with np.load(p) as d:
            raw = d["features"].astype(np.float32).reshape(-1, 3, 49)
            nodes = d["feedback_pupil"]
            ys.append(d["targets"].astype(np.float32))
        gx, gy = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
        mask = (gx * gx + gy * gy <= 1).astype(np.float32)
        # Linear interpolation is a fixed matrix: exactly the existing griddata convention.
        matrix = griddata(nodes, np.eye(49), (gx, gy), method="linear", fill_value=0) * mask[..., None]
        for start in range(0, len(raw), 100):
            maps = np.einsum("nfk,hwk->nfhw", raw[start:start + 100], matrix).astype(np.float32)
            xs.append(pool_maps(torch.from_numpy(maps)).numpy())
        sources.append(str(p.relative_to(ROOT)))
    x, y = np.concatenate(xs), np.concatenate(ys)
    source = {"files": sources, "states": len(y), "pool_grid": [16, 16], "derived_seconds": time.perf_counter() - t0}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, features=x, targets=y, source_json=json.dumps(source))
    return x, y, source


def prepare_baselines(system, epochs=25, max_train_seconds=300):
    paths = [checkpoint_path(system, m) for m in ("sensitivity_svd", "slor_mlp")]
    if all(p.exists() for p in paths):
        print(f"REUSE {system} fitted baselines", flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(20260905)
    np.random.seed(20260905)
    x, y, source = load_pooled_arrays(system)
    split = int(len(y) * .8)
    # KLA's eleven evaluation states are rows 0:11 in the source controller.
    first = 11 if system == "kla" else 0
    train_idx = np.arange(first, split)
    val_idx = np.arange(split, len(y))
    xm, xs = x[train_idx].mean(0), x[train_idx].std(0)
    xs[xs < 1e-6] = 1
    ym, ys = y[train_idx].mean(0), y[train_idx].std(0)
    ys[ys < 1e-6] = 1
    yn = (y - ym) / ys
    common = {"system": system, "group_dofs": SYSTEMS[system][1], "target_mean": ym, "target_std": ys,
              "train_indices": train_idx, "validation_indices": val_idx, "source": source,
              "training_seed": 20260905, "input_adapter": "existing maps averaged to 16x16 per field"}
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if not paths[0].exists():
        t0 = time.perf_counter()
        xf = (x - xm) / xs
        design = np.column_stack([yn[train_idx], np.ones(len(train_idx))]).astype(np.float64)
        coefficients = np.linalg.lstsq(design, xf[train_idx], rcond=1e-8)[0]
        jac, intercept = coefficients[:-1].T, coefficients[-1]
        u, s, vt = np.linalg.svd(jac, full_matrices=False)
        best = None
        # Damping is selected on the archived validation partition, never on the 11 cases.
        for fraction in (1e-4, 1e-3, 1e-2, .1, 1.):
            damping = fraction * s[0]
            inv = (vt.T * (s / (s * s + damping * damping))) @ u.T
            pred = (xf[val_idx] - intercept) @ inv.T
            rmse = float(np.sqrt(np.mean((pred - yn[val_idx]) ** 2)))
            if best is None or rmse < best[0]:
                best = rmse, fraction, inv
        model = build_proposal(system, "sensitivity_svd")
        for key, value in {"feature_mean": xm, "feature_std": xs, "sensitivity": jac,
                           "inverse": best[2], "intercept": intercept}.items():
            getattr(model, key).copy_(torch.as_tensor(value, dtype=torch.float32))
        meta = {"method": "Calibrated sensitivity-matrix SVD", "neural_training_epochs": 0,
                "calibration_states": len(train_idx), "validation_states": len(val_idx),
                "validation_pose_rmse_normalized": best[0], "damping_fraction": best[1],
                "fit_seconds": time.perf_counter() - t0, "singular_values": s.tolist()}
        torch.save(dict(common, state_dict=model.state_dict(), metadata=meta), paths[0])
        save_json(OUT / f"{system}_sensitivity_svd_fit.json", meta)
        print(f"FIT {system} sensitivity SVD {meta['fit_seconds']:.1f}s", flush=True)
    if paths[1].exists():
        return
    model = build_proposal(system, "slor_mlp").to(DEVICE)
    model.feature_mean.copy_(torch.from_numpy(xm).to(DEVICE))
    model.feature_std.copy_(torch.from_numpy(xs).to(DEVICE))
    tx = torch.from_numpy(x[train_idx]).to(DEVICE)
    ty = torch.from_numpy(yn[train_idx]).to(DEVICE)
    vx = torch.from_numpy(x[val_idx]).to(DEVICE)
    vy = torch.from_numpy(yn[val_idx]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=10)
    best_loss, best_weights, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(tx), device=DEVICE)
        for start in range(0, len(order), 250):
            idx = order[start:start + 250]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.forward_features(tx[idx]), ty[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            pred = torch.cat([model.forward_features(vx[i:i + 250]) for i in range(0, len(vx), 250)])
            val_loss = float(F.mse_loss(pred, vy))
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "validation_mse": val_loss, "elapsed_seconds": time.perf_counter() - started})
        if epoch == 1 or epoch % 5 == 0:
            print(f"TRAIN {system} Slor MLP {epoch}/{epochs} validation MSE={val_loss:.6f}", flush=True)
        if time.perf_counter() - started >= max_train_seconds:
            break
    meta = {"method": "Slor-style residual MLP", "paper": "https://arxiv.org/abs/2506.23173",
            "hidden_layers": 5, "hidden_width": 2048, "activation": "ReLU", "skip_every_hidden_layers": 2,
            "optimizer": "AdamW", "learning_rate": 1e-5, "batch_size": 250, "weight_decay": .01,
            "paper_epochs": 500, "epochs_run": len(history), "epoch_cap": epochs,
            "training_states": len(train_idx), "validation_states": len(val_idx),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "validation_pose_rmse_normalized": best_loss ** .5,
            "fit_seconds": time.perf_counter() - started, "history": history}
    torch.save(dict(common, state_dict=best_weights, metadata=meta), paths[1])
    save_json(OUT / f"{system}_slor_mlp_fit.json", meta)
    print(f"SAVED {system} Slor MLP {meta['fit_seconds']:.1f}s", flush=True)
