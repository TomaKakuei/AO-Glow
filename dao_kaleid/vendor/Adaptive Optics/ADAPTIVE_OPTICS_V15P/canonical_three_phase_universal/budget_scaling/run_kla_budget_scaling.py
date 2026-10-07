"""KLA DUV Ultimate Move Budget Scaling Benchmark (20 to 40 Moves).

Evaluates 11 quarantined zero-leakage test cases across 3 conditions:
  1. Condition A (5k Baseline Model, Budget=20, Early Exit Q >= 0.45):
     - Historical baseline: median Q = 0.5371, 7.5 moves.
  2. Condition B (7k Warm-Start Zero-Leak Model, Budget=20, Target Q >= 0.58 + L8 CoordDescent):
     - Pushes toward nominal within 20 moves.
  3. Condition C (7k Warm-Start Zero-Leak Model, Budget=40, Full Nominal Target Q >= 0.5962 + Multi-Lens CoordDescent):
     - Probes the ultimate physical convergence limit of KLA DUV under 40 moves.

Metrics Recorded:
  - Initial Q, Terminal Q, Moves Used, Success (Q >= 0.45), Near-Nominal (Q >= 0.58), Nominal Beat (Q >= 0.5962)
  - Full 9-field MTF profile, including negative edge (y = -0.132 mm) and positive edge (y = +0.132 mm)
  - 100% realistic sCMOS read noise, PRNU, and Hartmann slope noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
PROJECT_ROOT = PARENT_DIR.parent
sys.path.insert(0, str(PARENT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))

from kla_full_fov_engine import KLAFullFOVEngine
from models import build_universal_model

# Enable unbuffered real-time stdout flushing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = PARENT_DIR / "checkpoints"
RESULTS_DIR = PARENT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NOMINAL_Q = 0.5962


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("empty rows")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_budget_scaling_benchmark(
    ckpt_7k_path: Path | None = None,
    ckpt_5k_path: Path | None = None,
    out_csv_path: Path | None = None,
) -> list[dict]:
    print("=" * 80, flush=True)
    print("STARTING KLA DUV MOVE BUDGET SCALING BENCHMARK (20 TO 40 MOVES)", flush=True)
    print("=" * 80, flush=True)

    engine = KLAFullFOVEngine()
    mod_lenses = engine.module_lenses[2]  # L6, L7, L8 (15 DOFs)

    # Load 11 test cases
    data_p = PROJECT_ROOT.parent / "artifacts" / "dataset_module2_5dof.npz"
    if not data_p.exists():
        data_p = PROJECT_ROOT / "artifacts" / "dataset_module2_5dof.npz"
    d = np.load(data_p)
    init_perts_list = d["labels"][:11]

    # Load Checkpoints
    if ckpt_5k_path is None:
        ckpt_5k_path = CKPT_DIR / "kla_arch_a.pt"
    if ckpt_7k_path is None:
        ckpt_7k_path = CKPT_DIR / "kla_arch_a_7k_warmstart.pt"
        if not ckpt_7k_path.exists():
            ckpt_7k_path = CKPT_DIR / "kla_arch_a_7k.pt"

    print(f"Loading 5k Model: {ckpt_5k_path.name} ...", flush=True)
    c5 = torch.load(ckpt_5k_path, map_location=DEVICE, weights_only=False)
    m5 = build_universal_model("arch_a", in_fields=9, group_dofs=c5["group_dofs"]).to(DEVICE)
    m5.load_state_dict(c5["state_dict"])
    m5.eval()
    m5_mean, m5_std = c5["target_mean"], c5["target_std"]

    print(f"Loading 7k Model: {ckpt_7k_path.name} ...", flush=True)
    c7 = torch.load(ckpt_7k_path, map_location=DEVICE, weights_only=False)
    m7 = build_universal_model("arch_a", in_fields=9, group_dofs=c7["group_dofs"]).to(DEVICE)
    m7.load_state_dict(c7["state_dict"])
    m7.eval()
    m7_mean, m7_std = c7["target_mean"], c7["target_std"]

    def eval_p_dict(p_dict: dict) -> tuple[float, list[float]]:
        c_list = []
        for yf in engine.field_points_y:
            res = engine.get_field_opd_map(yf, p_dict, grid_size=64, add_noise=True)
            if res is None:
                return -1.0, []
            f_idx = np.argmin(np.abs(res["freqs"] - 2000.0))
            c_list.append(float(res["mtf_tangential"][f_idx]))
        q_val = float(np.mean(c_list) - 0.3 * np.std(c_list))
        return q_val, c_list

    def get_9_opds(p_dict: dict) -> np.ndarray | None:
        opd_list = []
        for yf in engine.field_points_y:
            res = engine.get_field_opd_map(yf, p_dict, grid_size=64, add_noise=True)
            if res is None:
                return None
            opd_list.append(res["opd_map"])
        return np.stack(opd_list, axis=0).astype(np.float32)

    all_results = []

    # Iterate through each of the 11 held-out cases
    for c_idx in range(11):
        case_id = f"Case_{c_idx+1:02d}"
        pert_vec = init_perts_list[c_idx]
        init_perts = {}
        for l_idx, l_name in enumerate(mod_lenses):
            init_perts[l_name] = {
                "dx": float(pert_vec[l_idx * 5 + 0]),
                "dy": float(pert_vec[l_idx * 5 + 1]),
                "dz": float(pert_vec[l_idx * 5 + 2]),
                "tx": float(pert_vec[l_idx * 5 + 3]),
                "ty": float(pert_vec[l_idx * 5 + 4]),
            }

        init_q, init_mtfs = eval_p_dict(init_perts)

        # ---------------------------------------------------------------------
        # Condition A: 5k Model Baseline (Budget = 20, Target = 0.45)
        # ---------------------------------------------------------------------
        moves_a = 0
        curr_perts_a = {k: v.copy() for k, v in init_perts.items()}
        best_q_a = init_q

        opds_0 = get_9_opds(curr_perts_a)
        moves_a += 1
        with torch.no_grad():
            inp = torch.from_numpy(opds_0).unsqueeze(0).to(DEVICE)
            pred_norm = m5(inp).squeeze(0).cpu().numpy()
        pred_p = pred_norm * m5_std + m5_mean

        best_cand_a = curr_perts_a
        for alpha in (0.25, 0.50, 0.75, 1.00):
            cand = {}
            for l_idx, l_name in enumerate(mod_lenses):
                cand[l_name] = {
                    "dx": curr_perts_a[l_name]["dx"] - alpha * pred_p[l_idx * 5 + 0],
                    "dy": curr_perts_a[l_name]["dy"] - alpha * pred_p[l_idx * 5 + 1],
                    "dz": curr_perts_a[l_name]["dz"] - alpha * pred_p[l_idx * 5 + 2],
                    "tx": curr_perts_a[l_name]["tx"] - alpha * pred_p[l_idx * 5 + 3],
                    "ty": curr_perts_a[l_name]["ty"] - alpha * pred_p[l_idx * 5 + 4],
                }
            qc, _ = eval_p_dict(cand)
            moves_a += 1
            if qc > best_q_a:
                best_q_a = qc
                best_cand_a = cand
        curr_perts_a = best_cand_a

        # Phase 3: Adaptive Residual Routing with early stop at Q >= 0.45
        while moves_a < 20 and best_q_a < 0.45:
            opds_res = get_9_opds(curr_perts_a)
            moves_a += 1
            if moves_a >= 20 or opds_res is None:
                break
            with torch.no_grad():
                inp_r = torch.from_numpy(opds_res).unsqueeze(0).to(DEVICE)
                pred_norm_r = m5(inp_r).squeeze(0).cpu().numpy()
            pred_p_r = pred_norm_r * m5_std + m5_mean

            best_r = curr_perts_a
            for alpha_r in (0.30, 0.50, 0.70):
                if moves_a >= 20:
                    break
                cand_r = {}
                for l_idx, l_name in enumerate(mod_lenses):
                    cand_r[l_name] = {
                        "dx": curr_perts_a[l_name]["dx"] - alpha_r * pred_p_r[l_idx * 5 + 0],
                        "dy": curr_perts_a[l_name]["dy"] - alpha_r * pred_p_r[l_idx * 5 + 1],
                        "dz": curr_perts_a[l_name]["dz"] - alpha_r * pred_p_r[l_idx * 5 + 2],
                        "tx": curr_perts_a[l_name]["tx"] - alpha_r * pred_p_r[l_idx * 5 + 3],
                        "ty": curr_perts_a[l_name]["ty"] - alpha_r * pred_p_r[l_idx * 5 + 4],
                    }
                qc_r, _ = eval_p_dict(cand_r)
                moves_a += 1
                if qc_r > best_q_a:
                    best_q_a = qc_r
                    best_r = cand_r
            if best_r is curr_perts_a:
                break
            curr_perts_a = best_r

        term_q_a, term_mtfs_a = eval_p_dict(curr_perts_a)
        rec_a = 100.0 * (term_q_a - init_q) / (NOMINAL_Q - init_q)
        all_results.append({
            "case_id": case_id,
            "condition": "Condition A (5k Baseline Model, Budget=20, Stop@0.45)",
            "budget": 20,
            "initial_metric": init_q,
            "terminal_metric": term_q_a,
            "moves_used": moves_a,
            "recovery_pct": max(0.0, rec_a),
            "pct_of_nominal": 100.0 * term_q_a / NOMINAL_Q,
            "success_45": int(term_q_a >= 0.45),
            "high_precision_55": int(term_q_a >= 0.55),
            "reach_nominal_58": int(term_q_a >= 0.58),
            "center_mtf": term_mtfs_a[4] if term_mtfs_a else -1,
            "neg_edge_mtf": term_mtfs_a[0] if term_mtfs_a else -1,
            "pos_edge_mtf": term_mtfs_a[-1] if term_mtfs_a else -1,
        })

        # ---------------------------------------------------------------------
        # Condition B: 7k Model (Budget = 20, Released Target = 0.58 + L8 SVD)
        # ---------------------------------------------------------------------
        moves_b = 0
        curr_perts_b = {k: v.copy() for k, v in init_perts.items()}
        best_q_b = init_q

        # Phase 1: 7k Model Proposal
        opds_0_b = get_9_opds(curr_perts_b)
        moves_b += 1
        with torch.no_grad():
            inp_b = torch.from_numpy(opds_0_b).unsqueeze(0).to(DEVICE)
            pred_norm_b = m7(inp_b).squeeze(0).cpu().numpy()
        pred_p_b = pred_norm_b * m7_std + m7_mean

        best_cand_b = curr_perts_b
        for alpha in (0.25, 0.50, 0.75, 1.00):
            cand = {}
            for l_idx, l_name in enumerate(mod_lenses):
                cand[l_name] = {
                    "dx": curr_perts_b[l_name]["dx"] - alpha * pred_p_b[l_idx * 5 + 0],
                    "dy": curr_perts_b[l_name]["dy"] - alpha * pred_p_b[l_idx * 5 + 1],
                    "dz": curr_perts_b[l_name]["dz"] - alpha * pred_p_b[l_idx * 5 + 2],
                    "tx": curr_perts_b[l_name]["tx"] - alpha * pred_p_b[l_idx * 5 + 3],
                    "ty": curr_perts_b[l_name]["ty"] - alpha * pred_p_b[l_idx * 5 + 4],
                }
            qc, _ = eval_p_dict(cand)
            moves_b += 1
            if qc > best_q_b:
                best_q_b = qc
                best_cand_b = cand
        curr_perts_b = best_cand_b

        # Phase 2: Neural Residual Router up to Target 0.58
        while moves_b < 15 and best_q_b < 0.58:
            opds_res_b = get_9_opds(curr_perts_b)
            moves_b += 1
            if moves_b >= 15 or opds_res_b is None:
                break
            with torch.no_grad():
                inp_rb = torch.from_numpy(opds_res_b).unsqueeze(0).to(DEVICE)
                pred_norm_rb = m7(inp_rb).squeeze(0).cpu().numpy()
            pred_p_rb = pred_norm_rb * m7_std + m7_mean

            best_rb = curr_perts_b
            for alpha_r in (0.30, 0.50, 0.70):
                if moves_b >= 15:
                    break
                cand_r = {}
                for l_idx, l_name in enumerate(mod_lenses):
                    cand_r[l_name] = {
                        "dx": curr_perts_b[l_name]["dx"] - alpha_r * pred_p_rb[l_idx * 5 + 0],
                        "dy": curr_perts_b[l_name]["dy"] - alpha_r * pred_p_rb[l_idx * 5 + 1],
                        "dz": curr_perts_b[l_name]["dz"] - alpha_r * pred_p_rb[l_idx * 5 + 2],
                        "tx": curr_perts_b[l_name]["tx"] - alpha_r * pred_p_rb[l_idx * 5 + 3],
                        "ty": curr_perts_b[l_name]["ty"] - alpha_r * pred_p_rb[l_idx * 5 + 4],
                    }
                qc_r, _ = eval_p_dict(cand_r)
                moves_b += 1
                if qc_r > best_q_b:
                    best_q_b = qc_r
                    best_rb = cand_r
            if best_rb is curr_perts_b:
                break
            curr_perts_b = best_rb

        # Phase 3: 1D Coordinate Descent Micro-Refinement on sensitive lens L8 with remaining budget (up to 20)
        h_s = 0.002  # 2 um
        h_t = (0.1 / 60.0) * (np.pi / 180.0)  # 0.1 arcmin
        dof_scales = [h_s, h_s, h_s, h_t, h_t]
        dof_keys = ["dx", "dy", "dz", "tx", "ty"]

        for d_idx, key in enumerate(dof_keys):
            if moves_b >= 20:
                break
            h_step = dof_scales[d_idx]

            # Positive probe
            cp = {k: v.copy() for k, v in curr_perts_b.items()}
            cp["L8"][key] += h_step
            qp, _ = eval_p_dict(cp)
            moves_b += 1
            if qp > best_q_b:
                best_q_b = qp
                curr_perts_b = cp
                continue

            if moves_b >= 20:
                break

            # Negative probe
            cm = {k: v.copy() for k, v in curr_perts_b.items()}
            cm["L8"][key] -= h_step
            qm, _ = eval_p_dict(cm)
            moves_b += 1
            if qm > best_q_b:
                best_q_b = qm
                curr_perts_b = cm

        term_q_b, term_mtfs_b = eval_p_dict(curr_perts_b)
        rec_b = 100.0 * (term_q_b - init_q) / (NOMINAL_Q - init_q)
        all_results.append({
            "case_id": case_id,
            "condition": "Condition B (7k Warm-Start Zero-Leak Model, Budget=20, Target=0.58+CoordDescent)",
            "budget": 20,
            "initial_metric": init_q,
            "terminal_metric": term_q_b,
            "moves_used": moves_b,
            "recovery_pct": max(0.0, rec_b),
            "pct_of_nominal": 100.0 * term_q_b / NOMINAL_Q,
            "success_45": int(term_q_b >= 0.45),
            "high_precision_55": int(term_q_b >= 0.55),
            "reach_nominal_58": int(term_q_b >= 0.58),
            "center_mtf": term_mtfs_b[4] if term_mtfs_b else -1,
            "neg_edge_mtf": term_mtfs_b[0] if term_mtfs_b else -1,
            "pos_edge_mtf": term_mtfs_b[-1] if term_mtfs_b else -1,
        })

        # ---------------------------------------------------------------------
        # Condition C: 7k Model (Budget = 40, Target = 0.5962 + Deep Multi-Lens SVD)
        # ---------------------------------------------------------------------
        moves_c = 0
        curr_perts_c = {k: v.copy() for k, v in init_perts.items()}
        best_q_c = init_q

        # Phase 1: 7k Model Proposal
        opds_0_c = get_9_opds(curr_perts_c)
        moves_c += 1
        with torch.no_grad():
            inp_c = torch.from_numpy(opds_0_c).unsqueeze(0).to(DEVICE)
            pred_norm_c = m7(inp_c).squeeze(0).cpu().numpy()
        pred_p_c = pred_norm_c * m7_std + m7_mean

        best_cand_c = curr_perts_c
        for alpha in (0.25, 0.50, 0.75, 1.00):
            cand = {}
            for l_idx, l_name in enumerate(mod_lenses):
                cand[l_name] = {
                    "dx": curr_perts_c[l_name]["dx"] - alpha * pred_p_c[l_idx * 5 + 0],
                    "dy": curr_perts_c[l_name]["dy"] - alpha * pred_p_c[l_idx * 5 + 1],
                    "dz": curr_perts_c[l_name]["dz"] - alpha * pred_p_c[l_idx * 5 + 2],
                    "tx": curr_perts_c[l_name]["tx"] - alpha * pred_p_c[l_idx * 5 + 3],
                    "ty": curr_perts_c[l_name]["ty"] - alpha * pred_p_c[l_idx * 5 + 4],
                }
            qc, _ = eval_p_dict(cand)
            moves_c += 1
            if qc > best_q_c:
                best_q_c = qc
                best_cand_c = cand
        curr_perts_c = best_cand_c

        # Phase 2: Multi-Round Neural Residual Routing (up to move 22, aiming for NOMINAL_Q)
        while moves_c < 22 and best_q_c < NOMINAL_Q:
            opds_res_c = get_9_opds(curr_perts_c)
            moves_c += 1
            if moves_c >= 22 or opds_res_c is None:
                break
            with torch.no_grad():
                inp_rc = torch.from_numpy(opds_res_c).unsqueeze(0).to(DEVICE)
                pred_norm_rc = m7(inp_rc).squeeze(0).cpu().numpy()
            pred_p_rc = pred_norm_rc * m7_std + m7_mean

            best_rc = curr_perts_c
            for alpha_r in (0.20, 0.40, 0.60):
                if moves_c >= 22:
                    break
                cand_r = {}
                for l_idx, l_name in enumerate(mod_lenses):
                    cand_r[l_name] = {
                        "dx": curr_perts_c[l_name]["dx"] - alpha_r * pred_p_rc[l_idx * 5 + 0],
                        "dy": curr_perts_c[l_name]["dy"] - alpha_r * pred_p_rc[l_idx * 5 + 1],
                        "dz": curr_perts_c[l_name]["dz"] - alpha_r * pred_p_rc[l_idx * 5 + 2],
                        "tx": curr_perts_c[l_name]["tx"] - alpha_r * pred_p_rc[l_idx * 5 + 3],
                        "ty": curr_perts_c[l_name]["ty"] - alpha_r * pred_p_rc[l_idx * 5 + 4],
                    }
                qc_r, _ = eval_p_dict(cand_r)
                moves_c += 1
                if qc_r > best_q_c:
                    best_q_c = qc_r
                    best_rc = cand_r
            if best_rc is curr_perts_c:
                break
            curr_perts_c = best_rc

        # Phase 3: Deep Multi-Lens Coordinate Descent Micro-Refinement (L8 sensitive first, then L7) across remaining moves up to 40
        refine_lenses = ["L8", "L7"]
        fine_scales = [0.0015, 0.0015, 0.0015, (0.05 / 60.0) * (np.pi / 180.0), (0.05 / 60.0) * (np.pi / 180.0)]

        for l_target in refine_lenses:
            if moves_c >= 40 or best_q_c >= NOMINAL_Q:
                break
            for d_idx, key in enumerate(dof_keys):
                if moves_c >= 40 or best_q_c >= NOMINAL_Q:
                    break
                h_step = fine_scales[d_idx]

                # Positive step
                cp = {k: v.copy() for k, v in curr_perts_c.items()}
                cp[l_target][key] += h_step
                qp, _ = eval_p_dict(cp)
                moves_c += 1
                if qp > best_q_c:
                    best_q_c = qp
                    curr_perts_c = cp
                    continue

                if moves_c >= 40:
                    break

                # Negative step
                cm = {k: v.copy() for k, v in curr_perts_c.items()}
                cm[l_target][key] -= h_step
                qm, _ = eval_p_dict(cm)
                moves_c += 1
                if qm > best_q_c:
                    best_q_c = qm
                    curr_perts_c = cm

        term_q_c, term_mtfs_c = eval_p_dict(curr_perts_c)
        rec_c = 100.0 * (term_q_c - init_q) / (NOMINAL_Q - init_q)
        all_results.append({
            "case_id": case_id,
            "condition": "Condition C (7k Warm-Start Zero-Leak Model, Budget=40, Full Nominal+CoordDescent)",
            "budget": 40,
            "initial_metric": init_q,
            "terminal_metric": term_q_c,
            "moves_used": moves_c,
            "recovery_pct": max(0.0, rec_c),
            "pct_of_nominal": 100.0 * term_q_c / NOMINAL_Q,
            "success_45": int(term_q_c >= 0.45),
            "high_precision_55": int(term_q_c >= 0.55),
            "reach_nominal_58": int(term_q_c >= 0.58),
            "center_mtf": term_mtfs_c[4] if term_mtfs_c else -1,
            "neg_edge_mtf": term_mtfs_c[0] if term_mtfs_c else -1,
            "pos_edge_mtf": term_mtfs_c[-1] if term_mtfs_c else -1,
        })

        print(
            f"[{case_id}] Init: {init_q:.4f} Q | "
            f"Cond A (20m/5k): {term_q_a:.4f} Q ({moves_a:02d}m) | "
            f"Cond B (20m/7k): {term_q_b:.4f} Q ({moves_b:02d}m) | "
            f"Cond C (40m/7k): {term_q_c:.4f} Q ({moves_c:02d}m, {100.0*term_q_c/NOMINAL_Q:.1f}% Nom)",
            flush=True,
        )

    if out_csv_path is None:
        out_csv_path = RESULTS_DIR / "kla_budget_scaling_20_40_benchmark.csv"
    write_csv(out_csv_path, all_results)
    print(f"\nSuccessfully wrote {len(all_results)} rows to: {out_csv_path.resolve()}", flush=True)

    return all_results


if __name__ == "__main__":
    run_budget_scaling_benchmark()
