from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
PYTHON_EXE = Path.home() / "anaconda3" / "envs" / "ao311" / "python.exe"
OUTPUT_DIR = ROOT / "artifacts" / "v10_vs_v10_1_mode12_case0100_tt3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CASE = {
    "overall_variation_mm": 0.1,
    "per_anchor_overrides": {
        surface_id: {
            "tilt_x_deg": 3.0 / 60.0,
            "tilt_y_deg": 3.0 / 60.0,
        }
        for surface_id in [68, 70, 72, 74, 77, 79, 81, 84]
    },
}

VERSIONS = [
    {
        "name": "v10",
        "folder": PROJECT_ROOT / "ADAPTIVE_OPTICS_V10",
        "backend_module": "ao_v10_backend",
        "num_modes": 8,
    },
    {
        "name": "v10_1",
        "folder": PROJECT_ROOT / "ADAPTIVE_OPTICS_V10_1",
        "backend_module": "ao_v10_1_backend",
        "num_modes": 12,
    },
]

PROFILES = ["realistic", "minimal", "hard"]


def _run_one(version: dict[str, object], profile: str) -> dict[str, object]:
    folder = Path(version["folder"])
    backend_module = str(version["backend_module"])
    output_name = f"{version['name']}_{profile}_case0100_tt3"
    payload = {
        "overall_variation_mm": CASE["overall_variation_mm"],
        "per_anchor_overrides": CASE["per_anchor_overrides"],
        "output_name": output_name,
        "shwfs_noise_profile": profile,
    }
    inline = f"""
import json, os
os.environ["AO_V3_ENABLE_TIPTILT"] = "1"
from {backend_module} import RunConfig, run_repair
payload = {json.dumps(payload)}
cfg = RunConfig(
    overall_variation_mm=float(payload["overall_variation_mm"]),
    per_anchor_overrides=payload["per_anchor_overrides"],
    output_name=str(payload["output_name"]),
    shwfs_noise_profile=str(payload["shwfs_noise_profile"]),
)
result = run_repair(cfg, log=lambda message: None)
summary = {{
    "run_dir": result.get("run_dir", ""),
    "repair_summary_path": result.get("repair_summary_path", ""),
    "summary_text": result.get("summary_text", ""),
}}
print(json.dumps(summary, ensure_ascii=True))
"""
    started = time.perf_counter()
    completed = subprocess.run(
        [str(PYTHON_EXE), "-c", inline],
        cwd=str(folder),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    elapsed = float(time.perf_counter() - started)
    last_line = completed.stdout.strip().splitlines()[-1]
    summary = json.loads(last_line)
    summary_path = Path(summary["repair_summary_path"])
    payload_json = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "version": str(version["name"]),
        "profile": profile,
        "elapsed_s": elapsed,
        "run_dir": str(summary["run_dir"]),
        "repair_summary_path": str(summary_path),
        "repaired_residual": float(payload_json["repaired"]["residual_norm"]),
        "repaired_true_residual": float(payload_json["repaired"]["true_residual_norm"]),
        "repaired_wrms": float(payload_json["repaired"]["wavefront_rms_waves"]),
        "repaired_strehl": float(payload_json["repaired"]["strehl_ratio"]),
        "nominal_wrms": float(payload_json["nominal"]["wavefront_rms_waves"]),
        "num_modes": int(version["num_modes"]),
        "slope_channels": int(payload_json["configuration"]["shwfs_slope_channels"]),
        "lenslet_count": int(payload_json["configuration"]["shwfs_lenslet_count"]),
    }


def main() -> int:
    results_json_path = OUTPUT_DIR / "results.json"
    if results_json_path.exists():
        rows = json.loads(results_json_path.read_text(encoding="utf-8"))
    else:
        rows = []
    completed = {(str(row["version"]), str(row["profile"])) for row in rows}
    for version in VERSIONS:
        for profile in PROFILES:
            key = (str(version["name"]), profile)
            if key in completed:
                print(f"Skipping {version['name']} / {profile} (already complete)...", flush=True)
                continue
            print(f"Running {version['name']} / {profile}...", flush=True)
            row = _run_one(version, profile)
            rows.append(row)
            completed.add(key)
            results_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "version,profile,elapsed_s,num_modes,lenslet_count,slope_channels,repaired_residual,repaired_true_residual,repaired_wrms,repaired_strehl,nominal_wrms,repair_summary_path"
    ]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row["version"]),
                    str(row["profile"]),
                    f"{float(row['elapsed_s']):.3f}",
                    str(row["num_modes"]),
                    str(row["lenslet_count"]),
                    str(row["slope_channels"]),
                    f"{float(row['repaired_residual']):.9f}",
                    f"{float(row['repaired_true_residual']):.9f}",
                    f"{float(row['repaired_wrms']):.9f}",
                    f"{float(row['repaired_strehl']):.9f}",
                    f"{float(row['nominal_wrms']):.9f}",
                    str(row["repair_summary_path"]),
                ]
            )
        )
    (OUTPUT_DIR / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
