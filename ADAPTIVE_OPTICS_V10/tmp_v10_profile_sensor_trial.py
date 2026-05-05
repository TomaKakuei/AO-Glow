from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ["AO_V3_ENABLE_TIPTILT"] = "1"

from ao_v10_backend import RunConfig, run_repair


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts" / "v10_profile_sensor_trial"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260504)
CASE_COUNT = 5


def _sample_global_tt_deg() -> tuple[float, float]:
    magnitude_arcmin = float(RNG.uniform(0.0, 3.0))
    angle = float(RNG.uniform(0.0, 2.0 * math.pi))
    tx_arcmin = magnitude_arcmin * math.cos(angle)
    ty_arcmin = magnitude_arcmin * math.sin(angle)
    return tx_arcmin / 60.0, ty_arcmin / 60.0


def _sample_case(index: int) -> dict[str, object]:
    overall_variation_mm = float(RNG.uniform(0.0, 0.1))
    tilt_x_deg, tilt_y_deg = _sample_global_tt_deg()
    overrides = {
        anchor_id: {
            "tilt_x_deg": tilt_x_deg,
            "tilt_y_deg": tilt_y_deg,
        }
        for anchor_id in (68, 70, 72, 74, 77, 79, 81, 84)
    }
    return {
        "case_index": index,
        "overall_variation_mm": overall_variation_mm,
        "global_tilt_x_deg": tilt_x_deg,
        "global_tilt_y_deg": tilt_y_deg,
        "global_tilt_radius_arcmin": float((tilt_x_deg**2 + tilt_y_deg**2) ** 0.5 * 60.0),
        "per_anchor_overrides": overrides,
    }


def _run_profile_case(profile: str, case_payload: dict[str, object]) -> dict[str, object]:
    config = RunConfig(
        overall_variation_mm=float(case_payload["overall_variation_mm"]),
        per_anchor_overrides=case_payload["per_anchor_overrides"],  # type: ignore[arg-type]
        output_name=f"v10_{profile}_trial_case{int(case_payload['case_index']):02d}",
        shwfs_noise_profile=str(profile),
        shwfs_forward_averages=1,
        optimizer_maxiter_limit=200,
    )
    result = run_repair(config, log=lambda _message: None)
    summary = json.loads(Path(result["repair_summary_path"]).read_text(encoding="utf-8"))
    repaired = summary["repaired"]
    moved = summary["moved"]
    config_summary = summary.get("configuration", {})
    return {
        "profile": profile,
        "case_index": int(case_payload["case_index"]),
        "overall_variation_mm": float(case_payload["overall_variation_mm"]),
        "global_tilt_x_deg": float(case_payload["global_tilt_x_deg"]),
        "global_tilt_y_deg": float(case_payload["global_tilt_y_deg"]),
        "global_tilt_radius_arcmin": float(case_payload["global_tilt_radius_arcmin"]),
        "sensor_lenslet_count": int(config_summary.get("shwfs_resolution", {}).get("lenslet_count", -1)),
        "sensor_slope_channels": int(config_summary.get("shwfs_resolution", {}).get("slope_channels", -1)),
        "nominal_residual": float(summary["nominal"]["residual_norm"]),
        "before_residual": float(moved["residual_norm"]),
        "repaired_residual": float(repaired["residual_norm"]),
        "before_true_residual": float(moved["true_residual_norm"]),
        "repaired_true_residual": float(repaired["true_residual_norm"]),
        "before_wrms": float(moved["wavefront_rms_waves"]),
        "repaired_wrms": float(repaired["wavefront_rms_waves"]),
        "before_strehl": float(moved["strehl_ratio"]),
        "repaired_strehl": float(repaired["strehl_ratio"]),
        "run_dir": str(result["run_dir"]),
        "repair_summary_path": str(result["repair_summary_path"]),
    }


def main() -> None:
    profiles = ["realistic", "minimal", "hard"]
    cases = [_sample_case(index + 1) for index in range(CASE_COUNT)]
    rows: list[dict[str, object]] = []
    csv_path = ARTIFACTS_DIR / "results.csv"
    json_path = ARTIFACTS_DIR / "results.json"
    cases_path = ARTIFACTS_DIR / "cases.json"

    def _save() -> None:
        with cases_path.open("w", encoding="utf-8") as handle:
            json.dump(cases, handle, indent=2)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    if json_path.exists():
        try:
            rows = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    completed = {(str(row["profile"]), int(row["case_index"])) for row in rows}

    for case_payload in cases:
        for profile in profiles:
            if (profile, int(case_payload["case_index"])) in completed:
                continue
            rows.append(_run_profile_case(profile, case_payload))
            _save()

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {cases_path}")


if __name__ == "__main__":
    main()
