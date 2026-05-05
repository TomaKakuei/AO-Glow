from __future__ import annotations

import importlib
import json
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "artifacts" / "v5_avg4_nominal_minimum_vs_realistic_two_cases.json"
ACTUATOR_IDS = (68, 70, 72, 74, 77, 79, 81, 84)


def _case_specs() -> list[tuple[int, float, float]]:
    rng = random.Random(20260502 + 2205)
    specs: list[tuple[int, float, float]] = []
    for case_index in range(1, 11):
        specs.append((case_index, rng.uniform(0.08, 0.10), rng.uniform(1.0, 5.0)))
    return specs


def _build_overrides(tt_arcmin: float) -> dict[int, dict[str, float]]:
    tilt_deg = tt_arcmin / 60.0
    return {
        anchor_id: {
            "tilt_x_deg": tilt_deg,
            "tilt_y_deg": tilt_deg,
        }
        for anchor_id in ACTUATOR_IDS
    }


def _configure_env(tt_arcmin: float, calibration_profile: str) -> None:
    tt_box_arcmin = 2.0 * tt_arcmin
    os.environ["AO_V3_ENABLE_TIPTILT"] = "1"
    os.environ["AO_V3_DISABLE_BLIND_SEED"] = "1"
    os.environ["AO_V5_SHWFS_NOISE_PROFILE"] = "realistic"
    os.environ["AO_V5_SHWFS_CALIBRATION_NOISE_PROFILE"] = calibration_profile
    os.environ["AO_V5_SHWFS_FORWARD_AVERAGES"] = "4"
    os.environ["AO_V5_SEARCH_BOX_XY_MM"] = "0.22"
    os.environ["AO_V5_SEARCH_BOX_Z_MM"] = "0.15"
    os.environ["AO_V5_TIPTILT_TEST_LIMIT_ARCMIN"] = f"{tt_box_arcmin:.6f}"
    os.environ["AO_V3_TIPTILT_LIMIT_ARCMIN"] = f"{tt_box_arcmin:.6f}"


def _run_case(case_index: int, overall_variation_mm: float, tt_arcmin: float, calibration_profile: str) -> dict:
    _configure_env(tt_arcmin, calibration_profile)
    for module_name in [
        "benchmark_freeform_real_shwfs_residual_120",
        "benchmark_1mm_real_shwfs_converging_920nm",
        "ao_v5_backend",
    ]:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
    if "ao_v5_backend" in sys.modules:
        ao_v5_backend = sys.modules["ao_v5_backend"]
    else:
        import ao_v5_backend  # type: ignore
    RunConfig = ao_v5_backend.RunConfig
    run_repair = ao_v5_backend.run_repair
    config = RunConfig(
        overall_variation_mm=overall_variation_mm,
        per_anchor_overrides=_build_overrides(tt_arcmin),
        output_name=f"v5_avg4_case{case_index:02d}_{calibration_profile}_nominal_compare",
        shwfs_noise_profile="realistic",
        shwfs_forward_averages=4,
    )
    result = run_repair(config)
    summary = json.loads(Path(result["repair_summary_path"]).read_text(encoding="utf-8"))
    return {
        "run_dir": result["run_dir"],
        "before_true_residual": summary["moved"]["true_residual_norm"],
        "repaired_true_residual": summary["repaired"]["true_residual_norm"],
        "before_wrms_waves": summary["moved"]["wavefront_rms_waves"],
        "repaired_wrms_waves": summary["repaired"]["wavefront_rms_waves"],
        "before_residual": summary["moved"]["residual_norm"],
        "repaired_residual": summary["repaired"]["residual_norm"],
        "runtime_s": summary["runtime_s"],
        "ray_tracing_runtime_s": summary["ray_tracing_runtime_s"],
        "shwfs_noise_profile": summary["configuration"]["shwfs_noise_profile"],
    }


def main() -> int:
    selected_cases = {2, 10}
    rows: list[dict] = []
    for case_index, overall_variation_mm, tt_arcmin in _case_specs():
        if case_index not in selected_cases:
            continue
        entry = {
            "case_index": case_index,
            "overall_variation_mm": overall_variation_mm,
            "tt_arcmin": tt_arcmin,
            "tt_box_arcmin": 2.0 * tt_arcmin,
            "realistic_nominal": _run_case(case_index, overall_variation_mm, tt_arcmin, "realistic"),
            "minimum_nominal": _run_case(case_index, overall_variation_mm, tt_arcmin, "minimum"),
        }
        rows.append(entry)
    OUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved_json={OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
