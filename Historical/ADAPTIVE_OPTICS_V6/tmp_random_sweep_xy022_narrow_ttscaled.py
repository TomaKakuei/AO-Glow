from __future__ import annotations

import csv
import importlib
import json
import os
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CSV_PATH = ARTIFACTS_DIR / "v6_random_xy022_narrow_ttscaled_10cases.csv"
JSON_PATH = ARTIFACTS_DIR / "v6_random_xy022_narrow_ttscaled_10cases.json"
ACTUATOR_IDS = (68, 70, 72, 74, 77, 79, 81, 84)


def _case_specs() -> list[tuple[int, float, float]]:
    rng = random.Random(20260502 + 2205)
    specs: list[tuple[int, float, float]] = []
    for case_index in range(1, 11):
        specs.append((case_index, rng.uniform(0.08, 0.10), rng.uniform(1.0, 5.0)))
    return specs


def _load_existing_rows() -> list[dict[str, float | int | str]]:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text(encoding="utf-8-sig"))
    return []


def _save_rows(rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _parse_range_args() -> tuple[int, int]:
    start_index = 1
    end_index = 10
    if len(sys.argv) >= 2:
        start_index = max(1, int(sys.argv[1]))
    if len(sys.argv) >= 3:
        end_index = min(10, int(sys.argv[2]))
    if start_index > end_index:
        raise ValueError(f"invalid range: {start_index}..{end_index}")
    return start_index, end_index


def _build_overrides(tt_arcmin: float) -> dict[int, dict[str, float]]:
    tilt_deg = tt_arcmin / 60.0
    return {
        anchor_id: {
            "tilt_x_deg": tilt_deg,
            "tilt_y_deg": tilt_deg,
        }
        for anchor_id in ACTUATOR_IDS
    }


def _tt_stats(summary: dict) -> tuple[float, float, float]:
    tt_initial = 0.0
    tt_final = 0.0
    for anchor_id in map(str, ACTUATOR_IDS):
        moved = summary["moved_position_by_anchor"][anchor_id]
        final = summary["final_position_by_anchor"][anchor_id]
        tt_initial += (abs(float(moved["tilt_x_deg"])) + abs(float(moved["tilt_y_deg"]))) / 2.0
        tt_final += (abs(float(final["tilt_x_deg"])) + abs(float(final["tilt_y_deg"]))) / 2.0
    tt_initial /= len(ACTUATOR_IDS)
    tt_final /= len(ACTUATOR_IDS)
    tt_recovery_pct = 100.0 * (1.0 - (tt_final / max(tt_initial, 1.0e-12)))
    return tt_initial, tt_final, tt_recovery_pct


def _xyz_correction(summary: dict) -> tuple[float, float, float]:
    corr_dx = 0.0
    corr_dy = 0.0
    corr_dz = 0.0
    for anchor_id in map(str, ACTUATOR_IDS):
        moved = summary["moved_position_by_anchor"][anchor_id]
        final = summary["final_position_by_anchor"][anchor_id]
        corr_dx += abs(float(final["dx_mm"]) - float(moved["dx_mm"]))
        corr_dy += abs(float(final["dy_mm"]) - float(moved["dy_mm"]))
        corr_dz += abs(float(final["dz_mm"]) - float(moved["dz_mm"]))
    count = float(len(ACTUATOR_IDS))
    return corr_dx / count, corr_dy / count, corr_dz / count


def _configure_env(tt_arcmin: float) -> None:
    tt_box_arcmin = 2.0 * tt_arcmin
    os.environ["AO_V3_ENABLE_TIPTILT"] = "1"
    os.environ["AO_V3_DISABLE_BLIND_SEED"] = "1"
    os.environ["AO_V6_SHWFS_NOISE_PROFILE"] = "realistic"
    os.environ["AO_V6_SHWFS_FORWARD_AVERAGES"] = "1"
    os.environ["AO_V6_SEARCH_BOX_XY_MM"] = "0.22"
    os.environ["AO_V6_SEARCH_BOX_Z_MM"] = "0.15"
    os.environ["AO_V6_TIPTILT_TEST_LIMIT_ARCMIN"] = f"{tt_box_arcmin:.6f}"
    os.environ["AO_V3_TIPTILT_LIMIT_ARCMIN"] = f"{tt_box_arcmin:.6f}"


def main() -> int:
    start_index, end_index = _parse_range_args()
    rows_by_case = {
        int(row["case_index"]): row
        for row in _load_existing_rows()
    }

    for case_index, overall_variation_mm, tt_arcmin in _case_specs():
        if case_index < start_index or case_index > end_index:
            continue
        if case_index in rows_by_case:
            print(f"case {case_index:02d}: already present, skipping")
            continue

        _configure_env(tt_arcmin)
        if "ao_v6_backend" in sys.modules:
            importlib.reload(sys.modules["ao_v6_backend"])
            ao_v6_backend = sys.modules["ao_v6_backend"]
        else:
            import ao_v6_backend  # type: ignore
        RunConfig = ao_v6_backend.RunConfig
        run_repair = ao_v6_backend.run_repair

        output_name = f"v6_random_xy022_narrow_ttscaled_case{case_index:02d}"
        config = RunConfig(
            overall_variation_mm=overall_variation_mm,
            per_anchor_overrides=_build_overrides(tt_arcmin),
            output_name=output_name,
            shwfs_noise_profile="realistic",
            shwfs_forward_averages=1,
        )
        result = run_repair(config)
        summary_path = Path(result["repair_summary_path"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tt_initial_deg, tt_final_deg, tt_recovery_pct = _tt_stats(summary)
        corr_dx_mean, corr_dy_mean, corr_dz_mean = _xyz_correction(summary)

        tt_box_arcmin = 2.0 * tt_arcmin
        rows_by_case[case_index] = {
            "case_index": case_index,
            "overall_variation_mm": overall_variation_mm,
            "tt_arcmin": tt_arcmin,
            "tt_box_arcmin": tt_box_arcmin,
            "run_dir": result["run_dir"],
            "before_residual": summary["moved"]["residual_norm"],
            "repaired_residual": summary["repaired"]["residual_norm"],
            "before_true_residual": summary["moved"]["true_residual_norm"],
            "repaired_true_residual": summary["repaired"]["true_residual_norm"],
            "before_wrms_waves": summary["moved"]["wavefront_rms_waves"],
            "repaired_wrms_waves": summary["repaired"]["wavefront_rms_waves"],
            "before_strehl": summary["moved"]["strehl_ratio"],
            "repaired_strehl": summary["repaired"]["strehl_ratio"],
            "tt_initial_deg": tt_initial_deg,
            "tt_final_deg": tt_final_deg,
            "tt_recovery_pct": tt_recovery_pct,
            "corr_dx_mean_mm": corr_dx_mean,
            "corr_dy_mean_mm": corr_dy_mean,
            "corr_dz_mean_mm": corr_dz_mean,
            "iteration_count": summary["iteration_count"],
            "runtime_s": summary["runtime_s"],
            "ray_tracing_runtime_s": summary["ray_tracing_runtime_s"],
        }
        rows = [rows_by_case[idx] for idx in sorted(rows_by_case)]
        _save_rows(rows)
        print(
            f"case {case_index:02d}: overall={overall_variation_mm:.6f} mm "
            f"tt={tt_arcmin:.6f} arcmin tt_box={tt_box_arcmin:.6f} arcmin "
            f"true_resid={summary['repaired']['true_residual_norm']:.6f} "
            f"wrms={summary['repaired']['wavefront_rms_waves']:.6f}"
        )

    rows = [rows_by_case[idx] for idx in sorted(rows_by_case)]
    _save_rows(rows)
    print(f"saved_csv={CSV_PATH}")
    print(f"saved_json={JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
