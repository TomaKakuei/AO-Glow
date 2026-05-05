import csv
from pathlib import Path

import numpy as np

import benchmark_1mm_real_shwfs_converging_920nm as bench
import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
import benchmark_freeform_wb_sensorless_refined_nominal_alignment_120 as refined_nominal
import w_bessel_core as wb
from optical_model_rayoptics import SurfacePerturbation


bench.NUM_MODES = 12
real_shwfs.NUM_MODES = 12
refined_nominal.NUM_MODES = 12

ACTUATOR_IDS = [68, 70, 72, 74, 77, 79, 81, 84]
CASE_SEEDS = [604348002, 42494520, 2083903238, 1507789355, 786839411, 2088058701, 35128567, 890464660, 1341259647, 493014582]
NUM_MODES = 12

ROOT = Path("artifacts")
ROOT.mkdir(exist_ok=True)

MODE_CSV = ROOT / "wb12_mode_definition.csv"
LONG_CSV = ROOT / "wb12_case_residual_components_long.csv"
WIDE_CSV = ROOT / "wb12_case_residual_components_wide.csv"


def build_shwfs():
    nominal_model = bench._build_model(
        pupil_samples=bench.FAST_PUPIL_SAMPLES,
        fft_samples=bench.FAST_FFT_SAMPLES,
    )
    return real_shwfs._build_shwfs_measurement_model(
        nominal_model,
        focus_shift_mm=bench.REFERENCE_FOCUS_MM,
        actuator_ids=ACTUATOR_IDS,
    )


def random_dxyz_case(amplitude_mm: float, seed: int) -> dict[int, SurfacePerturbation]:
    rng = np.random.default_rng(int(seed))
    state: dict[int, SurfacePerturbation] = {}
    for anchor in ACTUATOR_IDS:
        state[int(anchor)] = SurfacePerturbation(
            dx_mm=float(rng.uniform(-amplitude_mm, amplitude_mm)),
            dy_mm=float(rng.uniform(-amplitude_mm, amplitude_mm)),
            dz_mm=float(rng.uniform(-amplitude_mm, amplitude_mm)),
        )
    return state


def random_tilt_case(amplitude_arcmin: float, seed: int) -> dict[int, SurfacePerturbation]:
    amplitude_deg = float(amplitude_arcmin) / 60.0
    rng = np.random.default_rng(int(seed))
    state: dict[int, SurfacePerturbation] = {}
    for anchor in ACTUATOR_IDS:
        state[int(anchor)] = SurfacePerturbation(
            tilt_x_deg=float(rng.uniform(-amplitude_deg, amplitude_deg)),
            tilt_y_deg=float(rng.uniform(-amplitude_deg, amplitude_deg)),
        )
    return state


def estimate_coeffs(shwfs, state: dict[int, SurfacePerturbation]) -> np.ndarray:
    model = bench._build_model(
        pupil_samples=bench.FAST_PUPIL_SAMPLES,
        fft_samples=bench.FAST_FFT_SAMPLES,
    )
    model.set_surface_perturbations(state)
    return np.asarray(shwfs.estimate_coeffs(model), dtype=np.float64)


def state_component_dict(state: dict[int, SurfacePerturbation]) -> dict[str, float]:
    out: dict[str, float] = {}
    for anchor in ACTUATOR_IDS:
        perturb = state[int(anchor)]
        out[f"a{anchor}_dx_mm"] = float(perturb.dx_mm)
        out[f"a{anchor}_dy_mm"] = float(perturb.dy_mm)
        out[f"a{anchor}_dz_mm"] = float(perturb.dz_mm)
        out[f"a{anchor}_tilt_x_deg"] = float(perturb.tilt_x_deg)
        out[f"a{anchor}_tilt_y_deg"] = float(perturb.tilt_y_deg)
    return out


def main() -> None:
    shwfs = build_shwfs()
    reference_coeffs = np.asarray(shwfs.reference_coeffs, dtype=np.float64)
    nm_pairs = list(wb.nm_sequence(NUM_MODES))

    with MODE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode_index", "n", "m", "label", "real_form"])
        for mode_index, (n_val, m_val) in enumerate(nm_pairs, start=1):
            if m_val > 0:
                form = f"sqrt(2)*J_{abs(m_val)}(alpha_{{{abs(m_val)},q}}*rho)*cos({abs(m_val)}*theta)"
            elif m_val < 0:
                form = f"sqrt(2)*J_{abs(m_val)}(alpha_{{{abs(m_val)},q}}*rho)*sin({abs(m_val)}*theta)"
            else:
                form = "J_0(alpha_{0,q}*rho)"
            writer.writerow([mode_index, n_val, m_val, f"B{mode_index}", form])

    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []

    groups = [
        ("dxyz_mm", 0.010, random_dxyz_case),
        ("dxyz_mm", 0.100, random_dxyz_case),
        ("dxyz_mm", 0.500, random_dxyz_case),
        ("tilt_arcmin", 2.0, random_tilt_case),
        ("tilt_arcmin", 5.0, random_tilt_case),
        ("tilt_arcmin", 10.0, random_tilt_case),
    ]

    for group_name, level_value, builder in groups:
        for case_index, seed in enumerate(CASE_SEEDS, start=1):
            state = builder(level_value, int(seed))
            coeff_hat = estimate_coeffs(shwfs, state)
            residual_vec = coeff_hat - reference_coeffs
            residual_norm = float(np.linalg.norm(residual_vec))

            wide_row: dict[str, object] = {
                "group": group_name,
                "level": float(level_value),
                "case_index": int(case_index),
                "seed": int(seed),
                "residual_norm_l2": residual_norm,
            }
            wide_row.update(state_component_dict(state))

            for mode_index, (n_val, m_val) in enumerate(nm_pairs, start=1):
                coeff_value = float(coeff_hat[mode_index - 1])
                ref_value = float(reference_coeffs[mode_index - 1])
                residual_value = float(residual_vec[mode_index - 1])
                long_rows.append(
                    {
                        "group": group_name,
                        "level": float(level_value),
                        "case_index": int(case_index),
                        "seed": int(seed),
                        "mode_index": int(mode_index),
                        "n": int(n_val),
                        "m": int(m_val),
                        "coeff_hat": coeff_value,
                        "coeff_ref": ref_value,
                        "residual_component": residual_value,
                        "residual_component_abs": abs(residual_value),
                        "residual_norm_l2": residual_norm,
                    }
                )
                wide_row[f"mode_{mode_index:02d}_n"] = int(n_val)
                wide_row[f"mode_{mode_index:02d}_m"] = int(m_val)
                wide_row[f"mode_{mode_index:02d}_coeff_hat"] = coeff_value
                wide_row[f"mode_{mode_index:02d}_coeff_ref"] = ref_value
                wide_row[f"mode_{mode_index:02d}_residual"] = residual_value

            wide_rows.append(wide_row)

    long_fieldnames = [
        "group",
        "level",
        "case_index",
        "seed",
        "mode_index",
        "n",
        "m",
        "coeff_hat",
        "coeff_ref",
        "residual_component",
        "residual_component_abs",
        "residual_norm_l2",
    ]
    with LONG_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fieldnames)
        writer.writeheader()
        writer.writerows(long_rows)

    wide_fieldnames = list(wide_rows[0].keys())
    with WIDE_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fieldnames)
        writer.writeheader()
        writer.writerows(wide_rows)

    print(str(MODE_CSV))
    print(str(LONG_CSV))
    print(str(WIDE_CSV))


if __name__ == "__main__":
    main()
