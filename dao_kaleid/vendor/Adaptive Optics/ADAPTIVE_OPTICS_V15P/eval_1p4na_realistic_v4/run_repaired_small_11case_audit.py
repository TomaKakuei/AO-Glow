"""Run eleven reduced-perturbation cases on the repaired Nikon 1.4-NA plant.

The repaired patent prescription is first operated at the fixed center-corrected
mechanical bias saved by ``optimize_corrected_center_bias.py``.  Eleven hidden
30-coordinate perturbations (six groups x dx,dy,dz,tx,ty) are then drawn at
one repaired group at a time in dx/dy/tx/ty at +/-0.020 mm and +/-0.040 degree;
dz and the other groups remain fixed.  The controller sees optical-path residuals,
not the hidden state, and returns the plant toward the fixed operating bias.

Center performance is reported as absolute reference-sphere PTTD RMS WFE.
The +/-0.100-mm sample-plane edges retain their native prescription aberration,
so both absolute and bias-referenced differential PTTD RMS WFE are archived and
the differential maximum is the paper edge metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "eval_1p4na_hybrid" / "model_repair_scratch"
V3_DIR = ROOT / "eval_1p4na_hybrid" / "v3_corrected"
for path in (HERE, MODEL_DIR, V3_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import corrected_absolute_metric as absolute  # noqa: E402
import run_corrected_highna_validation as corrected  # noqa: E402
from validated_highna_model import build_validated_forward_model  # noqa: E402


SEEDS = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11)
TRANSLATION_LIMIT_MM = 0.020
TILT_LIMIT_DEG = 0.040
SMALL_SCALES = np.tile(
    np.asarray(
        [
            TRANSLATION_LIMIT_MM,
            TRANSLATION_LIMIT_MM,
            TRANSLATION_LIMIT_MM,
            TILT_LIMIT_DEG,
            TILT_LIMIT_DEG,
        ],
        dtype=float,
    ),
    len(corrected.GROUPS),
)
DIFFRACTION_LIMIT_WAVES = 0.070
OUT_DIR = HERE / "results_repaired_small"
BIAS_JSON = HERE / "center_bias.json"
BIAS_NPZ = HERE / "center_bias.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def configure_small_scales() -> None:
    corrected.TRANSLATION_LIMIT_MM = TRANSLATION_LIMIT_MM
    corrected.TILT_LIMIT_DEG = TILT_LIMIT_DEG
    corrected.SCALES = SMALL_SCALES.copy()


def load_bias() -> tuple[np.ndarray, np.ndarray, dict]:
    if not BIAS_JSON.exists() or not BIAS_NPZ.exists():
        raise FileNotFoundError("run optimize_corrected_center_bias.py first")
    metadata = json.loads(BIAS_JSON.read_text(encoding="utf-8"))
    arrays = np.load(BIAS_NPZ)
    bias_full_normalized = np.asarray(arrays["bias_normalized"], float)
    full_scales = np.asarray(arrays["coordinate_scales"], float)
    bias_physical = bias_full_normalized * full_scales
    bias_small_normalized = bias_physical / SMALL_SCALES
    return bias_small_normalized, bias_physical, metadata


class BiasedHighNAPlant(corrected.HighNAPlant):
    """High-NA plant whose optical reference is a fixed corrected bias."""

    def __init__(
        self,
        bias_q: np.ndarray,
        hidden_delta_q: np.ndarray | None = None,
        *,
        fields_mm: tuple[float, ...] = corrected.CONTROL_FIELDS_MM,
        pupil_nodes: np.ndarray = corrected.CONTROL_PUPIL_NODES,
        projector: np.ndarray = corrected.CONTROL_PTTD_PROJECTOR,
    ) -> None:
        self.bias_q = np.asarray(bias_q, float).copy()
        delta = (
            np.zeros_like(self.bias_q)
            if hidden_delta_q is None
            else np.asarray(hidden_delta_q, float).copy()
        )
        super().__init__(
            hidden_q=self.bias_q + delta,
            fields_mm=fields_mm,
            pupil_nodes=pupil_nodes,
            projector=projector,
        )
        target_opl, target_full = self._trace_opl_state(self.bias_q)
        self.nominal_opl_mm = target_opl
        self.nominal_full_rays = target_full
        self.cache.clear()
        self.request_count = 0
        self.trace_evaluation_count = 0
        self.ray_attempt_count = 0
        self.ray_trace_count = 0
        self.failed_trace_attempts = 0


def build_biased_response(
    bias_q: np.ndarray,
    *,
    force: bool,
) -> tuple[np.ndarray, dict]:
    npz_path = OUT_DIR / "biased_response_matrix.npz"
    json_path = OUT_DIR / "biased_response_matrix.json"
    if npz_path.exists() and json_path.exists() and not force:
        arrays = np.load(npz_path)
        return np.asarray(arrays["jacobian"], float), json.loads(json_path.read_text(encoding="utf-8"))
    plant = BiasedHighNAPlant(bias_q)
    started = time.perf_counter()
    matrix = corrected.finite_difference_jacobian(
        plant,
        np.zeros(len(SMALL_SCALES), dtype=float),
        step=1.0e-3,
        central=True,
    )
    singular = np.linalg.svd(matrix, compute_uv=False)
    metadata = {
        "shape": list(matrix.shape),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "coordinate_scales": SMALL_SCALES.tolist(),
        "linearization_point": "fixed center-corrected operating bias",
        "full_trace_optical_evaluations": int(plant.trace_evaluation_count),
        "ray_attempt_count": int(plant.ray_attempt_count),
        "wall_seconds": float(time.perf_counter() - started),
    }
    np.savez_compressed(
        npz_path,
        jacobian=matrix,
        bias_small_normalized=bias_q,
        coordinate_scales=SMALL_SCALES,
        pupil_nodes=corrected.CONTROL_PUPIL_NODES,
        field_heights_mm=np.asarray(corrected.CONTROL_FIELDS_MM),
    )
    write_json(json_path, metadata)
    return matrix, metadata


def absolute_stage_metrics(opm, effective_q: np.ndarray) -> dict:
    rows = {}
    for label, field_mm in zip(corrected.AUDIT_FIELD_LABELS, corrected.AUDIT_FIELDS_MM):
        metric = absolute.reference_sphere_rms(
            opm,
            effective_q,
            float(field_mm),
            corrected.AUDIT_PUPIL_NODES,
        )
        rows[label] = {
            "field_height_mm": float(field_mm),
            "absolute_pttd_rms_waves": float(metric["absolute_pttd_rms_waves"]),
            "ray_count": int(metric["ray_count"]),
            "reference_sphere_radius_mm": float(metric["reference_sphere_radius_mm"]),
        }
    edge_values = [
        rows["edge_minus_0p1mm"]["absolute_pttd_rms_waves"],
        rows["edge_plus_0p1mm"]["absolute_pttd_rms_waves"],
    ]
    rows["edge_max_absolute_pttd_rms_waves"] = float(max(edge_values))
    rows["edge_mean_absolute_pttd_rms_waves"] = float(np.mean(edge_values))
    return rows


def active_coordinate_indices(group_index: int) -> np.ndarray:
    base = 5 * group_index
    return np.asarray([base + 0, base + 1, base + 3, base + 4], dtype=int)


def acquire_active_group(
    plant: BiasedHighNAPlant,
    response_matrix: np.ndarray,
    active_indices: np.ndarray,
) -> tuple[np.ndarray, corrected.OpticalEvaluation, list[dict]]:
    """Fixed-bias response acquisition restricted to the addressed group."""
    q = np.zeros(len(SMALL_SCALES), dtype=float)
    history = []
    active_matrix = response_matrix[:, active_indices]
    for round_index in range(3):
        current = plant.evaluate(q)
        active_step, svd = corrected.svd_step(
            active_matrix,
            current.residual_waves.ravel(),
            damping=0.002,
        )
        maximum = float(np.max(np.abs(active_step)))
        if maximum > 1.5:
            active_step *= 1.5 / maximum
        step = np.zeros_like(q)
        step[active_indices] = active_step
        q_next, next_evaluation, line = corrected.select_line_search(
            plant,
            q,
            step,
            (0.25, 0.50, 0.75, 1.00, 1.25),
        )
        history.append(
            {
                "round": round_index + 1,
                "start_aggregate_rms_waves": float(current.aggregate_rms_waves),
                "selected_aggregate_rms_waves": float(next_evaluation.aggregate_rms_waves),
                "step_linf_normalized": float(np.max(np.abs(active_step))),
                "response_svd": svd,
                "line_search": line,
            }
        )
        if np.array_equal(q_next, q):
            break
        q = q_next
    return q, plant.evaluate(q), history


def active_group_jacobian(
    plant: BiasedHighNAPlant,
    q: np.ndarray,
    active_indices: np.ndarray,
    *,
    step: float = 5.0e-4,
) -> np.ndarray:
    columns = []
    for index in active_indices:
        delta = np.zeros_like(q)
        delta[index] = step
        plus = plant.evaluate(q + delta).residual_waves.ravel()
        minus = plant.evaluate(q - delta).residual_waves.ravel()
        columns.append((plus - minus) / (2.0 * step))
    return np.column_stack(columns)


def refine_active_group(
    plant: BiasedHighNAPlant,
    acquisition_command: np.ndarray,
    active_indices: np.ndarray,
    *,
    max_iterations: int,
) -> tuple[np.ndarray, corrected.OpticalEvaluation, list[dict]]:
    """Measured damped-SVD refinement on the four addressed coordinates."""
    q = acquisition_command.copy()
    history = []
    damping_fraction = 3.0e-4
    for iteration in range(max_iterations):
        current = plant.evaluate(q)
        if current.aggregate_rms_waves <= 0.002:
            break
        before = plant.trace_evaluation_count
        matrix = active_group_jacobian(plant, q, active_indices)
        singular_max = float(np.linalg.svd(matrix, compute_uv=False)[0])
        active_step, svd = corrected.svd_step(
            matrix,
            current.residual_waves.ravel(),
            damping=damping_fraction * singular_max,
        )
        maximum = float(np.max(np.abs(active_step)))
        if maximum > 0.75:
            active_step *= 0.75 / maximum
        step = np.zeros_like(q)
        step[active_indices] = active_step
        q_next, next_evaluation, line = corrected.select_line_search(
            plant,
            q,
            step,
            (1.00, 0.50, 0.25, 0.125, 0.0625),
        )
        improved = next_evaluation.aggregate_rms_waves < current.aggregate_rms_waves * (1.0 - 1e-10)
        history.append(
            {
                "iteration": iteration + 1,
                "start_aggregate_rms_waves": float(current.aggregate_rms_waves),
                "selected_aggregate_rms_waves": float(next_evaluation.aggregate_rms_waves),
                "jacobian_trace_evaluations": int(plant.trace_evaluation_count - before),
                "step_linf_normalized": float(np.max(np.abs(active_step))),
                "local_svd": svd,
                "line_search": line,
                "accepted": bool(improved),
            }
        )
        if not improved:
            damping_fraction *= 10.0
            if damping_fraction > 1.0:
                break
            continue
        q = q_next
        damping_fraction = max(1.0e-6, damping_fraction * 0.5)
    return q, plant.evaluate(q), history


def run_case(
    seed: int,
    case_index: int,
    bias_q: np.ndarray,
    response_matrix: np.ndarray,
    *,
    force: bool,
    max_iterations: int,
) -> dict:
    json_path = OUT_DIR / f"case_{case_index:02d}_seed_{seed:03d}.json"
    npz_path = OUT_DIR / f"case_{case_index:02d}_seed_{seed:03d}.npz"
    if json_path.exists() and npz_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    # Match the latest realistic pipeline's reduced case construction: one
    # four-axis group is perturbed per case.  The six repaired physical groups
    # rotate over the eleven deterministic seeds; the other 26 coordinates,
    # including every dz, remain at the center-corrected operating bias.
    active_group_index = (case_index - 1) % len(corrected.GROUPS)
    active_group = list(corrected.GROUPS)[active_group_index]
    rng = np.random.RandomState(seed)
    delta_q = np.zeros(len(SMALL_SCALES), dtype=float)
    base = 5 * active_group_index
    delta_q[base + 0] = float(rng.uniform(-1.0, 1.0))
    delta_q[base + 1] = float(rng.uniform(-1.0, 1.0))
    delta_q[base + 3] = float(rng.uniform(-1.0, 1.0))
    delta_q[base + 4] = float(rng.uniform(-1.0, 1.0))
    zero_command = np.zeros(len(SMALL_SCALES), dtype=float)
    active_indices = active_coordinate_indices(active_group_index)

    control_plant = BiasedHighNAPlant(bias_q, delta_q)
    initial_control = control_plant.evaluate(zero_command)
    acquisition_command, acquisition_control, acquisition_history = acquire_active_group(
        control_plant,
        response_matrix,
        active_indices,
    )
    final_command, final_control, local_history = refine_active_group(
        control_plant,
        acquisition_command,
        active_indices,
        max_iterations=max_iterations,
    )

    audit_plant = BiasedHighNAPlant(
        bias_q,
        delta_q,
        fields_mm=corrected.AUDIT_FIELDS_MM,
        pupil_nodes=corrected.AUDIT_PUPIL_NODES,
        projector=corrected.AUDIT_PTTD_PROJECTOR,
    )
    differential_stages = {
        "initial": audit_plant.evaluate(zero_command),
        "post_acquisition": audit_plant.evaluate(acquisition_command),
        "final": audit_plant.evaluate(final_command),
    }
    rim_plant = BiasedHighNAPlant(
        bias_q,
        delta_q,
        fields_mm=corrected.AUDIT_FIELDS_MM,
        pupil_nodes=corrected.AUDIT_RIM_NODES,
        projector=np.eye(len(corrected.AUDIT_RIM_NODES)),
    )
    rim_stages = {
        "initial": rim_plant.evaluate(zero_command),
        "post_acquisition": rim_plant.evaluate(acquisition_command),
        "final": rim_plant.evaluate(final_command),
    }

    absolute.disable_file_logging()
    absolute_model, _ = build_validated_forward_model()
    commands = {
        "initial": zero_command,
        "post_acquisition": acquisition_command,
        "final": final_command,
    }
    absolute_metrics = {
        name: absolute_stage_metrics(absolute_model, bias_q + delta_q + command)
        for name, command in commands.items()
    }
    differential_metrics = {
        name: corrected.evaluation_dict(value)
        for name, value in differential_stages.items()
    }
    full_trace = all(
        value.full_rays == value.expected_rays
        for value in list(differential_stages.values()) + list(rim_stages.values())
    )
    final_center_absolute = absolute_metrics["final"]["center"]["absolute_pttd_rms_waves"]
    final_edge_differential = differential_metrics["final"]["edge_max_rms_waves"]
    initial_edge_differential = differential_metrics["initial"]["edge_max_rms_waves"]
    record = {
        "schema_version": "dao-nikon-repaired-small-v1",
        "case": int(case_index),
        "seed": int(seed),
        "active_group": active_group,
        "generator": "numpy.random.RandomState(seed).uniform(-1,1,4) applied to dx,dy,tx,ty of one rotating group; dz and all other groups fixed",
        "perturbation_limits": {
            "dx_dy_dz_mm": [-TRANSLATION_LIMIT_MM, TRANSLATION_LIMIT_MM],
            "tx_ty_deg": [-TILT_LIMIT_DEG, TILT_LIMIT_DEG],
        },
        "coordinate_order": [
            f"{group}.{coordinate}"
            for group in corrected.GROUPS
            for coordinate in corrected.COORDINATES
        ],
        "applied_coordinates": [
            f"{active_group}.dx",
            f"{active_group}.dy",
            f"{active_group}.tx",
            f"{active_group}.ty",
        ],
        "states": {
            "fixed_bias_normalized_in_small_scale": bias_q.tolist(),
            "hidden_perturbation_normalized": delta_q.tolist(),
            "hidden_perturbation_physical": corrected.state_from_normalized(delta_q),
            "initial_total_physical": corrected.state_from_normalized(bias_q + delta_q),
            "post_acquisition_command_normalized": acquisition_command.tolist(),
            "terminal_command_normalized": final_command.tolist(),
            "terminal_residual_from_bias_normalized": (delta_q + final_command).tolist(),
            "terminal_residual_from_bias_physical": corrected.state_from_normalized(
                delta_q + final_command
            ),
            "terminal_total_physical": corrected.state_from_normalized(
                bias_q + delta_q + final_command
            ),
        },
        "absolute_metrics": absolute_metrics,
        "differential_metrics": differential_metrics,
        "control_metrics": {
            "initial": corrected.evaluation_dict(initial_control),
            "post_acquisition": corrected.evaluation_dict(acquisition_control),
            "final": corrected.evaluation_dict(final_control),
        },
        "controller": {
            "state_truth_used": False,
            "addressed_coordinate_indices": active_indices.tolist(),
            "acquisition": "three-pass fixed-bias response matrix restricted to the addressed four-axis group, with measured gain scan",
            "acquisition_history": acquisition_history,
            "refinement": "central-difference damped-SVD Gauss-Newton restricted to the addressed four-axis group, with measured trust-region scan",
            "local_history": local_history,
        },
        "evaluation_counts": {
            "controller_unique_full_optical_evaluations": int(control_plant.trace_evaluation_count),
            "controller_ray_attempts": int(control_plant.ray_attempt_count),
            "audit_ray_attempts": int(audit_plant.ray_attempt_count),
            "rim_ray_attempts": int(rim_plant.ray_attempt_count),
        },
        "case_gates": {
            "center_absolute_below_0p07": bool(final_center_absolute < DIFFRACTION_LIMIT_WAVES),
            "edge_differential_improved": bool(final_edge_differential < initial_edge_differential),
            "all_audit_and_rim_rays_complete": bool(full_trace),
            "pass": bool(
                final_center_absolute < DIFFRACTION_LIMIT_WAVES
                and final_edge_differential < initial_edge_differential
                and full_trace
            ),
        },
        "wall_seconds": float(time.perf_counter() - started),
    }
    np.savez_compressed(
        npz_path,
        bias_small_normalized=bias_q,
        hidden_perturbation_normalized=delta_q,
        initial_command_normalized=zero_command,
        post_acquisition_command_normalized=acquisition_command,
        terminal_command_normalized=final_command,
        audit_pupil_nodes=corrected.AUDIT_PUPIL_NODES,
        audit_fields_mm=np.asarray(corrected.AUDIT_FIELDS_MM),
        initial_differential_residual_waves=differential_stages["initial"].residual_waves,
        post_acquisition_differential_residual_waves=differential_stages[
            "post_acquisition"
        ].residual_waves,
        final_differential_residual_waves=differential_stages["final"].residual_waves,
    )
    write_json(json_path, record)
    print(
        "CASE {:02d}/{} seed={} center_abs {:.6f}->{:.6f}->{:.6f}; "
        "edge_diff {:.6f}->{:.6f}; pass={} ({:.1f}s)".format(
            case_index,
            len(SEEDS),
            seed,
            absolute_metrics["initial"]["center"]["absolute_pttd_rms_waves"],
            absolute_metrics["post_acquisition"]["center"]["absolute_pttd_rms_waves"],
            final_center_absolute,
            initial_edge_differential,
            final_edge_differential,
            record["case_gates"]["pass"],
            record["wall_seconds"],
        ),
        flush=True,
    )
    return record


def metric_stats(values: list[float]) -> dict:
    array = np.asarray(values, float)
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "values": [float(value) for value in values],
    }


def summarize(cases: list[dict], bias_metadata: dict, response_metadata: dict) -> dict:
    center_initial = [
        case["absolute_metrics"]["initial"]["center"]["absolute_pttd_rms_waves"]
        for case in cases
    ]
    center_capture = [
        case["absolute_metrics"]["post_acquisition"]["center"]["absolute_pttd_rms_waves"]
        for case in cases
    ]
    center_final = [
        case["absolute_metrics"]["final"]["center"]["absolute_pttd_rms_waves"]
        for case in cases
    ]
    edge_initial_diff = [
        case["differential_metrics"]["initial"]["edge_max_rms_waves"]
        for case in cases
    ]
    edge_capture_diff = [
        case["differential_metrics"]["post_acquisition"]["edge_max_rms_waves"]
        for case in cases
    ]
    edge_final_diff = [
        case["differential_metrics"]["final"]["edge_max_rms_waves"]
        for case in cases
    ]
    edge_final_abs = [
        case["absolute_metrics"]["final"]["edge_max_absolute_pttd_rms_waves"]
        for case in cases
    ]
    return {
        "schema_version": "dao-nikon-repaired-small-summary-v1",
        "case_count": len(cases),
        "seeds": [case["seed"] for case in cases],
        "all_cases_pass": bool(all(case["case_gates"]["pass"] for case in cases)),
        "center_final_below_0p07": int(
            sum(value < DIFFRACTION_LIMIT_WAVES for value in center_final)
        ),
        "center_metric": "absolute reference-sphere PTTD RMS WFE",
        "edge_reported_metric": "fixed-bias-referenced differential PTTD RMS WFE; maximum of +/-0.100-mm fields",
        "statistics": {
            "center_initial_absolute_waves": metric_stats(center_initial),
            "center_post_acquisition_absolute_waves": metric_stats(center_capture),
            "center_final_absolute_waves": metric_stats(center_final),
            "edge_initial_differential_waves": metric_stats(edge_initial_diff),
            "edge_post_acquisition_differential_waves": metric_stats(edge_capture_diff),
            "edge_final_differential_waves": metric_stats(edge_final_diff),
            "edge_final_absolute_waves": metric_stats(edge_final_abs),
        },
        "fixed_bias": bias_metadata,
        "response_matrix": response_metadata,
    }


def write_csv(cases: list[dict]) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "case": case["case"],
                "seed": case["seed"],
                "active_group": case["active_group"],
                "initial_center_absolute_waves": case["absolute_metrics"]["initial"]["center"]["absolute_pttd_rms_waves"],
                "post_acquisition_center_absolute_waves": case["absolute_metrics"]["post_acquisition"]["center"]["absolute_pttd_rms_waves"],
                "final_center_absolute_waves": case["absolute_metrics"]["final"]["center"]["absolute_pttd_rms_waves"],
                "initial_edge_differential_waves": case["differential_metrics"]["initial"]["edge_max_rms_waves"],
                "post_acquisition_edge_differential_waves": case["differential_metrics"]["post_acquisition"]["edge_max_rms_waves"],
                "final_edge_differential_waves": case["differential_metrics"]["final"]["edge_max_rms_waves"],
                "final_edge_absolute_waves": case["absolute_metrics"]["final"]["edge_max_absolute_pttd_rms_waves"],
                "pass": case["case_gates"]["pass"],
                "wall_seconds": case["wall_seconds"],
            }
        )
    with (OUT_DIR / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(force: bool, force_response: bool, max_iterations: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    absolute.disable_file_logging()
    bias_q, bias_physical, bias_metadata = load_bias()
    configure_small_scales()
    response_matrix, response_metadata = build_biased_response(
        bias_q,
        force=force_response,
    )
    print(
        f"RESPONSE shape={response_matrix.shape} rank={response_metadata['rank']} "
        f"condition={response_metadata['condition_number']:.3e}",
        flush=True,
    )
    cases = []
    for case_index, seed in enumerate(SEEDS, start=1):
        cases.append(
            run_case(
                seed,
                case_index,
                bias_q,
                response_matrix,
                force=force,
                max_iterations=max_iterations,
            )
        )
        summary = summarize(cases, bias_metadata, response_metadata)
        write_json(OUT_DIR / "summary_partial.json", summary)
    summary = summarize(cases, bias_metadata, response_metadata)
    summary["experiment"] = {
        "source_prescription": "Nikon-assigned US6519092B2 Embodiment 2, objective surfaces 1-24 and imaging-lens surfaces 28-33",
        "wavelength_nm": float(corrected.WAVELENGTH_NM),
        "sample_side_na": float(corrected.SAMPLE_SIDE_NA),
        "magnification": -99.50324044682563,
        "fields_mm": list(corrected.AUDIT_FIELDS_MM),
        "actuated_coordinates": 30,
        "applied_perturbation_coordinates_per_case": 4,
        "moving_groups": 6,
        "perturbation_translation_bound_mm": TRANSLATION_LIMIT_MM,
        "perturbation_tilt_bound_deg": TILT_LIMIT_DEG,
        "fixed_bias_physical": bias_physical.tolist(),
        "audit_pupil_nodes_per_field": int(len(corrected.AUDIT_PUPIL_NODES)),
        "additional_full_na_rim_nodes_per_field": int(len(corrected.AUDIT_RIM_NODES)),
        "state_truth_used_by_controller": False,
    }
    source = MODEL_DIR / "Nikon_1p4NA_source_copy.zmx"
    summary["hashes"] = {
        "run_script_sha256": sha256(Path(__file__)),
        "absolute_metric_script_sha256": sha256(HERE / "corrected_absolute_metric.py"),
        "bias_json_sha256": sha256(BIAS_JSON),
        "source_prescription_sha256": sha256(source),
        "validated_model_sha256": sha256(MODEL_DIR / "validated_highna_model.py"),
    }
    write_json(OUT_DIR / "audit_results.json", summary)
    write_csv(cases)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-response", action="store_true")
    parser.add_argument("--max-local-iterations", type=int, default=6)
    args = parser.parse_args()
    result = run(args.force, args.force_response, args.max_local_iterations)
    print(json.dumps({
        "case_count": result["case_count"],
        "all_cases_pass": result["all_cases_pass"],
        "center_final_below_0p07": result["center_final_below_0p07"],
        "center_final_absolute": result["statistics"]["center_final_absolute_waves"],
        "edge_final_differential": result["statistics"]["edge_final_differential_waves"],
        "edge_final_absolute": result["statistics"]["edge_final_absolute_waves"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
