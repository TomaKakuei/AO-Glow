"""Deterministic 30-coordinate validation on the audited 100x/1.4-NA plant.

This script is intentionally independent of the historical High-NA CNN run.
It uses the repaired physical sample-to-camera prescription, six rigid groups
with five genuinely applied coordinates each, and direct nominal-referenced
OPL measurements on fixed sample-side-NA pupil nodes.

The controller is model based.  A nominal response matrix provides a linearized
acquisition proposal; damped finite-difference Gauss--Newton iterations then
refine the same 30 stroke-normalized coordinates.  No knowledge of the drawn
mechanical state is supplied to either stage or to its optical objective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rayoptics.elem.surface import DecenterData


HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "model_repair_scratch"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from validated_highna_model import (  # noqa: E402
    WAVELENGTH_NM,
    build_validated_forward_model,
)
from probe_forward_model import trace_explicit  # noqa: E402


SCHEMA_VERSION = "dao-highna-audit-v1"
# A seed is eligible before recovery is inspected iff its initial state traces
# every independent audit-grid and full-NA rim ray at all three audit fields.
# The default run takes the first eleven eligible non-negative seeds and writes
# every accepted/rejected candidate before any recovery is evaluated.
TARGET_CASE_COUNT = 11
CONTROL_FIELDS_MM = (0.0, -0.05, 0.05)
AUDIT_FIELDS_MM = (0.0, -0.1, 0.1)
AUDIT_FIELD_LABELS = ("center", "edge_minus_0p1mm", "edge_plus_0p1mm")
GROUPS = {
    "G2": (6, 7),
    "G3": (8, 10),
    "G4": (11, 13),
    "G5": (14, 16),
    "G6": (17, 20),
    "G7": (21, 24),
}
COORDINATES = ("dx", "dy", "dz", "tx", "ty")
TRANSLATION_LIMIT_MM = 0.150
TILT_LIMIT_DEG = 1.5
SCALES = np.tile(
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
    len(GROUPS),
)
WAVELENGTH_MM = WAVELENGTH_NM * 1e-6
SAMPLE_INDEX = 1.52216
SAMPLE_SIDE_NA = 1.4


class FullTraceError(RuntimeError):
    """Raised when any fixed pupil ray does not traverse the full plant."""


def control_pupil_nodes() -> np.ndarray:
    """Return the fixed sparse pupil used only by the feedback objective."""
    points = [(0.0, 0.0)]
    for rho in (0.25, 0.50, 0.75, 1.00):
        for az in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            points.append((float(rho * np.cos(az)), float(rho * np.sin(az))))
    return np.asarray(points, dtype=float)


def audit_pupil_nodes() -> np.ndarray:
    """Return a spatially uniform 17x17 Cartesian disk (197 nodes)."""
    axis = np.linspace(-1.0, 1.0, 17)
    return np.asarray(
        [(float(x), float(y)) for y in axis for x in axis if x * x + y * y <= 1.0 + 1e-12],
        dtype=float,
    )


def audit_rim_nodes() -> np.ndarray:
    """Return 32 full-NA coverage nodes not already in the RMS grid."""
    rim = np.asarray(
        [
            (float(np.cos(az)), float(np.sin(az)))
            for az in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        ],
        dtype=float,
    )
    grid = audit_pupil_nodes()
    keep = [
        point
        for point in rim
        if not np.any(np.linalg.norm(grid - point[None, :], axis=1) < 1e-12)
    ]
    return np.asarray(keep, dtype=float)


CONTROL_PUPIL_NODES = control_pupil_nodes()
AUDIT_PUPIL_NODES = audit_pupil_nodes()
AUDIT_RIM_NODES = audit_rim_nodes()


def pttd_design(nodes: np.ndarray) -> np.ndarray:
    x = nodes[:, 0]
    y = nodes[:, 1]
    return np.column_stack((np.ones(len(x)), x, y, x * x + y * y))


CONTROL_PTTD_DESIGN = pttd_design(CONTROL_PUPIL_NODES)
CONTROL_PTTD_PROJECTOR = np.eye(len(CONTROL_PUPIL_NODES)) - CONTROL_PTTD_DESIGN @ np.linalg.pinv(CONTROL_PTTD_DESIGN)
AUDIT_PTTD_DESIGN = pttd_design(AUDIT_PUPIL_NODES)
AUDIT_PTTD_PROJECTOR = np.eye(len(AUDIT_PUPIL_NODES)) - AUDIT_PTTD_DESIGN @ np.linalg.pinv(AUDIT_PTTD_DESIGN)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_from_normalized(q: np.ndarray) -> dict[str, dict[str, float]]:
    physical = np.asarray(q, dtype=float) * SCALES
    state: dict[str, dict[str, float]] = {}
    offset = 0
    for group in GROUPS:
        state[group] = {
            coordinate: float(physical[offset + index])
            for index, coordinate in enumerate(COORDINATES)
        }
        offset += len(COORDINATES)
    return state


def initial_normalized_state(seed: int) -> np.ndarray:
    """Draw every translation/tilt independently within its stated limit."""
    rng = np.random.RandomState(seed)
    return rng.uniform(-1.0, 1.0, size=len(SCALES)).astype(float)


def apply_normalized_state(opm, q: np.ndarray) -> None:
    """Apply the configured rigid groups without moving uncommanded bodies."""
    sm = opm.seq_model
    state = state_from_normalized(q)
    for start, end in GROUPS.values():
        sm.ifcs[start].decenter = None
        sm.ifcs[end].decenter = None
    for group, (start, end) in GROUPS.items():
        p = state[group]
        enter = DecenterData(
            "decenter",
            x=p["dx"],
            y=p["dy"],
            alpha=p["tx"],
            beta=p["ty"],
        )
        leave = DecenterData(
            "reverse",
            x=p["dx"],
            y=p["dy"],
            alpha=p["tx"],
            beta=p["ty"],
        )
        # RayOptics stores axial translation in the third coordinate of the
        # transform vector; constructor keywords expose only x/y and angles.
        enter.dec[2] = p["dz"]
        leave.dec[2] = p["dz"]
        enter.update()
        leave.update()
        # A tilted body's final vertex moves by (R - I) * span. Include that
        # displacement in the return transform so it does not move the next
        # body, tube lens, or image surface. The body itself remains rigid.
        if enter.rot_mat is not None:
            span = np.array([0.0, 0.0, sum(gap.thi for gap in sm.gaps[start:end])])
            leave.dec += (enter.rot_mat - np.eye(3)) @ span
        sm.ifcs[start].decenter = enter
        sm.ifcs[end].decenter = leave
    sm.update_model()


@dataclass
class OpticalEvaluation:
    command_q: np.ndarray
    effective_q: np.ndarray
    opl_mm: np.ndarray
    residual_waves: np.ndarray
    field_rms_waves: np.ndarray
    aggregate_rms_waves: float
    full_rays: int
    expected_rays: int


class HighNAPlant:
    """Forward plant with a hidden initial state and cached command responses."""

    def __init__(
        self,
        hidden_q: np.ndarray | None = None,
        *,
        fields_mm: tuple[float, ...] = CONTROL_FIELDS_MM,
        pupil_nodes: np.ndarray = CONTROL_PUPIL_NODES,
        projector: np.ndarray = CONTROL_PTTD_PROJECTOR,
    ) -> None:
        self.hidden_q = (
            np.zeros(len(SCALES), dtype=float)
            if hidden_q is None
            else np.asarray(hidden_q, dtype=float).copy()
        )
        self.fields_mm = tuple(float(value) for value in fields_mm)
        self.pupil_nodes = np.asarray(pupil_nodes, dtype=float)
        self.projector = np.asarray(projector, dtype=float)
        self.opm, self.source_mapping = build_validated_forward_model()
        self.expected_interfaces = len(self.opm.seq_model.ifcs)
        self.expected_rays = len(self.fields_mm) * len(self.pupil_nodes)
        self.cache: dict[bytes, OpticalEvaluation] = {}
        self.request_count = 0
        self.trace_evaluation_count = 0
        self.ray_attempt_count = 0
        self.ray_trace_count = 0
        self.failed_trace_attempts = 0
        # The reference is the zero-misalignment prescription and is independent
        # of the hidden seed state.  It is traced before the hidden state is used.
        nominal = self._trace_opl_state(np.zeros(len(SCALES), dtype=float))
        self.nominal_opl_mm = nominal[0]
        self.nominal_full_rays = nominal[1]

    @staticmethod
    def _cache_key(q: np.ndarray) -> bytes:
        # Optimizer iterates are deterministic doubles.  Rounding only makes
        # exact line-search repeats reusable and is far below actuator steps.
        return np.round(np.asarray(q, dtype=np.float64), 14).tobytes()

    def _trace_opl_state(self, effective_q: np.ndarray) -> tuple[np.ndarray, int]:
        apply_normalized_state(self.opm, effective_q)
        rows = []
        full = 0
        for field_mm in self.fields_mm:
            field_values = []
            for px, py in self.pupil_nodes:
                self.ray_attempt_count += 1
                ux = SAMPLE_SIDE_NA * float(px) / SAMPLE_INDEX
                uy = SAMPLE_SIDE_NA * float(py) / SAMPLE_INDEX
                try:
                    package = trace_explicit(self.opm, field_mm, ux, uy)
                    ray = package.ray
                    if len(ray) != self.expected_interfaces:
                        raise FullTraceError(
                            f"partial trace at field={field_mm:+.3f} mm, "
                            f"pupil=({px:+.4f},{py:+.4f}): "
                            f"{len(ray)}/{self.expected_interfaces} interfaces"
                        )
                    value = float(package.op)
                    if not math.isfinite(value):
                        raise FullTraceError("non-finite optical path length")
                except FullTraceError:
                    self.failed_trace_attempts += 1
                    raise
                except Exception as exc:
                    self.failed_trace_attempts += 1
                    partial = getattr(exc, "ray_pkg", None)
                    count = len(partial[0]) if partial is not None and partial[0] else 0
                    raise FullTraceError(
                        f"trace failed at field={field_mm:+.3f} mm, "
                        f"pupil=({px:+.4f},{py:+.4f}); completed "
                        f"{count}/{self.expected_interfaces}: {type(exc).__name__}: {exc}"
                    ) from exc
                field_values.append(value)
                full += 1
                self.ray_trace_count += 1
            rows.append(field_values)
        return np.asarray(rows, dtype=float), full

    def evaluate(self, command_q: np.ndarray) -> OpticalEvaluation:
        self.request_count += 1
        command_q = np.asarray(command_q, dtype=float)
        key = self._cache_key(command_q)
        if key in self.cache:
            return self.cache[key]
        effective_q = self.hidden_q + command_q
        opl, full = self._trace_opl_state(effective_q)
        delta_waves = (opl - self.nominal_opl_mm) / WAVELENGTH_MM
        residual = np.asarray([self.projector @ row for row in delta_waves], dtype=float)
        rms = np.sqrt(np.mean(residual * residual, axis=1))
        result = OpticalEvaluation(
            command_q=command_q.copy(),
            effective_q=effective_q.copy(),
            opl_mm=opl,
            residual_waves=residual,
            field_rms_waves=rms,
            aggregate_rms_waves=float(np.sqrt(np.mean(residual * residual))),
            full_rays=full,
            expected_rays=self.expected_rays,
        )
        self.cache[key] = result
        self.trace_evaluation_count += 1
        return result


def finite_difference_jacobian(
    plant: HighNAPlant,
    q: np.ndarray,
    *,
    step: float,
    central: bool,
) -> np.ndarray:
    """Return d(projected differential OPL)/d(normalized coordinate)."""
    q = np.asarray(q, dtype=float)
    columns = []
    base = None if central else plant.evaluate(q).residual_waves.ravel()
    for index in range(len(q)):
        delta = np.zeros_like(q)
        delta[index] = step
        if central:
            plus = plant.evaluate(q + delta).residual_waves.ravel()
            minus = plant.evaluate(q - delta).residual_waves.ravel()
            column = (plus - minus) / (2.0 * step)
        else:
            plus = plant.evaluate(q + delta).residual_waves.ravel()
            column = (plus - base) / step
        columns.append(column)
    return np.column_stack(columns)


def svd_step(jacobian: np.ndarray, residual: np.ndarray, damping: float) -> tuple[np.ndarray, dict]:
    u, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
    coefficients = singular / (singular * singular + damping * damping)
    step = -(vt.T * coefficients) @ (u.T @ residual)
    positive = singular[singular > np.finfo(float).eps * singular[0]]
    info = {
        "singular_max": float(singular[0]),
        "singular_min": float(positive[-1]) if len(positive) else 0.0,
        "condition_number": float(singular[0] / positive[-1]) if len(positive) else math.inf,
        "numerical_rank": int(np.linalg.matrix_rank(jacobian)),
        "damping": float(damping),
    }
    return step, info


def select_line_search(
    plant: HighNAPlant,
    q: np.ndarray,
    step: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, OpticalEvaluation, list[dict]]:
    candidates = []
    best_q = q.copy()
    best_eval = plant.evaluate(best_q)
    candidates.append({"alpha": 0.0, "aggregate_rms_waves": best_eval.aggregate_rms_waves})
    for alpha in alphas:
        trial_q = q + float(alpha) * step
        try:
            trial = plant.evaluate(trial_q)
            value = trial.aggregate_rms_waves
            record = {"alpha": float(alpha), "aggregate_rms_waves": value, "full_trace": True}
            if value < best_eval.aggregate_rms_waves:
                best_q = trial_q
                best_eval = trial
        except FullTraceError as exc:
            record = {
                "alpha": float(alpha),
                "aggregate_rms_waves": None,
                "full_trace": False,
                "error": str(exc),
            }
        candidates.append(record)
    return best_q, best_eval, candidates


def acquire(
    plant: HighNAPlant,
    response_matrix: np.ndarray,
) -> tuple[np.ndarray, OpticalEvaluation, list[dict]]:
    """Linearized nominal-response acquisition from a zero actuator command."""
    # The controller is deliberately initialized at zero command.  The hidden
    # state resides only inside ``plant`` and is never read here.
    q = np.zeros(len(SCALES), dtype=float)
    history = []
    for round_index in range(3):
        current = plant.evaluate(q)
        raw_step, svd = svd_step(response_matrix, current.residual_waves.ravel(), damping=0.02)
        max_component = float(np.max(np.abs(raw_step)))
        if max_component > 1.5:
            raw_step *= 1.5 / max_component
        q_next, next_eval, line = select_line_search(
            plant,
            q,
            raw_step,
            (0.25, 0.50, 0.75, 1.00, 1.25),
        )
        history.append(
            {
                "round": round_index + 1,
                "start_aggregate_rms_waves": current.aggregate_rms_waves,
                "step_linf_normalized": float(np.max(np.abs(raw_step))),
                "response_svd": svd,
                "line_search": line,
                "selected_aggregate_rms_waves": next_eval.aggregate_rms_waves,
            }
        )
        if np.array_equal(q_next, q):
            break
        q = q_next
    return q, plant.evaluate(q), history


def refine(
    plant: HighNAPlant,
    acquisition_command_q: np.ndarray,
    *,
    max_iterations: int,
) -> tuple[np.ndarray, OpticalEvaluation, list[dict]]:
    """Damped local Gauss--Newton search on normalized physical strokes."""
    q = acquisition_command_q.copy()
    history = []
    damping = 0.05
    for iteration in range(max_iterations):
        current = plant.evaluate(q)
        if current.aggregate_rms_waves <= 0.005:
            break
        before_traces = plant.trace_evaluation_count
        jacobian = finite_difference_jacobian(plant, q, step=5e-4, central=True)
        raw_step, svd = svd_step(jacobian, current.residual_waves.ravel(), damping=damping)
        max_component = float(np.max(np.abs(raw_step)))
        if max_component > 0.75:
            raw_step *= 0.75 / max_component
        q_next, next_eval, line = select_line_search(
            plant,
            q,
            raw_step,
            (1.00, 0.50, 0.25, 0.125, 0.0625),
        )
        improved = next_eval.aggregate_rms_waves < current.aggregate_rms_waves * (1.0 - 1e-10)
        history.append(
            {
                "iteration": iteration + 1,
                "start_aggregate_rms_waves": current.aggregate_rms_waves,
                "jacobian_trace_evaluations": plant.trace_evaluation_count - before_traces,
                "step_linf_normalized": float(np.max(np.abs(raw_step))),
                "local_svd": svd,
                "line_search": line,
                "accepted": bool(improved),
                "selected_aggregate_rms_waves": next_eval.aggregate_rms_waves,
            }
        )
        if not improved:
            damping *= 10.0
            if damping > 50.0:
                break
            continue
        q = q_next
        damping = max(0.005, damping * 0.5)
        if float(np.max(np.abs(raw_step))) < 2e-5:
            break
    return q, plant.evaluate(q), history


def evaluation_dict(value: OpticalEvaluation) -> dict[str, Any]:
    edge_mean = float(np.mean(value.field_rms_waves[1:]))
    edge_max = float(np.max(value.field_rms_waves[1:]))
    return {
        "aggregate_rms_waves": value.aggregate_rms_waves,
        "center_rms_waves": float(value.field_rms_waves[0]),
        "negative_edge_rms_waves": float(value.field_rms_waves[1]),
        "positive_edge_rms_waves": float(value.field_rms_waves[2]),
        "edge_mean_rms_waves": edge_mean,
        "edge_max_rms_waves": edge_max,
        "full_rays": int(value.full_rays),
        "expected_rays": int(value.expected_rays),
    }


def save_evaluation_arrays(
    path: Path,
    audit_plant: HighNAPlant,
    stages: dict[str, OpticalEvaluation],
    rim_plant: HighNAPlant,
    rim_stages: dict[str, OpticalEvaluation],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "audit_pupil_nodes": AUDIT_PUPIL_NODES,
        "audit_rim_nodes": AUDIT_RIM_NODES,
        "audit_field_heights_mm": np.asarray(AUDIT_FIELDS_MM, dtype=float),
        "nominal_audit_opl_mm": audit_plant.nominal_opl_mm,
        "nominal_audit_full_trace_mask": np.ones_like(
            audit_plant.nominal_opl_mm, dtype=np.uint8
        ),
        "nominal_rim_opl_mm": rim_plant.nominal_opl_mm,
        "nominal_rim_full_trace_mask": np.ones_like(
            rim_plant.nominal_opl_mm, dtype=np.uint8
        ),
    }
    for name, evaluation in stages.items():
        arrays[f"{name}_command_normalized"] = evaluation.command_q
        arrays[f"{name}_effective_state_normalized"] = evaluation.effective_q
        arrays[f"{name}_effective_state_physical"] = evaluation.effective_q * SCALES
        arrays[f"{name}_opl_mm"] = evaluation.opl_mm
        arrays[f"{name}_differential_pttd_residual_waves"] = evaluation.residual_waves
        arrays[f"{name}_full_trace_mask"] = np.ones_like(evaluation.opl_mm, dtype=np.uint8)
    for name, evaluation in rim_stages.items():
        arrays[f"{name}_rim_opl_mm"] = evaluation.opl_mm
        arrays[f"{name}_rim_full_trace_mask"] = np.ones_like(
            evaluation.opl_mm, dtype=np.uint8
        )
    np.savez_compressed(path, **arrays)


def trace_initial_completion(seed: int) -> dict[str, Any]:
    """Apply only the deterministic hidden state and count audit/rim traversal."""
    hidden_q = initial_normalized_state(seed)
    opm, _ = build_validated_forward_model()
    apply_normalized_state(opm, hidden_q)
    expected_interfaces = len(opm.seq_model.ifcs)
    nodes = np.vstack((AUDIT_PUPIL_NODES, AUDIT_RIM_NODES))
    expected = len(AUDIT_FIELDS_MM) * len(nodes)
    completed = 0
    errors = []
    started = time.perf_counter()
    for field_mm in AUDIT_FIELDS_MM:
        for px, py in nodes:
            ux = SAMPLE_SIDE_NA * float(px) / SAMPLE_INDEX
            uy = SAMPLE_SIDE_NA * float(py) / SAMPLE_INDEX
            try:
                package = trace_explicit(opm, field_mm, ux, uy)
                segments = len(package.ray)
                if segments != expected_interfaces:
                    errors.append(
                        f"field={field_mm:+.3f}, pupil=({px:+.4f},{py:+.4f}), "
                        f"segments={segments}/{expected_interfaces}"
                    )
                    continue
                completed += 1
            except Exception as exc:
                partial = getattr(exc, "ray_pkg", None)
                segments = len(partial[0]) if partial is not None and partial[0] else 0
                errors.append(
                    f"field={field_mm:+.3f}, pupil=({px:+.4f},{py:+.4f}), "
                    f"segments={segments}/{expected_interfaces}, {type(exc).__name__}: {exc}"
                )
    return {
        "seed": int(seed),
        "eligible": completed == expected,
        "completed_rays": int(completed),
        "expected_rays": int(expected),
        "rms_grid_nodes_per_field": int(len(AUDIT_PUPIL_NODES)),
        "additional_rim_nodes_per_field": int(len(AUDIT_RIM_NODES)),
        "first_error": errors[0] if errors else "",
        "failure_count": int(len(errors)),
        "wall_seconds": float(time.perf_counter() - started),
    }


def determine_seeds(
    out_dir: Path,
    requested: str,
    *,
    force: bool,
) -> tuple[int, ...]:
    """Apply the pre-recovery full-trace eligibility rule and save its ledger."""
    ledger_path = out_dir / "seed_eligibility.csv"
    if requested.strip().lower() == "auto":
        candidate_seeds = range(1000)
        target = TARGET_CASE_COUNT
    else:
        explicit = parse_seeds(requested)
        candidate_seeds = explicit
        target = len(explicit)

    existing: dict[int, dict[str, str]] = {}
    if ledger_path.exists() and not force:
        with ledger_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                existing[int(row["seed"])] = row

    ledger: list[dict[str, Any]] = []
    eligible: list[int] = []
    for seed in candidate_seeds:
        if seed in existing:
            raw = existing[seed]
            record: dict[str, Any] = {
                "seed": seed,
                "eligible": str(raw["eligible"]).strip().lower() == "true",
                "completed_rays": int(raw["completed_rays"]),
                "expected_rays": int(raw["expected_rays"]),
                "rms_grid_nodes_per_field": int(raw["rms_grid_nodes_per_field"]),
                "additional_rim_nodes_per_field": int(raw["additional_rim_nodes_per_field"]),
                "first_error": raw.get("first_error", ""),
                "failure_count": int(raw["failure_count"]),
                "wall_seconds": float(raw["wall_seconds"]),
            }
        else:
            record = trace_initial_completion(seed)
        ledger.append(record)
        if record["eligible"]:
            eligible.append(seed)
        if len(eligible) == target:
            break
    if len(eligible) != target:
        raise RuntimeError(
            f"found only {len(eligible)}/{target} full-trace eligible seeds"
        )

    with ledger_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    if requested.strip().lower() != "auto" and tuple(eligible) != tuple(candidate_seeds):
        rejected = [row["seed"] for row in ledger if not row["eligible"]]
        raise FullTraceError(f"explicit seed list contains ineligible seeds: {rejected}")
    return tuple(eligible)


def run_seed(
    seed: int,
    out_dir: Path,
    response_matrix: np.ndarray,
    *,
    force: bool,
    max_iterations: int,
) -> dict[str, Any]:
    json_path = out_dir / f"seed_{seed:03d}.json"
    npz_path = out_dir / f"seed_{seed:03d}_terminal.npz"
    log_path = out_dir / f"seed_{seed:03d}.log"
    if json_path.exists() and npz_path.exists() and not force:
        return json.loads(json_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    hidden_q = initial_normalized_state(seed)
    zero_command = np.zeros(len(SCALES), dtype=float)
    plant = HighNAPlant(hidden_q=hidden_q)
    log_lines = [
        f"seed={seed}",
        "controller=model-based nominal-response acquisition + local damped Gauss-Newton",
        "state_truth_used_by_controller=false",
    ]
    initial_control = plant.evaluate(zero_command)
    log_lines.append(
        f"initial_control={json.dumps(evaluation_dict(initial_control), sort_keys=True)}"
    )
    acquisition_command, acquisition_control, acquisition_history = acquire(
        plant, response_matrix
    )
    log_lines.append(
        f"acquisition_control={json.dumps(evaluation_dict(acquisition_control), sort_keys=True)}"
    )
    final_command, final_control, local_history = refine(
        plant,
        acquisition_command,
        max_iterations=max_iterations,
    )
    log_lines.append(
        f"final_control={json.dumps(evaluation_dict(final_control), sort_keys=True)}"
    )

    # The paper metrics are recomputed on an independent, spatially uniform
    # Cartesian pupil and on held-out +/-0.100-mm fields.  All three fields are
    # traced together for each single saved command.
    audit_plant = HighNAPlant(
        hidden_q=hidden_q,
        fields_mm=AUDIT_FIELDS_MM,
        pupil_nodes=AUDIT_PUPIL_NODES,
        projector=AUDIT_PTTD_PROJECTOR,
    )
    audit_stages = {
        "initial": audit_plant.evaluate(zero_command),
        "acquisition": audit_plant.evaluate(acquisition_command),
        "final": audit_plant.evaluate(final_command),
    }
    rim_plant = HighNAPlant(
        hidden_q=hidden_q,
        fields_mm=AUDIT_FIELDS_MM,
        pupil_nodes=AUDIT_RIM_NODES,
        projector=np.eye(len(AUDIT_RIM_NODES)),
    )
    rim_stages = {
        "initial": rim_plant.evaluate(zero_command),
        "acquisition": rim_plant.evaluate(acquisition_command),
        "final": rim_plant.evaluate(final_command),
    }
    save_evaluation_arrays(npz_path, audit_plant, audit_stages, rim_plant, rim_stages)
    initial = audit_stages["initial"]
    acquisition = audit_stages["acquisition"]
    final = audit_stages["final"]
    log_lines.append(f"initial_audit={json.dumps(evaluation_dict(initial), sort_keys=True)}")
    log_lines.append(
        f"acquisition_audit={json.dumps(evaluation_dict(acquisition), sort_keys=True)}"
    )
    log_lines.append(f"final_audit={json.dumps(evaluation_dict(final), sort_keys=True)}")
    center_improved = bool(final.field_rms_waves[0] < initial.field_rms_waves[0])
    edge_improved = bool(np.max(final.field_rms_waves[1:]) < np.max(initial.field_rms_waves[1:]))
    full_trace = all(
        value.full_rays == value.expected_rays
        for value in list(audit_stages.values()) + list(rim_stages.values())
    )
    hidden_norm = float(np.linalg.norm(hidden_q))
    terminal_effective_norm = float(np.linalg.norm(hidden_q + final_command))
    norm_ratio = terminal_effective_norm / hidden_norm if hidden_norm > 0.0 else 0.0
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "dao-highna-seed-v1",
        "seed": int(seed),
        "generator": "numpy.random.RandomState(seed).uniform(-1,1,30)",
        "physical_limits": {
            "dx_dy_dz_mm": [-TRANSLATION_LIMIT_MM, TRANSLATION_LIMIT_MM],
            "tx_ty_deg": [-TILT_LIMIT_DEG, TILT_LIMIT_DEG],
        },
        "coordinate_order": [
            f"{group}.{coordinate}" for group in GROUPS for coordinate in COORDINATES
        ],
        "states": {
            "hidden_initial_normalized": hidden_q.tolist(),
            "hidden_initial_physical": state_from_normalized(hidden_q),
            "initial_command_normalized": zero_command.tolist(),
            "post_acquisition_command_normalized": acquisition_command.tolist(),
            "terminal_command_normalized": final_command.tolist(),
            "post_acquisition_effective_state_normalized": (
                hidden_q + acquisition_command
            ).tolist(),
            "terminal_effective_state_normalized": (hidden_q + final_command).tolist(),
            "terminal_effective_state_physical": state_from_normalized(
                hidden_q + final_command
            ),
        },
        "metrics": {name: evaluation_dict(value) for name, value in audit_stages.items()},
        "control_metrics": {
            "initial": evaluation_dict(initial_control),
            "post_acquisition": evaluation_dict(acquisition_control),
            "final": evaluation_dict(final_control),
        },
        "acquisition": {
            "name": "linearized nominal-response matrix acquisition",
            "state_truth_used": False,
            "rounds": acquisition_history,
        },
        "local_refinement": {
            "name": "stroke-normalized damped Gauss-Newton refinement",
            "state_truth_used": False,
            "iterations": local_history,
        },
        "evaluation_counts": {
            "controller_requests": int(plant.request_count),
            "unique_full_optical_evaluations": int(plant.trace_evaluation_count),
            "controller_ray_attempts_including_nominal": int(plant.ray_attempt_count),
            "controller_completed_rays_including_nominal": int(plant.ray_trace_count),
            "audit_unique_evaluations_excluding_nominal": int(audit_plant.trace_evaluation_count),
            "audit_ray_attempts_including_nominal": int(audit_plant.ray_attempt_count),
            "audit_completed_rays_including_nominal": int(audit_plant.ray_trace_count),
            "rim_unique_evaluations_excluding_nominal": int(rim_plant.trace_evaluation_count),
            "rim_ray_attempts_including_nominal": int(rim_plant.ray_attempt_count),
            "rim_completed_rays_including_nominal": int(rim_plant.ray_trace_count),
        },
        "same_terminal_state_center_edge": True,
        "posthoc_effective_state_norm_ratio": float(norm_ratio),
        "case_gates": {
            "center_improved": center_improved,
            "heldout_edge_max_improved": edge_improved,
            "all_reported_and_rim_rays_complete": full_trace,
            "audit_pass": bool(center_improved and edge_improved and full_trace),
        },
        "terminal_array_file": npz_path.name,
        "wall_seconds": float(elapsed),
    }
    _write_json(json_path, payload)
    log_lines.append(f"evaluation_counts={json.dumps(payload['evaluation_counts'], sort_keys=True)}")
    log_lines.append(f"wall_seconds={elapsed:.6f}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(
        f"seed {seed:03d}: center {initial.field_rms_waves[0]:.6f} -> "
        f"{acquisition.field_rms_waves[0]:.6f} -> {final.field_rms_waves[0]:.6f} waves; "
        f"edge mean {np.mean(initial.field_rms_waves[1:]):.6f} -> "
        f"{np.mean(acquisition.field_rms_waves[1:]):.6f} -> "
        f"{np.mean(final.field_rms_waves[1:]):.6f}; "
        f"{plant.trace_evaluation_count} optical evaluations, {elapsed:.1f} s",
        flush=True,
    )
    return payload


def build_response_matrix(out_dir: Path, *, force: bool) -> tuple[np.ndarray, dict[str, Any]]:
    path = out_dir / "nominal_response_matrix.npz"
    metadata_path = out_dir / "nominal_response_matrix.json"
    if path.exists() and metadata_path.exists() and not force:
        arrays = np.load(path)
        return np.asarray(arrays["jacobian_waves_per_normalized_coordinate"], float), json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    plant = HighNAPlant()
    started = time.perf_counter()
    jacobian = finite_difference_jacobian(
        plant,
        np.zeros(len(SCALES), dtype=float),
        step=1e-3,
        central=True,
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    metadata = {
        "method": "central finite difference at the nominal prescription",
        "step_normalized": 1e-3,
        "shape": list(jacobian.shape),
        "rank": int(np.linalg.matrix_rank(jacobian)),
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]),
        "full_trace_optical_evaluations": int(plant.trace_evaluation_count),
        "ray_attempt_count_including_nominal": int(plant.ray_attempt_count),
        "completed_ray_count_including_nominal": int(plant.ray_trace_count),
        "wall_seconds": float(time.perf_counter() - started),
    }
    np.savez_compressed(
        path,
        jacobian_waves_per_normalized_coordinate=jacobian,
        pupil_nodes=CONTROL_PUPIL_NODES,
        field_heights_mm=np.asarray(CONTROL_FIELDS_MM),
        coordinate_scales=SCALES,
    )
    _write_json(metadata_path, metadata)
    return jacobian, metadata


def write_summary(out_dir: Path, seeds: tuple[int, ...], response_metadata: dict[str, Any]) -> None:
    records = []
    for seed in seeds:
        path = out_dir / f"seed_{seed:03d}.json"
        if not path.exists():
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        return

    rows = []
    for record in records:
        initial = record["metrics"]["initial"]
        acquisition = record["metrics"]["acquisition"]
        final = record["metrics"]["final"]
        rows.append(
            {
                "seed": int(record["seed"]),
                "initial_center_waves": float(initial["center_rms_waves"]),
                "acquisition_center_waves": float(acquisition["center_rms_waves"]),
                "final_center_waves": float(final["center_rms_waves"]),
                "initial_edge_minus_waves": float(initial["negative_edge_rms_waves"]),
                "initial_edge_plus_waves": float(initial["positive_edge_rms_waves"]),
                "initial_edge_waves": float(initial["edge_max_rms_waves"]),
                "acquisition_edge_waves": float(acquisition["edge_max_rms_waves"]),
                "final_edge_minus_waves": float(final["negative_edge_rms_waves"]),
                "final_edge_plus_waves": float(final["positive_edge_rms_waves"]),
                "final_edge_waves": float(final["edge_max_rms_waves"]),
                "unique_full_optical_evaluations": int(
                    record["evaluation_counts"]["unique_full_optical_evaluations"]
                ),
                "wall_seconds": float(record["wall_seconds"]),
            }
        )
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    all_full = all(
        bool(record["case_gates"]["all_reported_and_rim_rays_complete"])
        for record in records
    )
    all_recovered = all(bool(record["case_gates"]["audit_pass"]) for record in records)
    complete = len(records) == len(seeds) == TARGET_CASE_COUNT
    acceptance_path = MODEL_DIR / "acceptance_results.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    source_path = Path(acceptance["source_zmx"])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "audit_pass": bool(
            complete and all_full and all_recovered and response_metadata["rank"] == 30
        ),
        "case_count": len(records),
        "metric_name": "nominal-referenced differential RMS WFE",
        "wavelength_nm": float(WAVELENGTH_NM),
        "sample_side_na": float(acceptance["maximum_traced_sample_side_na"]),
        "magnification": float(acceptance["signed_centroid_magnification"]),
        "field_coordinate": "sample-plane height",
        "field_unit": "mm",
        "center_field_value": 0.0,
        "edge_field_values": [-0.1, 0.1],
        "edge_statistic": "maximum of the two held-out +/-0.100-mm sample-plane edges",
        "active_coordinates": 30,
        "moving_groups": 6,
        "coordinate_basis": "per-group rigid-body dx, dy, dz (mm), tx, ty (deg), normalized by +/-0.150 mm and +/-1.5 deg strokes",
        "coordinate_order": [
            f"{group}.{coordinate}" for group in GROUPS for coordinate in COORDINATES
        ],
        "sensitivity_rank": int(response_metadata["rank"]),
        "full_stack_trace_verified": bool(all_full),
        "same_terminal_state_center_edge": True,
        "nominal_subtraction_pointwise": True,
        "removed_modes": ["piston", "tip", "tilt", "defocus"],
        "source_prescription": "Nikon-assigned US6519092B2 Embodiment 2, objective surfaces 1-24 and imaging-lens surfaces 28-33",
        "source_prescription_sha256": _sha256(source_path),
        "prescription_surfaces": int(acceptance["forward_prescription_surfaces"]),
        "traced_interfaces": int(acceptance["forward_interfaces_including_object_image"]),
        "full_trace_ray_count": int(
            len(AUDIT_FIELDS_MM) * (len(AUDIT_PUPIL_NODES) + len(AUDIT_RIM_NODES))
        ),
        "full_trace_expected_ray_count": int(
            len(AUDIT_FIELDS_MM) * (len(AUDIT_PUPIL_NODES) + len(AUDIT_RIM_NODES))
        ),
        "full_trace_count_scope": "per reported state: three audit fields, 197 RMS-grid nodes plus 28 nonduplicate rho=1 coverage nodes per field",
        "rms_grid_ray_count_per_state": int(len(AUDIT_FIELDS_MM) * len(AUDIT_PUPIL_NODES)),
        "rim_coverage_ray_count_per_state": int(len(AUDIT_FIELDS_MM) * len(AUDIT_RIM_NODES)),
        "reported_state_ray_count_all_cases": int(
            len(records)
            * 3
            * len(AUDIT_FIELDS_MM)
            * (len(AUDIT_PUPIL_NODES) + len(AUDIT_RIM_NODES))
        ),
        "actual_controller_ray_attempts_including_per_case_nominal": int(
            sum(
                record["evaluation_counts"]["controller_ray_attempts_including_nominal"]
                for record in records
            )
        ),
        "actual_audit_and_rim_ray_attempts_including_per_case_nominal": int(
            sum(
                record["evaluation_counts"]["audit_ray_attempts_including_nominal"]
                + record["evaluation_counts"]["rim_ray_attempts_including_nominal"]
                for record in records
            )
        ),
        "pupil_sampling": "independent 17x17 Cartesian unit-disk grid (197 spatially uniform RMS nodes) plus 28 nonduplicate 32-azimuth rho=1 full-NA coverage nodes at each audit field",
        "controller": "model-based linearized nominal-response acquisition followed by stroke-normalized damped Gauss-Newton local refinement",
        "control_fields_mm": list(CONTROL_FIELDS_MM),
        "control_pupil_sampling": "49 sparse nodes (center plus four 12-point rings); used only for feedback, never for paper WFE",
        "state_truth_used_by_controller": False,
        "initial_state_generator": "numpy.random.RandomState(seed).uniform(-1,1,30)",
        "seed_eligibility_rule": "first 11 non-negative seeds with complete initial traversal on all audit RMS-grid and independent full-NA rim rays; selected before recovery",
        "seeds": [int(record["seed"]) for record in records],
        "response_matrix": response_metadata,
    }
    _write_json(out_dir / "audit_metadata.json", metadata)

    summary = {
        "completed_cases": len(records),
        "requested_cases": len(seeds),
        "audit_pass": metadata["audit_pass"],
        "statistics": {},
    }
    for key in (
        "initial_center_waves",
        "acquisition_center_waves",
        "final_center_waves",
        "initial_edge_waves",
        "acquisition_edge_waves",
        "final_edge_waves",
    ):
        values = np.asarray([float(row[key]) for row in rows])
        summary["statistics"][key] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
        }
    _write_json(out_dir / "summary.json", summary)


def parse_seeds(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-response", action="store_true")
    parser.add_argument("--max-local-iterations", type=int, default=6)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    HERE.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        meta_path = HERE / "nominal_response_matrix.json"
        response_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        write_summary(HERE, seeds, response_metadata)
        return

    response_matrix, response_metadata = build_response_matrix(
        HERE,
        force=args.force_response,
    )
    print(
        f"response matrix: shape={response_matrix.shape}, "
        f"rank={response_metadata['rank']}, "
        f"condition={response_metadata['condition_number']:.3e}",
        flush=True,
    )
    for seed in seeds:
        run_seed(
            seed,
            HERE,
            response_matrix,
            force=args.force,
            max_iterations=args.max_local_iterations,
        )
        write_summary(HERE, seeds, response_metadata)


if __name__ == "__main__":
    main()
