"""Run paired, ray-traced Trepan2p controller ablations.

The optical metric is copied exactly from ``eval_11_cases_true_wrms.py``.
Success is fixed by the user: held-out nine-field mean WRMS must be strictly
below 0.07; values greater than or equal to 0.07 are failures.

The default endpoint for model/sensor ablations is the measured-gain stage.
This isolates the learned proposal and acquisition behavior without allowing
the expensive 40-DOF local optimizer to erase their differences. CTL-01 alone
recomputes that joint stage from each paired measured-gain state.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize

from benchmark_freeform_real_shwfs_residual_120 import _build_shwfs_measurement_model
from generate_speckle_dataset import _set_field_point, fresnel_propagate
from optical_model_rayoptics import RayOpticsPhysicsEngine
from trepan2p_selected_models import build_model, trainable_parameter_count


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "selected_trepan2p_ablation_protocol.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "trepan2p_selected_ablation_v1"
CHECKPOINT_ROOT = ARTIFACT_ROOT / "checkpoints"
RESULT_ROOT = ARTIFACT_ROOT / "results"
SUCCESS_THRESHOLD = 0.07
REFERENCE_FOCUS_MM = 4.647

SurfacePerturbation = collections.namedtuple(
    "SurfacePerturbation",
    ["dx_mm", "dy_mm", "dz_mm", "tilt_x_deg", "tilt_y_deg"],
)
NOMINAL = SurfacePerturbation(0.0, 0.0, 0.0, 0.0, 0.0)
FRONT_ANCHORS = [68, 70, 72, 74]
REAR_ANCHORS = [77, 79, 81, 84]
ALL_ANCHORS = FRONT_ANCHORS + REAR_ANCHORS

PAPER_RING = [
    (0.0, 0.0),
    (1.5, 0.0),
    (-1.5, 0.0),
    (0.0, 1.5),
    (0.0, -1.5),
    (1.06, 1.06),
    (-1.06, 1.06),
    (-1.06, -1.06),
    (1.06, -1.06),
]
TRAINING_GRID = [
    (-2.0, 0.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (2.0, 0.0),
    (0.0, -1.8),
    (0.0, -1.0),
    (0.0, 0.0),
    (0.0, 1.0),
    (0.0, 2.2),
]
WIDE_GRID = [
    (0.0, 0.0),
    (2.0, 0.0),
    (-2.0, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
    (1.414, 1.414),
    (-1.414, 1.414),
    (-1.414, -1.414),
    (1.414, -1.414),
]

BASELINE_ALIASES = {
    "paper_baseline",
    "sen01_all_nine",
    "net03_paper_baseline",
    "net04_blocks8",
    "dat01_n10000",
    "sen03_paper_nine_speckle",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def deterministic_scmos(image: np.ndarray, seed: int) -> np.ndarray:
    peak = float(np.max(image))
    if peak <= 0.0:
        return np.zeros_like(image, dtype=np.float32)
    rng = np.random.default_rng(seed)
    expected_e = np.clip(image * (4000.0 / peak), 0.0, 10000.0)
    expected_e *= rng.normal(1.0, 0.005, size=image.shape)
    shot_e = rng.poisson(np.clip(expected_e, 0.0, None)).astype(np.float64)
    read_e = rng.normal(0.0, 1.2, size=image.shape)
    dsnu = rng.normal(2.0, 0.5, size=image.shape)
    sensed_e = np.clip(shot_e + read_e + dsnu, 0.0, 10000.0)
    return np.clip(np.rint(sensed_e * 4.0 + 100.0), 0.0, 65535.0).astype(np.float32)


def build_engine(field: tuple[float, float]) -> RayOpticsPhysicsEngine:
    engine = RayOpticsPhysicsEngine(pupil_samples=64)
    engine.opm.update_model()
    _set_field_point(engine, float(field[0]), float(field[1]))
    return engine


class OpticalRuntime:
    def __init__(self, input_fields: list[tuple[float, float]], modality: str) -> None:
        self.input_fields = list(input_fields)
        self.modality = modality
        self.input_engines = [build_engine(field) for field in self.input_fields]
        # The paper controller scores gain and joint-refinement candidates at
        # field center even when the sensor stack begins with an off-axis view.
        self.center_engine = build_engine((0.0, 0.0))
        self.wide_engines = [build_engine(field) for field in WIDE_GRID]
        base = self.input_engines[0]
        self.wavelength_mm = float(base.wavelength_nm) * 1.0e-6
        self.wavelength_system = float(base.opm.nm_to_sys_units(base.wavelength_nm))
        self.dx_mm = float(base.pupil_diameter_mm) / 64.0
        self.diffuser = np.load(ROOT / "artifacts" / "diffuser_mask.npy")
        self.shwfs_models = None
        if modality == "shwfs":
            self.shwfs_models = [
                _build_shwfs_measurement_model(engine, focus_shift_mm=0.0)
                for engine in self.input_engines
            ]

    def compute_true_wrms(self, opd: np.ndarray, valid_mask: np.ndarray) -> float:
        ny, nx = opd.shape
        yy, xx = np.mgrid[-1:1:complex(ny), -1:1:complex(nx)]
        combined_mask = valid_mask & (np.sqrt(xx**2 + yy**2) <= 1.0)
        if not np.any(combined_mask):
            combined_mask = valid_mask
        y = yy[combined_mask]
        x = xx[combined_mask]
        z = np.asarray(opd, dtype=np.float64)[combined_mask]
        r2 = x**2 + y**2
        design = np.column_stack([x, y, r2, np.ones_like(x)])
        coefficients, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
        fit = coefficients[0] * x + coefficients[1] * y + coefficients[2] * r2 + coefficients[3]
        return float(np.std(z - fit) / self.wavelength_system)

    def center_wrms(self, state: dict[int, SurfacePerturbation]) -> float:
        engine = self.center_engine
        try:
            engine.set_surface_perturbations(state, clear_others=True)
            _, _, opd, valid = engine._sample_wavefront(
                num_rays=64,
                field_index=0,
                wavelength_nm=engine.wavelength_nm,
                focus=REFERENCE_FOCUS_MM,
            )
            if not np.any(valid):
                return 999.0
            return self.compute_true_wrms(opd, valid)
        except Exception:
            return 999.0

    def wide_wrms(self, state: dict[int, SurfacePerturbation]) -> tuple[float, list[float]]:
        values: list[float] = []
        for engine in self.wide_engines:
            try:
                engine.set_surface_perturbations(state, clear_others=True)
                _, _, opd, valid = engine._sample_wavefront(
                    num_rays=64,
                    field_index=0,
                    wavelength_nm=engine.wavelength_nm,
                    focus=REFERENCE_FOCUS_MM,
                )
                value = 999.0 if not np.any(valid) else self.compute_true_wrms(opd, valid)
            except Exception:
                value = 999.0
            values.append(float(value))
        return float(np.mean(values)), values

    def observe(
        self,
        state: dict[int, SurfacePerturbation],
        *,
        noise_seed: int,
        diffuser_shift_px: int = 0,
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        shifted_diffuser = np.roll(self.diffuser, shift=(diffuser_shift_px, diffuser_shift_px), axis=(0, 1))
        for field_index, engine in enumerate(self.input_engines):
            engine.set_surface_perturbations(state, clear_others=True)
            if self.modality == "shwfs":
                shwfs = self.shwfs_models[field_index]
                if hasattr(shwfs, "rng"):
                    shwfs.rng = np.random.default_rng(noise_seed + field_index)
                _, coefficients = shwfs.estimate_from_model(engine)
                rows.append(np.asarray(coefficients, dtype=np.float32))
                continue

            opd = np.asarray(engine.get_wavefront_opd(focus=REFERENCE_FOCUS_MM), dtype=np.float64)
            amplitude = (opd != 0.0).astype(np.float64)
            pupil = amplitude * np.exp(1j * (2.0 * np.pi / self.wavelength_mm) * opd)
            padded = np.zeros((128, 128), dtype=np.complex128)
            padded[32:96, 32:96] = pupil
            if self.modality == "speckle":
                sensor_field = fresnel_propagate(
                    padded * np.exp(1j * shifted_diffuser),
                    dx=self.dx_mm,
                    z=2.0,
                    wavelength=self.wavelength_mm,
                )
            elif self.modality == "psf":
                sensor_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
            else:
                raise ValueError(self.modality)
            rows.append(deterministic_scmos(np.abs(sensor_field) ** 2, noise_seed + field_index))
        return np.stack(rows, axis=0)


class ProposalPair:
    def __init__(self, experiment: str, device: torch.device, *, lazy_groups: bool = False) -> None:
        self.experiment = experiment
        self.device = device
        self.lazy_groups = bool(lazy_groups)
        self.models: dict[str, torch.nn.Module] = {}
        self.checkpoint_paths: dict[str, Path] = {}
        self.mech_mean: dict[str, np.ndarray] = {}
        self.mech_std: dict[str, np.ndarray] = {}
        self.feature_mean: dict[str, np.ndarray | None] = {}
        self.feature_std: dict[str, np.ndarray | None] = {}
        self.config: dict | None = None
        parameter_counts: list[int] = []
        for group in ("front", "rear"):
            if experiment in BASELINE_ALIASES:
                checkpoint_path = ROOT / "artifacts" / f"speckle_cnn_{group}.pth"
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                config = {
                    "model_kind": "paper_baseline",
                    "modality": "speckle",
                    "channel_indices": list(range(9)),
                    "in_channels": 9,
                    "latent_dim": 1024,
                    "mlp_width": 2048,
                    "mlp_blocks": 8,
                }
            else:
                checkpoint_path = CHECKPOINT_ROOT / f"{experiment}_{group}.pth"
                if not checkpoint_path.exists():
                    raise FileNotFoundError(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                config = dict(checkpoint["config"])
            self.checkpoint_paths[group] = checkpoint_path
            if self.config is None:
                self.config = config
            elif self.config != config:
                raise ValueError("Front/rear checkpoint configurations do not match")
            parameter_counts.append(
                int(
                    sum(
                        tensor.numel()
                        for key, tensor in checkpoint["state_dict"].items()
                        if not key.endswith(("running_mean", "running_var", "num_batches_tracked"))
                    )
                )
            )
            if not self.lazy_groups:
                model = build_model(config)
                model.load_state_dict(checkpoint["state_dict"])
                # Keep checkpoint weights on the CPU between proposal calls.
                # ``predict`` moves one model to the requested device for the
                # same FP32 forward pass and immediately moves it back.
                model.eval()
                self.models[group] = model
            self.mech_mean[group] = np.asarray(checkpoint["mech_mean"], dtype=np.float32)
            self.mech_std[group] = np.asarray(checkpoint["mech_std"], dtype=np.float32)
            self.feature_mean[group] = (
                None if checkpoint.get("feature_mean") is None else np.asarray(checkpoint["feature_mean"], dtype=np.float32)
            )
            self.feature_std[group] = (
                None if checkpoint.get("feature_std") is None else np.asarray(checkpoint["feature_std"], dtype=np.float32)
            )
            del checkpoint
        if len(set(parameter_counts)) != 1:
            raise ValueError("Front/rear checkpoint parameter counts do not match")
        self.parameter_count_per_model = parameter_counts[0]

    @property
    def modality(self) -> str:
        return str(self.config.get("modality", "speckle"))

    def _model_for_group(self, group: str) -> torch.nn.Module:
        model = self.models.get(group)
        if model is not None:
            return model
        checkpoint = torch.load(self.checkpoint_paths[group], map_location="cpu", weights_only=False)
        model = build_model(self.config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.models[group] = model
        del checkpoint
        gc.collect()
        return model

    def predict(self, group: str, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if self.modality in ("speckle", "psf"):
            maxima = np.max(values, axis=(-2, -1), keepdims=True)
            values = values / (maxima + 1.0e-6)
        else:
            values = (values - self.feature_mean[group]) / self.feature_std[group]
        model = self._model_for_group(group)
        model.to(self.device)
        try:
            tensor = torch.from_numpy(values).unsqueeze(0).to(self.device)
            with torch.no_grad():
                normalized = model(tensor).cpu().numpy()[0]
        finally:
            if self.device.type == "cuda":
                model.to("cpu")
                torch.cuda.empty_cache()
        return normalized * self.mech_std[group] + self.mech_mean[group]

    def release_group(self, group: str) -> None:
        """Release one group after its last proposal has been evaluated."""
        model = self.models.pop(group, None)
        if model is not None:
            del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def release(self) -> None:
        """Release proposal weights once a terminal task no longer needs them."""
        for group in list(self.models):
            self.release_group(group)


def fields_for_experiment(experiment: str, config: dict) -> list[tuple[float, float]]:
    if experiment == "sen01_one_center":
        return [(0.0, 0.0)]
    if experiment == "sen01_linear_three":
        return [(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]
    if experiment == "sen01_cross_five":
        return [(0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]
    if experiment == "sen01_all_nine":
        return TRAINING_GRID
    if str(config.get("family", "")) == "SEN-03" or experiment.startswith("sen03_"):
        return [(0.0, 0.0)] if experiment != "sen03_paper_nine_speckle" else PAPER_RING
    return PAPER_RING


def isolate_group(
    state: dict[int, SurfacePerturbation],
    anchors_to_keep: list[int],
) -> dict[int, SurfacePerturbation]:
    return {anchor: state[anchor] if anchor in anchors_to_keep else NOMINAL for anchor in ALL_ANCHORS}


def initial_state_for_seed(seed: int, scale: float = 1.0) -> dict[int, SurfacePerturbation]:
    translation_bound_mm = 0.100
    tilt_bound_deg = 2.5 / 60.0
    rng = np.random.RandomState(seed)
    state: dict[int, SurfacePerturbation] = {}
    for anchor in ALL_ANCHORS:
        state[anchor] = SurfacePerturbation(
            dx_mm=float(scale * rng.uniform(-translation_bound_mm, translation_bound_mm)),
            dy_mm=float(scale * rng.uniform(-translation_bound_mm, translation_bound_mm)),
            dz_mm=float(scale * rng.uniform(-translation_bound_mm, translation_bound_mm)),
            tilt_x_deg=float(scale * rng.uniform(-tilt_bound_deg, tilt_bound_deg)),
            tilt_y_deg=float(scale * rng.uniform(-tilt_bound_deg, tilt_bound_deg)),
        )
    return state


def vec_to_state_group(
    vector: np.ndarray,
    base_state: dict[int, SurfacePerturbation],
    group_anchors: list[int],
) -> dict[int, SurfacePerturbation]:
    result = copy.deepcopy(base_state)
    for index, anchor in enumerate(group_anchors):
        offset = index * 5
        result[anchor] = SurfacePerturbation(
            base_state[anchor].dx_mm - vector[offset],
            base_state[anchor].dy_mm - vector[offset + 1],
            base_state[anchor].dz_mm - vector[offset + 2],
            base_state[anchor].tilt_x_deg - vector[offset + 3],
            base_state[anchor].tilt_y_deg - vector[offset + 4],
        )
    return result


def combine_groups(
    initial_state: dict[int, SurfacePerturbation],
    front_state: dict[int, SurfacePerturbation],
    rear_state: dict[int, SurfacePerturbation],
) -> dict[int, SurfacePerturbation]:
    result = copy.deepcopy(initial_state)
    for anchor in FRONT_ANCHORS:
        result[anchor] = front_state[anchor]
    for anchor in REAR_ANCHORS:
        result[anchor] = rear_state[anchor]
    return result


def state_to_correction_vector(
    state: dict[int, SurfacePerturbation],
    initial_state: dict[int, SurfacePerturbation],
    anchors: list[int],
) -> np.ndarray:
    vector = np.zeros(5 * len(anchors), dtype=np.float64)
    for index, anchor in enumerate(anchors):
        offset = index * 5
        vector[offset : offset + 5] = [
            initial_state[anchor].dx_mm - state[anchor].dx_mm,
            initial_state[anchor].dy_mm - state[anchor].dy_mm,
            initial_state[anchor].dz_mm - state[anchor].dz_mm,
            initial_state[anchor].tilt_x_deg - state[anchor].tilt_x_deg,
            initial_state[anchor].tilt_y_deg - state[anchor].tilt_y_deg,
        ]
    return vector


def run_joint_refinement(
    *,
    measured_state: dict[int, SurfacePerturbation],
    initial_state: dict[int, SurfacePerturbation],
    runtime: OpticalRuntime,
) -> tuple[dict[int, SurfacePerturbation], dict]:
    """Recompute the paper joint stage from this run's paired measured state."""

    initial_vector = state_to_correction_vector(measured_state, initial_state, ALL_ANCHORS)
    objective_evaluations = 0

    def objective(vector: np.ndarray) -> float:
        nonlocal objective_evaluations
        objective_evaluations += 1
        candidate = vec_to_state_group(np.asarray(vector), initial_state, ALL_ANCHORS)
        return runtime.center_wrms(candidate)

    optimized = minimize(
        objective,
        initial_vector,
        method="L-BFGS-B",
        options={"maxiter": 30, "ftol": 1.0e-4, "eps": 1.0e-2},
    )
    final_state = vec_to_state_group(np.asarray(optimized.x), initial_state, ALL_ANCHORS)
    return final_state, {
        "joint_objective_evaluations": int(objective_evaluations),
        "joint_optimizer_success": bool(optimized.success),
        "joint_optimizer_message": str(optimized.message),
    }


def run_group_rounds(
    *,
    group: str,
    initial_state: dict[int, SurfacePerturbation],
    proposal_pair: ProposalPair,
    runtime: OpticalRuntime,
    policy: str,
    case_seed: int,
    diffuser_shift_px: int = 0,
) -> tuple[dict[int, SurfacePerturbation], list[dict], int]:
    anchors = FRONT_ANCHORS if group == "front" else REAR_ANCHORS
    group_offset = 100_000 if group == "front" else 200_000
    current_vector = np.zeros(20, dtype=np.float64)
    group_state = isolate_group(initial_state, anchors)
    logs: list[dict] = []
    objective_evaluations = 0

    for round_index, target_alpha in enumerate((1.0, 0.7, 0.3), start=1):
        observation = runtime.observe(
            group_state,
            noise_seed=case_seed * 1000 + group_offset + round_index * 100,
            diffuser_shift_px=diffuser_shift_px,
        )
        proposal = proposal_pair.predict(group, observation)

        if policy == "measured":
            objective_cache: dict[float, float] = {}

            def objective(alpha_array: np.ndarray) -> float:
                nonlocal objective_evaluations
                alpha = float(alpha_array[0])
                cache_key = round(alpha, 12)
                if cache_key in objective_cache:
                    return objective_cache[cache_key]
                objective_evaluations += 1
                candidate = vec_to_state_group(
                    current_vector + alpha * proposal,
                    initial_state,
                    anchors,
                )
                value = runtime.center_wrms(isolate_group(candidate, anchors))
                objective_cache[cache_key] = value
                return value

            grid = np.linspace(target_alpha - 0.2, target_alpha + 0.2, 5)
            grid_values = [(float(alpha), float(objective(np.asarray([alpha])))) for alpha in grid]
            best_grid_alpha, _ = min(grid_values, key=lambda item: item[1])
            optimized = minimize(
                objective,
                x0=[best_grid_alpha],
                method="BFGS",
                options={"eps": 1.0e-3, "maxiter": 5},
            )
            candidates = [
                (float(optimized.x[0]), float(optimized.fun), "BFGS"),
                (float(target_alpha), float(objective(np.asarray([target_alpha]))), "direct"),
                (0.0, float(objective(np.asarray([0.0]))), "zero"),
            ]
            accepted_alpha, accepted_center, accepted_source = min(candidates, key=lambda item: item[1])
        elif policy == "fixed_0p7":
            accepted_alpha = 0.7
            accepted_source = "fixed_0p7"
            candidate = vec_to_state_group(current_vector + accepted_alpha * proposal, initial_state, anchors)
            accepted_center = runtime.center_wrms(isolate_group(candidate, anchors))
        elif policy == "unit":
            accepted_alpha = 1.0
            accepted_source = "unit"
            candidate = vec_to_state_group(current_vector + accepted_alpha * proposal, initial_state, anchors)
            accepted_center = runtime.center_wrms(isolate_group(candidate, anchors))
        else:
            raise ValueError(policy)

        current_vector += accepted_alpha * proposal
        group_state = isolate_group(vec_to_state_group(current_vector, initial_state, anchors), anchors)
        logs.append(
            {
                "round": round_index,
                "target_alpha": target_alpha,
                "accepted_alpha": accepted_alpha,
                "accepted_source": accepted_source,
                "isolated_center_wrms": accepted_center,
            }
        )
    return group_state, logs, objective_evaluations


def raw_proposal_state(
    *,
    initial_state: dict[int, SurfacePerturbation],
    proposal_pair: ProposalPair,
    runtime: OpticalRuntime,
    case_seed: int,
    diffuser_shift_px: int = 0,
) -> dict[int, SurfacePerturbation]:
    group_states: dict[str, dict[int, SurfacePerturbation]] = {}
    for group, anchors, group_offset in (
        ("front", FRONT_ANCHORS, 100_000),
        ("rear", REAR_ANCHORS, 200_000),
    ):
        isolated = isolate_group(initial_state, anchors)
        observation = runtime.observe(
            isolated,
            noise_seed=case_seed * 1000 + group_offset + 100,
            diffuser_shift_px=diffuser_shift_px,
        )
        proposal = proposal_pair.predict(group, observation)
        group_states[group] = isolate_group(vec_to_state_group(proposal, initial_state, anchors), anchors)
    return combine_groups(initial_state, group_states["front"], group_states["rear"])


def run_gain_policy(
    *,
    initial_state: dict[int, SurfacePerturbation],
    proposal_pair: ProposalPair,
    runtime: OpticalRuntime,
    policy: str,
    case_seed: int,
    diffuser_shift_px: int = 0,
    release_proposal_groups: bool = False,
) -> tuple[dict[int, SurfacePerturbation], dict]:
    front_state, front_logs, front_evals = run_group_rounds(
        group="front",
        initial_state=initial_state,
        proposal_pair=proposal_pair,
        runtime=runtime,
        policy=policy,
        case_seed=case_seed,
        diffuser_shift_px=diffuser_shift_px,
    )
    if release_proposal_groups:
        proposal_pair.release_group("front")
    rear_state, rear_logs, rear_evals = run_group_rounds(
        group="rear",
        initial_state=initial_state,
        proposal_pair=proposal_pair,
        runtime=runtime,
        policy=policy,
        case_seed=case_seed,
        diffuser_shift_px=diffuser_shift_px,
    )
    if release_proposal_groups:
        proposal_pair.release_group("rear")
    return combine_groups(initial_state, front_state, rear_state), {
        "front_rounds": front_logs,
        "rear_rounds": rear_logs,
        "control_objective_evaluations": int(front_evals + rear_evals),
    }


def metrics_row(
    *,
    study: str,
    experiment: str,
    seed: int,
    endpoint: str,
    state: dict[int, SurfacePerturbation],
    runtime: OpticalRuntime,
    elapsed_seconds: float,
    extra: dict | None = None,
) -> dict:
    center = runtime.center_wrms(state)
    wide_mean, wide_values = runtime.wide_wrms(state)
    row = {
        "protocol_version": "trepan2p-selected-v1",
        "study": study,
        "experiment": experiment,
        "seed": int(seed),
        "endpoint": endpoint,
        "center_wrms": center,
        "wide_mean_wrms": wide_mean,
        "wide_field_wrms": wide_values,
        "success_threshold_wrms": SUCCESS_THRESHOLD,
        "success": bool(wide_mean < SUCCESS_THRESHOLD),
        "elapsed_seconds": float(elapsed_seconds),
    }
    if extra:
        row.update(extra)
    return row


class ResultWriter:
    def __init__(self, name: str) -> None:
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = RESULT_ROOT / f"{name}.jsonl"
        self.csv_path = RESULT_ROOT / f"{name}.csv"
        self.rows: list[dict] = []
        if self.jsonl_path.exists():
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.rows.append(json.loads(line))

    def completed(self, marker_key: str) -> bool:
        return any(row.get("marker_key") == marker_key and row.get("endpoint") == "complete" for row in self.rows)

    def append_many(self, rows: list[dict], marker_key: str) -> None:
        marker = {
            "protocol_version": "trepan2p-selected-v1",
            "endpoint": "complete",
            "marker_key": marker_key,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            for row in rows + [marker]:
                handle.write(json.dumps(row, default=_json_default, sort_keys=True) + "\n")
                handle.flush()
        self.rows.extend(rows + [marker])
        self.write_csv()

    def write_csv(self) -> None:
        flat_rows: list[dict] = []
        for row in self.rows:
            if row.get("endpoint") == "complete":
                continue
            flat_rows.append(
                {
                    "study": row.get("study"),
                    "experiment": row.get("experiment"),
                    "seed": row.get("seed"),
                    "endpoint": row.get("endpoint"),
                    "policy": row.get("policy"),
                    "scale": row.get("scale"),
                    "diffuser_shift_px": row.get("diffuser_shift_px"),
                    "center_wrms": row.get("center_wrms"),
                    "wide_mean_wrms": row.get("wide_mean_wrms"),
                    "success_threshold_wrms": row.get("success_threshold_wrms"),
                    "success": row.get("success"),
                    "parameter_count_per_model": row.get("parameter_count_per_model"),
                    "control_objective_evaluations": row.get("control_objective_evaluations"),
                    "joint_objective_evaluations": row.get("joint_objective_evaluations"),
                    "joint_optimizer_success": row.get("joint_optimizer_success"),
                    "joint_start": row.get("joint_start"),
                    "reused_from": row.get("reused_from"),
                    "elapsed_seconds_semantics": row.get("elapsed_seconds_semantics"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                }
            )
        fieldnames = list(flat_rows[0]) if flat_rows else ["study"]
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def tagged_name(base: str, result_tag: str) -> str:
    return base if not result_tag else f"{base}__{result_tag}"


def reusable_result(
    result_name: str,
    *,
    seed: int,
    study: str,
    endpoint: str,
    policy: str | None = None,
) -> dict:
    """Load one previously evaluated row whose optical experiment is identical.

    Reuse is intentionally limited to exact protocol identities: zero diffuser
    displacement is the frozen baseline acquisition, and ROB-01 at unit scale
    is the CTL-01 measured-then-joint endpoint for the same paired seed.
    """
    source_path = RESULT_ROOT / f"{result_name}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"Required reusable result is missing: {source_path}")
    matches: list[dict] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            int(row.get("seed", -1)) == int(seed)
            and row.get("study") == study
            and row.get("endpoint") == endpoint
            and (policy is None or row.get("policy") == policy)
        ):
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one reusable row in {source_path} for "
            f"seed={seed}, study={study}, endpoint={endpoint}, policy={policy}; "
            f"found {len(matches)}"
        )
    return dict(matches[0])


def run_ctl(seeds: list[int], *, device: torch.device, result_tag: str = "") -> None:
    writer = ResultWriter(tagged_name("ctl01_ctl02", result_tag))
    proposals = ProposalPair("paper_baseline", device)
    runtime = OpticalRuntime(PAPER_RING, modality="speckle")

    for seed in seeds:
        marker_key = f"ctl-seed-{seed}"
        if writer.completed(marker_key):
            print(f"SKIP {marker_key}", flush=True)
            continue
        case_start = time.perf_counter()
        initial = initial_state_for_seed(seed)
        rows = [
            metrics_row(
                study="CTL-01",
                experiment="paper_baseline",
                seed=seed,
                endpoint="initial",
                state=initial,
                runtime=runtime,
                elapsed_seconds=time.perf_counter() - case_start,
            )
        ]
        raw_state = raw_proposal_state(
            initial_state=initial,
            proposal_pair=proposals,
            runtime=runtime,
            case_seed=seed,
        )
        rows.append(
            metrics_row(
                study="CTL-01",
                experiment="paper_baseline",
                seed=seed,
                endpoint="cnn_only",
                state=raw_state,
                runtime=runtime,
                elapsed_seconds=time.perf_counter() - case_start,
            )
        )
        measured_row = None
        measured_state = None
        for policy in ("measured", "fixed_0p7", "unit"):
            policy_state, details = run_gain_policy(
                initial_state=initial,
                proposal_pair=proposals,
                runtime=runtime,
                policy=policy,
                case_seed=seed,
            )
            row = metrics_row(
                study="CTL-02",
                experiment="paper_baseline",
                seed=seed,
                endpoint="gain_acquisition",
                state=policy_state,
                runtime=runtime,
                elapsed_seconds=time.perf_counter() - case_start,
                extra={"policy": policy, **details},
            )
            rows.append(row)
            if policy == "measured":
                measured_row = dict(row)
                measured_state = policy_state

        if measured_row is not None:
            measured_row["study"] = "CTL-01"
            measured_row["endpoint"] = "measured_gain"
            rows.append(measured_row)
        if measured_state is None:
            raise RuntimeError("The paired measured-gain state was not produced")
        joint_state, joint_details = run_joint_refinement(
            measured_state=measured_state,
            initial_state=initial,
            runtime=runtime,
        )
        joint_row = metrics_row(
            study="CTL-01",
            experiment="paper_baseline",
            seed=seed,
            endpoint="joint_refinement",
            state=joint_state,
            runtime=runtime,
            elapsed_seconds=time.perf_counter() - case_start,
            extra={"joint_start": "paired_measured_gain", **joint_details},
        )
        rows.append(joint_row)
        writer.append_many(rows, marker_key)
        print(
            f"DONE {marker_key} measured={measured_row['wide_mean_wrms']:.6f} "
            f"joint={joint_row['wide_mean_wrms']:.6f}",
            flush=True,
        )


def run_model(experiment: str, seeds: list[int], *, device: torch.device, result_tag: str = "") -> None:
    writer = ResultWriter(tagged_name(f"model_{experiment}", result_tag))
    proposals = ProposalPair(experiment, device)
    fields = fields_for_experiment(experiment, proposals.config)
    runtime = OpticalRuntime(fields, modality=proposals.modality)
    for seed in seeds:
        marker_key = f"model-{experiment}-seed-{seed}"
        if writer.completed(marker_key):
            print(f"SKIP {marker_key}", flush=True)
            continue
        start = time.perf_counter()
        initial = initial_state_for_seed(seed)
        endpoint_state, details = run_gain_policy(
            initial_state=initial,
            proposal_pair=proposals,
            runtime=runtime,
            policy="measured",
            case_seed=seed,
        )
        row = metrics_row(
            study=str(proposals.config.get("family", "baseline")),
            experiment=experiment,
            seed=seed,
            endpoint="measured_gain",
            state=endpoint_state,
            runtime=runtime,
            elapsed_seconds=time.perf_counter() - start,
            extra={
                "policy": "measured",
                "parameter_count_per_model": proposals.parameter_count_per_model,
                "input_fields": fields,
                **details,
            },
        )
        writer.append_many([row], marker_key)
        print(
            f"DONE {marker_key} wide={row['wide_mean_wrms']:.6f} "
            f"success={row['success']}",
            flush=True,
        )


def run_rob01(
    seeds: list[int],
    scales: list[float],
    *,
    device: torch.device,
    result_tag: str = "",
) -> None:
    writer = ResultWriter(tagged_name("rob01_capture_range", result_tag))
    proposals: ProposalPair | None = None
    runtime: OpticalRuntime | None = None
    for scale in scales:
        for seed in seeds:
            marker_key = f"rob01-scale-{scale:g}-seed-{seed}"
            if writer.completed(marker_key):
                print(f"SKIP {marker_key}", flush=True)
                continue
            if float(scale) == 1.0:
                row = reusable_result(
                    "ctl01_ctl02",
                    seed=seed,
                    study="CTL-01",
                    endpoint="joint_refinement",
                )
                row.update(
                    {
                        "study": "ROB-01",
                        "experiment": "paper_baseline",
                        "scale": float(scale),
                        "policy": "measured_then_joint",
                        "reused_from": "ctl01_ctl02:CTL-01/joint_refinement",
                        "elapsed_seconds_semantics": "source_evaluation",
                    }
                )
                writer.append_many([row], marker_key)
                print(
                    f"REUSE {marker_key} wide={row['wide_mean_wrms']:.6f} "
                    f"success={row['success']}",
                    flush=True,
                )
                continue
            if proposals is None:
                proposals = ProposalPair("paper_baseline", device, lazy_groups=True)
            if runtime is None:
                runtime = OpticalRuntime(PAPER_RING, modality="speckle")
            start = time.perf_counter()
            initial = initial_state_for_seed(seed, scale=scale)
            endpoint_state, details = run_gain_policy(
                initial_state=initial,
                proposal_pair=proposals,
                runtime=runtime,
                policy="measured",
                # Keep sensor-noise draws paired across perturbation scales so
                # the phase boundary reflects capture magnitude, not a new
                # noise realization at every column.
                case_seed=seed,
                release_proposal_groups=True,
            )
            # The proposal networks are not consulted during the 40-DOF joint
            # refinement.  Releasing both 272.1 M-parameter models here avoids
            # retaining roughly 2.2 GB of idle FP32 weights per worker.  This
            # occurs only after every proposal vector and measured gain needed
            # by this case has already been fixed, so the numerical path below
            # is unchanged.
            proposals.release()
            proposals = None
            joint_state, joint_details = run_joint_refinement(
                measured_state=endpoint_state,
                initial_state=initial,
                runtime=runtime,
            )
            row = metrics_row(
                study="ROB-01",
                experiment="paper_baseline",
                seed=seed,
                endpoint="joint_refinement",
                state=joint_state,
                runtime=runtime,
                elapsed_seconds=time.perf_counter() - start,
                extra={
                    "scale": scale,
                    "policy": "measured_then_joint",
                    **details,
                    **joint_details,
                },
            )
            writer.append_many([row], marker_key)
            print(f"DONE {marker_key} wide={row['wide_mean_wrms']:.6f} success={row['success']}", flush=True)


def run_diffuser_drift(
    seeds: list[int],
    shifts: list[int],
    *,
    device: torch.device,
    result_tag: str = "",
) -> None:
    writer = ResultWriter(tagged_name("sen03_diffuser_drift", result_tag))
    proposals: ProposalPair | None = None
    runtime: OpticalRuntime | None = None
    for shift in shifts:
        for seed in seeds:
            marker_key = f"drift-px-{shift}-seed-{seed}"
            if writer.completed(marker_key):
                print(f"SKIP {marker_key}", flush=True)
                continue
            if int(shift) == 0:
                row = reusable_result(
                    "model_paper_baseline",
                    seed=seed,
                    study="baseline",
                    endpoint="measured_gain",
                    policy="measured",
                )
                row.update(
                    {
                        "study": "SEN-03",
                        "experiment": "diffuser_drift",
                        "diffuser_shift_px": int(shift),
                        "policy": "measured",
                        "reused_from": "model_paper_baseline:baseline/measured_gain",
                        "elapsed_seconds_semantics": "source_evaluation",
                    }
                )
                writer.append_many([row], marker_key)
                print(
                    f"REUSE {marker_key} wide={row['wide_mean_wrms']:.6f} "
                    f"success={row['success']}",
                    flush=True,
                )
                continue
            if proposals is None:
                proposals = ProposalPair("paper_baseline", device)
            if runtime is None:
                runtime = OpticalRuntime(PAPER_RING, modality="speckle")
            start = time.perf_counter()
            initial = initial_state_for_seed(seed)
            endpoint_state, details = run_gain_policy(
                initial_state=initial,
                proposal_pair=proposals,
                runtime=runtime,
                policy="measured",
                case_seed=seed,
                diffuser_shift_px=shift,
            )
            row = metrics_row(
                study="SEN-03",
                experiment="diffuser_drift",
                seed=seed,
                endpoint="measured_gain",
                state=endpoint_state,
                runtime=runtime,
                elapsed_seconds=time.perf_counter() - start,
                extra={"diffuser_shift_px": shift, "policy": "measured", **details},
            )
            writer.append_many([row], marker_key)
            print(f"DONE {marker_key} wide={row['wide_mean_wrms']:.6f} success={row['success']}", flush=True)


def parse_number_list(text: str, kind: type) -> list:
    return [kind(token.strip()) for token in text.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=["ctl", "model", "rob01", "drift"], required=True)
    parser.add_argument("--experiment", default="paper_baseline")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in protocol["paper_case_seeds"]))
    parser.add_argument("--scales", default=",".join(str(value) for value in protocol["rob01_scale_factors"]))
    parser.add_argument("--shifts", default=",".join(str(value) for value in protocol["diffuser_drift_pixels"]))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--result-tag", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_number_list(args.seeds, int)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.study == "ctl":
        run_ctl(seeds, device=device, result_tag=args.result_tag)
    elif args.study == "model":
        run_model(args.experiment, seeds, device=device, result_tag=args.result_tag)
    elif args.study == "rob01":
        run_rob01(
            seeds,
            parse_number_list(args.scales, float),
            device=device,
            result_tag=args.result_tag,
        )
    elif args.study == "drift":
        run_diffuser_drift(
            seeds,
            parse_number_list(args.shifts, int),
            device=device,
            result_tag=args.result_tag,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
