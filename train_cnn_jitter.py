"""Train a lightweight PSF CNN for phase-to-compensator prediction on 2P_AO.

This generator keeps the physical roles separate:

- Random jitter is applied only to non-compensator lens elements to create
  realistic wavefront/PSF errors.
- The CNN target is only the compensator actuator position on surface 74
  (representing the 74-76 group) needed to counter that phase error.

The label is derived from the disturbed wavefront through a pseudo-inverse
Jacobian from weighted-Bessel coefficients to compensator motion. Samples whose
derived compensator command would exceed the mechanical travel are rejected so
the training set stays inside the physically reachable envelope.

Important environment note for this Windows ao311 setup:
import Torch before NumPy or SciPy in the training process.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from feedback_controller import COMPENSATOR_SURFACE_INDEX, FeedbackController, build_lightweight_cnn
from optical_model_rayoptics import RayOpticsPhysicsEngine, SHIFT_LIMIT_MM
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1.0e-3
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_JITTER_LIMIT_MM = 0.01
DEFAULT_ACTIVATION_PROBABILITY = 0.15
DEFAULT_PUPIL_SAMPLES = 48
DEFAULT_FFT_SAMPLES = 256
DEFAULT_RANDOM_SEED = 20260412
DEFAULT_JACOBIAN_STEP_MM = 0.05
DEFAULT_COMPENSATOR_GROUP_SURFACES = (74, 75, 76)
DEFAULT_MAX_ACTIVE_SURFACES = 1
DEFAULT_PINV_RCOND = 1.0e-2
DEFAULT_TRAINING_NUM_MODES = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CNN that predicts the surface-74 compensator position."
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--jitter-limit-mm", type=float, default=DEFAULT_JITTER_LIMIT_MM)
    parser.add_argument(
        "--activation-probability",
        type=float,
        default=DEFAULT_ACTIVATION_PROBABILITY,
        help="Probability that an individual disturbance surface is active in a sample.",
    )
    parser.add_argument("--pupil-samples", type=int, default=DEFAULT_PUPIL_SAMPLES)
    parser.add_argument("--fft-samples", type=int, default=DEFAULT_FFT_SAMPLES)
    parser.add_argument("--num-modes", type=int, default=DEFAULT_TRAINING_NUM_MODES)
    parser.add_argument("--jacobian-step-mm", type=float, default=DEFAULT_JACOBIAN_STEP_MM)
    parser.add_argument("--max-active-surfaces", type=int, default=DEFAULT_MAX_ACTIVE_SURFACES)
    parser.add_argument("--pinv-rcond", type=float, default=DEFAULT_PINV_RCOND)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument(
        "--disturbance-surface-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional explicit disturbance surfaces. Defaults to all movable elements except 74-76.",
    )
    parser.add_argument(
        "--compensator-surface",
        type=int,
        default=COMPENSATOR_SURFACE_INDEX,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return parser.parse_args()


def sample_sparse_jitter(
    rng: np.random.Generator,
    num_surfaces: int,
    jitter_limit_mm: float,
    activation_probability: float,
    max_active_surfaces: int,
) -> np.ndarray:
    if max_active_surfaces <= 0:
        raise ValueError("max_active_surfaces must be positive.")

    perturbations = np.zeros((num_surfaces, 3), dtype=np.float64)
    active_mask = rng.random(num_surfaces) < activation_probability
    if not np.any(active_mask):
        active_mask[int(rng.integers(0, num_surfaces))] = True
    active_indices = np.flatnonzero(active_mask)
    if active_indices.size > max_active_surfaces:
        active_indices = np.sort(
            rng.choice(active_indices, size=max_active_surfaces, replace=False)
        )
        active_mask[:] = False
        active_mask[active_indices] = True

    perturbations[active_mask] = rng.uniform(
        low=-jitter_limit_mm,
        high=jitter_limit_mm,
        size=(int(active_mask.sum()), 3),
    )
    return perturbations


def estimate_compensator_jacobian(
    controller: FeedbackController,
    *,
    num_modes: int,
    step_mm: float,
) -> np.ndarray:
    if step_mm <= 0.0:
        raise ValueError("jacobian step must be positive.")

    base_position = np.zeros(3, dtype=np.float64)
    controller.set_compensator_position(base_position)
    base_coeffs = controller.extract_w_bessel_coeffs(num_modes=num_modes)
    jacobian = np.zeros((num_modes, 3), dtype=np.float64)

    for axis_index in range(3):
        offset_position = base_position.copy()
        offset_position[axis_index] += step_mm
        controller.set_compensator_position(offset_position)
        coeffs_offset = controller.extract_w_bessel_coeffs(num_modes=num_modes)
        jacobian[:, axis_index] = (coeffs_offset - base_coeffs) / step_mm

    controller.set_compensator_position(base_position)
    if np.linalg.norm(jacobian) <= 1.0e-12:
        raise RuntimeError("Compensator Jacobian is singular at the nominal configuration.")
    return jacobian


def select_disturbance_surface_ids(
    engine: RayOpticsPhysicsEngine,
    requested_surface_ids: list[int] | None,
) -> list[int]:
    if requested_surface_ids:
        surface_ids = [int(surface_id) for surface_id in requested_surface_ids]
    else:
        excluded = set(DEFAULT_COMPENSATOR_GROUP_SURFACES)
        surface_ids = [
            surface_id
            for surface_id in engine.get_movable_surface_ids(
                include_coverglass=True,
                include_sample_media=False,
            )
            if surface_id not in excluded
        ]

    if not surface_ids:
        raise ValueError("No disturbance surfaces were selected for dataset generation.")
    return surface_ids


def build_dataset(
    *,
    num_samples: int,
    disturbance_surface_ids: list[int],
    compensator_surface_id: int,
    jitter_limit_mm: float,
    activation_probability: float,
    seed: int,
    pupil_samples: int,
    fft_samples: int,
    num_modes: int,
    jacobian_step_mm: float,
    max_active_surfaces: int,
    pinv_rcond: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if jitter_limit_mm <= 0.0:
        raise ValueError("jitter_limit_mm must be positive.")
    if jitter_limit_mm > SHIFT_LIMIT_MM:
        raise ValueError(
            f"jitter_limit_mm={jitter_limit_mm} exceeds the physical shift limit of {SHIFT_LIMIT_MM} mm."
        )

    engine = RayOpticsPhysicsEngine(
        design_name="2P_AO",
        pupil_samples=pupil_samples,
        fft_samples=fft_samples,
    )
    controller = FeedbackController(
        engine,
        architecture="lightweight_cnn",
        compensator_surface_index=compensator_surface_id,
        prediction_mode="absolute",
        device="cpu",
    )
    jacobian = estimate_compensator_jacobian(
        controller,
        num_modes=num_modes,
        step_mm=jacobian_step_mm,
    )
    jacobian_pinv = np.linalg.pinv(jacobian, rcond=pinv_rcond)

    disturbance_descriptions = {
        str(surface_id): engine.surface_descriptions[
            engine.get_prescription_surface_index(surface_id)
        ]
        for surface_id in disturbance_surface_ids
    }
    compensator_description = engine.surface_descriptions[
        engine.get_prescription_surface_index(compensator_surface_id)
    ]

    images = np.zeros((num_samples, fft_samples, fft_samples), dtype=np.float32)
    targets = np.zeros((num_samples, 3), dtype=np.float32)
    coefficients = np.zeros((num_samples, num_modes), dtype=np.float32)

    rng = np.random.default_rng(seed)
    sample_index = 0
    rejected_limit = 0
    rejected_trace = 0
    attempt_count = 0
    max_attempts = max(num_samples * 25, 1000)
    t0 = time.perf_counter()

    while sample_index < num_samples:
        attempt_count += 1
        if attempt_count > max_attempts:
            raise RuntimeError(
                "Unable to generate enough physically reachable samples. "
                "Try reducing jitter_limit_mm or activation_probability."
            )

        controller.set_compensator_position(np.zeros(3, dtype=np.float64))
        perturbation_matrix = sample_sparse_jitter(
            rng=rng,
            num_surfaces=len(disturbance_surface_ids),
            jitter_limit_mm=jitter_limit_mm,
            activation_probability=activation_probability,
            max_active_surfaces=max_active_surfaces,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                engine.set_surface_perturbations(
                    {
                        surface_id: perturbation_matrix[row_index]
                        for row_index, surface_id in enumerate(disturbance_surface_ids)
                    }
                )
                psf_before = engine.get_psf_image(
                    pupil_samples=pupil_samples,
                    fft_samples=fft_samples,
                )
                coeff_vector = controller.extract_w_bessel_coeffs(num_modes=num_modes)
        except (RuntimeError, RuntimeWarning, ValueError, FloatingPointError):
            rejected_trace += 1
            continue

        compensator_target = -jacobian_pinv @ coeff_vector

        if (
            not np.all(np.isfinite(compensator_target))
            or np.any(np.abs(compensator_target) > SHIFT_LIMIT_MM)
        ):
            rejected_limit += 1
            continue

        images[sample_index] = psf_before
        targets[sample_index] = compensator_target.astype(np.float32, copy=False)
        coefficients[sample_index] = coeff_vector.astype(np.float32, copy=False)
        sample_index += 1

        if sample_index % 25 == 0 or sample_index == 1:
            elapsed = time.perf_counter() - t0
            avg_seconds = elapsed / float(sample_index)
            print(
                f"[dataset] {sample_index:4d}/{num_samples} "
                f"avg={avg_seconds:.3f}s/sample "
                f"rejected_limit={rejected_limit} rejected_trace={rejected_trace}"
            )

    engine.clear_perturbation()
    controller.set_compensator_position(np.zeros(3, dtype=np.float64))

    metadata: dict[str, object] = {
        "disturbance_surface_ids": [int(surface_id) for surface_id in disturbance_surface_ids],
        "disturbance_surface_descriptions": disturbance_descriptions,
        "compensator_surface_id": int(compensator_surface_id),
        "compensator_surface_description": compensator_description,
        "jacobian_matrix": jacobian,
        "rejected_limit_samples": int(rejected_limit),
        "rejected_trace_samples": int(rejected_trace),
        "attempt_count": int(attempt_count),
    }
    return images, targets, coefficients, metadata


def train_model(
    *,
    images: np.ndarray,
    targets: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[float], list[float]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie strictly between 0 and 1.")

    image_tensor = torch.from_numpy(images[:, None, :, :])
    target_tensor = torch.from_numpy(targets)
    dataset = TensorDataset(image_tensor, target_tensor)

    val_size = max(1, int(round(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples left for the training split.")

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_lightweight_cnn(output_dim=targets.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_history: list[float] = []
    val_history: list[float] = []

    for epoch_index in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_images, batch_targets in train_loader:
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_images)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch_images.shape[0]
            train_loss_sum += float(loss.item()) * batch_size_actual
            train_count += batch_size_actual

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for batch_images, batch_targets in val_loader:
                batch_images = batch_images.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_images)
                loss = criterion(predictions, batch_targets)
                batch_size_actual = batch_images.shape[0]
                val_loss_sum += float(loss.item()) * batch_size_actual
                val_count += batch_size_actual

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        train_history.append(train_loss)
        val_history.append(val_loss)
        print(
            f"[train] epoch {epoch_index + 1:02d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

    return model, train_history, val_history


def save_artifacts(
    *,
    output_dir: Path,
    images: np.ndarray,
    targets: np.ndarray,
    coefficients: np.ndarray,
    model: nn.Module,
    metadata: dict[str, object],
    train_history: list[float],
    val_history: list[float],
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_tag = f"{int(args.num_samples)}samples"
    dataset_path = output_dir / f"2p_ao_phase_to_comp74_{sample_tag}.npz"
    checkpoint_path = output_dir / f"2p_ao_phase_to_comp74_lightweight_cnn_{sample_tag}.pt"

    np.savez_compressed(
        dataset_path,
        psf_images=images,
        target_compensator_mm=targets,
        w_bessel_coeffs=coefficients,
        disturbance_surface_ids=np.asarray(
            metadata["disturbance_surface_ids"],
            dtype=np.int32,
        ),
        compensator_surface_id=np.asarray([metadata["compensator_surface_id"]], dtype=np.int32),
    )

    checkpoint = {
        "architecture": "lightweight_cnn",
        "model_state_dict": model.state_dict(),
        "output_dim": int(targets.shape[1]),
        "actuator_surface_indices": [int(metadata["compensator_surface_id"])],
        "prediction_mode": "absolute",
        "target_semantics": "absolute compensator position derived from wavefront coefficients",
        "disturbance_surface_ids": metadata["disturbance_surface_ids"],
        "disturbance_surface_descriptions": metadata["disturbance_surface_descriptions"],
        "compensator_surface_id": int(metadata["compensator_surface_id"]),
        "compensator_surface_description": metadata["compensator_surface_description"],
        "jacobian_matrix": np.asarray(metadata["jacobian_matrix"], dtype=np.float64),
        "rejected_limit_samples": int(metadata["rejected_limit_samples"]),
        "rejected_trace_samples": int(metadata["rejected_trace_samples"]),
        "attempt_count": int(metadata["attempt_count"]),
        "num_samples": int(args.num_samples),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "val_fraction": float(args.val_fraction),
        "jitter_limit_mm": float(args.jitter_limit_mm),
        "activation_probability": float(args.activation_probability),
        "num_modes": int(args.num_modes),
        "jacobian_step_mm": float(args.jacobian_step_mm),
        "max_active_surfaces": int(args.max_active_surfaces),
        "pinv_rcond": float(args.pinv_rcond),
        "pupil_samples": int(args.pupil_samples),
        "fft_samples": int(args.fft_samples),
        "seed": int(args.seed),
        "image_shape": list(images.shape[1:]),
        "train_loss_history": train_history,
        "val_loss_history": val_history,
    }
    torch.save(checkpoint, checkpoint_path)
    return dataset_path, checkpoint_path


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    preview_engine = RayOpticsPhysicsEngine(
        design_name="2P_AO",
        pupil_samples=args.pupil_samples,
        fft_samples=args.fft_samples,
    )
    disturbance_surface_ids = select_disturbance_surface_ids(
        preview_engine,
        args.disturbance_surface_ids,
    )
    preview_engine.clear_perturbation()

    print(f"[config] device={device}")
    print(f"[config] disturbance_surfaces={disturbance_surface_ids}")
    print(f"[config] compensator_surface={args.compensator_surface}")
    print(
        f"[config] num_samples={args.num_samples} epochs={args.epochs} "
        f"pupil_samples={args.pupil_samples} fft_samples={args.fft_samples} "
        f"num_modes={args.num_modes} max_active_surfaces={args.max_active_surfaces} "
        f"pinv_rcond={args.pinv_rcond}"
    )

    dataset_t0 = time.perf_counter()
    images, targets, coefficients, metadata = build_dataset(
        num_samples=args.num_samples,
        disturbance_surface_ids=disturbance_surface_ids,
        compensator_surface_id=args.compensator_surface,
        jitter_limit_mm=args.jitter_limit_mm,
        activation_probability=args.activation_probability,
        seed=args.seed,
        pupil_samples=args.pupil_samples,
        fft_samples=args.fft_samples,
        num_modes=args.num_modes,
        jacobian_step_mm=args.jacobian_step_mm,
        max_active_surfaces=args.max_active_surfaces,
        pinv_rcond=args.pinv_rcond,
    )
    dataset_elapsed = time.perf_counter() - dataset_t0
    print(
        f"[dataset] complete in {dataset_elapsed:.2f}s "
        f"rejected_limit={metadata['rejected_limit_samples']} "
        f"rejected_trace={metadata['rejected_trace_samples']} "
        f"attempts={metadata['attempt_count']}"
    )

    train_t0 = time.perf_counter()
    model, train_history, val_history = train_model(
        images=images,
        targets=targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=device,
    )
    train_elapsed = time.perf_counter() - train_t0
    print(f"[train] complete in {train_elapsed:.2f}s")

    dataset_path, checkpoint_path = save_artifacts(
        output_dir=args.output_dir,
        images=images,
        targets=targets,
        coefficients=coefficients,
        model=model,
        metadata=metadata,
        train_history=train_history,
        val_history=val_history,
        args=args,
    )

    verification_engine = RayOpticsPhysicsEngine(
        design_name="2P_AO",
        pupil_samples=args.pupil_samples,
        fft_samples=args.fft_samples,
    )
    verification_controller = FeedbackController(
        verification_engine,
        architecture="lightweight_cnn",
        compensator_surface_index=args.compensator_surface,
        checkpoint_path=checkpoint_path,
        prediction_mode="absolute",
        device=device,
    )
    verification_psf = verification_engine.get_psf_image()
    verification_prediction = verification_controller.infer(verification_psf)

    print(f"[artifact] dataset={dataset_path}")
    print(f"[artifact] checkpoint={checkpoint_path}")
    print(
        f"[verify] loaded checkpoint with output_dim={verification_prediction.size} "
        f"for compensator surface {args.compensator_surface}"
    )
    if train_history and val_history:
        print(
            f"[result] final_train_loss={train_history[-1]:.6f} "
            f"final_val_loss={val_history[-1]:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
