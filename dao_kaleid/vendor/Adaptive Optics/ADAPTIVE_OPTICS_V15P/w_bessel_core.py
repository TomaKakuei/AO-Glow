"""Weighted-Bessel basis generation for a circular optical pupil.

The public entry point is :func:`generate_w_bessel_basis`, which builds:

- a square Cartesian pupil grid,
- a Gaussian weight matrix ``W_xy`` based on a 1/e^2 beam fill ratio,
- the first ``num_modes`` standard Zernike modes,
- the matching Fourier-Bessel seed modes, and
- a weighted orthonormalized basis cube produced with modified Gram-Schmidt.

Assumption for ``beam_fill_ratio``:
    The value is interpreted as ``(1/e^2 beam diameter) / (pupil diameter)``.
    With a unit-radius pupil, the Gaussian intensity weight is
    ``W(r) = exp(-2 * (r / beam_fill_ratio)**2)`` inside the pupil.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.special import jn_zeros, jv


Array2D = NDArray[np.float64]
Array3D = NDArray[np.float64]
Bool2D = NDArray[np.bool_]

DEFAULT_NUM_MODES = 15


@dataclass(frozen=True)
class WBesselBasisResult:
    """Container returned by :func:`generate_w_bessel_basis`."""

    x: Array2D
    y: Array2D
    rho: Array2D
    theta: Array2D
    pupil_mask: Bool2D
    W_xy: Array2D
    nm_sequence: tuple[tuple[int, int], ...]
    zernike_cube: Array3D
    bessel_seed_cube: Array3D
    basis_cube: Array3D


def build_pupil_grid(grid_size: int) -> tuple[Array2D, Array2D, Array2D, Array2D, Bool2D]:
    """Build a square grid over a unit-radius circular pupil."""

    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1.")

    axis = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    pupil_mask = rho <= 1.0
    return x, y, rho, theta, pupil_mask


def gaussian_weight_matrix(
    rho: Array2D,
    pupil_mask: Bool2D,
    beam_fill_ratio: float,
) -> Array2D:
    """Return a Gaussian weight matrix clipped strictly to the pupil."""

    if beam_fill_ratio <= 0.0:
        raise ValueError("beam_fill_ratio must be positive.")

    W_xy = np.exp(-2.0 * (rho / beam_fill_ratio) ** 2, dtype=np.float64)
    W_xy = np.where(pupil_mask, W_xy, 0.0)
    return W_xy


def noll_to_nm(index: int) -> tuple[int, int]:
    """Map a 1-based Noll-like index to the corresponding (n, m) pair."""

    if index < 1:
        raise ValueError("index must be 1 or greater.")

    count = 0
    n = 0
    while True:
        for m in range(-n, n + 1, 2):
            count += 1
            if count == index:
                return n, m
        n += 1


def nm_sequence(num_modes: int = DEFAULT_NUM_MODES) -> tuple[tuple[int, int], ...]:
    """Return the first ``num_modes`` (n, m) pairs in Noll-like ordering."""

    if num_modes < 1:
        raise ValueError("num_modes must be at least 1.")
    return tuple(noll_to_nm(mode_index) for mode_index in range(1, num_modes + 1))


def _zernike_radial(n: int, m_abs: int, rho: Array2D, pupil_mask: Bool2D) -> Array2D:
    radial = np.zeros_like(rho, dtype=np.float64)
    if (n - m_abs) % 2 != 0:
        return radial

    max_s = (n - m_abs) // 2
    rho_in_pupil = np.where(pupil_mask, rho, 0.0)
    for s in range(max_s + 1):
        coeff = (
            (-1) ** s
            * factorial(n - s)
            / (
                factorial(s)
                * factorial((n + m_abs) // 2 - s)
                * factorial((n - m_abs) // 2 - s)
            )
        )
        radial += coeff * rho_in_pupil ** (n - 2 * s)

    radial[~pupil_mask] = 0.0
    return radial


def zernike_mode(n: int, m: int, rho: Array2D, theta: Array2D, pupil_mask: Bool2D) -> Array2D:
    """Return a standard RMS-normalized real-valued Zernike mode."""

    m_abs = abs(m)
    radial = _zernike_radial(n, m_abs, rho, pupil_mask)

    if m > 0:
        angular = np.cos(m_abs * theta)
        norm = sqrt(2.0 * (n + 1))
    elif m < 0:
        angular = np.sin(m_abs * theta)
        norm = sqrt(2.0 * (n + 1))
    else:
        angular = np.ones_like(theta, dtype=np.float64)
        norm = sqrt(n + 1.0)

    mode = norm * radial * angular
    mode[~pupil_mask] = 0.0
    return mode


def _bessel_radial_rank(n: int, m_abs: int) -> int:
    """Map a Zernike radial order to the corresponding Bessel root index."""

    return (n - m_abs) // 2 + 1


def bessel_seed_mode(n: int, m: int, rho: Array2D, theta: Array2D, pupil_mask: Bool2D) -> Array2D:
    """Return a real-valued Fourier-Bessel seed mode on the unit disk.

    The azimuthal order matches the Zernike ``m`` value. The radial order uses
    the same ordering depth as the corresponding Zernike mode so the first
    ``num_modes`` Bessel seeds follow the same modal progression.
    """

    m_abs = abs(m)
    radial_rank = _bessel_radial_rank(n, m_abs)
    alpha = jn_zeros(m_abs, radial_rank)[-1]

    rho_in_pupil = np.where(pupil_mask, rho, 0.0)
    radial = jv(m_abs, alpha * rho_in_pupil)

    if m > 0:
        angular = np.cos(m_abs * theta)
        angular_scale = sqrt(2.0)
    elif m < 0:
        angular = np.sin(m_abs * theta)
        angular_scale = sqrt(2.0)
    else:
        angular = np.ones_like(theta, dtype=np.float64)
        angular_scale = 1.0

    mode = angular_scale * radial * angular
    mode[~pupil_mask] = 0.0
    return mode


def generate_zernike_cube(
    grid_size: int,
    num_modes: int = DEFAULT_NUM_MODES,
) -> tuple[Array3D, tuple[tuple[int, int], ...], tuple[Array2D, Array2D, Array2D, Array2D, Bool2D]]:
    """Generate the first ``num_modes`` Zernike modes on a fresh pupil grid."""

    x, y, rho, theta, pupil_mask = build_pupil_grid(grid_size)
    sequence = nm_sequence(num_modes)
    cube = np.stack(
        [zernike_mode(n, m, rho, theta, pupil_mask) for n, m in sequence],
        axis=0,
    )
    return cube, sequence, (x, y, rho, theta, pupil_mask)


def generate_bessel_seed_cube(
    grid_size: int,
    num_modes: int = DEFAULT_NUM_MODES,
) -> tuple[Array3D, tuple[tuple[int, int], ...], tuple[Array2D, Array2D, Array2D, Array2D, Bool2D]]:
    """Generate the first ``num_modes`` Bessel seed modes on a fresh pupil grid."""

    x, y, rho, theta, pupil_mask = build_pupil_grid(grid_size)
    sequence = nm_sequence(num_modes)
    cube = np.stack(
        [bessel_seed_mode(n, m, rho, theta, pupil_mask) for n, m in sequence],
        axis=0,
    )
    return cube, sequence, (x, y, rho, theta, pupil_mask)


def weighted_inner_product(a: Array2D, b: Array2D, W_xy: Array2D) -> float:
    """Weighted inner product over the pupil."""

    return float(np.sum(a * b * W_xy, dtype=np.float64))


def modified_gram_schmidt_weighted(
    mode_cube: Array3D,
    W_xy: Array2D,
    pupil_mask: Bool2D,
    *,
    reorthogonalize: bool = True,
    atol: float = 1e-12,
) -> Array3D:
    """Weighted modified Gram-Schmidt orthonormalization.

    The basis is orthonormal under the discrete weighted inner product
    ``sum(A * B * W_xy)``. Values outside the pupil are forced to exactly zero
    throughout the algorithm.
    """

    if mode_cube.ndim != 3:
        raise ValueError("mode_cube must have shape (num_modes, grid_size, grid_size).")
    if mode_cube.shape[1:] != W_xy.shape or W_xy.shape != pupil_mask.shape:
        raise ValueError("mode_cube, W_xy, and pupil_mask shapes are inconsistent.")

    num_modes, grid_y, grid_x = mode_cube.shape
    flat_mask = pupil_mask.ravel()
    flat_weight = W_xy.ravel().astype(np.float64, copy=False)
    work = np.asarray(mode_cube, dtype=np.float64).reshape(num_modes, grid_y * grid_x).copy()
    work[:, ~flat_mask] = 0.0

    basis = np.zeros_like(work)
    num_passes = 2 if reorthogonalize else 1

    for i in range(num_modes):
        v = work[i].copy()
        v[~flat_mask] = 0.0

        for _ in range(num_passes):
            for j in range(i):
                projection = float(np.sum(v * basis[j] * flat_weight, dtype=np.float64))
                v -= projection * basis[j]
                v[~flat_mask] = 0.0

        norm_sq = float(np.sum(v * v * flat_weight, dtype=np.float64))
        if norm_sq <= atol:
            raise ValueError(
                f"Mode {i + 1} became numerically singular during weighted MGS."
            )

        basis[i] = v / np.sqrt(norm_sq)
        basis[i, ~flat_mask] = 0.0

    return basis.reshape(num_modes, grid_y, grid_x)


def generate_w_bessel_basis(
    grid_size: int,
    beam_fill_ratio: float,
    num_modes: int = DEFAULT_NUM_MODES,
) -> WBesselBasisResult:
    """Generate a weighted-Bessel basis for a circular optical pupil."""

    x, y, rho, theta, pupil_mask = build_pupil_grid(grid_size)
    W_xy = gaussian_weight_matrix(rho, pupil_mask, beam_fill_ratio)
    sequence = nm_sequence(num_modes)

    zernike_cube = np.stack(
        [zernike_mode(n, m, rho, theta, pupil_mask) for n, m in sequence],
        axis=0,
    )
    bessel_seed_cube = np.stack(
        [bessel_seed_mode(n, m, rho, theta, pupil_mask) for n, m in sequence],
        axis=0,
    )
    basis_cube = modified_gram_schmidt_weighted(bessel_seed_cube, W_xy, pupil_mask)

    return WBesselBasisResult(
        x=x,
        y=y,
        rho=rho,
        theta=theta,
        pupil_mask=pupil_mask,
        W_xy=W_xy,
        nm_sequence=sequence,
        zernike_cube=zernike_cube,
        bessel_seed_cube=bessel_seed_cube,
        basis_cube=basis_cube,
    )


def project_phase_map(
    phase_map: Array2D,
    basis_cube: Array3D,
    W_xy: Array2D,
    pupil_mask: Bool2D | None = None,
) -> NDArray[np.float64]:
    """Project a 2D phase map onto an orthonormal weighted basis cube."""

    phase = np.asarray(phase_map, dtype=np.float64)
    basis = np.asarray(basis_cube, dtype=np.float64)
    weight = np.asarray(W_xy, dtype=np.float64)

    if basis.ndim != 3:
        raise ValueError("basis_cube must have shape (num_modes, grid_size, grid_size).")
    if phase.shape != basis.shape[1:] or phase.shape != weight.shape:
        raise ValueError("phase_map, basis_cube, and W_xy shapes are inconsistent.")

    if pupil_mask is None:
        pupil_mask = weight > 0.0
    elif pupil_mask.shape != phase.shape:
        raise ValueError("pupil_mask must match the phase_map shape.")

    phase = phase.copy()
    phase[~pupil_mask] = 0.0

    if not np.all(np.isfinite(phase[pupil_mask])):
        raise ValueError("phase_map contains non-finite values inside the pupil.")

    coefficients = np.empty(basis.shape[0], dtype=np.float64)
    for mode_index in range(basis.shape[0]):
        coefficients[mode_index] = weighted_inner_product(phase, basis[mode_index], weight)
    return coefficients


def reconstruct_phase_map(coefficients: Iterable[float], basis_cube: Array3D) -> Array2D:
    """Reconstruct a phase map from modal coefficients and basis functions."""

    coeffs = np.asarray(tuple(coefficients), dtype=np.float64)
    if basis_cube.ndim != 3:
        raise ValueError("basis_cube must have shape (num_modes, grid_size, grid_size).")
    if coeffs.shape[0] != basis_cube.shape[0]:
        raise ValueError("Number of coefficients must match basis_cube.shape[0].")
    return np.tensordot(coeffs, basis_cube, axes=(0, 0))


__all__ = [
    "DEFAULT_NUM_MODES",
    "WBesselBasisResult",
    "build_pupil_grid",
    "gaussian_weight_matrix",
    "noll_to_nm",
    "nm_sequence",
    "zernike_mode",
    "bessel_seed_mode",
    "generate_zernike_cube",
    "generate_bessel_seed_cube",
    "weighted_inner_product",
    "modified_gram_schmidt_weighted",
    "generate_w_bessel_basis",
    "project_phase_map",
    "reconstruct_phase_map",
]
