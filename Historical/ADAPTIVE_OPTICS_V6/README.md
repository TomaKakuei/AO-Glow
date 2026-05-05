# Adaptive Optics V6

`ADAPTIVE_OPTICS_V6` builds on `V5` and keeps the same hidden-plant / controller truth boundary, while adding a more robust SHWFS estimator for noisy `13 x 13` sensor data.

## Truth Boundary

Allowed during repair:

- Current SHWFS measurement from the moved system
- Nominal SHWFS reference measurement captured at the ideal alignment
- A precomputed local mechanical-to-SHWFS response matrix built at nominal
- Mechanical bounds and gap constraints
- The current candidate relative command state inside the optimizer
- A zero-centered relative search box whose size is chosen from disturbance order, not hidden absolute moved coordinates

Forbidden during repair:

- Ground-truth moved `dx/dy/dz`
- Hidden absolute moved/start anchor position
- Any direct `nominal_reset = -known_delta` shortcut
- `true_residual_norm`
- Any seed built from hidden absolute actuator position
- Any direct readback of absolute anchor state by the repair/controller layer

Module boundary:

- `clamp/plant` may read the hidden absolute moved/start anchor state to apply mechanical limits and produce SHWFS measurements.
- `repair/controller` may not read hidden absolute moved/start anchor state and may only search in zero-centered relative command space.

## What V6 Adds

V6 keeps the grouped `XY -> Z -> global TT -> local TT -> polish` search structure from V5, but upgrades the SHWFS estimator with:

1. Dynamic thresholding from local background plus a configurable noise sigma multiple
2. `3 x 3` median filtering for hot-pixel-like outliers
3. Small-kernel Gaussian smoothing for shot-noise suppression
4. Windowed weighted centroiding with `I^n`
5. Tikhonov-regularized W-Bessel coefficient estimation
6. Mild active-mode truncation in noisy operation

The repair objective remains:

`cost = ||c_hat - c_ref||_2 + penalty(mechanical_violation)`

## Active Defaults

- SHWFS lenslet grid: `13 x 13`
- SHWFS slope ceiling: `256`
- Default noise profile: `realistic`
- Default forward averaging: `1`
- Weighted centroid power: `2`
- Dynamic threshold: `background + 3 sigma`
- Gaussian sigma: `0.7 px`
- Active W-Bessel modes in noisy solve: `6`
- Tikhonov lambda: `0.015`

## Entry Points

- [Launch_Adaptive_Optics_V6.cmd](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V6/Launch_Adaptive_Optics_V6.cmd>)
- [ao_v6_launcher.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V6/ao_v6_launcher.py>)
- [ao_v6_worker.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V6/ao_v6_worker.py>)
- [ao_v6_backend.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V6/ao_v6_backend.py>)

Core SHWFS / benchmark implementation:

- [benchmark_freeform_real_shwfs_residual_120.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V6/benchmark_freeform_real_shwfs_residual_120.py>)
- [benchmark_1mm_real_shwfs_converging_920nm.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V6/benchmark_1mm_real_shwfs_converging_920nm.py>)
