# Adaptive Optics V3

This folder is the closed-loop-focused successor to `ADAPTIVE_OPTICS_V2`.

The design goal for V3 is strict:

- The repair loop may use only information that would be available to a real closed-loop controller.
- Ideal or ground-truth quantities may still be computed for offline evaluation, figures, and regression checks.
- Those truth-only quantities must not participate in repair-time seed construction, search-time scoring, or repair-time state updates.

## Closed-Loop Truth Policy

Allowed during repair:

- Current SHWFS measurement from the moved system
- Nominal SHWFS reference measurement captured at the ideal alignment
- A precomputed local mechanical-to-SHWFS response matrix built at nominal
- Mechanical bounds and gap constraints
- The current candidate mechanical state inside the optimizer

Allowed as nominal calibration, not treated as leaked truth:

- Nominal PSF captured at the ideal position
- Nominal wavefront sensor reading captured at the ideal position

Forbidden during repair:

- Ground-truth moved `dx/dy/dz`
- Any direct `nominal_reset = -known_delta` shortcut
- `true_residual_norm`
- Ideal-PSF error terms inside the repair objective
- Any seed built from hidden absolute actuator position

Evaluation-only after or outside repair:

- Ideal diffraction-limited PSF
- Ground-truth OPD-derived modal coefficients
- `true_residual_norm`
- Plots, CSVs, and JSON summaries comparing repaired vs ideal

## V3 Blind Seed Logic

V2 leaked truth by constructing a nominal reset seed from the known moved state. V3 replaces that with a linearized SHWFS guess.

The new flow is:

1. At nominal, build the standard SHWFS modal response matrix.
2. At nominal, also sweep each of the 24 mechanical degrees of freedom with a small signed perturbation.
3. For each perturbation, record the induced SHWFS slope residual.
4. Stack those columns into a mechanical response matrix `R_mech`, where:

   `slope_residual ~= R_mech @ delta_mech`

5. Precompute the pseudo-inverse of `R_mech`.
6. During repair, measure the current SHWFS slope residual and estimate the hidden mechanical error:

   `delta_est ~= pinv(R_mech) @ slope_residual`

7. Use `-delta_est` as the blind `nominal_reset_seed`, clip it to legal bounds, and compare it against the hold seed.

This seed is still only a local linear guess, but it is derived from SHWFS-visible information rather than hidden moved-state truth.

## Current V3 Defaults

The current V3 configuration intentionally trades global range for better local observability:

- SHWFS lenslet grid: `8 x 8`
- SHWFS slope-channel ceiling: `256`
- Mechanical response learning step: `5 um`
- Mechanical response pseudo-inverse `rcond`: `1e-6`
- Search-box scale from the known disturbance order: `1.5x`
- Search-box minimum radius: `0.02 mm`
- Blind-seed axial weight: `0.25`

For example, a `100 um` disturbance-scale case now gets a `150 um` search box instead of an exact `100 um` one. The default lenslet grid stays well within the configured `256`-channel limit so the response-learning pass remains practical for repeated repair runs.

## Active V3 Entry Points

Use these files for the V3 workflow:

- [Launch_Adaptive_Optics_V3.cmd](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V3/Launch_Adaptive_Optics_V3.cmd>)
- [ao_v3_launcher.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V3/ao_v3_launcher.py>)
- [ao_v3_worker.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V3/ao_v3_worker.py>)
- [ao_v3_backend.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V3/ao_v3_backend.py>)

The copied `ao_v2_*` files remain in this folder only as reference snapshots and are not the intended V3 closed-loop path.
