# Adaptive Optics V4

This folder is the residual-guided block-search successor to `ADAPTIVE_OPTICS_V3`.

The design goal for V4 stays strict:

- The repair loop may use only information that would be available to a real closed-loop controller.
- Ideal or ground-truth quantities may still be computed for offline evaluation, figures, and regression checks.
- Those truth-only quantities must not participate in repair-time seed construction, search-time scoring, or repair-time state updates.

## Closed-Loop Truth Policy

Allowed during repair:

- Current SHWFS measurement from the moved system
- Nominal SHWFS reference measurement captured at the ideal alignment
- A precomputed local mechanical-to-SHWFS response matrix built at nominal
- Mechanical bounds and gap constraints
- The current candidate relative command state inside the optimizer

Allowed as nominal calibration, not treated as leaked truth:

- Nominal PSF captured at the ideal position
- Nominal wavefront sensor reading captured at the ideal position

Forbidden during repair:

- Ground-truth moved `dx/dy/dz`
- Hidden absolute moved/start anchor position
- Any direct `nominal_reset = -known_delta` shortcut
- `true_residual_norm`
- Ideal-PSF error terms inside the repair objective
- Any seed built from hidden absolute actuator position

Evaluation-only after or outside repair:

- Ideal diffraction-limited PSF
- Ground-truth OPD-derived modal coefficients
- `true_residual_norm`
- Plots, CSVs, and JSON summaries comparing repaired vs ideal

## V4 Search Logic

V4 keeps the V3 blind-seed policy, but changes the repair engine from one all-at-once search into a residual-guided block search.

The V4 flow is:

1. Build the nominal SHWFS modal response matrix and local mechanical-to-SHWFS response matrix at nominal.
2. Form the blind seed only from SHWFS-visible quantities.
3. Keep the absolute moved/start state inside the clamp-and-forward plant only.
4. Expose to the repair/controller only a zero-centered relative command space, where the pre-repair position is `0`.
5. Measure the current SHWFS modal residual:

   `r = c_hat - c_ref`

6. Split the residual into coarse signatures:

   `R = |r1| + |r5|`

   `A = |r2| + |r3| + |r8| + |r9|`

   `H = |r4| + |r6| + |r7| + |r10| + |r11| + |r12|`

7. Use those signatures to prioritize grouped searches over:

   `Z block = all dz`

   `XY block = all dx + dy`

   `TT block = all tilt_x + tilt_y`

8. Run the main translation blocks first, then test a smaller tip/tilt block, then do a final polish block.
9. If a block stalls, keep its current best state and continue into the next block instead of aborting the whole repair.
10. For strongly angular or tip/tilt-dominant cases, apply a light `XY/Z` regularization before the `TT` block so the controller is less eager to fake tip/tilt repair with translation-only compensation.
9. Keep the repair-time objective inside the same closed-loop constraints:

   `cost = ||c_hat - c_ref||_2 + penalty(mechanical_violation)`

This means V4 still obeys the same truth policy, but it no longer forces `tip/tilt` to compete with every other degree of freedom in one flat search space.

Module boundary:

- `clamp/plant` may read the hidden absolute moved/start anchor state to apply mechanical limits and produce SHWFS measurements.
- `repair/controller` may not read hidden absolute moved/start anchor state and may only search in the zero-centered relative command space.

## Current V4 Defaults

The current V4 configuration intentionally trades global range for better local observability:

- SHWFS lenslet grid: `8 x 8`
- SHWFS slope-channel ceiling: `256`
- Mechanical response learning step: `5 um`
- Mechanical response pseudo-inverse `rcond`: `1e-6`
- Search-box scale from the known disturbance order: `1.5x`
- Search-box minimum radius: `0.02 mm`
- Blind-seed axial weight: `0.25`
- Tip/tilt test-block limit: `5 arcmin`
- Stage-local no-improvement stop: `40 eval`
- TT-dominant angular-ratio trigger: `2.0`
- TT-dominant pre-TT `XY/Z` bias weight: `0.05`

For example, a `100 um` disturbance-scale case gets a `150 um` search box instead of an exact `100 um` one. The default lenslet grid stays well within the configured `256`-channel limit so the response-learning pass remains practical for repeated repair runs.

## Active V4 Entry Points

Use these files for the V4 workflow:

- [Launch_Adaptive_Optics_V4.cmd](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V4/Launch_Adaptive_Optics_V4.cmd>)
- [ao_v4_launcher.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V4/ao_v4_launcher.py>)
- [ao_v4_worker.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V4/ao_v4_worker.py>)
- [ao_v4_backend.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V4/ao_v4_backend.py>)

The copied `ao_v2_*` and `ao_v3_*` files remain in this folder only as reference snapshots and are not the intended V4 closed-loop path.
