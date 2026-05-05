# Adaptive Optics V10.2

`ADAPTIVE_OPTICS_V10_2` is the **12-mode analysis branch**.

This branch is where the 12-mode experiments live. The main 8-mode line remains in `V10.1`; `V10.2` is for mode-expansion analysis and case sweeps.

## Latest 12-Mode Result

The latest 12-mode study is the 5-case quantized sweep:

- [artifacts/v10_2_quantized12_mode5cases/results.csv](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/artifacts/v10_2_quantized12_mode5cases/results.csv>)
- [artifacts/v10_2_quantized12_mode5cases/mode_summary.csv](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/artifacts/v10_2_quantized12_mode5cases/mode_summary.csv>)
- [artifacts/v10_2_quantized12_mode5cases/mode_components_long.csv](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/artifacts/v10_2_quantized12_mode5cases/mode_components_long.csv>)

Run it again with:

```powershell
conda run -n ao311 python .\ADAPTIVE_OPTICS_V10_2\tmp_v10_2_quantized12_mode5cases.py
```

If you need the runner file directly:

- [tmp_v10_2_quantized12_mode5cases.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/tmp_v10_2_quantized12_mode5cases.py>)

## Branch Role

`V10.2` keeps the same hidden-plant closed-loop truth boundary as the other `V10` branches, but its purpose is to explore how the controller behaves after expanding the modal basis to 12 modes.

## Closed-Loop Truth Policy

Allowed during repair:

- Current SHWFS measurement from the moved system
- Nominal SHWFS reference measurement captured at the ideal alignment
- A precomputed local mechanical-to-SHWFS response matrix built at nominal
- Mechanical bounds and gap constraints
- The current candidate relative command state inside the optimizer
- A zero-centered relative search box whose size is chosen from the known disturbance order, not from hidden absolute moved coordinates

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
- Any direct readback of absolute anchor state by the repair/controller layer

Evaluation-only after or outside repair:

- Ideal diffraction-limited PSF
- Ground-truth OPD-derived modal coefficients
- `true_residual_norm`
- Plots, CSVs, and JSON summaries comparing repaired vs ideal

Module boundary:

- `clamp/plant` may read the hidden absolute moved/start anchor state to apply mechanical limits and produce SHWFS measurements.
- `repair/controller` may not read hidden absolute moved/start anchor state and may only search in the zero-centered relative command space.

## V5 Search Logic

V5 keeps the `global-TT-first` grouped search from V4.1:

1. Build the nominal SHWFS modal response matrix and local mechanical-to-SHWFS response matrix at nominal.
2. Keep the absolute moved/start state inside the clamp-and-forward plant only.
3. Expose to the repair/controller only a zero-centered relative command space, where the pre-repair position is `0`.
4. Split residual signatures into grouped search blocks over:

   `Z block = all dz`

   `XY block = all dx + dy`

   `TT block = all tilt_x + tilt_y`

5. Run the main translation blocks first.
6. Run a `global TT` block over common-mode `global_tilt_x` and `global_tilt_y`.
7. Follow with a `local TT refine` block over per-anchor TT residuals.
8. If a block stalls, keep its current best state and continue into the next block instead of aborting the whole repair.

The repair-time objective remains:

`cost = ||c_hat - c_ref||_2 + penalty(mechanical_violation)`

## V5 SHWFS Forward Model

The V5 SHWFS forward model uses a `13 x 13` lenslet grid with a `256`-slope ceiling. The old V3/V4 slope multiplier noise has been removed from the active path.

For each active lenslet:

1. Fit the local pupil phase to an ideal `x/y` slope.
2. Convert that ideal slope into a lenslet-spot centroid shift on a small synthetic sensor patch.
3. Build an ideal spot in the electron domain.
4. Inject physical sensor noise in this order:

   `I_gain = I_ideal * G`

   `I_base = I_gain + O`

   `I_shot ~ Poisson(I_base)`

   `I_final_e = I_shot + N_read`

   `I_adu = clamp(round(I_final_e / conversion_gain), 0, 1023)`

5. Recover the centroid from the quantized sensor patch.
6. Convert the centroid shift back into a measured SHWFS slope.

This means shot noise now scales with the actual electron count instead of being faked directly on the final slope vector.

## Noise Profiles

V5 supports four built-in SHWFS noise profiles:

- `none`
  Idealized readout with no sensor-domain noise.
- `minimum`
  Light physical noise, high signal, mild gain variation.
- `realistic`
  Recommended default. Thorlabs-style 8-bit operating point with `peak_e=800`, `analog_gain=0.305 ADU/e-`, `black_level=10 ADU`, moderate gain variation, background offset, shot noise, and low CMOS read noise.
- `enhanced`
  Stress-test noise. Lower signal and stronger fixed-pattern/background/read noise.

The current default is:

- `lenslet grid = 13 x 13`
- `slope limit = 256`
- `noise profile = realistic`
- `realistic analog_gain_adu_per_e = 0.305`
- `realistic black_level_adu = 10.0`
- `tip/tilt test block = 6 arcmin`
- `stage-local no-improvement stop = 40 eval`
- `TT-dominant XYZ bias = 0.03`
- `local TT variance weight = 0.02`

The implemented sensor-side noise parameters follow these physical categories:

- Gain non-uniformity
- Spatial background offset / dark-current map
- Photon shot noise in electrons
- Read noise in electrons
- ADU quantization and `10-bit` clipping

## Active V5 Entry Points

Use these files for the V5 workflow:

- [Launch_Adaptive_Optics_V5.cmd](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V5/Launch_Adaptive_Optics_V5.cmd>)
- [ao_v5_launcher.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V5/ao_v5_launcher.py>)
- [ao_v5_worker.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V5/ao_v5_worker.py>)
- [ao_v5_backend.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V5/ao_v5_backend.py>)

Reference implementation details for the noisy sensor model live in:

- [benchmark_freeform_real_shwfs_residual_120.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V5/benchmark_freeform_real_shwfs_residual_120.py>)
- [benchmark_1mm_real_shwfs_converging_920nm.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive Optics/ADAPTIVE_OPTICS_V5/benchmark_1mm_real_shwfs_converging_920nm.py>)
