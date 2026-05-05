# AO-Glow

Closed-loop active-alignment simulator for adaptive optics, built around a pure-Python `rayoptics` core so the application can be packaged into a portable Windows executable with PyInstaller.

## Current AO Benchmark Index

The active benchmark line now lives in the `ADAPTIVE_OPTICS_V10_*` branches. Use the links below to jump to the two most useful call paths:

- Best 8-mode reference: [ADAPTIVE_OPTICS_V10_1/artifacts/v10_profile_sensor_trial/results.csv](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_1/artifacts/v10_profile_sensor_trial/results.csv>)
  - The best profile in that 8-mode comparison is `minimal`.
  - Re-run the 8-mode mainline with:

```powershell
conda run -n ao311 python .\ADAPTIVE_OPTICS_V10_1\ao_v10_1_launcher.py
```

- Latest 12-mode study: [ADAPTIVE_OPTICS_V10_2/artifacts/v10_2_quantized12_mode5cases/results.csv](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/artifacts/v10_2_quantized12_mode5cases/results.csv>)
  - Supporting summaries:
    - [mode_summary.csv](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/artifacts/v10_2_quantized12_mode5cases/mode_summary.csv>)
    - [mode_components_long.csv](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/artifacts/v10_2_quantized12_mode5cases/mode_components_long.csv>)
  - Re-run the 5-case 12-mode runner with:

```powershell
conda run -n ao311 python .\ADAPTIVE_OPTICS_V10_2\tmp_v10_2_quantized12_mode5cases.py
```

If you only need the latest 12-mode runner file, start here:

- [tmp_v10_2_quantized12_mode5cases.py](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/ADAPTIVE_OPTICS_V10_2/tmp_v10_2_quantized12_mode5cases.py>)

Older branch snapshots are archived under [Historical/](</c:/Users/arthu/OneDrive/文档/_UCSB/ECE/Adaptive%20Optics/Historical>).

## What Is Included

- `optical_model_rayoptics.py`
  - RayOptics-based optical physics engine
  - 2P AO prescription injection path
  - Rigid lens-element / cemented-group perturbation support
  - Cemented-group handling for `74-76` and `81-83`
  - Mechanical envelopes with `+/- 2.0 mm` lateral travel and axial non-contact clamps
  - Wavefront OPD extraction and FFT PSF generation
- `feedback_controller.py`
  - `CNN Only` inference path
  - `CNN + SciPy` sensorless refinement
  - SHWFS / weighted-Bessel pseudo-inverse controller
- `train_cnn_jitter.py`
  - Simulated disturbance-to-compensator dataset generation
  - 1000-sample lightweight CNN training pipeline
  - Weighted-Bessel residual-to-compensator target generation for the rigid `74-76` group
- `main_gui.py`
  - PyQt6 GUI tying together model, controller, and view
  - Worker-threaded optimization
  - Matplotlib PSF / wavefront displays
- `main_gui.spec`
  - PyInstaller build spec for the portable Windows GUI bundle
- `w_bessel_core.py`
  - Weighted-Bessel basis generation and projection utilities

## Current Model Setup

- Optical design: injected `2P_AO` sequential prescription
- Fast correction target: rigid compensator group `74-76` commanded through the `surface 74` anchor
- Rigid element groups:
  - `68-69`, `70-71`, `72-73`, `74-76`, `77-78`, `79-80`, `81-83`, `84-85`
- Disturbance model for CNN training:
  - jitter is applied only to non-compensator movable lens groups
  - the CNN target is the absolute `[dx, dy, dz]` rigid-group command for `74-76`
  - weighted-Bessel coefficients are converted to residuals against the nominal baseline before solving the compensator target
- Hard mechanical constraints:
  - lateral translation: `+/- 2.0 mm`
  - axial translation: clamped by front/back non-contact clearance and then by `+/- 2.0 mm`
  - tilt: `+/- 2.0 deg`

## Trained Model

The repository keeps the lightweight checkpoint used by the GUI:

- `artifacts/2p_ao_psf_to_wbessel_comp7476_lightweight_cnn_1000samples.pt`

Large generated datasets and packaged binaries are intentionally excluded from git.

## Running From Source

Use the existing `ao311` Conda environment.

```powershell
python main_gui.py
```

The GUI defaults to loading:

- `artifacts/2p_ao_psf_to_wbessel_comp7476_lightweight_cnn_1000samples.pt`

## Training

Example training command:

```powershell
python train_cnn_jitter.py --num-samples 1000 --epochs 12 --batch-size 32 --pupil-samples 32 --fft-samples 128 --num-modes 5 --jacobian-step-mm 0.02 --jitter-limit-mm 0.01 --axial-jitter-scale 0.02 --activation-probability 0.15 --max-active-surfaces 1 --pinv-rcond 1e-2 --output-dir artifacts --device cpu
```

## Packaging

Build the portable GUI bundle with:

```powershell
python -m PyInstaller --clean --noconfirm main_gui.spec
```

The build output appears in:

- `dist/AdaptiveOpticsGUI/`

## Notes

- The current packaged app is built in `onedir` mode for stability with `torch`, `PyQt6`, `matplotlib`, and `rayoptics`.
- The GUI includes a `CNN Only` mode for fast correction and retains the slower refinement modes for testing and comparison.
- The CNN checkpoint is trained on `128x128` PSF inputs; the runtime controller automatically resizes the live PSF input from the display resolution when needed.
