# AO-Glow

Closed-loop active-alignment simulator for adaptive optics, built around a pure-Python `rayoptics` core so the application can be packaged into a portable Windows executable with PyInstaller.

## What Is Included

- `optical_model_rayoptics.py`
  - RayOptics-based optical physics engine
  - 2P AO prescription injection path
  - Per-surface perturbation support with hard mechanical clamps
  - Wavefront OPD extraction and FFT PSF generation
- `feedback_controller.py`
  - `CNN Only` inference path
  - `CNN + SciPy` sensorless refinement
  - SHWFS / weighted-Bessel pseudo-inverse controller
- `train_cnn_jitter.py`
  - Simulated disturbance-to-compensator dataset generation
  - 1000-sample lightweight CNN training pipeline
  - Surface-74 compensator target generation from weighted-Bessel coefficients
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
- Fast correction target: compensator on `surface 74`
- Disturbance model for CNN training:
  - jitter is applied only to non-compensator movable lens surfaces
  - the CNN target is the absolute `[dx, dy, dz]` command for `surface 74`
- Hard mechanical constraints:
  - translation: `+/- 1.0 mm`
  - tilt: `+/- 2.0 deg`

## Trained Model

The repository keeps the lightweight checkpoint used by the GUI:

- `artifacts/2p_ao_phase_to_comp74_lightweight_cnn_1000samples.pt`

Large generated datasets and packaged binaries are intentionally excluded from git.

## Running From Source

Use the existing `ao311` Conda environment.

```powershell
python main_gui.py
```

The GUI defaults to loading:

- `artifacts/2p_ao_phase_to_comp74_lightweight_cnn_1000samples.pt`

## Training

Example training command:

```powershell
python train_cnn_jitter.py --num-samples 1000 --epochs 12 --batch-size 32 --pupil-samples 48 --fft-samples 256 --num-modes 3 --jacobian-step-mm 0.02 --jitter-limit-mm 0.01 --activation-probability 0.15 --max-active-surfaces 1 --pinv-rcond 1e-2 --output-dir artifacts --device cpu
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
