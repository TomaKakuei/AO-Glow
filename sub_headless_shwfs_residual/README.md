# Shortest Non-UI SHWFS Residual Chain

This subfolder is a minimal runtime entry for the current SHWFS-residual optimization chain.

It keeps the existing model/control logic and adds a JSON lens displacement input path.

## Entry Script

- `run_short_chain.py`

## Input JSON

Use `displacement_input_example.json` as the template.

Each perturbation item supports either key style:

- `surface_index`, `dx_mm`, `dy_mm`, `dz_mm`
- `anchor_surface_id`, `perturbation_dx_mm`, `perturbation_dy_mm`, `perturbation_dz_mm`

## Quick Start (AO311)

From repo root:

```powershell
conda run -n ao311 python sub_headless_shwfs_residual/run_short_chain.py `
  --input-json sub_headless_shwfs_residual/displacement_input_example.json `
  --output-json sub_headless_shwfs_residual/outputs/short_chain_result.json
```

Fast smoke (skip optimizer):

```powershell
conda run -n ao311 python sub_headless_shwfs_residual/run_short_chain.py `
  --input-json sub_headless_shwfs_residual/displacement_input_example.json `
  --output-json sub_headless_shwfs_residual/outputs/short_chain_smoke.json `
  --maxiter 0 --maxfun 0
```

## Output

The output JSON includes:

- `nominal / moved / repaired` metrics
- `wavefront_rms_waves`, `wavefront_rms_nm`, `strehl_ratio`, `sharpness`, `residual_norm`
- best recovered actuator deltas and final positions
- optimizer settings and runtime

Default output path:

- `sub_headless_shwfs_residual/outputs/short_chain_result.json`
