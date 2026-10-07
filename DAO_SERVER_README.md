# DAO + Kaleid server runtime

This package includes four optical systems, trained predictors, RayOptics prescriptions, fast tracing, camera/noise simulation, streaming sample generation and feedback controllers. It runs from its own directory on a headless server. Historical training datasets and Python/CUDA installations are not needed to start it.

## Install on a server with PyTorch already available

Use Python 3.11 or newer in your existing PyTorch environment:

```bash
git clone https://github.com/TomaKakuei/AO-Glow.git
cd AO-Glow
bash scripts/install_dao_server.sh
python -m dao_kaleid doctor --system mod3 --device cuda
```

The installer installs the tested numerical dependencies, then installs RayOptics 0.9.8 and opticalglass 1.1.1 with `--no-deps`. Their declared Qt/notebook dependencies are unused by the included headless paths. PyTorch is reused. Normal `pip install -e .` also works but resolves the libraries' full GUI dependency lists.

For CPU use, pass `--device cpu`. The numerical engines and inference support CPU; Mod3 camera propagation and the networks can use CUDA. KLA's existing vectorized optical engine remains NumPy on CPU. CPU latency is system dependent.

## Included weights

The hash-checked registry is [model_registry.json](dao_kaleid/model_registry.json). It includes **19 distinct checkpoint files**:

| System | `closed_loop` | Additional profile |
|---|---|---|
| Mod3 | Two earlier weights used in the original noisy-sensorless 15-case feedback experiment | `latest`: the just-completed front epoch 43 / rear epoch 34, each trained for 50 epochs on 7000 samples with 500 selection-validation cases |
| Nikon | Original corrected G1/G2/G3 ArchA weights used by the complete published L4 controller | `latest`: coordinated 35-DOF epoch 49, with the native noisy 3×49 node branch |
| Trepan | Original rigid-tilt-corrected front/rear ArchA weights used by the published paired balanced20 controller | `latest` / `differentiable`: the checked front/rear differentiable candidates |
| KLA | Original module 2/3/4 ArchA weights used by the complete adaptive phase/MTF controller | `latest` / `differentiable`: checked module 2/3/4 candidates; `warmstart_7k`: the separate original Condition C checkpoint |

`closed_loop` identifies the original successful complete neural pipeline, not a posthoc ranking of all architectures on different tasks. The separate KLA Condition C checkpoint has a different task/controller and is not silently substituted for the three-module pipeline. No averaged or task-routed checkpoint selection is performed.

The latest Mod3 isolated-module center WRMS<0.07λ rates are 500/500 and 433/500. These selection-validation rates do not describe assembled whole-system feedback success. The original checkpoints remain available.

## Generate data without transferring old datasets

Each generator writes bounded NPZ batches and a manifest. Restart the same command to continue missing batches. Use a different output directory when changing the range or protocol. The nominal model and fixed noise/optical parameters are inside the package.

```bash
# Mod3: original heterogeneous independent rigid-body schedule, opposite branch nominal
python -m dao_kaleid generate --system mod3 --branch front --count 1000 --output runs/data/mod3_front --device cuda
python -m dao_kaleid generate --system mod3 --branch rear  --count 1000 --output runs/data/mod3_rear  --device cuda

# Trepan: existing fast speckle sampler, photon/read/PRNU/DSNU/ADC noise
python -m dao_kaleid generate --system trepan --branch front --count 1000 --output runs/data/trepan_front
python -m dao_kaleid generate --system trepan --branch rear  --count 1000 --output runs/data/trepan_rear

# Nikon: original noisy 3×49 nodes and derived maps, joint context retained
python -m dao_kaleid generate --system nikon --branch G1 --count 1000 --output runs/data/nikon_G1
# Repeat for G2 and G3.

# KLA: original noisy 9×64×64 optical maps; module 2/3/4 supported
python -m dao_kaleid generate --system kla --branch module2 --count 1000 --output runs/data/kla_module2
```

The KLA portable generator samples isolated target modules with the original physical amplitude box; it is a new stream, not a reconstruction of the historical sample IDs. The Nikon generator preserves the original generator's seed/group rules. The Trepan stream delegates to the existing fast batch sampler. Mod3 retains its assigned amplitude stratum on geometry rejection and refines continuous OPD fits on the same state/seed at the unchanged 0.0003λ threshold.

At 16-bit camera storage, 10,000 Mod3 samples contain approximately **2.75 GiB of uncompressed camera frames per branch** (9×128×128×2 bytes each), plus labels and metadata. Compression depends on the noisy images. Sampling writes batches rather than holding the whole dataset in memory. Do not commit generated datasets; `runs/` is ignored.

## Feedback and simulation

```bash
# Mod3: separate front/rear correction, assemble, then measured joint refinement
python -m dao_kaleid feedback --system mod3 --profile latest --seed 4101001 --budget 80 --module-budget 20 --device cuda --output runs/mod3_case.json

# Nikon: original L4 noisy measured residual loop, all seven bodies corrected jointly
python -m dao_kaleid feedback --system nikon --profile closed_loop --seed 4 --budget 80 --device cuda --output runs/nikon_case.json
python -m dao_kaleid feedback --system nikon --profile latest --seed 4 --budget 80 --device cuda --output runs/nikon_latest.json

# Portable bounded noisy-feedback adapters
python -m dao_kaleid feedback --system trepan --profile closed_loop --seed 136 --budget 80 --module-budget 20 --output runs/trepan_case.json
python -m dao_kaleid feedback --system kla --profile closed_loop --seed 156 --budget 80 --output runs/kla_case.json
```

Mod3 uses the original repaired module controller, independently reacquired noisy image pairs, learned module acceptance and measured-calibration joint correction. Only the simulator holds unknown physical states. Mechanical/travel interlocks stay on. Terminal ideal optical metrics are calculated after all control decisions are frozen.

Nikon retains its original noise-enabled node/center observation interface. These are simulated noisy optical residual observations, not speckle-camera frames or an added SHWFS interface. KLA also retains its original noise-enabled optical-map interface.

Trepan and KLA have two explicitly different entries:

- `feedback` is the portable bounded observation-driven adapter. Trepan uses independently reacquired noisy speckles for module acceptance and joint image refinement. KLA uses its original noisy maps and stored measured phase-response calibration. These adapter paths have functional checks, not new recovery-rate evidence; they do not inherit the published controller's performance claims.
- `replay-published` retains the original full controller logic and weights. **Historical Trepan replay uses exact center WRMS in its gain/joint decisions**. It is a simulation reconstruction, not the noisy sensorless deployment contract. KLA replay includes original module rounds, fixed/local response correction and broad/target MTF polishing. Nikon uses original L4 control. Mod3 uses the earlier checked weights and its original repaired noisy feedback rules.

```bash
# Original full histories; may take substantially longer than bounded functional checks
python -m dao_kaleid replay-published --system trepan --seed 136 --output runs/published_trepan.json
python -m dao_kaleid replay-published --system kla --seed 156 --output runs/published_kla.json
python -m dao_kaleid replay-published --system nikon --seed 4 --output runs/published_nikon.json
```

The original KLA 7k Condition C program is shipped in `dao_kaleid/vendor/Adaptive Optics/ADAPTIVE_OPTICS_V15P/canonical_three_phase_universal/budget_scaling/`; its dedicated checkpoint is in the registry. Its historical whole-suite CLI is intentionally not substituted into the three-module feedback entry.

## Keep a process running over SSH

```bash
mkdir -p runs
nohup python -u -m dao_kaleid generate --system mod3 --branch front --count 10000 --output runs/data/mod3_front --device cuda > runs/mod3_front.log 2>&1 &
echo $! > runs/mod3_front.pid
tail -f runs/mod3_front.log
```

Each system runs in its own process because original modules share short Python import names. Do not launch concurrent heavy optical jobs when measuring timing or using a small GPU. Logs and results stay on the server.

## Observation service

For an actual camera/control client, the optional standard-library HTTP server keeps the selected networks loaded:

```bash
DAO_API_TOKEN=your-local-token python -m dao_kaleid serve --system mod3 --profile latest --device cuda --host 127.0.0.1 --port 8000
```

`GET /health` describes the loaded profile. `POST /predict/front` or `/predict/rear` accepts an NPZ body containing `observation` (one noisy 9×128×128 raw ADU stack) and returns a pose and command. Nikon uses `/predict/joint` with 3×49 noisy nodes; KLA uses `/predict/module2`, etc. Supply `Authorization: Bearer …` when `DAO_API_TOKEN` is set. For remote access, use an SSH port forward or your existing reverse proxy.

Mod3 module networks require the opposite module to be nominal during module sensing, as in their training contract. Do not send an arbitrary assembled observation to both isolated networks. The complete simulation's assembled refinement uses measured image features instead.

## Evidence and provenance

Local functional checks are recorded in [package_validation.json](dao_kaleid/evidence/package_validation.json): all 19 checkpoints load, four optical paths generate noisy observations, short feedback paths execute, and the HTTP service returns a finite 35-axis command. Original project access and Qt/Notebook imports were blocked during portability checks. These checks ran on Windows; Linux server execution has not yet been tested.

The original KLA budget-scaling script also has its first 11 historical case labels packaged as a small fixture; its full training image archive is unnecessary for replay. The portable noisy Trepan adapter records any initial mechanical projection and rejects later commands that the simulator would silently clamp. Its recovery performance is not claimed to match the historical controller.

- [source_manifest.json](dao_kaleid/source_manifest.json) records original and packaged hashes and portable-only transformations.
- [model_registry.json](dao_kaleid/model_registry.json) records all checkpoint hashes and evidence scopes.
- `dao_kaleid/evidence/` contains saved model-selection/review results.
- `dao_kaleid/fixtures/` and `scripts/verify_dao_package.py` provide small regression observations and runtime verification.

The package includes training-free sampling and runtime, not a replacement for the historical training archive. Original optical and control sources are preserved as a curated snapshot; path/import/parent-directory adaptations are recorded. Training datasets, epoch histories, optimizer state, environments and unrelated SETSUNET content remain outside this release.
