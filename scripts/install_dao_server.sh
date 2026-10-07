#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import torch; print("Using existing PyTorch:", torch.__version__)'
python -m pip install -r requirements-dao.txt
python -m pip install --no-deps -r requirements-dao-optics.txt
python -m pip install --no-deps -e .
python -m dao_kaleid doctor
