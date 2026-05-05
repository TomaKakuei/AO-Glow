"""Process bootstrap for stable native-library startup on Windows ao311.

This module is imported before NumPy / Torch / RayOptics in entry points that
are sensitive to DLL search order. It makes the conda environment's native
directories discoverable and normalizes a few thread-related environment vars.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BOOTSTRAPPED = False


def _append_env_path(name: str, value: str) -> None:
    current = os.environ.get(name, "")
    if not current:
        os.environ[name] = value
        return
    parts = current.split(os.pathsep)
    if value not in parts:
        os.environ[name] = value + os.pathsep + current


def bootstrap_runtime() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")

    env_prefix = Path(sys.executable).resolve().parent.parent
    repo_root = Path(__file__).resolve().parent
    torch_lib_dir = env_prefix / "Lib" / "site-packages" / "torch" / "lib"
    bundled_internal_dir = repo_root.parent / "dist" / "AdaptiveOpticsGUI" / "_internal"
    bundled_torch_lib_dir = bundled_internal_dir / "torch" / "lib"
    dll_dirs = [
        bundled_internal_dir,
        bundled_torch_lib_dir,
        torch_lib_dir,
        env_prefix,
        env_prefix / "Library",
        env_prefix / "Library" / "bin",
        env_prefix / "DLLs",
        env_prefix / "Scripts",
    ]
    for dll_dir in dll_dirs:
        if not dll_dir.exists():
            continue
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(dll_dir))
        _append_env_path("PATH", str(dll_dir))

    _BOOTSTRAPPED = True


__all__ = ["bootstrap_runtime"]
