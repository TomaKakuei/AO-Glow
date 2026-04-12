# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


project_dir = Path(globals().get("SPECPATH", ".")).resolve()
checkpoint_path = (
    project_dir / "artifacts" / "2p_ao_phase_to_comp74_lightweight_cnn_1000samples.pt"
)

datas = []
binaries = []
hiddenimports = []

if checkpoint_path.exists():
    datas.append((str(checkpoint_path), "artifacts"))

for package_name in (
    "charset_normalizer",
    "matplotlib",
    "opticalglass",
    "rayoptics",
    "requests",
    "torchvision",
):
    datas += collect_data_files(package_name)
    hiddenimports += collect_submodules(package_name)
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass

for package_name in ("torch", "torchvision"):
    binaries += collect_dynamic_libs(package_name)
    hiddenimports += collect_submodules(package_name)
    try:
        datas += copy_metadata(package_name)
    except Exception:
        pass

hiddenimports += [
    "matplotlib.backends.backend_qtagg",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
]

excludes = [
    "IPython",
    "jedi",
    "PyQt5",
    "PySide2",
    "PySide6",
    "pytest",
    "setuptools",
    "tkinter",
]

a = Analysis(
    ["main_gui.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AdaptiveOpticsGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AdaptiveOpticsGUI",
)
