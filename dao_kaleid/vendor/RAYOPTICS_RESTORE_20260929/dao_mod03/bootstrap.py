"""Process-local DLL and project setup; does not modify any installed environment."""
from pathlib import Path
import ctypes
import os
import sys

HERE = Path(__file__).resolve().parent
RESTORE = HERE.parent
WORKSPACE = RESTORE.parent
DAO = WORKSPACE / 'Adaptive Optics' / 'ADAPTIVE_OPTICS_V15P'
RETRAIN = DAO / 'rigid_group_retrain_20260908'
ARCHIVE = RESTORE / 'design_archive' / 'mod03'
RESULTS = HERE / 'results'
DLL_HANDLES = []


def setup():
    # Conda's libiomp forwarding DLL points to LLVM OpenMP, whose exports do
    # not satisfy this PyTorch fbgemm. Select PyTorch's Intel runtime explicitly;
    # keep MKL sequential so that it does not initialize a second OpenMP runtime.
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if os.name == 'nt':
        prefix = Path(sys.prefix)
        os.environ['PATH'] = str(prefix) + ';' + str(prefix / 'Library/bin') + ';' + os.environ.get('PATH', '')
        if (prefix / 'Library/bin').is_dir():
            DLL_HANDLES.append(os.add_dll_directory(str(prefix / 'Library/bin')))
        runtime = prefix / 'Lib/site-packages/torch/lib/libiomp5md.dll'
        if runtime.is_file():
            DLL_HANDLES.append(ctypes.WinDLL(str(runtime)))
    for path in (RESTORE, RESTORE / 'patent_comparison', DAO,
                 DAO / 'canonical_three_phase_universal', RETRAIN):
        sys.path.insert(0, str(path))
    RESULTS.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(1)


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    import json
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def local_manifest():
    import json
    old = RETRAIN / 'active_manifest.json'
    source = json.loads(old.read_text(encoding='utf-8'))
    def relocate(value):
        if isinstance(value, str) and '/ADAPTIVE_OPTICS_V15P/' in value.replace('\\', '/'):
            suffix = value.replace('\\', '/').split('/ADAPTIVE_OPTICS_V15P/', 1)[1]
            return str(DAO / suffix)
        if isinstance(value, dict):
            return {relocate(k): relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value
    result = relocate(source)
    result['local_relocation'] = dict(original_manifest=str(old), original_sha256=sha(old),
                                    original_manifest_modified=False)
    write_json(HERE / 'active_manifest.local.json', result)
    return result


def activate_legacy_bridge():
    import control_bridge
    manifest = local_manifest()
    control_bridge.manifest = lambda: manifest
    return control_bridge
