from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
CHECKPOINTS = HERE / "checkpoints"
RESULTS = HERE / "results"
STATUS = HERE / "status.json"
ACTIVE = HERE / "active_manifest.json"
for path in (DATA, CHECKPOINTS, RESULTS):
    path.mkdir(parents=True, exist_ok=True)

for directory in (ROOT, ROOT / "canonical_three_phase_universal", ROOT / "kaleid_scope_repair_20260906"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


def atomic_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    try:
        # OneDrive can briefly lock the destination during synchronization.
        # Unique temporary names avoid writer collisions; retry only the final
        # same-directory replacement, which remains atomic.
        for attempt in range(50):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 49:
                    raise
                time.sleep(0.1)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def update_status(stage: str, **details) -> None:
    previous = {}
    if STATUS.exists() and stage != 'starting':
        try:
            previous = json.loads(STATUS.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    previous.update(details)
    if stage != "failed":
        previous.pop("error", None)
    previous.update({"stage": stage, "updated_unix": time.time(), "updated_local": time.strftime("%Y-%m-%d %H:%M:%S")})
    atomic_json(STATUS, previous)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_images(images):
    import numpy as np
    values = np.asarray(images, dtype=np.float32)
    axes = tuple(range(2, values.ndim))
    mean = values.mean(axis=axes, keepdims=True)
    std = values.std(axis=axes, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)
