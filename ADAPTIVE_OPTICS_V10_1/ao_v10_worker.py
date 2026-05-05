from __future__ import annotations

import json
import sys
from pathlib import Path

def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _parse_payload_text(arg: str) -> str:
    payload_text = arg
    if arg.startswith("@"):
        payload_text = Path(arg[1:]).read_text(encoding="utf-8-sig")
    elif Path(arg).exists():
        payload_text = Path(arg).read_text(encoding="utf-8-sig")
    return payload_text


def _parse_config(arg: str):
    from ao_v10_backend import RunConfig

    payload = json.loads(_parse_payload_text(arg))
    return RunConfig(
        overall_variation_mm=float(payload.get("overall_variation_mm", 1.0)),
        per_anchor_overrides=payload.get("per_anchor_overrides") or {},
        output_name=str(payload.get("output_name", "ui_case")),
        optimizer_maxiter_limit=(
            None
            if payload.get("optimizer_maxiter_limit") in (None, "")
            else int(payload.get("optimizer_maxiter_limit"))
        ),
        max_eval_limit=(
            None
            if payload.get("max_eval_limit") in (None, "")
            else int(payload.get("max_eval_limit"))
        ),
        shwfs_noise_profile=str(payload.get("shwfs_noise_profile", "realistic")),
        shwfs_forward_averages=int(payload.get("shwfs_forward_averages", 1)),
        shwfs_lenslet_count=(
            None if payload.get("shwfs_lenslet_count") in (None, "") else int(payload.get("shwfs_lenslet_count"))
        ),
        shwfs_slope_limit=(
            None if payload.get("shwfs_slope_limit") in (None, "") else int(payload.get("shwfs_slope_limit"))
        ),
        shwfs_canonical_active_lenslets=(
            None
            if payload.get("shwfs_canonical_active_lenslets") in (None, "")
            else int(payload.get("shwfs_canonical_active_lenslets"))
        ),
    )


def main() -> int:
    if len(sys.argv) < 2:
        _emit({"type": "error", "message": "Missing worker mode."})
        return 2
    mode = str(sys.argv[1]).strip().lower()
    try:
        if mode == "env":
            from runtime_bootstrap import bootstrap_runtime

            bootstrap_runtime()
            import sys as _sys

            torch_version = ""
            torch_cuda_available = False
            torch_error = ""
            ok = True
            try:
                import torch

                torch_version = str(torch.__version__)
                torch_cuda_available = bool(torch.cuda.is_available())
            except Exception as exc:
                ok = False
                torch_error = f"{type(exc).__name__}: {exc}"
            _emit(
                {
                    "type": "result",
                    "payload": {
                        "ok": ok,
                        "python_executable": str(_sys.executable),
                        "torch_version": torch_version,
                        "torch_cuda_available": torch_cuda_available,
                        "torch_error": torch_error,
                    },
                }
            )
            return 0

        if len(sys.argv) < 3:
            _emit({"type": "error", "message": "Missing worker config JSON."})
            return 2

        config = _parse_config(sys.argv[2])
        if mode == "before":
            from ao_v10_backend import generate_before_calibration

            result = generate_before_calibration(config, log=lambda message: _emit({"type": "log", "message": message}))
            _emit({"type": "result", "payload": result})
            return 0
        if mode == "repair":
            from ao_v10_backend import run_repair

            result = run_repair(config, log=lambda message: _emit({"type": "log", "message": message}))
            _emit({"type": "result", "payload": result})
            return 0

        _emit({"type": "error", "message": f"Unsupported worker mode: {mode}"})
        return 2
    except Exception as exc:
        _emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
