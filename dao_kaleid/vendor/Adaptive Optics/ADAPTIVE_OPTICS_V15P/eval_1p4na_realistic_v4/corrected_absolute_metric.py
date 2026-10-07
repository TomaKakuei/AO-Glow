"""Absolute reference-sphere WFE metric for the repaired Nikon plant."""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = ROOT / "eval_1p4na_hybrid" / "model_repair_scratch"
V3_DIR = ROOT / "eval_1p4na_hybrid" / "v3_corrected"
for path in (MODEL_DIR, V3_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rayoptics.zemax.zmxread as zmxread  # noqa: E402

import run_corrected_highna_validation as corrected  # noqa: E402
from probe_forward_model import trace_explicit  # noqa: E402
from validated_highna_model import build_validated_forward_model  # noqa: E402


REFERENCE_SPHERE_RADIUS_MM = 10.0


def disable_file_logging() -> None:
    loggers = [logging.getLogger()]
    loggers.extend(
        item for item in logging.root.manager.loggerDict.values() if isinstance(item, logging.Logger)
    )
    for logger in loggers:
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()
    zmxread.ZmxGlassHandler.save_replacements = lambda self: None


def reference_sphere_rms(
    opm,
    effective_q: np.ndarray,
    field_mm: float,
    nodes: np.ndarray,
    *,
    focus_offset_mm: float = 0.0,
) -> dict:
    """Trace absolute OPL to a common image-side reference sphere.

    The sphere is centered on the chief-ray camera intercept.  Rays are
    back-propagated along their final air-space directions to the 10-mm sphere;
    piston, image-plane tip/tilt, and defocus are then removed once from the
    resulting absolute OPL samples.
    """
    corrected.apply_normalized_state(opm, np.asarray(effective_q, float))
    chief = trace_explicit(opm, float(field_mm), 0.0, 0.0)
    chief_hit = np.asarray(chief.ray[-1][0], float)
    sphere_center = chief_hit + np.asarray([0.0, 0.0, float(focus_offset_mm)])
    values = []
    for px, py in np.asarray(nodes, float):
        ux = corrected.SAMPLE_SIDE_NA * float(px) / corrected.SAMPLE_INDEX
        uy = corrected.SAMPLE_SIDE_NA * float(py) / corrected.SAMPLE_INDEX
        package = trace_explicit(opm, float(field_mm), ux, uy)
        hit = np.asarray(package.ray[-1][0], float)
        direction = np.asarray(package.ray[-1][1], float)
        direction /= np.linalg.norm(direction)
        delta = hit - sphere_center
        projected = float(np.dot(delta, direction))
        radicand = (
            projected * projected
            - float(np.dot(delta, delta))
            + REFERENCE_SPHERE_RADIUS_MM**2
        )
        if radicand < 0.0:
            raise RuntimeError(
                f"reference-sphere miss at field={field_mm:+.3f}, "
                f"pupil=({px:+.4f},{py:+.4f})"
            )
        backtrack_mm = projected + math.sqrt(radicand)
        values.append(float(package.op) - backtrack_mm)
    opl = np.asarray(values, float)
    design = corrected.pttd_design(np.asarray(nodes, float))
    residual_mm = opl - design @ np.linalg.lstsq(design, opl, rcond=None)[0]
    residual_waves = residual_mm / corrected.WAVELENGTH_MM
    return {
        "absolute_pttd_rms_waves": float(np.sqrt(np.mean(residual_waves**2))),
        "reference_sphere_radius_mm": REFERENCE_SPHERE_RADIUS_MM,
        "chief_camera_hit_mm": chief_hit.tolist(),
        "focus_offset_mm": float(focus_offset_mm),
        "ray_count": int(len(nodes)),
        "residual_waves": residual_waves,
        "opl_to_reference_sphere_mm": opl,
    }


def evaluate_fields(
    effective_q: np.ndarray,
    fields_mm: tuple[float, ...] = corrected.AUDIT_FIELDS_MM,
    nodes: np.ndarray = corrected.AUDIT_PUPIL_NODES,
    *,
    focus_offset_mm: float = 0.0,
) -> dict:
    disable_file_logging()
    opm, _ = build_validated_forward_model()
    output = {}
    for field_mm in fields_mm:
        metric = reference_sphere_rms(
            opm,
            effective_q,
            float(field_mm),
            nodes,
            focus_offset_mm=focus_offset_mm,
        )
        output[f"{field_mm:+.3f}"] = {
            key: value
            for key, value in metric.items()
            if key not in {"residual_waves", "opl_to_reference_sphere_mm"}
        }
    return output


if __name__ == "__main__":
    import json
    from scipy.optimize import minimize_scalar

    disable_file_logging()
    model, _ = build_validated_forward_model()
    zero = np.zeros(len(corrected.SCALES))
    optimum = minimize_scalar(
        lambda offset: reference_sphere_rms(
            model,
            zero,
            0.0,
            corrected.AUDIT_PUPIL_NODES,
            focus_offset_mm=float(offset),
        )["absolute_pttd_rms_waves"],
        bounds=(-2.0, 2.0),
        method="bounded",
        options={"xatol": 1e-7},
    )
    print(json.dumps({
        "best_center_focus_offset_mm": float(optimum.x),
        "best_center_absolute_pttd_rms_waves": float(optimum.fun),
        "fields_at_best_center_focus": evaluate_fields(
            zero,
            focus_offset_mm=float(optimum.x),
        ),
    }, indent=2))
