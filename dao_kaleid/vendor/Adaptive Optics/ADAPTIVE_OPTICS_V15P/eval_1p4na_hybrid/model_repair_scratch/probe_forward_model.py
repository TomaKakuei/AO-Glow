"""Build and audit the patent Embodiment-2 microscope in its physical direction.

This scratch program intentionally does not import or modify the production
``build_reversed_combined`` module.  It selects the objective (surfaces 1--24)
and the configuration-2 Nikon imaging lens (surfaces 28--33) from the original
multiconfiguration ZMX file.  Surfaces 25--27 are ignored in configuration 2.

The ray tests use explicit object-space directions, so the reported sample-side
NA is not inferred from RayOptics' normalized-pupil machinery.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from opticalglass import opticalmedium as om
from rayoptics.optical.opticalmodel import OpticalModel
from rayoptics.raytr import opticalspec as opspec
from rayoptics.raytr import trace, wideangle
from rayoptics.zemax import zmxread


SOURCE_ZMX = Path(__file__).resolve().parent / "Nikon_1p4NA_source_copy.zmx"
WAVELENGTH_NM = 587.5618  # patent d line / primary wavelength in the ZMX
OBJECTIVE_SURFACES = tuple(range(1, 25))
TUBE_SURFACES = tuple(range(28, 34))  # active in multiconfiguration 2
CONFIG2_IMAGE_GAP_MM = 160.5963326923  # THIC 33, configuration 2

# Exact d-line indices listed in US6519092B2 Tables 1 and 3.  This avoids the
# silent n=1.5 placeholders and cross-catalog substitutes produced by the ZMX
# importer for CaF2, KZFH2, F5, and several legacy catalog names.
PATENT_ND_AFTER_SURFACE = {
    0: 1.52216,
    1: 1.52216,
    2: 1.51536,
    3: 1.51823,
    4: 2.02240,
    5: 1.0,
    6: 1.60300,
    7: 1.0,
    8: 1.52682,
    9: 1.49782,
    10: 1.0,
    11: 1.60342,
    12: 1.43385,
    13: 1.0,
    14: 1.61266,
    15: 1.43385,
    16: 1.0,
    17: 1.74950,
    18: 1.43385,
    19: 1.67163,
    20: 1.0,
    21: 1.77279,
    22: 1.80518,
    23: 1.77279,
    24: 1.0,
    28: 1.62280,
    29: 1.74950,
    30: 1.0,
    31: 1.66755,
    32: 1.61266,
    33: 1.0,
}


def _index(medium, wavelength_nm: float) -> float:
    return float(medium.rindex(wavelength_nm))


def _patent_medium(surface_idx: int):
    nd = PATENT_ND_AFTER_SURFACE[surface_idx]
    return om.Air() if nd == 1.0 else om.ConstantIndex(nd, f"patent_nd_s{surface_idx}")


def _append_exact_surface(opm: OpticalModel, src, src_idx: int, gap_idx: int) -> int:
    """Append a deep-copied source profile and its following source gap."""
    sm = opm.seq_model
    src_ifc = src.seq_model.ifcs[src_idx]
    src_gap = src.seq_model.gaps[gap_idx]
    sm.add_surface([0.0, float(src_gap.thi), _patent_medium(src_idx)])
    new_idx = sm.cur_surface
    dst_ifc = sm.ifcs[new_idx]
    dst_ifc.profile = copy.deepcopy(src_ifc.profile)
    dst_ifc.interact_mode = src_ifc.interact_mode
    dst_ifc.decenter = copy.deepcopy(src_ifc.decenter)
    dst_ifc.max_aperture = float(src_ifc.max_aperture)
    dst_ifc.clear_apertures = copy.deepcopy(src_ifc.clear_apertures)
    dst_ifc.edge_apertures = copy.deepcopy(src_ifc.edge_apertures)
    dst_ifc.label = f"src_{src_idx:02d}"
    return new_idx


def build_forward_model(image_gap_mm: float | None = None) -> tuple[OpticalModel, dict[int, int]]:
    """Return sample -> objective -> Nikon imaging lens -> camera model."""
    # The source ZMX represents NA using very wide angular fields.  RayOptics'
    # automatic chief-ray aimer cannot initialize that imported auxiliary
    # specification.  The repaired-model audit launches physical rays directly,
    # so a neutral aim record is sufficient during import/update.
    neutral_aim = lambda *_args, **_kwargs: {
        "aim_pt": np.array([0.0, 0.0]),
        "paraxial_pupil": None,
        "radius": 1.0,
    }
    trace.aim_chief_ray = neutral_aim
    opspec.aim_chief_ray = neutral_aim
    wideangle.find_real_enp = lambda *_args, **_kwargs: (neutral_aim(), None)
    src, _ = zmxread.read_lens_file(SOURCE_ZMX)
    opm = OpticalModel()
    opm.optical_spec.spectral_region.wavelengths = [WAVELENGTH_NM]
    opm.optical_spec.spectral_region.central_wvl = WAVELENGTH_NM

    sm = opm.seq_model
    sm.do_apertures = False
    # The object is on the outside face of the 0.17-mm cover glass.  The source
    # file represents this as surface 0 -> surface 1 with an infinite gap only
    # because its ZMX fields encode angular ray bundles; physical imaging uses
    # zero object distance here.
    sm.gaps[0].thi = 0.0
    sm.gaps[0].medium = _patent_medium(0)

    source_to_new: dict[int, int] = {}
    for i in OBJECTIVE_SURFACES:
        # The gap following objective surface 24 is the patent-specified 150 mm
        # separation to the active imaging lens.
        source_to_new[i] = _append_exact_surface(opm, src, i, i)

    for i in TUBE_SURFACES:
        source_to_new[i] = _append_exact_surface(opm, src, i, i)

    sm.gaps[-1].thi = float(
        CONFIG2_IMAGE_GAP_MM if image_gap_mm is None else image_gap_mm
    )

    sm.stop_surface = None
    # Explicit-ray validation only needs the sequential model.  A full optical
    # update would ask the paraxial normalized-pupil solver to infer an entrance
    # pupil for a zero object gap, which is deliberately outside this audit.
    sm.update_model()
    return opm, source_to_new


def trace_explicit(opm: OpticalModel, object_height_mm: float, ux: float, uy: float = 0.0):
    """Trace a ray from the sample point with explicit direction cosines."""
    uz2 = 1.0 - ux * ux - uy * uy
    if uz2 <= 0:
        raise ValueError("non-propagating direction")
    pt0 = np.array([float(object_height_mm), 0.0, 0.0])
    d0 = np.array([float(ux), float(uy), math.sqrt(uz2)])
    return trace.trace(opm.seq_model, pt0, d0, WAVELENGTH_NM, check_apertures=False)


def ray_record(opm: OpticalModel, object_height_mm: float, na_x: float, na_y: float = 0.0) -> dict:
    n_sample = _index(opm.seq_model.gaps[0].medium, WAVELENGTH_NM)
    ux, uy = na_x / n_sample, na_y / n_sample
    rec: dict = {
        "object_height_mm": object_height_mm,
        "na_x": na_x,
        "na_y": na_y,
        "sample_index": n_sample,
    }
    try:
        ray_pkg = trace_explicit(opm, object_height_mm, ux, uy)
        ray = ray_pkg.ray
        rec.update(
            ok=len(ray) == len(opm.seq_model.ifcs),
            segments=len(ray),
            expected_segments=len(opm.seq_model.ifcs),
            image_x_mm=float(ray[-1][0][0]),
            image_y_mm=float(ray[-1][0][1]),
            final_dir=[float(v) for v in ray[-1][1]],
            max_radius_mm=float(max(np.hypot(seg[0][0], seg[0][1]) for seg in ray)),
        )
    except Exception as exc:  # audit must retain the exact failure
        partial_ray = exc.ray_pkg[0] if getattr(exc, "ray_pkg", None) is not None else []
        rec.update(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            failed_surface=getattr(exc, "surf", None),
            failed_surface_label=(
                getattr(getattr(exc, "ifc", None), "label", None)
                if getattr(exc, "ifc", None) is not None
                else None
            ),
            partial_segments=(
                len(partial_ray) if partial_ray else None
            ),
            last_partial_point=(
                [float(v) for v in partial_ray[-1][0]] if partial_ray else None
            ),
            last_partial_direction=(
                [float(v) for v in partial_ray[-1][1]] if partial_ray else None
            ),
        )
    return rec


def sample_na_bundle(na: float, rings: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99, 1.0)):
    for rho in rings:
        if rho == 0:
            yield 0.0, 0.0
            continue
        for az_deg in range(0, 360, 30):
            az = math.radians(az_deg)
            yield na * rho * math.cos(az), na * rho * math.sin(az)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    opm, mapping = build_forward_model()
    sm = opm.seq_model
    n_sample = _index(sm.gaps[0].medium, WAVELENGTH_NM)

    records = []
    for field_height in (0.0, 0.1, -0.1):
        for na_x, na_y in sample_na_bundle(1.4):
            records.append(ray_record(opm, field_height, na_x, na_y))

    by_field = {}
    for h in (0.0, 0.1, -0.1):
        subset = [r for r in records if r["object_height_mm"] == h]
        valid = [r for r in subset if r.get("ok")]
        by_field[str(h)] = {
            "tested": len(subset),
            "complete": len(valid),
            "image_centroid_mm": [
                float(np.mean([r["image_x_mm"] for r in valid])) if valid else None,
                float(np.mean([r["image_y_mm"] for r in valid])) if valid else None,
            ],
            "image_rms_radius_mm": (
                float(
                    np.sqrt(
                        np.mean(
                            [
                                (r["image_x_mm"] - np.mean([q["image_x_mm"] for q in valid])) ** 2
                                + (r["image_y_mm"] - np.mean([q["image_y_mm"] for q in valid])) ** 2
                                for r in valid
                            ]
                        )
                    )
                )
                if valid
                else None
            ),
        }

    plus = by_field["0.1"]["image_centroid_mm"][0]
    minus = by_field["-0.1"]["image_centroid_mm"][0]
    magnification = (plus - minus) / 0.2 if plus is not None and minus is not None else None

    audit = {
        "source": str(SOURCE_ZMX),
        "wavelength_nm": WAVELENGTH_NM,
        "selected_source_surfaces": list(OBJECTIVE_SURFACES + TUBE_SURFACES),
        "excluded_multiconfiguration_surfaces": [25, 26, 27],
        "source_to_new_surface": mapping,
        "num_interfaces_including_object_image": len(sm.ifcs),
        "sample_index": n_sample,
        "marginal_half_angle_deg": math.degrees(math.asin(1.4 / n_sample)),
        "requested_sample_na": 1.4,
        "field_summary": by_field,
        "signed_lateral_magnification_from_centroids": magnification,
        "records": records,
    }
    path = out_dir / "forward_probe_results.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "records"}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
