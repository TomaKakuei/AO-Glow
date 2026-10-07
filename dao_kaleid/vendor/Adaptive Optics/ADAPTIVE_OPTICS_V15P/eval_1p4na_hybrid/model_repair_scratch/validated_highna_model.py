"""Reusable, audited builders for the patent-derived 100x/1.4-NA model.

Only files in ``model_repair_scratch`` use this module.  The production
High-NA scripts remain unchanged while the repaired prescription is reviewed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from rayoptics.optical.opticalmodel import OpticalModel

from probe_forward_model import WAVELENGTH_NM, build_forward_model


# Six actuated rigid lens groups used by the existing High-NA controller.  L1
# (source surfaces 3--4) and the coverslip/immersion interfaces remain fixed.
# Each range includes the air-side boundary that carries the forward decenter
# transform, matching the original controller's group convention.
MOVING_GROUP_SOURCE_BOUNDARIES = {
    "G7": (24, 21),
    "G6": (20, 17),
    "G5": (16, 14),
    "G4": (13, 11),
    "G3": (10, 8),
    "G2": (7, 6),
}


@dataclass(frozen=True)
class ReverseBuildMetadata:
    source_to_forward: dict[int, int]
    source_to_reverse: dict[int, int]
    moving_groups: dict[str, tuple[int, int]]
    dummy_interfaces: dict[str, int]


def _append_reversed_interface(
    reverse_opm: OpticalModel,
    forward_opm: OpticalModel,
    forward_idx: int,
) -> int:
    """Append the exact reciprocal of one forward interface."""
    rsm = reverse_opm.seq_model
    fsm = forward_opm.seq_model
    gap_before = fsm.gaps[forward_idx - 1]

    rsm.add_surface(
        [0.0, float(gap_before.thi), copy.deepcopy(gap_before.medium)]
    )
    reverse_idx = rsm.cur_surface
    rifc = rsm.ifcs[reverse_idx]
    fifc = fsm.ifcs[forward_idx]
    rifc.profile = copy.deepcopy(fifc.profile)
    rifc.profile.flip()
    rifc.interact_mode = fifc.interact_mode
    rifc.max_aperture = float(fifc.max_aperture)
    rifc.clear_apertures = copy.deepcopy(fifc.clear_apertures)
    rifc.edge_apertures = copy.deepcopy(fifc.edge_apertures)
    rifc.label = f"reverse_of_{fifc.label}"
    return reverse_idx


def build_validated_forward_model():
    """Build sample -> camera using exact d-line patent indices/configuration 2."""
    return build_forward_model()


def build_validated_reverse_model(
    *, insert_group_dummies: bool = True
) -> tuple[OpticalModel, ReverseBuildMetadata]:
    """Build camera -> sample as the exact reciprocal of the forward model.

    When requested, zero-power dummy interfaces close each moving-group
    coordinate break without changing optical path length.  The returned group
    mapping is directly usable for paired ``decenter`` / ``reverse`` transforms.
    """
    fwd, source_to_forward = build_forward_model()
    fsm = fwd.seq_model

    rev = OpticalModel()
    rev.optical_spec.spectral_region.wavelengths = [WAVELENGTH_NM]
    rev.optical_spec.spectral_region.central_wvl = WAVELENGTH_NM
    rsm = rev.seq_model
    rsm.do_apertures = False

    # Camera plane to the last imaging-lens surface.
    rsm.gaps[0].thi = float(fsm.gaps[-1].thi)
    rsm.gaps[0].medium = copy.deepcopy(fsm.gaps[-1].medium)

    forward_to_source = {v: k for k, v in source_to_forward.items()}
    source_to_reverse: dict[int, int] = {}
    dummy_interfaces: dict[str, int] = {}
    moving_groups: dict[str, tuple[int, int]] = {}
    group_by_reverse_end_source = {
        source_end: group
        for group, (_source_start, source_end) in MOVING_GROUP_SOURCE_BOUNDARIES.items()
    }

    for forward_idx in range(len(fsm.ifcs) - 2, 0, -1):
        reverse_idx = _append_reversed_interface(rev, fwd, forward_idx)
        source_idx = forward_to_source[forward_idx]
        source_to_reverse[source_idx] = reverse_idx

        if insert_group_dummies and source_idx in group_by_reverse_end_source:
            group = group_by_reverse_end_source[source_idx]
            original_gap = rsm.gaps[reverse_idx]
            original_thickness = float(original_gap.thi)
            original_medium = copy.deepcopy(original_gap.medium)
            rsm.gaps[reverse_idx].thi = 0.0
            rsm.add_surface([0.0, original_thickness, original_medium])
            dummy_idx = rsm.cur_surface
            rsm.ifcs[dummy_idx].label = f"{group}_transform_return"
            dummy_interfaces[group] = dummy_idx
            source_start, _ = MOVING_GROUP_SOURCE_BOUNDARIES[group]
            moving_groups[group] = (source_to_reverse[source_start], dummy_idx)

    if not insert_group_dummies:
        for group, (source_start, source_end) in MOVING_GROUP_SOURCE_BOUNDARIES.items():
            moving_groups[group] = (
                source_to_reverse[source_start],
                source_to_reverse[source_end],
            )

    rsm.stop_surface = None
    rsm.update_model()
    metadata = ReverseBuildMetadata(
        source_to_forward=source_to_forward,
        source_to_reverse=source_to_reverse,
        moving_groups=moving_groups,
        dummy_interfaces=dummy_interfaces,
    )
    return rev, metadata


__all__ = [
    "MOVING_GROUP_SOURCE_BOUNDARIES",
    "ReverseBuildMetadata",
    "WAVELENGTH_NM",
    "build_validated_forward_model",
    "build_validated_reverse_model",
]
