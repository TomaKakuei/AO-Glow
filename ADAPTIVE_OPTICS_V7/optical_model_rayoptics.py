"""Core ray-optics physics engine for active alignment simulation.

The module supports named prescription injection paths, including the current
`2P_AO` design, while staying on direct `rayoptics` imports for a smaller
PyInstaller surface area than `rayoptics.environment`.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft2, fftshift

from opticalglass.modelglass import ModelGlass
from opticalglass.opticalmedium import ConstantIndex
from rayoptics.elem.elements import render_lens_shape
from rayoptics.elem.surface import DecenterData
from rayoptics.optical.opticalmodel import OpticalModel
from rayoptics.raytr.analyses import eval_wavefront
from rayoptics.raytr.analyses import trace_ray_list
from rayoptics.raytr import trace
from rayoptics.raytr.opticalspec import FieldSpec, PupilSpec, WvlSpec


FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
PrescriptionRow = dict[str, Any]

DEFAULT_DESIGN_NAME = "2P_AO"
DEFAULT_WAVELENGTH_NM = 550.0
DEFAULT_MOCK_PUPIL_DIAMETER_MM = 10.0
DEFAULT_2P_AO_PUPIL_DIAMETER_MM = 2.0
DEFAULT_PUPIL_SAMPLES = 64
DEFAULT_FFT_SAMPLES = 256
SHIFT_LIMIT_MM = 2.0
TILT_LIMIT_DEG = 2.0
AXIAL_CLEARANCE_MARGIN_MM = 0.01
DEFAULT_TUBE_LENS_FOCAL_LENGTH_MM = 180.0
DEFAULT_TUBE_LENS_OBJECT_SPACE_MM = 100.0
DEFAULT_TUBE_LENS_IMAGE_SPACE_MM = 180.0
DEFAULT_TUBE_LENS_SEMI_DIAMETER_MM = 25.4
TUBE_LENS_SURFACE_ID = 180001

TWO_PHOTON_AO_RIGID_GROUPS: tuple[tuple[int, ...], ...] = (
    (68, 69),
    (70, 71),
    (72, 73),
    (74, 75, 76),
    (77, 78),
    (79, 80),
    (81, 82, 83),
    (84, 85),
)

TWO_PHOTON_AO_PRESCRIPTION: list[PrescriptionRow] = [
    {"surf": 67, "desc": "Obj", "radius": float("inf"), "thickness": 5.564, "material": "", "sd": None},
    {"surf": 68, "desc": "012-0340", "radius": -23.0, "thickness": 2.0, "material": "F-SILICA", "sd": 15.0},
    {"surf": 69, "desc": "", "radius": float("inf"), "thickness": 28.79, "material": "", "sd": 15.0},
    {"surf": 70, "desc": "LE1015", "radius": -171.6, "thickness": 6.2, "material": "N-BK7", "sd": 25.4},
    {"surf": 71, "desc": "", "radius": 65.2, "thickness": 0.32, "material": "", "sd": 25.4},
    {"surf": 72, "desc": "011-3460", "radius": float("inf"), "thickness": 10.3, "material": "N-BK7", "sd": 25.4},
    {"surf": 73, "desc": "", "radius": -46.71, "thickness": 0.98, "material": "", "sd": 25.4},
    {"surf": 74, "desc": "Custom 1", "radius": 153.895, "thickness": 14.574, "material": "S-PHM52", "sd": 25.4},
    {"surf": 75, "desc": "Custom 1", "radius": -37.831, "thickness": 3.122, "material": "S-TIH53", "sd": 25.4},
    {"surf": 76, "desc": "Assembly compensator", "radius": float("inf"), "thickness": 37.44, "material": "", "sd": 25.4},
    {"surf": 77, "desc": "011-3396", "radius": 36.33, "thickness": 13.4, "material": "N-BK7", "sd": 25.4},
    {"surf": 78, "desc": "", "radius": float("inf"), "thickness": 3.77, "material": "", "sd": 25.4},
    {"surf": 79, "desc": "LE1076", "radius": 30.3, "thickness": 9.7, "material": "N-BK7", "sd": 25.4},
    {"surf": 80, "desc": "", "radius": 65.8, "thickness": 0.18, "material": "", "sd": 25.4},
    {"surf": 81, "desc": "Custom 2", "radius": 24.757, "thickness": 9.26, "material": "S-PHM52", "sd": 15.0},
    {"surf": 82, "desc": "Custom 2", "radius": -53.335, "thickness": 2.104, "material": "S-TIH53", "sd": 15.0},
    {"surf": 83, "desc": "Assembly compensator", "radius": 16.072, "thickness": 10.857, "material": "", "sd": 8.5},
    {"surf": 84, "desc": "Coverglass", "radius": float("inf"), "thickness": 0.15, "material": "N-BK7", "sd": 3.0},
    {"surf": 85, "desc": "Tissue", "radius": float("inf"), "thickness": 0.275, "material": "seawater", "sd": None},
    {"surf": 86, "desc": "Image Plane", "radius": -263.157, "thickness": 0.0, "material": "", "sd": None},
]


class MechanicalLimitWarning(UserWarning):
    """Requested mechanical motion exceeded the allowed travel and was clamped."""


@dataclass(frozen=True)
class SurfacePerturbation:
    """Rigid-body perturbation request applied to a single surface."""

    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: float = 0.0
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0

    def is_zero(self) -> bool:
        return (
            self.dx_mm == 0.0
            and self.dy_mm == 0.0
            and self.dz_mm == 0.0
            and self.tilt_x_deg == 0.0
            and self.tilt_y_deg == 0.0
        )


@dataclass(frozen=True)
class MechanicalEnvelope:
    """Physical motion envelope for one rigid optical element/group."""

    lateral_limit_mm: float
    axial_min_mm: float
    axial_max_mm: float
    tilt_limit_deg: float
    front_clearance_mm: float
    back_clearance_mm: float
    member_surface_ids: tuple[int, ...]

    def clamp(self, perturbation: SurfacePerturbation) -> tuple[SurfacePerturbation, list[str]]:
        clamped_values = {
            "dx_mm": float(np.clip(perturbation.dx_mm, -self.lateral_limit_mm, self.lateral_limit_mm)),
            "dy_mm": float(np.clip(perturbation.dy_mm, -self.lateral_limit_mm, self.lateral_limit_mm)),
            "dz_mm": float(np.clip(perturbation.dz_mm, self.axial_min_mm, self.axial_max_mm)),
            "tilt_x_deg": float(np.clip(perturbation.tilt_x_deg, -self.tilt_limit_deg, self.tilt_limit_deg)),
            "tilt_y_deg": float(np.clip(perturbation.tilt_y_deg, -self.tilt_limit_deg, self.tilt_limit_deg)),
        }
        exceeded: list[str] = []
        for key, requested_value in (
            ("dx_mm", perturbation.dx_mm),
            ("dy_mm", perturbation.dy_mm),
            ("dz_mm", perturbation.dz_mm),
            ("tilt_x_deg", perturbation.tilt_x_deg),
            ("tilt_y_deg", perturbation.tilt_y_deg),
        ):
            clamped_value = clamped_values[key]
            if not np.isclose(requested_value, clamped_value):
                exceeded.append(f"{key}={requested_value:.6f} -> {clamped_value:.6f}")
        return SurfacePerturbation(**clamped_values), exceeded


class RayOpticsPhysicsEngine:
    """Sequential ray-trace wrapper with OPD and FFT PSF generation.

    `design_name="2P_AO"` injects the provided microscope objective path and
    keeps a map from the source prescription surface numbers to the live
    RayOptics sequential indices. `design_name="mock_objective"` restores the
    old six-surface starter model.
    """

    def __init__(
        self,
        *,
        design_name: str = DEFAULT_DESIGN_NAME,
        prescription: list[PrescriptionRow] | None = None,
        wavelength_nm: float = DEFAULT_WAVELENGTH_NM,
        pupil_diameter_mm: float | None = None,
        pupil_samples: int = DEFAULT_PUPIL_SAMPLES,
        fft_samples: int = DEFAULT_FFT_SAMPLES,
        propagation_direction: str = "forward",
        include_tube_lens: bool = False,
        object_space_mm: float | None = None,
        sample_medium_thickness_mm: float | None = None,
        tube_lens_focal_length_mm: float = DEFAULT_TUBE_LENS_FOCAL_LENGTH_MM,
        tube_lens_object_space_mm: float = DEFAULT_TUBE_LENS_OBJECT_SPACE_MM,
        tube_lens_image_space_mm: float = DEFAULT_TUBE_LENS_IMAGE_SPACE_MM,
        tube_lens_semi_diameter_mm: float = DEFAULT_TUBE_LENS_SEMI_DIAMETER_MM,
    ) -> None:
        if wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm must be positive.")
        if pupil_samples < 2:
            raise ValueError("pupil_samples must be at least 2.")
        if fft_samples < pupil_samples:
            raise ValueError("fft_samples must be greater than or equal to pupil_samples.")
        propagation_direction = str(propagation_direction).strip().lower()
        if propagation_direction not in {"forward", "reverse"}:
            raise ValueError("propagation_direction must be 'forward' or 'reverse'.")
        if object_space_mm is not None and not np.isfinite(float(object_space_mm)):
            raise ValueError("object_space_mm must be finite when provided.")
        if sample_medium_thickness_mm is not None and sample_medium_thickness_mm < 0.0:
            raise ValueError("sample_medium_thickness_mm must be non-negative.")
        if tube_lens_focal_length_mm <= 0.0:
            raise ValueError("tube_lens_focal_length_mm must be positive.")
        if tube_lens_object_space_mm < 0.0:
            raise ValueError("tube_lens_object_space_mm must be non-negative.")
        if tube_lens_image_space_mm < 0.0:
            raise ValueError("tube_lens_image_space_mm must be non-negative.")
        if tube_lens_semi_diameter_mm <= 0.0:
            raise ValueError("tube_lens_semi_diameter_mm must be positive.")

        self.design_name = str(design_name)
        self.propagation_direction = propagation_direction
        self.prescription = self._select_prescription(self.design_name, prescription)
        self.wavelength_nm = float(wavelength_nm)
        self.pupil_diameter_mm = self._default_pupil_diameter(
            self.design_name, pupil_diameter_mm
        )
        self.pupil_samples = int(pupil_samples)
        self.fft_samples = int(fft_samples)
        self.include_tube_lens = bool(include_tube_lens)
        self.object_space_mm = None if object_space_mm is None else float(object_space_mm)
        self.sample_medium_thickness_mm = (
            None if sample_medium_thickness_mm is None else float(sample_medium_thickness_mm)
        )
        self.tube_lens_focal_length_mm = float(tube_lens_focal_length_mm)
        self.tube_lens_object_space_mm = float(tube_lens_object_space_mm)
        self.tube_lens_image_space_mm = float(tube_lens_image_space_mm)
        self.tube_lens_semi_diameter_mm = float(tube_lens_semi_diameter_mm)
        self.tube_lens_surface_id = int(TUBE_LENS_SURFACE_ID)
        self.tube_lens_anchor_surface_id: int | None = None

        self.source_surface_to_seq_index: dict[int, int] = {}
        self.seq_index_to_source_surface: dict[int, int] = {}
        self.surface_descriptions: dict[int, str] = {}
        self.group_anchor_to_members: dict[int, tuple[int, ...]] = {}
        self.surface_to_group_anchor: dict[int, int] = {}
        self.group_mechanical_envelopes: dict[int, MechanicalEnvelope] = {}
        self.group_anchor_order: tuple[int, ...] = ()
        self.group_nominal_gap_by_pair: dict[tuple[int, int], float] = {}
        self.minimum_intergroup_gap_mm: float = 0.0

        self.opm = OpticalModel(radius_mode=True)
        self.opm.system_spec.title = self.design_name
        self.seq_model = self.opm["seq_model"]
        self.optical_spec = self.opm["optical_spec"]

        self._configure_optical_spec()
        self._build_active_design()
        if self.design_name == "2P_AO" and self.propagation_direction == "reverse":
            self._reverse_active_design()
        if self.design_name == "2P_AO":
            self._apply_spacing_overrides()
        if self.design_name == "2P_AO" and self.include_tube_lens:
            self._insert_tube_lens_stage()
        if self.design_name == "2P_AO" and self.propagation_direction == "reverse":
            self._auto_tune_reverse_object_pupil()
        self._configure_rigid_groups()
        self.seq_model.do_apertures = False
        self.opm.update_model()

        self._baseline_decenters: list[DecenterData | None] = []
        self._surface_perturbations: dict[int, SurfacePerturbation] = {}
        self.capture_current_baseline()

    def _select_prescription(
        self,
        design_name: str,
        prescription: list[PrescriptionRow] | None,
    ) -> list[PrescriptionRow] | None:
        if prescription is not None:
            return copy.deepcopy(prescription)
        if design_name == "2P_AO":
            return copy.deepcopy(TWO_PHOTON_AO_PRESCRIPTION)
        if design_name == "mock_objective":
            return None
        raise ValueError(
            f"Unknown design_name={design_name!r}. Use '2P_AO', "
            "'mock_objective', or pass an explicit prescription."
        )

    def _default_pupil_diameter(
        self, design_name: str, requested_value: float | None
    ) -> float:
        if requested_value is not None:
            if requested_value <= 0.0:
                raise ValueError("pupil_diameter_mm must be positive.")
            return float(requested_value)
        if design_name == "2P_AO" or self.prescription is not None:
            return DEFAULT_2P_AO_PUPIL_DIAMETER_MM
        return DEFAULT_MOCK_PUPIL_DIAMETER_MM

    def _configure_optical_spec(self) -> None:
        self.seq_model.gaps[0].thi = 1.0e10
        self.optical_spec["pupil"] = PupilSpec(
            self.optical_spec,
            key=("object", "epd"),
            value=self.pupil_diameter_mm,
        )
        self.optical_spec["fov"] = FieldSpec(
            self.optical_spec,
            key=("object", "angle"),
            value=0.0,
            flds=[0.0],
            is_relative=False,
        )
        self.optical_spec["wvls"] = WvlSpec([(self.wavelength_nm, 1.0)], ref_wl=0)

    def _build_active_design(self) -> None:
        if self.prescription is None:
            self._build_mock_objective()
        else:
            self._build_prescription_model(self.prescription)

    def _build_mock_objective(self) -> None:
        self.source_surface_to_seq_index.clear()
        self.seq_index_to_source_surface.clear()
        self.surface_descriptions.clear()

        self.opm.add_lens(power=1.0 / 120.0, bending=0.20, th=4.0, sd=10.0, t=2.0)
        self.opm.add_lens(power=1.0 / 160.0, bending=-0.10, th=3.5, sd=9.0, t=4.0)
        self.opm.add_lens(power=1.0 / 200.0, bending=0.15, th=3.0, sd=8.0, t=35.0)

    def _reverse_active_design(self) -> None:
        """Flip the injected 2P_AO sequence so tracing runs from image side to object side."""

        if self.prescription is None:
            return

        if len(self.seq_model.ifcs) <= 2:
            return

        # Use the sequential-layer flip directly because `OpticalModel.flip()`
        # traverses part-tree nodes that can be unset for prescription-injected
        # models in this environment.
        # Keep the terminal image surface (source id 86) at the far end and
        # only reverse the objective train itself.
        self.seq_model.flip(1, len(self.seq_model.ifcs) - 3)
        self.opm.rebuild_from_seq()
        self.opm.update_model()
        self._refresh_surface_mappings_from_seq_model()

        # The reverse build starts at the original image-side tissue boundary.
        # If the caller did not provide an explicit object-space override, use
        # the tissue thickness as the nominal source-to-first-surface spacing.
        if self.object_space_mm is None and len(self.prescription) >= 2:
            self.seq_model.gaps[0].thi = float(self.prescription[-2]["thickness"])

    def _build_prescription_model(self, prescription: list[PrescriptionRow]) -> None:
        if len(prescription) < 2:
            raise ValueError("Prescription must contain an object row and at least one surface.")

        self.source_surface_to_seq_index.clear()
        self.seq_index_to_source_surface.clear()
        self.surface_descriptions.clear()

        object_row = prescription[0]
        # In the default forward configuration we model an infinity-corrected
        # objective with parallel input rays, so keep object space at infinity.
        # Reverse mode still uses a finite launch spacing on the sample side.
        if self.propagation_direction == "forward":
            self.seq_model.gaps[0].thi = 1.0e10
        else:
            self.seq_model.gaps[0].thi = float(object_row["thickness"])
        self._tag_surface_metadata(
            seq_index=0,
            source_surface_id=int(object_row["surf"]),
            description=str(object_row["desc"] or "Obj"),
        )

        for row in prescription[1:]:
            self._append_prescription_surface(row)

    def _apply_spacing_overrides(self) -> None:
        """Apply user-requested object-space and sample-medium spacing overrides."""

        if self.prescription is None:
            return

        if self.object_space_mm is not None:
            self.seq_model.gaps[0].thi = float(self.object_space_mm)

        if self.sample_medium_thickness_mm is not None:
            for row_index, row in enumerate(self.prescription):
                description = str(row.get("desc", "") or "").strip().lower()
                material = str(row.get("material", "") or "").strip().lower()
                if description == "tissue" or material in {"seawater", "water"}:
                    seq_index = self.source_surface_to_seq_index.get(int(row["surf"]))
                    if seq_index is not None and seq_index < len(self.seq_model.gaps):
                        self.seq_model.gaps[seq_index].thi = float(self.sample_medium_thickness_mm)
                    break

    def _count_traceable_pupil_samples(
        self,
        *,
        num_rays: int = 9,
        pupil_radius_limit: float = 1.0,
        field_index: int = 0,
        wavelength_nm: float | None = None,
        focus: float = 0.0,
    ) -> int:
        wavelength_nm = self.wavelength_nm if wavelength_nm is None else float(wavelength_nm)
        fld = self.optical_spec["fov"].fields[field_index]
        grid = np.linspace(-pupil_radius_limit, pupil_radius_limit, int(num_rays), dtype=np.float64)
        pupil_coords = [
            (float(px), float(py))
            for px in grid
            for py in grid
            if float(px * px + py * py) <= float(pupil_radius_limit * pupil_radius_limit)
        ]
        ray_list = trace_ray_list(
            self.opm,
            pupil_coords,
            fld,
            wavelength_nm,
            float(focus),
            append_if_none=True,
            output_filter="last",
            rayerr_filter="summary",
            check_apertures=False,
        )
        return int(sum(1 for _, _, ray_pkg in ray_list if ray_pkg is not None))

    def _auto_tune_reverse_object_pupil(self) -> None:
        """Shrink object-side EPD in reverse mode until rays trace robustly."""

        if self.propagation_direction != "reverse":
            return

        pupil_spec = self.optical_spec["pupil"]
        if tuple(pupil_spec.key) != ("object", "epd"):
            return

        # Ensure transforms and refractive-index caches reflect any spacing
        # edits that were applied just before this tuning pass.
        self.opm.update_model()

        base_epd = float(pupil_spec.value)
        candidate_scales = (
            1.0,
            0.9,
            0.8,
            0.7,
            0.6,
            0.5,
            0.4,
            0.3,
            0.25,
            0.2,
            0.15,
            0.1,
        )
        best_epd = base_epd
        best_count = -1
        for scale in candidate_scales:
            candidate_epd = max(base_epd * float(scale), 1.0e-3)
            pupil_spec.value = candidate_epd
            count = self._count_traceable_pupil_samples(num_rays=9, pupil_radius_limit=1.0)
            if count > best_count or (count == best_count and candidate_epd > best_epd):
                best_count = count
                best_epd = candidate_epd

        pupil_spec.value = float(best_epd)

    def _append_prescription_surface(self, row: PrescriptionRow) -> None:
        surf_id = int(row["surf"])
        description = str(row["desc"] or f"Surf {surf_id}")
        radius = row["radius"]
        thickness = float(row["thickness"])
        material = self._resolve_material(str(row["material"]))
        semi_diameter = row["sd"]

        surf_data: list[Any] = [radius, thickness]
        if material is not None:
            surf_data.append(material)

        if semi_diameter is None:
            self.seq_model.add_surface(surf_data)
        else:
            sd_value = float(semi_diameter)
            if len(surf_data) == 2:
                self.seq_model.add_surface(surf_data, sd=sd_value)
            else:
                surf_data.append(sd_value)
                self.seq_model.add_surface(surf_data)

        seq_index = len(self.seq_model.ifcs) - 2
        self._tag_surface_metadata(
            seq_index=seq_index,
            source_surface_id=surf_id,
            description=description,
        )

    def _refresh_surface_mappings_from_seq_model(self) -> None:
        """Rebuild live-to-source surface maps after sequence edits."""

        self.source_surface_to_seq_index.clear()
        self.seq_index_to_source_surface.clear()
        self.surface_descriptions.clear()

        for seq_index, ifc in enumerate(self.seq_model.ifcs):
            source_surface_id = getattr(ifc, "source_surface_id", None)
            if source_surface_id is None:
                continue
            source_surface_id = int(source_surface_id)
            self.source_surface_to_seq_index[source_surface_id] = int(seq_index)
            self.seq_index_to_source_surface[int(seq_index)] = source_surface_id
            description = str(
                getattr(ifc, "description", "")
                or getattr(ifc, "label", "")
                or f"Surface {seq_index}"
            )
            self.surface_descriptions[int(seq_index)] = description

    def _seq_surface_z_positions_mm(self) -> list[float]:
        """Return the axial position of each live surface in sequence order."""

        z_positions: list[float] = []
        axial_z = 0.0
        for gap in self.seq_model.gaps:
            z_positions.append(float(axial_z))
            axial_z += float(getattr(gap, "thi", 0.0))
        return z_positions

    def _insert_tube_lens_stage(self) -> None:
        """Insert a paraxial tube lens and place the image surface 180 mm behind it."""

        image_surface_source_id = int(self.prescription[-1]["surf"])
        image_surface_seq_index = self.source_surface_to_seq_index.get(image_surface_source_id)
        if image_surface_seq_index is None:
            raise RuntimeError("Unable to locate the image surface for tube-lens insertion.")

        if self.propagation_direction == "reverse":
            # In reverse mode, source id 68 is the objective-side exit where
            # we append the tube-lens relay.
            objective_last_source_id = int(self.prescription[1]["surf"])
        else:
            objective_last_source_id = int(self.prescription[-2]["surf"])
        self.tube_lens_anchor_surface_id = int(objective_last_source_id)
        objective_last_seq_index = self.source_surface_to_seq_index.get(objective_last_source_id)
        if objective_last_seq_index is None:
            raise RuntimeError("Unable to locate the objective last surface for tube-lens insertion.")

        self.seq_model.gaps[objective_last_seq_index].thi = self.tube_lens_object_space_mm

        self.opm.add_thinlens(
            power=1.0 / self.tube_lens_focal_length_mm,
            indx=1.5,
            idx=image_surface_seq_index,
            t=self.tube_lens_image_space_mm,
            sd=self.tube_lens_semi_diameter_mm,
            lbl="Tube Lens",
        )

        if image_surface_seq_index < len(self.seq_model.ifcs):
            placeholder_ifc = self.seq_model.ifcs[image_surface_seq_index]
            placeholder_ifc.label = "Tube Lens Front"
            placeholder_ifc.description = "Tube lens entrance plane"
            placeholder_ifc.source_surface_id = None

        tube_lens_ifc = None
        for probe_index in range(
            image_surface_seq_index,
            min(image_surface_seq_index + 4, len(self.seq_model.ifcs)),
        ):
            if type(self.seq_model.ifcs[probe_index]).__name__ == "ThinLens":
                tube_lens_ifc = self.seq_model.ifcs[probe_index]
                break

        if tube_lens_ifc is None:
            raise RuntimeError("Unable to locate the inserted thin-lens interface.")

        tube_lens_ifc.label = "Tube Lens"
        tube_lens_ifc.description = "Paraxial tube lens"
        tube_lens_ifc.source_surface_id = self.tube_lens_surface_id

        self._refresh_surface_mappings_from_seq_model()

    def _resolve_material(self, material_name: str) -> Any | None:
        material_key = material_name.strip()
        if material_key == "":
            return None
        if material_key == "N-BK7":
            return "N-BK7, Schott"
        if material_key == "S-PHM52":
            return "S-PHM52, Ohara"
        if material_key == "S-TIH53":
            return "S-TIH53, Ohara"
        if material_key == "F-SILICA":
            # Opticalglass in this environment does not expose this alias
            # directly, so use a local ModelGlass equivalent.
            return ModelGlass(1.458458, 67.82, "F-SILICA")
        if material_key.lower() == "seawater":
            return ConstantIndex(1.339, "seawater")
        raise ValueError(f"Unsupported material alias: {material_name!r}")

    def _tag_surface_metadata(
        self,
        *,
        seq_index: int,
        source_surface_id: int,
        description: str,
    ) -> None:
        ifc = self.seq_model.ifcs[seq_index]
        ifc.label = description
        ifc.description = description
        ifc.source_surface_id = source_surface_id
        self.source_surface_to_seq_index[source_surface_id] = seq_index
        self.seq_index_to_source_surface[seq_index] = source_surface_id
        self.surface_descriptions[seq_index] = description

    def _configure_rigid_groups(self) -> None:
        self.group_anchor_to_members.clear()
        self.surface_to_group_anchor.clear()
        self.group_mechanical_envelopes.clear()
        self.group_anchor_order = ()
        self.group_nominal_gap_by_pair.clear()
        self.minimum_intergroup_gap_mm = 0.0

        if self.prescription is None:
            return

        prescription_surface_ids = {int(row["surf"]) for row in self.prescription}
        for group_members in TWO_PHOTON_AO_RIGID_GROUPS:
            members = tuple(
                int(surface_id)
                for surface_id in group_members
                if int(surface_id) in prescription_surface_ids
            )
            if not members:
                continue

            anchor_surface_id = int(members[0])
            self.group_anchor_to_members[anchor_surface_id] = members
            for surface_id in members:
                self.surface_to_group_anchor[int(surface_id)] = anchor_surface_id
            self.group_mechanical_envelopes[anchor_surface_id] = self._build_group_envelope(
                members
            )

        self.group_anchor_order = tuple(sorted(self.group_anchor_to_members))
        self._configure_group_gap_constraints()

    def _configure_group_gap_constraints(self) -> None:
        if self.prescription is None or not self.group_anchor_order:
            self.group_nominal_gap_by_pair.clear()
            self.minimum_intergroup_gap_mm = 0.0
            return

        row_lookup = {
            int(row["surf"]): (row_index, row)
            for row_index, row in enumerate(self.prescription)
        }
        gap_by_pair: dict[tuple[int, int], float] = {}
        finite_gaps_mm: list[float] = []
        anchors = list(self.group_anchor_order)
        for pair_index in range(len(anchors) - 1):
            left_anchor = int(anchors[pair_index])
            right_anchor = int(anchors[pair_index + 1])
            left_last_surface_id = int(self.group_anchor_to_members[left_anchor][-1])
            _, left_last_row = row_lookup[left_last_surface_id]
            nominal_gap_mm = float(left_last_row.get("thickness", 0.0))
            if not np.isfinite(nominal_gap_mm):
                continue
            gap_by_pair[(left_anchor, right_anchor)] = nominal_gap_mm
            if nominal_gap_mm > 0.0:
                finite_gaps_mm.append(nominal_gap_mm)

        self.group_nominal_gap_by_pair = gap_by_pair
        self.minimum_intergroup_gap_mm = float(min(finite_gaps_mm)) if finite_gaps_mm else 0.0

    def _build_group_envelope(self, member_surface_ids: Sequence[int]) -> MechanicalEnvelope:
        row_lookup = {
            int(row["surf"]): (row_index, row)
            for row_index, row in enumerate(self.prescription or [])
        }
        member_surface_ids = tuple(int(surface_id) for surface_id in member_surface_ids)
        first_surface_id = member_surface_ids[0]
        last_surface_id = member_surface_ids[-1]
        first_row_index, first_row = row_lookup[first_surface_id]
        last_row_index, last_row = row_lookup[last_surface_id]

        if first_row_index > 0:
            _, previous_row = row_lookup[int((self.prescription or [])[first_row_index - 1]["surf"])]
        else:
            previous_row = {"desc": "Obj", "thickness": float("inf")}

        next_row = (
            (self.prescription or [])[last_row_index + 1]
            if last_row_index + 1 < len(self.prescription or [])
            else {"desc": "Image Plane"}
        )

        previous_desc = str(previous_row.get("desc", "") or "").strip().lower()
        next_desc = str(next_row.get("desc", "") or "").strip().lower()

        front_clearance_mm = float(previous_row.get("thickness", float("inf")))
        back_clearance_mm = float(last_row.get("thickness", float("inf")))

        axial_min_mm = -max(front_clearance_mm - AXIAL_CLEARANCE_MARGIN_MM, 0.0)
        axial_max_mm = max(back_clearance_mm - AXIAL_CLEARANCE_MARGIN_MM, 0.0)

        # Only the 74-76 compensator group is allowed to use the full physical
        # axial clearance. All other groups keep the tighter of the physical
        # clearance and the fixed 2 mm travel envelope.
        if int(first_surface_id) != 74:
            axial_min_mm = max(axial_min_mm, -SHIFT_LIMIT_MM)
            axial_max_mm = min(axial_max_mm, SHIFT_LIMIT_MM)

        if axial_max_mm < axial_min_mm:
            midpoint = 0.5 * (axial_min_mm + axial_max_mm)
            axial_min_mm = midpoint
            axial_max_mm = midpoint

        return MechanicalEnvelope(
            lateral_limit_mm=SHIFT_LIMIT_MM,
            axial_min_mm=axial_min_mm,
            axial_max_mm=axial_max_mm,
            tilt_limit_deg=TILT_LIMIT_DEG,
            front_clearance_mm=float(front_clearance_mm),
            back_clearance_mm=float(back_clearance_mm),
            member_surface_ids=member_surface_ids,
        )

    def _anchor_surface_id_for_source_surface(self, source_surface_id: int) -> int:
        return int(self.surface_to_group_anchor.get(int(source_surface_id), int(source_surface_id)))

    def _group_members_for_source_surface(self, source_surface_id: int) -> tuple[int, ...]:
        anchor_surface_id = self._anchor_surface_id_for_source_surface(source_surface_id)
        return self.group_anchor_to_members.get(anchor_surface_id, (int(source_surface_id),))

    def get_group_anchor_surface_ids(
        self,
        *,
        include_coverglass: bool = True,
        include_sample_media: bool = False,
    ) -> list[int]:
        movable_surface_ids = self.get_movable_surface_ids(
            include_coverglass=include_coverglass,
            include_sample_media=include_sample_media,
        )
        anchors: list[int] = []
        seen: set[int] = set()
        for surface_id in movable_surface_ids:
            anchor_surface_id = self._anchor_surface_id_for_source_surface(surface_id)
            if anchor_surface_id not in seen:
                seen.add(anchor_surface_id)
                anchors.append(anchor_surface_id)
        return anchors

    def get_group_mechanical_envelope(self, surface_index: int) -> MechanicalEnvelope:
        seq_index = self._resolve_surface_index(surface_index)
        source_surface_id = self.seq_index_to_source_surface.get(seq_index, seq_index)
        anchor_surface_id = self._anchor_surface_id_for_source_surface(source_surface_id)
        envelope = self.group_mechanical_envelopes.get(anchor_surface_id)
        if envelope is None:
            return MechanicalEnvelope(
                lateral_limit_mm=SHIFT_LIMIT_MM,
                axial_min_mm=-SHIFT_LIMIT_MM,
                axial_max_mm=SHIFT_LIMIT_MM,
                tilt_limit_deg=TILT_LIMIT_DEG,
                front_clearance_mm=np.inf,
                back_clearance_mm=np.inf,
                member_surface_ids=(int(source_surface_id),),
            )
        return envelope

    def get_movable_surface_ids(
        self,
        *,
        include_coverglass: bool = True,
        include_sample_media: bool = False,
    ) -> list[int]:
        """Return source-surface ids for movable optical elements in the prescription.

        For injected prescriptions, movable elements are identified by rows that
        introduce a non-empty optical material. Air gaps and the image plane are
        excluded automatically. Sample media such as seawater are excluded by
        default because they are not usually mounted as actuated optics.
        """

        if self.prescription is None:
            return [
                self.seq_index_to_source_surface.get(seq_index, seq_index)
                for seq_index in range(1, max(len(self.seq_model.ifcs) - 1, 1))
            ]

        movable_surface_ids: list[int] = []
        for row in self.prescription[1:]:
            material_name = str(row.get("material", "")).strip()
            if material_name == "":
                continue

            description = str(row.get("desc", "")).strip().lower()
            if not include_coverglass and description == "coverglass":
                continue
            if not include_sample_media and material_name.lower() in {"seawater", "water"}:
                continue

            movable_surface_ids.append(int(row["surf"]))

        return movable_surface_ids

    def get_surface_catalog(
        self,
        *,
        include_object: bool = False,
        include_image: bool = False,
        include_sample_media: bool = False,
    ) -> list[dict[str, Any]]:
        """Return display metadata for physical prescription surfaces.

        The GUI uses this to render every optical interface instead of only the
        compensator surface. For injected prescriptions we keep the original
        source surface ids so the UI can address the same indices the user sees
        in the optical design.
        """

        if self.prescription is None:
            catalog: list[dict[str, Any]] = []
            for seq_index in range(len(self.seq_model.ifcs)):
                ifc = self.seq_model.ifcs[seq_index]
                if ifc.interact_mode in {"dummy", "phantom"}:
                    continue
                catalog.append(
                    {
                        "surface_id": int(self.seq_index_to_source_surface.get(seq_index, seq_index)),
                        "seq_index": int(seq_index),
                        "label": str(getattr(ifc, "label", "") or f"Surface {seq_index}"),
                        "description": str(getattr(ifc, "description", "") or ""),
                        "material": "",
                        "radius": float(getattr(getattr(ifc, "profile", None), "cv", 0.0) or 0.0),
                        "thickness": float(getattr(self.seq_model.gaps[seq_index], "thi", 0.0)),
                        "semi_diameter": None,
                        "is_compensator": False,
                    }
                )
            return catalog

        catalog = []
        seq_z_positions = self._seq_surface_z_positions_mm()
        seen_seq_indices: set[int] = set()
        for row_index, row in enumerate(self.prescription):
            surface_id = int(row["surf"])
            description = str(row.get("desc", "") or "")
            material = str(row.get("material", "") or "")

            if not include_object and description.lower() == "obj":
                continue
            if not include_image and description.lower() in {"image plane", "image surface"}:
                continue
            if not include_sample_media and material.lower() in {"seawater", "water"}:
                continue

            seq_index = self.source_surface_to_seq_index.get(surface_id)
            if seq_index is None:
                continue

            catalog.append(
                {
                    "surface_id": surface_id,
                    "seq_index": int(seq_index),
                    "label": description or f"Surf {surface_id}",
                    "description": description,
                    "material": material,
                    "radius": float(row["radius"]) if np.isfinite(float(row["radius"])) else np.inf,
                    "thickness": float(row["thickness"]),
                    "semi_diameter": None if row.get("sd") is None else float(row["sd"]),
                    "is_compensator": surface_id in {74, 75, 76},
                    "group_anchor_surface_id": self._anchor_surface_id_for_source_surface(surface_id),
                    "group_surface_ids": list(self._group_members_for_source_surface(surface_id)),
                    "is_glued_group": len(self._group_members_for_source_surface(surface_id)) > 2,
                    "mechanical_envelope": self.get_group_mechanical_envelope(surface_id),
                    "nominal_z_mm": float(seq_z_positions[seq_index]) if seq_index < len(seq_z_positions) else float("nan"),
                    "render_polygon_mm": self._render_polygon_for_prescription_row(row_index),
                    "render_end_surface_id": self._render_end_surface_id_for_prescription_row(row_index),
                }
            )
            seen_seq_indices.add(int(seq_index))

        for seq_index, ifc in enumerate(self.seq_model.ifcs):
            if seq_index in seen_seq_indices:
                continue
            if getattr(ifc, "interact_mode", "") in {"dummy", "phantom"}:
                continue

            source_surface_id = getattr(ifc, "source_surface_id", None)
            if source_surface_id is None:
                continue

            source_surface_id = int(source_surface_id)
            label = str(getattr(ifc, "label", "") or f"Surface {source_surface_id}")
            description = str(getattr(ifc, "description", "") or label)
            if not include_object and description.lower() == "obj":
                continue
            if not include_image and description.lower() in {"image plane", "image surface"}:
                continue

            semi_diameter = getattr(ifc, "surface_od", lambda: None)()
            render_polygon_mm = None
            if hasattr(ifc, "full_profile"):
                try:
                    sd = float(semi_diameter) if semi_diameter is not None else self.tube_lens_semi_diameter_mm
                    profile = ifc.full_profile((-sd, sd))
                    render_polygon_mm = [
                        [[float(point[0]), float(point[1])] for point in profile]
                    ]
                except Exception:
                    render_polygon_mm = None

            catalog.append(
                {
                    "surface_id": source_surface_id,
                    "seq_index": int(seq_index),
                    "label": label,
                    "description": description,
                    "material": "",
                    "radius": np.inf,
                    "thickness": float(self.seq_model.gaps[seq_index].thi),
                    "semi_diameter": None if semi_diameter is None else float(semi_diameter),
                    "is_compensator": False,
                    "group_anchor_surface_id": self._anchor_surface_id_for_source_surface(source_surface_id),
                    "group_surface_ids": list(self._group_members_for_source_surface(source_surface_id)),
                    "is_glued_group": len(self._group_members_for_source_surface(source_surface_id)) > 2,
                    "mechanical_envelope": self.get_group_mechanical_envelope(source_surface_id),
                    "nominal_z_mm": float(seq_z_positions[seq_index]) if seq_index < len(seq_z_positions) else float("nan"),
                    "render_polygon_mm": render_polygon_mm,
                    "render_end_surface_id": None,
                    "optical_power": float(getattr(ifc, "optical_power", 0.0)),
                    "is_tube_lens": source_surface_id == self.tube_lens_surface_id,
                }
            )

        catalog.sort(key=lambda item: int(item["seq_index"]))
        return catalog

    def _render_end_surface_id_for_prescription_row(self, row_index: int) -> int | None:
        if self.prescription is None or row_index < 0 or row_index >= len(self.prescription) - 1:
            return None
        row = self.prescription[row_index]
        material = str(row.get("material", "") or "").strip()
        if material == "":
            return None
        return int(self.prescription[row_index + 1]["surf"])

    def _render_polygon_for_prescription_row(
        self,
        row_index: int,
    ) -> list[list[list[float]]] | None:
        if self.prescription is None or row_index < 0 or row_index >= len(self.prescription) - 1:
            return None

        row = self.prescription[row_index]
        next_row = self.prescription[row_index + 1]
        material = str(row.get("material", "") or "").strip()
        if material == "":
            return None

        semi_diameter_candidates = [
            float(value)
            for value in (
                row.get("sd"),
                next_row.get("sd"),
            )
            if value is not None
        ]
        if not semi_diameter_candidates:
            return None
        sd = max(semi_diameter_candidates)
        half_aperture_mm = float(sd)
        thickness_mm = float(row["thickness"])

        def sag_mm(radius_mm: float, y_mm: FloatArray) -> FloatArray:
            if not np.isfinite(radius_mm):
                return np.zeros_like(y_mm)
            radius_abs_mm = abs(float(radius_mm))
            # Keep the sampled profile inside the real sphere even when the
            # requested semi-diameter is slightly larger than the curvature.
            clipped = np.clip(np.abs(y_mm), 0.0, max(radius_abs_mm - 1.0e-9, 0.0))
            root = np.sqrt(np.maximum(radius_abs_mm * radius_abs_mm - clipped * clipped, 0.0))
            return float(radius_mm) - np.sign(float(radius_mm)) * root

        y_samples = np.linspace(-half_aperture_mm, half_aperture_mm, 129, dtype=np.float64)
        x_front = sag_mm(float(row["radius"]), y_samples)
        # The next prescription row is the rear interface of the current lens.
        # For cemented groups we use the direct signed sag about the rear
        # vertex. For isolated singlets followed by an air gap, some catalog
        # sign conventions are effectively mirrored in the reference drawings,
        # so prefer the orientation that stays inside the available air gap.
        direct_back = thickness_mm + sag_mm(float(next_row["radius"]), y_samples)
        mirrored_back = thickness_mm - sag_mm(float(next_row["radius"]), y_samples)
        x_back = direct_back
        next_is_air_interface = str(next_row.get("material", "") or "").strip() == ""
        if next_is_air_interface:
            air_gap_mm = float(next_row.get("thickness", 0.0))
            allowed_back_profile = np.full_like(y_samples, thickness_mm + air_gap_mm)
            if row_index + 2 < len(self.prescription):
                following_row = self.prescription[row_index + 2]
                following_material = str(following_row.get("material", "") or "").strip()
                if following_material != "":
                    following_sd = following_row.get("sd")
                    overlap_half_aperture_mm = half_aperture_mm
                    if following_sd is not None:
                        overlap_half_aperture_mm = min(overlap_half_aperture_mm, float(following_sd))
                    overlap_mask = np.abs(y_samples) <= overlap_half_aperture_mm + 1.0e-9
                    following_front = (
                        thickness_mm
                        + air_gap_mm
                        + sag_mm(float(following_row["radius"]), y_samples)
                    )
                    allowed_back_profile = np.where(overlap_mask, following_front, np.inf)

            direct_overrun_mm = float(np.max(direct_back - allowed_back_profile))
            mirrored_overrun_mm = float(np.max(mirrored_back - allowed_back_profile))
            direct_fits = direct_overrun_mm <= 1.0e-9
            mirrored_fits = mirrored_overrun_mm <= 1.0e-9
            if mirrored_fits and not direct_fits:
                x_back = mirrored_back
            elif direct_fits and not mirrored_fits:
                x_back = direct_back
            elif mirrored_fits and direct_fits:
                x_back = direct_back
            elif mirrored_overrun_mm < direct_overrun_mm:
                x_back = mirrored_back

        polygon = np.column_stack(
            (
                np.concatenate([x_front, x_back[::-1]]),
                np.concatenate([y_samples, y_samples[::-1]]),
            )
        )
        return [
            [[float(point[0]), float(point[1])] for point in polygon]
        ]

    def get_prescription_surface_index(self, source_surface_id: int) -> int:
        """Return the RayOptics sequential index for a source prescription surface."""

        try:
            return self.source_surface_to_seq_index[int(source_surface_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown source surface id: {source_surface_id}") from exc

    def capture_current_baseline(self) -> None:
        """Treat the current sequential model as the new nominal alignment state."""

        self.opm.update_model()
        self._baseline_decenters = [
            copy.deepcopy(ifc.decenter) for ifc in self.seq_model.ifcs
        ]
        self._surface_perturbations.clear()

    def _resolve_surface_index(self, surface_index: int) -> int:
        seq_index = int(surface_index)
        if 0 <= seq_index < len(self.seq_model.ifcs):
            return seq_index
        mapped_index = self.source_surface_to_seq_index.get(seq_index)
        if mapped_index is not None:
            return mapped_index
        raise IndexError(
            f"surface_index={surface_index} is neither a valid sequential index nor "
            "a known source prescription surface id."
        )

    def _validate_surface_index(self, seq_index: int) -> None:
        if seq_index < 0 or seq_index >= len(self.seq_model.ifcs):
            raise IndexError(
                f"surface_index={seq_index} is outside the valid sequential range "
                f"[0, {len(self.seq_model.ifcs) - 1}]."
            )
        if self.seq_model.ifcs[seq_index].interact_mode in {"dummy", "phantom"}:
            raise ValueError(
                "surface_index must reference a physical optical surface, not the "
                "object/image dummy interfaces."
            )

    def _clamp_perturbation(
        self,
        source_surface_id: int,
        dx: float,
        dy: float,
        dz: float,
        tilt_x: float,
        tilt_y: float,
    ) -> SurfacePerturbation:
        requested = SurfacePerturbation(
            dx_mm=float(dx),
            dy_mm=float(dy),
            dz_mm=float(dz),
            tilt_x_deg=float(tilt_x),
            tilt_y_deg=float(tilt_y),
        )
        envelope = self.get_group_mechanical_envelope(int(source_surface_id))
        clamped, exceeded = envelope.clamp(requested)

        if exceeded:
            member_tag = ",".join(str(surface_id) for surface_id in envelope.member_surface_ids)
            warnings.warn(
                "Perturbation exceeded mechanical limits for rigid group "
                f"[{member_tag}] and was clamped: " + ", ".join(exceeded),
                MechanicalLimitWarning,
                stacklevel=2,
            )

        return clamped

    def _anchor_state_map(self) -> dict[int, SurfacePerturbation]:
        state: dict[int, SurfacePerturbation] = {}
        for anchor_surface_id in self.group_anchor_order:
            anchor_seq_index = self._resolve_surface_index(anchor_surface_id)
            state[int(anchor_surface_id)] = copy.deepcopy(
                self._surface_perturbations.get(anchor_seq_index, SurfacePerturbation())
            )
        return state

    def _project_anchor_state_to_gap_constraints(
        self,
        candidate_state: dict[int, SurfacePerturbation],
        *,
        adjustable_anchor_ids: set[int],
    ) -> tuple[dict[int, SurfacePerturbation], list[str]]:
        if not self.group_nominal_gap_by_pair or self.minimum_intergroup_gap_mm <= 0.0:
            return candidate_state, []

        projected = {
            int(anchor): copy.deepcopy(perturbation)
            for anchor, perturbation in candidate_state.items()
        }
        bounds = {
            int(anchor): self.group_mechanical_envelopes[int(anchor)]
            for anchor in self.group_anchor_order
            if int(anchor) in self.group_mechanical_envelopes
        }
        warnings_list: list[str] = []
        tolerance = 1.0e-12

        for _ in range(max(1, len(self.group_anchor_order) * 4)):
            changed = False
            for left_anchor, right_anchor in zip(self.group_anchor_order[:-1], self.group_anchor_order[1:]):
                nominal_gap_mm = self.group_nominal_gap_by_pair.get((int(left_anchor), int(right_anchor)))
                if nominal_gap_mm is None:
                    continue
                required_delta = float(self.minimum_intergroup_gap_mm - nominal_gap_mm)
                left_state = projected[int(left_anchor)]
                right_state = projected[int(right_anchor)]
                current_delta = float(right_state.dz_mm - left_state.dz_mm)
                deficit = required_delta - current_delta
                if deficit <= tolerance:
                    continue

                left_envelope = bounds[int(left_anchor)]
                right_envelope = bounds[int(right_anchor)]
                left_room = (
                    float(left_state.dz_mm - left_envelope.axial_min_mm)
                    if int(left_anchor) in adjustable_anchor_ids
                    else 0.0
                )
                right_room = (
                    float(right_envelope.axial_max_mm - right_state.dz_mm)
                    if int(right_anchor) in adjustable_anchor_ids
                    else 0.0
                )
                move_left = min(0.5 * deficit, max(left_room, 0.0))
                move_right = min(0.5 * deficit, max(right_room, 0.0))
                remaining = deficit - move_left - move_right
                if remaining > tolerance:
                    extra_right = min(remaining, max(right_room - move_right, 0.0))
                    move_right += extra_right
                    remaining -= extra_right
                if remaining > tolerance:
                    extra_left = min(remaining, max(left_room - move_left, 0.0))
                    move_left += extra_left
                    remaining -= extra_left

                if move_left <= tolerance and move_right <= tolerance:
                    warnings_list.append(
                        "Minimum inter-group gap constraint could not be satisfied for "
                        f"adjacent rigid groups {self.group_anchor_to_members[int(left_anchor)]} and "
                        f"{self.group_anchor_to_members[int(right_anchor)]}."
                    )
                    continue

                if move_left > tolerance:
                    projected[int(left_anchor)] = SurfacePerturbation(
                        dx_mm=float(left_state.dx_mm),
                        dy_mm=float(left_state.dy_mm),
                        dz_mm=float(left_state.dz_mm - move_left),
                        tilt_x_deg=float(left_state.tilt_x_deg),
                        tilt_y_deg=float(left_state.tilt_y_deg),
                    )
                    changed = True
                if move_right > tolerance:
                    right_state = projected[int(right_anchor)]
                    projected[int(right_anchor)] = SurfacePerturbation(
                        dx_mm=float(right_state.dx_mm),
                        dy_mm=float(right_state.dy_mm),
                        dz_mm=float(right_state.dz_mm + move_right),
                        tilt_x_deg=float(right_state.tilt_x_deg),
                        tilt_y_deg=float(right_state.tilt_y_deg),
                    )
                    changed = True
            if not changed:
                break

        for left_anchor, right_anchor in zip(self.group_anchor_order[:-1], self.group_anchor_order[1:]):
            nominal_gap_mm = self.group_nominal_gap_by_pair.get((int(left_anchor), int(right_anchor)))
            if nominal_gap_mm is None:
                continue
            left_state = projected[int(left_anchor)]
            right_state = projected[int(right_anchor)]
            actual_gap_mm = float(nominal_gap_mm + right_state.dz_mm - left_state.dz_mm)
            if actual_gap_mm + tolerance < float(self.minimum_intergroup_gap_mm):
                warnings_list.append(
                    "Minimum inter-group gap constraint remains violated for "
                    f"adjacent rigid groups {self.group_anchor_to_members[int(left_anchor)]} and "
                    f"{self.group_anchor_to_members[int(right_anchor)]}: "
                    f"{actual_gap_mm:.6f} mm < {self.minimum_intergroup_gap_mm:.6f} mm."
                )

        return projected, warnings_list

    def _restore_baseline_decenters(self) -> None:
        for ifc, baseline in zip(self.seq_model.ifcs, self._baseline_decenters):
            ifc.decenter = copy.deepcopy(baseline)

    def _decenter_from_perturbation(self, perturbation: SurfacePerturbation) -> DecenterData:
        decenter = DecenterData(
            "dec and return",
            x=perturbation.dx_mm,
            y=perturbation.dy_mm,
            alpha=perturbation.tilt_x_deg,
            beta=perturbation.tilt_y_deg,
        )
        decenter.dec[2] = perturbation.dz_mm
        decenter.update()
        return decenter

    def _coerce_surface_perturbation(
        self,
        source_surface_id: int,
        value: SurfacePerturbation | Sequence[float] | dict[str, float],
    ) -> SurfacePerturbation:
        if isinstance(value, SurfacePerturbation):
            return self._clamp_perturbation(
                source_surface_id,
                value.dx_mm,
                value.dy_mm,
                value.dz_mm,
                value.tilt_x_deg,
                value.tilt_y_deg,
            )

        if isinstance(value, dict):
            return self._clamp_perturbation(
                source_surface_id,
                float(value.get("dx", value.get("dx_mm", 0.0))),
                float(value.get("dy", value.get("dy_mm", 0.0))),
                float(value.get("dz", value.get("dz_mm", 0.0))),
                float(value.get("tilt_x", value.get("tilt_x_deg", 0.0))),
                float(value.get("tilt_y", value.get("tilt_y_deg", 0.0))),
            )

        components = np.asarray(value, dtype=np.float64).reshape(-1)
        if components.size == 3:
            return self._clamp_perturbation(
                source_surface_id,
                float(components[0]),
                float(components[1]),
                float(components[2]),
                0.0,
                0.0,
            )
        if components.size == 5:
            return self._clamp_perturbation(
                source_surface_id,
                float(components[0]),
                float(components[1]),
                float(components[2]),
                float(components[3]),
                float(components[4]),
            )
        raise ValueError("Surface perturbation values must contain 3 or 5 floats.")

    def _apply_all_perturbations(self) -> None:
        self._restore_baseline_decenters()
        for seq_index, perturbation in self._surface_perturbations.items():
            self.seq_model.ifcs[seq_index].decenter = self._decenter_from_perturbation(
                perturbation
            )
        self.opm.update_model()

    def perturb_lens(
        self,
        surface_index: int,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        tilt_x: float = 0.0,
        tilt_y: float = 0.0,
    ) -> SurfacePerturbation:
        """Apply a locally returning decenter/tilt perturbation to one surface.

        `surface_index` may be either the live RayOptics sequential index or,
        for injected prescriptions such as `2P_AO`, the original source surface
        id from the prescription.
        """

        seq_index = self._resolve_surface_index(surface_index)
        self._validate_surface_index(seq_index)
        source_surface_id = self.seq_index_to_source_surface.get(seq_index, seq_index)
        perturbation = self._clamp_perturbation(
            source_surface_id,
            dx,
            dy,
            dz,
            tilt_x,
            tilt_y,
        )
        anchor_surface_id = self._anchor_surface_id_for_source_surface(source_surface_id)
        anchor_state = self._anchor_state_map()
        anchor_state[int(anchor_surface_id)] = perturbation
        anchor_state, gap_warnings = self._project_anchor_state_to_gap_constraints(
            anchor_state,
            adjustable_anchor_ids={int(anchor_surface_id)},
        )
        perturbation = anchor_state[int(anchor_surface_id)]
        if gap_warnings:
            for message in gap_warnings:
                warnings.warn(message, MechanicalLimitWarning, stacklevel=2)

        member_surface_ids = self._group_members_for_source_surface(source_surface_id)

        for member_surface_id in member_surface_ids:
            member_seq_index = self._resolve_surface_index(member_surface_id)
            if perturbation.is_zero():
                self._surface_perturbations.pop(member_seq_index, None)
            else:
                self._surface_perturbations[member_seq_index] = perturbation

        self._apply_all_perturbations()
        return perturbation

    def set_surface_perturbations(
        self,
        perturbations: dict[int, SurfacePerturbation | Sequence[float] | dict[str, float]],
    ) -> dict[int, SurfacePerturbation]:
        """Update multiple surface perturbations and refresh the model once.

        The input keys may be either live sequential indices or injected
        prescription surface ids. Surfaces omitted from `perturbations` keep
        their current perturbation state.
        """

        applied: dict[int, SurfacePerturbation] = {}
        grouped_requests: dict[int, SurfacePerturbation] = {}
        for surface_index, requested_value in perturbations.items():
            seq_index = self._resolve_surface_index(surface_index)
            self._validate_surface_index(seq_index)
            source_surface_id = self.seq_index_to_source_surface.get(seq_index, seq_index)
            anchor_surface_id = self._anchor_surface_id_for_source_surface(source_surface_id)
            perturbation = self._coerce_surface_perturbation(source_surface_id, requested_value)

            existing_request = grouped_requests.get(anchor_surface_id)
            if existing_request is not None and existing_request != perturbation:
                raise ValueError(
                    f"Conflicting perturbation requests were provided for rigid group "
                    f"{self._group_members_for_source_surface(source_surface_id)}."
                )
            grouped_requests[anchor_surface_id] = perturbation

            applied[int(surface_index)] = perturbation

        anchor_state = self._anchor_state_map()
        for anchor_surface_id, perturbation in grouped_requests.items():
            anchor_state[int(anchor_surface_id)] = perturbation

        anchor_state, gap_warnings = self._project_anchor_state_to_gap_constraints(
            anchor_state,
            adjustable_anchor_ids=set(int(anchor_surface_id) for anchor_surface_id in grouped_requests),
        )
        if gap_warnings:
            for message in gap_warnings:
                warnings.warn(message, MechanicalLimitWarning, stacklevel=2)

        for anchor_surface_id in sorted(grouped_requests):
            perturbation = anchor_state[int(anchor_surface_id)]
            for member_surface_id in self._group_members_for_source_surface(anchor_surface_id):
                member_seq_index = self._resolve_surface_index(member_surface_id)
                if perturbation.is_zero():
                    self._surface_perturbations.pop(member_seq_index, None)
                else:
                    self._surface_perturbations[member_seq_index] = perturbation

        self._apply_all_perturbations()
        return applied

    def clear_perturbation(self, surface_index: int | None = None) -> None:
        """Clear one perturbation or all perturbations and restore the baseline."""

        if surface_index is None:
            self._surface_perturbations.clear()
        else:
            seq_index = self._resolve_surface_index(surface_index)
            source_surface_id = self.seq_index_to_source_surface.get(seq_index, seq_index)
            for member_surface_id in self._group_members_for_source_surface(source_surface_id):
                member_seq_index = self._resolve_surface_index(member_surface_id)
                self._surface_perturbations.pop(member_seq_index, None)
        self._apply_all_perturbations()

    def get_surface_perturbations(self) -> dict[int, SurfacePerturbation]:
        """Return the current perturbation state keyed by source surface id."""

        perturbations: dict[int, SurfacePerturbation] = {}
        for seq_index, perturbation in self._surface_perturbations.items():
            source_surface_id = self.seq_index_to_source_surface.get(seq_index, seq_index)
            perturbations[int(source_surface_id)] = copy.deepcopy(perturbation)
        return perturbations

    def _sample_wavefront(
        self,
        *,
        num_rays: int,
        field_index: int,
        wavelength_nm: float,
        focus: float,
    ) -> tuple[FloatArray, FloatArray, FloatArray, NDArray[np.bool_]]:
        if num_rays < 2:
            raise ValueError("num_rays must be at least 2.")
        if field_index < 0 or field_index >= len(self.optical_spec["fov"].fields):
            raise IndexError("field_index is outside the available field list.")

        fld = self.optical_spec["fov"].fields[field_index]
        wavefront_grid = eval_wavefront(
            self.opm,
            fld,
            wavelength_nm,
            focus,
            num_rays=num_rays,
            check_apertures=False,
            value_if_none=np.nan,
        )

        pupil_x = np.asarray(wavefront_grid[:, :, 0], dtype=np.float64)
        pupil_y = np.asarray(wavefront_grid[:, :, 1], dtype=np.float64)
        opd_waves = np.asarray(wavefront_grid[:, :, 2], dtype=np.float64)

        circular_pupil = np.hypot(pupil_x, pupil_y) <= 1.0
        valid_mask = circular_pupil & np.isfinite(opd_waves)

        wavelength_sys_units = self.opm.nm_to_sys_units(wavelength_nm)
        opd_sys_units = np.zeros_like(opd_waves, dtype=np.float64)
        opd_sys_units[valid_mask] = opd_waves[valid_mask] * wavelength_sys_units

        return pupil_x, pupil_y, opd_sys_units, valid_mask

    def _wavefront_metrics_at_focus(
        self,
        *,
        focus: float,
        num_rays: int | None = None,
        field_index: int = 0,
        wavelength_nm: float | None = None,
    ) -> dict[str, float]:
        """Return wavefront RMS metrics at a specified defocus setting."""

        pupil_samples = self.pupil_samples if num_rays is None else int(num_rays)
        wavelength_nm = self.wavelength_nm if wavelength_nm is None else float(wavelength_nm)
        wavelength_sys_units = float(self.opm.nm_to_sys_units(wavelength_nm))

        pupil_x, pupil_y, opd_sys_units, valid_mask = self._sample_wavefront(
            num_rays=pupil_samples,
            field_index=field_index,
            wavelength_nm=wavelength_nm,
            focus=float(focus),
        )
        if not np.any(valid_mask):
            raise RuntimeError("Wavefront sampling returned no valid pupil points.")

        pupil_values = np.asarray(opd_sys_units, dtype=np.float64)[valid_mask]
        pupil_values = pupil_values - float(np.mean(pupil_values))

        rms_sys_units = float(np.sqrt(np.mean(pupil_values * pupil_values, dtype=np.float64)))
        rms_waves = float(rms_sys_units / max(wavelength_sys_units, 1.0e-15))
        rms_nm = float(rms_waves * wavelength_nm)

        valid_x = np.asarray(pupil_x, dtype=np.float64)[valid_mask]
        valid_y = np.asarray(pupil_y, dtype=np.float64)[valid_mask]
        radial = valid_x * valid_x + valid_y * valid_y

        low_order_basis = np.stack(
            [
                np.ones_like(pupil_values, dtype=np.float64),
                valid_x,
                valid_y,
                radial,
            ],
            axis=1,
        )
        low_order_coeffs, *_ = np.linalg.lstsq(low_order_basis, pupil_values, rcond=None)
        low_order_residual = pupil_values - low_order_basis @ low_order_coeffs
        low_order_rms_sys_units = float(
            np.sqrt(np.mean(low_order_residual * low_order_residual, dtype=np.float64))
        )
        low_order_rms_waves = float(low_order_rms_sys_units / max(wavelength_sys_units, 1.0e-15))
        low_order_rms_nm = float(low_order_rms_waves * wavelength_nm)

        return {
            "best_focus_shift_mm": float(focus),
            "wavefront_rms_sys_units": float(rms_sys_units),
            "wavefront_rms_waves": float(rms_waves),
            "wavefront_rms_nm": float(rms_nm),
            "wavefront_rms_low_order_removed_sys_units": float(low_order_rms_sys_units),
            "wavefront_rms_low_order_removed_waves": float(low_order_rms_waves),
            "wavefront_rms_low_order_removed_nm": float(low_order_rms_nm),
            "pupil_valid_fraction": float(np.mean(valid_mask.astype(np.float64))),
        }

    def find_best_focus(
        self,
        *,
        num_rays: int | None = None,
        field_index: int = 0,
        wavelength_nm: float | None = None,
        focus_center: float | None = None,
        half_range_mm: float | None = None,
        coarse_points: int = 31,
        refine_points: int = 17,
        refine_passes: int = 2,
    ) -> dict[str, float]:
        """Sweep focus around a paraxial seed and return the best low-order RMS."""

        if coarse_points < 3:
            raise ValueError("coarse_points must be at least 3.")
        if refine_points < 3:
            raise ValueError("refine_points must be at least 3.")
        if refine_passes < 1:
            raise ValueError("refine_passes must be at least 1.")

        if focus_center is None:
            try:
                focus_center = float(trace.refocus(self.opm))
            except Exception:
                focus_center = float(self.opm["optical_spec"]["focus"].focus_shift)
        if half_range_mm is None:
            half_range_mm = max(0.5, 0.25 * abs(float(focus_center)) + 0.05)

        center = float(focus_center)
        search_half_range = float(half_range_mm)
        best_metrics: dict[str, float] | None = None
        best_focus = center

        for pass_index in range(int(refine_passes)):
            num_points = int(coarse_points if pass_index == 0 else refine_points)
            candidate_focuses = np.linspace(
                center - search_half_range,
                center + search_half_range,
                num_points,
                dtype=np.float64,
            )
            for focus in candidate_focuses:
                metrics = self._wavefront_metrics_at_focus(
                    focus=float(focus),
                    num_rays=num_rays,
                    field_index=field_index,
                    wavelength_nm=wavelength_nm,
                )
                score = float(metrics["wavefront_rms_low_order_removed_waves"])
                if best_metrics is None or score < float(best_metrics["wavefront_rms_low_order_removed_waves"]):
                    best_metrics = metrics
                    best_focus = float(focus)
            center = best_focus
            search_half_range *= 0.25

        if best_metrics is None:
            raise RuntimeError("Best-focus search failed to evaluate any candidate focus.")
        best_metrics = dict(best_metrics)
        best_metrics["best_focus_shift_mm"] = float(best_focus)
        return best_metrics

    def measure_exit_ray_slopes(
        self,
        *,
        num_rays: int = 9,
        field_index: int = 0,
        wavelength_nm: float | None = None,
        focus: float = 0.0,
        pupil_radius_limit: float = 1.0,
    ) -> dict[str, float | list[dict[str, float]]]:
        """Measure the exit ray angles across the pupil.

        The slope magnitudes are the direction-cosine magnitudes of the final
        ray segment after the last physical interface.
        """

        if num_rays < 3:
            raise ValueError("num_rays must be at least 3.")
        if pupil_radius_limit <= 0.0:
            raise ValueError("pupil_radius_limit must be positive.")

        wavelength_nm = self.wavelength_nm if wavelength_nm is None else float(wavelength_nm)
        fld = self.optical_spec["fov"].fields[field_index]
        grid = np.linspace(-pupil_radius_limit, pupil_radius_limit, int(num_rays), dtype=np.float64)
        pupil_coords = [
            (float(px), float(py))
            for px in grid
            for py in grid
            if float(px * px + py * py) <= float(pupil_radius_limit * pupil_radius_limit)
        ]
        ray_list = trace_ray_list(
            self.opm,
            pupil_coords,
            fld,
            wavelength_nm,
            float(focus),
            append_if_none=True,
            output_filter="last",
            rayerr_filter="summary",
            check_apertures=False,
        )

        samples: list[dict[str, float]] = []
        slope_mags: list[float] = []
        edge_slope_mags: list[float] = []
        for pupil_x, pupil_y, ray_pkg in ray_list:
            if ray_pkg is None:
                continue
            last_seg = ray_pkg[0][-1]
            after_dir = np.asarray(last_seg[1], dtype=np.float64)
            slope_mag = float(np.hypot(after_dir[0], after_dir[1]))
            samples.append(
                {
                    "pupil_x": float(pupil_x),
                    "pupil_y": float(pupil_y),
                    "slope_x": float(after_dir[0]),
                    "slope_y": float(after_dir[1]),
                    "slope_mag": float(slope_mag),
                }
            )
            slope_mags.append(slope_mag)
            if np.isclose(np.hypot(pupil_x, pupil_y), pupil_radius_limit):
                edge_slope_mags.append(slope_mag)

        if not slope_mags:
            raise RuntimeError("No exit rays could be traced for slope measurement.")

        return {
            "focus_mm": float(focus),
            "wavelength_nm": float(wavelength_nm),
            "pupil_radius_limit": float(pupil_radius_limit),
            "num_rays": int(num_rays),
            "rms_slope_mag": float(np.sqrt(np.mean(np.square(slope_mags), dtype=np.float64))),
            "max_slope_mag": float(np.max(slope_mags)),
            "edge_rms_slope_mag": float(
                np.sqrt(np.mean(np.square(edge_slope_mags), dtype=np.float64))
                if edge_slope_mags
                else np.nan
            ),
            "mean_slope_x": float(np.mean([sample["slope_x"] for sample in samples])),
            "mean_slope_y": float(np.mean([sample["slope_y"] for sample in samples])),
            "samples": samples,
        }

    def get_wavefront_opd(
        self,
        *,
        num_rays: int | None = None,
        field_index: int = 0,
        wavelength_nm: float | None = None,
        focus: float = 0.0,
    ) -> FloatArray:
        """Return a 2D OPD map in system units with zeros outside the pupil."""

        _, _, opd_sys_units, _ = self._sample_wavefront(
            num_rays=self.pupil_samples if num_rays is None else int(num_rays),
            field_index=field_index,
            wavelength_nm=self.wavelength_nm if wavelength_nm is None else float(wavelength_nm),
            focus=float(focus),
        )
        return opd_sys_units

    def get_wavefront_opd_at_best_focus(
        self,
        *,
        num_rays: int | None = None,
        field_index: int = 0,
        wavelength_nm: float | None = None,
        half_range_mm: float | None = None,
    ) -> tuple[FloatArray, float, dict[str, float]]:
        """Return the OPD map at best focus along with the focus shift used."""

        best_focus = self.find_best_focus(
            num_rays=num_rays,
            field_index=field_index,
            wavelength_nm=wavelength_nm,
            half_range_mm=half_range_mm,
        )
        focus_shift = float(best_focus["best_focus_shift_mm"])
        opd = self.get_wavefront_opd(
            num_rays=num_rays,
            field_index=field_index,
            wavelength_nm=wavelength_nm,
            focus=focus_shift,
        )
        return opd, focus_shift, best_focus

    def get_psf_image(
        self,
        *,
        pupil_samples: int | None = None,
        fft_samples: int | None = None,
        field_index: int = 0,
        wavelength_nm: float | None = None,
        focus: float = 0.0,
    ) -> Float32Array:
        """Return a min-max normalized FFT PSF image as float32."""

        pupil_samples = self.pupil_samples if pupil_samples is None else int(pupil_samples)
        fft_samples = self.fft_samples if fft_samples is None else int(fft_samples)
        wavelength_nm = self.wavelength_nm if wavelength_nm is None else float(wavelength_nm)

        if fft_samples < pupil_samples:
            raise ValueError("fft_samples must be greater than or equal to pupil_samples.")

        _, _, opd_sys_units, valid_mask = self._sample_wavefront(
            num_rays=pupil_samples,
            field_index=field_index,
            wavelength_nm=wavelength_nm,
            focus=float(focus),
        )

        if not np.any(valid_mask):
            raise RuntimeError("Wavefront sampling returned no valid rays inside the pupil.")

        opd_for_phase = opd_sys_units.copy()
        opd_for_phase[valid_mask] -= np.mean(opd_for_phase[valid_mask])

        amplitude_mask = valid_mask.astype(np.float64)
        wavelength_sys_units = self.opm.nm_to_sys_units(wavelength_nm)
        complex_pupil = amplitude_mask * np.exp(
            1j * 2.0 * np.pi * opd_for_phase / wavelength_sys_units
        )

        pad_total = fft_samples - complex_pupil.shape[0]
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        padded_pupil = np.pad(
            complex_pupil,
            ((pad_before, pad_after), (pad_before, pad_after)),
            mode="constant",
            constant_values=0.0,
        )

        intensity = np.abs(fftshift(fft2(padded_pupil))) ** 2
        intensity = np.asarray(intensity, dtype=np.float64)

        intensity_min = float(np.min(intensity))
        intensity_max = float(np.max(intensity))
        if np.isclose(intensity_max, intensity_min):
            return np.zeros_like(intensity, dtype=np.float32)

        normalized = (intensity - intensity_min) / (intensity_max - intensity_min)
        return normalized.astype(np.float32, copy=False)

    def get_psf_image_at_best_focus(
        self,
        *,
        pupil_samples: int | None = None,
        fft_samples: int | None = None,
        field_index: int = 0,
        wavelength_nm: float | None = None,
        half_range_mm: float | None = None,
    ) -> tuple[Float32Array, float, dict[str, float]]:
        """Return a normalized PSF at the best focus, plus the focus shift used."""

        best_focus = self.find_best_focus(
            num_rays=pupil_samples,
            field_index=field_index,
            wavelength_nm=wavelength_nm,
            half_range_mm=half_range_mm,
        )
        focus_shift = float(best_focus["best_focus_shift_mm"])
        psf = self.get_psf_image(
            pupil_samples=pupil_samples,
            fft_samples=fft_samples,
            field_index=field_index,
            wavelength_nm=wavelength_nm,
            focus=focus_shift,
        )
        return psf, focus_shift, best_focus


__all__ = [
    "AXIAL_CLEARANCE_MARGIN_MM",
    "DEFAULT_2P_AO_PUPIL_DIAMETER_MM",
    "DEFAULT_DESIGN_NAME",
    "DEFAULT_FFT_SAMPLES",
    "DEFAULT_MOCK_PUPIL_DIAMETER_MM",
    "DEFAULT_PUPIL_SAMPLES",
    "DEFAULT_WAVELENGTH_NM",
    "MechanicalLimitWarning",
    "MechanicalEnvelope",
    "RayOpticsPhysicsEngine",
    "SHIFT_LIMIT_MM",
    "SurfacePerturbation",
    "TILT_LIMIT_DEG",
    "TWO_PHOTON_AO_PRESCRIPTION",
    "TWO_PHOTON_AO_RIGID_GROUPS",
]
