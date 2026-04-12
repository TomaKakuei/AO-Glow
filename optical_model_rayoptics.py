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
from rayoptics.elem.surface import DecenterData
from rayoptics.optical.opticalmodel import OpticalModel
from rayoptics.raytr.analyses import eval_wavefront
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
SHIFT_LIMIT_MM = 1.0
TILT_LIMIT_DEG = 2.0

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
    ) -> None:
        if wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm must be positive.")
        if pupil_samples < 2:
            raise ValueError("pupil_samples must be at least 2.")
        if fft_samples < pupil_samples:
            raise ValueError("fft_samples must be greater than or equal to pupil_samples.")

        self.design_name = str(design_name)
        self.prescription = self._select_prescription(self.design_name, prescription)
        self.wavelength_nm = float(wavelength_nm)
        self.pupil_diameter_mm = self._default_pupil_diameter(
            self.design_name, pupil_diameter_mm
        )
        self.pupil_samples = int(pupil_samples)
        self.fft_samples = int(fft_samples)

        self.source_surface_to_seq_index: dict[int, int] = {}
        self.seq_index_to_source_surface: dict[int, int] = {}
        self.surface_descriptions: dict[int, str] = {}

        self.opm = OpticalModel(radius_mode=True)
        self.opm.system_spec.title = self.design_name
        self.seq_model = self.opm["seq_model"]
        self.optical_spec = self.opm["optical_spec"]

        self._configure_optical_spec()
        self._build_active_design()
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

    def _build_prescription_model(self, prescription: list[PrescriptionRow]) -> None:
        if len(prescription) < 2:
            raise ValueError("Prescription must contain an object row and at least one surface.")

        self.source_surface_to_seq_index.clear()
        self.seq_index_to_source_surface.clear()
        self.surface_descriptions.clear()

        object_row = prescription[0]
        self.seq_model.gaps[0].thi = float(object_row["thickness"])
        self._tag_surface_metadata(
            seq_index=0,
            source_surface_id=int(object_row["surf"]),
            description=str(object_row["desc"] or "Obj"),
        )

        for row in prescription[1:]:
            self._append_prescription_surface(row)

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
        dx: float,
        dy: float,
        dz: float,
        tilt_x: float,
        tilt_y: float,
    ) -> SurfacePerturbation:
        requested = {
            "dx_mm": float(dx),
            "dy_mm": float(dy),
            "dz_mm": float(dz),
            "tilt_x_deg": float(tilt_x),
            "tilt_y_deg": float(tilt_y),
        }
        limits = {
            "dx_mm": SHIFT_LIMIT_MM,
            "dy_mm": SHIFT_LIMIT_MM,
            "dz_mm": SHIFT_LIMIT_MM,
            "tilt_x_deg": TILT_LIMIT_DEG,
            "tilt_y_deg": TILT_LIMIT_DEG,
        }

        clamped: dict[str, float] = {}
        exceeded: list[str] = []
        for key, value in requested.items():
            limit = limits[key]
            clamped_value = float(np.clip(value, -limit, limit))
            clamped[key] = clamped_value
            if not np.isclose(clamped_value, value):
                exceeded.append(f"{key}={value:.6f} -> {clamped_value:.6f}")

        if exceeded:
            warnings.warn(
                "Perturbation exceeded mechanical limits and was clamped: "
                + ", ".join(exceeded),
                MechanicalLimitWarning,
                stacklevel=2,
            )

        return SurfacePerturbation(**clamped)

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
        value: SurfacePerturbation | Sequence[float] | dict[str, float],
    ) -> SurfacePerturbation:
        if isinstance(value, SurfacePerturbation):
            return self._clamp_perturbation(
                value.dx_mm,
                value.dy_mm,
                value.dz_mm,
                value.tilt_x_deg,
                value.tilt_y_deg,
            )

        if isinstance(value, dict):
            return self._clamp_perturbation(
                float(value.get("dx", value.get("dx_mm", 0.0))),
                float(value.get("dy", value.get("dy_mm", 0.0))),
                float(value.get("dz", value.get("dz_mm", 0.0))),
                float(value.get("tilt_x", value.get("tilt_x_deg", 0.0))),
                float(value.get("tilt_y", value.get("tilt_y_deg", 0.0))),
            )

        components = np.asarray(value, dtype=np.float64).reshape(-1)
        if components.size == 3:
            return self._clamp_perturbation(
                float(components[0]),
                float(components[1]),
                float(components[2]),
                0.0,
                0.0,
            )
        if components.size == 5:
            return self._clamp_perturbation(
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
        perturbation = self._clamp_perturbation(dx, dy, dz, tilt_x, tilt_y)

        if perturbation.is_zero():
            self._surface_perturbations.pop(seq_index, None)
        else:
            self._surface_perturbations[seq_index] = perturbation

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
        for surface_index, requested_value in perturbations.items():
            seq_index = self._resolve_surface_index(surface_index)
            self._validate_surface_index(seq_index)
            perturbation = self._coerce_surface_perturbation(requested_value)

            if perturbation.is_zero():
                self._surface_perturbations.pop(seq_index, None)
            else:
                self._surface_perturbations[seq_index] = perturbation

            applied[int(surface_index)] = perturbation

        self._apply_all_perturbations()
        return applied

    def clear_perturbation(self, surface_index: int | None = None) -> None:
        """Clear one perturbation or all perturbations and restore the baseline."""

        if surface_index is None:
            self._surface_perturbations.clear()
        else:
            seq_index = self._resolve_surface_index(surface_index)
            self._surface_perturbations.pop(seq_index, None)
        self._apply_all_perturbations()

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


__all__ = [
    "DEFAULT_2P_AO_PUPIL_DIAMETER_MM",
    "DEFAULT_DESIGN_NAME",
    "DEFAULT_FFT_SAMPLES",
    "DEFAULT_MOCK_PUPIL_DIAMETER_MM",
    "DEFAULT_PUPIL_SAMPLES",
    "DEFAULT_WAVELENGTH_NM",
    "MechanicalLimitWarning",
    "RayOpticsPhysicsEngine",
    "SHIFT_LIMIT_MM",
    "SurfacePerturbation",
    "TILT_LIMIT_DEG",
    "TWO_PHOTON_AO_PRESCRIPTION",
]
