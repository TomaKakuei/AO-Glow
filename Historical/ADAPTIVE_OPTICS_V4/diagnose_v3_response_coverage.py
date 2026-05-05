import json
from typing import Any

import numpy as np

from ao_v3_backend import (
    ACTUATOR_IDS,
    SEARCH_BOX_MIN_MM,
    SEARCH_BOX_SCALE,
    bench,
    real_shwfs,
)


def main() -> None:
    model = bench._build_model(pupil_samples=bench.FAST_PUPIL_SAMPLES, fft_samples=bench.FAST_FFT_SAMPLES)
    shwfs = real_shwfs._build_shwfs_measurement_model(
        model,
        focus_shift_mm=bench.REFERENCE_FOCUS_MM,
        actuator_ids=ACTUATOR_IDS,
    )

    response = np.asarray(shwfs.mechanical_response_matrix, dtype=np.float64)
    singular_values = np.linalg.svd(response, compute_uv=False)
    threshold = float(real_shwfs.MECHANICAL_RESPONSE_RCOND) * float(singular_values[0])
    effective_rank = int(np.count_nonzero(singular_values > threshold))
    column_norms = np.linalg.norm(response, axis=0)
    max_norm = float(np.max(column_norms)) if column_norms.size else 1.0

    rows: list[dict[str, Any]] = []
    weak_axes: list[dict[str, Any]] = []
    for idx, anchor_surface_id in enumerate(ACTUATOR_IDS):
        triplet = column_norms[3 * idx : 3 * idx + 3]
        row = {
            "anchor_surface_id": int(anchor_surface_id),
            "dx_norm": float(triplet[0]),
            "dy_norm": float(triplet[1]),
            "dz_norm": float(triplet[2]),
            "dx_relative_to_max": float(triplet[0] / max(max_norm, 1.0e-15)),
            "dy_relative_to_max": float(triplet[1] / max(max_norm, 1.0e-15)),
            "dz_relative_to_max": float(triplet[2] / max(max_norm, 1.0e-15)),
        }
        rows.append(row)
        for axis_name, value in zip(("dx", "dy", "dz"), triplet):
            if float(value) / max(max_norm, 1.0e-15) < 1.0e-3:
                weak_axes.append(
                    {
                        "anchor_surface_id": int(anchor_surface_id),
                        "axis": axis_name,
                        "relative_norm": float(value / max(max_norm, 1.0e-15)),
                    }
                )

    summary = {
        "shwfs_lenslet_count": int(shwfs.lenslet_count),
        "slope_channels": int(shwfs.reference_slopes.size),
        "slope_limit": int(real_shwfs.SHWFS_SLOPE_LIMIT),
        "mechanical_steps_mm": [float(v) for v in shwfs.mechanical_steps_mm],
        "mechanical_response_rcond": float(real_shwfs.MECHANICAL_RESPONSE_RCOND),
        "response_shape": [int(v) for v in response.shape],
        "matrix_rank": int(np.linalg.matrix_rank(response)),
        "effective_rank_at_rcond": effective_rank,
        "sigma_max": float(singular_values[0]),
        "sigma_min": float(singular_values[-1]),
        "condition_number": float(singular_values[0] / max(float(singular_values[-1]), 1.0e-15)),
        "search_box_scale": float(SEARCH_BOX_SCALE),
        "search_box_min_mm": float(SEARCH_BOX_MIN_MM),
        "per_anchor_axis_norms": rows,
        "weak_axes_below_1e3_of_max": weak_axes,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
