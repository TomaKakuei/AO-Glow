import argparse
import json

import numpy as np

import benchmark_freeform_real_shwfs_residual_120 as real_shwfs
from ao_v3_backend import RunConfig, run_repair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overall-variation-mm", type=float, default=0.1)
    parser.add_argument("--output-name", type=str, default="v3_noseed_case")
    args = parser.parse_args()

    original = real_shwfs.ShwfsMeasurementModel.estimate_nominal_correction_seed

    def zero_seed(self: real_shwfs.ShwfsMeasurementModel, optical_model) -> np.ndarray:
        return np.zeros(3 * len(self.actuator_ids), dtype=np.float64)

    real_shwfs.ShwfsMeasurementModel.estimate_nominal_correction_seed = zero_seed
    try:
        result = run_repair(
            RunConfig(
                overall_variation_mm=float(args.overall_variation_mm),
                output_name=str(args.output_name),
            )
        )
    finally:
        real_shwfs.ShwfsMeasurementModel.estimate_nominal_correction_seed = original

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
