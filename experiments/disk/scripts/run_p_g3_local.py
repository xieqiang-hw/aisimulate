"""Run the full FreeInference replay with P-local G3 and HBM-only decode."""
from pathlib import Path

from run_experiment import BASE, main


if __name__ == "__main__":
    raise SystemExit(main(
        default_config=BASE / "configs/p_g3_local.yaml",
        default_output_root=Path.home() / "project/results/glm52-1p1d-p-g3-local",
    ))
