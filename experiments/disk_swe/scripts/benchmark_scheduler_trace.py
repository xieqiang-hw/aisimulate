"""Measure trace overhead on an identical real request prefix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path.home() / "project/data/swe_chat/derived/mooncake-v1/claude-code-main-glm52-128k-overlay-2x"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/aisim-observation-benchmark"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=3000)
    args = parser.parse_args()
    runner = BASE.parent / "disk/scripts/run_experiment.py"
    common = [sys.executable, "-u", str(runner), "--config",
              str(BASE / "configs/swe_p_local_8ssd_p128gb_1x_hbm2048.yaml"),
              "--bundle", str(DEFAULT_BUNDLE), "--max-requests", str(args.max_requests)]
    for mode, options in [
        ("v2-gzip", ["--capture-scheduler-trace", "--scheduler-trace-compression", "gzip"]),
        ("v2-none", ["--capture-scheduler-trace", "--scheduler-trace-compression", "none"]),
        ("off-v2", []),
    ]:
        for _ in range(args.repeats):
            subprocess.run(common + options + ["--output-root", str(args.output_root / mode)], check=True)


if __name__ == "__main__":
    main()
