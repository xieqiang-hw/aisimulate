"""Measure trace overhead on an identical real request prefix."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/aisim-observation-benchmark"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-requests", type=int, default=3000)
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration to benchmark")
    parser.add_argument("--bundle", type=Path, required=True, help="Mooncake bundle to benchmark")
    args = parser.parse_args()
    runner = BASE.parent / "disk/scripts/run_experiment.py"
    common = [sys.executable, "-u", str(runner), "--config",
              str(args.config.expanduser().resolve()),
              "--bundle", str(args.bundle.expanduser().resolve()), "--max-requests", str(args.max_requests)]
    for mode, options in [
        ("v2-gzip", ["--capture-scheduler-trace", "--scheduler-trace-compression", "gzip"]),
        ("v2-none", ["--capture-scheduler-trace", "--scheduler-trace-compression", "none"]),
        ("off-v2", []),
    ]:
        for _ in range(args.repeats):
            subprocess.run(common + options + ["--output-root", str(args.output_root / mode)], check=True)


if __name__ == "__main__":
    main()
