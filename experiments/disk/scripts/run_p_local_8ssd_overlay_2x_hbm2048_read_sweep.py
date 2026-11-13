"""Scan SSD read bandwidth for the 2x overlay with 128 GB DRAM and 2048 HBM blocks."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

PROJECT = Path.home() / "project"
BUNDLE = PROJECT / ("data/freeinference_agentic_trace/derived/dynosim-openloop-v1/"
                    "428f809eef5952012f9c-overlay-2x")
OUTPUT = PROJECT / "results/glm52-overlay-2x-hbm2048-p128gb-ssd-read-sweep"
READ_BANDWIDTHS = (12, 16, 24, 32, 40, 48, 56)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--read-gbps", type=int, nargs="+", choices=READ_BANDWIDTHS,
                        default=READ_BANDWIDTHS)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    if not (BUNDLE / "manifest.json").is_file():
        parser.error(f"missing 2x overlay bundle: {BUNDLE}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    failures = []
    bandwidths = dict.fromkeys(args.read_gbps)
    print(f"Sweep directory: {output}", flush=True)
    for bandwidth in bandwidths:
        name = f"p_local_8ssd_p128gb_overlay_2x_hbm2048_read{bandwidth}"
        print(f"Running {name}: SSD read {bandwidth} GB/s per TP shard", flush=True)
        command = [
            sys.executable, str(BASE / "scripts/run_experiment.py"),
            "--config", str(BASE / f"configs/{name}.yaml"),
            "--bundle", str(BUNDLE), "--output-root", str(output / name),
            "--speedup", "1", "--max-requests", str(args.max_requests),
        ]
        if args.dry_run:
            command.append("--dry-run")
        if subprocess.run(command, check=False).returncode:
            failures.append(name)
            print(f"Failed: {name}. See {output / name}", file=sys.stderr, flush=True)

    print(f"Succeeded: {len(bandwidths) - len(failures)}/{len(bandwidths)}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
