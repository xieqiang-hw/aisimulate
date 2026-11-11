"""Sweep prefill HBM KV capacity at fixed 384 GB DRAM and 1x."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

HBM_BLOCKS = (0, 16384, 12288, 8192)
BASELINE = "p_local_8ssd_p384gb_1x_ep8_384k"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-ep8-384k-hbm-sweep")
    parser.add_argument("--max-requests", type=int, default=0, help="0 replays the full trace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hbm-blocks", nargs="+", type=int, choices=HBM_BLOCKS,
                        default=HBM_BLOCKS, help="0 uses the automatic baseline")
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    print(f"Local SSD sweep directory: {output}", flush=True)
    for blocks in dict.fromkeys(args.hbm_blocks):
        name = BASELINE if blocks == 0 else f"{BASELINE}_hbm{blocks}"
        print(f"Running {name}: P HBM KV blocks {blocks or 'default'}", flush=True)
        command = [sys.executable, str(BASE / "scripts/run_p_g3_local.py"),
                   "--config", str(BASE / f"configs/{name}.yaml"),
                   "--bundle", str(args.bundle),
                   "--output-root", str(output / name),
                   "--max-requests", str(args.max_requests), "--speedup", "1"]
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"{name} failed. See {output / name}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
