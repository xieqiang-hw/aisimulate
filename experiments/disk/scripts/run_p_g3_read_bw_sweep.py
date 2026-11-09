"""Run the four P-local SSD read bandwidth experiments sequentially."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

DRAM_BLOCKS = (4096, 8192)
READ_BANDWIDTHS = (2.5, 5.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-1p1d-p-g3-read-bw")
    parser.add_argument("--max-requests", type=int, default=0, help="0 replays the full trace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dram-blocks", nargs="+", type=int, choices=DRAM_BLOCKS,
                        default=DRAM_BLOCKS, help="Select cases, for example to retry a failed case")
    parser.add_argument("--read-gbps", nargs="+", type=float, choices=READ_BANDWIDTHS,
                        default=READ_BANDWIDTHS)
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    print(f"Read bandwidth sweep directory: {output}", flush=True)
    cases = ((blocks, bw) for blocks in dict.fromkeys(args.dram_blocks)
             for bw in dict.fromkeys(args.read_gbps))
    for blocks, bw in cases:
        tag = format(bw, "g").replace(".", "p")
        name = f"p_g3_local_dram_{blocks}_read_{tag}"
        print(f"Running {name}: {blocks:,} DRAM blocks, {bw:g} GB/s SSD read", flush=True)
        command = [sys.executable, str(BASE / "scripts/run_p_g3_local.py"),
                   "--config", str(BASE / f"configs/{name}.yaml"),
                   "--bundle", str(args.bundle),
                   "--output-root", str(output / name),
                   "--max-requests", str(args.max_requests)]
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"{name} failed. See {output / name}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
