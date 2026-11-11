"""Run the nine local-SSD and independent P/D DRAM experiments sequentially."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

P_DRAM_GB = (384, 256, 128)
SPEEDUPS = (1, 2, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/minimax-m25-h100-local-8ssd")
    parser.add_argument("--max-requests", type=int, default=0, help="0 replays the full trace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--p-dram-gb", nargs="+", type=int, choices=P_DRAM_GB,
                        default=P_DRAM_GB, help="Select cases, for example to retry a failed case")
    parser.add_argument("--speedups", nargs="+", type=int, choices=SPEEDUPS,
                        default=SPEEDUPS)
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    print(f"Local SSD sweep directory: {output}", flush=True)
    cases = ((gb, speed) for gb in dict.fromkeys(args.p_dram_gb)
             for speed in dict.fromkeys(args.speedups))
    for gb, speed in cases:
        name = f"h100_p_local_8ssd_p{gb}gb_{speed}x"
        print(f"Running {name}: P DRAM {gb} GB, {speed}x arrivals", flush=True)
        command = [sys.executable, str(BASE / "scripts/run_p_g3_local.py"),
                   "--config", str(BASE / f"configs/{name}.yaml"),
                   "--bundle", str(args.bundle),
                   "--output-root", str(output / name),
                   "--max-requests", str(args.max_requests), "--speedup", str(speed)]
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"{name} failed. See {output / name}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
