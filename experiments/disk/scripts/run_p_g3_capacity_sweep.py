"""Run the four P-local SSD capacity experiments sequentially."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

MULTIPLIERS = (4, 16, 64, 128)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-1p1d-p-g3-capacity")
    parser.add_argument("--max-requests", type=int, default=0, help="0 replays the full trace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--multipliers", nargs="+", type=int, choices=MULTIPLIERS,
                        default=MULTIPLIERS, help="Select cases, for example to retry a failed case")
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    print(f"Capacity sweep directory: {output}", flush=True)
    for multiplier in dict.fromkeys(args.multipliers):
        name = f"p_g3_local_{multiplier}x"
        print(f"Running {name}: {16384 * multiplier:,} SSD blocks", flush=True)
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
