"""Replay the 16-week 128K trace at 1x with three P-side HBM capacities."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

HBM_BLOCKS = (3072, 4096, 8192)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-128k-1x-hbm-sweep")
    parser.add_argument("--max-requests", type=int, default=0)
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    print(f"HBM sweep directory: {output}", flush=True)
    failed = []
    for blocks in HBM_BLOCKS:
        name = f"p_local_8ssd_p384gb_1x_hbm{blocks}"
        print(f"Running {name}: {blocks:,} P HBM KV blocks", flush=True)
        result = subprocess.run([
            sys.executable, str(BASE / "scripts/run_experiment.py"),
            "--config", str(BASE / f"configs/{name}.yaml"),
            "--bundle", str(args.bundle),
            "--output-root", str(output / name),
            "--speedup", "1", "--max-requests", str(args.max_requests),
        ], check=False)
        if result.returncode:
            failed.append(name)
    print(f"Completed {len(HBM_BLOCKS) - len(failed)}/{len(HBM_BLOCKS)} cases", flush=True)
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr, flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
