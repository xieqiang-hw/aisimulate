"""Run nine local SSD cases with overlaid FreeInference arrivals."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_experiment import BASE

P_DRAM_GB = (384, 256, 128)
OVERLAYS = (2, 4, 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Original 128K bundle; overlays are its siblings")
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-1p1d-local-8ssd-overlay")
    parser.add_argument("--max-requests", type=int, default=0, help="0 replays the full trace")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--p-dram-gb", nargs="+", type=int, choices=P_DRAM_GB,
                        default=P_DRAM_GB, help="Select cases, for example to retry a failed case")
    parser.add_argument("--overlays", nargs="+", type=int, choices=(*OVERLAYS, 16),
                        default=OVERLAYS)
    parser.add_argument('--p-hbm-blocks', type=int, choices=(2048,))
    args = parser.parse_args()
    if args.p_hbm_blocks and args.overlays != [2]:
        parser.error('--p-hbm-blocks 2048 requires --overlays 2')
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    if args.p_hbm_blocks and args.output_root == parser.get_default('output_root'):
        args.output_root = args.output_root.with_name('glm52-overlay-2x-hbm2048')
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    print(f"Local SSD sweep directory: {output}", flush=True)
    parent = args.bundle.expanduser().resolve()
    bundles = {}
    for factor in dict.fromkeys(args.overlays):
        bundle = parent.with_name(f"{parent.name}-overlay-{factor}x")
        manifest_path = bundle / "manifest.json"
        if not manifest_path.is_file():
            parser.error(f"Missing overlay manifest: {manifest_path}")
        overlay = json.loads(manifest_path.read_text())["overlay"]
        if (overlay["factor"], overlay["replay_speedup"], overlay["parent_bundle"]) != (factor, 1, parent.name):
            parser.error(f"Unexpected overlay metadata: {manifest_path}")
        bundles[factor] = bundle

    failures = []
    for factor, gb in ((factor, gb) for factor in bundles for gb in dict.fromkeys(args.p_dram_gb)):
        name = f"p_local_8ssd_p{gb}gb_overlay_{factor}x"
        config = BASE / f"configs/p_local_8ssd_p{gb}gb_1x.yaml"
        if args.p_hbm_blocks:
            name += f'_hbm{args.p_hbm_blocks}'
            config = BASE / f'configs/{name}.yaml'
        print(f"Running {name}: P DRAM {gb} GB, {factor}x overlay, engine speedup 1", flush=True)
        command = [sys.executable, str(BASE / "scripts/run_p_g3_local.py"),
                   "--config", str(config),
                   "--bundle", str(bundles[factor]),
                   "--output-root", str(output / name),
                   "--max-requests", str(args.max_requests), "--speedup", "1"]
        if args.dry_run:
            command.append("--dry-run")
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"{name} failed. See {output / name}", file=sys.stderr)
            if not args.p_hbm_blocks:
                return result.returncode
            failures.append(name)
    if failures:
        print('Failed cases: ' + ', '.join(failures), file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
