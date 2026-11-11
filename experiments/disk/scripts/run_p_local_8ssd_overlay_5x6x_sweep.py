"""Run the selected 5x and 6x overlay bandwidth cases serially."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml
from aisimulate.config.cli import CorePredictionConfig
from run_experiment import BASE

BASELINES = [(factor, gb, 8, 8) for factor in (5, 6) for gb in (128, 256, 384)]
PRIORITY = [(6, 128, read, 8) for read in (12, 16, 24)] + [(5, 128, 16, 8)]
REMAINING = [(factor, gb, 24, 8) for factor, gb, _, _ in BASELINES
             if (factor, gb) != (6, 128)]
CASES = BASELINES + PRIORITY + REMAINING


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path, help="Original 128K bundle")
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-overlay-5x-6x-bandwidth")
    parser.add_argument("--list", action="store_true", help="Print cases without starting replay")
    args = parser.parse_args()
    if args.list:
        for case in CASES:
            print(*case)
        return
    parent = args.bundle.expanduser().resolve()
    bundles = {}
    for factor in (5, 6):
        bundle = parent.with_name(f"{parent.name}-overlay-{factor}x")
        overlay = json.loads((bundle / "manifest.json").read_text())["overlay"]
        if (overlay["factor"], overlay["replay_speedup"], overlay["parent_bundle"]) != (factor, 1, parent.name):
            parser.error(f"Unexpected overlay metadata in {bundle}")
        bundles[factor] = bundle

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    configs = output / "configs"
    configs.mkdir(parents=True)
    print(f"Sweep directory: {output}", flush=True)
    planned = []
    for factor, gb, read, write in CASES:
        name = f"p{gb}gb_overlay_{factor}x_read{read}_write{write}"
        config = configs / f"{name}.yaml"
        template = BASE / f"configs/p_local_8ssd_p{gb}gb_1x.yaml"
        settings = yaml.safe_load(template.read_text())
        g3 = settings["engine"]["workers"]["prefill"]["kv_cache"]["g3_offload"]
        g3.update(read_bandwidth_gbps=read, write_bandwidth_gbps=write)
        settings["execution"]["resources"]["memory_limit_gb"] = 21.0
        config.write_text(yaml.safe_dump(settings, sort_keys=False))
        CorePredictionConfig.from_yaml(str(config))
        planned.append((factor, name, config))

    failures = []
    for index, (factor, name, config) in enumerate(planned, 1):
        print(f"[{index}/{len(planned)}] Running {name}", flush=True)
        result = subprocess.run([sys.executable, str(BASE / "scripts/run_p_g3_local.py"),
                        "--config", str(config), "--bundle", str(bundles[factor]),
                        "--output-root", str(output / name), "--speedup", "1"], check=False)
        if result.returncode:
            failures.append(name)
            print(f"Failed {name} (exit {result.returncode}); continuing. See {output / name}", flush=True)

    print(f"Sweep finished: {len(planned) - len(failures)}/{len(planned)} succeeded", flush=True)
    if failures:
        print("Failed cases: " + ", ".join(failures), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
