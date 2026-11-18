"""Sweep P-side DRAM capacity and SSD read bandwidth on H100 NVL."""
import argparse
from itertools import product
import math
from pathlib import Path

import yaml

from run_experiment_suite import BASE, BUNDLE, run_cases, validate_bundle

FACTORS = (8, 16, 32, 64)
DRAM_GB = (512, 384, 256, 128, 64, 32, 16)
READ_GBPS = (2, 4, 8)
# GLM-5.2 FP8 DSA, 64 tokens per KV block, eight Attention shards.
BLOCK_BYTES = 60_216 * 64 * 8
TEMPLATE = BASE / "configs/swe_h100_nvl_local_8ssd.yaml"
DEFAULT_SYSTEMS = BASE.parents[1] / "python/aisimulate/src/aisimulate_core/systems"
OUTPUT = Path.home() / "project/results/swe-chat-glm52-h100-nvl-local-8ssd"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--systems-root", type=Path, default=DEFAULT_SYSTEMS,
                        help="Directory containing h100_nvl.yaml and its performance data")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--factors", nargs="+", type=int, choices=FACTORS, default=FACTORS)
    parser.add_argument("--p-dram-gb", nargs="+", type=int, choices=DRAM_GB, default=DRAM_GB)
    parser.add_argument("--read-bandwidths", nargs="+", type=int, choices=READ_GBPS,
                        default=READ_GBPS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--memory-limit-gb", type=float)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--capture-scheduler-trace", action="store_true")
    parser.add_argument("--scheduler-trace-compression", choices=("gzip", "none"), default="gzip")
    args = parser.parse_args()
    if args.jobs < 1 or args.max_requests < 0:
        parser.error("--jobs must be positive and --max-requests nonnegative")
    if args.memory_limit_gb is not None and (
        not math.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0
    ):
        parser.error("--memory-limit-gb must be finite and positive")

    parent = args.bundle.expanduser().resolve()
    matrix = list(product(dict.fromkeys(args.factors), dict.fromkeys(args.p_dram_gb),
                          dict.fromkeys(args.read_bandwidths)))
    selected = []
    for factor, gb, read in matrix:
        name = f"swe_h100_nvl_p{gb}gb_read{read}_overlay_{factor}x"
        bundle = parent.with_name(f"{parent.name}-overlay-{factor}x")
        selected.append(("bandwidth", name, name, bundle))
    if args.plan:
        for _, name, _, bundle in selected:
            print(f"{name} -> {bundle}")
        print(f"Total: {len(selected)} cases")
        return 0

    systems_root = args.systems_root.expanduser().resolve()
    profile = systems_root / "h100_nvl.yaml"
    if not profile.is_file():
        parser.error(f"H100 NVL profile missing: {profile}. Supply --systems-root.")
    profile_data = yaml.safe_load(profile.read_text())
    data_root = systems_root / profile_data.get("data_dir", "data/h100_nvl")
    if not data_root.is_dir() or not any(data_root.rglob("*.parquet")):
        parser.error(f"H100 NVL performance data missing: {data_root}")

    for bundle in dict.fromkeys(item[3] for item in selected):
        try:
            count = validate_bundle(bundle, parent)
        except (OSError, ValueError, KeyError, StopIteration) as error:
            parser.error(f"Invalid overlay bundle {bundle}: {error}")
        print(f"Bundle ready: {bundle.name}, {count:,} requests", flush=True)

    settings = {}
    for (_, gb, read), (_, name, _, _) in zip(matrix, selected, strict=True):
        cfg = yaml.safe_load(TEMPLATE.read_text())
        cfg["engine"]["systems_paths"] = [str(systems_root)]
        kv = cfg["engine"]["workers"]["prefill"]["kv_cache"]
        kv["host_offload"]["num_host_blocks"] = round(gb * 10**9 / BLOCK_BYTES)
        kv["g3_offload"]["num_g3_blocks"] = round(56 * 10**12 / BLOCK_BYTES)
        kv["g3_offload"]["read_bandwidth_gbps"] = float(read)
        if args.memory_limit_gb is not None:
            cfg.setdefault("execution", {}).setdefault("resources", {})[
                "memory_limit_gb"] = args.memory_limit_gb
        settings[name] = cfg
    return run_cases(args, parent, selected, settings)


if __name__ == "__main__":
    raise SystemExit(main())
