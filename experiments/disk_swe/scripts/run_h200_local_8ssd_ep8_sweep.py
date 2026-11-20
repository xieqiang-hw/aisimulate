"""Run the verified SWE-chat H200 TP8/EP8 local SSD sweep serially."""
import argparse
import json
import math
import re
from itertools import product
from pathlib import Path

import yaml

from run_experiment_suite import BASE, run_cases, validate_bundle

PARENT = Path.home() / (
    "project/data/swe_chat/derived/verified_interleave_mooncake_v1/"
    "claude-code-main-glm52-128k"
)
OUTPUT = Path.home() / "project/results/swe-chat-glm52-h200-ep8-16x"
TEMPLATE = BASE / "configs/swe_h200_local_8ssd_ep8.yaml"
DRAM_GB = (384, 256, 128, 64)
EXTENSION_DRAM_GB = (512, *DRAM_GB)
READ_GBPS = (2, 4, 8)
BLOCK_BYTES = 60_216 * 64 * 8  # GLM-5.2 FP8 DSA, entire TP8 instance.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=PARENT.with_name(
        f"{PARENT.name}-overlay-16x"), help="Verified SWE-chat overlay bundle")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--p-dram-gb", nargs="+", type=int, choices=(*EXTENSION_DRAM_GB, 1000))
    parser.add_argument("--read-bandwidths", nargs="+", type=int, choices=(1, *READ_GBPS, 16))
    parser.add_argument("--memory-limit-gb", type=float, default=None,
                        help="Process-tree memory budget in decimal GB; default is automatic")
    parser.add_argument("--reserve-memory-gb", type=float, default=None,
                        help="Override host memory reserve while keeping the experiment memory limit")
    parser.add_argument("--max-requests", type=int, default=0, help="0 replays the full trace")
    parser.add_argument("--extension", action="store_true",
                        help="Scan both 16x and 32x at five DRAM capacities")
    parser.add_argument("--supplement-8x", action="store_true",
                        help="Add read16 at existing capacities and 1000 GB at read1/2/4/8/16")
    parser.add_argument("--skip-completed-from", action="append", type=Path, default=[],
                        help="Skip completed cases listed in a previous sweep.json")
    parser.add_argument("--plan", action="store_true", help="Print configurations without replay")
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    args.jobs = 1
    if args.memory_limit_gb is not None and (not math.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0):
        parser.error("--memory-limit-gb must be finite and positive")
    if args.reserve_memory_gb is not None and (not math.isfinite(args.reserve_memory_gb) or args.reserve_memory_gb < 0):
        parser.error("--reserve-memory-gb must be finite and nonnegative")
    args.capture_scheduler_trace = False
    args.scheduler_trace_compression = "gzip"

    bundle = args.bundle.expanduser().resolve()
    overlay = re.fullmatch(r"(.+)-overlay-(\d+)x", bundle.name)
    if not overlay or int(overlay[2]) < 2:
        parser.error("--bundle must name an overlay bundle")
    parent = bundle.with_name(overlay[1])
    selected = []
    settings = {}
    capacities = args.p_dram_gb or (EXTENSION_DRAM_GB if args.extension else DRAM_GB)
    matrix = {factor: dict.fromkeys(capacities) for factor in ((16, 32) if args.extension else (int(overlay[2]),))}
    completed = set()
    for previous in args.skip_completed_from:
        plan = json.loads(previous.expanduser().read_text(encoding="utf-8"))
        if Path(plan["bundle"]).resolve() != parent or plan["max_requests"] != args.max_requests:
            parser.error(f"Previous sweep uses a different bundle or request limit: {previous}")
        completed.update((case["name"], Path(case["bundle"]).resolve()) for case in plan["cases"]
                         if case["status"] == "completed")
    bandwidths = args.read_bandwidths or ((1, *READ_GBPS) if args.extension else READ_GBPS)
    combinations = ((factor, gb, read) for factor, capacities in matrix.items()
                    for gb, read in product(capacities, dict.fromkeys(bandwidths)))
    if args.supplement_8x:
        if args.extension or args.p_dram_gb or args.read_bandwidths or int(overlay[2]) != 8:
            parser.error("--supplement-8x requires an 8x bundle and no other matrix selectors")
        combinations = [(8, gb, 16) for gb in EXTENSION_DRAM_GB] + [
            (8, 1000, read) for read in (1, 2, 4, 8, 16)
        ]
    for factor, gb, read in combinations:
        trace = parent.with_name(f"{parent.name}-overlay-{factor}x")
        name = f"swe_h200_ep8_p{gb}gb_read{read}_overlay_{factor}x"
        if (name, trace) in completed:
            continue
        config = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
        if args.memory_limit_gb is not None:
            config.setdefault("execution", {}).setdefault("resources", {})["memory_limit_gb"] = args.memory_limit_gb
        if args.reserve_memory_gb is not None:
            config.setdefault("execution", {}).setdefault("resources", {})["reserve_memory_gb"] = args.reserve_memory_gb
        kv = config["engine"]["workers"]["prefill"]["kv_cache"]
        kv["host_offload"]["num_host_blocks"] = round(gb * 10**9 / BLOCK_BYTES)
        kv["g3_offload"]["num_g3_blocks"] = round(56 * 10**12 / BLOCK_BYTES)
        kv["g3_offload"]["read_bandwidth_gbps"] = float(read)
        selected.append(("bandwidth", name, name, trace))
        settings[name] = config
        if args.plan:
            print(f"{name}: DRAM {kv['host_offload']['num_host_blocks']:,} blocks, "
                  f"SSD {kv['g3_offload']['num_g3_blocks']:,} blocks")
    if args.plan:
        print(f"Total: {len(selected)} cases, serial, memory budget: {args.memory_limit_gb or 'auto'} GB")
        return 0
    if not selected:
        print("All selected cases are already complete", flush=True)
        return 0
    for trace in dict.fromkeys(case[3] for case in selected):
        try:
            count = validate_bundle(trace, parent)
        except (OSError, ValueError, KeyError, StopIteration) as error:
            parser.error(f"Invalid overlay bundle {trace}: {error}")
        print(f"Bundle ready: {trace.name}, {count:,} requests", flush=True)
    return run_cases(args, parent, selected, settings)


if __name__ == "__main__":
    raise SystemExit(main())
