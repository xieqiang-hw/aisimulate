"""MiniMax-M2.5 / H100 64 GB / FP8 compute + BF16 KV; serial 8x sweep."""
import argparse
import json
import math
from pathlib import Path
import yaml
from sweep_common import BASE, run_cases

MODEL = "MiniMaxAI/MiniMax-M2.5"
DEFAULT_BUNDLE = Path.home() / (
    "project/data/swe_chat/derived/verified_interleave_mooncake_v1/"
    "claude-code-main-minimax-m25-shape128k-overlay-8x"
)
CAPACITIES = (64, 128, 256, 384, 512, 1000)
BANDWIDTHS = (1, 2, 4, 8, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/d/aisim-results/swe-chat-minimax-h100-64gb-ep8-8x-bf16kv"))
    parser.add_argument("--p-dram-gb", nargs="+", type=int, choices=CAPACITIES, default=CAPACITIES)
    parser.add_argument("--read-bandwidths", nargs="+", type=int, choices=BANDWIDTHS, default=BANDWIDTHS)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--memory-limit-gb", type=float, default=None)
    parser.add_argument("--capture-scheduler-trace", action="store_true", default=True,
                        help="Always capture native P/D scheduler traces for this sweep (default: enabled)")
    parser.add_argument("--scheduler-trace-compression", choices=("gzip",), default="gzip",
                        help="Scheduler traces use gzip compression")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    if args.memory_limit_gb is not None and (
        not math.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0
    ):
        parser.error("--memory-limit-gb must be finite and positive")
    args.jobs = 1
    selected, settings = [], {}
    bundle = args.bundle.expanduser().resolve()
    # Prioritize lower SSD bandwidth, then lower DRAM capacity.
    for read in sorted(set(args.read_bandwidths)):
        for gb in sorted(set(args.p_dram_gb)):
            name = f"swe_minimax_h100_64gb_ep8_p{gb}gb_read{read}_overlay_8x"
            config = yaml.safe_load((BASE / "configs" / (name + ".yaml")).read_text())
            expected = {
                "gemm_quant_mode": "fp8_block", "moe_quant_mode": "fp8_block",
                "fmha_quant_mode": "bfloat16", "kvcache_quant_mode": "bfloat16",
            }
            if any(config["engine"].get(key) != value for key, value in expected.items()):
                parser.error(f"Expected explicit FP8 compute and BF16 KV: {name}")
            # Resolve the local hardware profile relative to this checkout.
            config["engine"]["systems_paths"] = [
                str(BASE / "configs/systems_h100_64gb"), "default"
            ]
            if args.memory_limit_gb is not None:
                config.setdefault("execution", {}).setdefault("resources", {})[
                    "memory_limit_gb"
                ] = args.memory_limit_gb
            settings[name] = config
            selected.append(("bandwidth", name, name, bundle))
            if args.plan:
                kv = config["engine"]["workers"]["prefill"]["kv_cache"]
                print(f"{name}: G2={kv['host_offload']['num_host_blocks']}, "
                      f"G3={kv['g3_offload']['num_g3_blocks']}")
    if args.plan:
        print(f"Total: {len(selected)} cases, serial; host memory budget auto unless specified; "
              f"capture_scheduler_trace={args.capture_scheduler_trace}, "
              f"scheduler_trace_compression={args.scheduler_trace_compression}; KV=bfloat16")
        return 0
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
        if (manifest.get("model_mapping", {}).get("target_model") != MODEL
            or manifest.get("format") != "mooncake"
            or manifest.get("trace_block_size") != 16
            or manifest.get("context_limit") != 131072
            or manifest.get("overlay", {}).get("factor") != 8
            or manifest["overlay"].get("replay_speedup") != 1):
            raise ValueError("Expected MiniMax shape-controlled Mooncake 128K / 8x bundle")
        if not manifest.get("shape_comparison"):
            raise ValueError("Missing controlled workload-shape provenance")
        if (bundle / "requests.jsonl").stat().st_size != manifest["files"]["requests.jsonl"]["size"]:
            raise ValueError("Trace size differs from manifest")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    # The common runner independently verifies SHA256 before each replay.
    return run_cases(args, bundle, selected, settings)


if __name__ == "__main__":
    raise SystemExit(main())
