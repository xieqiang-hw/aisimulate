"""Rerun SWE-chat experiments with planned output token IDs."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
import signal
import time
from pathlib import Path
import subprocess
import sys

import yaml

BASE = Path(__file__).resolve().parents[1]
RUNNER = BASE.parent / "disk/scripts/run_experiment.py"

SUITES = ("baseline", "hbm", "overlay", "bandwidth")
FACTORS = (2, 4, 8, 16, 32, 64)
HIGH_FACTORS = (128,)
LOW_BANDWIDTH_FACTORS = (1, 2, 4)
BANDWIDTH_FACTORS = (*LOW_BANDWIDTH_FACTORS, 8, 16, 32, 64, *HIGH_FACTORS)
DRAM_GB = (384, 256, 128)
BUNDLE = Path.home() / ("project/data/swe_chat/derived/mooncake-v1/"
                        "claude-code-main-glm52-128k")
OUTPUT = Path.home() / "project/results/swe-chat-glm52-1p1d"


def cases(bundle, suites, factors, overlay_dram_gb=DRAM_GB):
    for suite in suites:
        if suite == "baseline":
            for gb in DRAM_GB:
                name = f"swe_p_local_8ssd_p{gb}gb_1x"
                yield suite, name, name, bundle
        elif suite == "hbm":
            for blocks in (4096, 3072, 2048):
                name = f"swe_p_local_8ssd_p384gb_1x_hbm{blocks}"
                yield suite, name, name, bundle
        elif suite == "overlay":
            for factor in factors:
                if factor == 1:
                    continue
                for gb in overlay_dram_gb:
                    name = f"swe_p_local_8ssd_p{gb}gb_overlay_{factor}x_hbm2048"
                    config = f"swe_p_local_8ssd_p{gb}gb_1x_hbm2048"
                    yield suite, name, config, bundle.with_name(f"{bundle.name}-overlay-{factor}x")
        elif suite == "bandwidth":
            for factor in factors:
                if factor not in BANDWIDTH_FACTORS:
                    continue
                bandwidths = (8, 16, 32, 48, 64, 128) if factor in HIGH_FACTORS else (16, 32, 48, 64)
                if factor in LOW_BANDWIDTH_FACTORS:
                    bandwidths = (8, 16, 32, 48, 64)
                trace = bundle if factor == 1 else bundle.with_name(f"{bundle.name}-overlay-{factor}x")
                for bandwidth in bandwidths:
                    name = f"swe_p_local_8ssd_p128gb_overlay_{factor}x_hbm2048_read{bandwidth}"
                    config = f"swe_p_local_8ssd_p128gb_1x_hbm2048_read{bandwidth}"
                    if bandwidth == 8:
                        config = "swe_p_local_8ssd_p128gb_1x_hbm2048"
                    yield suite, name, config, trace


def validate_bundle(bundle, parent):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if (manifest["format"] != "mooncake"
            or manifest["model_mapping"]["target_model"] != "zai-org/GLM-5.2-FP8"
            or manifest["context_limit"] != 131072
            or manifest["trace_block_size"] != 16):
        raise ValueError(f"Unexpected bundle format, model or context: {bundle}")
    if bundle != parent:
        overlay = manifest["overlay"]
        factor = int(bundle.name.rsplit("-", 1)[1].removesuffix("x"))
        if (overlay["factor"], overlay["replay_speedup"], overlay["parent_bundle"]) != (
                factor, 1, parent.name):
            raise ValueError(f"Overlay metadata differs from selected parent: {bundle}")
    trace = bundle / "requests.jsonl"
    if trace.stat().st_size != manifest["files"]["requests.jsonl"]["size"]:
        raise ValueError(f"Trace size differs from manifest: {trace}")
    with trace.open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    ids = first.get("output_token_ids")
    if (not isinstance(ids, list) or len(ids) != first["output_length"]
            or any(type(token) is not int or not 0 <= token < 2**32 for token in ids)):
        raise ValueError(f"Expected output_token_ids matching output_length: {trace}")
    return manifest["request_count"]


def save_plan(path, plan):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=SUITES)
    parser.add_argument("--factors", nargs="+", type=int, choices=(1, *FACTORS, *HIGH_FACTORS), default=FACTORS)
    parser.add_argument("--p-dram-gb", nargs="+", type=int, choices=DRAM_GB, default=DRAM_GB,
                        help="Select P-side DRAM capacities for the overlay suite")
    parser.add_argument("--read-bandwidths", nargs="+", type=int, choices=(8, 16, 32, 48, 64, 128))
    parser.add_argument("--max-requests", type=int, default=0, help="0 replays the full trace")
    parser.add_argument("--plan", action="store_true", help="List cases without executing or writing output")
    parser.add_argument("--capture-scheduler-trace", action="store_true", help="Stream per-iteration P/D metrics for every case")
    parser.add_argument("--scheduler-trace-compression", choices=("gzip","none"), default="gzip")
    parser.add_argument("--jobs", type=int, default=1, help="Maximum simultaneous experiments")
    parser.add_argument("--memory-limit-gb", type=float, help="Per-experiment process-tree memory budget")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.memory_limit_gb is not None and (not math.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0):
        parser.error("--memory-limit-gb must be finite and positive")
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    parent = args.bundle.expanduser().resolve()
    selected = list(cases(parent, dict.fromkeys(args.suites), dict.fromkeys(args.factors), dict.fromkeys(args.p_dram_gb)))
    if not args.read_bandwidths and args.factors == FACTORS and "--factors" not in sys.argv:
        selected = [row for row in selected if row[0] != "bandwidth" or not any(
            f"_overlay_{factor}x_" in row[1] for factor in LOW_BANDWIDTH_FACTORS)]
    if args.read_bandwidths:
        selected = [row for row in selected if row[0] != "bandwidth"
                    or int(row[1].rsplit("read", 1)[1]) in args.read_bandwidths]

    if not selected:
        parser.error("No cases selected")
    if args.plan:
        for suite, name, _, bundle in selected:
            print(f"{suite}: {name} -> {bundle}")
        print(f"Total: {len(selected)} cases")
        return 0

    from aisimulate.config.cli import CorePredictionConfig
    settings = {}
    # Preflight all required bundles and YAML files before launching a replay.
    try:
        for bundle in dict.fromkeys(item[3] for item in selected):
            count = validate_bundle(bundle, parent)
            print(f"Bundle ready: {bundle.name}, {count:,} requests", flush=True)
        for _, _, config, _ in selected:
            source = BASE / f"configs/{config}.yaml"
            config_data = yaml.safe_load(source.read_text())
            config_data["evaluation"]["sla"].update(ttft_ms=1000.0, itl_ms=30.0)
            if args.memory_limit_gb is not None:
                config_data.setdefault("execution", {}).setdefault("resources", {})["memory_limit_gb"] = args.memory_limit_gb
            settings[config] = config_data
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        parser.error(f"Preflight failed: {exc}. Finish generating all selected bundles first.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = args.output_root.expanduser().resolve() / stamp
    configs = output / "configs"
    configs.mkdir(parents=True, exist_ok=False)
    for name, config_data in settings.items():
        config = configs / f"{name}.yaml"
        config.write_text(yaml.safe_dump(config_data, sort_keys=False), encoding="utf-8")
        CorePredictionConfig.from_yaml(str(config))

    plan = {
        "bundle": str(parent), "max_requests": args.max_requests,
        "jobs": args.jobs, "memory_limit_gb": args.memory_limit_gb,
        "capture_scheduler_trace": args.capture_scheduler_trace,
        "scheduler_trace_compression": args.scheduler_trace_compression,
        "scheduler_trace_schema": 2 if args.capture_scheduler_trace else None,
        "sla": {"ttft_ms": 1000.0, "tpot_ms": 30.0},
        "cases": [{"suite": suite, "name": name, "config": str(configs / f"{config}.yaml"),
                   "bundle": str(bundle), "status": "pending"}
                  for suite, name, config, bundle in selected],
    }
    plan_path = output / "sweep.json"
    save_plan(plan_path, plan)
    print(f"Sweep directory: {output}", flush=True)
    pending = iter(enumerate(plan["cases"], 1))
    active = {}
    exhausted = False
    try:
        while active or not exhausted:
            while len(active) < args.jobs and not exhausted:
                item = next(pending, None)
                if item is None:
                    exhausted = True
                    break
                index, case = item
                case_output = output / case["suite"] / case["name"]
                case_output.mkdir(parents=True)
                command = [
                    sys.executable, "-u", str(RUNNER), "--config", case["config"],
                    "--bundle", case["bundle"], "--output-root", str(case_output),
                    "--speedup", "1", "--max-requests", str(args.max_requests),
                ]
                if args.capture_scheduler_trace:
                    command.extend(["--capture-scheduler-trace", "--scheduler-trace-compression", args.scheduler_trace_compression])
                log = (case_output / "runner.log").open("w")
                try:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except OSError as error:
                    log.close()
                    case.update(status="failed", error=str(error))
                    save_plan(plan_path, plan)
                    print(f"Failed to start {case['name']}: {error}", flush=True)
                    continue
                case.update(status="running", pid=process.pid, log=str(case_output / "runner.log"))
                active[process] = (case, log)
                save_plan(plan_path, plan)
                print(f"[{index}/{len(selected)}] Running {case['suite']}/{case['name']} "
                      f"PID={process.pid}; log={case['log']}", flush=True)
            for process, (case, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                case.update(status="completed" if code == 0 else "failed", returncode=code)
                del active[process]
                save_plan(plan_path, plan)
                print(f"{case['status']}: {case['suite']}/{case['name']}", flush=True)
            if active:
                time.sleep(0.5)
    except BaseException:
        for process, (case, log) in active.items():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            case["status"] = "interrupted"
        save_plan(plan_path, plan)
        for process, (_, log) in active.items():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            log.close()
        raise
    failed = sum(case["status"] != "completed" for case in plan["cases"])
    print(f"Finished: {len(selected) - failed}/{len(selected)} completed. Plan: {plan_path}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
