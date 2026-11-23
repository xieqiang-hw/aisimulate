#!/usr/bin/env python3
"""Run GLM-5.2/H200 chat baselines through the existing AISim experiment runner."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from prepare_chat import DATASETS, sha256
from validate_chat_bundle import validate

BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
RUNNER = BASE.parent / "disk/scripts/run_experiment.py"
TEMPLATE = BASE / "configs/glm52_h200_ep8_baseline.yaml"
MODEL = "zai-org/GLM-5.2-FP8"
# Same GLM DSA layout/capacity convention as disk_swe's H200 TP8/EP8 sweep:
# 60,216 bytes/token/TP shard * 64 tokens/block * 8 TP shards, decimal GB/TB.
INSTANCE_BLOCK_BYTES = 60_216 * 64 * 8


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def case_name(dataset, dram_gb, read_gbps):
    rate = f"{read_gbps:g}".replace(".", "p")
    return f"{dataset}_glm52_h200_ep8_p{dram_gb}gb_read{rate}_1x"


def configure(template, bundle, dram_gb, read_gbps, args):
    config = deepcopy(template)
    engine = config["engine"]
    expected = {"model": MODEL, "hardware": "h200_sxm", "mode": "disaggregated",
                "backend": "vllm", "backend_version": "0.24.0", "context_length": 131072,
                "estimation_mode": "op_level", "fallback_policy": "deny"}
    if any(engine.get(key) != value for key, value in expected.items()):
        raise ValueError("template must retain the GLM-5.2/H200 vLLM 0.24.0 baseline identity")
    for role in ("prefill", "decode"):
        worker = engine["workers"][role]
        parallelism = worker["parallelism"]
        if any(parallelism.get(key) != value for key, value in {
            "replicas": 1, "tensor": 8, "pipeline": 1,
            "attention_data": 1, "moe_tensor": 1, "moe_expert": 8,
        }.items()) or worker["kv_cache"]["block_size"] != 64:
            raise ValueError("capacity conversion requires the SWE TP8/DP1/EP8, 64-token layout")
    config["traffic"]["source"].update(paths=[str(bundle / "requests.jsonl")], format="mooncake", block_size=16)
    config["traffic"]["load"].update(type="trace_timestamps", speedup=args.speedup)
    kv = engine["workers"]["prefill"]["kv_cache"]
    kv["host_offload"]["num_host_blocks"] = round(dram_gb * 10**9 / INSTANCE_BLOCK_BYTES)
    kv["g3_offload"]["num_g3_blocks"] = round(56 * 10**12 / INSTANCE_BLOCK_BYTES)
    kv["g3_offload"]["read_bandwidth_gbps"] = read_gbps
    resources = config.setdefault("execution", {}).setdefault("resources", {})
    if args.memory_limit_gb is not None:
        resources["memory_limit_gb"] = args.memory_limit_gb
    if args.reserve_memory_gb is not None:
        resources["reserve_memory_gb"] = args.reserve_memory_gb
    return config


def build_cases(args):
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate import _runtime

    template = yaml.safe_load(args.template.read_text())
    runtime_identity = {
        "git_revision": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "aisim_version": importlib.metadata.version("aisim"),
        "runtime_sha256": sha256(Path(_runtime.__file__)),
    }
    names = list(DATASETS) if "all" in args.datasets else list(dict.fromkeys(args.datasets))
    cases = []
    for dataset in names:
        bundle = (args.bundle_root / dataset).resolve()
        report = validate(bundle)
        manifest = json.loads((bundle / "manifest.json").read_text())
        if (manifest.get("dataset") != dataset or manifest["model_mapping"]["target_model"] != MODEL
                or manifest["context_limit"] != 131072):
            raise ValueError(f"unexpected dataset/model/context in bundle: {bundle}")
        for dram, read in product(dict.fromkeys(args.p_dram_gb), dict.fromkeys(args.read_bandwidths)):
            config = configure(template, bundle, dram, read, args)
            CorePredictionConfig.model_validate(config)
            identity = {"bundle": str(bundle), "manifest_sha256": sha256(bundle / "manifest.json"),
                        "config": config, "max_requests": args.max_requests,
                        "capture_scheduler_trace": args.capture_scheduler_trace,
                        "export_html": args.export_html,
                        "runtime_identity": runtime_identity,
                        "runner_sha256": sha256(RUNNER)}
            cases.append({"name": case_name(dataset, dram, read), "dataset": dataset,
                          "bundle": str(bundle), "request_count": report["requests"],
                          "session_count": report["sessions"], "span_seconds": report["span_seconds"],
                          "p_dram_gb": dram, "read_gbps_per_tp": read,
                          "config": config, "fingerprint": canonical_hash(identity),
                          "manifest_sha256": identity["manifest_sha256"],
                          "runtime_identity": runtime_identity,
                          "status": "pending"})
    return cases


def reusable_cases(paths):
    result = {}
    for path in paths:
        previous = json.loads(path.read_text())
        if previous.get("schema") != "chat-baseline-v1":
            raise ValueError(f"not a chat baseline sweep: {path}")
        for case in previous["cases"]:
            if case["status"] != "completed":
                continue
            run_path = Path(case["run_json"])
            run = json.loads(run_path.read_text())
            if run.get("status") != "completed" or run.get("returncode") != 0 or run.get("dry_run"):
                raise ValueError(f"completed case has no successful real replay: {run_path}")
            if not (run_path.parent / "prediction/prediction.json").is_file():
                raise ValueError(f"completed case prediction is missing: {run_path}")
            result[case["fingerprint"]] = str(run_path)
    return result


def command_for(case, config_path, case_output, args):
    command = [sys.executable, "-u", str(RUNNER), "--config", str(config_path),
               "--bundle", case["bundle"], "--output-root", str(case_output),
               "--speedup", str(args.speedup), "--max-requests", str(args.max_requests)]
    if args.dry_run:
        command.append("--dry-run")
    if args.capture_scheduler_trace:
        command.extend(["--capture-scheduler-trace", "--scheduler-trace-compression", "gzip"])
    if args.export_html:
        command.append("--export-html")
    return command


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def record_child(case, directory, returncode, dry_run):
    candidates = list(directory.glob("*/run.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one child run.json, got {len(candidates)} in {directory}")
    path = candidates[0]
    run = json.loads(path.read_text())
    case.update(run_json=str(path), prediction_dir=str(path.parent / "prediction"),
                selected_requests=run["selected_requests"], wall_seconds=run.get("wall_seconds"))
    if returncode != 0:
        raise ValueError(f"experiment runner exited {returncode}")
    if dry_run:
        if not run.get("dry_run") or run["status"] != "prepared":
            raise ValueError("child did not produce a valid dry-run record")
        case["status"] = "validated"
    else:
        if run.get("dry_run") or run.get("status") != "completed":
            raise ValueError("child did not complete a real replay")
        prediction = path.parent / "prediction/prediction.json"
        if not prediction.is_file():
            raise ValueError("successful runner produced no prediction.json")
        # Read only the summary prefix: full per-request output may be very large.
        with prediction.open() as stream:
            header = stream.read(100_000).split('"per_request":', 1)[0]
        marker = '"completed_requests":'
        if marker not in header:
            raise ValueError("completed_requests is absent from prediction summary")
        completed = json.JSONDecoder().raw_decode(header.split(marker, 1)[1].lstrip())[0]
        case["completed_requests"] = completed
        if completed != run["selected_requests"]:
            raise ValueError(f"only {completed}/{run['selected_requests']} requests completed")
        case["status"] = "completed"


def execute(cases, args):
    reusable = reusable_cases(args.skip_completed_from)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output_root / stamp).resolve()
    configs = output / "configs"
    configs.mkdir(parents=True, exist_ok=False)
    for case in cases:
        path = configs / f"{case['name']}.yaml"
        path.write_text(yaml.safe_dump(case.pop("config"), sort_keys=False))
        case["config_path"] = str(path)
        if not args.dry_run and case["fingerprint"] in reusable:
            case.update(status="skipped_completed", prior_run_json=reusable[case["fingerprint"]])
    plan = {"schema": "chat-baseline-v1", "created_utc": stamp, "output": str(output),
            "reference": str(BASE.parent / "disk_swe/glm52_experiments.html"),
            "reference_sha256": sha256(BASE.parent / "disk_swe/glm52_experiments.html"),
            "template": str(args.template), "template_sha256": sha256(args.template),
            "runner": str(RUNNER), "runner_sha256": sha256(RUNNER),
            "script_sha256": sha256(Path(__file__)),
            "git_revision": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "python": sys.version, "dry_run": args.dry_run, "jobs": args.jobs,
            "max_active_runners": 0,
            "overlay_factor": 1, "speedup": args.speedup, "max_requests": args.max_requests,
            "sla": {"ttft_ms": 1000, "tpot_ms": 30}, "cases": cases}
    plan_path = output / "sweep.json"
    save(plan_path, plan)
    print(f"Sweep: {plan_path}", flush=True)
    failed, exhausted, halt_launch = False, False, False
    pending = iter(enumerate(cases, 1))
    active = {}

    def cancel_active(reason):
        for process, (case, log, _, start) in list(active.items()):
            stop_child(process)
            log.close()
            case.update(status="interrupted", error=reason,
                        runner_wall_seconds=time.monotonic() - start)
        active.clear()
        save(plan_path, plan)

    try:
        while active or not exhausted:
            while not halt_launch and not exhausted and len(active) < args.jobs:
                item = next(pending, None)
                if item is None:
                    exhausted = True
                    break
                index, case = item
                if case["status"] == "skipped_completed":
                    print(f"[{index}/{len(cases)}] skipped: {case['name']}", flush=True)
                    continue
                directory = output / "cases" / case["name"]
                command = command_for(case, Path(case["config_path"]), directory, args)
                log, process = None, None
                start = time.monotonic()
                try:
                    directory.mkdir(parents=True)
                    log = (directory / "runner.log").open("w")
                    case.update(status="running", command=command, log=str(directory / "runner.log"),
                                started_utc=datetime.now(timezone.utc).isoformat())
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[process] = (case, log, directory, start)
                    case["pid"] = process.pid
                    plan["max_active_runners"] = max(plan["max_active_runners"], len(active))
                    save(plan_path, plan)
                    print(f"[{index}/{len(cases)}] running ({len(active)}/{args.jobs}): {case['name']}", flush=True)
                except OSError as error:
                    if process in active:
                        raise
                    if log is not None:
                        log.close()
                    failed = True
                    case.update(status="failed", error=str(error), runner_wall_seconds=time.monotonic() - start)
                    save(plan_path, plan)
                    halt_launch = args.fail_fast
            for process, (case, log, directory, start) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[process]
                case["returncode"] = code
                try:
                    record_child(case, directory, code, args.dry_run)
                except (OSError, ValueError, KeyError) as error:
                    failed = True
                    case.update(status="failed", error=str(error))
                    halt_launch = args.fail_fast
                case.update(runner_wall_seconds=time.monotonic() - start,
                            finished_utc=datetime.now(timezone.utc).isoformat())
                save(plan_path, plan)
                print(f"{case['status']}: {case['name']}; log={case['log']}", flush=True)
            if halt_launch:
                exhausted = True
                plan["halt_reason"] = "fail-fast: no further cases started; existing cases allowed to finish"
                save(plan_path, plan)
            if active:
                time.sleep(0.1)
    except KeyboardInterrupt:
        cancel_active("interrupted by user")
        return 130
    except BaseException:
        cancel_active("coordinator aborted")
        raise
    print(f"Finished: {sum(case['status'] == 'completed' for case in cases)} completed; plan: {plan_path}", flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=[*DATASETS, "all"], default=["all"])
    parser.add_argument("--bundle-root", type=Path, default=Path.home() / "project/data/chat/derived/mooncake-v1")
    parser.add_argument("--output-root", type=Path, help="defaults to the baseline or sweep results directory")
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--sweep", action="store_true", help="preset: 7 datasets x 6 DRAM sizes x 5 SSD read rates")
    parser.add_argument("--jobs", type=int, default=1, help="maximum simultaneous experiment runners (use 4 for the sweep)")
    parser.add_argument("--p-dram-gb", nargs="+", type=int)
    parser.add_argument("--read-bandwidths", nargs="+", type=float, choices=(1, 2, 4, 8, 16))
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--max-requests", type=int, default=0, help="0: full trace; prefix includes boundary timestamp ties")
    parser.add_argument("--memory-limit-gb", type=float)
    parser.add_argument("--reserve-memory-gb", type=float)
    parser.add_argument("--capture-scheduler-trace", action="store_true")
    parser.add_argument("--export-html", action="store_true")
    parser.add_argument("--skip-completed-from", type=Path, action="append", default=[])
    parser.add_argument("--fail-fast", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="validate inputs and print plan without writing/running")
    mode.add_argument("--dry-run", action="store_true", help="materialize configs and check runner; no simulation")
    mode.add_argument("--write-configs", type=Path, help="write reviewable/directly runnable YAMLs only")
    args = parser.parse_args(argv)
    args.p_dram_gb = args.p_dram_gb or ([64, 128, 256, 384, 512, 1000] if args.sweep else [384])
    args.read_bandwidths = args.read_bandwidths or ([1, 2, 4, 8, 16] if args.sweep else [8.0])
    if args.output_root is None:
        args.output_root = Path.home() / ("project/results/chat-glm52-h200-sweep" if args.sweep else "project/results/chat-glm52-h200-baseline")
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    if args.max_requests < 0 or any(gb <= 0 for gb in args.p_dram_gb):
        parser.error("max-requests must be nonnegative; DRAM GB must be positive")
    for name, values in (("read-bandwidths", args.read_bandwidths), ("speedup", [args.speedup]),
                         ("memory-limit-gb", [] if args.memory_limit_gb is None else [args.memory_limit_gb])):
        if any(not math.isfinite(value) or value <= 0 for value in values):
            parser.error(f"{name} must be finite and positive")
    if args.reserve_memory_gb is not None and (not math.isfinite(args.reserve_memory_gb) or args.reserve_memory_gb < 0):
        parser.error("reserve-memory-gb must be finite and nonnegative")
    args.bundle_root = args.bundle_root.expanduser().resolve()
    args.template = args.template.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.skip_completed_from = [path.expanduser().resolve() for path in args.skip_completed_from]
    try:
        cases = build_cases(args)
        if args.plan:
            preview = [{key: value for key, value in case.items() if key != "config"} for case in cases]
            print(json.dumps({"case_count": len(cases), "jobs": args.jobs, "speedup": args.speedup,
                              "max_requests": args.max_requests, "overlay_factor": 1, "cases": preview}, indent=2))
            return 0
        if args.write_configs:
            destination = args.write_configs.expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            rendered = {destination / f"{case['name']}.yaml": yaml.safe_dump(case["config"], sort_keys=False) for case in cases}
            for path, text in rendered.items():
                if path.exists() and path.read_text() != text:
                    raise ValueError(f"refusing to overwrite a different config: {path}")
            for path, text in rendered.items():
                path.write_text(text)
                print(path)
            return 0
        return execute(cases, args)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"baseline: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
