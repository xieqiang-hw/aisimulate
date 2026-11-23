"""Shared bundle validation and batch execution for active SWE-chat sweeps."""
from datetime import datetime, timezone
import json
import os
import signal
import time
from pathlib import Path
import subprocess
import sys
import yaml

BASE = Path(__file__).resolve().parents[1]
RUNNER = BASE.parent / "disk/scripts/run_experiment.py"
BUNDLE = Path.home() / ("project/data/swe_chat/derived/mooncake-v1/"
                        "claude-code-main-glm52-128k")

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

def run_cases(args, parent, selected, settings):
    from aisimulate.config.cli import CorePredictionConfig

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
