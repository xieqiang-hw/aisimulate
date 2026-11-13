"""Validate scheduler observations against unchanged replay behavior."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from scheduler_trace import iter_iterations, iter_events

BASE = Path(__file__).resolve().parents[1]


def load_rows(path):
    if path.name.startswith("iteration_metrics"):
        return list(iter_iterations(path.parent))
    if path.name.startswith("scheduling_events"):
        return list(iter_events(path.parent))
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        return [json.loads(line) for line in stream]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/aisim-scheduler-observation-validation"))
    parser.add_argument("--compression", choices=("gzip", "none"), default="gzip")
    args = parser.parse_args()
    root = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    root.mkdir(parents=True)
    trace = root / "requests.jsonl"
    trace.write_text("".join(json.dumps({
        "timestamp": stamp, "input_length": length, "output_length": 2,
        "hash_ids": list(range(key, key + (length + 15) // 16)),
        "output_token_ids": [300001, 300002],
    }) + "\n" for stamp, length, key in ((0,129,0),(1000,257,100),(2000,129,0))))
    # Use the existing small-capacity/fixed-timing fixture to force eviction.
    config = yaml.safe_load((BASE.parent / "disk/configs/p_g3_local_validation.yaml").read_text())
    config["traffic"]["source"]["paths"] = [str(trace)]
    config["engine"]["workers"]["decode"]["kv_cache"]["host_offload"] = {
        "scope": "dp_rank_local", "num_host_blocks": 8, "connector": "offloading_v0_24"}
    cfg = root / "resolved.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    reports = {}
    for enabled in (False, True):
        out = root / ("enabled" if enabled else "disabled")
        env = os.environ.copy()
        env.pop("AISIM_SCHEDULER_TRACE_DIR", None)
        if enabled:
            env["AISIM_SCHEDULER_TRACE_DIR"] = str(out)
            env["AISIM_SCHEDULER_TRACE_COMPRESSION"] = args.compression
        with (root / f"{out.name}.log").open("w") as log:
            subprocess.run([str(Path(sys.executable).parent / "aisim"), "predict", "--stack", "engine",
                            "--config", str(cfg), "--output-dir", str(out), "--capture-per-request"],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[enabled] = json.loads((out / "prediction.json").read_text())
    plain = load_rows(root / "disabled/requests.jsonl")
    observed = load_rows(root / "enabled/requests.jsonl")
    for rows in (plain, observed):
        for row in rows:
            row.pop("uuid")
    assert plain == observed, "Per-request execution changed with recording enabled"
    for key in ("completed_requests", "p99_ttft_ms", "mean_ttft_ms", "mean_tpot_ms", "g3_offload"):
        assert reports[False][key] == reports[True][key], key
    assert not (root / "disabled/iteration_metrics.jsonl.gz").exists()
    iterations = load_rows(root / "enabled/iteration_metrics.jsonl.gz")
    events = load_rows(root / "enabled/scheduling_events.jsonl.gz")
    assert {row["role"] for row in iterations} == {"prefill", "decode"}
    last = defaultdict(lambda: float("-inf"))
    for row in iterations:
        identity = row["role"], row["worker_id"], row["dp_rank"]
        assert row["start_ms"] >= last[identity]
        last[identity] = row["start_ms"]
        assert row["start_ms"] <= row["compute_start_ms"] <= row["end_ms"]
        assert row["hbm_active_blocks"] + row["hbm_inactive_blocks"] <= row["hbm_total_blocks"]
        assert row["prefill_bs"] <= 1 and row["decode_bs"] <= 1
        assert row["gauge_boundary"] == "after_schedule_before_emission"
    p_rows = [row for row in iterations if row["role"] == "prefill"]
    d_rows = [row for row in iterations if row["role"] == "decode"]
    assert sum(row["prefill_new_tokens"] for row in p_rows) == 387
    assert sum(row["decode_generated_tokens"] for row in d_rows) == 6
    assert max(row["g3_pending_read_jobs"] or 0 for row in p_rows) > 0
    assert max(row["h2d_loading_requests"] for row in iterations) > 0
    assert reports[True]["g3_offload"]["read"]["completed_bytes"] > 0
    assert observed[2]["first_admission_g3_reused_input_tokens"] == 128
    event_counts = Counter((row["role"], row["event"]) for row in events)
    assert event_counts[("prefill","admission")] == event_counts[("decode","admission")] == 3
    # A separate D pressure fixture must produce an actual D-local G2 readmission.
    pressure_trace = root / "decode-pressure.jsonl"
    pressure_trace.write_text("".join(json.dumps({
        "timestamp": stamp, "input_length": 65, "output_length": output,
        "hash_ids": list(range(key, key + 5)), "output_token_ids": [key + 300001] * output,
    }) + "\n" for stamp, output, key in ((0,140,0),(10,100,100))))
    config["traffic"]["source"]["paths"] = [str(pressure_trace)]
    config["engine"]["workers"]["decode"]["scheduler"]["max_sequences"] = 8
    config["engine"]["workers"]["decode"]["kv_cache"]["capacity"]["blocks"] = 4
    config["engine"]["workers"]["decode"]["kv_cache"]["host_offload"]["num_host_blocks"] = 16
    pressure_cfg = root / "decode-pressure.yaml"
    pressure_cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    pressure_rows = {}
    for enabled in (False, True):
        out = root / ("pressure-enabled" if enabled else "pressure-disabled")
        env = os.environ.copy()
        env.pop("AISIM_SCHEDULER_TRACE_DIR", None)
        if enabled:
            env["AISIM_SCHEDULER_TRACE_DIR"] = str(out)
            env["AISIM_SCHEDULER_TRACE_COMPRESSION"] = args.compression
        with (root / f"{out.name}.log").open("w") as log:
            subprocess.run([str(Path(sys.executable).parent / "aisim"), "predict", "--stack", "engine",
                            "--config", str(pressure_cfg), "--output-dir", str(out), "--capture-per-request"],
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        pressure_rows[enabled] = load_rows(out / "requests.jsonl")
        for row in pressure_rows[enabled]:
            row.pop("uuid")
    assert pressure_rows[False] == pressure_rows[True]
    d_events = load_rows(root / "pressure-enabled/scheduling_events.jsonl.gz")
    d_iterations = load_rows(root / "pressure-enabled/iteration_metrics.jsonl.gz")
    preemptions = [row for row in d_events if row["role"] == "decode" and row["event"] == "preemption"]
    readmissions = [row for row in d_events if row["role"] == "decode" and row["event"] == "readmission"]
    assert preemptions and readmissions
    assert max(row["h2d_loading_requests"] for row in d_iterations if row["role"] == "decode") > 0
    history = [a for row in pressure_rows[True] for a in row["admission_history"]
               if a["pool"] == "decode" and a["is_readmission"]]
    assert len(readmissions) == len(history)
    assert sum(row["reused_input_tokens"] for row in readmissions) == sum(row["reused_input_tokens"] for row in history) == 64

    summary = {
        "passed": True, "run_dir": str(root),
        "decode_pressure_exact_match": True, "decode_readmissions": len(readmissions),
        "decode_g2_readmission_reused_tokens": 64, "completed_requests": len(observed),
        "per_request_exact_match_excluding_uuid": True, "report_metrics_exact_match": True,
        "iteration_rows": len(iterations),
        "events": {f"{role}/{event}": count for (role,event),count in event_counts.items()},
        "prefill_new_tokens": sum(row["prefill_new_tokens"] for row in p_rows),
        "decode_generated_tokens": sum(row["decode_generated_tokens"] for row in d_rows),
        "max_g3_read_jobs": max(row["g3_pending_read_jobs"] or 0 for row in p_rows),
        "max_h2d_loading_requests": max(row["h2d_loading_requests"] for row in iterations),
    }
    (root / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
