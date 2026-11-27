"""Build the offline GLM dashboard and its per-configuration JS assets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np

from throughput_capacity_data import report_header
from throughput_execution import TOOL, make_observation

BASE = Path(__file__).resolve().parents[1]
DECODE1_PLAN = Path("/mnt/d/aisim-results/swe-chat-glm52-decode1/20261126T095908.068749Z/sweep.json")
BASELINES = {8: "baseline_8x.json", 16: "overlay_16x.json", 32: "overlay_32x.json"}
BIN_MS = 300_000


def write(path, content):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def js(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def request_metrics(prediction, single_token):
    arrival, end, admit = [], [], []
    p_wait = d_wait = d_count = 0
    with (prediction / "requests.jsonl").open("rb") as stream:
        for line in stream:
            r = json.loads(line)
            arrival.append(r["arrival_time_ms"])
            end.append(r["terminal_time_ms"])
            admit.append(r["prefill_admit_ms"])
            p_wait += r["prefill_wait"]["scheduling_wait_ms"]
            held, reserved = r["source_held_ms"], r["destination_reserved_ms"]
            if held is not None and reserved is not None:
                d_wait += reserved - held
                d_count += 1
            if single_token and (r["output_length"], r["requested_output_length"]) != (1, 1):
                raise ValueError("Expected single-token output for every request")
    return (np.array(arrival), np.array(end), np.array(admit),
            dict(mean_p_wait_ms=p_wait / len(arrival), mean_d_reserve_ms=d_wait / d_count if d_count else None))


def capped_case(row, family, arrival_ref):
    case_dir = Path(row["config"]).parent
    run_path = max(case_dir.glob("runs/*/run.json"))
    run = json.loads(run_path.read_text())
    prediction = run_path.parent / "prediction"
    assert run["status"] == "completed" and run["output_token_limit"] == 1
    assert run["speedup"] == family["speedup"] == 1 and run["capture_scheduler_trace"]
    assert run["source_trace_sha256"] == family["requests_sha256"]
    a, t, p, waits = request_metrics(prediction, True)
    assert np.array_equal(a, arrival_ref) and np.all((a <= p) & (p <= t))
    header = report_header(prediction)
    assert len(a) == header["completed_requests"] == family["request_count"]
    count = len(family["demand"]["requests"])
    edges = np.arange(1, count + 1) * BIN_MS
    arrivals, terminals, admissions = map(np.sort, (a, t, p))
    buckets = np.bincount((a / BIN_MS).astype(int), minlength=count).tolist()
    assert buckets == family["demand"]["requests"]
    completed = np.bincount((t / BIN_MS).astype(int), minlength=count).tolist()
    assert len(completed) == count and sum(completed) == len(a)
    arrived_at_edges = np.searchsorted(arrivals, edges, side="right")
    unfinished = (arrived_at_edges - np.searchsorted(terminals, edges, side="right")).tolist()
    waiting = (arrived_at_edges - np.searchsorted(admissions, edges, side="right")).tolist()
    peak = np.searchsorted(arrivals, arrivals, side="right") - np.searchsorted(terminals, arrivals, side="right")
    manifest = json.loads((Path(run["bundle"]) / "manifest.json").read_text())

    def window(end_ms):
        arrived = int(np.searchsorted(arrivals, end_ms))
        done = int(np.searchsorted(terminals, end_ms))
        return dict(arrived_requests=arrived, completed_requests=done, backlog_start=0,
                    arrival_request_per_s=arrived / (end_ms / 1000),
                    completion_request_per_s=done / (end_ms / 1000),
                    backlog_end=arrived-done, peak_inflight_requests=int(peak[:arrived].max()))

    with TemporaryDirectory(prefix="glm52-decode1-") as tmp:
        target = Path(tmp) / "execution.json"
        subprocess.run([str(TOOL), "--bin-seconds", "300", "--overview-only",
                        str(prediction), str(target)], check=True)
        native = json.loads(target.read_text())
    assert abs(native["end_s"] - header["duration_ms"] / 1000) < 0.001
    item = dict(case=case_dir.name, dram_gb=256, read_gbps_per_tp=2, speedup=1,
                source_result=str(prediction / "requests.jsonl"), output_token_limit=1,
                total_input_tokens=sum(family["demand"]["input"]), total_output_tokens=len(a),
                duration_ms=header["duration_ms"], committed_prefill_tokens=header["committed_prefill_tokens"],
                mean_ttft_ms=header["mean_ttft_ms"], p99_ttft_ms=header["p99_ttft_ms"],
                mean_tpot_ms=None, sla_percent=None, **waits,
                reuse_ratio=header["first_admission_prefix_cache_reused_ratio"],
                ssd_read_tb=header["g3_offload"]["read"]["completed_bytes"] / 1e12,
                main=window(manifest["overlay"]["segment_ms"]), whole=window(a.max() + 1),
                series=dict(completed_requests=completed, completed_output_tokens=completed,
                            unfinished_requests=unfinished, p_preadmission_waiting=waiting),
                execution=make_observation(native, count, prediction, header["committed_prefill_tokens"], len(a)))
    return {**family, "cases": [item], "demand": {**family["demand"], "output": buckets}}


def build_pair(factor, analysis, row):
    family = json.loads((analysis / BASELINES[factor]).read_text())
    family["factor"] = factor
    family["cases"] = [c for c in family["cases"] if c["dram_gb"] == 256 and c["read_gbps_per_tp"] == 2]
    assert len(family["cases"]) == 1
    original = family["cases"][0]
    assert original.get("speedup", 1) == family["speedup"] == 1 and original["execution"]
    assert sum(original["series"]["completed_requests"]) == family["request_count"]
    family["execution_ready_count"] = 1
    baseline = Path(row["baseline"])
    run = json.loads(baseline.read_text())
    assert run["speedup"] == 1 and run["source_trace_sha256"] == family["requests_sha256"]
    assert original["execution"]["source_directory"] == str(baseline.parent / "prediction")
    arrival, _, _, waits = request_metrics(Path(original["execution"]["source_directory"]), False)
    original.update(waits)
    print(f"Aggregating {factor}× single-token run", flush=True)
    return factor, family, capped_case(row, family, arrival)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=BASE / "analysis/throughput_glm52")
    parser.add_argument("--output", type=Path, default=BASE / "throughput_glm52.html")
    parser.add_argument("--decode1-plan", type=Path, default=DECODE1_PLAN)
    args = parser.parse_args()
    plan = {row["factor"]: row for row in json.loads(args.decode1_plan.read_text()) if row["status"] == "completed"}
    if missing := BASELINES.keys() - plan.keys():
        parser.error(f"Missing completed single-token runs: {sorted(missing)}")
    assets = args.output.with_name(args.output.stem + "_data")
    assets.mkdir(parents=True, exist_ok=True)
    entries = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(build_pair, factor, args.analysis, plan[factor]) for factor in BASELINES]
        for job in jobs:
            factor, original, one = job.result()
            for mode, data in (("original", original), ("one", one)):
                key = f"{factor}x-{mode}"
                file = key + ".js"
                write(assets / file, f"globalThis.glm52ThroughputData[{js(key)}]={js(data)};\n")
                entries.append(dict(factor=factor, mode=mode, key=key, file=file))
    index = dict(entries=entries, updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
    write(assets / "index.js", f"globalThis.glm52ThroughputData={{}};\nglobalThis.glm52ThroughputIndex={js(index)};\n")
    template = Path(__file__).with_name("throughput_dashboard.html").read_text(encoding="utf-8")
    write(args.output, template.replace("__DATA_DIRECTORY__", assets.name))
    print(f"Saved {args.output} with {len(entries)} configurations", flush=True)


if __name__ == "__main__":
    main()
