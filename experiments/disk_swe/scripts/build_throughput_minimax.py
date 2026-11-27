"""Build offline MiniMax throughput assets from completed runs, without replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from throughput_capacity_data import report_header
from throughput_execution import TOOL, make_observation

BASE = Path(__file__).resolve().parents[1]
ASSETS = BASE / "throughput_minimax_data"
ROOTS = [
    Path("/mnt/d/aisim-results/swe-chat-minimax-h100-64gb-ep8-32x-bf16kv"),
    Path("/mnt/d/aisim-results/swe-chat-scheduler-trace-supplement"),
    Path("/mnt/e/aisim-results/swe-chat-minimax-256x-large-dram"),
]
ROWS = BASE / "analysis/model_reports/minimax_rows.json"
CASE = re.compile(r"p(\d+)gb_read(\d+)")
NUMBER = rb"(-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
FIELDS = [b"arrival_time_ms", b"terminal_time_ms", b"prefill_admit_ms", b"input_length", b"output_length"]
PATTERN = re.compile(rb'"(' + b"|".join(FIELDS) + rb')":\s*' + NUMBER)
FILES = ["prediction.json", "requests.jsonl", "iteration_metrics.jsonl.gz",
         "scheduler_states.jsonl.gz", "scheduling_events.jsonl.gz", "scheduler_trace_metadata.json"]


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def script(path, key, value):
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    write(path, f"globalThis.minimaxThroughput[{json.dumps(key)}]={data};\n")


def cases():
    rows = json.loads(ROWS.read_text())["rows"]
    index = {}
    for r in rows:
        if r.get("output_token_limit"):
            continue
        factor, speed = r["factor"], r["speedup"]
        if (factor, speed) not in ((32, 1), (256, .125)):
            continue
        g, bw = r["dram_gb"], int(r["read_gbps"])
        key = f"{factor}x-p{g}-r{bw}"
        index[key] = dict(key=key, factor=factor, speed=speed, dram=g, read=bw,
                          status=r["status"], values=r["values"])
    sources = {}
    for root in ROOTS:
        for path in root.rglob("run.json"):
            match = CASE.search(str(path))
            if not match or "glm52_p" in str(path):
                continue
            try:
                run = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            bundle = Path(run.get("bundle", ""))
            factor = 256 if "overlay-256x" in bundle.name else 32
            g, bw = map(int, match.groups())
            key = f"{factor}x-p{g}-r{bw}"
            if key not in index or run.get("selected_requests") != 607190:
                continue
            pred = path.parent / "prediction"
            if run.get("status") == "completed" and all((pred / n).is_file() and (pred / n).stat().st_size for n in FILES):
                sources[key] = (pred, bundle)
                index[key]["status"] = "已完成"
    return index, sources


def interval_peak(arrival, departure, count):
    events = np.concatenate((arrival, departure))
    changes = np.concatenate((np.ones(len(arrival), dtype=int), -np.ones(len(departure), dtype=int)))
    order = np.argsort(events, kind="stable")
    t = events[order]
    cumulative = np.cumsum(changes[order])
    last = np.r_[t[1:] != t[:-1], True]
    bins = np.floor(t[last] / 300000).astype(int)
    edges = np.arange(count) * 300000
    at_start = np.searchsorted(arrival, edges, side="right") - np.searchsorted(departure, edges, side="right")
    np.maximum.at(at_start, bins, cumulative[last])
    return at_start


def aggregate(row, pred, bundle):
    key = row["key"]
    suffix = hashlib.sha256(json.dumps([(n, (pred/n).stat().st_size, (pred/n).stat().st_mtime_ns)
                                          for n in FILES]).encode() + str(pred).encode() + b"glm-view-v2").hexdigest()[:12]
    stem = key + "-" + suffix
    hour, detail = stem + "-hour.js", stem + "-5min.js"
    meta_path = ASSETS / (stem + ".json")
    if meta_path.is_file() and (ASSETS/hour).is_file() and (ASSETS/detail).is_file():
        return json.loads(meta_path.read_text())
    print("Aggregating", key, flush=True)
    header = report_header(pred)
    native_path = ASSETS / (stem + ".native.tmp")
    subprocess.run(["nice", "-n", "12", str(TOOL), "--bin-seconds", "300", "--overview-only",
                    str(pred), str(native_path)], check=True)
    native = json.loads(native_path.read_text())
    native_path.unlink()
    columns = {k: [] for k in FIELDS}
    with (pred/"requests.jsonl").open("rb") as stream:
        for line in stream:
            values = dict(PATTERN.findall(line))
            for k in FIELDS:
                columns[k].append(float(values[k]))
    a, t, p, inp, out = (np.asarray(columns[k]) for k in FIELDS)
    assert len(a) == header["completed_requests"] == 607190 and np.all(p>=a) and np.all(t>=p)
    count = math.floor(max(t.max(), a.max(), header["duration_ms"]) / 300000) + 1
    sa, st, sp = np.sort(a), np.sort(t), np.sort(p)
    bucket = lambda times, values=None: np.bincount((times/300000).astype(int), weights=values, minlength=count)
    stream = make_observation(native, count, pred, header["committed_prefill_tokens"], int(out.sum()))["streams"]
    pre, dec = stream["prefill"], stream["decode"]
    edges = np.arange(1, count + 1) * 300000
    demand = dict(requests=bucket(a), input=bucket(a, inp), output=bucket(a, out))
    series = dict(completed_requests=bucket(t), completed_output_tokens=bucket(t, out),
                  unfinished_requests=np.searchsorted(sa, edges, side="left")-np.searchsorted(st, edges, side="left"),
                  p_preadmission_waiting=np.searchsorted(sa, edges, side="left")-np.searchsorted(sp, edges, side="left"))
    backlog_peak, waiting_peak = interval_peak(sa, st, count), interval_peak(sa, sp, count)
    bounds = json.loads((bundle/"manifest.json").read_text())["overlay"]
    assert int(bounds["factor"]) == row["factor"]
    assert sum(demand["requests"]) == sum(series["completed_requests"]) == 607190
    main_end, whole_end = bounds["segment_ms"]/row["speed"], a.max()+1
    windows = {}
    for name, end in (("main", main_end), ("whole", whole_end)):
        arrived, completed = np.searchsorted(sa,end), np.searchsorted(st,end)
        exact = sa[sa<end]
        peak = np.max(np.searchsorted(sa,exact,side="right")-np.searchsorted(st,exact,side="right"))
        windows[name] = dict(arrival_request_per_s=float(arrived/(end/1000)),
                             completion_request_per_s=float(completed/(end/1000)),
                             backlog_end=int(arrived-completed),peak_inflight_requests=int(peak))
    observation = make_observation(native, count, pred, header["committed_prefill_tokens"], int(out.sum()))
    fields = ("busy_percent","active_mean_bs","max_bs","prefill_tokens","generated_tokens",
              "g3_read_jobs_max","h2d_loading_max")
    def groups(v):
        return [v[i:i+12] for i in range(0,count,12)]
    def reduce(v, fn):
        return [fn(group) for group in groups(v)]
    def pack(v):
        return [round(float(x), 5) if x is not None else None for x in v]
    def view(hourly):
        ds = {k:pack(reduce(v,sum) if hourly else v) for k,v in demand.items()}
        ss = {k:pack((reduce(v,sum) if k.startswith("completed") else reduce(v,lambda g:g[-1]))
                     if hourly else v) for k,v in series.items()}
        exec_data = {}
        for role, stream in observation["streams"].items():
            metrics = {k:pack(stream[k]) for k in fields}
            if hourly:
                metrics = {k:pack(reduce(stream[k],sum if k.endswith("tokens") else
                       (lambda g:sum(g)/len(g)) if k=="busy_percent" else max))
                       for k in fields if k!="active_mean_bs"}
                metrics["active_mean_bs"] = [
                    sum((v or 0)*w for v,w in zip(vs,ws))/sum(ws) if sum(ws) else None
                    for vs,ws in zip(groups(stream["active_mean_bs"]),groups(stream["busy_percent"]))]
            exec_data[role] = metrics
        peaks = dict(input=pack(reduce(demand["input"],max)), backlog=pack(reduce(backlog_peak,max)),
                     waiting=pack(reduce(waiting_peak,max)),
                     **{role+"_"+k:pack(reduce(stream[k],max))
                       for role,stream in observation["streams"].items()
                       for k in ("prefill_tokens","generated_tokens","busy_percent")}) if hourly else {}
        return dict(demand=ds,series=ss,streams=exec_data,peaks=peaks,
                    seconds=3600 if hourly else 300)
    peak_bin = int(np.argmax(demand["input"][:max(1,math.floor(main_end/300000))]))
    busy_bin = int(np.argmax(series["unfinished_requests"][:max(1,math.floor(main_end/300000))]))
    data = dict(hour=hour,detail=detail,count=count,main_days=float(main_end/86400000),
                whole_days=float(whole_end/86400000),peak=dict(start_day=peak_bin/288,end_day=(peak_bin+1)/288),
                busy_day=busy_bin//288,requests=len(a),main_request_share=float(np.mean(a<main_end)),
                source=str(pred),chart_ready=True,format=2)
    info = dict(**windows,mean_ttft_ms=header["mean_ttft_ms"],p99_ttft_ms=header["p99_ttft_ms"],
                mean_tpot_ms=None if row.get("output_token_limit") else header["mean_tpot_ms"],
                sla_percent=None if row.get("output_token_limit") else header["goodput_completed_requests"]/607190*100,
                reuse_ratio=header["first_admission_prefix_cache_reused_ratio"],
                ssd_read_tb=header["g3_offload"]["read"]["completed_bytes"]/1e12,
                reconciliation=observation["reconciliation"])
    script(ASSETS/hour, "hour", {**data, **view(True), "info":info})
    script(ASSETS/detail, "detail", view(False))
    write(meta_path, json.dumps(data, separators=(",", ":")))
    print("Ready", key, flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    template = Path(__file__).with_name("throughput_minimax_dashboard.html")
    published = {}
    failed = set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {}
        while True:
            index, sources = cases()
            for key, future in list(jobs.items()):
                if future.done():
                    try:
                        published[key] = future.result()
                    except Exception as exc:
                        print("Failed", key, repr(exc), flush=True)
                        failed.add(key)
                    del jobs[key]
            for key, row in index.items():
                row.update(published.get(key, {}))
                if key in failed:
                    row["aggregation_failed"] = True
            entries = sorted(index.values(), key=lambda x: (x["factor"],x["dram"],x["read"]))
            script(ASSETS/"index.js", "index", entries)
            if template.exists():
                write(BASE/"throughput_minimax.html", template.read_text())
            candidates = [key for key in sources if key not in published and key not in jobs and key not in failed]
            candidates.sort(key=lambda key: (index[key]["dram"]!=256, index[key]["read"]!=2, index[key]["factor"], key))
            for key in candidates[:max(0,2-len(jobs))]:
                jobs[key] = pool.submit(aggregate,index[key],*sources[key])
            terminal = not any(r["status"] in ("运行中","待启动") for r in entries)
            if not jobs and not candidates and (terminal or not args.watch):
                print(f"Done: {len(published)} configurations, {len(failed)} failures", flush=True)
                break
            time.sleep(5)


if __name__ == "__main__":
    main()
