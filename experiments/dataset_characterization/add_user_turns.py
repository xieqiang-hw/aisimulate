"""Add observed human-input turns to existing replay and upstream reports."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from xml.etree import ElementTree

import pyarrow.parquet as pq

from characterize_datasets import DATA, report, stats
from characterize_upstream import render


def summarize(counts, method, matched=None):
    histogram = Counter(counts)
    bounds = ((0, 0), (1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float("inf")))
    distribution = [sum(count for turns, count in histogram.items() if lo <= turns <= hi) for lo, hi in bounds]
    result = dict(stats=stats(histogram), distribution=distribution, method=method)
    if matched is not None:
        result["matched_requests"] = matched
    assert sum(distribution) == result["stats"]["n"]
    return result


def sidecar(bundle):
    with (Path(bundle) / "source_requests.jsonl").open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def free_turns(root, bundle):
    selected = defaultdict(dict)
    for row in sidecar(bundle):
        selected[row["week"]][row["source_row_index"]] = (
            row["session_id"], row["source_model"], row["step"])
    original, replay_groups = [], defaultdict(set)
    matched = 0
    for path in sorted(root.glob("release/*/processed/requests.parquet")):
        week = path.parents[1].name
        sessions = defaultdict(list)
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=["session_id", "step", "model", "step_trigger"]):
            for row in batch.to_pylist():
                sid, step = row["session_id"], row["step"]
                retained = selected[week].get(row_index)
                if retained is not None and retained != (sid, row["model"], step):
                    raise ValueError(f"FreeInference source mismatch at {week}, row {row_index}")
                sessions[sid].append((step, row["step_trigger"], retained))
                row_index += 1
        for sid, calls in sessions.items():
            user_turn = 0
            for step, trigger, retained in sorted(calls, key=lambda call: call[0]):
                if trigger == "user":
                    user_turn += 1
                if retained is not None:
                    turns = replay_groups[(week, sid, retained[1])]
                    if user_turn:
                        turns.add(user_turn)
                    matched += 1
            original.append(user_turn)
    if matched != sum(map(len, selected.values())):
        raise ValueError("FreeInference replay sidecar is only partially matched")
    return (
        summarize(original, "每个上游周内会话按 step 顺序统计 step_trigger=user 的请求。工具触发的请求归入最近一次用户输入。"),
        summarize(map(len, replay_groups.values()),
                  "通过周次和原表行号对齐回放请求。只计至少关联一条保留请求的用户输入，同一用户输入关联多次调用只计一轮。0 表示匹配到调用但原记录没有可归属的用户输入。", matched),
    )


def swe_turns(root, bundle):
    retained = defaultdict(Counter)
    for row in sidecar(bundle):
        retained[row["session_id"]][row["response_id"]] += 1
    prompts, response_positions = defaultdict(set), defaultdict(dict)
    path = root / "conversations.parquet"
    columns = ["session_id", "turn_number", "turn_type", "response_id"]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=65536):
        for row in batch.to_pylist():
            sid, order, rid = row["session_id"], row["turn_number"], row["response_id"]
            if row["turn_type"] == "user_prompt":
                prompts[sid].add(order)
            if rid and rid in retained.get(sid, ()):
                positions = response_positions[sid]
                positions[rid] = min(order, positions.get(rid, order))
    replay_counts = []
    matched = 0
    for sid, positions in response_positions.items():
        user_orders = sorted(prompts.get(sid, ()))
        seen = {bisect_right(user_orders, position) for position in positions.values()}
        replay_counts.append(len(seen - {0}))
        matched += sum(retained[sid][rid] for rid in positions)
    original_sessions = pq.read_table(root / "sessions.parquet", columns=["session_id"])
    session_ids = set(original_sessions["session_id"].to_pylist())
    if not prompts.keys() <= session_ids:
        raise ValueError("SWE-chat user prompt session missing from sessions index")
    return (
        summarize((len(prompts.get(sid, ())) for sid in session_ids),
                  "按上游 sessions 索引统计每个会话的 user_prompt 事件。每条人工提示计一轮，工具回传不计。"),
        summarize(replay_counts, "以 session_id 和 response_id 匹配回放调用与主会话事件，把匹配的调用归入前一条 user_prompt。同一提示只计一次。仅覆盖至少匹配一条调用的回放会话，0 表示已有匹配调用但找不到更早的提示。", matched),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path(__file__).resolve().parent / "reports")
    parser.add_argument("--free-root", type=Path, default=DATA / "freeinference_agentic_trace/upstream")
    parser.add_argument("--swe-root", type=Path, default=DATA / "swe_chat/upstream")
    args = parser.parse_args()
    replay_path = args.reports / "dataset_characteristics.json"
    raw_path = args.reports / "upstream_characteristics.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    upstream = json.loads(raw_path.read_text(encoding="utf-8"))
    datasets = replay["datasets"]
    for data, source, analyze, root in zip(datasets, upstream, (free_turns, swe_turns),
                                           (args.free_root, args.swe_root)):
        source["user_turns"], data["user_turns"] = analyze(root, data["bundle"])
        print(data["name"], data["user_turns"], flush=True)
    for data in datasets + upstream:
        for key in ("turns", "raw_input"):
            if key in data:
                groups = data[key]["groups"]
                data[key]["groups"] = {int(k): value for k, value in groups.items()}
    for path, payload, page in (
        (replay_path, replay, report(datasets)),
        (raw_path, upstream, render(upstream, args.free_root, args.swe_root)),
    ):
        charts = [ElementTree.fromstring(svg) for svg in re.findall(r"<svg\b.*?</svg>", page, re.S)]
        assert sum("用户交互轮数分布" in chart.get("aria-label", "") for chart in charts) == 1
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        path.with_suffix(".html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
