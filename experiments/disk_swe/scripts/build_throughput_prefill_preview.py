"""Build the standalone throughput preview with per-configuration diagnostics."""
import gzip
import json
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ANALYSIS = BASE / "analysis"
CACHE = Path.home() / "project/tmp/throughput_prefill_preview"
CACHE.mkdir(parents=True, exist_ok=True)


def request(row):
    wait = row.get("prefill_wait") or row
    arrival = row["arrival_time_ms"]
    ttft = row.get("ttft_ms")
    if ttft is None and row.get("first_token_ms") is not None:
        ttft = row["first_token_ms"] - arrival
    adjusted = wait.get("ttft_excluding_prefill_scheduling_ms")
    schedule = wait.get("scheduling_wait_ms")
    if adjusted is None and ttft is not None and schedule is not None:
        adjusted = max(0, ttft - schedule)
    return [
        arrival, row.get("prefill_admit_ms"), ttft, adjusted,
        row.get("source_held_ms"), row.get("destination_reserved_ms"),
        row.get("source_released_ms"), row.get("uuid", "")[:8],
    ]


def batch(row):
    return [row[2], row[3], row[4], row[6], row[5]]


def extract(factor, pred, lo, hi):
    files = [pred / n for n in
             ("iteration_metrics.jsonl.gz", "scheduler_states.jsonl.gz", "requests.jsonl")]
    arrays = ANALYSIS / "throughput_glm52" / f"requests_{factor}x.npz"
    identity = [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in [*files, arrays]]
    key = [2, lo, hi, identity]
    cache = CACHE / f"{factor}x.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["key"] == key:
            return saved["data"]
    batches, requests, states = [], [], []
    print(f"{factor}× extracting batches", flush=True)
    with gzip.open(files[0], "rb") as stream:
        for line in stream:
            if not line.startswith(b"[0,"):
                continue
            row = json.loads(line)
            if row[2] > hi:
                break
            if row[4] > lo and row[5] > 0:
                batches.append(batch(row))
    print(f"{factor}× extracting states and requests", flush=True)
    prior = {}
    with gzip.open(files[1], "rb") as stream:
        for line in stream:
            sid, _, at, _ = line.split(b",", 3)
            t = float(at)
            if t > hi:
                break
            if t < lo:
                prior[sid] = line
                continue
            states.append(json.loads(line))
    states = [json.loads(line) for line in prior.values()] + states
    with np.load(arrays) as series:
        arrival, admit = series["arrival_time_ms"], series["prefill_admit_ms"]
    indexes = np.flatnonzero((arrival <= hi) & ((arrival >= lo) | (admit >= lo))).tolist()
    positions = iter(indexes)
    target = next(positions, None)
    with files[2].open("rb") as stream:
        for i, line in enumerate(stream):
            if i != target:
                continue
            row = json.loads(line)
            assert row["arrival_time_ms"] == arrival[i] and row["prefill_admit_ms"] == admit[i]
            requests.append(request(row))
            target = next(positions, None)
            if target is None:
                break
    assert target is None and batches and requests
    data = dict(batches=batches, requests=requests, states=states)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(key=key, data=data), separators=(",", ":")))
    tmp.replace(cache)
    return data


def main():
    old = ANALYSIS / "prefill_queue_diagnosis"
    original = json.loads((old / "extended_window_data.json").read_text())
    cases = {
        "8": dict(
            domain=[original["start_ms"], original["end_ms"]],
            window=[682800000, 683100000],
            batches=[batch(row) for row in original["iterations"] if row[0] == 0 and row[5] > 0],
            requests=[request(row) for row in original["requests"]],
            states=[[0, i, t, state] for t, i, state in original["states"]],
            cases=[
                dict(label="BS 未满，token 预算已用完", focus=[683087834.955, 683100433.885],
                     note="连续五轮的 BS 均为 1，每轮新算 16,384 token。单个长请求用完了整批 token 预算，后续请求仍在等待。"),
                dict(label="批次已提交，新请求等待下一轮", focus=[683014908.181, 683017350.820],
                     note="尾段仅新算 974 token，BS 为 1。三条请求在该批窗口附近到达，已提交的批次无法再插入请求。此例沿用原诊断中的调度时序核验。"),
            ],
        )
    }
    for factor in (16, 32):
        report = json.loads((ANALYSIS / "glm_pd_backpressure" / f"overlay_{factor}x.json").read_text())
        episode = report["main"]["top_idle_queue_episodes"][0]
        start, end = episode["start_ms"], episode["end_ms"]
        # Place the entire example inside a five-minute window.
        center = (start + end) / 2
        domain = [center - 3600000, center + 3600000]
        pred = Path(report["prediction"])
        data = extract(factor, pred, *domain)
        focus = [start - 3000, end + 3000]
        note = (f"这个片段中 P 约 {episode['duration_s']:.2f} 秒未执行计算，"
                f"普通准入队列最多 {episode['p_queue_max']} 条。"
                f"同时观测到 P 保留交接 KV、D 等待预留，"
                f"P 恢复待办重叠 {episode['restore_pending_s']:.3f} 秒。")
        if factor == 16:
            note += "已核验的一条源端请求等待 D 预留约 32.56 秒。其源端 KV 释放约 19.56 ms 后，另一条等待请求首次准入 P。"
        else:
            note += "本片段恢复待办与 D 待预留并存，时序可用来观察两者重叠，单独的延迟贡献尚未量化。"
        cases[str(factor)] = dict(
            **data, domain=domain, window=[center - 150000, center + 150000],
            cases=[dict(label="等待 D 预留与 P 准入" if factor == 16 else "D 待预留与 KV 恢复同时出现",
                        focus=focus, note=note)],
        )

    for factor, arrival in [("8", 682286635), ("16", 840095971)]:
        data = cases[factor]
        row = next(r for r in data["requests"] if r[0] == arrival and r[2] is not None
                   and r[3] is not None and r[2] - r[3] > 5000)
        wait = (row[2] - row[3]) / 1000
        data["cases"].append(dict(
            label="单请求的首次准入等待", focus=[arrival - 2000, row[1] + 3000],
            request_id=row[7],
            note=f"请求 {row[7]} 的 TTFT 为 {row[2]/1000:.3f} 秒，扣排队后为 "
                 f"{row[3]/1000:.3f} 秒，扣除的首次准入前调度等待为 {wait:.3f} 秒。"
                 "可在批次与等待曲线上对照它到达至首次准入的时序。",
        ))

    data = cases["32"]
    start, end = 155225090.48151055, 155234815.96393454
    full = [r for r in data["batches"] if start <= r[1] < end]
    assert len(full) == 4 and all(r[3] == 16384 for r in full)
    assert [r[4] for r in full] == [1, 1, 1, 10]
    data["cases"].append(dict(
        label="连续四轮用满 token 预算", focus=[start - 2000, end + 2000],
        note="连续四轮各新算 16,384 token，BS 分别为 1、1、1、10。"
             "这段计算中普通准入队列非空，序列数仍低于 32。",
    ))
    start, end = 154111104.79593953, 154114835.91415253
    state = next(row[3] for row in data["states"] if row[0] == 0 and row[2] == start)
    assert state["waiting_admission_requests"] == 33 and state["source_held_requests"] == 0
    d_state = max((row for row in data["states"] if row[0] == 1 and row[2] <= start),
                  key=lambda row: row[2])[3]
    assert d_state["pending_destination_requests"] == 0
    data["cases"].append(dict(
        label="P 有恢复待办，D 没有待预留", focus=[start - 2500, end + 2500],
        note="片段起点的 P 快照记录了 33 条普通队列等待、22 条连接器等待、"
             "2 条 H2D 加载和 12 个 SSD 待读任务。此时 P 没有保留交接 KV，最近一条 D 快照"
             "也没有待预留请求。这个片段可与 D 待预留和恢复同时出现的案例对照。",
    ))

    for data in cases.values():
        for item in data["cases"]:
            center = sum(item["focus"]) / 2
            start = max(data["domain"][0], min(center - 150000, data["domain"][1] - 300000))
            item["window"] = [start, start + 300000]

    # Keep compact state counts for causal context and discard other snapshot fields.
    for data in cases.values():
        compact = {0: [], 1: []}
        last = {}
        for sid, _, t, state in sorted(data["states"], key=lambda row: row[2]):
            if sid == 0:
                values = [state.get(k) or 0 for k in ("waiting_admission_requests",
                          "source_held_requests", "connector_waiting_requests",
                          "h2d_loading_requests", "g3_pending_read_jobs")]
            else:
                values = [state.get("pending_destination_requests") or 0]
            if last.get(sid) != values:
                compact[sid].append([t, *values])
                last[sid] = values
        data["states"] = compact

    source = (BASE / "throughput_glm52.html").read_text()
    ui = Path(__file__).with_name("throughput_prefill_preview.html").read_text()
    payload = json.dumps(cases, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    ui = ui.replace("__PREFILL_CASES__", payload)
    marker = '<section id="execution">'
    assert source.count(marker) == 1
    rendered = source.replace(marker, ui + "\n" + marker)
    rendered = rendered.replace('<a href="#execution">', '<a href="#prefillLocal">局部诊断</a><a href="#execution">', 1)
    rendered = rendered.replace('<h2>3. P/D 执行状态</h2>', '<h2>4. P/D 执行状态</h2>')
    rendered = rendered.replace('<h2>4. 当前窗口汇总</h2>', '<h2>5. 当前窗口汇总</h2>')
    target = BASE / "throughput_glm52_prefill_preview.html"
    tmp = target.with_suffix(".html.tmp")
    tmp.write_text(rendered)
    tmp.replace(target)
    print("Saved", target, [(f, len(v["batches"]), len(v["requests"])) for f, v in cases.items()])


if __name__ == "__main__":
    main()
