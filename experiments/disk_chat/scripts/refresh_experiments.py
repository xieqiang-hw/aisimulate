#!/usr/bin/env python3
"""Refresh chat experiment MD/HTML from full baseline runs; never run simulations."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import yaml

from prepare_chat import DATASETS, sha256
from build_chat_dashboard import render as render_dashboard, template as dashboard_template

BASE = Path(__file__).resolve().parents[1]
ROOT = Path.home() / "project/results/chat-glm52-h200-baseline"
SWEEP_ROOT = Path.home() / "project/results/chat-glm52-h200-sweep"
LABELS = {"completed": "已完成", "pending": "待启动", "running": "运行中",
          "failed": "失败", "interrupted": "已中断", "skipped_completed": "复用已完成",
          "invalid": "结果校验失败"}
FIELDS = {
    "hbm": "first_admission_g1_reused_input_tokens",
    "dram": "first_admission_g2_direct_reused_input_tokens",
    "ssd": "first_admission_g3_reused_input_tokens",
}
METRICS = [
    ("sla", "联合 SLA 达标率", "%"), ("p99_ttft", "P99 TTFT", "秒"),
    ("mean_ttft", "平均 TTFT", "秒"), ("tpot", "平均 TPOT", "ms"),
    ("adjusted_mean", "扣排队后平均 TTFT", "秒"), ("adjusted_p99", "扣排队后 P99 TTFT", "秒"),
    ("hbm", "HBM 复用率", "%"), ("dram", "DRAM 直接复用率", "%"),
    ("ssd", "SSD 恢复复用率", "%"), ("reuse", "总前缀复用率", "%"),
    ("ssd_read", "SSD 累计读", "TB"), ("wall_seconds", "实际耗时", "秒"),
    ("memory", "内存峰值", "GB"), ("rps", "完成请求吞吐", "请求/秒"),
]
COLUMNS = [("dataset", "数据集"), ("status", "状态"), ("completed", "完成请求"),
           *[(key, f"{label}（{unit}）") for key, label, unit in METRICS[:13]]]


def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summary_prefix(path):
    """The engine emits a summary followed by a potentially huge per_request list."""
    text = ""
    with path.open(encoding="utf-8") as stream:
        while chunk := stream.read(65536):
            text += chunk
            match = re.search(r'"per_request"\s*:', text)
            if match:
                return json.loads(text[:match.start()].rstrip().rstrip(",") + "}")
            if len(text) > 4_000_000:
                raise ValueError("prediction summary prefix exceeds 4 MB")
    return json.loads(text)


def percentile99(values):
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * 0.99 + 0.5)] if ordered else None


def request_metrics(path, expected):
    totals, coverage = Counter(), Counter()
    adjusted = []
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("terminal_status", "completed") != "completed":
                raise ValueError("per-request output contains non-completed requests")
            length = row["input_length"]
            if type(length) is not int or length <= 0:
                raise ValueError("invalid per-request input_length")
            totals["input"] += length
            for key, field in FIELDS.items():
                if field in row:
                    value = row[field]
                    if type(value) is not int or not 0 <= value <= length:
                        raise ValueError(f"invalid {field}")
                    totals[key] += value
                    coverage[key] += 1
            value = row.get("prefill_wait", {}).get("ttft_excluding_prefill_scheduling_ms")
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("invalid adjusted TTFT")
                adjusted.append(value)
            count += 1
    if count != expected:
        raise ValueError(f"request detail count {count} != completed count {expected}")
    result = {key: 100 * totals[key] / totals["input"] if coverage[key] == count else None for key in FIELDS}
    result["reuse"] = sum(result[key] for key in FIELDS) if all(result[key] is not None for key in FIELDS) else None
    if result["reuse"] is not None and result["reuse"] > 100 + 1e-9:
        raise ValueError("first-admission reuse layers exceed input tokens")
    result.update(input_tokens=totals["input"], adjusted_coverage=len(adjusted),
                  adjusted_mean=math.fsum(adjusted) / count / 1000 if len(adjusted) == count else None,
                  adjusted_p99=percentile99(adjusted) / 1000 if len(adjusted) == count else None)
    return result


def completed_metrics(run_path, expected, cache_dir):
    run = read_json(run_path)
    if run.get("dry_run") or run.get("status") != "completed" or run.get("returncode") != 0:
        raise ValueError("not a successful real replay")
    if run.get("max_requests") != 0 or run.get("selected_requests") != expected:
        raise ValueError("not a full dataset replay")
    prediction = run_path.parent / "prediction"
    paths = [run_path, prediction / "prediction.json", prediction / "requests.jsonl",
             prediction / "resource-runtime.json", Path(__file__)]
    signature = [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]
    key = hashlib.sha256(str(run_path).encode()).hexdigest()[:24]
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        cache = read_json(cache_path)
        if cache.get("signature") == signature:
            return cache["metrics"]
    summary = summary_prefix(prediction / "prediction.json")
    count = summary["completed_requests"]
    if count != expected or count <= 0:
        raise ValueError(f"completed {count} != full dataset size {expected}")
    result = request_metrics(prediction / "requests.jsonl", count)
    resource = read_json(prediction / "resource-runtime.json")
    if resource.get("status") != "completed" or resource.get("exit_code") != 0:
        raise ValueError("resource monitor did not report successful termination")
    peak = resource.get("peak_observed_rss_bytes")
    memory_budget = resource.get("budget", {}).get("memory_limit_bytes")
    read_bytes = summary.get("g3_offload", {}).get("read", {}).get("completed_bytes")
    duration_ms = summary.get("duration_ms")
    result.update(
        completed=count, sla=100 * summary["goodput_completed_requests"] / count,
        p99_ttft=summary["p99_ttft_ms"] / 1000,
        mean_ttft=summary["mean_ttft_ms"] / 1000, tpot=summary["mean_tpot_ms"],
        ssd_read=None if read_bytes is None else read_bytes / 1e12,
        wall_seconds=run.get("wall_seconds"), memory=None if peak is None else peak / 1e9,
        memory_budget_gb=None if memory_budget is None else memory_budget / 1e9,
        memory_budget_mode=resource.get("requested_resources", {}).get("memory_limit_gb"),
        rps=summary.get("request_throughput_rps"),
        simulation_seconds=None if duration_ms is None else duration_ms / 1000,
    )
    atomic(cache_path, json.dumps({"signature": signature, "metrics": result}, indent=2) + "\n")
    return result


def child_path(case, sweep_path):
    if case.get("status") == "skipped_completed":
        return Path(case["prior_run_json"])
    if case.get("run_json"):
        return Path(case["run_json"])
    candidates = sorted((sweep_path.parent / "cases" / case["name"]).glob("*/run.json"))
    return candidates[-1] if candidates else None


def collect(paths, cache_dir):
    chosen, warnings, sources = {}, [], []
    for sweep_path in sorted(paths):
        try:
            plan = read_json(sweep_path)
            if plan.get("schema") != "chat-baseline-v1" or plan.get("dry_run") or plan.get("max_requests") != 0:
                continue
            sources.append(str(sweep_path))
            for case in plan["cases"]:
                if case["dataset"] not in DATASETS:
                    continue
                config_path = Path(case["config_path"])
                config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
                budget = config.get("engine", {}).get("workers", {}).get("prefill", {}).get("scheduler", {}).get("max_batched_tokens")
                key = (case["dataset"], case["p_dram_gb"], case["read_gbps_per_tp"], plan["speedup"], budget)
                stamp = plan.get("created_utc", sweep_path.parent.name)
                if key not in chosen or stamp >= chosen[key][1].get("created_utc", chosen[key][0].parent.name):
                    chosen[key] = (sweep_path, plan, case)
        except (OSError, ValueError, KeyError) as error:
            warnings.append(f"{sweep_path}: {error}")
    rows = []
    for sweep_path, plan, case in chosen.values():
        row = {"dataset": case["dataset"], "status_code": case["status"],
               "status": LABELS.get(case["status"], case["status"]),
               "expected": case["request_count"], "sessions": case.get("session_count"),
               "span_seconds": case.get("span_seconds"), "p_dram_gb": case["p_dram_gb"],
               "read_gbps": case["read_gbps_per_tp"], "speedup": plan["speedup"],
               "sweep": str(sweep_path), "bundle": case["bundle"], "error": case.get("error"),
               "run_json": None, "values": {}, "config_path": case.get("config_path")}
        try:
            manifest_path = Path(case["bundle"]) / "manifest.json"
            manifest = read_json(manifest_path)
            if sha256(manifest_path) != case["manifest_sha256"]:
                raise ValueError("source manifest changed since the run")
            row.update(arrival_kind=manifest["arrival_kind"], source_block_size=manifest["source_block_size"],
                       source_requests=manifest["diagnostics"]["source_requests"],
                       filtered_requests=manifest["diagnostics"]["source_requests"] - manifest["request_count"])
            config = yaml.safe_load(Path(case["config_path"]).read_text())
            row["budget"] = config["engine"]["workers"]["prefill"]["scheduler"]["max_batched_tokens"]
            run_path = child_path(case, sweep_path)
            if run_path and run_path.exists():
                run = read_json(run_path)
                row["run_json"] = str(run_path)
                if run.get("dry_run"):
                    raise ValueError("dry-run cannot supply full experiment results")
                if run.get("status") == "completed":
                    row["values"] = completed_metrics(run_path, case["request_count"], cache_dir)
                    row.update(status_code="completed", status="已完成")
                elif run.get("status") in LABELS:
                    row.update(status_code=run["status"], status=LABELS[run["status"]])
            elif case["status"] in ("completed", "skipped_completed"):
                raise ValueError("completed case has no run.json")
        except (OSError, ValueError, KeyError) as error:
            row.update(status_code="invalid", status=LABELS["invalid"], error=str(error))
            row["values"] = {}
        rows.append(row)
    rank = {name: index for index, name in enumerate(DATASETS)}
    rows.sort(key=lambda row: (rank[row["dataset"]], row["budget"] if "budget" in row else 0,
                               row["speedup"], row["p_dram_gb"], row["read_gbps"]))
    return rows, warnings, sources


def fmt(value, digits=2):
    if value is None:
        return "—"
    if isinstance(value, str):
        return value.replace("|", "\\|").replace("\n", " ")
    if type(value) is int:
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "".join(
        "| " + " | ".join(fmt(value) for value in row) + " |\n" for row in rows)


def result_cell(row, key):
    if key in ("dataset", "status"):
        return row[key]
    value = row["values"].get(key)
    if value is not None and key in ("sla", "hbm", "dram", "ssd", "reuse", "ssd_read", "memory"):
        return f"{value:,.3f}"
    return value


def markdown(data):
    rows = data["rows"]
    counts = Counter(row["status"] for row in rows)
    text = f"# GLM-5.2 · Chat 数据集 baseline 实验\n\n更新时间：{data['generated']}。" + "，".join(f"{key} {value} 组" for key, value in counts.items()) + "。\n\n"
    text += "## 1. 配置与统计口径\n\n"
    text += table(["项目", "配置"], data["config_info"])
    text += "\n仅读取正式全量 baseline；排除 smoke、dry-run 和 max-requests 非零的回放。每个数据集、DRAM、读带宽、speedup 组合采用最新一次正式尝试，不用旧成功值填补新的待运行或失败记录。\n\n"
    text += "扣排队 TTFT 只扣除 P 首次准入前调度等待，保留 KV 恢复、后续 chunk、交接与 D 等待；P99 沿用 SWE 的最近秩口径。分层复用为首次准入各层复用 token 总数 ÷ 输入 token 总数。GB/TB 为十进制，SSD 累计读包含重复恢复。内存峰值取 resource-runtime.json 的实测进程树 RSS，不是模拟的 DRAM 容量。缺失指标记为 —。\n\n"
    text += "## 2. 数据集与负载\n\n"
    text += table(["数据集", "会话数", "输入请求", "原始记录", "过滤请求", "跨度（秒）", "到达时间", "源块（token）"], [
        [row["dataset"], row["sessions"], row["expected"], row.get("source_requests"), row.get("filtered_requests"),
         row["span_seconds"], "合成 Poisson" if row.get("arrival_kind") == "synthetic_poisson" else "真实观测", row.get("source_block_size")]
        for row in rows])
    text += "\nQwen API/coder/thinking 和 Mooncake agent/synthetic 是独立对照组。Qwen 包保留原文件的请求类型，以发布的 token 长度模拟；不同数据集的请求量、长度及持续时间不同，跨数据集图表是描述性比较。Mooncake 未提供显式会话 ID。所有包均无真实 output_token_ids，不能等同于 SWE 的 D 输出 token 级复用。\n\n"
    text += "## 3. 实验结果\n\n"
    groups = sorted({(row.get("budget"), row["speedup"], row["p_dram_gb"], row["read_gbps"]) for row in rows}, key=str)
    for budget, speed, dram, read in groups:
        selected = [row for row in rows if (row.get("budget"), row["speedup"], row["p_dram_gb"], row["read_gbps"]) == (budget, speed, dram, read)]
        text += f"### 预算 {fmt(budget)} token · 原始时间轴 1× · speedup={speed:g} · DRAM {dram} GB · SSD 读 {read:g} GB/s/TP\n\n"
        text += table([label for _, label in COLUMNS], [
            [result_cell(row, key) for key, _ in COLUMNS] for row in selected]) + "\n"
    text += "## 4. 运行资源\n\n"
    text += table(["数据集", "内存预算模式", "实际预算（GB）", "RSS 峰值（GB）", "虚拟时长（秒）", "实际耗时（秒）"], [
        [row["dataset"], *[row["values"].get(key) for key in ("memory_budget_mode", "memory_budget_gb", "memory", "simulation_seconds", "wall_seconds")]] for row in rows])
    text += "\n## 5. 更新与来源\n\n"
    text += "- [交互 HTML](glm52_experiments.html)；[机器可读快照](analysis/model_reports/glm52_rows.json)。\n"
    text += "- 更新命令：`~/.venvs/aisim/bin/python /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/refresh_experiments.py`。刷新只读取结果，不启动仿真。\n"
    for source in data["sources"]:
        text += f"- 正式 sweep：`{source}`。\n"
    for row in rows:
        text += f"- {row['dataset']}：`{row['run_json'] or row['sweep']}`。\n"
        if row.get("error"):
            text += f"  - 状态说明：{fmt(row['error'])}。\n"
    for warning in data["warnings"]:
        text += f"- 读取警告：{fmt(warning)}。\n"
    return text


def refresh(results_root=None, sweeps=None, output_dir=BASE):
    state = output_dir / "analysis/model_reports"
    state.mkdir(parents=True, exist_ok=True)
    cache_dir = state / "metrics_cache"
    cache_dir.mkdir(exist_ok=True)
    with (state / "refresh.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        roots = [results_root] if results_root is not None else [ROOT, SWEEP_ROOT]
        paths = sweeps or [path for root in roots for path in root.glob("*/sweep.json")]
        rows, warnings, sources = collect(paths, cache_dir)
        if not rows:
            raise ValueError("no formal full baseline cases found; smoke/dry-run are excluded")
        info = [
            ["模型 / GPU", "zai-org/GLM-5.2-FP8 / H200 SXM"],
            ["部署 / 并行", "1P1D，P/D 各 8 卡；Attention TP8/DP1，MoE TP1/EP8"],
            ["性能模型", "vLLM 0.24.0，op_level，fallback=deny"],
            ["HBM KV", "自动容量，memory_fraction=0.9，64-token 块，FP8 DSA"],
            ["缓存", "P：HBM + 本地 DRAM + worker_local SSD；D：仅 HBM"],
            ["本地 SSD", "约 56 TB；baseline 读写各 8 GB/s/TP"],
            ["传输", "H2D/D2H 各 25 GB/s/TP；P→D 50 GB/s/TP，destination_missing"],
            ["调度", "Chunked Prefill；baseline P/D 预算均 16,384 token；序列上限 32/256"],
            ["联合 SLA", "TTFT ≤ 1 秒且单请求平均 TPOT ≤ 30 ms"],
            ["负载", "128K 输入+输出上限；冷缓存，零预热；原始时间轴 1×，baseline speedup=1"],
            ["运行内存", "默认自动预算；实际预算和 RSS 单独列出"],
        ]
        data = {"generated": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
                "rows": rows, "warnings": warnings, "sources": sources,
                "config_info": info, "metrics": METRICS, "columns": COLUMNS}
        atomic(state / "glm52_rows.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        atomic(output_dir / "glm52_experiments.md", markdown(data))
        atomic(Path(__file__).with_name("chat_experiments_dashboard.html"), dashboard_template())
        atomic(output_dir / "glm52_experiments.html", render_dashboard(data))
    print(json.dumps({"rows": len(rows), "statuses": dict(Counter(row["status"] for row in rows)),
                      "markdown": str(output_dir / "glm52_experiments.md"),
                      "html": str(output_dir / "glm52_experiments.html")}, ensure_ascii=False))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, help="override default baseline + sweep discovery")
    parser.add_argument("--sweep", type=Path, action="append", help="explicit formal sweep.json; repeatable")
    parser.add_argument("--output-dir", type=Path, default=BASE)
    args = parser.parse_args()
    try:
        refresh(args.results_root.expanduser().resolve() if args.results_root else None,
                [path.expanduser().resolve() for path in args.sweep] if args.sweep else None,
                args.output_dir.expanduser().resolve())
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"report: {error}\n")


if __name__ == "__main__":
    main()
