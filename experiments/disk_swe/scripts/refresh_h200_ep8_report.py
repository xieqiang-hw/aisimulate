"""Render or watch the H200 EP8 16x local SSD sweep reports."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from build_results_dashboard import build
from result_metrics import result_cells

BASE = Path(__file__).resolve().parents[1]
CASE = re.compile(r"swe_h200_ep8_p(\d+)gb_read(\d+)_overlay_(\d+)x")
ADJUSTED_TTFT = re.compile(rb'"ttft_excluding_prefill_scheduling_ms"\s*:\s*([0-9.eE+-]+)')
LABELS = {"pending": "待启动", "running": "运行中", "failed": "失败", "interrupted": "已中断"}
COLUMNS = ("状态", "完成请求", "联合 SLA 达标率", "P99 TTFT（秒）", "平均 TTFT（秒）",
           "TPOT（ms）", "HBM 复用", "DRAM 直接复用", "SSD 恢复复用", "总复用率",
           "SSD 累计读", "实际耗时", "扣排队后平均 TTFT（秒）", "扣排队后 P99 TTFT（秒）",
           "内存峰值（GB）")


SHRINK_ANALYSIS = '## 2. 32× 下缩容后扣排队 TTFT 降低的原因\n\n以 SSD 读带宽 1 GB/s／TP 分片为例，32× 高负载下，DRAM 从 128 GB 缩到 64 GB，不仅减少了能保留的前缀缓存，也压缩了 SSD 恢复 KV 的落地缓冲空间。多个请求争用这部分空间时，恢复可能因预留失败而停止，已恢复的块也可能在使用前被挤掉；由于前缀复用要求连续匹配，部分块缺失就会缩短最终可复用的前缀。恢复和执行时序的变化还会影响请求准入时的 HBM 命中情况，因此本次总复用率从 84.428% 降到 74.186%，更多输入需要重新计算，增加 P 侧处理压力并加重排队。与此同时，每条请求实际用上的 SSD 恢复 KV 减少，计入的平均 KV 恢复等待从 6.87 秒降到 3.49 秒，使“扣排队后平均 TTFT”从 11.80 秒降到 8.51 秒；但被扣除的调度等待大幅增加，原始平均 TTFT 反而从约 2.7 万秒升到 19.9 万秒。现有代码和统计支持这条解释，各环节的具体贡献仍需进一步量化。\n'

def finished_cells(run):
    path = next(run.glob("*/run.json"))
    info = json.loads(path.read_text(encoding="utf-8"))
    if info["status"] != "completed":
        raise ValueError(f"Inconsistent completed case: {path}")
    values = result_cells(path, info["wall_seconds"], info["selected_requests"])
    prediction = path.parent / "prediction"
    with (prediction / "requests.jsonl").open("rb") as stream:
        times = [float(match[1]) for line in stream if (match := ADJUSTED_TTFT.search(line))]
    times.sort()
    if times:
        rank = math.floor((len(times) - 1) * 0.99 + 0.5)
        values += [f"{math.fsum(times) / len(times) / 1000:.2f}", f"{times[rank] / 1000:.2f}"]
    else:
        values += ["—", "—"]
    resource_path = prediction / "resource-runtime.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8")) if resource_path.exists() else {}
    peak = resource.get("peak_observed_rss_bytes")
    return values + [f"{peak / 1e9:.3f}" if peak is not None else "—"]


def render(sweep, plan, cache, stopped):
    parent = Path(plan["bundle"])
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    overlays = sorted({Path(c["bundle"]) for c in plan["cases"]})
    densities = []
    for overlay in overlays:
        meta = json.loads((overlay / "manifest.json").read_text(encoding="utf-8"))["overlay"]
        densities.append(f'{meta["factor"]}× → {meta["effective_density_multiplier"]:.3f}×')
    capacities = "／".join(str(n) for n in sorted(
        {int(CASE.fullmatch(c["name"])[1]) for c in plan["cases"]}, reverse=True))
    counts = Counter(c["status"] for c in plan["cases"])
    table = [["标称叠加倍率", "P 侧 DRAM", "SSD 读带宽（GB/s／TP 分片）", *COLUMNS]]
    table.append(["---"] * len(table[0]))
    failures = []
    for case in plan["cases"]:
        match = CASE.fullmatch(case["name"])
        if not match:
            raise ValueError(f"Unexpected case: {case['name']}")
        dram, bandwidth, factor = match.groups()
        status = case["status"]
        run = Path(case["sweep"]) / case["suite"] / case["name"]
        if status == "completed":
            if case["name"] not in cache:
                cache[case["name"]] = finished_cells(run)
            values = cache[case["name"]]
        else:
            state = "已中断" if stopped and status == "running" else LABELS[status]
            values = [state, *(["—"] * (len(COLUMNS) - 1))]
        table.append([f"{factor}×", f"{dram} GB", bandwidth, *values])
        if status == "failed" and case.get("log"):
            logs = sorted(run.glob("*/run.log")) or [Path(case["log"])]
            lines = logs[-1].read_text(errors="replace").splitlines()
            reason = next((line for line in reversed(lines) if line.strip()), "查看运行日志")
            failures.append(f"- `{case['name']}` · {reason}")

    body = "\n".join("| " + " | ".join(row) + " |" for row in table)
    note = " 调度进程已退出，未完成的组将在本报告中保留其最后状态。" if stopped else ""
    diagnostics = "\n\n## 失败记录\n\n" + "\n".join(failures) if failures else ""
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %z")
    return f"""# SWE-chat 场景 1 · H200 EP8 本地 SSD 扫描

<!-- report:ep8-local-ssd -->

新 SWE-chat Claude Code 数据集，128K 上下文。每组 {manifest["request_count"]:,} 条请求，标称会话叠加倍率对应的全程平均到达密度为 {"，".join(densities)}，按 trace 时间戳回放。

GLM-5.2-FP8，H200 SXM，1P1D。P/D 分别为 Attention TP8／DP1、MoE TP1／EP8，自动 HBM KV 容量。P 侧 DRAM 为 {capacities} GB，SSD 约 56 TB。D 侧仅使用 HBM。SSD 写带宽固定 8 GB/s，H2D 和 D2H 各 25 GB/s，均按每 TP 分片计。P/D 单步 token 预算均为 16,384，序列上限分别为 32／256。{len(plan["cases"])} 组串行，使用 AISim 自动内存预算。

联合 SLA 为 TTFT ≤ 1 秒且单请求平均 TPOT ≤ 30 ms。扣排队后的两列使用逐请求 TTFT 扣除 P 首次准入前调度等待后汇总，保留 KV 恢复等待。各层复用按首次准入复用 token 占全部输入 token 的比例计算。SSD 累计读包含重复恢复。内存峰值为进程树 RSS，采用十进制 GB。

<!-- results:status:start -->
完成 {counts["completed"]}／{len(plan["cases"])} 组，运行 {0 if stopped else counts["running"]} 组，失败 {counts["failed"]} 组，待启动 {counts["pending"]} 组。最后更新 {now}。{note}
<!-- results:status:end -->

数据目录 `{parent}`  
结果目录 `{sweep}`

## 1. DRAM × SSD 读带宽扫描

{body}

{SHRINK_ANALYSIS}{diagnostics}
"""


def process_identity(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return stat[19] if stat[0] != "Z" else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--include-sweep", type=Path, action="append", default=[])
    parser.add_argument("--watch-pid", type=int, help="Refresh on each status change until this sweep exits")
    args = parser.parse_args()
    sweep = args.sweep.expanduser().resolve()
    roots = [*(p.expanduser().resolve() for p in args.include_sweep), sweep]
    paths = [root / "sweep.json" for root in roots]
    doc = BASE / "analysis/model_reports/source_documents/experiments_h200_ep8_16x.md"
    html = doc.with_suffix(".html")
    identity = process_identity(args.watch_pid) if args.watch_pid else None
    cache = {}
    previous = None
    while True:
        raw = tuple(path.read_text(encoding="utf-8") for path in paths)
        plans = [json.loads(value) for value in raw]
        plan = dict(plans[-1])
        if any(item["bundle"] != plan["bundle"] for item in plans):
            raise ValueError("Reports can only combine sweeps of the same base bundle")
        cases = {}
        for root, item in zip(roots, plans, strict=True):
            for case in item["cases"]:
                cases[case["name"]] = dict(case, sweep=str(root))
        plan["cases"] = sorted(cases.values(), key=lambda case: (
            int(CASE.fullmatch(case["name"])[3]), -int(CASE.fullmatch(case["name"])[1]),
            int(CASE.fullmatch(case["name"])[2])))
        terminal = all(c["status"] in ("completed", "failed", "interrupted") for c in plan["cases"])
        stopped = bool(args.watch_pid and (identity is None or process_identity(args.watch_pid) != identity)
                       and not terminal)
        state = (raw, stopped)
        if state != previous:
            updated = render(sweep, plan, cache, stopped)
            temp = doc.with_suffix(".md.tmp")
            temp.write_text(updated, encoding="utf-8")
            temp.replace(doc)
            build(markdown_path=doc, output=html)
            from refresh_model_reports import refresh
            refresh("glm52")
            print(Counter(c["status"] for c in plan["cases"]), flush=True)
            previous = state
        if terminal or stopped or not args.watch_pid:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
