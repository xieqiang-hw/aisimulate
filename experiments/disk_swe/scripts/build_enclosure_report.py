"""Build offline reports for full compact enclosure experiments."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo
import yaml

BASE = Path(__file__).resolve().parents[1]
ROOT = Path.home() / "project/results/swe-chat-enclosure-h200-ep8-compact"
TOKEN_KEYS = ["first_admission_g1_reused_input_tokens",
              "first_admission_g2_direct_reused_input_tokens",
              "first_admission_g3_reused_input_tokens"]
COLUMNS = ["状态","完成请求","联合 SLA 达标率","P99 TTFT（秒）","平均 TTFT（秒）",
           "TPOT（ms）","HBM 复用","DRAM 直接复用","盘框恢复复用","总复用率",
           "盘框累计读（TB）","实际耗时","扣排队后平均 TTFT（秒）",
           "扣排队后 P99 TTFT（秒）","内存峰值（GB）","盘框累计写（TB）",
           "D 生成块写回（块）","P 复用 D 生成块（块次）"]
METRICS = [
 ("sla","联合 SLA 达标率","%"),("mean_ttft","平均 TTFT","秒"),
 ("p99_ttft","P99 TTFT","秒"),("adjusted_mean","扣排队后平均 TTFT","秒"),
 ("adjusted_p99","扣排队后 P99 TTFT","秒"),("tpot","TPOT","ms"),
 ("hbm","HBM 复用率","%"),("dram","DRAM 直接复用率","%"),
 ("enclosure","盘框恢复复用率","%"),("reuse","总前缀复用率","%"),
 ("enclosure_read","整框累计读取量","TB"),("enclosure_write","整框累计写入量","TB"),
 ("decode_writeback","D 生成块写回","块"),("pd_reuse","P 复用 D 生成块","块次"),
 ("memory","进程树内存峰值","GB")]
LABELS = {"running":"运行中","prepared":"待启动","completed":"已完成",
          "failed":"失败","interrupted":"已中断"}

def atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

def fmt(value, decimals=3, suffix=""):
    return "—" if value is None else f"{value:,.{decimals}f}{suffix}"

def summary_header(path):
    # Native latency and enclosure counters precede the large per_request array.
    with path.open() as stream:
        text = stream.read(262144).split('"per_request":', 1)[0]
    result = {}
    for key in ["completed_requests","goodput_completed_requests",
                "mean_ttft_ms","p99_ttft_ms","mean_tpot_ms","g3_offload"]:
        marker = '"' + key + '":'
        if marker in text:
            result[key] = json.JSONDecoder().raw_decode(text.split(marker,1)[1].lstrip())[0]
    if "completed_requests" not in result:
        raise ValueError("completed_requests missing from prediction header")
    return result

def completed_metrics(run, expected):
    prediction = run / "prediction"
    summary = summary_header(prediction / "prediction.json")
    total_input = records = 0
    totals = [0,0,0]
    adjusted = []
    with (prediction / "requests.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if any(key not in row for key in TOKEN_KEYS):
                raise ValueError("missing first-admission tier statistics")
            total_input += row["input_length"]
            for i,key in enumerate(TOKEN_KEYS):
                totals[i] += row[key] or 0
            value = (row.get("prefill_wait") or {}).get("ttft_excluding_prefill_scheduling_ms")
            if value is not None:
                adjusted.append(value)
            records += 1
    if records != expected or summary["completed_requests"] != expected or total_input <= 0:
        raise ValueError(f"full-run count mismatch: records={records}, expected={expected}")
    rates = [100*v/total_input for v in totals]
    result = dict(zip(["hbm","dram","enclosure"], rates))
    result.update(reuse=sum(rates),completed=expected,
        sla=100*summary["goodput_completed_requests"]/expected,
        mean_ttft=summary["mean_ttft_ms"]/1000,p99_ttft=summary["p99_ttft_ms"]/1000,
        tpot=summary.get("mean_tpot_ms"),adjusted_mean=None,adjusted_p99=None,memory=None,
        enclosure_read=None,enclosure_write=None,decode_writeback=None,pd_reuse=None)
    if adjusted:
        adjusted.sort()
        result["adjusted_mean"] = math.fsum(adjusted)/len(adjusted)/1000
        result["adjusted_p99"] = adjusted[math.floor((len(adjusted)-1)*.99+.5)]/1000
    io = summary.get("g3_offload") or {}
    for key,direction in [("enclosure_read","read"),("enclosure_write","write")]:
        count = (io.get(direction) or {}).get("completed_bytes")
        result[key] = count/1e12 if count is not None else None
    result["decode_writeback"] = io.get("decode_writeback_blocks")
    result["pd_reuse"] = io.get("prefill_decode_reuse_blocks")
    path = prediction / "resource-runtime.json"
    if path.exists():
        peak = json.loads(path.read_text()).get("peak_observed_rss_bytes")
        result["memory"] = peak/1e9 if peak is not None else None
    return result

def cells(row):
    v = row["values"]
    seconds = row.get("wall_seconds")
    elapsed = "—" if seconds is None else f"{round(seconds)//60} 分 {round(seconds)%60} 秒"
    return [row["status"],fmt(v.get("completed"),0),fmt(v.get("sla"),3,"%"),
      fmt(v.get("p99_ttft"),2),fmt(v.get("mean_ttft"),2),fmt(v.get("tpot"),2),
      *[fmt(v.get(k),3,"%") for k in ["hbm","dram","enclosure","reuse"]],
      fmt(v.get("enclosure_read")),elapsed,fmt(v.get("adjusted_mean"),2),
      fmt(v.get("adjusted_p99"),2),fmt(v.get("memory")),fmt(v.get("enclosure_write")),
      fmt(v.get("decode_writeback"),0),fmt(v.get("pd_reuse"),0)]

def collect(root, cache):
    rows = []
    archive = root / "bundle_manifest_archive.json"
    archived_manifests = json.loads(archive.read_text()) if archive.is_file() else {}
    for path in sorted(root.glob("*/run.json")):
        meta = json.loads(path.read_text())
        run = path.parent
        cfg_path = run / "resolved.yaml"
        bundle = Path(meta["bundle"])
        manifest_path = bundle / "manifest.json"
        if not cfg_path.exists():
            continue
        manifest = (json.loads(manifest_path.read_text()) if manifest_path.is_file()
                    else archived_manifests.get(str(bundle)))
        if manifest is None:
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        e = cfg["engine"]
        enclosure = e.get("kv_transfer",{}).get("enclosure")
        if (not enclosure or manifest.get("hash_ids_scope") != "input_output"
            or e["model"] != "zai-org/GLM-5.2-FP8" or meta.get("max_requests",0) != 0):
            continue
        p = e["workers"]["prefill"]
        per_tp = p["kv_cache"].get("bytes_per_token","auto")
        # Verified GLM-5.2 FP8 DSA layout; explicit overrides remain authoritative.
        per_tp = 60216 if per_tp == "auto" else int(per_tp)
        block_bytes = per_tp*p["kv_cache"]["block_size"]*p["parallelism"]["tensor"]
        host = p["kv_cache"].get("host_offload",{}).get("num_host_blocks",0)
        overlay = manifest.get("overlay",{})
        row = dict(id=run.name,source=str(run),
          budget=p["scheduler"]["max_batched_tokens"],factor=overlay.get("factor",1),
          speedup=cfg["traffic"]["load"].get("speedup",1),
          dram_gb=round(host*block_bytes/1e9),dram_actual_gb=host*block_bytes/1e9,
          read_gbps=enclosure["read_bandwidth_gbps"],write_gbps=enclosure["write_bandwidth_gbps"],
          enclosure_tb=enclosure["num_blocks"]*block_bytes/1e12,
          status=LABELS.get(meta.get("status"),"待启动"),values={},
          wall_seconds=meta.get("wall_seconds"),bundle=str(bundle),manifest=manifest,config=cfg)
        if meta.get("status") == "completed":
            files = [run/"prediction"/name for name in
                     ["prediction.json","requests.jsonl","resource-runtime.json"]]
            signature = json.dumps([(str(p),p.stat().st_size,p.stat().st_mtime_ns)
                                    for p in files if p.exists()])
            try:
                archived = run/"prediction/report_summary.json"
                if not files[1].is_file() and archived.is_file():
                    preserved = json.loads(archived.read_text())
                    if (preserved.get("report") != "enclosure" or preserved.get("source") != str(run)
                            or preserved["values"]["completed"] != meta["selected_requests"]):
                        raise ValueError("Preserved report summary does not match this run")
                    row["values"] = preserved["values"]
                else:
                    if signature not in cache:
                        cache[signature] = completed_metrics(run,meta["selected_requests"])
                    row["values"] = cache[signature]
            except (OSError,ValueError,KeyError) as error:
                row["status"] = "失败"
                row["error"] = f"报告校验失败：{error}"
        if meta.get("error"):
            row["error"] = meta["error"]
        if meta.get("status") in {"failed", "interrupted"}:
            resource = run/"prediction/resource-runtime.json"
            if resource.exists():
                usage = json.loads(resource.read_text())
                row["error"] = row.get("error") or usage.get("reason")
                peak = usage.get("peak_observed_rss_bytes")
                if peak is not None:
                    row["values"]["memory"] = peak/1e9
        row["cells"] = cells(row)
        rows.append(row)
    # Keep every attempt in the source index, but use one successful result
    # per configuration so an older failed attempt cannot mask a later retry.
    grouped = {}
    for row in rows:
        key = (row["budget"],row["factor"],row["speedup"],row["dram_gb"],
               row["read_gbps"],row["write_gbps"],row["bundle"],
               json.dumps(row["config"]["engine"],sort_keys=True))
        grouped.setdefault(key,[]).append(row)
    selected = []
    for attempts in grouped.values():
        successful = [r for r in attempts if r["status"]=="已完成"]
        chosen = dict((successful or attempts)[-1])
        chosen["attempts"] = [
            {k:r.get(k) for k in ["id","source","status","error","wall_seconds"]}
            for r in attempts]
        selected.append(chosen)
    return sorted(selected,key=lambda r:r["id"])

def config_info(rows):
    if not rows:
        raise ValueError("no full compact enclosure runs found")
    row = rows[-1]
    drams = "／".join(f"{v:g}" for v in sorted({r["dram_gb"] for r in rows},reverse=True))
    reads = "／".join(f"{v:g}" for v in sorted({r["read_gbps"] for r in rows},reverse=True))
    writes = "／".join(f"{v:g}" for v in sorted({r["write_gbps"] for r in rows},reverse=True))
    cfg,m = row["config"],row["manifest"]
    p,d = cfg["engine"]["workers"]["prefill"],cfg["engine"]["workers"]["decode"]
    overlay = m.get("overlay",{})
    timeline = (f"标称 {row['factor']}×；全程 {overlay['replay_span_ms']/86400000:.2f} 天，"
                f"全程平均密度 {overlay['effective_density_multiplier']:.3f}×") if overlay else "原速 1×"
    return dict(name="GLM-5.2",model=cfg["engine"]["model"],
      hardware="H200 SXM；算子级性能模型",
      precision="GLM-5.2-FP8；KV FP8，容量使用已验证的 GLM DSA 布局",
      deployment="1P1D；P/D 各 8 卡，共 16 卡",
      parallelism="Attention TP8 / DP1；MoE TP1 / EP8",
      hbm="P/D 自动 KV 容量，memory_fraction=0.9；64 token / KV 块",
      cache=f"P：HBM + {drams} GB 本地 DRAM；D：HBM；P、D 共用盘框，无本地 SSD",
      enclosure=f"{row['enclosure_tb']:.3f} TB；整框读 {reads} / 写 {writes} GB/s；固定时延 0.1 ms",
      transfer="H2D/D2H 各 25 GB/s/TP 分片；盘框直达 HBM；P→D 经盘框交接",
      scheduler=f"P/D 每轮 {row['budget']:,} token；序列上限 {p['scheduler']['max_sequences']}/{d['scheduler']['max_sequences']}；Chunked Prefill",
      memory=f"进程树预算 {cfg.get('execution',{}).get('resources',{}).get('memory_limit_gb','自动')} GB",
      sla="原始 TTFT ≤ 1 秒，且单请求平均 TPOT ≤ 30 ms",
      context="输入＋输出合计不超过 128K token",requests=m["request_count"],
      sessions=overlay.get("session_count",10962),
      source="SWE-chat Claude Code 主会话，verified-interleave 版",
      encoding="compact Mooncake；连续输入＋输出哈希；每段 16 token",
      timeline=timeline,arrival="按时间戳独立到达，未启用 session_id＋delay 完成依赖",
      note="只收录正式 compact 场景 2 全量实验。盘框读写为整框共享预算，与场景 1 每 TP 分片本地 SSD 的带宽口径不同。")

def markdown(rows, info, stamp):
    counts = Counter(r["status"] for r in rows)
    text = f"# GLM-5.2 · SWE-chat 场景 2 共享盘框实验\n\n更新时间：{stamp}。共 {len(rows)} 组；"
    text += "，".join(f"{k} {v} 组" for k,v in counts.items())+"。\n\n"
    text += "## 1. 配置与数据\n\n| 项目 | 内容 |\n|---|---|\n"
    fields = [("model","模型"),("hardware","GPU／性能模型"),("precision","精度"),
      ("deployment","部署"),("parallelism","并行"),("hbm","HBM KV"),("cache","缓存层级"),
      ("enclosure","共享盘框"),("transfer","传输"),("scheduler","调度"),("memory","内存预算"),
      ("sla","联合 SLA"),("context","上下文"),("source","数据来源"),("encoding","数据编码"),
      ("timeline","叠加时间"),("arrival","到达语义")]
    for key,title in fields:
        text += f"| {title} | {info[key]} |\n"
    text += f"\n每组 {info['requests']:,} 条请求，原始会话 {info['sessions']:,} 个。叠加保留会话内相对间隔，speedup=1。\n\n"
    text += "## 2. 实验结果\n\n"
    for budget,factor,speed in sorted({(r["budget"],r["factor"],r["speedup"]) for r in rows}):
        selected = [r for r in rows if (r["budget"],r["factor"],r["speedup"])==(budget,factor,speed)]
        text += f"### 预算 {budget:,} token · {factor}× 叠加 · speedup={speed:g}\n\n"
        columns = ["P 侧 DRAM","盘框读带宽（GB/s，整框）","盘框写带宽（GB/s，整框）",*COLUMNS]
        text += "| "+" | ".join(columns)+" |\n| "+" | ".join(["---"]*len(columns))+" |\n"
        for row in selected:
            text += "| "+" | ".join([f"{row['dram_gb']} GB",f"{row['read_gbps']:g}",f"{row['write_gbps']:g}",*row["cells"]])+" |\n"
        text += "\n"
    text += "## 3. 盘框路径与统计口径\n\n"
    text += "- P→D 经共享盘框交接。P 写已计算的输入块，D 写已计算的完整生成块；后续 P 可命中 D 写回块。\n"
    text += "- 分层复用率＝P 首次准入各层复用 token 总数／全部输入 token 总数。盘框恢复直达 HBM，不要求经过本地 DRAM。\n"
    text += "- 累计读写为整框字节量，覆盖 P 恢复、P→D 交接和 D 写回等任务，不等同于首次准入实际复用量。D 写回与 D→P 复用计数不表示独立请求数。\n"
    text += "- 扣排队后 TTFT 仅扣除 P 首次准入前调度等待，保留 KV 恢复、后续 chunk 等待、交接及 D 侧等待。\n"
    text += "- SLA 使用原始 TTFT 与请求级平均 TPOT；单 token 输出跳过 TPOT 判定。原始分位取引擎报告，扣排队后分位取逐请求最近秩。\n"
    text += "- 未完成项留空，不填 0，不绘制为已完成结果。GB/TB 为十进制；容量按已验证的 GLM 每分片 60,216 字节/token × 64 × TP8 换算。\n\n"
    text += "## 4. 更新与来源\n\n"
    text += "- HTML：[enclosure_experiments.html](enclosure_experiments.html)，单文件离线使用。\n"
    q = chr(96)
    text += f"- 更新：仓库根目录运行 {q}~/.venvs/aisim/bin/python experiments/disk_swe/scripts/build_enclosure_report.py{q}；随后刷新浏览器。\n"
    text += f"- 数值与来源索引：{q}analysis/enclosure_reports/rows.json{q}。构建脚本不启动后台监视任务。\n"
    for row in rows:
        text += f"- {row['id']}：{q}{row['source']}{q}；数据：{q}{row['bundle']}{q}。\n"
        if row.get("error"):
            text += f"  - {row['error']}\n"
        for attempt in row.get("attempts",[]):
            if attempt["id"]!=row["id"]:
                text += f"  - 历史尝试 {attempt['id']}：{attempt['status']}；{q}{attempt['source']}{q}。\n"
    return text

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root",type=Path,default=ROOT)
    parser.add_argument("--output-dir",type=Path,default=BASE)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    state = output/"analysis/enclosure_reports"
    cache_path = state/"metrics_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rows = collect(args.results_root.expanduser().resolve(),cache)
    info = config_info(rows)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    payload = dict(model="enclosure",info=info,rows=rows,columns=COLUMNS,
      metrics=[dict(key=k,label=l,unit=u) for k,l,u in METRICS],generated=stamp,
      md_path="enclosure_experiments.md",
      scope_note="盘框带宽按整框共享计，DRAM↔HBM 按每 TP 分片计。P→D 经盘框交接；未完成实验仅展示状态。")
    template = Path(__file__).with_name("enclosure_report_dashboard.html").read_text()
    assert template.count("__DATA__")==1
    html = template.replace("__DATA__",json.dumps(payload,ensure_ascii=False,separators=(",",":")).replace("</","<\\/"))
    atomic(output/"enclosure_experiments.md",markdown(rows,info,stamp))
    atomic(output/"enclosure_experiments.html",html)
    atomic(state/"rows.json",json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    atomic(cache_path,json.dumps(cache,ensure_ascii=False)+"\n")
    print(f"Built scene-2 report: {len(rows)} runs; "+", ".join(f"{k}={v}" for k,v in Counter(r["status"] for r in rows).items()))

if __name__=="__main__":
    main()
