"""Adapt the repository's shared SWE report template for per-dataset chat scans.

The visual template is read from disk_swe, not independently restyled here.
Only chat data bindings, scope text and the dataset selector are adapted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SHARED_TEMPLATE = BASE.parent / "disk_swe/scripts/model_report_dashboard.html"
COLUMNS = [
    ("status", "状态"), ("completed", "完成请求"), ("sla", "联合 SLA 达标率"),
    ("p99_ttft", "P99 TTFT（秒）"), ("mean_ttft", "平均 TTFT（秒）"), ("tpot", "TPOT（ms）"),
    ("hbm", "HBM 复用"), ("dram", "DRAM 直接复用"), ("ssd", "SSD 恢复复用"),
    ("reuse", "总复用率"), ("ssd_read", "SSD 累计读"), ("wall_seconds", "实际耗时"),
    ("adjusted_mean", "扣排队后平均 TTFT（秒）"), ("adjusted_p99", "扣排队后 P99 TTFT（秒）"),
    ("memory", "内存峰值（GB）"),
]


def cell(row, key):
    if key == "status":
        return row["status"]
    value = row["values"].get(key)
    if value is None:
        return "—"
    if key == "completed":
        return f"{value:,}"
    if key in ("sla", "hbm", "dram", "ssd", "reuse"):
        return f"{value:.3f}%"
    if key == "ssd_read":
        return f"{value:.3f} TB"
    if key == "wall_seconds":
        seconds = round(value)
        return f"{seconds // 60} 分 {seconds % 60} 秒"
    return f"{value:.{3 if key == 'memory' else 2}f}"


def payload(data):
    rows, datasets = [], {}
    for row in data["rows"]:
        rows.append({**row, "dram_gb": row["p_dram_gb"], "factor": 1,
                     "source": row.get("run_json") or row["sweep"],
                     "cells": [cell(row, key) for key, _ in COLUMNS]})
        datasets[row["dataset"]] = {
            "requests": row["expected"], "sessions": row.get("sessions"),
            "span_seconds": row.get("span_seconds"), "arrival_kind": row.get("arrival_kind"),
            "source_block_size": row.get("source_block_size"),
        }
    return {
        "model": "glm52", "generated": data["generated"], "rows": rows,
        "datasets": datasets, "warnings": data["warnings"],
        "columns": [label for _, label in COLUMNS],
        "metrics": [{"key": key, "label": label, "unit": unit} for key, label, unit in data["metrics"]],
        "info": {
            "name": "GLM-5.2 · Chat", "model": "zai-org/GLM-5.2-FP8",
            "hardware": "H200 SXM；使用 H200 性能模型",
            "precision": "GLM-5.2-FP8；KV 使用框架解析的 FP8 DSA 布局",
            "deployment": "1P1D；P/D 各 8 卡，共 16 卡",
            "parallelism": "Attention TP8 / DP1；MoE TP1 / EP8",
            "hbm": "P/D 自动 KV 容量，memory_fraction=0.9；块大小 64 token",
            "cache": "P：HBM + 本地 DRAM + worker_local SSD；D：仅 HBM",
            "ssd": "约 56 TB；写 8 GB/s/TP 分片",
            "transfer": "H2D / D2H 各 25 GB/s/TP；P→D 50 GB/s/TP，destination_missing",
            "scheduler": "P/D 序列上限 32/256；开启 Chunked Prefill",
            "sla": "TTFT ≤ 1 秒，且单请求平均 TPOT ≤ 30 ms",
            "context": "128K 输入+输出上限；保留源 token 长度，无真实 output_token_ids",
            "source": "Qwen / Mooncake 各文件独立回放，数据集单独筛选",
            "requests": 0,
            "note": "只记录正式全量回放。不同数据集独立筛选；无真实输出 token ID，不能等同于 SWE 的 D 输出 token 级复用。运行内存默认自动，实测 RSS 与模拟 DRAM 容量分开。",
        },
    }


def replace_once(page, before, after):
    if page.count(before) != 1:
        raise ValueError(f"shared SWE template anchor changed: {before[:90]!r}")
    return page.replace(before, after, 1)


def template():
    original = SHARED_TEMPLATE.read_text()
    page = original
    changes = [
        ("SWE-CHAT · 场景 1 · 本地 DRAM 与 SSD", "CHAT TRACE · 场景 1 · 本地 DRAM 与 SSD"),
        ("grid-template-columns:repeat(5,1fr)", "grid-template-columns:repeat(6,1fr)"),
        ('<div class="controls"><label>单轮 token 预算', '<div class="controls"><label>数据集<select id="datasetSelect"></select></label><label>单轮 token 预算'),
        ("__DATA__", "__REPORT_DATA__"),
        ("function loadRows(){return DATA.rows.filter(r=>r.budget===+$('budget').value&&r.factor===+$('factor').value&&r.speedup===+$('speed').value)}",
         "function datasetRows(){return DATA.rows.filter(r=>r.dataset===$('datasetSelect').value)}\nfunction loadRows(){return datasetRows().filter(r=>r.budget===+$('budget').value&&r.factor===+$('factor').value&&r.speedup===+$('speed').value)}"),
        ("const rows=DATA.rows.filter(r=>r.budget===+$('budget').value);options('factor'", "const rows=datasetRows().filter(r=>r.budget===+$('budget').value);options('factor'"),
        ("const rows=DATA.rows.filter(r=>r.budget===+$('budget').value&&r.factor===+$('factor').value);options('speed'", "const rows=datasetRows().filter(r=>r.budget===+$('budget').value&&r.factor===+$('factor').value);options('speed'"),
        ("function render(){selection=loadRows()", "function render(){renderDatasetInfo();selection=loadRows()"),
        ("$('status').textContent=`${$('budget').value} token / 轮", "$('status').textContent=`${$('datasetSelect').value} · ${$('budget').value} token / 轮"),
        ("$('selectionNote').textContent=DATA.model==='minimax'?'MiniMax 为 FP8 计算、BF16 KV；自动 HBM KV 容量随单轮预算变化，不能将预算扫描视作只改变调度而容量不变。':'本页只汇总 H200 EP8 的场景 1 扫描；原始 8× 包的额外吞吐加压、Prefill 排队分析保持独立。';",
         "$('selectionNote').textContent='当前数据集独立绘制 DRAM 与读带宽扫描；未完成和缺失指标不画成 0。原始时间轴 1×，speedup 与会话叠加倍率分别记录。';"),
        ("options('referenceDram',drams.slice(1).map", "options('referenceDram',(drams.length>1?drams.slice(1):drams).map"),
        ("$('intro').textContent='按模型统一查看 DRAM、SSD 带宽与调度预算扫描。不同负载、speedup 和预算独立筛选，避免将不同配置结果混在一条曲线中。';",
         "$('intro').textContent='按数据集查看 DRAM、SSD 带宽与请求性能。页面布局与 SWE 实验页保持一致，不同数据集、speedup 和预算独立筛选。';"),
        ("<p id=\"modelNote\"></p>", '<p id="modelNote"></p><details><summary>所选组合的原始结果与运行资源</summary><div id="sourcePaths"></div></details>'),
        ("renderCharts();$('tableHead').innerHTML=", "renderCharts();renderSources();$('tableHead').innerHTML="),
        ("<th>预算 token</th><th>标称倍率</th>", "<th>数据集</th><th>预算 token</th><th>标称倍率</th>"),
        ("const k=[r.budget,r.factor,r.speedup].join('/')", "const k=[r.dataset,r.budget,r.factor,r.speedup].join('/')"),
        ("[fmt(r.budget,0),r.factor+'×',r.speedup,rs.length", "[r.dataset,fmt(r.budget,0),r.factor+'×',r.speedup,rs.length"),
        ("n('失败')+n('已中断')", "n('失败')+n('已中断')+n('结果校验失败')"),
        ("来源索引随结果内嵌于本 HTML；旧分散报告存档在 analysis/model_reports/source_documents/，不再作为顶层入口。原始实验目录与数据包保留。",
         "来源索引随结果内嵌于本 HTML；原始结果与数据包保留。刷新脚本：experiments/disk_chat/scripts/refresh_experiments.py。"),
        ("['budget','overlay_factor','speedup','dram_GB','SSD_read_GBps_per_TP',...DATA.columns]", "['dataset','budget','overlay_factor','speedup','dram_GB','SSD_read_GBps_per_TP',...DATA.columns]"),
        ("[r.budget,r.factor,r.speedup,r.dram_gb,r.read_gbps,...r.cells]", "[r.dataset,r.budget,r.factor,r.speedup,r.dram_gb,r.read_gbps,...r.cells]"),
        ("a.download=DATA.model+'_experiments.csv'", "a.download=$('datasetSelect').value+'_glm52_experiments.csv'"),
        ("options('budget',values(DATA.rows,'budget').map(v=>[v,fmt(v,0)+' token']),'16384');",
         "options('datasetSelect',Object.keys(DATA.datasets).map(v=>[v,v]),'qwen-chat');options('budget',values(datasetRows(),'budget').map(v=>[v,fmt(v,0)+' token']),'16384');"),
        ("$('budget').onchange=()=>configure(0);", "$('datasetSelect').onchange=()=>{options('budget',values(datasetRows(),'budget').map(v=>[v,fmt(v,0)+' token']),'16384');configure(0)};$('budget').onchange=()=>configure(0);"),
        ("window.reportTest={model:DATA.model,rows:DATA.rows.length,selected:selection.length,groups:scope.size}",
         "window.reportTest={model:DATA.model,rows:DATA.rows.length,selected:selection.length,groups:scope.size,dataset:$('datasetSelect').value}"),
    ]
    for before, after in changes:
        page = replace_once(page, before, after)
    extra = r'''
function renderDatasetInfo(){
 const name=$('datasetSelect').value,meta=DATA.datasets[name];if(!meta)return;
 $('dataset').innerHTML=infoRows([['数据集',name],['回放规模','每组 '+fmt(meta.requests,0)+' 条请求'],['会话数',meta.sessions==null?'未提供显式会话 ID':fmt(meta.sessions,0)],['时间跨度',fmt(meta.span_seconds,3)+' 秒'],['到达时间',meta.arrival_kind==='synthetic_poisson'?'合成 Poisson 到达':'真实观测到达'],['上下文',DATA.info.context],['时间口径','原始时间轴 1×；speedup 小于 1 会放大请求间隔'],['联合 SLA',DATA.info.sla]]);
 $('datasetFootnote').innerHTML='<strong>适用范围</strong><br>'+infoEscape(DATA.info.note);
}
function renderSources(){
 $('sourcePaths').innerHTML=selection.map(r=>'<div style="padding:12px 0;border-top:1px solid var(--line)"><strong>'+infoEscape(r.dataset)+' · '+fmt(r.dram_gb,0)+' GB · '+fmt(r.read_gbps)+' GB/s/TP · '+infoEscape(r.status)+'</strong><p class="small">运行：<code>'+infoEscape(r.run_json||r.sweep)+'</code></p><p class="small">配置：<code>'+infoEscape(r.config_path||'—')+'</code></p><p class="small">实际内存预算：'+fmt(r.values.memory_budget_gb)+' GB；RSS 峰值：'+fmt(r.values.memory)+' GB</p>'+(r.error?'<p class="small">'+infoEscape(r.error)+'</p>':'')+'</div>').join('');
}
'''
    page = replace_once(page, "const info=DATA.info;", extra + "\nconst info=DATA.info;")
    digest = hashlib.sha256(original.encode()).hexdigest()
    return "<!-- Generated from disk_swe/scripts/model_report_dashboard.html; SHA256 " + digest + "; adapted by build_chat_dashboard.py. -->\n" + page


def render(data):
    encoded = json.dumps(payload(data), ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template().replace("__REPORT_DATA__", encoded)
