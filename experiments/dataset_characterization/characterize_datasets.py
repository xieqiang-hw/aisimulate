"""Original, stdlib-only analysis of experiment bundles; no simulator dependencies.

Defaults match experiments/disk and disk_swe. Reads compact source metadata for
FreeInference and scalar lengths from SWE requests (never materializes hash IDs).
Reports describe retained replay cohorts, not the unfiltered upstream datasets.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
from itertools import zip_longest
import json
import math
from pathlib import Path
import re

DATA = Path.home() / "project/data"
FREE = DATA / "freeinference_agentic_trace/derived/dynosim-openloop-v1/428f809eef5952012f9c"
SWE = DATA / "swe_chat/derived/mooncake-v1/claude-code-main-glm52-128k"
SCALARS = {k: re.compile(rb'"' + k.encode() + rb'"\s*:\s*(\d+)\s*[,}]')
           for k in ("timestamp", "input_length", "output_length")}
EDGES = [0, 128, 512, 2048, 8192, 32768, 65536, 98304, 131072]
TURN_LABELS = ["1", "2–5", "6–10", "11–20", "21–50", "51–100", "101+"]
LIGHT = """<style>
body{background:#f5f7fa;color:#172439}
.card,figure,details{background:#fff;border-color:#d6e1eb}
th,td{border-color:#d6e1eb}th,p,figcaption{color:#41556d}
svg text{fill:#344960}h2,a,.legend{color:#006b68}.tag,.legend b{color:#924900}
svg text[style="fill:white"]{fill:#172439!important}
</style>"""


class Turns:
    def __init__(self):
        self.counts = Counter()
        self.groups = {}
        self.lengths = (Counter(), Counter())

    def add(self, session, ordinal, input_length, output_length):
        self.counts[session] += 1
        group = min(max(0, (ordinal - 1) // 5), 2)
        count, lengths = self.groups.get(group, (0, None))
        if lengths is None:
            lengths = (Counter(), Counter())
        self.groups[group] = (count + 1, lengths)
        for length, hist, total in zip((input_length, output_length), lengths, self.lengths):
            if type(length) is int and length >= 0:
                hist[length] += 1
                total[length] += 1

    def summary(self):
        return dict(
            stats=stats(Counter(self.counts.values())),
            lengths=[dict(stats=stats(h), bins=length_bins(h)) for h in self.lengths],
            distribution=[sum(1 for n in self.counts.values() if lo <= n <= hi)
                          for lo, hi in zip((1,2,6,11,21,51,101),(1,5,10,20,50,100,math.inf))],
            groups={i: dict(count=count, axes=[dict(stats=stats(h), bins=length_bins(h))
                   for h in lengths]) for i,(count,lengths) in sorted(self.groups.items())},
        )
LABELS = ["<128", "128–511", "512–2K", "2K–8K", "8K–32K", "32K–64K", "64K–96K", "96K–128K", "≥128K"]


def stats(hist):
    n = sum(hist.values())
    if not n:
        return {"n": 0}
    total = sum(v * c for v, c in hist.items())
    mean = total / n
    result = dict(n=n, total=total, mean=mean, min=min(hist), max=max(hist))
    for name, q in [("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)]:
        rank, seen = math.ceil(n * q), 0
        for value, count in sorted(hist.items()):
            seen += count
            if seen >= rank:
                result[name] = value
                break
    result["cv"] = math.sqrt(max(0, sum(c * (v - mean)**2 for v, c in hist.items()) / n)) / mean if mean else None
    remaining, top = math.ceil(.01 * n), 0
    for value, count in sorted(hist.items(), reverse=True):
        take = min(remaining, count)
        top += take * value
        remaining -= take
        if not remaining:
            break
    result["top1_share"] = top / total if total else None
    return result


def bucket(value):
    return sum(value >= edge for edge in EDGES) - 1


def length_bins(hist):
    bins = [0] * len(EDGES)
    for value, count in hist.items():
        bins[bucket(value)] += count
    return bins


def analyze(bundle, name, window_ms):
    manifest_raw = (bundle / "manifest.json").read_bytes()
    m = json.loads(manifest_raw)
    source_path = bundle / "source_requests.jsonl"
    hist = {k: Counter() for k in ("input", "output", "total", "gap", "session_gap")}
    bins, sessions, models, roles, finishes = {}, {}, Counter(), Counter(), Counter()
    turns = Turns()
    joint = [[0] * len(EDGES) for _ in EDGES]
    hashes = {"source_requests.jsonl": hashlib.sha256()}
    scalar_trace = name == "SWE-chat"
    request_file = (bundle / "requests.jsonl").open("rb") if scalar_trace else None
    if scalar_trace:
        hashes["requests.jsonl"] = hashlib.sha256()
    previous, first, n, zero, cached, prompt, cache_observed = None, None, 0, 0, 0, 0, 0
    try:
        with source_path.open("rb") as source:
            pairs = zip_longest(source, request_file) if request_file else ((raw, None) for raw in source)
            for raw, request in pairs:
                if raw is None or scalar_trace and request is None:
                    raise ValueError("Sidecar/trace count mismatch")
                hashes["source_requests.jsonl"].update(raw)
                row = json.loads(raw)
                if scalar_trace:
                    hashes["requests.jsonl"].update(request)
                    found = {k: pattern.search(request) for k, pattern in SCALARS.items()}
                    if not all(found.values()):
                        raise ValueError(f"Missing scalar at row {n}")
                    t, x, y = (int(found[k][1]) for k in SCALARS)
                    if t != row["timestamp"] or row.get("trace_row", n) != n:
                        raise ValueError(f"Unaligned sidecar at row {n}")
                else:
                    t, x, y = (row[k] for k in ("timestamp", "input_length", "output_length"))
                if any(type(v) is not int or v < 0 for v in (t, x, y)):
                    raise ValueError(f"Invalid scalar at row {n}")
                if previous is not None:
                    if t < previous:
                        raise ValueError("Non-monotonic timestamps")
                    hist["gap"][t - previous] += 1
                first = t if first is None else first
                previous = t
                for key, value in (("input", x), ("output", y), ("total", x+y)):
                    hist[key][value] += 1
                joint[bucket(x)][bucket(y)] += 1
                b = bins.setdefault(t // window_ms, [0, 0, 0])
                for i, v in enumerate((1, x, y)):
                    b[i] += v
                sid = (row.get("week", ""), row.get("source_model", ""), row["session_id"])
                if sid in sessions:
                    s = sessions[sid]
                    hist["session_gap"][t - s[2]] += 1
                    s[0] += 1
                    s[2] = t
                else:
                    sessions[sid] = [1, t, t]
                ordinal = row.get("call_index") if scalar_trace else row.get("step")
                turns.add(sid, ordinal + (0 if scalar_trace else 1) if type(ordinal) is int else sessions[sid][0], x, y)
                models[row.get("source_model", row.get("model", "未知"))] += 1
                roles.update(row.get("role_tokens", {}))
                finishes[row.get("finish_reason", "未提供")] += 1
                provider = row.get("provider_tokens", {})
                if type(provider.get("cached")) is int and type(provider.get("prompt")) is int:
                    cached += provider["cached"]
                    prompt += provider["prompt"]
                    cache_observed += 1
                zero += y == 0
                n += 1
                if n % 100000 == 0:
                    print(f"{name}: {n:,} rows", flush=True)
    finally:
        if request_file:
            request_file.close()
    if not n or n != m["request_count"]:
        raise ValueError(f"Manifest count mismatch: {n}")
    identities = {}
    for filename, digest in hashes.items():
        actual = digest.hexdigest()
        expected = m.get("files", {}).get(filename, {}).get("sha256")
        if expected and actual != expected:
            raise ValueError(f"Checksum mismatch: {filename}")
        identities[filename] = dict(sha256=actual, verified_against_manifest=bool(expected))
    recipe = m.get("recipe", {})
    # Original FreeInference declares a complete observation interval, including idle edges.
    begin = recipe.get("replay_start_ms", 0) - recipe.get("replay_start_ms", 0)
    end = recipe.get("score_end_offset_ms", previous + 1) if "overlay" not in m else previous + 1
    begin_bin, end_bin = begin // window_ms, (end - 1) // window_ms
    counts = Counter(bins.get(i, [0])[0] for i in range(begin_bin, end_bin + 1))
    summary = {k: stats(v) for k, v in hist.items()}
    summary["requests_per_window"] = stats(counts)
    summary["requests_per_session"] = stats(Counter(s[0] for s in sessions.values()))
    summary["session_span_ms"] = stats(Counter(s[2]-s[1] for s in sessions.values()))
    hist_bins = {k: [sum(c for v, c in hist[k].items() if bucket(v) == i) for i in range(len(EDGES))]
                 for k in ("input", "output")}
    if sum(v[0] for v in bins.values()) != n or sum(k*c for k,c in counts.items()) != n:
        raise ValueError("Observation interval does not cover all requests")
    for key, value in m.get('payload_totals', {}).items():
        if key in ('input_tokens', 'output_tokens') and value != summary[key.split('_')[0]]['total']:
            raise ValueError(f"Manifest token total mismatch: {key}")
    return dict(name=name, bundle=str(bundle), n=n, sessions=len(sessions), summary=summary,
                histogram=hist_bins, joint=joint, models=dict(models), roles=dict(roles), finish_reasons=dict(finishes),
                zero_output=zero, cached_tokens=cached, provider_prompt_tokens=prompt, cache_observed=cache_observed,
                window_ms=window_ms, observed_span_ms=previous-first+1, observation_ms=end-begin,
                bins=[[i*window_ms, *bins.get(i, [0, 0, 0])] for i in range(begin_bin, end_bin+1)],
                provenance=dict(manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), files=identities),
                manifest=m, turns=turns.summary())


def esc(v):
    return html.escape(str(v))


def fmt(v):
    return '—' if v is None else f'{v:,.2f}' if isinstance(v, float) else f'{v:,}'


def bars(series, labels, title, *, scale=None):
    """SVG grouped bars, each distribution normalized separately."""
    colors = ['#36c9c1', '#ffb45b']
    width, height = 1000, 300
    values = [[v / max(1, sum(a)) for v in a] for a in series]
    maximum = scale or max((v for a in values for v in a), default=1) or 1
    out = [f'<figure><h3>{esc(title)}</h3><svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">']
    for j, a in enumerate(values):
        for i, v in enumerate(a):
            x = 55+i*100+j*32
            h = v/maximum*195
            out.append(f'<rect x="{x}" y="{235-h}" width="29" height="{h}" fill="{colors[j]}"><title>{esc(labels[i])}: {v:.2%}</title></rect><text x="{x+14}" y="{225-h}" text-anchor="middle">{v:.1%}</text>')
    for i, label in enumerate(labels):
        out.append(f'<text x="{85+i*100}" y="265" text-anchor="middle">{esc(label)}</text>')
    return ''.join(out) + '</svg></figure>'


def lineplot(datasets, *, shared_domain_ms=None, shared_peak=None, input_tokens=False):
    window_ms = datasets[0]["window_ms"]
    if any(d["window_ms"] != window_ms for d in datasets):
        raise ValueError("Shared arrival chart requires an equal window width")
    domain_ms = shared_domain_ms or max(d["observation_ms"] for d in datasets)
    stride = max(1, math.ceil(max(len(d["bins"]) for d in datasets) / 450))
    field = 2 if input_tokens else 1
    conversion = 1 / 1000000 if input_tokens else 1000 / window_ms
    peak = shared_peak or max(row[field] for d in datasets for row in d["bins"]) * conversion
    power = 10 ** math.floor(math.log10(peak / 4)) if peak else 1
    rate_tick = next(step * power for step in (1, 2, 2.5, 5, 10) if step * power * 4 >= peak)
    left, right, top, bottom = 80, 960, 65, 305
    title = "FreeInference 与 SWE-chat 输入 token 总量" if input_tokens else "FreeInference 与 SWE-chat 到达速率"
    label = "共享坐标轴的输入 token 总量对比" if input_tokens else "共享坐标轴的到达速率对比"
    axis = f"百万 token / {window_ms/60000:g} 分钟" if input_tokens else "请求 / 秒"
    out = [f'<figure><h3>{title}</h3>'
           f'<svg viewBox="0 0 1000 365" role="img" aria-label="{label}">'
           f'<text x="80" y="48">{axis}</text>']
    for i in range(5):
        y = bottom - i / 4 * (bottom - top)
        out.append(f'<line x1="{left}" x2="{right}" y1="{y}" y2="{y}" stroke="#d6e1eb"/>'
                   f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{i * rate_tick:g}</text>')
    for i in range(6):
        x = left + i / 5 * (right - left)
        day = i / 5 * domain_ms / 86400000
        out.append(f'<line x1="{x}" x2="{x}" y1="{top}" y2="{bottom}" stroke="#e4ebf1"/>'
                   f'<text x="{x}" y="{bottom + 19}" text-anchor="middle">{day:.1f}</text>')
    out.append(f'<text x="{(left+right)/2}" y="350" text-anchor="middle">距各数据集回放起点（天）</text>')
    for index, (d, color) in enumerate(zip(datasets, ("#007872", "#bf6400"))):
        rows = d["bins"]
        selected = [max(rows[i:i + stride], key=lambda row: row[field]) for i in range(0, len(rows), stride)]
        if selected[-1] != rows[-1]:
            selected.append(rows[-1])
        points = [
            (left + row[0] / domain_ms * (right - left),
             bottom - row[field] * conversion / (4 * rate_tick) * (bottom - top))
            for row in selected
        ]
        coords = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
        legend = 485 + index * 235
        maximum = max(row[field] for row in selected) * conversion
        out.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.65" points="{coords}"/>'
                   f'<line x1="{legend}" x2="{legend + 20}" y1="25" y2="25" stroke="{color}" stroke-width="3"/>'
                   f'<text x="{legend + 26}" y="29">{esc(d["name"])}（峰值 {maximum:.3f}）</text>')
    caption = (f'每 {window_ms/60000:g} 分钟加总到达请求的已知输入 token。' if input_tokens
               else f'每 {window_ms/60000:g} 分钟统计请求数，换算成请求每秒。')
    out.append(f'</svg><figcaption>{caption}'
               f'每连续 {stride} 窗绘制峰值所在时间点以保留尖峰，末尾补绘最后一个窗口。'
               '横轴使用同一相对时间尺度，各数据集的实际日期独立。</figcaption></figure>')
    return ''.join(out)


def heatmap(d):
    out=[f'<figure><h3>{d["name"]} · 输入 × 输出联合分布</h3><svg viewBox="0 0 1000 525" role="img" aria-label="输入输出长度联合分布">']
    peak=max(max(row) for row in d['joint']) or 1
    for i,row in enumerate(d['joint']):
        out.append(f'<text x="90" y="{55+i*45}" text-anchor="end">{esc(LABELS[i])}</text>')
        for j,n in enumerate(row):
            opacity=.08+.92*math.sqrt(n/peak)
            out.append(f'<rect x="{105+j*95}" y="{30+i*45}" width="91" height="41" fill="#36c9c1" opacity="{opacity}"/><text x="{150+j*95}" y="{55+i*45}" text-anchor="middle" style="fill:white">{n/d["n"]:.1%}</text>')
    for j,label in enumerate(LABELS):
        out.append(f'<text x="{150+j*95}" y="460" text-anchor="middle">{esc(label)}</text>')
    return ''.join(out)+'</svg><figcaption>纵轴为输入长度，横轴为输出长度；格内为全部请求占比，颜色按本图最大格平方根缩放。两图颜色深浅不可直接比较。</figcaption></figure>'


def report(datasets):
    a,b = datasets
    out = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FreeInference × SWE-chat 数据集画像</title><style>
    *{box-sizing:border-box}body{margin:0;background:#0c1424;color:#e8edf8;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1200px;margin:auto;padding:48px 28px}h1{font-size:42px;line-height:1.25}h2{margin-top:48px;color:#36c9c1}h3{margin:0 0 16px}p,figcaption{color:#b8c4d8}.tag{color:#ffb45b;letter-spacing:3px}.cards,.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card,figure,details{background:#152238;border:1px solid #293c59;border-radius:14px;padding:24px;margin:0 0 20px}.big{font-size:32px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #293c59}th{color:#aab9d0}svg{width:100%;height:auto}svg text{fill:#b8c4d8;font:12px system-ui}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}summary{cursor:pointer}a{color:#36c9c1}.legend{color:#36c9c1}.legend b{color:#ffb45b}@media(max-width:700px){.grid,.cards{grid-template-columns:1fr}main{padding:20px}h1{font-size:30px}.table{overflow:auto}}
    </style><main><div class="tag">WORKLOAD ATLAS / 实验输入画像</div><h1>FreeInference × SWE-chat</h1><p>从请求形态、长尾、时间变化与会话结构理解两种回放负载。全部统计来自实际保留的数据包，未抽样。</p>''']
    out.append('<div class="cards">')
    for d in datasets:
        out.append(f'<div class="card"><h3>{d["name"]}</h3><div class="big">{d["n"]:,} 请求</div><p>{d["sessions"]:,} 个观测会话 · {d["observation_ms"]/86400000:.1f} 天观察区间</p></div>')
    out.append('</div><h2>01 / 先明确比较口径</h2><p>FreeInference 是多模型生产 agentic trace，输入为 o200k 角色 token 总和，输出为 provider completion tokens；映射到 GLM-5.2 不代表重新分词。SWE-chat 是 Claude Code 主会话重建，使用 GLM tokenizer；原始 system prompts 与 tool schemas 缺失，压缩或歧义处终止提取。SWE 时间是首个响应事件代理，不能视为真实请求到达或端到端延迟。这里的 SWE 指 SWE-chat，不等同于 SWE-bench 题目集。</p><p>基础数据包刻画原始回放负载；overlay 改变会话时间安排与密度，不能把其突发视为自然流量。token 比例只能提示负载形态，不能直接换算为 P/D 耗时或最优配比。</p>')
    out.append(f'<div class="card"><h3>这批数据呈现什么特征？</h3><p>FreeInference 的输入中位数为 {a["summary"]["input"]["p50"]:,} token，SWE 为 {b["summary"]["input"]["p50"]:,}；输出中位数分别为 {a["summary"]["output"]["p50"]:,} 和 {b["summary"]["output"]["p50"]:,}。最长 1% 输出请求分别贡献 {a["summary"]["output"]["top1_share"]:.1%}、{b["summary"]["output"]["top1_share"]:.1%} 的输出 token，均需关注输出长尾。</p><p>每五分钟请求数的 CV 分别为 {a["summary"]["requests_per_window"]["cv"]:.2f}、{b["summary"]["requests_per_window"]["cv"]:.2f}。这些差异描述当前保留样本，受观察区间、提取过滤与分词口径影响。</p></div>'.replace('每五分钟',f'每 {a["window_ms"]/60000:g} 分钟'))
    out.append('<h2>02 / 长度分布与长尾</h2><p class="legend">■ FreeInference　<b>■ SWE-chat</b>　悬停柱形查看精确占比</p>')
    for key,label in [('input','输入 token 长度'),('output','输出 token 长度')]:
        out.append(bars([d['histogram'][key] for d in datasets],LABELS,label))
    rows=[]
    for key,label in [('input','输入'),('output','输出')]:
        for metric,mlabel in [('mean','均值'),('p50','P50'),('p90','P90'),('p99','P99'),('max','最大值'),('top1_share','最长 1% 请求的 token 占比')]:
            vals=[d['summary'][key][metric] for d in datasets]
            rows.append([label+' / '+mlabel,*[f'{v:.2%}' if metric=='top1_share' else fmt(v) for v in vals]])
    rows.append(['总输入 / 总输出 token',*[fmt(d['summary']['input']['total']/max(1,d['summary']['output']['total'])) for d in datasets]])
    rows.append(['输出 token 占总长度',*[f'{d["summary"]["output"]["total"]/max(1,d["summary"]["total"]["total"]):.2%}' for d in datasets]])
    out.append(table(rows))
    out.extend(heatmap(d) for d in datasets)
    out.append('<h2>03 / 到达速率与突发性</h2>')
    out.append(lineplot(datasets))
    out.append(table([[label,*[fmt(func(d)) for d in datasets]] for label,func in [
        ('平均到达 req/s',lambda d:d['n']/(d['observation_ms']/1000)),
        ('每窗请求数 P99',lambda d:d['summary']['requests_per_window']['p99']),
        ('每窗请求数 CV',lambda d:d['summary']['requests_per_window']['cv']),
        ('到达间隔 P50 / ms',lambda d:d['summary']['gap'].get('p50')),
        ('到达间隔 P99 / ms',lambda d:d['summary']['gap'].get('p99'))]]))
    out.append('<p>CV = 标准差 / 均值；在相同窗口宽度下，较大值表示时间分布更不均匀。平均速率使用完整观察区间，含空窗；SWE 仅覆盖首尾事件范围，FreeInference 使用 manifest 声明区间。</p><h2>04 / 会话与上下文结构</h2>')
    out.append(table([[label,*[fmt(d['summary'][key].get(metric)) for d in datasets]] for label,key,metric in [
        ('每会话请求数 P50','requests_per_session','p50'),('每会话请求数 P99','requests_per_session','p99'),
        ('会话跨度 P50 / ms','session_span_ms','p50'),('会话跨度 P99 / ms','session_span_ms','p99'),
        ('会话内相邻请求间隔 P50 / ms','session_gap','p50')]]))
    out.append('<p>会话是保留样本中的观测片段，不保证完整。FreeInference 按 week、source_model、session_id 联合分组，匹配其前缀命名空间；SWE 按 session_id 分组。会话跨度与间隔不能解释为推理时间。</p><div class="grid">')
    for d in datasets:
        top=d['models']
        out.append('<div class="card"><h3>'+d['name']+' · 保留请求来源模型</h3>'+''.join(f'<div>{esc(k)}：{v:,} ({v/d["n"]:.1%})</div>' for k,v in sorted(top.items(),key=lambda x:-x[1])[:8])+'</div>')
    out.append('</div><h2>05 / 缓存与角色组成的可观测边界</h2>')
    if a['roles']:
        total=sum(a['roles'].values())
        out.append('<p>FreeInference 输入角色 token 组成：'+ '；'.join(f'{esc(k)} {v/total:.1%}' for k,v in a['roles'].items())+'。</p>')
    if a['provider_prompt_tokens']:
        out.append(f'<p>FreeInference provider cached / prompt token 总量比为 {a["cached_tokens"]/a["provider_prompt_tokens"]:.2%}，覆盖 {a["cache_observed"]:,} 条保留请求。该比值使用 provider 自身口径，不与 o200k 输入长度相除。</p>')
    out.append('<p>SWE 数据包未提供可比较的 provider cache-read 与角色 token 计数，缺失不记作零。本报告不推断模拟器实际缓存命中率；真实命中还取决于路由、缓存容量、逐出与时间安排。完整 block hash 扫描未启用，FreeInference 的 25 GB requests 文件不为本报告重复解码。</p><h2>06 / 可复现性与数据质量</h2><p>分位数使用 nearest-rank（ceil(q×N)）；长度直方图与统计精确计算。流式读取、校验行数、时间单调性及可用 manifest SHA256。JSON 伴随文件包含完整统计、联合长度分布、时间窗与元信息。未重新验证上游 parquet 或未读取的 requests 文件。</p>')
    for d in datasets:
        meta={k:d[k] for k in ('bundle','provenance','zero_output')}
        meta['manifest']={k:({sk:sv for sk,sv in v.items() if sk!='exclusion_bins'} if isinstance(v,dict) else v)
                          for k,v in d['manifest'].items()}
        out.append(f'<details><summary>{d["name"]} · 输入路径、过滤策略与来源元信息</summary><pre>{esc(json.dumps(meta,ensure_ascii=False,indent=2))}</pre></details>')
    out.append('<p>FreeInference 来源与 CC-BY-4.0 署名以输入包 ATTRIBUTION.md 为准，来源 URL 与固定 revision 保存在上述 manifest。此报告仅发布聚合统计，不包含原始对话。</p></main></html>')
    turn_note = ('<p>回放中一次模型请求计一轮。FreeInference 使用原始 step 加 1，'
                 'SWE-chat 使用原始 call_index。筛选留下的轮次空缺保留在序号中。</p>')
    return ''.join(out).replace('<main>', LIGHT+'<main>', 1).replace(
        '</main>', turn_section(datasets, note=turn_note)+user_turn_section(datasets)+'</main>', 1)


def user_turn_section(datasets):
    if not all("user_turns" in dataset for dataset in datasets):
        return ""
    values = [d["user_turns"] for d in datasets]
    rows = [
        ["覆盖会话数", *[fmt(v["stats"]["n"]) for v in values]],
        ["与源记录匹配的回放请求数", *[fmt(v.get("matched_requests")) for v in values]],
        ["用户交互轮数总和", *[fmt(v["stats"].get("total")) for v in values]],
    ]
    rows.extend([[label, *[fmt(v["stats"].get(key)) for v in values]]
                 for key, label in (("mean", "均值"), ("p50", "P50"), ("p95", "P95"), ("max", "最大值"))])
    return (
        '<h2>每会话用户交互轮数</h2>'
        + ''.join(f'<p><strong>{esc(d["name"])}</strong> {esc(v["method"])}</p>'
                  for d, v in zip(datasets, values))
        + bars([v["distribution"] for v in values],
               ["0", "1", "2–5", "6–10", "11–20", "21–50", "51–100", "101+"],
               "用户交互轮数分布（覆盖会话占比）")
        + table(rows)
    )


def turn_section(datasets, units=("token", "token"), chart_labels=LABELS, note=""):
    out = ['<h2>会话轮数与分轮段长度</h2>',
           '<p>轮数为每个会话实际记录的单位数。分轮段按源记录中的序号分组。均值和分位数只使用该指标的有效观测。</p>',
           note]
    out.append(bars([d['turns']['distribution'] for d in datasets], TURN_LABELS, '会话轮数分布（会话占比）'))
    metrics = [('mean','均值'),('p50','P50'),('p90','P90'),('p95','P95'),('p99','P99'),('max','最大值')]
    out.append(table([[label,*[fmt(d['turns']['stats'].get(k)) for d in datasets]]
                      for k,label in metrics]))
    for i,label in enumerate(("第 1–5 轮", "第 6–10 轮", "第 10 轮之后（第 11 轮起）")):
        groups = [d['turns']['groups'].get(i, {}) for d in datasets]
        out.append(f'<h3>{label}</h3>')
        rows = [['记录数', *[fmt(g.get('count',0)) for g in groups]]]
        for axis, name in enumerate(('输入','输出')):
            values = [g.get('axes', [{},{}])[axis] for g in groups]
            rows.extend([[f'{name} · {metric}', *[fmt(v.get('stats',{}).get(key)) for v in values]]
                         for key,metric in [('n','有效记录数'),*metrics]])
            if units[0] == units[1]:
                out.append(bars([v.get('bins',[0]*len(EDGES)) for v in values], chart_labels,
                                f'{name}长度分布（{units[0]}）'))
            else:
                for d,v,unit in zip(datasets,values,units):
                    out.append(bars([v.get('bins',[0]*len(EDGES))],chart_labels,
                                    f'{d["name"]} · {name}长度分布（{unit}）'))
        out.append(table(rows))
    return ''.join(out)


def table(rows):
    return '<div class="table"><table><thead><tr><th>指标</th><th>FreeInference</th><th>SWE-chat</th></tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeinfer',type=Path,default=FREE)
    p.add_argument('--swe',type=Path,default=SWE)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'reports/dataset_characteristics.html')
    p.add_argument('--window-minutes',type=int,default=5)
    args=p.parse_args()
    if args.window_minutes<=0:
        p.error('--window-minutes must be positive')
    data=[analyze(args.freeinfer,'FreeInference',args.window_minutes*60000),analyze(args.swe,'SWE-chat',args.window_minutes*60000)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(report(data),encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(),datasets=data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(args.output)


if __name__=='__main__':
    main()
