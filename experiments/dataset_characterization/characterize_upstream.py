"""Aggregate the original published tables without replay filtering or retokenization."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq

from characterize_datasets import DATA, LABELS, LIGHT, Turns, bars, esc, fmt, stats, table, turn_section, user_turn_section


def date_range(times):
    return [datetime.fromtimestamp(t / 1000, timezone.utc).isoformat() for t in times] if times else []


def freeinference(root):
    files = sorted(root.glob("release/*/processed/requests.parquet"))
    if not files:
        raise ValueError(f"No weekly request files in {root}")
    turns, models, native_input = Turns(), Counter(), Counter()
    providers, model_sessions, provider_sessions = Counter(), defaultdict(set), defaultdict(set)
    periods = []
    limit_exceeded = zero_outputs = 0
    columns = ["session_id", "step", "start_ms", "model", "role_tokens",
               "completion_tokens", "prompt_tokens", "provider"]
    for path in files:
        parquet = pq.ParquetFile(path)
        week = path.parents[1].name
        lo, hi = float("inf"), float("-inf")
        for batch in parquet.iter_batches(columns=columns, batch_size=32768):
            for row in batch.to_pylist():
                roles = row["role_tokens"]
                x = sum(roles.values()) if roles and all(type(v) is int for v in roles.values()) else None
                y = row["completion_tokens"]
                session = (week, row["session_id"])
                turns.add(session, row["step"] + 1, x, y)
                model, provider = row["model"] or "未知", row["provider"] or "未知"
                models[model] += 1
                providers[provider] += 1
                model_sessions[model].add(session)
                provider_sessions[provider].add(session)
                if row["prompt_tokens"] is not None:
                    native_input[row["prompt_tokens"]] += 1
                if x is not None and y is not None:
                    limit_exceeded += x + y > 131072
                zero_outputs += y == 0
                lo = min(lo, row["start_ms"])
                hi = max(hi, row["start_ms"])
        periods.append([lo, hi])
        print(f"FreeInference {week}: {parquet.metadata.num_rows:,}", flush=True)
    return dict(name="FreeInference", n=sum(models.values()), sessions=len(turns.counts),
                turns=turns.summary(), models=dict(models), files=[str(p) for p in files],
                observed_utc=date_range([min(p[0] for p in periods), max(p[1] for p in periods)]),
                zero_outputs=zero_outputs, over_128k=limit_exceeded, provider_prompt=stats(native_input),
                providers=dict(providers), model_sessions={k: len(v) for k,v in model_sessions.items()},
                provider_sessions={k: len(v) for k,v in provider_sessions.items()})


def swechat(root):
    path = root / "conversations.parquet"
    parquet = pq.ParquetFile(path)
    session_index = pq.read_table(root / "sessions.parquet", columns=["session_id", "agent"])
    agents = {r["session_id"]: r["agent"] or "未知" for r in session_index.to_pylist()}
    columns = ["session_id", "turn_number", "response_id",
               "turn_type", "role", "model", "agent", "input_tokens", "output_tokens",
               "cache_creation_input_tokens", "cache_read_input_tokens", "timestamp"]
    responses = defaultdict(dict)
    human_prompts = Counter()
    for batch in parquet.iter_batches(columns=columns, batch_size=65536):
        # Only response-bearing rows and user prompts are needed. Never read content.
        for row in batch.to_pylist():
            sid = row["session_id"]
            kind = row["turn_type"]
            if sid and kind == "user_prompt":
                human_prompts[sid] += 1
            if row["role"] != "assistant" and kind not in ("assistant_response", "tool_use"):
                continue
            ordinal = row["turn_number"]
            response = row["response_id"]
            if not sid or not response or ordinal is None:
                continue
            record = responses[sid].setdefault(response, [ordinal, None, None, None, None, False, None])
            record[0] = min(ordinal, record[0])
            for index, field in ((1,"input_tokens"), (2,"output_tokens")):
                value = row[field]
                cache_usage = sum(row[k] or 0 for k in ("cache_creation_input_tokens", "cache_read_input_tokens"))
                # Upstream also uses zero on content fragments without usage.
                if type(value) is int and (value > 0 or value == 0 and (
                    (row["output_tokens" if index == 1 else "input_tokens"] or 0) > 0 or cache_usage > 0
                )):
                    record[index] = max(value, record[index] if record[index] is not None else 0)
                    if index == 1:
                        total = value + cache_usage
                        record[6] = max(total, record[6] or 0)
            if row["model"]:
                record[3] = row["model"]
            if row["agent"] == "Claude Code":
                record[5] = True
            if row["timestamp"] and (record[4] is None or row["timestamp"] < record[4]):
                record[4] = row["timestamp"]
    turns, raw_input, models = Turns(), Turns(), Counter()
    clients, client_sessions = Counter(), Counter()
    client_models, model_sessions = defaultdict(Counter), defaultdict(set)
    lo, hi = None, None
    for sid, records in responses.items():
        client = agents.get(sid, "未知")
        client_sessions[client] += 1
        clients[client] += len(records)
        for i, (_, x, y, model, time, claude_client, total) in enumerate(sorted(records.values(), key=lambda record: record[0]), 1):
            claude_usage = claude_client and (model or "").lower().startswith("claude-")
            turns.add(sid, i, total if claude_usage else None, y)
            raw_input.add(sid, i, x, None)
            models[model or "未知"] += 1
            model_sessions[model or "未知"].add(sid)
            client_models[client][model or "未知"] += 1
            if time is not None:
                lo = time if lo is None else min(lo, time)
                hi = time if hi is None else max(hi, time)
    return dict(name="SWE-chat", n=sum(models.values()), sessions=len(turns.counts),
                turns=turns.summary(), models=dict(models), files=[str(path)],
                observed_utc=[t.isoformat() for t in (lo,hi) if t is not None],
                native_response_ids=sum(models.values()),
                human_prompts=sum(human_prompts.values()), raw_input=raw_input.summary(),
                user_prompts_per_session=stats(Counter(human_prompts.values())),
                clients=dict(clients), client_sessions=dict(client_sessions),
                model_sessions={k: len(v) for k,v in model_sessions.items()},
                client_models={k: dict(v) for k,v in client_models.items()})


def distribution(title, counts, sessions, total, unit, color):
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    peak = ranked[0][1] if ranked else 1
    out = [f'<figure><h3>{esc(title)}</h3><table><thead><tr><th>标签</th>'
           f'<th>{unit}数</th><th>占比</th><th>覆盖会话数</th></tr></thead><tbody>']
    for name, count in ranked[:10]:
        out.append(f'<tr><td>{esc(name)}</td><td>{fmt(count)}</td><td>{count/total:.2%}'
                   f'<div style="height:5px;width:{100*count/peak:.2f}%;background:{color}"></div></td>'
                   f'<td>{fmt(sessions.get(name))}</td></tr>')
    out.append('</tbody></table></figure>')
    if len(ranked) > 10:
        out.append(f'<details><summary>全部 {len(ranked):,} 个标签</summary>'
                   f'<table><thead><tr><th>标签</th><th>{unit}数</th><th>占比</th>'
                   '<th>覆盖会话数</th></tr></thead><tbody>')
        out.extend(f'<tr><td>{esc(name)}</td><td>{fmt(count)}</td><td>{count/total:.2%}</td>'
                   f'<td>{fmt(sessions.get(name))}</td></tr>' for name, count in ranked)
        out.append('</tbody></table></details>')
    return ''.join(out)


def render(datasets, free_root, swe_root):
    a, b = datasets
    out = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>原始数据集画像 · FreeInference × SWE-chat</title>',
           '<style>body{font:16px/1.65 system-ui,sans-serif;margin:0}main{max-width:1120px;margin:auto;padding:32px 20px}figure{margin:18px 0;padding:18px;border:1px solid}svg{width:100%;font:13px system-ui}table{width:100%;border-collapse:collapse;background:white;margin:14px 0 26px}th,td{text-align:left;padding:8px;border-bottom:1px solid}h3{margin-top:30px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
           LIGHT, '<main><h1>原始数据集画像 · FreeInference × SWE-chat</h1>',
           '<p>统计本地上游发布表，保留原模型、原 token 字段及原序列。FreeInference 按请求计轮，SWE-chat 按带原生响应 ID 的主会话模型响应计轮。蓝绿色代表 FreeInference，橙色代表 SWE-chat。</p>',
           '<p>FreeInference 输入为发布表的 o200k 角色 token 总和，输出为 provider completion tokens。SWE-chat 总输入长度仅统计 Claude Code 中使用 Claude 模型且有有效 usage 的响应。总输入等于同一条 usage 记录的 input_tokens、cache_read_input_tokens 和 cache_creation_input_tokens 之和。同一响应有多个 usage 记录时取最大总输入。SWE-chat 输出仍统计所有客户端中有有效 output_tokens 的响应。</p>',
           '<p>SWE-chat 扫描主会话事件表中的全部客户端和模型。模型响应需要同时具备 session_id、原生 response_id 和事件顺序，再按会话和响应 ID 合并内容块。有至少一个符合条件的响应才计入模型响应会话数。长度统计另按有效 usage 取样，只在另一项 usage 或缓存字段证明 usage 存在时保留零值。</p>',
           table([['请求 / 可识别响应',fmt(a['n']),fmt(b['n'])],
                  ['有请求或响应的会话',fmt(a['sessions']),fmt(b['sessions'])],
                  ['观测起止 UTC',*[' 至 '.join(d['observed_utc']) for d in datasets]]]),
           f'<p>FreeInference 有 {a["zero_outputs"]:,} 条零输出请求，{a["over_128k"]:,} 条请求的输入与输出之和超过 131,072 token。SWE-chat 有 {b["native_response_ids"]:,} 个响应由原生响应 ID 合并，另有 {b["human_prompts"]:,} 条人工提示记录。</p>']
    prompts = b["user_prompts_per_session"]
    out.append(f'<p>SWE-chat 有人工提示的会话共 {fmt(prompts.get("n"))} 个，每会话人工提示数均值 {fmt(prompts.get("mean"))}，P50 为 {fmt(prompts.get("p50"))}，P95 为 {fmt(prompts.get("p95"))}。</p>')
    for i, axis in enumerate(("输入", "输出")):
        values = [d["turns"]["lengths"][i] for d in datasets]
        out.append(bars([v["bins"] for v in values], LABELS, f"整体{axis}长度分布（有效记录占比）"))
        out.append(table([[label,*[fmt(v["stats"].get(key)) for v in values]] for key,label in (
            ("n","有效记录数"),("mean","均值"),("p50","P50"),("p95","P95"),("p99","P99"),("max","最大值")
        )]))
    out.append(turn_section(datasets))
    out.append(user_turn_section(datasets))
    out.append('<h2>客户端和来源模型分布</h2><p>SWE-chat 使用会话索引的 agent 字段。'
               'FreeInference 请求表提供 provider 标签，未发布客户端字段。模型名称保留上游标签。'
               '占比分母为各自的全部有效请求或原生响应；使用多个模型的会话会出现在多行。</p>')
    out.append(distribution("SWE-chat · 客户端", b["clients"], b["client_sessions"], b["n"], "响应", "#f29f4c"))
    out.append(distribution("SWE-chat · 来源模型", b["models"], b["model_sessions"], b["n"], "响应", "#f29f4c"))
    out.append(distribution("FreeInference · 服务提供方", a["providers"], a["provider_sessions"], a["n"], "请求", "#269b95"))
    out.append(distribution("FreeInference · 来源模型", a["models"], a["model_sessions"], a["n"], "请求", "#269b95"))
    for client, counts in sorted(b["client_models"].items(), key=lambda item: -b["clients"][item[0]]):
        out.append(f'<details><summary>SWE-chat · {esc(client)} 的来源模型</summary>'
                   '<table><thead><tr><th>来源模型</th><th>响应数</th><th>客户端内占比</th></tr></thead><tbody>')
        out.extend(f'<tr><td>{esc(model)}</td><td>{fmt(count)}</td><td>{count/b["clients"][client]:.2%}</td></tr>'
                   for model, count in sorted(counts.items(), key=lambda item: -item[1]))
        out.append('</tbody></table></details>')
    out.append(f'<p>数据来源说明见本地发布材料 <code>{esc(free_root / "README.md")}</code> 与 <code>{esc(swe_root / "README.md")}</code>。</p></main></html>')
    return ''.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free-root", type=Path, default=DATA/"freeinference_agentic_trace/upstream")
    parser.add_argument("--swe-root", type=Path, default=DATA/"swe_chat/upstream")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"reports/upstream_characteristics.html")
    args = parser.parse_args()
    datasets = [freeinference(args.free_root), swechat(args.swe_root)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(datasets, args.free_root, args.swe_root), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(datasets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
