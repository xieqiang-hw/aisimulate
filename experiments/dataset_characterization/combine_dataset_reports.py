"""Build the combined upstream and replay HTML report from validated aggregates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree

from characterize_datasets import LABELS, TURN_LABELS, bars, esc, fmt, heatmap, lineplot
from combined_temporal import HOUR, SPAN_LABELS, raw_free, raw_swe, replay_duration


def table(headers, rows):
    columns = [scope_attr("raw") if label.startswith("原始") else
               scope_attr("replay") if label.startswith("实验输入") else "" for label in headers]
    head = "".join(f"<th scope='col'{scope}>{esc(label)}</th>" for label, scope in zip(headers, columns))
    body = "".join("<tr>" + "".join(f"<td{scope}>{esc(value)}</td>" for value, scope in zip(row, columns))
                   + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def scope_attr(scope):
    return f' data-scope="{scope}"' if scope else ""


def details(title, content, *, scope=None):
    return f"<details{scope_attr(scope)}><summary>{esc(title)}</summary>{content}</details>"


def pairs(original, replay):
    return ("<div class='pair paired-cohorts'><div data-scope='raw'><p class='cohort'>原始数据</p>" + original
            + "</div><div data-scope='replay'><p class='cohort'>实验输入数据</p>" + replay + "</div></div>")


def ranked(counts, total, unit, sessions=None):
    rows = [[name, fmt(n), f"{n / total:.2%}", fmt(sessions.get(name)) if sessions else "—"]
            for name, n in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    headers = ("标签", unit, "占比", "覆盖会话数")
    preview = table(headers, rows[:5])
    if len(rows) > 5:
        preview += details(f"查看全部 {len(rows)} 个标签", table(headers, rows))
    return preview


def free_replay_metadata(bundle):
    providers, by_provider, by_model, sessions = Counter(), {}, {}, set()
    with (Path(bundle) / "source_requests.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            sid = (row["week"], row["session_id"])
            provider = row.get("provider") or "未知"
            model = row.get("source_model") or "未知"
            providers[provider] += 1
            by_provider.setdefault(provider, set()).add(sid)
            by_model.setdefault(model, set()).add(sid)
            sessions.add(sid)
    return providers, {k: len(v) for k,v in by_provider.items()}, {k: len(v) for k,v in by_model.items()}, len(sessions)


def paired_bars(raw_bins, replay_bins, labels, title):
    maximum = 0
    for counts in raw_bins + replay_bins:
        if sum(counts):
            maximum = max(maximum, max(counts) / sum(counts))
    return pairs(bars(raw_bins, labels, title, scale=maximum),
                 bars(replay_bins, labels, title, scale=maximum))


def metrics(raw_values, replay_values, measures):
    return table(("指标", "原始 FreeInference", "原始 SWE-chat", "实验输入 FreeInference", "实验输入 SWE-chat"), [
        [label, *[fmt(value.get(key)) for value in raw_values + replay_values]]
        for key, label in measures
    ])


LENGTH_METRICS = (("n", "有效样本数"), ("mean", "均值"), ("p50", "P50"),
                  ("p90", "P90"), ("p95", "P95"), ("p99", "P99"), ("max", "最大值"))


def length_section(raw_axes, replay_axes, axis):
    title = f"{axis} token 长度"
    chart = paired_bars([v["bins"] for v in raw_axes], [v["bins"] for v in replay_axes], LABELS, title)
    return chart + details(f"{axis}长度统计表", metrics([v["stats"] for v in raw_axes],
                                                          [v["stats"] for v in replay_axes], LENGTH_METRICS))


def temporal_sections(raw_arrival, replay, raw_duration, input_duration, raw_counts, input_counts):
    out = ['<h3>到达速率与突发性</h3>']
    datasets = raw_arrival + replay
    domain = max(d["observation_ms"] for d in datasets)
    peak = max(row[1] * 1000 / d["window_ms"] for d in datasets for row in d["bins"])
    original = lineplot(raw_arrival, shared_domain_ms=domain, shared_peak=peak)
    experimental = lineplot(replay, shared_domain_ms=domain, shared_peak=peak)
    out.append(pairs(original, experimental).replace("距各数据集回放起点", "距各数据集观测起点"))
    out.append('<p>两张图使用相同的时间和速率尺度。横轴从各数据集自身的观测起点计，'
               '同比例位置对应相同的相对天数。SWE-chat 使用已归并响应的首个可用事件时间作为到达代理。</p>')
    out.append(details("查看到达速率统计", metrics(
        [dict(n=d["n"], days=d["observation_ms"]/86400000, rps=d["n"]*1000/d["observation_ms"]) for d in raw_arrival],
        [dict(n=d["n"], days=d["observation_ms"]/86400000, rps=d["n"]*1000/d["observation_ms"]) for d in replay],
        (("n","有时间戳的请求 / 响应数"), ("days","观测天数"), ("rps","平均请求 / 秒")))))
    out.append('<h3>输入 token 总量随时间变化</h3>')
    token_peak = max(row[2] for d in datasets for row in d["bins"]) / 1000000
    original_tokens = lineplot(raw_arrival, shared_domain_ms=domain, shared_peak=token_peak, input_tokens=True)
    experimental_tokens = lineplot(replay, shared_domain_ms=domain, shared_peak=token_peak, input_tokens=True)
    out.append(pairs(original_tokens, experimental_tokens).replace("距各数据集回放起点", "距各数据集观测起点"))
    out.append('<p>同一窗口内的输入长度直接相加。同一会话的历史上下文可能随多个请求重复计入。</p>')
    coverage = raw_arrival[1]["input_observations"] / raw_arrival[1]["n"]
    out.append(f'<p data-scope="raw">原始 SWE-chat 只加总有有效总输入计数的 Claude Code / Claude 响应，'
               f'覆盖 {raw_arrival[1]["input_observations"]:,} 条，占有时间戳响应的 {coverage:.2%}。'
               '未观测到的输入不加到窗口总量中。</p>')
    out.append(details("查看输入 token 时间统计", metrics(
        [dict(n=d["input_observations"], total=sum(row[2] for row in d["bins"]),
              peak=max(row[2] for row in d["bins"])) for d in raw_arrival],
        [dict(n=d["n"], total=sum(row[2] for row in d["bins"]),
              peak=max(row[2] for row in d["bins"])) for d in replay],
        (("n", "有时间戳和输入计数的请求 / 响应数"), ("total", "已观测输入 token 总量"), ("peak", "峰值窗口输入 token 数")))))
    out.append('<h3>会话持续时间</h3><p>按会话首条到末条有时间戳的模型调用计算跨度。'
               '只有一个时间点或多个事件时间相同的会话计为 0。FreeInference 两侧按周次和会话 ID 合并，SWE-chat 按会话 ID。'
               '跨度包含同一会话内的空闲时间。</p>')
    out.append(paired_bars([d["bins"] for d in raw_duration], [d["bins"] for d in input_duration],
                           SPAN_LABELS, "会话持续时间分布"))
    measures = (("n", "有时间戳的会话数"), ("coverage", "时间戳覆盖率"),
                ("mean", "均值 / 小时"), ("p50", "P50 / 小时"), ("p90", "P90 / 小时"),
                ("p95", "P95 / 小时"), ("p99", "P99 / 小时"), ("max", "最长 / 小时"))

    def in_hours(values, totals):
        rows = []
        for value, total in zip(values, totals):
            row = {key: value["stats"][key]/HOUR for key in ("mean", "p50", "p90", "p95", "p99", "max")}
            row["n"] = value["stats"]["n"]
            row["coverage"] = f'{row["n"]/total:.2%}'
            rows.append(row)
        return rows

    values = [in_hours(raw_duration, raw_counts), in_hours(input_duration, input_counts)]
    rows = [[label, *[v[key] if key == "coverage" else fmt(v[key]) for v in values[0] + values[1]]]
            for key, label in measures]
    out.append(details("查看会话持续时间统计", table(
        ("指标", "原始 FreeInference", "原始 SWE-chat", "实验输入 FreeInference", "实验输入 SWE-chat"), rows)))
    return ''.join(out)


def render(raw, replay, raw_arrival, raw_duration, input_duration):
    a, b = raw
    x, y = replay
    providers, provider_sessions, model_sessions, session_count = free_replay_metadata(x["bundle"])
    if sum(providers.values()) != x["n"]:
        raise ValueError("FreeInference provider counts do not match replay aggregate")
    diagnostic = y["manifest"]["diagnostics"]
    selection = x["manifest"]["selection"]["counts"]
    target = y["manifest"]["model_mapping"]["target_model"]
    out = ['''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FreeInference 与 SWE-chat 数据集对照</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#172439;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1450px;margin:auto;padding:40px 26px 70px}h1{font-size:clamp(30px,5vw,47px);line-height:1.12;margin:16px 0}
h2{border-bottom:2px solid #c4dcd8;padding-bottom:9px;margin:44px 0 18px;color:#075d59}h3{margin:20px 0 9px}
p,figcaption{color:#41556d}nav{display:flex;gap:12px;flex-wrap:wrap}nav a{color:#006660;border:1px solid #c4dcd8;background:#fff;border-radius:6px;padding:5px 10px}
.pair{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;margin:16px 0}
.pair>div,figure,details,.note{background:#fff;border:1px solid #d9e3ed;border-radius:10px;padding:15px;margin:0 0 12px}
.pair figure{border:0;padding:0;margin:0}.pair h3,figure h3{margin:4px 0 10px}.cohort{font-weight:700;color:#075d59;margin:0 0 10px}
table{border-collapse:collapse;width:100%;text-align:left;font-variant-numeric:tabular-nums;background:#fff}
.scroll{overflow:auto}th,td{padding:8px 10px;border-bottom:1px solid #e0e8ef;overflow-wrap:anywhere}th{background:#eaf2f6}
summary{cursor:pointer;font-weight:600;color:#075d59}details .scroll{margin-top:12px}
svg{width:100%;height:auto}svg text{fill:#344960;font:12px system-ui}code{overflow-wrap:anywhere}.free{color:#006f6a}.swe{color:#9a4c04}
.view-control{display:flex;align-items:center;flex-wrap:wrap;gap:7px;border:0;padding:0;margin:16px 0}
.view-control legend{font-weight:650;margin-bottom:5px}.view-control label{background:#fff;border:1px solid #c4dcd8;border-radius:6px;padding:5px 11px;cursor:pointer}
.view-control label:has(input:checked){background:#dff1ed;border-color:#428a80;color:#075d59;font-weight:650}
body[data-view="raw"] [data-scope="replay"],body[data-view="replay"] [data-scope="raw"]{display:none}
body:not([data-view="both"]) .paired-cohorts{grid-template-columns:minmax(0,1fr)}
@media(max-width:790px){main{padding:22px 14px}.pair{grid-template-columns:1fr}}
</style></head><body data-view="raw"><main><h1>FreeInference 与 SWE-chat 数据集对照</h1>
<nav aria-label="报告目录"><a href="#overview">总览、口径与筛选</a><a href="#sources">来源分布</a><a href="#details">详细统计</a></nav>
<fieldset class="view-control"><legend>显示数据</legend>
<label><input type="radio" name="cohort-view" value="raw" checked> 仅原始数据</label>
<label><input type="radio" name="cohort-view" value="replay"> 仅实验输入数据</label>
<label><input type="radio" name="cohort-view" value="both"> 原始数据 + 实验输入数据</label>
</fieldset>
<h2 id="overview">原始数据与实验输入总览</h2>''']
    overview_headers = ("指标", "原始数据", "实验输入数据")
    out.append('<div class="pair overview-tables"><div><h3>FreeInference</h3>' + table(overview_headers, [
        ("模型请求数", fmt(a["n"]), fmt(x["n"])),
        ("有模型请求的会话数", fmt(a["sessions"]), fmt(session_count)),
    ]) + '</div><div><h3>SWE-chat</h3>' + table(overview_headers, [
        ("可识别响应数 / 实验请求数", fmt(b["n"]), fmt(y["n"])),
        ("有响应或请求的会话数", fmt(b["sessions"]), fmt(y["sessions"])),
    ]) + '</div></div>')
    out.append(
        '<p>会话总览按「周次 + 会话 ID」为 FreeInference 去重，按「会话 ID」为 SWE-chat 去重。'
        '一次用户提示可以触发多次模型调用，用户交互轮数另列。</p>'
        '<div data-scope="raw"><p>FreeInference 原始表一行是一次模型请求，覆盖 16 周。SWE-chat 原始主会话表按'
        ' <code>session_id + response_id</code> 归并模型响应，要求原生响应 ID 和有效事件顺序。'
        '</p><p>FreeInference 输入为 o200k 角色 token 总量，输出为源模型 completion tokens。'
        'SWE-chat 原始总输入只覆盖有有效 usage 的 Claude Code / Claude 响应，含缓存读写 token；'
        '原始输出覆盖所有有有效输出 usage 的已识别响应。</p></div>'
        '<div data-scope="replay"><p>实验输入统计采用两个实际使用的原速基础包。'
        f'FreeInference 的会话轮数图按「周次 + 来源模型 + 会话 ID」分组（{x["sessions"]:,} 组）。</p>'
        '<p>FreeInference 沿用源数据的 o200k 输入 token 数和 completion tokens。'
        'SWE-chat 对重建的历史和输出用 GLM tokenizer 重新分词。</p></div>'
        '<p>各长度指标按有效样本计算，分位数采用 nearest-rank。</p>'
    )
    out.append('<div data-scope="replay"><h3 id="selection">实验输入的筛选与转换</h3><div class="pair">'
        f'<div><h3>FreeInference</h3><p>原始 {a["n"]:,} 条请求中，排除 '
        f'{selection["zero_output_excluded"]:,} 条零输出请求和 {selection["context_excluded"]:,} 条'
        f'输入加输出超过 131,072 token 的请求，保留 {x["n"]:,} 条。'
        f'统一映射到 {esc(target)}，长度沿用来源计数。</p></div>'
        f'<div><h3>SWE-chat</h3><p>只重建有原始日志的 Claude Code 主会话。交错响应通过 requestId 和'
        '工具调用关联核验，歧义调用舍弃，遇到明确的独立上下文起点才恢复。'
        f'从 {diagnostic["attempted_sessions"]:,} 个会话提取 {diagnostic["extracted_calls"]:,} 次调用，'
        f'其中 {diagnostic["over_context_limit"]:,} 条超过 131,072 token，'
        f'{diagnostic["calls_missing_initial_history"]:,} 条缺少初始历史。'
        f'最终以 {esc(target)} tokenizer 生成 {y["n"]:,} 条实验输入请求。</p></div></div></div>')
    out.append('<h2 id="sources">客户端、服务与来源模型</h2>')
    out.append('<p>FreeInference 上游提供匿名服务标签；SWE-chat 用会话索引的客户端标签。来源模型保留原标签，'
               '包括「未知」。同一会话使用多个模型时可覆盖多个模型类别。</p>')
    out.append('<h3>SWE-chat 客户端</h3>' + pairs(
        ranked(b["clients"], b["n"], "响应数", b["client_sessions"]),
        ranked({"Claude Code": y["n"]}, y["n"], "请求数", {"Claude Code": y["sessions"]})))
    out.append('<h3>FreeInference 服务提供方</h3>' + pairs(
        ranked(a["providers"], a["n"], "请求数", a["provider_sessions"]),
        ranked(providers, x["n"], "请求数", provider_sessions)))
    out.append('<h3>FreeInference 来源模型</h3>' + pairs(
        ranked(a["models"], a["n"], "请求数", a["model_sessions"]),
        ranked(x["models"], x["n"], "请求数", model_sessions)))
    out.append('<h3>SWE-chat 来源模型</h3>' + pairs(
        ranked(b["models"], b["n"], "响应数", b["model_sessions"]),
        ranked(y["models"], y["n"], "请求数")))
    out.append(details("查看原始 SWE-chat 各客户端的模型分布", "".join(
        details(client, ranked(counts, b["clients"][client], "响应数"))
        for client, counts in sorted(b["client_models"].items(), key=lambda item: -b["clients"][item[0]])
    ), scope="raw"))
    out.append('<h2 id="details">详细统计图表</h2><p><strong class="free">■ FreeInference</strong>　'
               '<strong class="swe">■ SWE-chat</strong>。成对柱状图使用相同的区间与频率尺度。'
               '各柱表示相应数据集有效样本的占比。</p>')
    out.append('<h3>整体输入与输出长度</h3>')
    for axis, label in enumerate(("输入", "输出")):
        out.append(length_section([d["turns"]["lengths"][axis] for d in raw],
                                  [d["turns"]["lengths"][axis] for d in replay], label))
    out.append('<h3>每会话轮数</h3>')
    for title, key, labels in (("模型调用轮数", "turns", TURN_LABELS),
                               ("用户交互轮数", "user_turns", ["0","1","2–5","6–10","11–20","21–50","51–100","101+"])):
        out.append(paired_bars([d[key]["distribution"] for d in raw],
                               [d[key]["distribution"] for d in replay], labels, title))
        text = ""
        if key == "user_turns":
            for cohort, datasets in (("原始数据", raw), ("实验输入", replay)):
                for d in datasets:
                    method = d[key]["method"].replace("回放请求", "实验输入请求").replace(
                        "回放调用", "实验输入调用").replace("回放会话", "实验输入会话")
                    scope = "raw" if cohort == "原始数据" else "replay"
                    text += f'<p{scope_attr(scope)}>{cohort} · {esc(d["name"])}：{esc(method)}</p>'
        out.append(details(f"{title}统计表及口径", metrics([d[key]["stats"] for d in raw],
                                                            [d[key]["stats"] for d in replay], LENGTH_METRICS) + text))
    out.append('<h3>分轮段长度</h3><p>轮段按源调用顺序分组，实验输入保留被过滤调用造成的轮次空缺。</p>')
    for i, label in enumerate(("第 1–5 轮", "第 6–10 轮", "第 11 轮及以后（10+）")):
        out.append(f'<h4>{label}</h4>')
        for axis, name in enumerate(("输入", "输出")):
            out.append(length_section([d["turns"]["groups"][str(i)]["axes"][axis] for d in raw],
                                      [d["turns"]["groups"][str(i)]["axes"][axis] for d in replay], name))
    out.append(temporal_sections(raw_arrival, replay, raw_duration, input_duration,
                                 [a["sessions"], b["sessions"]], [session_count, y["sessions"]]))
    out.append(details("查看实验输入突发性统计", table(("指标","FreeInference","SWE-chat"), [
        ["每窗请求数 P99", *[fmt(d["summary"]["requests_per_window"]["p99"]) for d in replay]],
        ["每窗请求数变异系数", *[fmt(d["summary"]["requests_per_window"]["cv"]) for d in replay]],
        ["到达间隔 P50 / ms", *[fmt(d["summary"]["gap"]["p50"]) for d in replay]],
        ["到达间隔 P99 / ms", *[fmt(d["summary"]["gap"]["p99"]) for d in replay]],
    ]), scope="replay"))
    out.append(details("查看实验输入的输入与输出长度联合热图",
                       '<div class="pair"><div>' + heatmap(x) + '</div><div>' + heatmap(y) + '</div></div>', scope="replay"))
    out.append(details("查看实验输入会话内请求间隔与缓存统计", table(("指标","FreeInference","SWE-chat"), [
        ["会话内请求间隔 P50 / ms", *[fmt(d["summary"]["session_gap"]["p50"]) for d in replay]],
        ["provider 缓存读 token / prompt token", f'{x["cached_tokens"]/x["provider_prompt_tokens"]:.2%}' if x["provider_prompt_tokens"] else "—", "—"],
    ]), scope="replay"))
    out.append(details("查看来源文件和原始报告链接",
        '<p><a href="upstream_characteristics.html">完整原始数据报告</a> · '
        '<a href="dataset_characteristics.html">完整实验输入报告</a></p>'
        '<div data-scope="raw"><p>原始来源文件：</p><ul>' +
        ''.join(f'<li><code>{esc(path)}</code></li>' for d in raw for path in d["files"]) +
        '</ul></div><div data-scope="replay"><p>实验输入数据包：</p><ul>' +
        ''.join(f'<li><code>{esc(d["bundle"])}</code></li>' for d in replay) + '</ul></div>'))
    out.append('''</main><script>
const viewOptions = document.querySelectorAll('input[name="cohort-view"]');
for (const option of viewOptions) {
  option.addEventListener('change', () => {
    if (option.checked) document.body.dataset.view = option.value;
  });
}
const initialView = document.querySelector('input[name="cohort-view"]:checked');
if (initialView) document.body.dataset.view = initialView.value;
</script></body></html>''')
    return ''.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", default=Path(__file__).resolve().parent / "reports", type=Path)
    args = parser.parse_args()
    raw = json.loads((args.reports / "upstream_characteristics.json").read_text(encoding="utf-8"))
    replay = json.loads((args.reports / "dataset_characteristics.json").read_text(encoding="utf-8"))["datasets"]
    window = replay[0]["window_ms"]
    free_arrival, free_duration = raw_free(raw[0]["files"], window)
    print("FreeInference upstream time coverage", free_arrival["n"], free_duration["stats"]["n"], flush=True)
    swe_arrival, swe_duration = raw_swe(raw[1]["files"][0], window)
    print("SWE-chat upstream time coverage", swe_arrival["n"], swe_duration["stats"]["n"], flush=True)
    input_duration = [replay_duration(d) for d in replay]
    assert free_arrival["n"] <= raw[0]["n"] and swe_arrival["n"] <= raw[1]["n"]
    assert all(sum(row[1] for row in d["bins"]) == d["n"] for d in [free_arrival, swe_arrival] + replay)
    assert sum(row[2] for row in free_arrival["bins"]) == raw[0]["turns"]["lengths"][0]["stats"]["total"]
    for d in replay:
        assert sum(row[2] for row in d["bins"]) == d["summary"]["input"]["total"]
    result = render(raw, replay, [free_arrival, swe_arrival], [free_duration, swe_duration], input_duration)
    charts = [ElementTree.fromstring(svg) for svg in re.findall(r"<svg\b.*?</svg>", result, re.S)]
    assert len(charts) == 28 and " open" not in "".join(re.findall(r"<details[^>]*>", result))
    target = args.reports / "combined_dataset_characteristics.html"
    with NamedTemporaryFile(dir=target.parent, mode="w", encoding="utf-8", delete=False) as temp:
        temp.write(result)
    Path(temp.name).replace(target)
    print(f"{target}: {len(charts)} valid SVGs")


if __name__ == "__main__":
    main()
