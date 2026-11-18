"""Arrival windows and session spans from original tables and replay sidecars."""
from collections import Counter, defaultdict
from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq

from characterize_datasets import stats

MINUTE = 60000
HOUR = 60 * MINUTE
DAY = 24 * HOUR
SPAN_EDGES = (0, 1, MINUTE, 5 * MINUTE, 30 * MINUTE, 2 * HOUR,
              8 * HOUR, DAY, 7 * DAY)
SPAN_LABELS = ["0", "<1分钟", "1–5分钟", "5–30分钟", "30分–2时", "2–8小时", "8–24小时", "1–7天", "≥7天"]


def record_span(spans, session, time):
    span = spans.get(session)
    if span is None:
        spans[session] = (time, time)
    else:
        spans[session] = (min(span[0], time), max(span[1], time))


def duration(spans):
    histogram = Counter(end - start for start, end in spans.values())
    bins = [0] * len(SPAN_EDGES)
    for value, count in histogram.items():
        bins[sum(value >= edge for edge in SPAN_EDGES) - 1] += count
    assert sum(bins) == len(spans)
    return dict(bins=bins, stats=stats(histogram))


def raw_free(files, window_ms):
    times = Counter()
    inputs = Counter()
    input_observations = 0
    spans = {}
    for file in map(Path, files):
        week = file.parents[1].name
        for batch in pq.ParquetFile(file).iter_batches(columns=["session_id", "start_ms", "role_tokens"]):
            for sid, time, roles in zip(*(column.to_pylist() for column in batch.columns)):
                if time is not None:
                    times[time] += 1
                    record_span(spans, (week, sid), time)
                    if roles and all(type(v) is int for v in roles.values()):
                        inputs[time] += sum(roles.values())
                        input_observations += 1
    return arrival("FreeInference", times, window_ms, inputs, input_observations), duration(spans)


def raw_swe(file, window_ms):
    responses = defaultdict(dict)
    columns = ["session_id", "response_id", "turn_number", "role", "turn_type", "model", "agent",
               "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
               "timestamp"]
    for batch in pq.ParquetFile(file).iter_batches(columns=columns, batch_size=65536):
        values = [column.to_pylist() for column in batch.columns[:-1]]
        values.append(batch.columns[-1].cast(pa.int64()).to_pylist())
        for sid, rid, order, role, kind, model, agent, x, y, cache_write, cache_read, microseconds in zip(*values):
            if not sid or not rid or order is None:
                continue
            if role != "assistant" and kind not in ("assistant_response", "tool_use"):
                continue
            record = responses[sid].setdefault(rid, [None, None, False, None])
            if microseconds is not None:
                time = microseconds // 1000
                record[0] = time if record[0] is None else min(time, record[0])
            if model:
                record[1] = model
            if agent == "Claude Code":
                record[2] = True
            cached = (cache_write or 0) + (cache_read or 0)
            if type(x) is int and (x > 0 or x == 0 and ((y or 0) > 0 or cached > 0)):
                observed = x + cached
                record[3] = observed if record[3] is None else max(observed, record[3])
    times = Counter()
    inputs = Counter()
    input_observations = 0
    spans = {}
    for sid, calls in responses.items():
        for time, model, claude_client, tokens in calls.values():
            if time is None:
                continue
            times[time] += 1
            record_span(spans, sid, time)
            if claude_client and (model or "").lower().startswith("claude-") and tokens is not None:
                inputs[time] += tokens
                input_observations += 1
    return arrival("SWE-chat", times, window_ms, inputs, input_observations), duration(spans)


def arrival(name, times, window_ms, inputs, input_observations):
    origin = min(times)
    counts = Counter()
    token_counts = Counter()
    for time, count in times.items():
        counts[(time - origin) // window_ms] += count
    for time, tokens in inputs.items():
        token_counts[(time - origin) // window_ms] += tokens
    return dict(name=name, n=sum(counts.values()), window_ms=window_ms,
                observation_ms=max(times) - origin + 1,
                input_observations=input_observations,
                bins=[[index * window_ms, counts[index], token_counts[index]] for index in range(max(counts) + 1)])


def replay_duration(dataset):
    spans = {}
    with (Path(dataset["bundle"]) / "source_requests.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            sid = (row.get("week", ""), row["session_id"])
            record_span(spans, sid, row["timestamp"])
    return duration(spans)
