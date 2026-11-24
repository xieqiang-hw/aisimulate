#!/usr/bin/env python3
"""Measure workload shape from the original-speed AISim replay bundles."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import zip_longest
import json
from pathlib import Path
import re

import numpy as np

DATA = Path.home() / 'project/data'
SWE = 'swe-claude-code'
CHAT = ('qwen-chat', 'qwen-api', 'qwen-thinking', 'qwen-coder',
        'mooncake-chat', 'mooncake-agent', 'mooncake-synthetic')
HASHES = re.compile(rb'"hash_ids"\s*:\s*\[([^\]]*)\]')
NUMBER = {name: re.compile(rb'"' + name.encode() + rb'"\s*:\s*(\d+(?:\.\d+)?)')
          for name in ('input_length', 'output_length', 'timestamp')}


def distribution(values):
    if len(values) == 0:
        return None
    a = np.asarray(values)
    quantiles = np.percentile(a, [50, 95, 99])
    return dict(zip(('p50', 'p95', 'p99'), map(float, quantiles))) | {
        'mean': float(a.mean()), 'max': float(a.max())}


def numeric(line, field):
    return NUMBER[field].search(line)[1]


def common_blocks(a, b):
    """Count equal leading integer IDs in the canonical comma-separated lists."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    count = a.count(b',', 0, lo)
    if lo and all(lo == len(s) or s[lo] == 44 for s in (a, b)) and a[lo - 1] != 44:
        count += 1
    return count


def inspect(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    inputs, outputs, times, session_gaps, pair_overlap = [], [], [], [], []
    sessions, types, last = Counter(), Counter(), {}
    started = ended = None
    equal_time_pairs = adjacent_ordinals = paired_input = reused16 = reused64 = 0
    with (path / 'requests.jsonl').open('rb') as requests, (path / 'source_requests.jsonl').open('rb') as sources:
        for idx, (line, source) in enumerate(zip_longest(requests, sources)):
            if line is None or source is None:
                raise ValueError(f'{path}: misaligned files')
            side = json.loads(source)
            t = float(numeric(line, 'timestamp'))
            if side['trace_row'] != idx or side['timestamp'] != t or (times and t < times[-1]):
                raise ValueError(f'{path}: invalid alignment or time at row {idx}')
            ni, no = (int(numeric(line, key)) for key in ('input_length', 'output_length'))
            inputs.append(ni)
            outputs.append(no)
            times.append(t)
            if source_time := side.get('first_response_event_at'):
                if started is None:
                    started = source_time
                ended = source_time
            if kind := side.get('request_type'):
                types[kind] += 1
            if session := side.get('session_id'):
                sessions[session] += 1
                ordinal = side.get('call_index', side.get('turn'))
                # Both converters write nonnegative decimal IDs, with either
                # comma-space separators (SWE) or compact separators (Chat).
                hashes = HASHES.search(line)[1].replace(b' ', b'')
                previous = last.get(session)
                if previous is not None:
                    before, prev_length, prev_hashes, prev_ordinal = previous
                    gap = (t - before) / 1000
                    session_gaps.append(gap)
                    equal_time_pairs += (gap == 0)
                    adjacent_ordinals += (ordinal is not None and prev_ordinal is not None and ordinal == prev_ordinal + 1)
                    blocks = min(common_blocks(prev_hashes, hashes), ni // 16, prev_length // 16)
                    paired_input += ni
                    reused16 += blocks * 16
                    reused64 += blocks // 4 * 64
                    pair_overlap.append(blocks * 16 / ni)
                last[session] = t, ni, hashes, ordinal
    count = len(times)
    if count != manifest['request_count'] or manifest['trace_block_size'] != 16:
        raise ValueError(f'{path}: unexpected request count or hash size')
    ts = np.asarray(times) / 1000
    dels = np.diff(ts)
    span = float(ts[-1] - ts[0])
    positions = ts - ts[0]
    bins = np.bincount(positions.astype(np.int64))
    minute = np.bincount((positions / 60).astype(np.int64))
    inp, out = np.asarray(inputs), np.asarray(outputs)
    contexts = inp + out
    s = np.asarray(list(sessions.values()))
    result = {
        'bundle': str(path), 'manifest_request_count': count, 'start': started, 'end': ended,
        'span_seconds': span, 'rps': count / span if span else None,
        'input': distribution(inp), 'output': distribution(out), 'context': distribution(contexts),
        'input_tokens': int(inp.sum()), 'output_tokens': int(out.sum()),
        'pct_context_gt_32k': float(100 * (contexts > 32768).mean()),
        'pct_context_gt_64k': float(100 * (contexts > 65536).mean()),
        'arrivals_per_second': distribution(bins), 'arrivals_per_minute': distribution(minute),
        'active_second_pct': float(100 * (bins > 0).mean()),
        'same_timestamp_adjacent_pct': float(100 * (dels == 0).mean()) if len(dels) else None,
        'positive_arrival_gap_ms': distribution(dels[dels > 0] * 1000),
        'types': dict(types), 'sessions': len(s) if len(s) else None,
        'session_requests': distribution(s), 'single_request_session_pct': float(100 * (s == 1).mean()) if len(s) else None,
        'same_session_pairs': len(pair_overlap), 'same_session_gap_seconds': distribution(session_gaps),
        'equal_time_session_pairs': equal_time_pairs, 'consecutive_ordinal_pairs': adjacent_ordinals,
        'paired_input_tokens': paired_input, 'prefix16_weighted_pct': 100 * reused16 / paired_input if paired_input else None,
        'prefix64_weighted_pct': 100 * reused64 / paired_input if paired_input else None,
        'prefix64_all_input_pct': 100 * reused64 / int(inp.sum()) if len(s) else None,
        'prefix16_request_ratio': distribution(pair_overlap),
        'prefix16_pairs_ge90_pct': float(100 * (np.asarray(pair_overlap) >= 0.9).mean()) if pair_overlap else None,
        'source': {k: manifest.get(k) for k in ('timestamp_policy', 'scope', 'source_block_size', 'model_mapping_policy',
                                               'arrival_kind', 'selection', 'diagnostics', 'time_range')},
    }
    print(f'{path.name}: {count:,} requests, {len(sessions):,} sessions', flush=True)
    return result


def charts(data, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(data)
    y = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for ax, key, title in [(axes[0, 0], 'input', 'Input tokens per request'),
                           (axes[0, 1], 'output', 'Output tokens per request')]:
        for k, marker, label, color in [('p50', 'o', 'P50', '#157b65'), ('p95', 'D', 'P95', '#bd7430'),
                                        ('p99', '|', 'P99', '#435f9c')]:
            ax.plot([data[n][key][k] for n in names], y, linestyle='none', marker=marker, label=label, color=color)
        ax.set_xscale('log')
        ax.set_title(title, loc='left', pad=31)
        ax.legend(ncol=3, frameon=False, loc='lower left', bbox_to_anchor=(0, 1), borderaxespad=0)
    ax = axes[1, 0]
    ax.plot([data[n]['rps'] for n in names], y, 'o', color='#157b65', label='Mean RPS')
    ax.plot([data[n]['arrivals_per_minute']['max'] / 60 for n in names], y, 'D', color='#bd7430', label='Peak minute / 60')
    ax.set_xscale('log')
    ax.set_title('Arrival intensity (requests/s)', loc='left', pad=31)
    ax.legend(ncol=2, frameon=False, loc='lower left', bbox_to_anchor=(0, 1), borderaxespad=0)
    ax = axes[1, 1]
    for i, name in enumerate(names):
        value = data[name]['prefix64_weighted_pct']
        if value is None:
            ax.text(1, i, 'N/A', va='center', color='#666')
        else:
            ax.barh(i, value, height=.55, color='#157b65')
            ax.text(value + 1, i, f'{value:.1f}%', va='center')
    ax.set(xlim=(0, 108), xticks=[0, 25, 50, 75, 100])
    ax.set_title('Adjacent same-session prefix (64-token aligned, %)', loc='left')
    for ax in axes.flat:
        ax.set_yticks(y, names)
        ax.set_ylim(len(names) - .5, -.5)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#ddd', alpha=.8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('SWE and Chat replay traces | original timing', fontsize=17, fontweight='bold')
    fig.savefig(target, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--swe', type=Path, default=DATA / 'swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k')
    parser.add_argument('--chat', type=Path, default=DATA / 'chat/derived/mooncake-v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundles = [(SWE, args.swe), *((name, args.chat / name) for name in CHAT)]
    results = {name: inspect(path) for name, path in bundles}
    snapshot = {'generated': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'), 'datasets': results}
    (args.output / 'metrics.json').write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n')
    charts(results, args.output / 'trace_comparison.png')


if __name__ == '__main__':
    main()
