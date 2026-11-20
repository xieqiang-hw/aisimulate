"""Build an offline dashboard from existing 8x throughput analysis caches."""
from pathlib import Path
import argparse
import json
import math
import re

import numpy as np

from throughput_execution import collect
from throughput_capacity_data import build_capacity_families

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=BASE / 'analysis/throughput_8x')
    parser.add_argument('--output', type=Path, default=BASE / 'throughput_8x.html')
    args = parser.parse_args()
    folder = args.analysis.resolve()
    report = json.loads((folder / 'results.json').read_text())
    trace = np.load(folder / 'cache/trace.npz')
    timestamps = trace['timestamp']
    first = float(timestamps.min())
    indexes = np.floor((timestamps - first) / 300000).astype(int)
    count = math.ceil((timestamps.max() - first + 1) / 300000)
    demand = {
        'requests': np.bincount(indexes, minlength=count).astype(int).tolist(),
        'input': np.bincount(indexes, weights=trace['input_length'], minlength=count).astype(np.int64).tolist(),
        'output': np.bincount(indexes, weights=trace['output_length'], minlength=count).astype(np.int64).tolist(),
    }
    assert sum(demand['requests']) == len(timestamps)
    assert sum(demand['input']) == report['demand'][1]['input_tokens']
    summaries = {(r['case'], r['window']): r for r in report['windows']}
    decoder = json.JSONDecoder()
    cases = []
    for case in sorted(report['whole_runs'], key=lambda r: (r['dram_gb'], r['read_gbps_per_tp'])):
        if case['dram_gb'] not in (64, 128, 256) or case['read_gbps_per_tp'] != 2:
            continue
        cached = dict(np.load(folder / 'cache' / (case['case'] + '.npz')))
        assert all(len(a) == count for a in cached.values())
        prediction = Path(case['source_result']).with_name('prediction.json')
        with prediction.open() as stream:
            header = stream.read(20000).split('"per_request":', 1)[0]

        def metric(name):
            key = f'"{name}":'
            if key not in header:
                return None
            return decoder.raw_decode(header.split(key, 1)[1].lstrip())[0]

        g3 = metric('g3_offload')
        item = dict(case, mean_ttft_ms=metric('mean_ttft_ms'), p99_ttft_ms=metric('p99_ttft_ms'),
                    mean_tpot_ms=metric('mean_tpot_ms'), sla_percent=metric('goodput_completed_requests')/metric('completed_requests')*100, reuse_ratio=metric('first_admission_prefix_cache_reused_ratio'),
                    ssd_read_tb=g3['read']['completed_bytes']/1e12,
                    main=summaries[(case['case'], 'main')], whole=summaries[(case['case'], 'whole_arrivals')])
        item['series'] = {k: v.astype(np.int64).tolist() for k, v in cached.items()}
        cases.append(item)
    payload = dict(demand=demand, cases=cases, bin_seconds=300,
                   main_days=report['main_end_days'], whole_days=report['whole_arrival_days'],
                   windows=report['demand'], peak=report['common_peak_window'][0],
                   source=report['bundle'], requests_sha256=report['requests_sha256'],
                   request_count=len(timestamps), unavailable=report['unavailable'])
    observations = collect(report, folder, count)
    for item in cases:
        item['execution'] = observations.get(item['case'])
    payload['execution_ready_count'] = len(observations)
    payload['unavailable'] = ['GPU hardware utilization', 'maximum sustainable throughput', 'execution traces for cases with execution=null']
    payload = build_capacity_families(report, trace, payload, folder)
    template = Path(__file__).with_name('throughput_dashboard.html').read_text()
    assert template.count('__DASHBOARD_DATA__') == 1
    html = template.replace('__DASHBOARD_DATA__', json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    temp = args.output.with_suffix('.html.tmp')
    temp.write_text(html, encoding='utf-8')
    temp.replace(args.output)
    total=sum(len(f['cases']) for f in payload['families'].values())
    done=sum(r['status']=='completed' for r in payload['capacity_overview'])
    print(f'Saved {args.output}: {len(payload["families"])} strengths, {total} configuration/strength combinations; capacity {done}/{len(payload["capacity_overview"])} complete; {len(html.encode()):,} bytes')


if __name__ == '__main__':
    main()
