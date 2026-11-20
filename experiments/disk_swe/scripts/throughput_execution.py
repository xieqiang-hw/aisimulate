"""Aggregate completed capacity baseline traces into five-minute dashboard data."""
from pathlib import Path
import hashlib
import json
import subprocess

TOOL = Path.home() / 'build/trace-summary/release/aisim-trace-summary'
ROOT = Path.home() / 'project/results/swe-chat-glm52-throughput-capacity'
VERSION = 1


def header(path):
    with path.open() as stream:
        text = stream.read(20000).split('"per_request":', 1)[0]
    result = {}
    for key in ('completed_requests', 'committed_prefill_tokens', 'mean_ttft_ms', 'p99_ttft_ms'):
        result[key] = json.JSONDecoder().raw_decode(text.split(f'"{key}":', 1)[1].lstrip())[0]
    return result


def signature(prediction):
    return [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(prediction.iterdir())
            if p.name.startswith(('iteration_metrics', 'scheduler_states', 'scheduling_events', 'scheduler_trace_metadata'))]


def collect(report, folder, bin_count):
    if not TOOL.is_file():
        raise FileNotFoundError(f'Native aggregator missing: {TOOL}')
    tool_hash = hashlib.sha256(TOOL.read_bytes()).hexdigest()
    original = {c['dram_gb']: c for c in report['whole_runs'] if c['read_gbps_per_tp'] == 2}
    candidates = {}
    for plan_path in sorted(ROOT.glob('*/sweep.json')):
        plan = json.loads(plan_path.read_text())
        selection = plan.get('selection', {})
        if selection.get('max_requests') != 0 or selection.get('bundle') != report['bundle']:
            continue
        for case in plan['cases']:
            if case['status'] == 'completed' and case['speedup'] == 1 and case['dram_gb'] in original:
                prediction = Path(case['prediction'])
                run = json.loads(prediction.with_name('run.json').read_text())
                if run.get('source_trace_sha256') != report['requests_sha256']:
                    continue
                old = original[case['dram_gb']]
                if header(prediction / 'prediction.json') != header(Path(old['source_result']).with_name('prediction.json')):
                    print(f'Execution metrics not merged: baseline changed for DRAM {case["dram_gb"]}', flush=True)
                    continue
                candidates[case['dram_gb']] = (case, prediction)
    cache_dir = folder / 'cache/execution'
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = {}
    for dram, (case, prediction) in candidates.items():
        target = cache_dir / (case['name'] + '.json')
        expected = json.loads(json.dumps(signature(prediction)))
        cached = json.loads(target.read_text()) if target.is_file() else {}
        if (cached.get('export_version') != VERSION or cached.get('source_signature') != expected
                or cached.get('tool_sha256') != tool_hash):
            print(f'Aggregating five-minute execution data: DRAM {dram} GB', flush=True)
            subprocess.run(['nice', '-n', '10', str(TOOL), '--bin-seconds', '300', '--overview-only',
                            str(prediction), str(target)], check=True)
            cached = json.loads(target.read_text())
            cached.update(export_version=VERSION, source_signature=expected, tool_sha256=tool_hash,
                          source_directory=str(prediction))
            target.write_text(json.dumps(cached, separators=(',', ':')))
        observation = make_observation(cached, bin_count, prediction,
                                       original[dram]['committed_prefill_tokens'], original[dram]['total_output_tokens'])
        output[original[dram]['case']] = observation
        streams = observation['streams']
        print(f'Validated DRAM {dram}: P computed {streams["prefill"]["total_prefill_tokens"]:,}; '
              f'D generated {streams["decode"]["total_generated_tokens"]:,}', flush=True)
    return output

def make_observation(cached, bin_count, prediction, report_prefill_tokens, report_output_tokens):
    assert cached['overview_bin_seconds'] == 300
    assert cached['focus'] == []
    assert len(cached['overview']) == 2, 'Expected one P and one D stream'
    streams = {}
    for pool in cached['overview']:
        role = pool['identity']['role']
        assert pool['identity']['dp_rank'] == 0
        columns = cached['overview_columns']
        values = {name: [0] * bin_count for name in ['busy_percent', 'active_mean_bs', 'max_bs',
                   'prefill_tokens', 'generated_tokens', 'h2d_loading_max', 'g3_read_jobs_max',
                   'waiting_admission_max', 'preemptions']}
        values['active_mean_bs'] = [None] * bin_count
        for row in pool['rows']:
            r = dict(zip(columns, row))
            index = int(row[0])
            if index >= bin_count:
                assert r['prefill_tokens'] == 0 and r['decode_generated_tokens'] == 0
                continue
            assert 0 <= r['busy_percent'] <= 100.000001
            values['busy_percent'][index] = r['busy_percent']
            values['active_mean_bs'][index] = r['active_time_weighted_mean_bs'] if r['busy_percent'] > 0 else None
            for key in ['max_bs','prefill_tokens','h2d_loading_max','g3_read_jobs_max','waiting_admission_max','preemptions']:
                values[key][index] = r[key]
            values['generated_tokens'][index] = r['decode_generated_tokens']
        values['iterations'] = pool['iterations']
        values['total_prefill_tokens'] = pool['prefill_tokens']
        values['total_generated_tokens'] = pool['decode_generated_tokens']
        values['bs_limit'] = pool['identity']['max_sequences']
        streams[role] = values
    assert set(streams) == {'prefill', 'decode'}
    p_tokens = streams['prefill']['total_prefill_tokens']
    d_tokens = streams['decode']['total_generated_tokens']
    assert sum(streams['prefill']['prefill_tokens']) == p_tokens
    assert sum(streams['decode']['generated_tokens']) == d_tokens
    # The report counts completed Prefill work across BOTH stages. The
    # scheduler's prefill_new_tokens field is phase-filtered on Decode workers.
    assert p_tokens <= report_prefill_tokens
    assert d_tokens <= report_output_tokens
    reconciliation = dict(
        report_prefill_tokens=report_prefill_tokens,
        p_recorded_prefill_tokens=p_tokens,
        other_stage_prefill_tokens=report_prefill_tokens-p_tokens,
        report_output_tokens=report_output_tokens,
        decode_recorded_output_tokens=d_tokens,
        output_tokens_outside_decode_only_counter=report_output_tokens-d_tokens)
    return dict(source_directory=str(prediction), bin_seconds=300, streams=streams, reconciliation=reconciliation)
