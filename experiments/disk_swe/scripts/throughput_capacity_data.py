"""Prepare dashboard families for time-compression experiments on one 8x trace."""
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import numpy as np

from throughput_execution import ROOT, TOOL, signature, make_observation

PATTERN = re.compile(rb'"(arrival_time_ms|terminal_time_ms|prefill_admit_ms|output_length)"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8')
    temp.replace(path)


def report_header(prediction):
    with (prediction / 'prediction.json').open() as stream:
        text = stream.read(20000).split('"per_request":', 1)[0]
    result = {}
    keys = ['completed_requests','duration_ms','committed_prefill_tokens','mean_ttft_ms','p99_ttft_ms',
            'mean_tpot_ms','goodput_completed_requests','first_admission_prefix_cache_reused_ratio','g3_offload']
    for key in keys:
        result[key] = json.JSONDecoder().raw_decode(text.split(f'"{key}":',1)[1].lstrip())[0]
    return result


def request_series(case, source, destination):
    prediction = Path(case['prediction'])
    path = prediction / 'requests.jsonl'
    identity = [str(path), path.stat().st_size, path.stat().st_mtime_ns]
    meta_path = destination / (case['name'] + '.requests.json')
    npz_path = meta_path.with_suffix('.npz')
    if meta_path.is_file() and npz_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get('identity') == identity and meta.get('version') == 1:
            return meta, dict(np.load(npz_path))
    print('Request windows:',case['name'],flush=True)
    fields = ['arrival_time_ms','terminal_time_ms','prefill_admit_ms','output_length']
    cols = {key: array('d') for key in fields}
    with path.open('rb') as stream:
        for line in stream:
            row = dict(PATTERN.findall(line))
            for key in fields:
                cols[key].append(float(row[key.encode()]))
    a, t, p, output = [np.asarray(cols[k]) for k in fields]
    expected = np.sort(source['timestamp'] / case['speedup'])
    assert len(a) == len(expected)
    assert np.allclose(np.sort(a),expected,rtol=0,atol=1e-6)
    assert output.sum() == source['output_length'].sum()
    assert np.all(t>=a) and np.all(p>=a) and np.all(p<=t)
    sa,st,sp = np.sort(a),np.sort(t),np.sort(p)
    peak = np.searchsorted(sa,sa,side='right')-np.searchsorted(st,sa,side='right')
    count = math.ceil(max(t.max()+1,a.max()+1)/300000)+1
    edges = np.arange(1,count+1)*300000
    series = dict(completed_requests=np.bincount(np.floor(t/300000).astype(int),minlength=count),
                  completed_output_tokens=np.bincount(np.floor(t/300000).astype(int),weights=output,minlength=count).astype(np.int64),
                  unfinished_requests=np.searchsorted(sa,edges,side='right')-np.searchsorted(st,edges,side='right'),
                  p_preadmission_waiting=np.searchsorted(sa,edges,side='right')-np.searchsorted(sp,edges,side='right'))
    windows = {}
    for label,end in [('main',case['main_window_end_ms']),('whole',(source['timestamp'].max()+1)/case['speedup'])]:
        arrived=int(np.sum(a<end));completed=int(np.sum(t<end))
        windows[label]=dict(arrived_requests=arrived,completed_requests=completed,arrival_request_per_s=arrived/(end/1000),
            completion_request_per_s=completed/(end/1000),backlog_end=arrived-completed,
            peak_inflight_requests=int(peak[sa<end].max()),backlog_start=0)
    meta=dict(identity=identity,version=1,count=count,windows=windows,total_output_tokens=int(output.sum()))
    np.savez_compressed(npz_path,**series)
    save(meta_path,meta)
    return meta,series


def execution_summary(case, folder, header):
    prediction=Path(case['prediction'])
    target=folder/f'p{case["dram_gb"]}gb_speed{case["speedup"]:g}_execution.json'
    meta_path=target.with_suffix('.meta.json')
    fingerprint=dict(source_directory=str(prediction),source_signature=json.loads(json.dumps(signature(prediction))),
                     tool_sha256=hashlib.sha256(TOOL.read_bytes()).hexdigest(),bin_seconds=300)
    valid=target.is_file() and meta_path.is_file() and json.loads(meta_path.read_text())==fingerprint
    if not valid:
        print('Execution windows:',case['name'],flush=True)
        subprocess.run(['nice','-n','10',str(TOOL),'--bin-seconds','300','--overview-only',str(prediction),str(target)],check=True)
        save(meta_path,fingerprint)
    native=json.loads(target.read_text())
    assert abs(native['end_s']-header['duration_ms']/1000)<0.001
    return native


def overview_row(case,item,main_days):
    out={k:case[k] for k in ['name','dram_gb','speedup','status']}
    if item is None:
        return out
    out.update(main_completion_rps=item['main']['completion_request_per_s'],
               main_end_backlog=item['main']['backlog_end'],main_peak_unfinished=item['main']['peak_inflight_requests'],
               mean_ttft_s=item['mean_ttft_ms']/1000,p99_ttft_s=item['p99_ttft_ms']/1000,
               mean_tpot_ms=item['mean_tpot_ms'],sla_percent=item['sla_percent'])
    n=math.floor(main_days*288)
    if item.get('execution'):
        for role in ['prefill','decode']:
            stream=item['execution']['streams'][role]
            out[role+'_busy_percent']=sum(stream['busy_percent'][:n])/n
            out[role+'_preemptions']=sum(stream['preemptions'][:n])
    return out


def build_capacity_families(report,source,baseline,folder):
    families={'1':baseline}
    plans=[]
    for path in sorted(ROOT.glob('*/sweep.json')):
        plan=json.loads(path.read_text())
        selection=plan.get('selection',{})
        if selection.get('max_requests')==0 and selection.get('bundle')==report['bundle']:
            plans.append((path,plan))
    if not plans:
        return dict(families=families,default_speedup='1',capacity_overview=[],updated_at=datetime.now(timezone.utc).isoformat())
    plan_path,plan=plans[-1]
    work=folder.parent/'throughput_capacity'
    request_cache=work/'dashboard_requests'/plan_path.parent.name
    request_cache.mkdir(parents=True,exist_ok=True)
    finished=[case for case in plan['cases'] if case['status']=='completed' and case['speedup']!=1]
    prepared={}
    # Parsing requests is bounded to a few numeric arrays. Native execution
    # aggregation uses two low-priority streaming workers, never loading the
    # multi-GB raw JSONL files into memory.
    def prepare(case):
        pred=Path(case['prediction'])
        run=json.loads(pred.with_name('run.json').read_text())
        assert run['source_trace_sha256']==report['requests_sha256']
        assert run['speedup']==case['speedup'] and run['capture_scheduler_trace']
        h=report_header(pred)
        meta,series=request_series(case,source,request_cache)
        native=execution_summary(case,work,h)
        return case,h,meta,series,native
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(prepare,case) for case in finished]
        for job in as_completed(jobs):
            case,h,meta,series,native=job.result()
            prepared[case['name']]=(case,h,meta,series,native)
            print('Ready:',case['name'],flush=True)
    for speed in sorted({case['speedup'] for case in finished}):
        group=[entry for entry in prepared.values() if entry[0]['speedup']==speed]
        count=max(entry[2]['count'] for entry in group)
        arrival=source['timestamp']/speed
        indexes=np.floor(arrival/300000).astype(int)
        demand=dict(requests=np.bincount(indexes,minlength=count).astype(int).tolist(),
            input=np.bincount(indexes,weights=source['input_length'],minlength=count).astype(np.int64).tolist(),
            output=np.bincount(indexes,weights=source['output_length'],minlength=count).astype(np.int64).tolist())
        assert sum(demand['requests'])==report['demand'][1]['requests']
        main_days=report['main_end_days']/speed
        whole_days=report['whole_arrival_days']/speed
        full=math.floor(main_days*288)
        peak=int(np.argmax(demand['input'][:full]))
        cases=[]
        for case,h,meta,series,native in sorted(group,key=lambda e:e[0]['dram_gb']):
            item=dict(case=case['name'],dram_gb=case['dram_gb'],read_gbps_per_tp=2,speedup=speed,
                committed_prefill_tokens=h['committed_prefill_tokens'],duration_ms=h['duration_ms'],
                total_input_tokens=int(source['input_length'].sum()),total_output_tokens=meta['total_output_tokens'],
                source_result=str(Path(case['prediction'])/'requests.jsonl'),
                mean_ttft_ms=h['mean_ttft_ms'],p99_ttft_ms=h['p99_ttft_ms'],mean_tpot_ms=h['mean_tpot_ms'],
                sla_percent=h['goodput_completed_requests']/h['completed_requests']*100,
                reuse_ratio=h['first_admission_prefix_cache_reused_ratio'],ssd_read_tb=h['g3_offload']['read']['completed_bytes']/1e12,
                main=meta['windows']['main'],whole=meta['windows']['whole'])
            item['series']={k:np.pad(v,(0,count-len(v))).astype(np.int64).tolist() for k,v in series.items()}
            item['execution']=make_observation(native,count,Path(case['prediction']),h['committed_prefill_tokens'],meta['total_output_tokens'])
            cases.append(item)
        families[format(speed,'g')]=dict(demand=demand,cases=cases,bin_seconds=300,main_days=main_days,whole_days=whole_days,
            source=report['bundle'],requests_sha256=report['requests_sha256'],request_count=len(arrival),
            peak=dict(start_day=peak/288,end_day=(peak+1)/288),execution_ready_count=len(cases),
            main_request_share=report['demand'][0]['request_share'],speedup=speed)
    baseline['speedup']=1
    baseline['main_request_share']=report['demand'][0]['request_share']
    rows=[]
    for case in plan['cases']:
        family=families.get(format(case['speedup'],'g'))
        item=next((c for c in family['cases'] if c['dram_gb']==case['dram_gb'] and c['read_gbps_per_tp']==2),None) if family else None
        rows.append(overview_row(case,item if case['status']=='completed' else None,family['main_days'] if family else 0))
    save(work/'dashboard_overview.json',dict(source_plan=str(plan_path),rows=rows))
    return dict(families=families,default_speedup='1',capacity_overview=rows,source_plan=str(plan_path),
                updated_at=datetime.now(timezone.utc).isoformat())
