"""Prepare bounded five-minute aggregates for the original 8x / P256 GB run."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import numpy as np

BASE = Path(__file__).resolve().parents[1]
FOLDER = BASE/'analysis/prefill_queue_diagnosis'


def prepare():
    baseline=json.loads((BASE/'analysis/throughput_8x/results.json').read_text())
    cached=json.loads((BASE/'analysis/throughput_8x/cache/execution/swe_h200_ep8_capacity_p256gb_speed1.json').read_text())
    prediction=Path(cached['source_directory'])
    end=baseline['main_end_days']*86400000
    tool=Path.home()/'build/trace-summary/release/prefill_workload_summary'
    sources=[prediction/'requests.jsonl',prediction/'iteration_metrics.jsonl.gz']
    identity=dict(version=1,sources=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sources],end_ms=end,tool_sha256=hashlib.sha256(tool.read_bytes()).hexdigest())
    identity=json.loads(json.dumps(identity))
    target=FOLDER/'main_workload.json'
    if target.exists():
        data=json.loads(target.read_text())
        if data.get('identity')==identity:
            return data
    native_path=FOLDER/'main_workload_native.json'
    subprocess.run(['nice','-n','10',str(tool),str(prediction),str(native_path),str(end)],check=True)
    native=json.loads(native_path.read_text())
    expected=next(x['prefill_tokens'] for x in cached['overview'] if x['identity']['role']=='prefill')
    assert native['all_recorded_prefill_tokens']==expected
    pattern=re.compile(rb'"(arrival_time_ms|prefill_admit_ms)"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
    arrivals=[];admits=[]
    with (prediction/'requests.jsonl').open('rb') as stream:
        for line in stream:
            row=dict(pattern.findall(line));a=float(row[b'arrival_time_ms']);p=float(row[b'prefill_admit_ms'])
            assert p>=a
            arrivals.append(a);admits.append(p)
    a=np.asarray(arrivals);p=np.asarray(admits)
    assert len(a)==607190
    times=np.concatenate([a,p]);deltas=np.concatenate([np.ones(len(a),dtype=np.int64),-np.ones(len(p),dtype=np.int64)])
    order=np.argsort(times,kind='stable');times=times[order];deltas=deltas[order]
    unique,starts=np.unique(times,return_index=True)
    counts=np.cumsum(np.add.reduceat(deltas,starts))
    assert counts.min()>=0 and counts[-1]==0
    n=len(native['rows']);lo=np.arange(n)*300000.;hi=np.minimum(lo+300000,end)
    def values(at,side):
        pos=np.searchsorted(unique,at,side=side)-1
        return np.where(pos>=0,counts[np.maximum(pos,0)],0)
    peaks=values(lo,'right').copy();ends=values(hi,'left')
    valid=(unique>=0)&(unique<end)
    np.maximum.at(peaks,np.floor(unique[valid]/300000).astype(int),counts[valid])
    series=[]
    for i,row in enumerate(native['rows']):
        assert row[2] is None or row[2]<=16384
        assert row[3] is None or 0<row[3]<=row[2]
        series.append(row+[int(peaks[i]),int(ends[i]),float(hi[i]-lo[i])])
    data=dict(identity=identity,source=str(prediction),bin_ms=300000,end_ms=end,main_days=end/86400000,token_budget=16384,
              columns=native['columns']+['waiting_peak','waiting_end_left','bin_duration_ms'],rows=series,
              total_requests=len(a),compute_passes=sum(r[0] for r in series),main_new_tokens=sum(r[1] for r in series),
              full_budget_passes=sum(r[4] for r in series),main_waiting_peak=int(max(peaks)),
              main_end_waiting=int(values(np.asarray([end]),'left')[0]),all_recorded_prefill_tokens=expected)
    target.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')))
    print('Prepared main workload:',len(series),'bins; peak waiting',max(peaks),flush=True)
    return data


if __name__=='__main__':
    prepare()
