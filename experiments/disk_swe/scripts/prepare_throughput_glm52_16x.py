"""Aggregate the completed GLM 16x scheduler supplement without replaying."""
import json,math,subprocess
from pathlib import Path
import numpy as np
from throughput_execution import TOOL,make_observation
from throughput_capacity_data import report_header
BASE=Path(__file__).resolve().parents[1]
FOLDER=BASE/'analysis/throughput_glm52'
ROOT=Path('/mnt/d/aisim-results/swe-chat-scheduler-trace-supplement/20261125T122617.930725Z')
case=ROOT/'glm52_p256gb_read2_overlay16_speed1'
run=sorted((case/'runs').glob('*/run.json'))[-1]
meta=json.loads(run.read_text());assert meta['status']=='completed' and meta['speedup']==1
pred=run.parent/'prediction'
bundle=Path(meta['bundle']);manifest=json.loads((bundle/'manifest.json').read_text())
assert manifest['overlay']['factor']==16
raw=FOLDER/'requests_16x.npz'
if not raw.exists():
    import re
    from array import array
    pat=re.compile(rb'"(arrival_time_ms|terminal_time_ms|prefill_admit_ms|input_length|output_length)"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
    keys=['arrival_time_ms','terminal_time_ms','prefill_admit_ms','input_length','output_length']
    cols={k:array('d') for k in keys}
    with (pred/'requests.jsonl').open('rb') as stream:
        for i,line in enumerate(stream,1):
            row=dict(pat.findall(line))
            for k in keys:cols[k].append(float(row[k.encode()]))
            if i%100000==0:print('Request rows',i,flush=True)
    np.savez_compressed(raw,**{k:np.asarray(v) for k,v in cols.items()})
source=dict(np.load(raw))
a,t,p,inp,out=[source[k] for k in ['arrival_time_ms','terminal_time_ms','prefill_admit_ms','input_length','output_length']]
assert len(a)==manifest['request_count']==607190
h=report_header(pred)
count=math.ceil(max(t.max()+1,a.max()+1)/300000)+1
index=np.floor(a/300000).astype(int)
demand={k:np.bincount(index,weights=v,minlength=count).astype(np.int64).tolist() for k,v in [('requests',np.ones(len(a))),('input',inp),('output',out)]}
sa,st,sp=np.sort(a),np.sort(t),np.sort(p)
edges=np.arange(1,count+1)*300000
series=dict(completed_requests=np.bincount(np.floor(t/300000).astype(int),minlength=count),
            completed_output_tokens=np.bincount(np.floor(t/300000).astype(int),weights=out,minlength=count).astype(np.int64),
            unfinished_requests=np.searchsorted(sa,edges,side='right')-np.searchsorted(st,edges,side='right'),
            p_preadmission_waiting=np.searchsorted(sa,edges,side='right')-np.searchsorted(sp,edges,side='right'))
main_end=manifest['overlay']['segment_ms'];whole_end=a.max()+1
peak=np.searchsorted(sa,sa,side='right')-np.searchsorted(st,sa,side='right')
windows={}
for name,end in [('main',main_end),('whole',whole_end)]:
    arrived=int(np.sum(a<end));completed=int(np.sum(t<end))
    windows[name]=dict(arrived_requests=arrived,completed_requests=completed,arrival_request_per_s=arrived/(end/1000),
                      completion_request_per_s=completed/(end/1000),backlog_end=arrived-completed,
                      peak_inflight_requests=int(peak[sa<end].max()),backlog_start=0)
target=FOLDER/'execution_16x.json'
if not target.exists():
    print('Aggregating 16x execution traces',flush=True)
    subprocess.run([str(TOOL),'--bin-seconds','300','--overview-only',str(pred),str(target)],check=True)
native=json.loads(target.read_text())
assert abs(native['end_s']-h['duration_ms']/1000)<.001
item=dict(case=case.name,dram_gb=256,read_gbps_per_tp=2,speedup=1,
          committed_prefill_tokens=h['committed_prefill_tokens'],duration_ms=h['duration_ms'],
          total_input_tokens=int(inp.sum()),total_output_tokens=int(out.sum()),source_result=str(pred/'requests.jsonl'),
          mean_ttft_ms=h['mean_ttft_ms'],p99_ttft_ms=h['p99_ttft_ms'],mean_tpot_ms=h['mean_tpot_ms'],
          sla_percent=h['goodput_completed_requests']/h['completed_requests']*100,
          reuse_ratio=h['first_admission_prefix_cache_reused_ratio'],ssd_read_tb=h['g3_offload']['read']['completed_bytes']/1e12,
          main=windows['main'],whole=windows['whole'],series={k:v.astype(np.int64).tolist() for k,v in series.items()})
item['execution']=make_observation(native,count,pred,h['committed_prefill_tokens'],int(out.sum()))
peak_bin=int(np.argmax(demand['input'][:math.floor(main_end/300000)]))
family=dict(demand=demand,cases=[item],bin_seconds=300,main_days=main_end/86400000,whole_days=whole_end/86400000,
            source=str(bundle),requests_sha256=meta['source_trace_sha256'],request_count=len(a),
            peak=dict(start_day=peak_bin/288,end_day=(peak_bin+1)/288),execution_ready_count=1,
            main_request_share=float(np.sum(a<main_end)/len(a)),speedup=1,factor=16)
assert sum(demand['requests'])==607190
assert sum(series['completed_requests'])==607190
(FOLDER/'overlay_16x.json').write_text(json.dumps(family,ensure_ascii=False,separators=(',',':')))
print('Saved full 16x family',flush=True)
