"""Offline comparison of P128/P256 scheduling and arrival-cohort latency."""
from pathlib import Path
import gzip,json,re,subprocess
import numpy as np

BASE=Path(__file__).resolve().parents[1]
FOLDER=BASE/'analysis/prefill_queue_diagnosis'
KEYS=['arrival_time_ms','prefill_admit_ms','first_token_ms','input_length','scheduling_wait_ms','kv_wait_ms','ttft_ms','ttft_excluding_prefill_scheduling_ms']
PATTERN=re.compile(rb'"('+'|'.join(KEYS).encode()+rb')"\s*:\s*(-?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')

def ranks(x):
    order=np.argsort(x,kind='stable');v=x[order];starts=np.r_[0,np.flatnonzero(v[1:]!=v[:-1])+1];ends=np.r_[starts[1:],len(x)]
    out=np.empty(len(x));out[order]=np.repeat((starts+ends-1)/2,ends-starts);return out

def corr(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])

def prepare_case(gb):
    cached=json.loads((BASE/f'analysis/throughput_8x/cache/execution/swe_h200_ep8_capacity_p{gb}gb_speed1.json').read_text())
    source=Path(cached['source_directory']);seed=json.loads((FOLDER/'window_data.json').read_text())
    lo=(seed['window_start_ms']+seed['original_window_end_ms'])/2-3600000;hi=lo+7200000
    end=json.loads((BASE/'analysis/throughput_8x/results.json').read_text())['main_end_days']*86400000
    folder=FOLDER/f'p{gb}gb';folder.mkdir(exist_ok=True)
    signature=[[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in [source/'requests.jsonl',source/'iteration_metrics.jsonl.gz',source/'scheduler_states.jsonl.gz']]
    target=folder/'comparison_data.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old.get('signature')==signature and old.get('schema_version')==2:return old
    if gb==256:
        main=json.loads((FOLDER/'main_workload.json').read_text());extended=json.loads((FOLDER/'extended_window_data.json').read_text())
    else:
        nativepath=folder/'main_workload_native.json'
        if not nativepath.exists():
            subprocess.run([str(Path.home()/'build/trace-summary/release/prefill_workload_summary'),str(source),str(nativepath),str(end)],check=True)
        main=json.loads(nativepath.read_text())
        iterations=[];states=[];last=None
        print('Extract window',gb,flush=True)
        with gzip.open(source/'iteration_metrics.jsonl.gz','rb') as f:
            for line in f:
                if not line.startswith(b'[0,'):continue
                r=json.loads(line)
                if r[2]>hi:break
                if r[4]>=lo:iterations.append(r)
        with gzip.open(source/'scheduler_states.jsonl.gz','rb') as f:
            for line in f:
                if not line.startswith(b'[0,'):continue
                sid,it,t,s=json.loads(line);r=[t,it,{k:(v if v is not None else 0) for k,v in s.items()}]
                if t>hi:break
                if t<lo:last=r
                else:states.append(r)
        if last:states.insert(0,last)
        extended=dict(iterations=iterations,states=states)
    print('Read request timings',gb,flush=True)
    values=[];window=[]
    with (source/'requests.jsonl').open('rb') as f:
        for line in f:
            q={k.decode():float(v) for k,v in PATTERN.findall(line)}
            a=q['arrival_time_ms'];p=q['prefill_admit_ms'];ttft=q['ttft_ms'];wait=q['scheduling_wait_ms'];adjusted=q['ttft_excluding_prefill_scheduling_ms']
            assert p>=a and wait>=0 and abs(ttft-adjusted-wait)<1e-4
            assert abs(ttft-(q['first_token_ms']-a))<1e-4
            values.append([a,p,ttft,adjusted,wait,q['input_length'],q['kv_wait_ms']])
            if a<hi and (p>=lo or a>=lo):
                q['uuid']=re.search(rb'"uuid"\s*:\s*"([^"]+)"',line).group(1).decode();window.append(q)
    arr=np.asarray(values);assert len(arr)==607190
    arrivals=arr[:,0];admits=arr[:,1]
    # Q(a-): exclude both arrivals and admissions at that timestamp.
    sa=np.sort(arrivals);sp=np.sort(admits)
    before=np.searchsorted(sa,arrivals,'left')-np.searchsorted(sp,arrivals,'left')
    assert before.min()>=0
    for q in window:q['waiting_before_arrival']=int(np.searchsorted(sa,q['arrival_time_ms'],'left')-np.searchsorted(sp,q['arrival_time_ms'],'left'))
    valid=arrivals<end;ix=np.floor(arrivals[valid]/300000).astype(int);n=int(np.ceil(end/300000))
    sums=[np.bincount(ix,minlength=n),*[np.bincount(ix,weights=arr[valid,j],minlength=n) for j in (2,3,4)]]
    latency=[list(map(float,row)) for row in zip(*sums)]
    times=np.concatenate([arrivals,admits]);deltas=np.r_[np.ones(len(arr),dtype=np.int64),-np.ones(len(arr),dtype=np.int64)];order=np.argsort(times,kind='stable');times=times[order];deltas=deltas[order]
    unique,starts=np.unique(times,return_index=True);counts=np.cumsum(np.add.reduceat(deltas,starts));assert min(counts)>=0 and counts[-1]==0
    def at(t,side):
        pos=np.searchsorted(unique,t,side)-1;return np.where(pos>=0,counts[np.maximum(pos,0)],0)
    bs=np.arange(n)*300000.;ends=np.minimum(bs+300000,end);peaks=at(bs,'right').copy();ends_q=at(ends,'left');v=(unique>=0)&(unique<end)
    np.maximum.at(peaks,np.floor(unique[v]/300000).astype(int),counts[v])
    if gb==128:
        main['rows']=[row+[int(peaks[i]),int(ends_q[i]),float(ends[i]-bs[i])] for i,row in enumerate(main['rows'])]
        main.update(main_days=end/86400000,main_waiting_peak=int(max(peaks)),compute_passes=sum(r[0] for r in main['rows']),main_new_tokens=sum(r[1] for r in main['rows']),full_budget_passes=sum(r[4] for r in main['rows']))
    else:assert [r[7] for r in main['rows']]==list(peaks)
    main['latency_rows']=latency
    correlations=[]
    for j,name in [(2,'原始 TTFT'),(3,'扣排队后 TTFT'),(4,'被扣除的调度等待')]:
        x=before[valid];y=arr[valid,j]
        correlations.append(dict(metric=name,pearson=corr(x,y),spearman=corr(ranks(x),ranks(y))))
    data=dict(schema_version=2,signature=signature,source=str(source),configuration={**seed['configuration'],'dram_gb':gb},main_workload=main,
        window_start_ms=seed['window_start_ms'],original_window_end_ms=seed['original_window_end_ms'],window_end_ms=seed['window_end_ms'],
        extended_start_ms=lo,extended_end_ms=hi,extended_center_ms=(lo+hi)/2,iterations=extended['iterations'],states=extended['states'],requests=window,
        correlation=dict(scope='主要负载阶段，按请求到达前的等待数；同刻到达请求不计入',requests=int(valid.sum()),rows=correlations),
        validation=dict(requests=len(arr),window_requests=len(window),main_latency_requests=int(sums[0].sum()),max_wait_decomposition_error_ms=float(np.max(np.abs(arr[:,2]-arr[:,3]-arr[:,4])))))
    temp=target.with_suffix('.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')));temp.replace(target)
    print('Prepared',gb,data['validation'],flush=True);return data

def build():
    cases={str(gb):prepare_case(gb) for gb in (256,128)}
    payload=dict(cases=cases,default='256')
    template=Path(__file__).with_name('prefill_queue_comparison_dashboard.html').read_text();assert template.count('__DATA__')==1
    page=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    (BASE/'prefill_queue_diagnosis.html').write_text(page)
    p=BASE/'prefill_queue_diagnosis.md';md=p.read_text();marker='\n## 7. 128 GB 对照与延迟联动\n'
    md=md.split(marker)[0]+marker+'\nHTML 新增 P DRAM 128／256 GB 配置选择，其余保持原始 8×、speedup=1、SSD 读 2 GB/s/TP、16,384-token 预算。两组共用同一主要时段和局部窗口。原两种秒级案例的解释只对应 256 GB 配置。\n\n主要阶段按请求到达时间分桶，以请求数加权显示原始 TTFT、扣排队后 TTFT 和两者差额的均值；局部图每个点为一个请求，横轴是到达时刻，点的延迟包含其后直至首 token 的等待与执行。三幅图共用时间轴和游标。差额等于报告的首次准入前调度等待，是定义关系，不是独立的因果验证；扣排队后 TTFT 仍含 KV 恢复、后续 chunk 和 D 侧等时间。\n\n相关性以主要阶段每个请求到达前的 P 等待数与其延迟配对，输出 Pearson 与 Spearman 系数；不以相关性证明算力饱和或排队是唯一成因。\n'
    for gb,d in cases.items():
        md+=f'\n### P DRAM {gb} GB\n\n| 延迟项 | Pearson | Spearman |\n|---|---:|---:|\n'
        for r in d['correlation']['rows']:md+=f'| {r["metric"]} | {r["pearson"]:.4f} | {r["spearman"]:.4f} |\n'
    p.write_text(md)
    print('Built dual-configuration diagnosis page',flush=True)

if __name__=='__main__':build()
