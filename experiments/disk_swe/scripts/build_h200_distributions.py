"""Build offline request distributions for the exact rows of the H200 EP8 report."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

BASE=Path(__file__).resolve().parents[1]
FOLDER=BASE/'analysis/h200_request_distributions'
TOOL=Path.home()/'build/trace-summary/release/request_distribution_summary'
REPORT=BASE/'analysis/model_reports/source_documents/experiments_h200_ep8_16x.md'
METRICS=[
 dict(key='ttft',name='TTFT',unit='秒',field='ttft_ms / 1000',note='请求到达到首 token，包含所有阶段的等待。'),
 dict(key='adjusted_ttft',name='扣排队后 TTFT',unit='秒',field='prefill_wait.ttft_excluding_prefill_scheduling_ms / 1000',note='只扣除 P 首次准入前的调度等待；保留 KV 恢复、后续 chunk 等待、交接与 D 侧等待。'),
 dict(key='tpot',name='TPOT',unit='ms',field='itl_ms',note='单请求 (末 token 时间 − 首 token 时间) / (输出 token 数 − 1)。不是单次间隔最大值；输出不足两个 token 时无定义。'),
 dict(key='scheduling_wait',name='P 首次准入前调度等待',unit='秒',field='prefill_wait.scheduling_wait_ms / 1000',note='首次准入前总等待扣除路由和已标记 KV 等待的剩余项，不等同于纯算力排队。'),
 dict(key='kv_wait',name='P 首次准入前 KV 恢复等待',unit='秒',field='prefill_wait.kv_wait_ms / 1000',note='记录到的 KV 等待区间并集，不能简单由各传输方向等待相加替代。'),
 dict(key='ssd_wait',name='SSD → DRAM 等待',unit='秒',field='prefill_wait.ssd_to_dram_wait_ms / 1000',note='P 首次准入前记录的 SSD 恢复等待，包含对应任务排队，不是只计算字节搬运耗时。'),
 dict(key='h2d_wait',name='DRAM → HBM 等待',unit='ms',field='prefill_wait.dram_to_hbm_wait_ms',note='P 首次准入前记录的 H2D 恢复等待，包含对应任务排队。'),
 dict(key='latency',name='请求总延迟',unit='秒',field='e2e_latency_ms / 1000',note='请求到达至最后一个输出 token。'),
]


def header(prediction):
    with (prediction/'prediction.json').open() as stream:text=stream.read(30000).split('"per_request":',1)[0]
    result={}
    for k in ['completed_requests','mean_ttft_ms','p99_ttft_ms','mean_tpot_ms']:
        result[k]=json.JSONDecoder().raw_decode(text.split(f'"{k}":',1)[1].lstrip())[0]
    return result


def candidates():
    output={}
    for root in [Path.home()/'project/results',Path('/mnt/d/aisim-results')]:
        paths=set(root.glob('swe-chat-glm52-h200-ep8*/*/sweep.json'))
        paths.update(root.glob('swe-chat-glm52-h200-ep8*/*/*/sweep.json'))
        for path in sorted(paths):
            plan=json.loads(path.read_text())
            for c in plan.get('cases',[]):
                if c.get('status')!='completed':continue
                old=re.fullmatch(r'swe_h200_ep8_p(\d+)gb_read(\d+)_overlay_(\d+)x',c['name'])
                new=re.fullmatch(r'swe_h200_ep8_p(\d+)gb_read(\d+)_overlay(\d+)_speed0p5',c['name'])
                if not (old or new):continue
                gb,bw,factor=map(int,(old or new).groups());speed=.5 if new else 1.0
                preds=[Path(c['prediction'])] if c.get('prediction') else [p.parent/'prediction' for p in (path.parent/c.get('suite','bandwidth')/c['name']).glob('*/run.json') if json.loads(p.read_text()).get('status')=='completed']
                for pred in preds:
                    if (pred/'requests.jsonl').is_file():output.setdefault((factor,speed,gb,bw),[]).append(pred)
    return output


def select():
    index=candidates();out=[]
    for line in REPORT.read_text().splitlines():
        c=[v.strip() for v in line.strip().strip('|').split('|')]
        if len(c)<19 or not re.fullmatch(r'\d+×',c[0]):continue
        factor=int(c[0][:-1]);speed=float(c[1]);gb=int(c[2].split()[0]);bw=int(c[3]);key=(factor,speed,gb,bw)
        case=dict(id=f'overlay{factor}_speed{speed:g}_p{gb}gb_read{bw}',factor=factor,speedup=speed,dram_gb=gb,read_gbps=bw,status=c[4],expected_requests=int(c[5].replace(',','')) if c[5]!='—' else None)
        if case['status']!='已完成':out.append(case);continue
        matching=[]
        for pred in index.get(key,[]):
            h=header(pred)
            if h['completed_requests']!=case['expected_requests']:continue
            checks=[(h['mean_ttft_ms']/1000,float(c[8])),(h['p99_ttft_ms']/1000,float(c[7])),(h['mean_tpot_ms'],float(c[9]))]
            if all(abs(a-b)<=.00501 for a,b in checks):matching.append((pred,h))
        if not matching:raise ValueError(f'No matching request-level result for report row {key}')
        pred,h=max(matching,key=lambda pair:(pair[0]/'requests.jsonl').stat().st_mtime_ns)
        case.update(source=str(pred),report_summary=h);out.append(case)
    assert len(out)==100
    return out


def prepare(case,tool_hash):
    source=Path(case['source'])/'requests.jsonl';target=FOLDER/'cache'/(case['id']+'.json')
    identity=dict(version=1,source=str(source),size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns,tool_sha256=tool_hash)
    if target.exists():
        result=json.loads(target.read_text())
        if result.get('identity')==identity:return dict(case,distribution=result)
    temp=target.with_suffix('.tmp.json')
    subprocess.run(['nice','-n','10',str(TOOL),str(source),str(temp)],check=True)
    result=json.loads(temp.read_text());result['identity']=identity
    assert result['completed']==case['expected_requests']
    m=result['metrics'];h=case['report_summary']
    assert abs(m['ttft']['mean']*1000-h['mean_ttft_ms'])<=max(.001,abs(h['mean_ttft_ms'])*1e-9)
    assert abs(m['tpot']['mean']-h['mean_tpot_ms'])<=max(.001,abs(h['mean_tpot_ms'])*1e-9)
    for metric in m.values():
        if not metric['count']:continue
        assert metric['missing']+metric['count']==result['completed']
        for view in metric['views'].values():
            assert sum(view['counts'])==view['included']
            assert view['included']+view['excluded']==metric['count']
        assert sum(metric['views']['full']['counts'])==metric['count']
    target.write_text(json.dumps(result,separators=(',',':')));temp.unlink()
    return dict(case,distribution=result)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--jobs',type=int,default=2);args=ap.parse_args()
    (FOLDER/'cache').mkdir(parents=True,exist_ok=True)
    selected=select();(FOLDER/'selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2));print(f'Matched {len(selected)} report rows; {sum("source" in c for c in selected)} completed sources',flush=True)
    tool_hash=hashlib.sha256(TOOL.read_bytes()).hexdigest();ready={};errors=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs={pool.submit(prepare,c,tool_hash):c for c in selected if 'source' in c}
        for job in as_completed(jobs):
            case=jobs[job]
            try:ready[case['id']]=job.result();print(f'[{len(ready)}/{len(jobs)}] {case["id"]}',flush=True)
            except Exception as e:errors.append(dict(case=case['id'],error=str(e)));print('ERROR',case['id'],str(e),flush=True)
    if errors:
        (FOLDER/'errors.json').write_text(json.dumps(errors,indent=2));raise RuntimeError(f'{len(errors)} distributions failed; cached completed work is reusable')
    payload=dict(generated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source_report=str(REPORT),metrics=METRICS,cases=[ready.get(c['id'],c) for c in selected],base_bins=400)
    (FOLDER/'results.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    (FOLDER/'validation.json').write_text(json.dumps(dict(report_rows=len(selected),available=len(ready),unavailable=len(selected)-len(ready),checks=['source summary matches exact report row','completed request counts match report','TTFT and TPOT means reconcile','adjusted TTFT identity checked per request','full histograms conserve all valid samples','trimmed histograms report excluded tail counts']),indent=2))
    if not args.prepare_only:
        template=Path(__file__).with_name('request_distribution_dashboard.html').read_text();assert template.count('__DATA__')==1
        text=template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
        output=BASE/'experiments_h200_ep8_distributions.html';temp=output.with_suffix('.tmp.html');temp.write_text(text,encoding='utf-8');temp.replace(output);print('Saved',output,flush=True)


if __name__=='__main__':main()
