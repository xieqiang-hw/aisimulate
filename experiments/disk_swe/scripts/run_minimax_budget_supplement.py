"""Serial MiniMax budget supplements; retain validated summary results on E:."""
from __future__ import annotations
import argparse, copy, datetime, fcntl, hashlib, json, os, re, shutil, signal, subprocess, sys, time
from collections import Counter
from pathlib import Path
import yaml
BASE=Path(__file__).resolve().parents[1]
REPO=BASE.parents[1]
RUNNER=BASE.parent/'disk/scripts/run_experiment.py'
DEFAULT_ROOT=Path('/mnt/e/aisim-results/swe-chat-minimax-h100-64gb-budget-supplement')
MODEL='MiniMaxAI/MiniMax-M2.5'
DRAMS=(64,128,256,384,512,1000)
READS=(1,2,4,8,16)
LOADS=((32,1.0),(128,0.25))
from refresh_h200_ep8_report import finished_cells, COLUMNS
def save(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def matrix():
    for budget,loads in ((131072,LOADS),(16384,LOADS[1:]),(65536,LOADS)):
        for factor,speed in loads:
            for bw in READS:
                for gb in DRAMS:
                    name=f'minimax_budget{budget}_overlay{factor}_speed{str(speed).replace(".","p")}_read{bw}_p{gb}gb'
                    yield dict(name=name,budget=budget,factor=factor,speedup=speed,dram_gb=gb,read_gbps=bw,status='pending')
def bundle(factor):
    if factor==32:
        return Path.home()/'project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-minimax-m25-shape128k-overlay-32x'
    return Path('/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-minimax-m25-shape128k-overlay-128x')
def config_for(c,sample=None):
    p=BASE/'configs'/f'swe_minimax_h100_64gb_ep8_p{c["dram_gb"]}gb_read{c["read_gbps"]}_overlay_32x.yaml'
    cfg=yaml.safe_load(p.read_text())
    assert cfg['engine']['model']==MODEL and cfg['engine']['kvcache_quant_mode']=='bfloat16'
    for role in ('prefill','decode'): cfg['engine']['workers'][role]['scheduler']['max_batched_tokens']=c['budget']
    cfg['traffic']['source']['paths']=[str(sample or bundle(c['factor'])/'requests.jsonl')]
    cfg['traffic']['load']['speedup']=c['speedup']
    cfg.setdefault('execution',{}).setdefault('resources',{})['initialization_timeout_seconds']=3600
    return cfg
def preflight(root,cases):
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.capacity import materialize_aic_num_gpu_blocks
    dest=root/'configs';dest.mkdir(exist_ok=True)
    groups={}
    for c in cases:
        path=dest/(c['name']+'.yaml')
        cfg=config_for(c);path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        parsed=CorePredictionConfig.from_yaml(str(path));c['config']=str(path);c['bundle']=str(bundle(c['factor']))
        if c['budget'] not in groups:
            try:
                spec=prediction_to_replay_spec(parsed);roles={}
                for role in ('prefill','decode'):
                    diag={};raw=materialize_aic_num_gpu_blocks(getattr(spec.backend_deployment,role+'_engine_args'),memory_diagnostics=diag)
                    if raw['num_gpu_blocks']<2048: raise ValueError('KV pool cannot hold a full 128K request')
                    roles[role]=dict(blocks=raw['num_gpu_blocks'],diagnostics=diag)
                groups[c['budget']]=dict(status='passed',roles=roles)
            except Exception as e:groups[c['budget']]=dict(status='failed',error=str(e))
    save(root/'preflight.json',groups)
    for c in cases:
        if groups[c['budget']]['status']=='failed':
            c.update(status='skipped',reason=groups[c['budget']]['error'])
    for f in (32,128):
        b=bundle(f);m=json.loads((b/'manifest.json').read_text())
        assert m['model_mapping']['target_model']==MODEL and m['context_limit']==131072
        assert m['overlay']['factor']==f and m['request_count']==607190
        assert (b/'requests.jsonl').stat().st_size==m['files']['requests.jsonl']['size']
    return groups
def report(root,plan):
    from refresh_model_reports import refresh
    refresh('minimax',budget_root=root)
def compact(case_dir,expected):
    runs=sorted((case_dir/'runs').glob('*/run.json'))
    if len(runs)!=1:raise ValueError('Expected exactly one run for this attempt')
    path=runs[0];meta=json.loads(path.read_text());pred=path.parent/'prediction'
    if meta['status']!='completed':raise ValueError('Replay did not complete')
    cells=finished_cells(case_dir/'runs')
    if len(cells)!=len(COLUMNS) or cells[0]!='已完成' or int(cells[1].replace(',',''))!=expected:
        raise ValueError('Incomplete summary; retaining detailed files')
    if any(v=='—' for v in cells):raise ValueError('Missing report metric; retaining detailed files')
    header=(pred/'prediction.json').open()
    with header: prefix=header.read(20000).split('"per_request":',1)[0]
    resource=pred/'resource-runtime.json'
    obj=dict(schema_version=1,created_at=stamp(),metrics_columns=COLUMNS,metrics_cells=cells,
             expected_requests=expected,completed_requests=expected,
             run_metadata=meta,report_header_prefix=prefix,
             resources=json.loads(resource.read_text()) if resource.exists() else {})
    save(case_dir/'summary.json',obj)
    if json.loads((case_dir/'summary.json').read_text())['metrics_cells']!=cells:raise ValueError('Summary round trip failed')
    return cells,pred
def prune(root,pred):
    target=pred.resolve();allowed=root.resolve()
    if not target.is_relative_to(allowed) or target==allowed:raise ValueError('Unsafe cleanup path')
    for name in ('prediction.json','requests.jsonl','selected_requests.jsonl'):
        p=target/name
        if p.is_file():p.unlink()
    selected=target.parent/'selected_requests.jsonl'
    if selected.is_file():selected.unlink()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--resume',type=Path)
    p.add_argument('--smoke',action='store_true',help='Run 30 real requests for each budget, validating compaction')
    a=p.parse_args()
    if a.plan:
        print(Counter((c['budget'],c['factor'],c['speedup']) for c in matrix()));print(f'Total {len(list(matrix()))}; serial; E: compact output');return
    root=(a.resume or a.output_root/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')).resolve()
    if not root.is_relative_to(Path('/mnt/e')) or not os.path.ismount('/mnt/e'):raise ValueError('Output must be on mounted E:')
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.resume:
        plan=json.loads((root/'sweep.json').read_text())
        for c in plan['cases']:
            if c['status']=='running':raise RuntimeError('Check previous process before resuming a running case')
    else:
        cs=list(matrix());preflight(root,cs)
        if a.smoke:
            cs=[next(c for c in cs if c['budget']==b) for b in (131072,16384,65536)]
            small=root/'smoke-bundle';small.mkdir()
            trace=small/'requests.jsonl'
            with (bundle(32)/'requests.jsonl').open() as inp,trace.open('w') as out:
                for i,line in enumerate(inp):
                    if i==30:break
                    out.write(line)
            manifest=json.loads((bundle(32)/'manifest.json').read_text())
            manifest['request_count']=30
            with trace.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
            manifest['files']={'requests.jsonl':{'size':trace.stat().st_size,'sha256':sha}}
            save(small/'manifest.json',manifest)
            for c in cs:c['bundle']=str(small)
        plan=dict(schema_version=1,created_at=stamp(),jobs=1,compact_results=True,capture_scheduler_trace=False,cases=cs)
    save(root/'sweep.json',plan)
    print('Sweep directory:',root,flush=True)
    if a.validate_only:print('All configurations preflighted; no full replay started',flush=True);return
    if not a.smoke:report(root,plan)
    active=None
    def stop(sig,frame):
        if active is not None and active.poll() is None:os.killpg(active.pid,signal.SIGTERM)
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for i,c in enumerate(plan['cases']):
        if c['status'] in ('completed','skipped'):continue
        if shutil.disk_usage(root).free<10*10**9:raise RuntimeError('E: free space below 10GB; stopping without deleting evidence')
        folder=root/c['name'];folder.mkdir(exist_ok=True)
        if (folder/'runs').exists():raise RuntimeError('Existing unprocessed attempt; retain evidence and use a fresh output directory')
        cmd=[sys.executable,'-u',str(RUNNER),'--config',c['config'],'--bundle',c['bundle'],'--speedup',str(c['speedup']),'--output-root',str(folder/'runs'),'--max-requests','30' if a.smoke else '0']
        c.update(status='running',started_at=stamp());save(root/'sweep.json',plan)
        if not a.smoke:report(root,plan)
        print(f'[{i+1}/{len(plan["cases"])}] {c["name"]}',flush=True)
        env=os.environ.copy();env.pop('AISIM_SCHEDULER_TRACE_DIR',None);env.pop('AISIM_SCHEDULER_TRACE_COMPRESSION',None)
        scratch=folder/'tmp';scratch.mkdir(exist_ok=True);env['TMPDIR']=str(scratch)
        try:
            with (folder/'runner.log').open('w') as log:
                active=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                c['pid']=active.pid;save(root/'sweep.json',plan)
                rc=active.wait()
            if rc:raise RuntimeError(f'Runner exited {rc}; see {folder}/runner.log')
            meta_path=next((folder/'runs').glob('*/run.json'));expected=json.loads(meta_path.read_text())['selected_requests']
            cells,pred=compact(folder,expected);c.update(status='completed',metrics_cells=cells,summary=str(folder/'summary.json'),finished_at=stamp())
            save(root/'sweep.json',plan)
            if not a.smoke:report(root,plan)
            prune(root,pred);c['details_pruned']=True
        except KeyboardInterrupt:
            c.update(status='interrupted',finished_at=stamp());save(root/'sweep.json',plan);raise
        except Exception as e:
            c.update(status='failed',error=str(e),finished_at=stamp());print('FAILED:',e,flush=True)
        save(root/'sweep.json',plan)
        if not a.smoke:report(root,plan)
        print(c['status'],c['name'],flush=True)
    return 1 if any(c['status']=='failed' for c in plan['cases']) else 0
if __name__=='__main__':raise SystemExit(main())
