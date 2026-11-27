"""Serial scheduler trace supplements: GLM16/32, MiniMax256 at speed .125."""
import argparse, datetime, fcntl, json, subprocess, sys
from pathlib import Path
import yaml
BASE=Path(__file__).resolve().parents[1]
DATA=Path.home()/'project/data/swe_chat/derived/verified_interleave_mooncake_v1'
DROOT=Path('/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1')
RUNNER=BASE.parent/'disk/scripts/run_experiment.py'
def prepare_minimax():
    target=DROOT/'claude-code-main-minimax-m25-shape128k-overlay-256x'
    if target.exists():
        m=json.loads((target/'manifest.json').read_text())
        assert m['overlay']['factor']==256 and m['model_mapping']['target_model']=='MiniMaxAI/MiniMax-M2.5'
        assert (target/'requests.jsonl').stat().st_size==m['files']['requests.jsonl']['size']
        return target
    from prepare_swe_chat_overlay import index_bundle, overlay
    original=DATA/'claude-code-main-glm52-128k'
    raw=(original/'manifest.json').read_bytes()
    records,starts=index_bundle(original,json.loads(raw))
    DROOT.mkdir(parents=True,exist_ok=True)
    stage=DROOT/('.'+target.name+'-building')
    if stage.exists():
        raise RuntimeError(f'Incomplete prior data preparation: {stage}; inspect before retrying')
    overlay(original,stage,raw,records,starts,256)
    m=json.loads((stage/'manifest.json').read_text())
    m['model_mapping']['target_model']='MiniMaxAI/MiniMax-M2.5'
    m['shape_comparison']={
        'source_bundle':str(original),
        'source_target_model':'zai-org/GLM-5.2-FP8',
        'tokenization':'inherited GLM-5.2 tokenization; not retokenized with MiniMax',
        'output_token_ids':'inherited symbolic replay IDs, not MiniMax-generated token IDs',
        'overlay_factor':256}
    (stage/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    stage.rename(target)
    return target
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-root',type=Path,default=Path('/mnt/d/aisim-results/swe-chat-scheduler-trace-supplement'))
    ap.add_argument('--minimax-large-dram', action='store_true',
                    help='Run only the 2048/4096 GB MiniMax 256x cases')
    ap.add_argument('--plan',action='store_true')
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--max-requests',type=int,default=0)
    a=ap.parse_args()
    if a.max_requests<0: ap.error('max-requests must be nonnegative')
    if a.prepare_only:
        print('Prepared bundle:',prepare_minimax(),flush=True)
        return
    cases = []
    if not a.minimax_large_dram:
        template='/home/xieqiang/project/results/swe-chat-glm52-throughput-capacity/20261119T075854.538229Z/cases/swe_h200_ep8_capacity_p256gb_speed1/runs/20261119T075907.105935Z/resolved.yaml'
        for factor in (16,32):
            c=yaml.safe_load(Path(template).read_text())
            cases.append((f'glm52_p256gb_read2_overlay{factor}_speed1',c,DATA/f'claude-code-main-glm52-128k-overlay-{factor}x',1.0))
    capacities = (2048, 4096) if a.minimax_large_dram else (64, 128, 256, 384, 512, 1000)
    for read in (1,2,4,8,16):
        for gb in capacities:
            template_gb = 1000 if a.minimax_large_dram else gb
            source=BASE/'configs'/f'swe_minimax_h100_64gb_ep8_p{template_gb}gb_read{read}_overlay_32x.yaml'
            c=yaml.safe_load(source.read_text())
            c['engine']['systems_paths']=[str(BASE/'configs/systems_h100_64gb'),'default']
            if a.minimax_large_dram:
                # BF16 KV: 31,744 bytes/token per rank, 64 tokens per block, TP8.
                host = c['engine']['workers']['prefill']['kv_cache']['host_offload']
                bytes_per_block = 31_744 * 64 * 8
                assert host['num_host_blocks'] == 1000 * 10**9 // bytes_per_block
                host['num_host_blocks'] = gb * 10**9 // bytes_per_block
            cases.append((f'minimax_p{gb}gb_read{read}_overlay256_speed0p125',c,DROOT/'claude-code-main-minimax-m25-shape128k-overlay-256x',0.125))
    from aisimulate.config.cli import CorePredictionConfig
    for name,c,b,s in cases:
        for role in ('prefill','decode'):
            c['engine']['workers'][role]['scheduler']['max_batched_tokens']=16384
        c.setdefault('execution',{}).setdefault('resources',{}).update(memory_limit_gb=20.0,initialization_timeout_seconds=3600.0)
        c['traffic']['load']['speedup']=s
        c['traffic']['source']['paths']=[str(b/'requests.jsonl')]
        assert c['engine']['workers']['prefill']['kv_cache']['g3_offload']['write_bandwidth_gbps']==8
        print(f'{name}: budget=16384, memory=20 GB, scheduler trace=gzip',flush=True)
    print(f'Total {len(cases)} cases; serial; capture scheduler traces ON',flush=True)
    if a.plan:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            for name,c,b,s in cases:
                q=Path(td)/(name+'.yaml');q.write_text(yaml.safe_dump(c,sort_keys=False))
                CorePredictionConfig.from_yaml(str(q))
        print(f'All {len(cases)} configurations passed schema validation',flush=True)
        return
    a.output_root.mkdir(parents=True,exist_ok=True)
    with (a.output_root/'.supplement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        bundle=DROOT/'claude-code-main-minimax-m25-shape128k-overlay-256x'
        if not (bundle/'manifest.json').is_file():
            raise RuntimeError('Generate the 256x bundle first with --prepare-only')
        out=a.output_root/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        out.mkdir()
        plan=[]
        for name,c,b,s in cases:
            case=out/name;case.mkdir()
            cfg=case/'config.yaml';cfg.write_text(yaml.safe_dump(c,sort_keys=False))
            CorePredictionConfig.from_yaml(str(cfg))
            plan.append(dict(name=name,status='pending',config=str(cfg),bundle=str(b),speedup=s))
        def save():
            t=out/'sweep.json.tmp';t.write_text(json.dumps(plan,indent=2)+'\n');t.replace(out/'sweep.json')
        save()
        for i,(item,(_,c,b,s)) in enumerate(zip(plan,cases),1):
            case=Path(item['config']).parent
            item['status']='running';save()
            print(f'[{i}/{len(cases)}] Running {item["name"]}; log={case/"runner.log"}',flush=True)
            cmd=[sys.executable,'-u',str(RUNNER),'--config',item['config'],'--bundle',str(b),
                 '--speedup',str(s),'--max-requests',str(a.max_requests),'--output-root',str(case/'runs'),
                 '--capture-scheduler-trace','--scheduler-trace-compression','gzip']
            with (case/'runner.log').open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            item.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
            save();print(f'{item["status"]}: {item["name"]}',flush=True)
        print(f'Finished; summary={out/"sweep.json"}',flush=True)
        if any(x['status']=='failed' for x in plan):raise SystemExit(1)
if __name__=='__main__':main()
