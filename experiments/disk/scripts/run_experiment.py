"""GLM-5.2-FP8 1P1D experiment using the official native engine, no Dynamo.
Adapted experiment workflow: HWACSLab/aisimulate at
4f1b31c941c4a434f66b9f23359e9259b963ec21,
experiments/pd_unified/run_baseline.py (Apache-2.0).
This modified runner uses the official CLI instead of the old Dynamo SDK.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import yaml

BASE = Path(__file__).resolve().parents[1]
MODEL = 'zai-org/GLM-5.2-FP8'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main(*, default_config=BASE/'configs/hbm_only.yaml',
         default_output_root=Path.home()/'project/results/glm52-1p1d'):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=default_config, help=f'Experiment YAML (default: {default_config})')
    p.add_argument('--bundle', type=Path, required=True, help='Completed converter bundle directory containing manifest.json')
    p.add_argument('--output-root', type=Path, default=default_output_root)
    p.add_argument('--speedup', type=float, default=1.0)
    p.add_argument('--max-requests', type=int, default=0, help='0 means all; otherwise chronological prefix, including tied boundary timestamps')
    p.add_argument('--dry-run', action='store_true', help='Verify bundle, save and validate config without running simulation')
    p.add_argument('--export-html', action='store_true', help='Capture telemetry and dashboard; can be large for 16 weeks')
    args=p.parse_args()
    if not math.isfinite(args.speedup) or args.speedup<=0 or args.max_requests<0:
        p.error('speedup must be finite and positive; max-requests must be nonnegative')
    bundle=args.bundle.expanduser().resolve()
    manifest_path=bundle/'manifest.json'
    if not manifest_path.is_file():
        p.error('manifest.json missing: conversion may still be running; use only a completed bundle')
    manifest=json.loads(manifest_path.read_text())
    if manifest.get('model_mapping',{}).get('target_model') != MODEL:
        p.error(f'bundle target_model must be {MODEL}')
    if manifest.get('format') not in ('mooncake','mooncake-spooled') or manifest.get('trace_block_size')!=16:
        p.error('expected converter Mooncake bundle with 16-token hash blocks')
    trace=bundle/'requests.jsonl'
    expected=manifest['files']['requests.jsonl']
    print('Verifying requests.jsonl SHA256 ...',flush=True)
    actual=digest(trace)
    if trace.stat().st_size!=expected['size'] or actual!=expected['sha256']:
        p.error('requests.jsonl size/SHA256 differs from manifest')
    config_path=args.config.expanduser().resolve()
    config=yaml.safe_load(config_path.read_text())
    if manifest['context_limit']>config['engine']['context_length']:
        p.error('bundle context limit exceeds configured model context')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out=args.output_root.expanduser().resolve()/stamp
    out.mkdir(parents=True,exist_ok=False)
    selected=manifest['request_count']
    if args.max_requests:
        selected=0
        boundary=None
        dest=out/'selected_requests.jsonl'
        with trace.open() as inp, dest.open('w') as output:
            for line in inp:
                row=json.loads(line)
                if boundary is not None and row['timestamp']>boundary:
                    break
                output.write(line)
                selected+=1
                if selected==args.max_requests:
                    boundary=row['timestamp']
        trace=dest
    config['traffic']['source']['paths']=[str(trace)]
    # Official engine accepts the same JSONL via mooncake (including u32 hash IDs).
    config['traffic']['source']['format']='mooncake'
    config['traffic']['load']['speedup']=args.speedup
    cfg=out/'resolved.yaml'
    cfg.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    # Public schema validation; performance/capacity is resolved by the official CLI.
    from aisimulate.config.cli import CorePredictionConfig
    CorePredictionConfig.from_yaml(str(cfg))
    command=[str(Path(sys.executable).parent/'aisim'),'predict','--stack','engine',
             '--config',str(cfg),'--output-dir',str(out/'prediction'),'--capture-per-request']
    if args.export_html:
        command.append('--export-html')
    metadata={'config_source':str(config_path),'resolved_config_sha256':digest(cfg),
              'bundle':str(bundle),'manifest_sha256':digest(manifest_path),
              'source_trace_sha256':actual,'selected_trace_sha256':digest(trace),
              'selected_requests':selected,'max_requests':args.max_requests,
              'speedup':args.speedup,'cache_initial_state':'cold','warmup_seconds':0,
              'python':sys.version,'aisim':importlib.metadata.version('aisim'),
              'command':command,'status':'prepared','dry_run':args.dry_run,
              'engine_revision':subprocess.check_output(['git','-C',str(BASE.parents[1]),'rev-parse','HEAD'],text=True).strip()}
    save(out/'run.json',metadata)
    print(f'Run directory: {out}',flush=True)
    if args.dry_run:
        print('Bundle and YAML validated; no simulation executed.',flush=True)
        return 0
    start=time.monotonic()
    metadata['status']='running'
    save(out/'run.json',metadata)
    try:
        with (out/'run.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
        metadata.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode)
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc))
        raise
    finally:
        metadata['wall_seconds']=time.monotonic()-start
        save(out/'run.json',metadata)
    print(f"{metadata['status']}; see {out/'run.log'}",flush=True)
    return result.returncode

if __name__=='__main__':
    raise SystemExit(main())
