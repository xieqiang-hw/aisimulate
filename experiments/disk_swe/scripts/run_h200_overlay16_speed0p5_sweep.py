"""Replay the original 16x DRAM/SSD matrix with speedup=0.5, serially on D:."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from run_throughput_capacity_sweep import (
    RUNNER, completed_artifacts, digest, engine_identity, live_previous_child,
    now, save, stop_child, summary,
)

BASE = Path(__file__).resolve().parents[1]
CONFIGS = BASE / 'configs/h200_overlay16_speed0p5'
BUNDLE = Path.home() / 'project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-overlay-16x'
OUTPUT = Path('/mnt/d/aisim-results/swe-chat-glm52-h200-ep8-overlay16-speed0p5')
DRAMS = (1000, 512, 384, 256, 128, 64)
READS = (1, 2, 4, 8, 16)
SPEEDUP = 0.5


def name_for(dram, read):
    return f'swe_h200_ep8_p{dram}gb_read{read}_overlay16_speed0p5'


def load_bundle(bundle):
    path = bundle / 'manifest.json'
    data = json.loads(path.read_text())
    overlay = data.get('overlay', {})
    if (data.get('format') != 'mooncake' or data.get('trace_block_size') != 16
            or data.get('context_limit') != 131072
            or data.get('model_mapping', {}).get('target_model') != 'zai-org/GLM-5.2-FP8'
            or overlay.get('factor') != 16 or overlay.get('replay_speedup') != 1):
        raise ValueError('Expected the uncompressed GLM-5.2 128K Mooncake 16x overlay bundle')
    trace = bundle / 'requests.jsonl'
    if trace.stat().st_size != data['files']['requests.jsonl']['size']:
        raise ValueError('Trace size differs from manifest')
    with trace.open() as stream:
        first = json.loads(next(stream))
    if 'session_id' in first:
        raise ValueError('Use the original independent-request bundle for this comparison, not a new session-delay bundle')
    if not isinstance(first.get('output_token_ids'), list) or len(first['output_token_ids']) != first['output_length']:
        raise ValueError('Expected output_token_ids matching output_length')
    return data, digest(path)


def config_text(dram, read, memory):
    path = CONFIGS / (name_for(dram, read) + '.yaml')
    data = yaml.safe_load(path.read_text())
    engine = data['engine']
    kv = engine['workers']['prefill']['kv_cache']
    if (engine['model'] != 'zai-org/GLM-5.2-FP8' or engine['hardware'] != 'h200_sxm'
            or kv['g3_offload']['read_bandwidth_gbps'] != read
            or kv['g3_offload']['write_bandwidth_gbps'] != 8
            or data['traffic']['load']['speedup'] != SPEEDUP):
        raise ValueError(f'Unexpected experiment configuration: {path}')
    resources = data.setdefault('execution', {}).setdefault('resources', {})
    resources.update(memory_limit_gb=memory, initialization_timeout_seconds=3600.0)
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--p-dram-gb', type=int, nargs='+', choices=DRAMS, default=[v for v in DRAMS if v != 1000])
    parser.add_argument('--read-bandwidths', type=int, nargs='+', choices=READS, default=list(READS))
    parser.add_argument('--speedup', type=float, choices=(SPEEDUP,), default=SPEEDUP)
    parser.add_argument('--memory-limit-gb', type=float, default=18.0)
    parser.add_argument('--max-requests', type=int, default=0, help='0: full trace; positive values: validation only')
    parser.add_argument('--capture-scheduler-trace', action='store_true', default=True,
                        help='Always enabled for this experiment (accepted explicitly in the command)')
    parser.add_argument('--scheduler-trace-compression', choices=('gzip',), default='gzip')
    parser.add_argument('--resume', type=Path, help='Saved sweep.json or its directory; repeat any nondefault selection/options')
    parser.add_argument('--plan', action='store_true', help='List cases without writing files or launching AISim')
    parser.add_argument('--dry-run', action='store_true', help='Save and validate configs; do not launch AISim')
    args = parser.parse_args()
    if args.max_requests < 0 or not math.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0:
        parser.error('Invalid request limit or memory budget')
    bundle = args.bundle.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    disk = Path('/mnt/d')
    if not os.path.ismount(disk) or not output_root.is_relative_to(disk.resolve()):
        parser.error('D: must be mounted and --output-root must resolve beneath /mnt/d')
    drams = list(dict.fromkeys(args.p_dram_gb))
    reads = list(dict.fromkeys(args.read_bandwidths))
    manifest, manifest_hash = load_bundle(bundle)
    matrix = [(dram, read) for dram in drams for read in reads]
    settings = {name_for(gb, read): config_text(gb, read, args.memory_limit_gb) for gb, read in matrix}
    main_end_ms = manifest['overlay']['segment_ms'] / SPEEDUP
    for gb, read in matrix:
        print(f'{name_for(gb,read)}: P DRAM {gb} GB; SSD read {read}/write 8 GB/s/TP; '
              f'16x bundle speedup={SPEEDUP}; main window {main_end_ms/86400000:.5f} days', flush=True)
    print(f'Total: {len(matrix)}; serial; memory {args.memory_limit_gb:g} GB; '
          f'scheduler trace v2/gzip ON; results and TMPDIR on D:', flush=True)
    if args.plan:
        return 0

    from aisimulate.config.cli import CorePredictionConfig

    selection = dict(bundle=str(bundle), manifest_sha256=manifest_hash,
                     max_requests=args.max_requests, memory_limit_gb=args.memory_limit_gb,
                     p_dram_gb=drams, read_bandwidths=reads, speedup=SPEEDUP,
                     capture_scheduler_trace=True, scheduler_trace_compression='gzip',
                     engine=engine_identity())
    if args.resume:
        plan_path = args.resume.expanduser().resolve()
        if plan_path.is_dir():
            plan_path = plan_path / 'sweep.json'
        if not plan_path.is_relative_to(disk.resolve()) or not plan_path.is_file():
            parser.error('Resume plan must exist on D:')
        output = plan_path.parent
    else:
        output = output_root / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        output.mkdir(parents=True, exist_ok=False)
        plan_path = output / 'sweep.json'

    with (output / '.sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This sweep is already running')
        if args.resume:
            plan = json.loads(plan_path.read_text())
            if plan.get('selection') != selection:
                parser.error('Resume requires the same bundle, selection, budget, engine and runner')
            for case in plan['cases']:
                if live_previous_child(case):
                    parser.error(f'Previous runner is still active: PID {case["pid"]}')
                actual = hashlib.sha256(settings[case['name']].encode()).hexdigest()
                if actual != case['config_sha256'] or digest(Path(case['config'])) != actual:
                    parser.error(f'Configuration changed: {case["name"]}')
        else:
            configs = output / 'configs'
            configs.mkdir()
            cases = []
            for gb, read in matrix:
                name = name_for(gb, read)
                config = configs / (name + '.yaml')
                config.write_text(settings[name])
                CorePredictionConfig.from_yaml(str(config))
                cases.append(dict(name=name, dram_gb=gb, read_gbps_per_tp=read,
                                  speedup=SPEEDUP, config=str(config), config_sha256=digest(config),
                                  bundle=str(bundle), status='pending', attempts=0,
                                  main_window_end_ms=main_end_ms))
            plan = dict(schema_version=1, experiment='overlay16_speed0p5', created_at=now(),
                        selection=selection, jobs=1, request_count=manifest['request_count'],
                        overlay_factor=16, speedup=SPEEDUP, capture_scheduler_trace=True,
                        scheduler_trace_compression='gzip',
                        sla=dict(ttft_ms=1000, tpot_ms=30), cases=cases)
        plan['updated_at'] = now()
        save(plan_path, plan)
        print(f'Sweep directory: {output}\nResume plan: {plan_path}', flush=True)
        if args.dry_run:
            for case in plan['cases']:
                CorePredictionConfig.from_yaml(case['config'])
            print('All YAML files validated; no simulation launched.', flush=True)
            return 0

        def interrupt(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupt)
        for index, case in enumerate(plan['cases'], 1):
            if case['status'] == 'completed' and completed_artifacts(case):
                print(f'[{index}/{len(matrix)}] Skip completed {case["name"]}', flush=True)
                continue
            case_dir = output / 'cases' / case['name']
            case_dir.mkdir(parents=True, exist_ok=True)
            scratch = case_dir / 'tmp'
            scratch.mkdir(exist_ok=True)
            env = os.environ.copy()
            env['TMPDIR'] = str(scratch)
            case['attempts'] += 1
            log_path = case_dir / f'runner-attempt-{case["attempts"]:03d}.log'
            command = [sys.executable, '-u', str(RUNNER), '--config', case['config'],
                       '--bundle', str(bundle), '--output-root', str(case_dir / 'runs'),
                       '--speedup', '0.5', '--max-requests', str(args.max_requests),
                       '--capture-scheduler-trace', '--scheduler-trace-compression', 'gzip']
            print(f'[{index}/{len(matrix)}] Running {case["name"]}; log={log_path}', flush=True)
            process = None
            started = time.monotonic()
            try:
                with log_path.open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, env=env)
                    case.update(status='running', pid=process.pid, log=str(log_path),
                                started_at=now(), tmpdir=str(scratch), command=command)
                    save(plan_path, plan)
                    code = process.wait()
                case.update(status='completed' if code == 0 else 'failed', returncode=code)
                if code == 0:
                    prediction = completed_artifacts(case)
                    if prediction is None:
                        raise RuntimeError('Successful exit without complete scheduler trace artifacts')
                    case['prediction'] = str(prediction)
                    case['summary'] = summary(prediction)
                case.pop('error', None)
            except KeyboardInterrupt:
                if process is not None:
                    stop_child(process)
                case.update(status='interrupted', error='Stopped by signal or keyboard interrupt')
                print('Interrupted; use --resume to keep completed work.', flush=True)
                return 130
            except Exception as exc:
                if process is not None:
                    stop_child(process)
                case.update(status='failed', error=repr(exc))
                print(f'Failed {case["name"]}: {exc}; continuing', flush=True)
            finally:
                case.update(wall_seconds=time.monotonic()-started, finished_at=now())
                plan['updated_at'] = now()
                save(plan_path, plan)
            print(f'{case["status"]}: {case["name"]}; wall={case["wall_seconds"]:.1f}s', flush=True)
        failed = sum(c['status'] != 'completed' for c in plan['cases'])
        print(f'Finished: {len(matrix)-failed}/{len(matrix)} completed. Plan: {plan_path}', flush=True)
        return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
