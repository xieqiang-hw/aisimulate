"""Measure the capacity of one fixed 8x trace using serial time-compression runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

BASE = Path(__file__).resolve().parents[1]
RUNNER = BASE.parent / 'disk/scripts/run_experiment.py'
CONFIGS = BASE / 'configs/throughput_capacity'
BUNDLE = Path.home() / 'project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-overlay-8x'
OUTPUT = Path.home() / 'project/results/swe-chat-glm52-throughput-capacity'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def speed_label(value):
    return format(value, '.12g').replace('.', 'p')


def case_name(dram, speedup):
    return f'swe_h200_ep8_capacity_p{dram}gb_speed{speed_label(speedup)}'


def config_for(dram, speedup, memory):
    source = CONFIGS / (case_name(dram, speedup) + '.yaml')
    if not source.is_file():
        source = CONFIGS / (case_name(dram, 1) + '.yaml')
    config = yaml.safe_load(source.read_text(encoding='utf-8'))
    config['traffic']['load']['speedup'] = speedup
    resources = config.setdefault('execution', {}).setdefault('resources', {})
    resources['memory_limit_gb'] = memory
    resources['initialization_timeout_seconds'] = 3600.0
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def bundle_metadata(bundle):
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('format') != 'mooncake'
            or manifest.get('trace_block_size') != 16
            or manifest.get('context_limit') != 131072
            or manifest.get('model_mapping', {}).get('target_model') != 'zai-org/GLM-5.2-FP8'
            or manifest.get('overlay', {}).get('factor') != 8
            or manifest['overlay'].get('replay_speedup') != 1):
        raise ValueError('Expected the GLM-5.2 128K Mooncake 8x overlay bundle, without prior time compression')
    trace = bundle / 'requests.jsonl'
    if trace.stat().st_size != manifest['files']['requests.jsonl']['size']:
        raise ValueError('Trace size differs from manifest')
    with trace.open() as stream:
        first = json.loads(next(stream))
    if 'session_id' in first:
        raise ValueError('This experiment compares the existing independent-request 8x bundle; do not switch to a session-delay bundle')
    if not isinstance(first.get('output_token_ids'), list) or len(first['output_token_ids']) != first['output_length']:
        raise ValueError('Expected planned output_token_ids matching output_length')
    return manifest, digest(manifest_path)


def engine_identity():
    origin = importlib.util.find_spec('aisimulate._runtime').origin
    return {'aisim_version': importlib.metadata.version('aisim'),
            'runtime_sha256': digest(Path(origin)), 'runner_sha256': digest(RUNNER)}


def completed_artifacts(case):
    runs = sorted((Path(case['log']).parent / 'runs').glob('*/run.json'))
    for metadata_path in reversed(runs):
        try:
            meta = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            continue
        prediction = metadata_path.parent / 'prediction'
        if (meta.get('status') == 'completed' and meta.get('capture_scheduler_trace')
                and meta.get('scheduler_trace_schema') == 2
                and meta.get('speedup') == case['speedup']
                and (prediction / 'requests.jsonl').is_file()):
            names = ['scheduler_trace_metadata.json', 'iteration_metrics.jsonl.gz',
                     'scheduler_states.jsonl.gz', 'scheduling_events.jsonl.gz']
            if all((prediction / name).is_file() and (prediction / name).stat().st_size > 0 for name in names):
                return prediction
    return None


def summary(prediction):
    with (prediction / 'prediction.json').open() as stream:
        header = stream.read(20000).split('"per_request":', 1)[0]
    result = {}
    for key in ('completed_requests', 'duration_ms', 'committed_prefill_tokens',
                'mean_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'output_throughput_tok_s'):
        marker = f'"{key}":'
        if marker in header:
            result[key] = json.JSONDecoder().raw_decode(header.split(marker, 1)[1].lstrip())[0]
    return result


def live_previous_child(case):
    if case.get('status') != 'running' or not case.get('pid'):
        return False
    try:
        cmd = Path(f'/proc/{case["pid"]}/cmdline').read_bytes()
        return b'run_experiment.py' in cmd and str(case['config']).encode() in cmd
    except (OSError, PermissionError):
        return False


def stop_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--p-dram-gb', type=int, nargs='+', choices=(256, 128, 64), default=[256, 128, 64])
    parser.add_argument('--speedups', type=float, nargs='+', default=[1, 2, 4, 8], help='Compression relative to the SAME 8x bundle; may extend beyond 8')
    parser.add_argument('--memory-limit-gb', type=float, default=20.0)
    parser.add_argument('--max-requests', type=int, default=0, help='0 is the complete trace; positive values are only for validation')
    parser.add_argument('--resume', type=Path, help='Previous sweep.json (or its directory); requires the same selection/options')
    parser.add_argument('--plan', action='store_true', help='Print the matrix without writing results or running AISim')
    parser.add_argument('--dry-run', action='store_true', help='Save and validate all YAML files without launching simulations')
    args = parser.parse_args()
    if args.max_requests < 0 or not math.isfinite(args.memory_limit_gb) or args.memory_limit_gb <= 0:
        parser.error('Invalid request limit or memory budget')
    if any(not math.isfinite(s) or s <= 0 for s in args.speedups):
        parser.error('Every speedup must be finite and positive')
    speeds = sorted(set(args.speedups))
    drams = list(dict.fromkeys(args.p_dram_gb))
    bundle = args.bundle.expanduser().resolve()
    manifest, manifest_hash = bundle_metadata(bundle)
    matrix = [(dram, speed) for speed in speeds for dram in drams]
    configurations = {case_name(dram, speed): config_for(dram, speed, args.memory_limit_gb)
                      for dram, speed in matrix}
    if len(configurations) != len(matrix):
        parser.error('Speedups are too close to produce distinct case names')
    for dram, speed in matrix:
        print(f'{case_name(dram,speed)}: P DRAM {dram} GB; SSD read 2/write 8 GB/s/TP; '
              f'8x bundle speedup={speed:g}; main window {manifest["overlay"]["segment_ms"]/86400000/speed:.5f} days', flush=True)
    print(f'Total: {len(matrix)}; serial; memory {args.memory_limit_gb:g} GB per process tree; scheduler trace v2/gzip enabled', flush=True)
    if args.plan:
        return 0

    from aisimulate.config.cli import CorePredictionConfig

    identity = engine_identity()
    selection = dict(bundle=str(bundle), manifest_sha256=manifest_hash, max_requests=args.max_requests,
                     memory_limit_gb=args.memory_limit_gb, speedups=speeds, p_dram_gb=drams,
                     engine=identity)
    if args.resume:
        plan_path = args.resume.expanduser().resolve()
        if plan_path.is_dir():
            plan_path = plan_path / 'sweep.json'
        output = plan_path.parent
        if not plan_path.is_file():
            parser.error(f'Resume plan not found: {plan_path}')
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        output = args.output_root.expanduser().resolve() / stamp
        output.mkdir(parents=True, exist_ok=False)
        plan_path = output / 'sweep.json'

    with (output / '.sweep.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('This sweep is already being run by another process')
        if args.resume:
            plan = json.loads(plan_path.read_text())
            if plan.get('selection') != selection:
                parser.error('Resume requires the same bundle, engine, selection, memory budget and request limit as the saved sweep')
            for case in plan['cases']:
                if live_previous_child(case):
                    parser.error(f'A previous child is still running: PID {case["pid"]}, {case["name"]}')
                text = configurations[case['name']]
                if hashlib.sha256(text.encode()).hexdigest() != case['config_sha256']:
                    parser.error(f'Configuration changed since the saved sweep: {case["name"]}')
                if digest(Path(case['config'])) != case['config_sha256']:
                    parser.error(f'Saved configuration was edited: {case["config"]}')
        else:
            configs = output / 'configs'
            configs.mkdir()
            cases = []
            for dram, speed in matrix:
                name = case_name(dram, speed)
                path = configs / (name + '.yaml')
                text = configurations[name]
                path.write_text(text)
                CorePredictionConfig.from_yaml(str(path))
                cases.append(dict(name=name, dram_gb=dram, speedup=speed, bundle=str(bundle),
                                  config=str(path), config_sha256=digest(path), status='pending',
                                  main_window_end_ms=manifest['overlay']['segment_ms']/speed,
                                  attempts=0))
            plan = dict(schema_version=1, created_at=now(), selection=selection, jobs=1,
                        capture_scheduler_trace=True, scheduler_trace_compression='gzip',
                        base_overlay_factor=8, request_count=manifest['request_count'],
                        sla=dict(ttft_ms=1000, tpot_ms=30), cases=cases)
        plan['updated_at'] = now()
        save(plan_path, plan)
        print(f'Sweep directory: {output}\nResume plan: {plan_path}', flush=True)
        if args.dry_run:
            for case in plan['cases']:
                CorePredictionConfig.from_yaml(case['config'])
            print('All configs validated; no simulation launched.', flush=True)
            return 0

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
        for index, case in enumerate(plan['cases'], 1):
            if case['status'] == 'completed' and completed_artifacts(case):
                print(f'[{index}/{len(matrix)}] Skip completed {case["name"]}', flush=True)
                continue
            case_dir = output / 'cases' / case['name']
            case_dir.mkdir(parents=True, exist_ok=True)
            case['attempts'] += 1
            log_path = case_dir / f'runner-attempt-{case["attempts"]:03d}.log'
            command = [sys.executable, '-u', str(RUNNER), '--config', case['config'],
                       '--bundle', str(bundle), '--output-root', str(case_dir / 'runs'),
                       '--speedup', format(case['speedup'], '.17g'),
                       '--max-requests', str(args.max_requests), '--capture-scheduler-trace',
                       '--scheduler-trace-compression', 'gzip']
            print(f'[{index}/{len(matrix)}] Running {case["name"]}; log={log_path}', flush=True)
            process = None
            started = time.monotonic()
            try:
                with log_path.open('w') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    case.update(status='running', pid=process.pid, log=str(log_path), started_at=now())
                    save(plan_path, plan)
                    code = process.wait()
                case.update(status='completed' if code == 0 else 'failed', returncode=code)
                if code == 0:
                    prediction = completed_artifacts(case)
                    if prediction is None:
                        raise RuntimeError('Child exited successfully but completed trace artifacts are missing')
                    case['prediction'] = str(prediction)
                    case['summary'] = summary(prediction)
                case.pop('error', None)
            except KeyboardInterrupt:
                if process is not None:
                    stop_child(process)
                case.update(status='interrupted', error='Stopped by signal or keyboard interrupt')
                print('Interrupted; completed cases are retained for --resume.', flush=True)
                return 130
            except Exception as error:
                if process is not None:
                    stop_child(process)
                case.update(status='failed', error=repr(error))
                print(f'Failed {case["name"]}: {error}; continuing to the next case', flush=True)
            finally:
                case.update(wall_seconds=time.monotonic()-started, finished_at=now())
                plan['updated_at'] = now()
                save(plan_path, plan)
            print(f'{case["status"]}: {case["name"]}; wall={case["wall_seconds"]:.1f}s', flush=True)
        failed = sum(case['status'] != 'completed' for case in plan['cases'])
        print(f'Finished: {len(matrix)-failed}/{len(matrix)} completed. Plan: {plan_path}', flush=True)
        return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
