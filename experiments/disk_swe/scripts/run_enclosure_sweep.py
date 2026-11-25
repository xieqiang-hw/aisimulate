"""Serial scene-2 P DRAM x shared enclosure read-bandwidth sweep."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml

from run_enclosure_sample import DEFAULT_BUNDLE, validate_compact_bundle
from build_enclosure_report import summary_header

BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path.home() / "project/results/swe-chat-enclosure-h200-ep8-compact"

def normalized_config(config):
    result = deepcopy(config)
    result["traffic"]["source"].pop("paths", None)
    return result

def find_completed(root, config, bundle, manifest_sha, expected):
    target = normalized_config(config)
    for path in sorted(root.glob("*/run.json"), reverse=True):
        try:
            meta = json.loads(path.read_text())
            if (meta.get("status") != "completed" or meta.get("max_requests",0) != 0
                    or meta.get("selected_requests") != expected
                    or meta.get("manifest_sha256") != manifest_sha
                    or Path(meta["bundle"]).resolve() != bundle):
                continue
            actual = yaml.safe_load((path.parent/"resolved.yaml").read_text())
            if normalized_config(actual) != target:
                continue
            report = summary_header(path.parent/"prediction/prediction.json")
            if report["completed_requests"] == expected:
                return path.parent
        except (OSError, ValueError, KeyError):
            continue
    return None

def save(path, plan):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(path)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--dram-gb",type=int,nargs="+",choices=[256,128,64],default=[256,128,64])
    parser.add_argument("--read-gbps",type=int,nargs="+",choices=[80,40,20],default=[80,40,20])
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--stop-on-error",action="store_true")
    args = parser.parse_args(argv)
    bundle = args.bundle.expanduser().resolve()
    manifest = validate_compact_bundle(bundle)
    if manifest.get("overlay",{}).get("factor") != 8:
        parser.error("this sweep requires the compact 8x bundle")
    manifest_sha = hashlib.sha256((bundle/"manifest.json").read_bytes()).hexdigest()
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    # One sweep process at a time, including across different shell sessions.
    lock = (root/".enclosure_sweep.lock").open("a")
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another enclosure sweep is already using this output root")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = root/"sweeps"/stamp
    folder.mkdir(parents=True)
    cases = []
    for dram in dict.fromkeys(args.dram_gb):
        for read in dict.fromkeys(args.read_gbps):
            name = f"p{dram}gb_read{read}_8x"
            path = BASE/f"configs/swe_enclosure_h200_ep8_{name}.yaml"
            cfg = yaml.safe_load(path.read_text())
            prior = find_completed(root,cfg,bundle,manifest_sha,manifest["request_count"])
            cases.append(dict(name=name,dram_gb=dram,read_gbps=read,config=str(path),
                              status="skipped" if prior else "pending",
                              reused_run=str(prior) if prior else None))
    plan = dict(bundle=str(bundle),request_count=manifest["request_count"],serial=True,
                memory_limit_gb=20,write_bandwidth_gbps_total=80,cases=cases)
    plan_path = folder/"sweep.json"
    save(plan_path,plan)
    print(f"Sweep plan: {plan_path}",flush=True)
    print(f"Total {len(cases)}; skip {sum(c['status']=='skipped' for c in cases)}; "
          f"remaining {sum(c['status']=='pending' for c in cases)}; serial; memory 20 GB",flush=True)
    failed = False
    for index,case in enumerate(cases,1):
        if case["status"] == "skipped":
            print(f"[{index}/{len(cases)}] SKIP {case['name']}: {case['reused_run']}",flush=True)
            continue
        if args.dry_run:
            print(f"[{index}/{len(cases)}] WOULD RUN {case['name']}",flush=True)
            continue
        log = folder/(case["name"]+".log")
        case.update(status="running",log=str(log))
        save(plan_path,plan)
        print(f"[{index}/{len(cases)}] RUN {case['name']}; log={log}",flush=True)
        try:
            with log.open("w") as stream:
                result = subprocess.run([
                    sys.executable,str(BASE/"scripts/run_enclosure_sample.py"),
                    "--bundle",str(bundle),"--config",case["config"],
                    "--output-root",str(root),"--max-requests","0",
                ],stdout=stream,stderr=subprocess.STDOUT)
            cfg = yaml.safe_load(Path(case["config"]).read_text())
            completed = find_completed(root,cfg,bundle,manifest_sha,manifest["request_count"])
            case.update(status="completed" if result.returncode==0 and completed else "failed",
                        returncode=result.returncode,run=str(completed) if completed else None)
            failed |= case["status"] == "failed"
        except KeyboardInterrupt:
            case["status"] = "interrupted"
            save(plan_path,plan)
            raise
        save(plan_path,plan)
        print(f"{case['status']}: {case['name']}",flush=True)
        subprocess.run([sys.executable,str(BASE/"scripts/build_enclosure_report.py"),
                        "--results-root",str(root)],check=False)
        if case["status"]=="failed" and args.stop_on_error:
            break
    print(f"Sweep {'plan ready' if args.dry_run else 'finished'}: {plan_path}",flush=True)
    return 1 if failed else 0

if __name__=="__main__":
    raise SystemExit(main())
