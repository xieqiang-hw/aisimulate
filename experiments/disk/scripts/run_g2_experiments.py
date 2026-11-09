"""依次运行仅 P 开启共享 G2、P/D 接入同一个共享 G2 池的对照实验。
复用 run_experiment.py 的数据校验、原生 CLI 调用及结果记录。
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
import subprocess
import sys
import time

import yaml
from aisimulate.config.cli import CorePredictionConfig
from run_experiment import BASE, digest, save

CASES = ("p_g2_shared", "pd_g2_shared")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home()/"project/results/glm52-1p1d-g2")
    parser.add_argument("--num-host-blocks", type=int, default=None,
                        help="共享池总块数，不是每角色块数；块字节数按 TP 分片记账")
    parser.add_argument("--d2h-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--h2d-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--shared-d2h-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--shared-h2d-bandwidth-gbps", type=float, default=None)
    parser.add_argument("--latency-ms", type=float, default=None)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--speedup", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-html", action="store_true")
    args = parser.parse_args()
    if (args.num_host_blocks is not None and args.num_host_blocks <= 0) or args.max_requests < 0:
        parser.error("num-host-blocks 必须为正数；max-requests 不能为负数")
    for name in ("d2h_bandwidth_gbps", "h2d_bandwidth_gbps",
                 "shared_d2h_bandwidth_gbps", "shared_h2d_bandwidth_gbps", "latency_ms"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"{name} 必须是有限非负数；带宽 0 表示不限制")
    if not math.isfinite(args.speedup) or args.speedup <= 0:
        parser.error("speedup 必须是有限正数")
    sources = {case: BASE/f"configs/{case}.yaml" for case in CASES}
    configs = {case: yaml.safe_load(path.read_text()) for case, path in sources.items()}
    overrides = {
        "num_host_blocks": args.num_host_blocks,
        "d2h_bandwidth_gbps": args.d2h_bandwidth_gbps,
        "h2d_bandwidth_gbps": args.h2d_bandwidth_gbps,
        "shared_d2h_bandwidth_gbps": args.shared_d2h_bandwidth_gbps,
        "shared_h2d_bandwidth_gbps": args.shared_h2d_bandwidth_gbps,
        "latency_to_first_byte_ms": args.latency_ms,
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    for case, config in configs.items():
        for role in ("prefill", "decode"):
            cache = config["engine"]["workers"][role]["kv_cache"]
            required = role == "prefill" or case == "pd_g2_shared"
            if ("host_offload" in cache) != required or "g3_offload" in cache:
                parser.error(f"{case}: G2 角色或 G3 配置与实验名称不一致")
            if required:
                cache["host_offload"].update(overrides)
                if cache["host_offload"]["scope"] != "cluster_shared":
                    parser.error(f"{case}: 必须使用 cluster_shared G2")
    pool = configs["p_g2_shared"]["engine"]["workers"]["prefill"]["kv_cache"]["host_offload"]
    for config in configs.values():
        for worker in config["engine"]["workers"].values():
            current = worker["kv_cache"].get("host_offload")
            if current is not None and current != pool:
                parser.error("两组 G2 参数必须一致，以便公平对照")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = args.output_root.expanduser().resolve()/stamp
    out.mkdir(parents=True, exist_ok=False)
    bundle = args.bundle.expanduser().resolve()
    suite = {"status": "prepared", "dry_run": args.dry_run, "bundle": str(bundle),
             "config_sources": {case: {"path": str(path), "sha256": digest(path)}
                                for case, path in sources.items()},
             "g2": pool, "overrides": overrides,
             "max_requests": args.max_requests, "speedup": args.speedup,
             "cases": {}}
    # 先校验两份配置，再启动任一组；模板、输入和现有基线结果不修改。
    for case, config in configs.items():
        config["traffic"]["source"]["paths"] = [str(bundle/"requests.jsonl")]
        config["traffic"]["load"]["speedup"] = args.speedup
        path = out/f"{case}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        CorePredictionConfig.from_yaml(str(path))
        suite["cases"][case] = {"status": "pending", "config": str(path)}
    save(out/"suite.json", suite)
    print(f"实验组目录：{out}", flush=True)
    start = time.monotonic()
    active = None
    try:
        suite["status"] = "running"
        for case in CASES:
            active = case
            item = suite["cases"][case]
            command = [sys.executable, str(BASE/"scripts/run_experiment.py"),
                       "--config", item["config"], "--bundle", str(bundle),
                       "--output-root", str(out/case),
                       "--max-requests", str(args.max_requests),
                       "--speedup", str(args.speedup)]
            if args.dry_run:
                command.append("--dry-run")
            if args.export_html:
                command.append("--export-html")
            item.update(status="running", command=command)
            save(out/"suite.json", suite)
            print(f"开始：{case}", flush=True)
            result = subprocess.run(command, check=False)
            item.update(status=("validated" if args.dry_run else "completed")
                        if result.returncode == 0 else "failed", returncode=result.returncode)
            # 即使一组失败，也保留证据并继续另一组。
            save(out/"suite.json", suite)
        failed = any(v["status"] == "failed" for v in suite["cases"].values())
        suite["status"] = "failed" if failed else ("validated" if args.dry_run else "completed")
        return 1 if failed else 0
    except BaseException as exc:
        suite["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        suite["error"] = repr(exc)
        if active and suite["cases"][active]["status"] == "running":
            suite["cases"][active]["status"] = suite["status"]
        raise
    finally:
        suite["wall_seconds"] = time.monotonic()-start
        save(out/"suite.json", suite)
        print(f"状态：{suite['status']}；记录：{out/'suite.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
