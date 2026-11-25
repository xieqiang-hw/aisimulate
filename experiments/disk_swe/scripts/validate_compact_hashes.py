"""Run matching exact/compact Mooncake samples and compare simulation reports."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import yaml

BASE = Path(__file__).resolve().parents[1]


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()
                if k not in ("uuid", "wall_time_ms", "processed_tokens_per_s",
                             "processed_output_tokens_per_s", "prediction_provenance")}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def differences(a, b, path="", result=None):
    if result is None:
        result = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                result.append({"field": path + "/" + key, "kind": "missing"})
            else:
                differences(a[key], b[key], path + "/" + key, result)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            result.append({"field": path, "kind": "length"})
        for i, (x, y) in enumerate(zip(a, b)):
            differences(x, y, path + "/" + str(i), result)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9):
            result.append({"field": path, "exact": a, "compact": b})
    elif a != b:
        result.append({"field": path, "exact": a, "compact": b})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-bundle", type=Path, required=True)
    parser.add_argument("--compact-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=BASE / "configs/swe_enclosure_h200_ep8_p256gb_8x.yaml")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text())
    config["execution"]["resources"]["memory_limit_gb"] = 4.0
    cfg = root / "validation.yaml"
    cfg.write_text(yaml.safe_dump(config, sort_keys=False))
    reports = {}
    request_records = {}
    paths = {}
    for name, bundle in (("exact", args.exact_bundle), ("compact", args.compact_bundle)):
        print(f"Running {name} sample ...", flush=True)
        log = root / (name + "-runner.log")
        with log.open("w") as stream:
            subprocess.run([sys.executable, str(BASE.parent / "disk/scripts/run_experiment.py"),
                            "--bundle", str(bundle.resolve()), "--config", str(cfg),
                            "--output-root", str(root / name), "--max-requests", "0"],
                           check=True, stdout=stream, stderr=subprocess.STDOUT)
        path = next((root / name).glob("*/prediction/prediction.json"))
        paths[name] = str(path)
        reports[name] = json.loads(path.read_text())
        records = path.parent / "requests.jsonl"
        request_records[name] = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else reports[name].get("per_request", [])
    summary = differences(normalize(reports["exact"]), normalize(reports["compact"]))
    rows = differences(normalize(request_records["exact"]), normalize(request_records["compact"]))
    result = {
        "passed": not summary and not rows,
        "completed_requests": {k: v["completed_requests"] for k, v in reports.items()},
        "request_records": {k: len(v) for k, v in request_records.items()},
        "summary_differences": summary[:20], "request_differences": rows[:20],
        "comparison": "All report fields and per-request records except random UUIDs, actual wall clock and timing provenance; numeric tolerance abs=1e-9, rel=1e-12",
        "reports": paths,
        "sizes": {name: (bundle / "requests.jsonl").stat().st_size for name, bundle in (("exact", args.exact_bundle), ("compact", args.compact_bundle))},
    }
    (root / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
