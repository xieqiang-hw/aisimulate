"""Verify P-local G3 and D-local G2 in the same 1P1D replay."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml

from validate_p_g3_local import BASE, require


def replay(root, name, base, rows, *, decode_g2=True, decode_blocks=5):
    out = root / name
    out.mkdir()
    trace = out / "requests.jsonl"
    with trace.open("w") as stream:
        for timestamp, length, output, key in rows:
            print(json.dumps({"timestamp": timestamp, "input_length": length,
                              "output_length": output,
                              "hash_ids": list(range(key, key + (length + 15) // 16))}), file=stream)
    config = deepcopy(base)
    config["traffic"]["source"]["paths"] = [str(trace)]
    decode = config["engine"]["workers"]["decode"]
    decode["scheduler"]["max_sequences"] = 8
    decode["kv_cache"]["capacity"]["blocks"] = decode_blocks
    if decode_g2:
        decode["kv_cache"]["host_offload"] = {
            "scope": "dp_rank_local", "num_host_blocks": 16, "connector": "offloading_v0_24",
        }
    path = out / "resolved.yaml"
    path.write_text(yaml.safe_dump(config))
    prediction = out / "prediction"
    with (out / "run.log").open("w") as log:
        process = subprocess.run(
            [str(Path(sys.executable).with_name("aisim")), "predict", "--stack", "engine",
             "-c", str(path), "--output-dir", str(prediction), "--capture-per-request"],
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    require(process.returncode == 0, f"See {out / 'run.log'}")
    report = json.loads((prediction / "prediction.json").read_text())
    requests = [json.loads(line) for line in (prediction / "requests.jsonl").read_text().splitlines()]
    require(report["completed_requests"] == len(rows) == len(requests), "missing completions")
    for row, (_, _, output, _) in zip(requests, rows):
        require(row["terminal_status"] == "completed" and row["output_length"] == output,
                "incomplete output")
        require(row["source_released_ms"] is not None and row["decode_admit_ms"] is not None
                and row["first_token_ms"] >= row["decode_admit_ms"], "expected P-to-D handoff")
    return report, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-1p1d-g3-d-g2-validation")
    args = parser.parse_args()
    out = args.output_root.expanduser().resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out.mkdir(parents=True)
    print(f"Validation directory: {out}", flush=True)
    config = yaml.safe_load((BASE / "configs/p_g3_local_validation.yaml").read_text())
    report, requests = replay(out, "prefill_ssd", config, [(0,129,2,0),(1000,257,2,100),(2000,129,2,0)])
    restored = requests[-1]
    g3 = report["g3_offload"]
    require(g3["read"]["completed_jobs"] > 0 and g3["write"]["completed_jobs"] > 0
            and g3["cross_worker_read_blocks"] == 0 and g3["pending_blocks"] == 0,
            "expected completed local SSD transfers")
    require(restored["first_admission_g1_reused_input_tokens"] == 0
            and restored["first_admission_host_reused_input_tokens"] == 128,
            "expected restoration of two SSD blocks through P G2")

    # A occupies four D blocks as it grows. B must give up its KV, then resume.
    pressure = [(0, 65, 140, 0), (10, 65, 100, 100)]
    readmissions = {}
    for enabled in (False, True):
        name = "decode_g2" if enabled else "decode_hbm_only"
        _, rows = replay(out, name, config, pressure, decode_g2=enabled, decode_blocks=4)
        history = [a for a in rows[1]["admission_history"] if a["pool"] == "decode" and a["is_readmission"]]
        require(len(history) == 1 and history[0]["reused_input_tokens"] == (64 if enabled else 0),
                f"{name}: unexpected reuse on decode readmission: {history}")
        readmissions[name] = history[0]
    result = {"status": "passed", "prefill_repeat": restored,
              "g3_offload": g3, "decode_readmissions": readmissions}
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    print(f"PASS: {out / 'validation.json'}")


if __name__ == "__main__":
    main()
