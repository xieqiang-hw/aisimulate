"""验证 P 本地 SSD 写入、淘汰后恢复和 P/D 交接。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml

BASE = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_case(out, trace, enable_g3, *, host_blocks=4, repeat_ms=2000):
    out.mkdir()
    config = yaml.safe_load((BASE / "configs/p_g3_local_validation.yaml").read_text())
    config["traffic"]["source"]["paths"] = [str(trace)]
    cache = config["engine"]["workers"]["prefill"]["kv_cache"]
    cache["host_offload"]["num_host_blocks"] = host_blocks
    if not enable_g3:
        del cache["g3_offload"]
    config_path = out / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with (out / "run.log").open("w") as log:
        process = subprocess.run(
            [str(Path(sys.executable).parent / "aisim"), "predict", "--stack", "engine",
             "--config", str(config_path), "--output-dir", str(out / "prediction"),
             "--capture-per-request"],
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    require(process.returncode == 0, f"{out.name} failed. See {out / 'run.log'}")
    prediction = out / "prediction"
    report = json.loads((prediction / "prediction.json").read_text())
    rows = [json.loads(line) for line in (prediction / "requests.jsonl").read_text().splitlines()]
    expected_requests = len(trace.read_text().splitlines())
    require(report["completed_requests"] == len(rows) == expected_requests, "expected all completions")
    for row in rows:
        direct = row["first_admission_g2_direct_reused_input_tokens"]
        restored = row["first_admission_g3_reused_input_tokens"]
        require(direct + restored == row["first_admission_host_reused_input_tokens"],
                "G2/G3 split must match total Host reuse")
        require(row["terminal_status"] == "completed" and row["output_length"] == 2,
                "expected complete two-token responses")
        require(row["source_released_ms"] is not None and row["decode_admit_ms"] is not None
                and row["first_token_ms"] >= row["decode_admit_ms"], "expected P-to-D handoff")
    repeat = next(row for row in rows if row["arrival_time_ms"] == repeat_ms)
    return report.get("g3_offload"), repeat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path.home() / "project/results/glm52-1p1d-g3-validation")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = args.output_root.expanduser().resolve() / stamp
    out.mkdir(parents=True)
    print(f"Validation directory: {out}", flush=True)
    trace = out / "requests.jsonl"
    # B fills five G1 blocks and four complete G2 blocks, displacing A in both.
    # A returns with the same hash IDs. G3 has enough room for both prefixes.
    def write_trace(path, rows):
        path.write_text("".join(json.dumps({
            "timestamp": timestamp, "input_length": length, "output_length": 2,
            "hash_ids": list(range(key, key + (length + 15) // 16)),
        }) + "\n" for timestamp, length, key in rows))

    write_trace(trace, [(0, 129, 0), (1000, 257, 100), (2000, 129, 0)])
    control_g3, control = run_case(out / "g2_only", trace, False)
    g3, enabled = run_case(out / "p_g3_local", trace, True)
    require(control_g3 is None, "control must have no SSD activity")
    require(control["first_admission_reused_input_tokens"] == 0,
            "control A must miss in both G1 and G2")
    require(g3 is not None and g3["lookup_hits"] >= 2, "expected a G3 prefix hit")
    for direction in ("read", "write"):
        io = g3[direction]
        require(io["completed_jobs"] > 0 and io["completed_bytes"] > 0
                and io["submitted_jobs"] == io["completed_jobs"],
                f"expected completed G3 {direction} without outstanding jobs")
    require(g3["cross_worker_read_blocks"] == 0 and g3["pending_blocks"] == 0,
            "expected worker-local IO with no pending blocks")
    require(enabled["first_admission_g1_reused_input_tokens"] == 0
            and enabled["first_admission_host_reused_input_tokens"] == 128,
            "expected both SSD blocks restored through G2 before P admission")
    _, direct = run_case(out / "g2_direct", trace, False, host_blocks=8)
    mixed_trace = out / "mixed_requests.jsonl"
    write_trace(mixed_trace, [(0, 193, 0), (1000, 257, 100), (2000, 193, 0)])
    _, mixed = run_case(out / "g2_g3_mixed", mixed_trace, True, host_blocks=6)
    later_trace = out / "later_requests.jsonl"
    write_trace(later_trace, [(0, 129, 0), (500, 65, 0), (1000, 257, 100),
                              (2000, 129, 0), (3000, 193, 200), (4000, 129, 0)])
    _, later = run_case(out / "later_g2_reuse", later_trace, True, host_blocks=5, repeat_ms=4000)
    for row, expected in [(control, (0, 0)), (enabled, (0, 128)),
                          (direct, (128, 0)), (mixed, (64, 128)), (later, (64, 0))]:
        actual = (row["first_admission_g2_direct_reused_input_tokens"],
                  row["first_admission_g3_reused_input_tokens"])
        require(actual == expected, f"expected {expected} G2/G3 tokens, got {actual}")
    result = {"status": "passed", "control_repeat": control, "direct_repeat": direct,
              "mixed_repeat": mixed, "later_repeat": later,
              "g3_repeat": enabled, "g3_offload": g3}
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: {out / 'validation.json'}")


if __name__ == "__main__":
    main()
