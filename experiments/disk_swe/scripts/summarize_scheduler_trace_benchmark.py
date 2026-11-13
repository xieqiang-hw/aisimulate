"""Summarize identical-data trace benchmarks and verify v1/v2 parity."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
from statistics import median

from scheduler_trace import iter_iterations


def request_rows(path):
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            row.pop("uuid", None)
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/aisim-observation-benchmark"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    modes = ("off", "v1", "v2-gzip", "v2-none", "off-v2")
    paths = {}
    summaries = {}
    hashes = set()
    for mode in modes:
        runs = []
        for path in sorted((args.root / mode).glob("*/run.json")):
            meta = json.loads(path.read_text())
            if meta["status"] != "completed":
                continue
            hashes.add(meta["selected_trace_sha256"])
            directory = path.parent / "prediction"
            size = sum(p.stat().st_size for p in directory.iterdir()
                       if p.name.startswith(("iteration_metrics", "scheduling_events", "scheduler_states", "scheduler_trace_metadata")))
            runs.append({"path": str(path.parent), "wall_seconds": meta["wall_seconds"], "trace_bytes": size,
                         "selected_requests": meta["selected_requests"]})
        assert len(runs) >= 2, (mode, len(runs))
        paths[mode] = [Path(row["path"]) for row in runs]
        summaries[mode] = {"runs": runs, "median_wall_seconds": median(row["wall_seconds"] for row in runs),
                           "median_trace_bytes": median(row["trace_bytes"] for row in runs)}
    assert len(hashes) == 1, "Benchmarks used different selected input data"
    reference = paths["off"][0] / "prediction/requests.jsonl"
    for mode in modes:
        for path in paths[mode]:
            for index, (a,b) in enumerate(zip_longest(request_rows(reference), request_rows(path / "prediction/requests.jsonl"))):
                assert a == b, (mode, path, "request mismatch", index)
    old = paths["v1"][0] / "prediction"
    new = paths["v2-gzip"][0] / "prediction"
    iterations = 0
    for index, (a,b) in enumerate(zip_longest(iter_iterations(old), iter_iterations(new))):
        assert a is not None and b is not None, "Iteration count changed"
        for key in a:
            if key != "schema_version":
                assert key in b and a[key] == b[key], (index, key, a[key], b.get(key))
        iterations += 1
    result = {
        "passed": True, "selected_trace_sha256": next(iter(hashes)),
        "request_parity_all_runs": True, "v1_v2_iteration_and_state_exact_match": True,
        "verified_iteration_rows": iterations, "variants": summaries,
        "v2_gzip_speedup_over_v1": summaries["v1"]["median_wall_seconds"] / summaries["v2-gzip"]["median_wall_seconds"],
        "v2_gzip_overhead_vs_off": summaries["v2-gzip"]["median_wall_seconds"] / summaries["off-v2"]["median_wall_seconds"] - 1,
        "v2_gzip_trace_reduction": 1 - summaries["v2-gzip"]["median_trace_bytes"] / summaries["v1"]["median_trace_bytes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
