"""Compare 384 GB and 128 GB at 8x using paired requests."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median


def load(path):
    rows = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["session_id"], row["turn_index"])
            if key in rows:
                raise ValueError(f"Duplicate request key {key} in {path}")
            arrival, p, reserve, d, first = (
                row[field] for field in ("arrival_time_ms", "prefill_admit_ms",
                                         "destination_reserved_ms", "decode_admit_ms",
                                         "first_token_ms")
            )
            stages = tuple(y - x if x is not None and y is not None else None
                           for x, y in zip((arrival, p, reserve, d), (p, reserve, d, first)))
            rows[key] = (arrival, row["input_length"], row["requested_output_length"],
                         row["terminal_status"], row["ttft_ms"], row["itl_ms"], stages,
                         row["readmission_count"], row["first_admission_g2_direct_reused_input_tokens"],
                         row["first_admission_g3_reused_input_tokens"])
    return rows


def status(row):
    if row[3] != "completed" or row[4] is None:
        return "incomplete"
    failures = []
    if row[4] > 5000:
        failures.append("ttft")
    if row[5] is not None and row[5] > 50:
        failures.append("itl")
    return "+".join(failures) or "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep", type=Path)
    args = parser.parse_args()
    paths = []
    expected = []
    for gb in (384, 128):
        path, = args.sweep.glob(f"p_local_8ssd_p{gb}gb_8x/*/prediction/requests.jsonl")
        paths.append(path)
        with path.with_name("prediction.json").open() as stream:
            header = stream.read(16000)
        expected.append(int(header.split('"goodput_completed_requests":', 1)[1].split(",", 1)[0]))
    old, new = map(load, paths)
    if old.keys() != new.keys():
        raise ValueError("The two runs contain different request keys")

    transitions = Counter()
    cohorts = defaultdict(lambda: defaultdict(list))
    observed = [0, 0]
    for key, a in old.items():
        b = new[key]
        if a[:3] != b[:3]:
            raise ValueError(f"Request identity or workload differs for {key}")
        x, y = status(a), status(b)
        transitions[f"{x} -> {y}"] += 1
        observed[0] += x == "pass"
        observed[1] += y == "pass"
        if (x == "pass") == (y == "pass"):
            continue
        cohort = cohorts["gained" if y == "pass" else "lost"]
        names = ("ttft_ms", "itl_ms", "arrival_to_p_admit_ms", "p_admit_to_d_reserve_ms",
                 "d_reserve_to_admit_ms", "d_admit_to_first_token_ms", "readmissions",
                 "g2_direct_tokens", "g3_reused_tokens")
        for name, left, right in zip(names, (*a[4:6], *a[6], *a[7:]),
                                     (*b[4:6], *b[6], *b[7:])):
            if left is not None and right is not None:
                cohort[name].append(right - left)

    if observed != expected:
        raise ValueError(f"Per-request SLA totals {observed} disagree with reports {expected}")
    print(json.dumps({
        "paired_requests": len(old), "sla_pass_384gb_128gb": observed,
        "transitions": dict(transitions),
        "delta_128gb_minus_384gb": {
            label: {name: {"median": median(values), "mean": sum(values) / len(values),
                           "samples": len(values), "lower": sum(v < 0 for v in values)}
                    for name, values in metrics.items()}
            for label, metrics in cohorts.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
