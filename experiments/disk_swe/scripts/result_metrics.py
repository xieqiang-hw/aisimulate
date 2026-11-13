"""Read summary metrics for a completed replay."""
from collections import Counter
from functools import cache
import json
import re

METRICS = (
    b"input_length", b"first_admission_g1_reused_input_tokens",
    b"first_admission_g2_direct_reused_input_tokens",
    b"first_admission_g3_reused_input_tokens",
)
TOKEN_FIELD = re.compile(rb'"(' + b'|'.join(METRICS) + rb')":\s*(\d+)')


@cache
def result_cells(path, seconds, expected_count):
    prediction = path.parent / "prediction"
    with (prediction / "prediction.json").open() as stream:
        header = stream.read(20_000).split('"per_request":', 1)[0]

    def metric(name):
        return json.JSONDecoder().raw_decode(header.split(f'"{name}":', 1)[1].lstrip())[0]

    count = metric("completed_requests")
    totals = Counter()
    with (prediction / "requests.jsonl").open("rb") as stream:
        for line in stream:
            for key, value in TOKEN_FIELD.findall(line):
                totals[key] += int(value)
    rates = [100 * totals[key] / totals[METRICS[0]] for key in METRICS[1:]]
    duration = round(seconds)
    return [
        "\u5df2\u5b8c\u6210" if count == expected_count else "\u90e8\u5206\u5b8c\u6210",
        f"{count:,}", f'{100 * metric("goodput_completed_requests") / count:.3f}%',
        f'{metric("p99_ttft_ms") / 1000:.2f}', f'{metric("mean_ttft_ms") / 1000:.2f}',
        f'{metric("mean_tpot_ms"):.2f}',
        *(f"{rate:.3f}%" for rate in (*rates, sum(rates))),
        f'{metric("g3_offload")["read"]["completed_bytes"] / 1e12:.3f} TB',
        f"{duration // 60} \u5206 {duration % 60} \u79d2",
    ]
