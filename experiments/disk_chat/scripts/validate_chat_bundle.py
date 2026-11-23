#!/usr/bin/env python3
"""Stream-validate an AISim chat bundle without expanding synthetic tokens."""
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

from prepare_chat import BLOCK_SIZE, integer, sha256


def validate(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["format"] != "mooncake" or manifest["trace_block_size"] != BLOCK_SIZE:
        raise ValueError("expected a Mooncake bundle with 16-token blocks")
    for name, record in manifest["files"].items():
        path = bundle / name
        if path.parent != bundle or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
            raise ValueError(f"file size or SHA256 mismatch: {name}")
    count, previous, sessions = 0, -1, set()
    with (bundle / "requests.jsonl").open() as requests, (bundle / "source_requests.jsonl").open() as sources:
        for count, pair in enumerate(zip_longest(requests, sources), 1):
            if None in pair:
                raise ValueError("trace and source sidecar have different line counts")
            row, source = map(json.loads, pair)
            if set(row) != {"timestamp", "input_length", "output_length", "hash_ids"}:
                raise ValueError(f"row {count}: unexpected fields")
            timestamp = row["timestamp"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < previous:
                raise ValueError(f"row {count}: timestamps must be finite and monotonic")
            if count == 1 and timestamp != 0:
                raise ValueError("first arrival must be zero")
            previous = timestamp
            length = integer(row["input_length"], "input_length", 1)
            output = integer(row["output_length"], "output_length", 1)
            if length + output > manifest["context_limit"]:
                raise ValueError(f"row {count}: context limit exceeded")
            hashes = row["hash_ids"]
            if len(hashes) != math.ceil(length / BLOCK_SIZE) or any(type(value) is not int or not 0 <= value < 2**32 for value in hashes):
                raise ValueError(f"row {count}: invalid block hashes")
            if source["trace_row"] != count - 1 or source["timestamp"] != timestamp:
                raise ValueError(f"row {count}: sidecar alignment mismatch")
            if source["session_id"] is not None:
                sessions.add(source["session_id"])
    if count != manifest["request_count"] or not count:
        raise ValueError("request count mismatch or empty bundle")
    if manifest["session_count"] is not None and manifest["session_count"] != len(sessions):
        raise ValueError("session count mismatch")
    return {"bundle": str(bundle), "requests": count, "sessions": manifest["session_count"], "span_seconds": previous / 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundles", nargs="+", type=Path)
    args = parser.parse_args()
    for bundle in args.bundles:
        print(json.dumps(validate(bundle.expanduser().resolve())))


if __name__ == "__main__":
    main()
