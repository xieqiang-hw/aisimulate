"""Overlay SWE-chat replay arrivals without splitting sessions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

FACTORS = (2, 4, 8, 16)
SUPPORTED_FACTORS = (*FACTORS, 32, 64, 128)
REQUESTS = "requests.jsonl"
SOURCES = "source_requests.jsonl"
TIMESTAMP = re.compile(rb'"timestamp"\s*:\s*(\d+)\s*[,}]')


def index_bundle(bundle, manifest):
    """Index paired rows by byte offset so only one row is loaded during export."""
    records, starts = [], {}
    names = (REQUESTS, SOURCES)
    digests = [hashlib.sha256(), hashlib.sha256()]
    previous = -1
    with (bundle / REQUESTS).open("rb") as requests, (bundle / SOURCES).open("rb") as sources:
        while raw_source := sources.readline():
            offset = requests.tell()
            raw_request = requests.readline()
            row = json.loads(raw_source)
            timestamp = row["timestamp"]
            session = row["session_id"]
            match = TIMESTAMP.search(raw_request)
            if (type(timestamp) is not int or timestamp < 0 or timestamp < previous
                    or row["trace_row"] != len(records) or not session
                    or not match or int(match[1]) != timestamp):
                raise ValueError(f"Invalid source row {len(records)}")
            starts.setdefault(session, timestamp)
            records.append((timestamp, starts[session], offset, sources.tell() - len(raw_source)))
            previous = timestamp
            for digest, raw in zip(digests, (raw_request, raw_source)):
                digest.update(raw)
        if requests.read(1) or len(records) != manifest["request_count"] or not records:
            raise ValueError("Source row count mismatch or empty trace")
    for name, digest in zip(names, digests):
        if digest.hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {name}")
    return records, starts


def overlay(bundle, target, manifest_bytes, records, starts, factor):
    # Integer milliseconds; include the final timestamp in the source interval.
    span = records[-1][0] + 1
    width = (span + factor - 1) // factor
    shifts = {t: (t // width) * width for t in starts.values()}
    ordered = sorted(
        ((timestamp - shifts[start], i) for i, (timestamp, start, _, _) in enumerate(records))
    )
    manifest = json.loads(manifest_bytes)
    files = {}
    overflow_sessions = set()
    crossing_sessions = set()
    overflow_requests = 0
    with TemporaryDirectory(prefix=".swe-overlay-", dir=target.parent) as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        hashes = [hashlib.sha256(), hashlib.sha256()]
        with (
            (bundle / REQUESTS).open("rb") as requests,
            (bundle / SOURCES).open("rb") as sources,
            (stage / REQUESTS).open("wb") as trace,
            (stage / SOURCES).open("wb") as sidecar,
        ):
            for ordinal, (new_time, parent_row) in enumerate(ordered):
                timestamp, start, request_offset, source_offset = records[parent_row]
                requests.seek(request_offset)
                sources.seek(source_offset)
                raw = requests.readline()
                row = json.loads(sources.readline())
                match = TIMESTAMP.search(raw)
                assert match is not None
                updated = raw[:match.start(1)] + str(new_time).encode() + raw[match.end(1):]
                if timestamp // width != start // width:
                    crossing_sessions.add(row["session_id"])
                if new_time >= width:
                    overflow_requests += 1
                    overflow_sessions.add(row["session_id"])
                row.update(trace_row=ordinal, parent_trace_row=parent_row, timestamp=new_time)
                encoded = (json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
                trace.write(updated)
                sidecar.write(encoded)
                hashes[0].update(updated)
                hashes[1].update(encoded)

        for name, digest in zip((REQUESTS, SOURCES), hashes):
            files[name] = {"size": (stage / name).stat().st_size, "sha256": digest.hexdigest()}
        actual_span = ordered[-1][0] - ordered[0][0] + 1
        manifest["files"] = files
        manifest["timestamp_policy"] = (
            "Relative milliseconds from parent replay, shifted by session start segment; "
            "first_response_event_at remains the original arrival proxy"
        )
        manifest["overlay"] = {
            "factor": factor,
            "segment_ms": width,
            "replay_speedup": 1,
            "parent_bundle": bundle.name,
            "parent_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "session_assignment": "first retained request",
            "source_span_ms": span,
            "replay_span_ms": actual_span,
            "effective_density_multiplier": span / actual_span,
            "session_count": len(starts),
            "crossing_sessions": len(crossing_sessions),
            "overflow_sessions": len(overflow_sessions),
            "overflow_requests": overflow_requests,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if target.exists():
            raise FileExistsError(target)
        stage.rename(target)
    print(f"Created {target}: {len(records):,} requests, {len(starts):,} sessions, "
          f"effective {span / actual_span:.3f}x, {overflow_requests:,} tail requests", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path.home() / "project/data/swe_chat/derived/mooncake-v1/claude-code-main-glm52-128k")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--factors", type=int, nargs="+", choices=SUPPORTED_FACTORS, default=FACTORS)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    output = (args.output_root or bundle.parent).resolve()
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("format") != "mooncake" or "overlay" in manifest:
        parser.error("expected an original SWE-chat Mooncake bundle")
    targets = {f: output / f"{bundle.name}-overlay-{f}x" for f in dict.fromkeys(args.factors)}
    for target in targets.values():
        if target.exists():
            parser.error(f"output already exists: {target}")
    output.mkdir(parents=True, exist_ok=True)
    records, starts = index_bundle(bundle, manifest)
    print(f"Indexed {len(records):,} requests across {len(starts):,} sessions", flush=True)
    for factor, target in targets.items():
        overlay(bundle, target, manifest_bytes, records, starts, factor)


if __name__ == "__main__":
    main()
