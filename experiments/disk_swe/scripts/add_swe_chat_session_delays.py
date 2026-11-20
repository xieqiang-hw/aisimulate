"""Add causal session IDs and event-derived delays to an existing SWE-chat 1x bundle.

Token arrays and hashes are copied unchanged. The first request of each retained
session segment keeps its original timestamp. Later turns wait for the simulated
previous completion plus the nonnegative interval from its last assistant event
to the latest input event for the next call. These are log-event approximations,
not measured API start/end times. Missing timing uses delay=0 and is counted.
Filtered-out calls and history resets start a new replay session segment.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory

from prepare_swe_chat import extract_calls

DATA = Path.home() / "project/data"
TIMESTAMP = re.compile(rb'"timestamp"\s*:\s*(\d+)')


def milliseconds(value):
    try:
        date = datetime.fromisoformat(value)
        return round(date.timestamp() * 1000) if date.utcoffset() is not None else None
    except (TypeError, ValueError):
        return None


def session_delays(calls, retained):
    by_index = {call["call_index"]: call for call in calls}
    previous = None
    segment = 0
    for index, response_id, history_reset in retained:
        call = by_index.get(index)
        if call is None or call["response_id"] != response_id:
            raise ValueError(f"Source call {index}/{response_id} differs from current transcript extraction")
        first = previous is None or index != previous["call_index"] + 1 or history_reset
        if first:
            segment += 1
        inputs = [(milliseconds(event.get("timestamp")), event.get("timestamp"))
                  for event in call["input_delta"]]
        inputs = [(ms, stamp) for ms, stamp in inputs if ms is not None]
        ready_ms, ready_stamp = max(inputs, default=(None, None))
        end_stamp = previous.get("last_response_event_at") if previous else None
        end_ms = milliseconds(end_stamp)
        first_ms = milliseconds(call["first_response_event_at"])
        delay = 0
        reason = "segment_start"
        if not first:
            if ready_ms is None or end_ms is None:
                reason = "missing_timing_zero_delay"
            elif first_ms is None or ready_ms > first_ms:
                reason = "inconsistent_timing_zero_delay"
            elif ready_ms < end_ms:
                reason = "overlapping_events_zero_delay"
            else:
                delay = ready_ms - end_ms
                reason = "last_response_to_input_ready"
        yield index, segment, delay, reason, ready_stamp, end_stamp
        previous = call


def convert(bundle, upstream, output):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("format") != "mooncake" or "overlay" in manifest or "session_replay" in manifest:
        raise ValueError("Expected the original 1x SWE-chat Mooncake bundle")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with TemporaryDirectory(prefix=".swe-sessions-", dir=output.parent) as temporary:
        stage = Path(temporary)
        with sqlite3.connect(stage / "timings.sqlite") as db:
            db.execute("""CREATE TABLE calls (
                ordinal INTEGER PRIMARY KEY, session TEXT, call_index INTEGER,
                response TEXT, transcript TEXT, history_reset INTEGER,
                replay_session TEXT, delay INTEGER, reason TEXT, ready TEXT, previous_end TEXT)""")
            with (bundle / "source_requests.jsonl").open("rb") as sources:
                for ordinal, raw in enumerate(sources):
                    row = json.loads(raw)
                    if row["trace_row"] != ordinal:
                        raise ValueError(f"Source row index mismatch at {ordinal}")
                    db.execute("INSERT INTO calls VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)",
                               (ordinal, row["session_id"], row["call_index"], row["response_id"],
                                row["transcript_path"], bool(row.get("history_reset"))))
            db.execute("CREATE INDEX session_calls ON calls(session, call_index)")
            db.commit()
            for session, relative in db.execute("SELECT DISTINCT session, transcript FROM calls"):
                transcript = (upstream / relative).resolve()
                if not transcript.is_relative_to(upstream) or not transcript.is_file():
                    raise ValueError(f"Invalid transcript path: {relative}")
                extracted = extract_calls(transcript, session, Counter())
                retained = list(db.execute(
                    "SELECT call_index, response, history_reset FROM calls WHERE session=? ORDER BY ordinal",
                    (session,)))
                for index, segment, delay, reason, ready, end in session_delays(extracted, retained):
                    replay_session = session if segment == 1 else f"{session}#segment{segment}"
                    db.execute("""UPDATE calls SET replay_session=?, delay=?, reason=?, ready=?, previous_end=?
                                  WHERE session=? AND call_index=?""",
                               (replay_session, delay, reason, ready, end, session, index))
                    counts[reason] += 1
                counts["original_sessions"] += 1
                counts["replay_sessions"] += segment
                if counts["original_sessions"] % 100 == 0:
                    db.commit()
                    print(f"Prepared timing for {counts['original_sessions']:,} sessions", flush=True)
            db.commit()
            target = stage / "bundle"
            target.mkdir()
            names = ("requests.jsonl", "source_requests.jsonl")
            incoming = [hashlib.sha256(), hashlib.sha256()]
            outgoing = [hashlib.sha256(), hashlib.sha256()]
            with ((bundle / names[0]).open("rb") as requests,
                  (bundle / names[1]).open("rb") as sources,
                  (target / names[0]).open("wb") as trace,
                  (target / names[1]).open("wb") as sidecar):
                for ordinal, session, delay, reason, ready, end in db.execute(
                        "SELECT ordinal, replay_session, delay, reason, ready, previous_end FROM calls ORDER BY ordinal"):
                    raw, raw_source = requests.readline(), sources.readline()
                    incoming[0].update(raw)
                    incoming[1].update(raw_source)
                    source = json.loads(raw_source)
                    match = TIMESTAMP.search(raw)
                    if not match or int(match[1]) != source["timestamp"] or session is None:
                        raise ValueError(f"Request/source timing mismatch at row {ordinal}")
                    body = raw.rstrip()
                    if not body.endswith(b"}"):
                        raise ValueError(f"Expected one JSON object on request row {ordinal}")
                    extra = json.dumps({"session_id": session, "delay": delay},
                                       ensure_ascii=False, separators=(",", ":")).encode()
                    updated = body[:-1] + b"," + extra[1:] + b"\n"
                    source.update(replay_session_id=session, delay=delay, delay_source=reason,
                                  input_ready_event_at=ready, previous_response_last_event_at=end)
                    source_bytes = (json.dumps(source, ensure_ascii=False) + "\n").encode()
                    for stream, digest, value in ((trace, outgoing[0], updated), (sidecar, outgoing[1], source_bytes)):
                        stream.write(value)
                        digest.update(value)
                    counts["requests"] += 1
                if requests.read(1) or sources.read(1) or counts["requests"] != manifest["request_count"]:
                    raise ValueError("Request count differs from source manifest")
            for name, digest in zip(names, incoming):
                if digest.hexdigest() != manifest["files"][name]["sha256"]:
                    raise ValueError(f"Source checksum mismatch: {name}")
            manifest["files"] = {
                name: {"size": (target / name).stat().st_size, "sha256": digest.hexdigest()}
                for name, digest in zip(names, outgoing)}
            manifest["session_replay"] = {
                "policy": "previous_simulated_completion_plus_event_delay",
                "first_arrival": "original_first_assistant_event_timestamp",
                "delay": "last_previous_assistant_event_to_latest_current_input_event",
                "timing_quality": "approximate_log_events",
                "missing_or_inconsistent_delay": 0,
                "segment_policy": "split_at_filtered_call_gaps_and_history_resets",
                "parent_bundle": str(bundle), "counts": dict(counts),
            }
            (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            target.rename(output)
    print(f"Created {counts['requests']:,} requests in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DATA / "swe_chat/upstream")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    convert(args.bundle.expanduser().resolve(), args.upstream.expanduser().resolve(),
            args.output_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
