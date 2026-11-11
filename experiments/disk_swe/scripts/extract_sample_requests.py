"""Extract a small sample of Claude Code calls from SWE-chat raw transcripts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path.home() / "project/data/swe_chat"


def extract_calls(path, session_id, counts):
    calls = []
    pending = []
    current = None
    seen = set()
    tool_ids = set()
    with path.open(encoding="utf-8") as transcript:
        for number, line in enumerate(transcript, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("isSidechain"):
                continue
            if event.get("subtype") == "compact_boundary" or event.get("isCompactSummary"):
                counts["stopped_at_compaction"] += 1
                break
            kind = event.get("type")
            message = event.get("message") or {}
            content = message.get("content")
            if kind == "user" and content:
                if event.get("isMeta"):
                    counts["meta_user_events"] += 1
                blocks = content if isinstance(content, list) else []
                for block in blocks:
                    if block.get("type") == "tool_result" and block.get("tool_use_id") not in tool_ids:
                        counts["unmatched_tool_results"] += 1
                pending.append({"line": number, "timestamp": event.get("timestamp"), "content": content})
            elif kind == "assistant" and content:
                response_id = message.get("id")
                if not response_id or message.get("model") == "<synthetic>" or event.get("isApiErrorMessage"):
                    counts["stopped_at_ambiguous_assistant"] += 1
                    break
                if current is None or current["response_id"] != response_id:
                    if response_id in seen:
                        counts["stopped_at_noncontiguous_response"] += 1
                        break
                    seen.add(response_id)
                    current = {
                        "session_id": session_id,
                        "call_index": len(calls) + 1,
                        "response_id": response_id,
                        "api_request_id": event.get("requestId"),
                        "model": message.get("model"),
                        "first_response_event_at": event.get("timestamp"),
                        "time_source": "first_response_event" if event.get("timestamp") else None,
                        "input_delta": pending,
                        "response_events": [],
                    }
                    calls.append(current)
                    pending = []
                elif pending:
                    counts["stopped_at_interleaved_input"] += 1
                    break
                current["response_events"].append({"line": number, "content": content})
                if isinstance(content, list):
                    tool_ids.update(block["id"] for block in content
                                    if block.get("type") == "tool_use" and block.get("id"))
    counts["trailing_input_events"] += len(pending)
    return calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DATA / "upstream")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be positive")

    root = args.upstream.resolve()
    sessions = pq.read_table(root / "sessions.parquet", columns=[
        "session_id", "agent", "kind", "tool_call_count", "compact_transcript_delta_only",
    ]).to_pylist()
    logs = pq.read_table(root / "session_logs.parquet",
                         columns=["session_id", "transcript_path"]).to_pylist()
    paths = {row["session_id"]: row["transcript_path"] for row in logs if row["transcript_path"]}
    counts = Counter()
    selected = []
    records = []
    for session in sessions:
        if session["agent"] != "Claude Code" or session["kind"] or session["compact_transcript_delta_only"]:
            continue
        if not (session["tool_call_count"] or 0) >= 2:
            continue
        session_id = session["session_id"]
        if session_id not in paths:
            counts["missing_transcripts"] += 1
            continue
        path = (root / paths[session_id]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            counts["missing_transcripts"] += 1
            continue
        try:
            calls = extract_calls(path, session_id, counts)
        except (ValueError, UnicodeError):
            counts["invalid_transcripts"] += 1
            continue
        if len(calls) < 2:
            counts["insufficient_calls"] += 1
            continue
        selected.append({"session_id": session_id, "transcript_path": paths[session_id],
                         "calls": len(calls)})
        records.extend(calls)
        if len(selected) == args.sessions:
            break
    if len(selected) != args.sessions:
        parser.error(f"found only {len(selected)} usable sessions")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "source_requests.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"sessions": selected, "requests": len(records), "diagnostics": dict(counts)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                  encoding="utf-8")
    print(f"Extracted {len(records)} calls from {len(selected)} sessions into {args.output_dir}")


if __name__ == "__main__":
    main()
