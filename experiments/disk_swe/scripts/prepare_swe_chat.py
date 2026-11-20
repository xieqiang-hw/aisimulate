"""Convert SWE-chat Claude Code main sessions to ordinary AISim Mooncake."""
import argparse
from array import array
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
import hashlib
from itertools import islice
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from transformers import AutoTokenizer

DATA = Path.home() / 'project/data'
MODEL = 'zai-org/GLM-5.2-FP8'
BLOCK_SIZE = 16


def linked_tool_results(pending, current_tools):
    """Only defer results tied to tools already emitted by this response."""
    for event in pending:
        blocks = event["content"]
        if not isinstance(blocks, list) or not blocks:
            return False
        for block in blocks:
            tool_id = block.get("tool_use_id")
            if block.get("type") != "tool_result" or tool_id not in current_tools:
                return False
            origin = event.get("tool_origin")
            if origin and origin != current_tools[tool_id]:
                return False
    return True


def independent_root(event, content):
    # An explicit root human input starts a new logged history. A compaction
    # summary alone cannot establish which original messages were preserved.
    if ("parentUuid" not in event or event["parentUuid"] is not None
            or not event.get("uuid") or event.get("isMeta") or event.get("isCompactSummary")):
        return False
    if isinstance(content, str):
        return bool(content.strip())
    return isinstance(content, list) and bool(content) and all(
        block.get("type") in {"text", "image", "document"} for block in content
    )


def extract_calls(path, session_id, counts):
    calls = []
    pending = []
    current = None
    seen = set()
    tool_ids = set()
    current_tools = {}
    waiting_for_root = False
    history_reset = False

    def interrupt(reason, discard_current):
        nonlocal current, waiting_for_root, history_reset
        counts[reason] += 1
        if discard_current and current is not None:
            calls.pop()
            counts["incomplete_calls_discarded"] += 1
        current = None
        pending.clear()
        current_tools.clear()
        tool_ids.clear()
        waiting_for_root = True
        history_reset = True

    with path.open(encoding="utf-8") as transcript:
        for number, line in enumerate(transcript, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("isSidechain"):
                continue
            kind = event.get("type")
            message = event.get("message") or {}
            content = message.get("content")
            if waiting_for_root:
                if kind != "user" or not independent_root(event, content):
                    continue
                waiting_for_root = False
                counts["resumed_at_explicit_root"] += 1
            if event.get("subtype") == "compact_boundary" or event.get("isCompactSummary"):
                interrupt("paused_at_compaction", discard_current=False)
                continue
            if kind == "user" and content:
                if event.get("isMeta"):
                    counts["meta_user_events"] += 1
                blocks = content if isinstance(content, list) else []
                for block in blocks:
                    if block.get("type") == "tool_result" and block.get("tool_use_id") not in tool_ids:
                        counts["unmatched_tool_results"] += 1
                pending.append({
                    "line": number, "timestamp": event.get("timestamp"), "content": content,
                    "tool_origin": event.get("sourceToolAssistantUUID"),
                })
            elif kind == "assistant" and content:
                response_id = message.get("id")
                request_id = event.get("requestId")
                if not response_id or message.get("model") == "<synthetic>" or event.get("isApiErrorMessage"):
                    interrupt("paused_at_ambiguous_assistant", discard_current=True)
                    continue
                if current is None or current["response_id"] != response_id:
                    if response_id in seen:
                        interrupt("paused_at_noncontiguous_response", discard_current=True)
                        continue
                    seen.add(response_id)
                    current = {
                        "session_id": session_id,
                        "call_index": len(calls) + 1,
                        "response_id": response_id,
                        "api_request_id": request_id,
                        "model": message.get("model"),
                        "first_response_event_at": event.get("timestamp"),
                        "time_source": "first_response_event" if event.get("timestamp") else None,
                        "input_delta": pending,
                        "response_events": [],
                        "history_reset": history_reset,
                    }
                    calls.append(current)
                    history_reset = False
                    pending = []
                    current_tools.clear()
                elif pending:
                    if not request_id or current["api_request_id"] != request_id or not linked_tool_results(pending, current_tools):
                        interrupt("paused_at_unresolved_interleave", discard_current=True)
                        continue
                    if current.get("verified_interleave_count", 0) != len(pending):
                        counts["verified_interleave_boundaries"] += 1
                        current["verified_interleave_count"] = len(pending)
                elif request_id and current["api_request_id"] and request_id != current["api_request_id"]:
                    interrupt("paused_at_conflicting_request_id", discard_current=True)
                    continue
                current["response_events"].append({"line": number, "content": content})
                current["last_response_event_at"] = event.get("timestamp")
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "tool_use" and block.get("id"):
                            tool_ids.add(block["id"])
                            current_tools[block["id"]] = event.get("uuid")
    counts["trailing_input_events"] += len(pending)
    return calls


def input_messages(events):
    for event in events:
        content = event["content"]
        if isinstance(content, str):
            yield {"role": "user", "content": content}
            continue
        for block in content:
            if block["type"] == "tool_result":
                result = block.get("content") or ""
                if isinstance(result, list):
                    # Claude Code can return tool names without the original tool schemas.
                    # Keep those names as text instead of entering GLM's schema lookup.
                    result = [
                        {"type": "text", "text": "[tool reference: " + str(
                            item.get("tool_name") or item.get("name") or "unknown"
                        ) + "]"}
                        if isinstance(item, dict) and item.get("type") == "tool_reference"
                        else item
                        for item in result
                    ]
                yield {"role": "tool", "content": result}
            else:
                yield {"role": "user", "content": [block]}


def assistant_message(events):
    text, reasoning, tools = [], [], []
    for event in events:
        content = event["content"]
        if isinstance(content, str):
            text.append(content)
            continue
        for block in content:
            kind = block["type"]
            if kind == "text":
                text.append(block["text"])
            elif kind == "thinking":
                reasoning.append(block.get("thinking") or "")
            elif kind == "tool_use":
                tools.append({"function": {"name": block["name"], "arguments": block["input"]}})
    return {"role": "assistant", "content": "".join(text),
            "reasoning_content": "".join(reasoning), "tool_calls": tools}


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def session_rows(root, counts):
    logs = pq.read_table(root / "session_logs.parquet",
                         columns=["session_id", "transcript_path"]).to_pylist()
    paths = {row["session_id"]: row["transcript_path"] for row in logs if row["transcript_path"]}
    sessions = pq.ParquetFile(root / "sessions.parquet")
    seen = set()
    for batch in sessions.iter_batches(columns=[
        "session_id", "agent", "kind", "compact_transcript_delta_only",
    ]):
        for row in batch.to_pylist():
            counts["indexed_sessions"] += 1
            if row["agent"] != "Claude Code":
                counts["other_client_sessions"] += 1
                continue
            if row["kind"] or row["compact_transcript_delta_only"]:
                counts["non_main_or_delta_only_sessions"] += 1
                continue
            session = row["session_id"]
            if session in seen:
                counts["duplicate_sessions"] += 1
                continue
            seen.add(session)
            relative = paths.get(session)
            path = (root / relative).resolve() if relative else None
            if path is None or not path.is_relative_to(root) or not path.is_file():
                counts["missing_transcripts"] += 1
                continue
            yield session, relative, path


def stage_session(session, relative, calls, tokenizer, template, limit, counts, models, ids, include_input_ids=False, include_output_ids=True):
    if calls and not list(input_messages(calls[0]['input_delta'])):
        counts['sessions_missing_initial_input'] += 1
        counts['calls_missing_initial_history'] += len(calls)
        models.update(row['model'] or '<missing>' for row in calls)
        counts['extracted_calls'] += len(calls)
        return
    history = []
    for row in calls:
        counts["extracted_calls"] += 1
        models[row["model"] or "<missing>"] += 1
        if row.get("history_reset"):
            history.clear()
        history.extend(input_messages(row["input_delta"]))
        answer = assistant_message(row["response_events"])
        prompt = tokenizer.apply_chat_template(
            history, chat_template=template, tokenize=False, add_generation_prompt=True)
        history.append(answer)
        rendered = tokenizer.apply_chat_template(
            history, chat_template=template, tokenize=False, add_generation_prompt=False)
        if not rendered.startswith(prompt):
            counts["unaligned_template_calls"] += 1
            continue
        tokens = tokenizer.encode(prompt, add_special_tokens=False, verbose=False)
        output = tokenizer.encode(rendered[len(prompt):], add_special_tokens=False, verbose=False)
        if tokenizer.eos_token_id is not None:
            output.append(tokenizer.eos_token_id)
        counts["tokenized_input_tokens"] += len(tokens)
        if len(tokens) + len(output) > limit:
            counts["over_context_limit"] += 1
            counts["over_context_input_tokens"] += len(tokens)
            continue
        if not output:
            counts["empty_output"] += 1
            continue
        stamp = row["first_response_event_at"]
        try:
            date = datetime.fromisoformat(stamp)
            if date.utcoffset() is None:
                raise ValueError("missing timezone")
        except (TypeError, ValueError):
            counts["invalid_or_missing_timestamp"] += 1
            continue

        hashes = []
        parent = bytes(32)
        for start in range(0, len(tokens), BLOCK_SIZE):
            block = tokens[start:start + BLOCK_SIZE]
            if len(block) == BLOCK_SIZE:
                parent = hashlib.sha256(parent + struct.pack("<16I", *block)).digest()
            else:
                parent = hashlib.sha256(parent + struct.pack(f"<{len(block)}I", *block)
                                        + f"{session}:{row['call_index']}".encode()).digest()
            hashes.append(ids(parent))
        trace = {"input_length": len(tokens), "output_length": len(output), "hash_ids": hashes}
        if include_output_ids:
            trace["output_token_ids"] = output
        if include_input_ids:
            trace["input_token_ids"] = tokens
        source = {key: row[key] for key in ("session_id", "call_index", "response_id",
                                          "first_response_event_at", "time_source", "model")}
        source.update(transcript_path=relative, api_request_id=row["api_request_id"],
                      history_reset=row.get("history_reset", False))
        if include_output_ids:
            source["output_token_ids"] = output
        yield int(date.timestamp() * 1000), session, row["call_index"], trace, source
        counts["emitted_requests"] += 1
        counts["emitted_input_tokens"] += len(tokens)


def session_requests(session, relative, path, tokenizer, template, limit, counts, models, ids,
                     include_input_ids, include_output_ids):
    counts["attempted_sessions"] += 1
    try:
        calls = extract_calls(path, session, counts)
    except (OSError, ValueError, AttributeError, TypeError):
        counts["invalid_transcripts"] += 1
        return
    if not calls:
        counts["sessions_without_calls"] += 1
    yield from stage_session(session, relative, calls, tokenizer, template, limit, counts, models, ids,
                             include_input_ids, include_output_ids)


def load_tokenizer(path):
    return (
        AutoTokenizer.from_pretrained(path, local_files_only=True),
        (path / "chat_template.jinja").read_text(encoding="utf-8"),
    )


def initialize_worker(tokenizer_path, limit, include_input_ids, include_output_ids):
    global worker_options
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer, template = load_tokenizer(tokenizer_path)
    worker_options = tokenizer, template, limit, include_input_ids, include_output_ids


def prepare_session(job):
    tokenizer, template, limit, include_input_ids, include_output_ids = worker_options
    counts, models, local_ids = Counter(), Counter(), {}

    def local_id(digest):
        return local_ids.setdefault(digest, len(local_ids))

    rows = []
    for row in session_requests(*job, tokenizer, template, limit, counts, models, local_id,
                                include_input_ids, include_output_ids):
        trace = row[3]
        trace["hash_ids"] = array("I", trace["hash_ids"])
        rows.append(row)
    return counts, models, list(local_ids), rows


def parallel_sessions(jobs, workers, tokenizer_path, limit, include_input_ids, include_output_ids):
    """Commit results in source order with a bounded queue of session batches."""
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize_worker,
        initargs=(tokenizer_path, limit, include_input_ids, include_output_ids),
    ) as pool:
        pending = deque(pool.submit(prepare_session, job) for job in islice(jobs, workers * 2))
        while pending:
            result = pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(prepare_session, job))
            yield result


def insert_requests(db, rows):
    db.executemany("INSERT INTO requests VALUES (?, ?, ?, ?, ?)", (
        (ms, session, call, json.dumps(trace), json.dumps(source))
        for ms, session, call, trace, source in rows
    ))
    db.commit()


def remap_requests(rows, global_ids):
    for row in rows:
        trace = row[3]
        trace["hash_ids"] = [global_ids[index] for index in trace["hash_ids"]]
        yield row


def print_progress(counts):
    if counts["attempted_sessions"] % 100 == 0:
        print(f"Scanned {counts['attempted_sessions']:,} sessions, emitted "
              f"{counts['emitted_requests']:,} requests", flush=True)


def export(db, directory):
    with (directory / "requests.jsonl").open("w", encoding="utf-8") as requests, (
        directory / "source_requests.jsonl"
    ).open("w", encoding="utf-8") as sources:
        origin = None
        for index, (ms, request, source) in enumerate(db.execute(
            "SELECT ms, request, source FROM requests ORDER BY ms, session, call"
        )):
            if origin is None:
                origin = ms
            trace_row, source_row = json.loads(request), json.loads(source)
            trace_row["timestamp"] = ms - origin
            source_row.update(trace_row=index, timestamp=ms - origin)
            requests.write(json.dumps(trace_row) + "\n")
            sources.write(json.dumps(source_row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DATA / "swe_chat/upstream")
    parser.add_argument("--tokenizer", type=Path, default=DATA / "GLM_5.2_tokenizer")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "project/data/swe_chat/derived/mooncake-v1/claude-code-main-glm52-128k")
    parser.add_argument("--context-limit", type=int, default=131072)
    parser.add_argument("--max-sessions", type=int, default=0, help="0 scans all eligible sessions")
    parser.add_argument("--include-input-token-ids", action="store_true",
                        help="export exact prompt tokens for enclosure cross-turn reuse")
    parser.add_argument("--omit-output-token-ids", action="store_true",
                        help="export output lengths without output token IDs")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel session conversion workers (default: 1)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.context_limit < 1 or args.max_sessions < 0:
        parser.error("context limit must be positive and max sessions nonnegative")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error("output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, models = Counter(), Counter()
    with TemporaryDirectory(prefix=".swe-mooncake-", dir=output.parent) as tmp:
        work = Path(tmp)
        with sqlite3.connect(work / "staging.sqlite") as db:
            db.execute("CREATE TABLE requests (ms INTEGER, session TEXT, call INTEGER, "
                       "request TEXT, source TEXT, PRIMARY KEY (ms, session, call))")
            db.execute("CREATE TABLE hashes (id INTEGER PRIMARY KEY, digest BLOB UNIQUE)")
            next_id = 1

            @lru_cache(maxsize=65536)
            def ids(digest):
                nonlocal next_id
                match = db.execute("SELECT id FROM hashes WHERE digest = ?", (digest,)).fetchone()
                if match:
                    return match[0]
                if next_id >= 2**32:
                    raise ValueError("too many distinct prefix blocks for u32 IDs")
                value = next_id
                db.execute("INSERT INTO hashes VALUES (?, ?)", (value, digest))
                next_id += 1
                return value

            jobs = session_rows(args.upstream.resolve(), counts)
            if args.max_sessions:
                jobs = islice(jobs, args.max_sessions)
            include_output_ids = not args.omit_output_token_ids
            if args.workers == 1:
                tokenizer, template = load_tokenizer(args.tokenizer)
                for job in jobs:
                    rows = session_requests(*job, tokenizer, template, args.context_limit, counts, models,
                                            ids, args.include_input_token_ids, include_output_ids)
                    insert_requests(db, rows)
                    print_progress(counts)
            else:
                results = parallel_sessions(jobs, args.workers, args.tokenizer, args.context_limit,
                                            args.include_input_token_ids, include_output_ids)
                for session_counts, session_models, digests, rows in results:
                    global_ids = [ids(digest) for digest in digests]
                    insert_requests(db, remap_requests(rows, global_ids))
                    counts.update(session_counts)
                    models.update(session_models)
                    print_progress(counts)
            if not counts["emitted_requests"]:
                parser.error("no usable requests")
            bundle = work / "bundle"
            bundle.mkdir()
            export(db, bundle)
        manifest = {
            "format": "mooncake", "trace_block_size": BLOCK_SIZE,
            "includes_input_token_ids": args.include_input_token_ids,
            "context_limit": args.context_limit, "request_count": counts["emitted_requests"],
            "model_mapping": {"target_model": MODEL},
            "scope": "Claude Code main transcripts with verified tool interleaving and explicit-root recovery",
            "extraction_policy": "requestId and tool ID linkage required to join interleaved fragments",
            "includes_output_token_ids": not args.omit_output_token_ids,
            "max_sessions": args.max_sessions,
            "timestamp_policy": "first response event proxy, relative milliseconds",
            "history_policy": "logged messages only; system prompts and tool schemas omitted",
            "output_policy": "rendered completion plus EOS; lengths always recorded",
            "source_models_by_extracted_call": dict(models.most_common()),
            "diagnostics": dict(counts),
            "tokenizer_files": {name: sha256(args.tokenizer / name) for name in
                                ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")},
            "files": {path.name: {"size": path.stat().st_size, "sha256": sha256(path)}
                      for path in bundle.iterdir()},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")
        bundle.rename(output)
    print(f"Created {counts['emitted_requests']:,} requests in {output}", flush=True)


if __name__ == "__main__":
    main()
