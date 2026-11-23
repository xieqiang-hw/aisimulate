#!/usr/bin/env python3
"""Convert downloaded Qwen/Bailian and Mooncake traces to AISim SWE-style bundles.

Standard library only. No tokenization: source token counts and observed hash
relationships are retained. This is static timestamp replay, not agentic replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory


DATA = Path.home() / "project/data"
BLOCK_SIZE = 16
DATASETS = {
    "qwen-chat": ("qwen", "qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl", 16, 1000),
    "qwen-api": ("qwen", "qwen-bailian-usagetraces-anon/qwen_traceB_blksz_16.jsonl", 16, 1000),
    "qwen-thinking": ("qwen", "qwen-bailian-usagetraces-anon/qwen_thinking_blksz_16.jsonl", 16, 1000),
    "qwen-coder": ("qwen", "qwen-bailian-usagetraces-anon/qwen_coder_blksz_16.jsonl", 16, 1000),
    "mooncake-chat": ("mooncake", "Mooncake-FAST25-release/conversation_trace.jsonl", 512, 1),
    "mooncake-agent": ("mooncake", "Mooncake-FAST25-release/toolagent_trace.jsonl", 512, 1),
    "mooncake-synthetic": ("mooncake", "Mooncake-FAST25-release/synthetic_trace.jsonl", 512, 1),
}
UPSTREAM = {
    "qwen": "https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon",
    "mooncake": "https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release",
}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def milliseconds(value, multiplier):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("timestamp must be a finite nonnegative number")
    try:
        result = Decimal(str(value)) * multiplier
    except InvalidOperation as error:
        raise ValueError("invalid timestamp") from error
    if not result.is_finite() or result < 0:
        raise ValueError("timestamp must be finite and nonnegative")
    return result


def relative_timestamp(value, origin):
    result = value - origin
    return int(result) if result == result.to_integral_value() else float(result)


def parse_rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each line must be a JSON object")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            yield line_number, row


class HashIDs:
    """Disk-backed collision-free interning; cache size bounds conversion RAM."""

    def __init__(self, db):
        self.db = db
        self.next_id = 0
        db.execute("CREATE TABLE hashes (key BLOB PRIMARY KEY, id INTEGER NOT NULL)")
        self.intern = lru_cache(maxsize=65536)(self._intern)

    def private(self):
        value = self.next_id
        self.next_id += 1
        if value >= 2**32:
            raise ValueError("too many hash identities for the AISim u32 bridge")
        return value

    def _intern(self, key):
        result = self.db.execute("SELECT id FROM hashes WHERE key = ?", (key,)).fetchone()
        if result is not None:
            return result[0]
        value = self.private()
        self.db.execute("INSERT INTO hashes VALUES (?, ?)", (key, value))
        return value

    def convert(self, hashes, length, source_block_size):
        result = []
        parent = 0
        for index, raw_hash in enumerate(hashes):
            available = min(source_block_size, length - index * source_block_size)
            if available <= 0:
                break
            if available < source_block_size:
                # We cannot recover any finer prefix within a partially observed
                # source block. Keep it private, including its full 16-token pieces.
                result.extend(self.private() for _ in range(math.ceil(available / BLOCK_SIZE)))
                break
            # Intern the exact (parent identity, source block hash) pair. Unlike a
            # truncated digest, this cannot collide. Qwen hashes individual blocks;
            # Mooncake already hashes prefixes. Both retain correct prefix equality.
            key = b"p" + parent.to_bytes(8, "big") + raw_hash.to_bytes(8, "big")
            parent = self.intern(key) + 1
            for subblock in range(source_block_size // BLOCK_SIZE):
                key = b"s" + parent.to_bytes(8, "big") + subblock.to_bytes(2, "big")
                result.append(self.intern(key))
        if len(result) != math.ceil(length / BLOCK_SIZE):
            raise ValueError("converted block count does not cover the full prompt")
        return result


def stage_source(db, source, family, source_block, scale, args):
    db.execute("CREATE TABLE raw (line INTEGER PRIMARY KEY, ms REAL, payload TEXT)")
    db.execute("CREATE TABLE links (chat INTEGER PRIMARY KEY, parent INTEGER, root INTEGER)")
    counts = Counter()
    minimum = maximum = None
    for line_number, row in parse_rows(source):
        try:
            timestamp = milliseconds(row.get("timestamp"), scale)
            input_length = integer(row.get("input_length"), "input_length", 1)
            output_length = integer(row.get("output_length"), "output_length")
            hashes = row.get("hash_ids")
            if not isinstance(hashes, list) or len(hashes) != math.ceil(input_length / source_block):
                raise ValueError("hash_ids count must equal ceil(input_length / source_block_size)")
            if any(type(value) is not int or not 0 <= value < 2**64 for value in hashes):
                raise ValueError("hash_ids must be unsigned 64-bit integers")
            if family == "qwen":
                chat = integer(row.get("chat_id"), "chat_id")
                parent = integer(row.get("parent_chat_id"), "parent_chat_id", -1)
                integer(row.get("turn"), "turn", 1)
                db.execute("INSERT INTO links VALUES (?, ?, NULL)", (chat, parent))
            db.execute("INSERT INTO raw VALUES (?, ?, ?)", (line_number, float(timestamp), dump(row)))
        except (ValueError, sqlite3.IntegrityError) as error:
            raise ValueError(f"{source}:{line_number}: {error}") from error
        counts["source_requests"] += 1
        minimum = timestamp if minimum is None else min(minimum, timestamp)
        maximum = timestamp if maximum is None else max(maximum, timestamp)
    if minimum is None:
        raise ValueError(f"empty input: {source}")
    db.execute("CREATE INDEX raw_time ON raw(ms, line)")
    db.commit()
    if family == "qwen":
        # Resolve all links before filtering: a parent can appear later in the file
        # or be outside the selected window/context limit. Detect broken chains.
        links = dict(db.execute("SELECT chat, parent FROM links"))
        roots = {}
        for chat in links:
            trail, visited, current = [], set(), chat
            while current not in roots:
                if current in visited:
                    raise ValueError(f"cycle in Qwen parent chain at chat_id={current}")
                visited.add(current)
                trail.append(current)
                if current not in links:
                    raise ValueError(f"missing Qwen parent chat_id={current}")
                parent = links[current]
                if parent == -1:
                    root = current
                    break
                current = parent
            else:
                root = roots[current]
            for item in trail:
                roots[item] = root
        db.executemany("UPDATE links SET root=? WHERE chat=?", ((root, chat) for chat, root in roots.items()))
        counts["source_sessions"] = len(set(roots.values()))
        db.commit()
    return counts, minimum, maximum


def source_revision(source):
    if not (source.parent / ".git").exists():
        return None
    result = subprocess.run(["git", "-C", str(source.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def prepare(name, source, output, args):
    family, _, source_block, scale = DATASETS[name]
    if output.exists():
        raise ValueError(f"output already exists (use a new --output-root): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    source_digest = sha256(source)
    with TemporaryDirectory(prefix=f".{name}-", dir=output.parent) as temporary:
        work = Path(temporary)
        with sqlite3.connect(work / "staging.sqlite") as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            counts, source_min, source_max = stage_source(db, source, family, source_block, scale, args)
            identities = HashIDs(db)
            bundle = work / "bundle"
            bundle.mkdir()
            sessions, types = set(), Counter()
            first = last = None
            first_selected_source_time = None
            min_input = min_output = None
            max_input = max_output = total_input = total_output = 0
            max_turn = 0
            window_start = source_min + Decimal(str(args.start_seconds)) * 1000
            window_end = None if args.duration_seconds is None else window_start + Decimal(str(args.duration_seconds)) * 1000
            with (bundle / "requests.jsonl").open("w", encoding="utf-8") as trace, (bundle / "source_requests.jsonl").open("w", encoding="utf-8") as sidecar:
                rows = db.execute("SELECT line, payload FROM raw ORDER BY ms, line")
                for line_number, payload in rows:
                    row = json.loads(payload)
                    timestamp = milliseconds(row["timestamp"], scale)
                    if timestamp < window_start or (window_end is not None and timestamp >= window_end):
                        counts["excluded_time_window"] += 1
                        continue
                    if args.types and row.get("type") not in args.types:
                        counts["excluded_request_type"] += 1
                        continue
                    length, output_length = row["input_length"], row["output_length"]
                    if output_length == 0:
                        counts["excluded_zero_output"] += 1
                        continue
                    if length + output_length > args.context_limit:
                        counts["excluded_context_limit"] += 1
                        continue
                    if args.max_requests and counts["emitted_requests"] >= args.max_requests:
                        counts["excluded_request_limit"] += 1
                        continue
                    if first is None:
                        first = timestamp
                        first_selected_source_time = row["timestamp"]
                    last = timestamp
                    trace_row = {
                        "timestamp": relative_timestamp(timestamp, first),
                        "input_length": length,
                        "output_length": output_length,
                        "hash_ids": identities.convert(row["hash_ids"], length, source_block),
                    }
                    source_row = {
                        "trace_row": counts["emitted_requests"],
                        "source_line": line_number,
                        "source_timestamp": row["timestamp"],
                        "timestamp": trace_row["timestamp"],
                        "request_type": row.get("type"),
                    }
                    if family == "qwen":
                        root = db.execute("SELECT root FROM links WHERE chat=?", (row["chat_id"],)).fetchone()[0]
                        session_id = f"{name}:{root}"
                        source_row.update(session_id=session_id, request_id=f"{name}:{row['chat_id']}",
                                          source_chat_id=row["chat_id"], parent_chat_id=row["parent_chat_id"], turn=row["turn"])
                        sessions.add(session_id)
                        max_turn = max(max_turn, row["turn"])
                    else:
                        source_row.update(session_id=None, request_id=f"{name}:line-{line_number}")
                    trace.write(dump(trace_row) + "\n")
                    sidecar.write(dump(source_row) + "\n")
                    counts["emitted_requests"] += 1
                    types[str(row.get("type", "unspecified"))] += 1
                    total_input += length
                    total_output += output_length
                    min_input = length if min_input is None else min(min_input, length)
                    min_output = output_length if min_output is None else min(min_output, output_length)
                    max_input, max_output = max(max_input, length), max(max_output, output_length)
            if not counts["emitted_requests"]:
                raise ValueError(f"{name}: no usable requests after filtering")
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("source file changed during conversion")
            manifest = {
                "format": "mooncake", "schema_version": 1, "dataset": name,
                "trace_block_size": BLOCK_SIZE, "source_block_size": source_block,
                "context_limit": args.context_limit, "request_count": counts["emitted_requests"],
                "session_count": len(sessions) if family == "qwen" else None,
                "model_mapping": {"target_model": args.target_model},
                "includes_input_token_ids": False, "includes_output_token_ids": False,
                "model_mapping_policy": "source token counts retained; no retokenization to target model",
                "scope": "static per-request replay of observed timestamps; session metadata in sidecar only",
                "timestamp_policy": "relative milliseconds, normalized to first emitted request; no speedup",
                "arrival_kind": "synthetic_poisson" if name == "mooncake-synthetic" else "observed",
                "prefix_policy": "exact chain interning; full source blocks split into 16-token virtual blocks; incomplete source blocks private",
                "output_policy": "source output lengths only; output tokens and D-to-next-P reuse cannot be recovered",
                "source": {"path": str(source), "size": before.st_size, "sha256": source_digest,
                           "repository": UPSTREAM[family], "checkout_revision": source_revision(source),
                           "license": "Apache-2.0", "modified": True},
                "time_range": {"source_first_ms": float(source_min), "source_last_ms": float(source_max),
                               "source_span_seconds": float((source_max - source_min) / 1000),
                               "selected_first_source_timestamp": first_selected_source_time,
                               "selected_span_seconds": float((last - first) / 1000)},
                "selection": {"start_seconds": args.start_seconds, "duration_seconds": args.duration_seconds,
                              "max_requests": args.max_requests, "request_types": args.types},
                "statistics": {"input_tokens_total": total_input, "output_tokens_total": total_output,
                               "input_tokens_min": min_input, "input_tokens_max": max_input,
                               "output_tokens_min": min_output, "output_tokens_max": max_output,
                               "maximum_source_turn": max_turn if family == "qwen" else None,
                               "request_types": dict(types)},
                "diagnostics": dict(counts),
                "files": {path.name: {"size": path.stat().st_size, "sha256": sha256(path)} for path in bundle.iterdir()},
            }
            # Bundle metadata attributes derived data; code is original to this repo.
            license_file = source.parent / "LICENSE"
            if license_file.is_file():
                (bundle / "SOURCE_LICENSE").write_bytes(license_file.read_bytes())
                manifest["files"]["SOURCE_LICENSE"] = {"size": license_file.stat().st_size, "sha256": sha256(license_file)}
            (bundle / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        bundle.rename(output)
    print(f"{name}: {counts['emitted_requests']:,} requests -> {output}", flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=[*DATASETS, "all"], default=["qwen-chat", "mooncake-chat"])
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--input", type=Path, help="override source path; requires one dataset")
    parser.add_argument("--output-root", type=Path, default=DATA / "chat/derived/mooncake-v1")
    parser.add_argument("--target-model", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--context-limit", type=int, default=131072, help="filter on input + output; never truncate")
    parser.add_argument("--max-requests", type=int, default=0, help="0: full trace; otherwise first N eligible requests")
    parser.add_argument("--start-seconds", type=float, default=0, help="window offset from source's first timestamp")
    parser.add_argument("--duration-seconds", type=float, help="half-open selection window duration")
    parser.add_argument("--types", nargs="+", help="optional Qwen type filter, e.g. --types text; default retains all")
    args = parser.parse_args(argv)
    if args.context_limit <= 0 or args.max_requests < 0:
        parser.error("context-limit must be positive; max-requests must be nonnegative")
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        parser.error("start-seconds must be finite and nonnegative")
    if args.duration_seconds is not None and (not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0):
        parser.error("duration-seconds must be finite and positive")
    names = list(DATASETS) if "all" in args.dataset else list(dict.fromkeys(args.dataset))
    if args.input and len(names) != 1:
        parser.error("--input requires exactly one --dataset")
    if args.types and any(DATASETS[name][0] != "qwen" for name in names):
        parser.error("--types is only supported for Qwen datasets")
    jobs = []
    for name in names:
        source = (args.input or args.data_root / DATASETS[name][1]).expanduser().resolve()
        output = (args.output_root / name).expanduser().resolve()
        if not source.is_file():
            parser.error(f"source does not exist: {source}")
        if output.exists():
            parser.error(f"output already exists: {output}")
        jobs.append((name, source, output))
    for name, source, output in jobs:
        try:
            prepare(name, source, output, args)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{name}: {error}\n")


if __name__ == "__main__":
    main()
