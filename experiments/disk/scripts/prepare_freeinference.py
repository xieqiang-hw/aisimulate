"""Convert a FreeInference interval to a portable, disk-backed request bundle.

This original adapter consumes the CC-BY-4.0 dataset at the revision below;
it does not copy upstream implementation code. Derived data and attribution
are written to the shared trace store, never to the simulator repository.
PyArrow is needed only for conversion, not for --help or the unit tests.
"""

from __future__ import annotations

import argparse
import array
from collections import Counter, OrderedDict
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile


REVISION = "90a627adefb7f7b88708484bcbea64010f76492a"
DATASET_URL = "https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace"
ROLE_NAMES = ("system", "user", "assistant", "tool", "tool_definitions")
METADATA_COLUMNS = (
    "session_id", "step", "start_ms", "end_ms", "model", "provider",
    "prompt_tokens", "completion_tokens", "cache_read_tokens", "role_tokens",
    "finish_reason",
)
TRACE_BLOCK_SIZE = 16
BIN_MS = 300_000
DEFAULT_TARGET_MODEL = "deepseek-ai/DeepSeek-V3.2"


def week_bounds(week: str) -> tuple[int, int]:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}", week):
        raise ValueError("week must be YYYY-MM-DD_YYYY-MM-DD or all")
    first, last = (dt.date.fromisoformat(x) for x in week.split("_"))
    if (last - first).days != 6:
        raise ValueError("week must cover seven consecutive calendar days")
    return (parse_utc(first.isoformat() + "T00:00:00Z"),
            parse_utc((last + dt.timedelta(days=1)).isoformat() + "T00:00:00Z"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def utc_text(milliseconds: int) -> str:
    return dt.datetime.fromtimestamp(milliseconds / 1000, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> int:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Time must include a UTC offset, for example 2026-09-03T08:00:00Z")
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    delta = parsed.astimezone(dt.timezone.utc) - epoch
    microseconds = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    if microseconds % 1000:
        raise ValueError("Sub-millisecond boundaries are not supported")
    return microseconds // 1000


def integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def make_options(*, week: str, source_model: str, start_utc: str, end_utc: str,
                 warmup_seconds: int = 0, context_limit: int = 131072,
                 context_policy: str = "error", target_model: str = DEFAULT_TARGET_MODEL,
                 trace_format: str = "mooncake") -> dict:
    if week != "all":
        week_start, week_end = week_bounds(week)
    if not isinstance(source_model, str) or not source_model.strip():
        raise ValueError("source-model must be a nonempty exact model label or all")
    if not isinstance(target_model, str) or not target_model.strip():
        raise ValueError("target-model must be a nonempty model ID")
    if trace_format not in {"mooncake", "mooncake-spooled"}:
        raise ValueError("format must be mooncake or mooncake-spooled")
    integer(warmup_seconds, "warmup-seconds")
    integer(context_limit, "context-limit", minimum=1)
    if context_policy not in {"error", "filter"}:
        raise ValueError("context-policy must be error or filter")
    start, end = parse_utc(start_utc), parse_utc(end_utc)
    replay_start = start - warmup_seconds * 1000
    if not replay_start <= start < end:
        raise ValueError("Scoring end must be later than scoring start")
    if week != "all" and not week_start <= replay_start <= start < end <= week_end:
        raise ValueError("Warmup and scoring interval must stay within the selected UTC week")
    return {
        "week": week, "source_model": source_model, "target_model": target_model, "format": trace_format,
        "score_start_ms": start, "score_end_ms": end,
        "replay_start_ms": replay_start,
        "score_start_offset_ms": start - replay_start,
        "score_end_offset_ms": end - replay_start,
        "warmup_seconds": warmup_seconds,
        "start_utc": utc_text(start), "end_utc": utc_text(end),
        "context_limit": context_limit, "context_policy": context_policy,
    }


class SelectionAccumulator:
    """Scalar-only accounting; memory grows with time bins, never request count."""

    def __init__(self, options: dict):
        self.options = options
        self.counts = Counter({key: 0 for key in (
            "source_rows", "matching_requests", "zero_output_excluded", "context_excluded",
            "emitted_requests", "matching_warmup_requests", "matching_score_requests",
            "emitted_warmup_requests", "emitted_score_requests",
        )})
        self.model_counts = {}
        start = options["score_start_ms"]
        first_bin = (options["replay_start_ms"] - start) // BIN_MS
        last_bin = (options["score_end_ms"] - 1 - start) // BIN_MS
        self.bins = {index: {
            "bin_start_ms": start + index * BIN_MS,
            "bin_end_ms": start + (index + 1) * BIN_MS,
            "matching_requests": 0, "emitted_requests": 0,
            "zero_output_excluded": 0, "context_excluded": 0,
            "matching_input_tokens": 0, "emitted_input_tokens": 0,
            "matching_output_tokens": 0, "emitted_output_tokens": 0,
        } for index in range(first_bin, last_bin + 1)}

    def consume(self, row_index: int, row: dict) -> dict | None:
        options, counts = self.options, self.counts
        start = options["score_start_ms"]
        counts["source_rows"] += 1
        integer(row_index, "source row index")
        timestamp = integer(row.get("start_ms"), f"row {row_index} start_ms")
        if options["source_model"] != "all" and row.get("model") != options["source_model"]:
            return None
        if not options["replay_start_ms"] <= timestamp < options["score_end_ms"]:
            return None
        if not isinstance(row.get("model"), str) or not row["model"]:
            raise ValueError(f"row {row_index}: source model is missing")
        roles = row.get("role_tokens")
        if not isinstance(roles, dict) or set(roles) != set(ROLE_NAMES):
            raise ValueError(f"row {row_index}: role_tokens must contain the five declared roles")
        isl = sum(integer(roles[name], f"row {row_index} role_tokens.{name}") for name in ROLE_NAMES)
        integer(isl, f"row {row_index} o200k input length", minimum=1)
        osl = integer(row.get("completion_tokens"), f"row {row_index} completion_tokens")
        for field in ("prompt_tokens", "cache_read_tokens"):
            if row.get(field) is not None:
                integer(row[field], f"row {row_index} {field}")
        if row.get("end_ms") is not None:
            integer(row["end_ms"], f"row {row_index} end_ms")
        if not isinstance(row.get("session_id"), str) or not row["session_id"]:
            raise ValueError(f"row {row_index}: session_id is missing")
        integer(row.get("step"), f"row {row_index} step")
        phase = "warmup" if timestamp < start else "score"
        counts["matching_requests"] += 1
        counts[f"matching_{phase}_requests"] += 1
        bucket = self.bins[(timestamp - start) // BIN_MS]
        model_counts = self.model_counts.setdefault(row["model"], Counter({
            "matching_requests": 0, "emitted_requests": 0,
            "zero_output_excluded": 0, "context_excluded": 0}))
        model_counts["matching_requests"] += 1
        bucket["matching_requests"] += 1
        bucket["matching_input_tokens"] += isl
        bucket["matching_output_tokens"] += osl
        # First-round TTFT/ITL cohort excludes zero-completion tool-only events.
        # Never invent one output token. This exclusion is explicit in every bin.
        if osl == 0:
            reason = "zero_output_excluded"
        elif isl >= options["context_limit"] or isl + osl > options["context_limit"]:
            reason = "context_excluded"
        else:
            reason = None
        if reason:
            counts[reason] += 1
            bucket[reason] += 1
            model_counts[reason] += 1
            return None
        counts["emitted_requests"] += 1
        counts[f"emitted_{phase}_requests"] += 1
        bucket["emitted_requests"] += 1
        bucket["emitted_input_tokens"] += isl
        bucket["emitted_output_tokens"] += osl
        model_counts["emitted_requests"] += 1
        return {**row, "source_row_index": row_index,
                "input_length": isl, "output_length": osl, "phase": phase}

    def finish(self) -> dict:
        options, counts = self.options, self.counts
        if counts["context_excluded"] and options["context_policy"] == "error":
            raise ValueError(
                f"{counts['context_excluded']} of {counts['matching_requests']} matching requests exceed "
                f"context={options['context_limit']}. No output was published. "
                "Choose another window/model or explicitly use --context-policy filter."
            )
        if not counts["emitted_score_requests"]:
            raise ValueError(f"No scoring requests remain; selection counts={dict(counts)}")
        return {**options, "counts": dict(counts), "exclusion_bins": list(self.bins.values()),
                "source_model_counts": {key: dict(value) for key, value in sorted(self.model_counts.items())},
                "exclusion_bin_ms": BIN_MS,
                "timestamp_policy": "Stable chronological order by start_ms, source week, original parquet row index; gaps and ties retained.",
                "zero_output_policy": "Explicitly excluded; no fabricated output token; counts retained per time bin.",
                "input_length_semantics": "sum(role_tokens), o200k_base proxy; not exact target-model tokenization.",
                "output_length_semantics": "Unmodified provider completion_tokens; target tokenizer conversion unavailable.",
                "cache_semantics": "Chained o200k 16-token prefixes; request-private unknown tail; provider cache_read_tokens is provenance only.",
                "arrival_semantics": "Open-loop request replay; no agent dependency feedback or original provider service latency injected.",
                "cache_initial_state": "cold at replay_start; preceding warmup requests simulated when requested"}


def select_requests(indexed_rows, options: dict) -> tuple[list[dict], dict]:
    """Small-fixture helper. Production selection is spooled by DiskPreparation."""
    accumulator = SelectionAccumulator(options)
    selected, seen_indices = [], set()
    for row_index, row in indexed_rows:
        key = (row.get("source_week", options["week"]), row_index)
        if key in seen_indices:
            raise ValueError(f"Duplicate source row index {key}")
        seen_indices.add(key)
        result = accumulator.consume(row_index, row)
        if result is not None:
            selected.append(result)
    selected.sort(key=lambda r: (r["start_ms"], r.get("source_week", options["week"]), r["source_row_index"]))
    return selected, accumulator.finish()


def verify_snapshot_source(root: Path, week: str) -> tuple[Path, dict]:
    week_bounds(week)
    description = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "upstream_manifest.json").read_text(encoding="utf-8"))
    if description.get("revision") != REVISION or manifest.get("revision") != REVISION:
        raise ValueError("Source metadata does not match the adapter's pinned dataset revision")
    relative = f"release/{week}/processed/requests.parquet"
    entries = [entry for entry in manifest["files"] if entry.get("path") == relative]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one source manifest entry for {relative}")
    entry = entries[0]
    path = root / "upstream" / relative
    expected_hash = entry.get("lfs_sha256")
    if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("Selected parquet must have a preserved LFS SHA256")
    if not path.is_file() or path.stat().st_size != entry.get("size"):
        raise ValueError(f"Source file missing or size mismatch: {path}")
    digest = file_sha256(path)
    if digest != expected_hash:
        raise ValueError(f"Source SHA256 mismatch: {path}")
    return path, {"revision": REVISION, "url": DATASET_URL,
                  "relative_path": relative, "sha256": digest,
                  "size": path.stat().st_size, "license": "CC-BY-4.0"}


def resolve_source_weeks(root: Path, options: dict) -> list[str]:
    """Resolve all to the pinned snapshot weeks covering the entire interval."""
    if options["week"] != "all":
        return [options["week"]]
    manifest = json.loads((root / "upstream_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("revision") != REVISION:
        raise ValueError("Source metadata does not match the adapter's pinned dataset revision")
    weeks = []
    for entry in manifest["files"]:
        match = re.fullmatch(r"release/(\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2})/processed/requests.parquet",
                             entry.get("path", ""))
        if match:
            week = match.group(1)
            begin, end = week_bounds(week)
            if begin < options["score_end_ms"] and end > options["replay_start_ms"]:
                weeks.append(week)
    weeks.sort()
    covered = options["replay_start_ms"]
    for index, week in enumerate(weeks):
        begin, end = week_bounds(week)
        if begin > covered or (index and begin != covered):
            raise ValueError("Selected interval contains missing, overlapping or duplicate source weeks")
        covered = end
    if not weeks or covered < options["score_end_ms"]:
        raise ValueError("Selected interval extends outside the available source weeks")
    return weeks


def combine_sources(sources: list[dict]) -> dict:
    """Keep single-source SHA semantics; name the multi-source digest explicitly."""
    if len(sources) == 1:
        return {**sources[0], "files": sources, "sha256_semantics": "source_file_bytes"}
    identities = [{key: source[key] for key in ("relative_path", "sha256", "size")}
                  for source in sources]
    return {"revision": REVISION, "url": DATASET_URL, "license": "CC-BY-4.0",
            "relative_path": None, "size": sum(source["size"] for source in sources),
            "sha256": hashlib.sha256(json_bytes(identities)).hexdigest(),
            "sha256_semantics": "sha256 of canonical JSON ordered file identities (relative_path, sha256, size); not a file hash",
            "files": sources}


def metadata_rows(parquet):
    offset = 0
    for batch in parquet.iter_batches(columns=list(METADATA_COLUMNS), batch_size=2048, use_threads=False):
        for index, row in enumerate(batch.to_pylist()):
            yield offset + index, row
        offset += batch.num_rows


def validate_blocks(row: dict, blocks) -> None:
    if not isinstance(blocks, (list, tuple, array.array)):
        raise ValueError(f"row {row['source_row_index']}: block_ids must be a list")
    if not TRACE_BLOCK_SIZE * len(blocks) <= row["input_length"] < TRACE_BLOCK_SIZE * (len(blocks) + 1):
        raise ValueError(f"row {row['source_row_index']}: role token sum is incompatible with the 16-token blocks")
    if any(type(value) is not int or not 0 <= value < 2**64 for value in blocks):
        raise ValueError(f"row {row['source_row_index']}: block IDs must be unsigned 64-bit integers")


class BlockInterner:
    """Collision-free, deterministic IDs within one explicitly namespaced bundle."""

    def __init__(self, first_id: int = 1):
        integer(first_id, "first block ID")
        self.ids = {}
        self.next_id = first_id

    def allocate(self) -> int:
        if self.next_id >= 2**64:
            raise ValueError("Too many unique block identities for Mooncake u64 IDs")
        result = self.next_id
        self.next_id += 1
        return result

    def known(self, week: str, model: str, block: int) -> int:
        key = (week, model, block)
        if key not in self.ids:
            self.ids[key] = self.allocate()
        return self.ids[key]

    def map_blocks(self, week: str, model: str, blocks) -> list[int]:
        return [self.known(week, model, block) for block in blocks]


class DiskBlockInterner(BlockInterner):
    """First-appearance IDs with a disk index and a strictly bounded hot cache.

    Both raw and mapped IDs are stored as unsigned, big-endian 8-byte blobs:
    SQLite INTEGER cannot represent the upper half of the u64 domain. The
    per-request dictionary is bounded by context_limit / TRACE_BLOCK_SIZE.
    """

    def __init__(self, db: sqlite3.Connection, cache_entries: int = 32768, first_id: int = 1):
        integer(cache_entries, "interner cache entries")
        integer(first_id, "first block ID")
        self.db, self.cache_entries = db, cache_entries
        self.cache = OrderedDict()
        self.next_id = first_id
        db.execute("CREATE TABLE interned (week TEXT, model TEXT, raw BLOB, mapped BLOB NOT NULL, "
                   "PRIMARY KEY (week, model, raw)) WITHOUT ROWID")

    def known(self, week: str, model: str, block: int) -> int:
        return self.map_blocks(week, model, [block])[0]

    def map_blocks(self, week: str, model: str, blocks) -> list[int]:
        resolved, missing = {}, []
        for block in dict.fromkeys(blocks):
            key = (week, model, block)
            if key in self.cache:
                resolved[block] = self.cache[key]
                self.cache.move_to_end(key)
            else:
                missing.append(block)
        # Stay below even older SQLite builds' 999-parameter limit.
        for offset in range(0, len(missing), 500):
            chunk = missing[offset:offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            cursor = self.db.execute(
                f"SELECT raw, mapped FROM interned WHERE week=? AND model=? AND raw IN ({placeholders})",
                (week, model, *(block.to_bytes(8, "big") for block in chunk)))
            for raw, mapped in cursor:
                resolved[int.from_bytes(raw, "big")] = int.from_bytes(mapped, "big")
        inserts = []
        for block in missing:
            if block not in resolved:
                resolved[block] = self.allocate()
                inserts.append((week, model, block.to_bytes(8, "big"), resolved[block].to_bytes(8, "big")))
        self.db.executemany("INSERT INTO interned VALUES (?, ?, ?, ?)", inserts)
        for block, mapped in resolved.items():
            key = (week, model, block)
            self.cache[key] = mapped
            self.cache.move_to_end(key)
            if len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)
        return [resolved[block] for block in blocks]


class DiskPreparation:
    """Bounded-batch selection, external time sorting and prefix-block spooling.

    The database is disposable staging state, not a restart checkpoint. Its
    cache is capped at 64 MiB; temporary sorting is allowed to spill to disk.
    Arrow may retain decompression pages in addition to each requested batch.
    """

    def __init__(self, path: Path, options: dict, *, progress: bool = False):
        self.options = options
        self.progress = progress
        self.added_weeks = set()
        self.accumulator = SelectionAccumulator(options)
        self.db = sqlite3.connect(path)
        for pragma in ("journal_mode=OFF", "synchronous=OFF", "cache_size=-65536",
                       "temp_store=FILE", "mmap_size=0"):
            self.db.execute(f"PRAGMA {pragma}")
        self.db.execute("CREATE TABLE selected (week TEXT, row_index INTEGER, start_ms INTEGER NOT NULL, "
                        "input_length INTEGER NOT NULL, metadata TEXT NOT NULL, "
                        "PRIMARY KEY (week, row_index)) WITHOUT ROWID")
        self.db.execute("CREATE INDEX chronological ON selected (start_ms, week, row_index)")
        self.db.execute("CREATE TABLE blocks (week TEXT, row_index INTEGER, payload BLOB NOT NULL, "
                        "PRIMARY KEY (week, row_index)) WITHOUT ROWID")
        # Legacy Mooncake re-interns IDs to zero-based values in its loader.
        # The spooled loader skips that map, so pre-intern from zero for parity.
        self.interner = DiskBlockInterner(self.db, first_id=0 if options.get("format") == "mooncake-spooled" else 1)

    def add_metadata(self, week: str, indexed_rows) -> None:
        if week in self.added_weeks:
            raise ValueError(f"Source week already scanned: {week}")
        self.added_weeks.add(week)
        # A parquet source must yield physical rows in increasing order. This
        # detects accidental duplicate/out-of-order readers without a giant set.
        previous = -1
        for row_index, row in indexed_rows:
            integer(row_index, "source row index")
            if row_index <= previous:
                raise ValueError("Metadata reader must yield strictly increasing physical row indices")
            previous = row_index
            selected = self.accumulator.consume(row_index, {**row, "source_week": week})
            if selected is not None:
                self.db.execute("INSERT INTO selected VALUES (?, ?, ?, ?, ?)",
                                (week, row_index, selected["start_ms"], selected["input_length"],
                                 json.dumps(selected, ensure_ascii=False, separators=(",", ":"))))
            if row_index % 2048 == 2047:
                self.db.commit()
            if self.progress and row_index % 100_000 == 99_999:
                print(f"Metadata {week}: {row_index + 1:,} source rows scanned", flush=True)
        self.db.commit()

    def add_blocks(self, week: str, row_index: int, blocks) -> None:
        found = self.db.execute("SELECT input_length FROM selected WHERE week=? AND row_index=?",
                                (week, row_index)).fetchone()
        if found is None:
            raise ValueError(f"Blocks refer to an unselected source row: {week}/{row_index}")
        validate_blocks({"source_row_index": row_index, "input_length": found[0]}, blocks)
        self.db.execute("INSERT INTO blocks VALUES (?, ?, ?)",
                        (week, row_index, array.array("Q", blocks).tobytes()))

    def spool_parquet_blocks(self, week: str, parquet) -> None:
        group_offset = 0
        next_progress = 100_000
        for group in range(parquet.metadata.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            exists = self.db.execute(
                "SELECT 1 FROM selected WHERE week=? AND row_index>=? AND row_index<? LIMIT 1",
                (week, group_offset, group_offset + count)).fetchone()
            if exists:
                offset = group_offset
                for batch in parquet.iter_batches(row_groups=[group], columns=["block_ids"],
                                                   batch_size=64, use_threads=False):
                    # At most one batch of scalar row indices is materialized.
                    wanted = self.db.execute(
                        "SELECT row_index FROM selected WHERE week=? AND row_index>=? AND row_index<?",
                        (week, offset, offset + batch.num_rows)).fetchall()
                    column = batch.column(0)
                    for (row_index,) in wanted:
                        self.add_blocks(week, row_index, column[row_index - offset].as_py())
                    offset += batch.num_rows
                    if self.progress and offset >= next_progress:
                        print(f"Prefix blocks {week}: {offset:,} source rows scanned", flush=True)
                        next_progress = (offset // 100_000 + 1) * 100_000
                self.db.commit()
            group_offset += count

    def rows(self):
        missing = self.db.execute(
            "SELECT s.week, s.row_index FROM selected AS s LEFT JOIN blocks AS b "
            "ON s.week=b.week AND s.row_index=b.row_index WHERE b.row_index IS NULL LIMIT 1").fetchone()
        if missing:
            raise ValueError(f"Missing blocks for selected source row {missing}")
        cursor = self.db.execute(
            "SELECT s.metadata, b.payload FROM selected AS s INDEXED BY chronological JOIN blocks AS b "
            "ON s.week=b.week AND s.row_index=b.row_index ORDER BY s.start_ms, s.week, s.row_index")
        for metadata, payload in cursor:
            row = json.loads(metadata)
            values = array.array("Q")
            values.frombytes(payload)
            row["_blocks"] = values
            yield row

    def close(self):
        self.db.close()


def converted_rows(selected, blocks_for_row, options: dict, *, interner=None):
    interner = interner if interner is not None else BlockInterner(
        first_id=0 if options.get("format") == "mooncake-spooled" else 1)
    for output_row, row in enumerate(selected):
        blocks = blocks_for_row(row["source_row_index"]) if blocks_for_row is not None else row["_blocks"]
        validate_blocks(row, blocks)
        week = row.get("source_week", options["week"])
        hashes = interner.map_blocks(week, row["model"], blocks)
        tail = row["input_length"] - TRACE_BLOCK_SIZE * len(blocks)
        if tail:
            hashes.append(interner.allocate())
        request = {"timestamp": row["start_ms"] - options["replay_start_ms"],
                   "input_length": row["input_length"], "output_length": row["output_length"],
                   "hash_ids": hashes}
        provenance = {
            "output_row": output_row, "source_row_index": row["source_row_index"],
            "week": week, "session_id": row["session_id"], "step": row["step"],
            "source_model": row["model"], "provider": row.get("provider"),
            "start_ms": row["start_ms"], "end_ms": row.get("end_ms"),
            "timestamp": request["timestamp"], "phase": row["phase"],
            "provider_tokens": {"prompt": row.get("prompt_tokens"), "completion": row["completion_tokens"],
                                "cached": row.get("cache_read_tokens")},
            "role_tokens": row["role_tokens"], "finish_reason": row.get("finish_reason"),
            "input_length": row["input_length"], "output_length": row["output_length"],
            "known_prefix_tokens": TRACE_BLOCK_SIZE * len(blocks), "unknown_tail_tokens": tail,
        }
        yield request, provenance


def attribution_text() -> str:
    return f"""# FreeInference trace attribution

Source: [{DATASET_URL}]({DATASET_URL})

Pinned dataset revision: `{REVISION}`.

Dataset: FreeInference Agentic Trace, accompanying *From Requests to Sessions:
A Large-Scale Characterization of Human-Driven Agentic Workloads* (2026).
Authors: William Nixon, Muxin Tian, Yunjia Zheng, Haryadi S. Gunawi, Juncheng Yang.

License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
No endorsement by the original authors is implied.

Changes: selected the declared original model(s) and continuous UTC interval
across the declared source week(s), plus explicit warmup; chronologically
ordered requests; excluded zero-completion events and,
only when explicitly configured, over-context requests. Input length uses the
released o200k_base role-token sum; output length retains provider token counts.
Known 16-token prefix blocks were remapped within a week/model namespace, with
request-private IDs for unknown partial tails. Timestamps were shifted by the
declared replay origin. Original session identity is retained in a sidecar;
the simulator input represents open-loop independent requests, not agent feedback.
Provider cache hits and serving latencies are not injected as simulated costs.
All selected source-model traffic is intended for the one target model named in
the manifest: a counterfactual workload mapping, not reconstructed original
model routing, tokenizer behavior, hardware, capacity or model switching.

See `manifest.json` for exact selection, exclusions, hashes and source provenance.
This derivative does not restore target-model tokenization or original deployment.
"""


def write_bundle(stage: Path, selected, blocks_for_row, selection: dict,
                 source: dict, recipe: dict, *, interner=None, progress: bool = False) -> dict:
    stage.mkdir()
    request_count = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "hash_ids": 0}
    trace_format = selection.get("format", "mooncake")
    with (stage / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as requests, \
            (stage / "source_requests.jsonl").open("w", encoding="utf-8", newline="\n") as sidecar:
        for request, provenance in converted_rows(selected, blocks_for_row, selection, interner=interner):
            if trace_format == "mooncake-spooled" and any(value >= 2**32 for value in request["hash_ids"]):
                raise ValueError("mooncake-spooled requires globally interned u32 hash IDs; bundle has too many identities")
            requests.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            sidecar.write(json.dumps(provenance, ensure_ascii=False, separators=(",", ":")) + "\n")
            request_count += 1
            totals["input_tokens"] += request["input_length"]
            totals["output_tokens"] += request["output_length"]
            totals["hash_ids"] += len(request["hash_ids"])
            if progress and request_count % 10_000 == 0:
                print(f"Converted {request_count:,}/{selection['counts']['emitted_requests']:,} requests; "
                      f"{totals['hash_ids']:,} prefix IDs written", flush=True)
    if request_count != selection["counts"]["emitted_requests"]:
        raise ValueError("Converted request count does not match the selected cohort; nothing published")
    (stage / "ATTRIBUTION.md").write_text(attribution_text(), encoding="utf-8", newline="\n")
    files = {name: {"sha256": file_sha256(stage / name), "size": (stage / name).stat().st_size}
             for name in ("requests.jsonl", "source_requests.jsonl", "ATTRIBUTION.md")}
    manifest = {"schema_version": 1, "format": trace_format, "trace_block_size": TRACE_BLOCK_SIZE,
                "request_count": request_count, "context_limit": selection["context_limit"],
                "payload_totals": totals,
                "hash_id_encoding": {
                    "order": "first appearance in chronological output, including private tails",
                    "namespace": "source_week,source_model,chained_block_id",
                    "first_id": 0 if trace_format == "mooncake-spooled" else 1,
                    "max_bits": 32 if trace_format == "mooncake-spooled" else 64,
                },
                "resource_estimates": {
                    "eager_input_token_bytes_u32": 4 * totals["input_tokens"],
                    "compact_native_spool_payload_bytes": 4 * (totals["hash_ids"] + totals["output_tokens"]),
                    "semantics": "Payload byte lower bounds only; excludes record headers, metadata, SQLite staging, JSONL and result files.",
                },
                "model_mapping": {
                    "mode": "single_target_counterfactual",
                    "target_model": selection.get("target_model", DEFAULT_TARGET_MODEL),
                    "source_model_role": "provenance_and_hash_namespace",
                    "semantics": "All selected source models use one target model; no model switching or original deployment reconstruction.",
                    "target_tokenization_exact": False,
                },
                "source": source, "selection": selection, "files": files, "recipe": recipe}
    (stage / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def publish_bundle(stage: Path, target: Path) -> bool:
    """Publish a complete directory atomically; never overwrite divergent output."""
    if not target.exists():
        try:
            stage.rename(target)
            return True
        except FileExistsError:
            pass
    if not target.is_dir():
        raise ValueError(f"Existing output is not a directory: {target}")
    for source in stage.iterdir():
        existing = target / source.name
        if not existing.is_file() or existing.stat().st_size != source.stat().st_size \
                or file_sha256(existing) != file_sha256(source):
            raise ValueError(f"Existing derived output differs; nothing overwritten: {existing}")
    return False


def prepare(root: Path, options: dict) -> Path:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Conversion requires pyarrow; replay of the resulting JSONL bundle does not") from exc
    root = root.expanduser().resolve()
    weeks = resolve_source_weeks(root, options)
    options = {**options, "weeks": weeks}
    sources, paths = [], []
    for week in weeks:
        path, item = verify_snapshot_source(root, week)
        sources.append(item)
        paths.append(path)
        print(f"Verified {week} source SHA256: {item['sha256']}", flush=True)
    source = combine_sources(sources)
    recipe = {"recipe_version": 2, "source_revision": REVISION,
              "source_sha256": source["sha256"], "converter_sha256": file_sha256(Path(__file__)),
              **options, "zero_output_policy": "exclude", "hash_namespace": "week,source_model,chained_block_id",
              "token_coordinates": "o200k input; provider completion output",
              "model_mapping": "single_target_counterfactual"}
    recipe_id = hashlib.sha256(json_bytes(recipe)).hexdigest()[:20]
    parent = root / "derived" / "dynosim-openloop-v1"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / recipe_id
    # Selected metadata, sorted traversal, prefix lists and the global hash
    # namespace all live on disk. Only bounded batches and one request's blocks
    # are decoded at once; no token-ID sequence is expanded by this converter.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=parent) as temporary:
        temporary_path = Path(temporary).resolve()
        if not temporary_path.is_relative_to(parent.resolve()):
            raise ValueError("Temporary workspace escaped the intended derivative directory")
        spool = DiskPreparation(temporary_path / "preparation.sqlite", options, progress=True)
        try:
            for week, path in zip(weeks, paths):
                with pq.ParquetFile(path) as parquet:
                    spool.add_metadata(week, metadata_rows(parquet))
                print(f"Scanned metadata: {week}", flush=True)
            selection = spool.accumulator.finish()
            print(json.dumps(selection["counts"], ensure_ascii=False), flush=True)
            for week, path in zip(weeks, paths):
                with pq.ParquetFile(path) as parquet:
                    spool.spool_parquet_blocks(week, parquet)
                print(f"Spooled prefix blocks: {week}", flush=True)
            stage = temporary_path / "bundle"
            write_bundle(stage, spool.rows(), None, selection, source, recipe,
                         interner=spool.interner, progress=True)
        finally:
            spool.close()
        created = publish_bundle(stage, target)
    print(f"{'Created' if created else 'Reused identical'} bundle: {target}", flush=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = "/home/xieqiang/project/data/freeinference_agentic_trace"
    parser.add_argument("--root", type=Path, default=default_root, help="Shared FreeInference dataset directory")
    parser.add_argument("--week", required=True, help="YYYY-MM-DD_YYYY-MM-DD or all (weeks covering the requested interval)")
    parser.add_argument("--source-model", required=True, help="Exact original model label or all; kept as provenance, not model routing")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL,
                        help="Declared single-model counterfactual target; replay config must use the same model")
    parser.add_argument("--format", choices=("mooncake", "mooncake-spooled"), default="mooncake",
                        help="mooncake-spooled requires the patched native disk-backed loader")
    parser.add_argument("--start-utc", required=True, help="Inclusive scoring start with explicit timezone")
    parser.add_argument("--end-utc", required=True, help="Exclusive scoring end with explicit timezone")
    parser.add_argument("--warmup-seconds", type=int, default=0)
    parser.add_argument("--context-limit", type=int, default=131072)
    parser.add_argument("--context-policy", choices=("error", "filter"), default="error")
    args = parser.parse_args()
    try:
        options = make_options(week=args.week, source_model=args.source_model,
                               start_utc=args.start_utc, end_utc=args.end_utc,
                               warmup_seconds=args.warmup_seconds, context_limit=args.context_limit,
                               context_policy=args.context_policy, target_model=args.target_model,
                               trace_format=args.format)
        prepare(args.root, options)
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
