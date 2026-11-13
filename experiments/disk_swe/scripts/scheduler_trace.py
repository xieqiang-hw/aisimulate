"""Read complete v1/v2 scheduler traces without materializing full timelines."""
import gzip
import json
from pathlib import Path


def trace_path(directory, stem):
    directory = Path(directory)
    for suffix in (".jsonl.gz", ".jsonl"):
        path = directory / (stem + suffix)
        if path.exists():
            return path
    raise FileNotFoundError(f"{stem} trace not found in {directory}")


def rows(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
        for line in stream:
            yield json.loads(line)


def metadata(directory):
    path = Path(directory) / "scheduler_trace_metadata.json"
    return json.loads(path.read_text()) if path.exists() else None


def iter_iterations(directory):
    meta = metadata(directory)
    path = trace_path(directory, "iteration_metrics")
    if not meta:
        yield from rows(path)
        return
    if meta["schema_version"] != 2:
        raise ValueError("Unsupported scheduler trace schema")
    changes = iter(rows(trace_path(directory, "scheduler_states")))
    upcoming = next(changes, None)
    states = {}
    for values in rows(path):
        if len(values) != len(meta["iteration_columns"]):
            raise ValueError("Iteration column count mismatch")
        row = dict(zip(meta["iteration_columns"], values, strict=True))
        sid, iteration = row["stream_id"], row["iteration_id"]
        if upcoming is not None and upcoming[:2] == [sid, iteration]:
            states[sid] = upcoming[3]
            upcoming = next(changes, None)
        if sid not in states:
            raise ValueError("Missing baseline state for scheduler stream")
        yield {
            "schema_version": 2, "gauge_boundary": meta["gauge_boundary"],
            **meta["streams"][sid], **states[sid], **row,
            "duration_ms": row["end_ms"] - row["start_ms"],
            "compute_duration_ms": row["end_ms"] - row["compute_start_ms"],
            "pre_compute_wait_ms": row["compute_start_ms"] - row["start_ms"],
        }
    if upcoming is not None:
        raise ValueError("Unmatched scheduler state transition")


def iter_events(directory):
    meta = metadata(directory)
    for row in rows(trace_path(directory, "scheduling_events")):
        if meta:
            row = {**meta["streams"][row["stream_id"]], **row}
            if row["event"] in ("admission", "readmission"):
                row["is_readmission"] = row["event"] == "readmission"
            if row.get("detail"):
                row["reason"] = row["detail"]["kind"]
        yield row
