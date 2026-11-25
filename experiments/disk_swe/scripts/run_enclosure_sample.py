"""Run compact scene-2 data using the formal H200 GLM-5.2 TP8/EP8 configuration.

The historical filename is retained for existing commands. The default now runs
the complete compact 8x bundle; --max-requests selects a chronological sample.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path("/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x")
DEFAULT_CONFIG = BASE / "configs/swe_enclosure_h200_ep8_p256gb_8x.yaml"


def validate_compact_bundle(bundle):
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("manifest.json missing: wait for data generation to finish")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != "mooncake"
            or manifest.get("hash_ids_scope") != "input_output"
            or manifest.get("hash_block_size") != 16
            or manifest.get("trace_block_size") != 16):
        raise ValueError("expected compact input_output hashes with 16-token hash blocks")
    if manifest.get("includes_input_token_ids") or manifest.get("includes_output_token_ids"):
        raise ValueError("compact manifest must not declare token ID arrays")
    if not isinstance(manifest.get("request_count"), int) or manifest["request_count"] <= 0:
        raise ValueError("compact bundle must contain requests")
    trace = bundle / "requests.jsonl"
    expected = manifest.get("files", {}).get("requests.jsonl")
    if not expected or not trace.is_file() or trace.stat().st_size != expected["size"]:
        raise ValueError("requests.jsonl missing or its size differs from manifest")
    # Fast format check only. The common runner verifies the complete file SHA256,
    # and the native loader validates every row, avoiding an extra full JSON scan.
    with trace.open() as stream:
        row = json.loads(stream.readline())
    total = row["input_length"] + row["output_length"]
    hashes = row.get("hash_ids")
    if (row.get("hash_ids_scope") != "input_output"
            or row.get("hash_block_size") != 16
            or not isinstance(hashes, list) or len(hashes) != (total + 15) // 16
            or any(type(value) is not int or not 0 <= value < 2**64 for value in hashes)
            or "input_token_ids" in row or "output_token_ids" in row):
        raise ValueError("first row is not a valid compact input_output request")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=Path.home() / "project/results/swe-chat-enclosure-h200-ep8-compact")
    parser.add_argument("--max-requests", type=int, default=0, help="0 runs all requests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--capture-scheduler-trace", action="store_true")
    args = parser.parse_args(argv)
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    bundle = args.bundle.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not config.is_file():
        parser.error(f"config missing: {config}")
    try:
        manifest = validate_compact_bundle(bundle)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    factor = manifest.get("overlay", {}).get("factor", 1)
    print(f"Compact bundle: {bundle}; requests={manifest['request_count']:,}; overlay={factor}x; speedup=1", flush=True)
    print(f"Configuration: {config}", flush=True)
    command = [
        sys.executable, str(BASE.parent / "disk/scripts/run_experiment.py"),
        "--config", str(config), "--bundle", str(bundle),
        "--output-root", str(args.output_root.expanduser()),
        "--max-requests", str(args.max_requests),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.capture_scheduler_trace:
        command.append("--capture-scheduler-trace")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
