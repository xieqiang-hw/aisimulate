"""Convert exact SWE-chat Mooncake bundles to compact input+output block IDs.

The source is never changed. Each chain ID covers a contiguous 16-token
segment of input_token_ids + output_token_ids. Only complete native KV blocks
are cacheable; partial final segments are retained for length reconstruction.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys
import time


def combined_hash_ids(tokens, block_size=16):
    """SHA-256 chain over little-endian uint32 segments, serialized as uint64 IDs."""
    parent = bytes(32)
    result = []
    for start in range(0, len(tokens), block_size):
        chunk = array("I", tokens[start:start + block_size])
        if chunk.itemsize != 4:
            raise RuntimeError("uint32 array is required")
        if sys.byteorder != "little":
            chunk.byteswap()
        parent = hashlib.sha256(parent + chunk.tobytes()).digest()
        result.append(int.from_bytes(parent[:8], "little"))
    return result


def convert_row(row, block_size=16):
    if row.get("hash_ids_scope") is not None:
        raise ValueError("source must contain exact token arrays, not compact hashes")
    for key, length in (("input_token_ids", "input_length"), ("output_token_ids", "output_length")):
        ids = row.get(key)
        if not isinstance(ids, list) or len(ids) != row[length]:
            raise ValueError(f"{key} must contain exactly {length} tokens")
        if any(type(token) is not int or not 0 <= token <= 0xffffffff for token in ids):
            raise ValueError(f"{key} must contain uint32 integers")
    result = {k: v for k, v in row.items()
              if k not in ("input_token_ids", "output_token_ids", "hash_ids")}
    result["hash_ids"] = combined_hash_ids(row["input_token_ids"] + row["output_token_ids"], block_size)
    result["hash_ids_scope"] = "input_output"
    result["hash_block_size"] = block_size
    return result


def digest(path):
    with path.open("rb") as stream:
        return {"size": path.stat().st_size,
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path.home() / "project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-exact")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=0, help="0 converts the complete bundle")
    args = parser.parse_args()
    source, target = args.bundle.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    if source == target or source in target.parents:
        parser.error("use a separate output directory outside the source bundle")
    if target.exists():
        parser.error("output directory already exists; choose a new path")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("trace_block_size") != 16:
        parser.error("expected 16-token source hashes")
    target.mkdir(parents=True)
    started = time.monotonic()
    count = input_total = output_total = 0
    source_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    with (source / "requests.jsonl").open("rb") as inp, (target / "requests.jsonl").open("wb") as out:
        for raw in inp:
            if not raw.strip():
                raise ValueError("blank JSONL rows are not supported")
            source_digest.update(raw)
            row = convert_row(json.loads(raw))
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
            out.write(encoded)
            output_digest.update(encoded)
            count += 1
            input_total += row["input_length"]
            output_total += row["output_length"]
            if count % 1000 == 0:
                print(f"Converted {count:,} requests; {time.monotonic() - started:.1f}s; source read {inp.tell() / 1e9:.2f} GB", flush=True)
            if args.max_requests and count >= args.max_requests:
                break
    if not count:
        raise ValueError("empty source")
    if not args.max_requests:
        expected = manifest["files"]["requests.jsonl"]
        if source_digest.hexdigest() != expected["sha256"] or (source / "requests.jsonl").stat().st_size != expected["size"]:
            raise ValueError("source size/SHA256 differs from manifest")
        if count != manifest["request_count"]:
            raise ValueError("source request count differs from manifest")
    # Keep the per-request provenance sidecar so the existing session-overlay
    # script can process this bundle without rereading token arrays.
    sidecar = source / "source_requests.jsonl"
    files = {"requests.jsonl": {"size": (target / "requests.jsonl").stat().st_size,
                                "sha256": output_digest.hexdigest()}}
    if sidecar.exists():
        with sidecar.open("rb") as inp, (target / sidecar.name).open("wb") as out:
            for _ in range(count):
                raw = inp.readline()
                if not raw:
                    raise ValueError("source metadata is shorter than requests")
                out.write(raw)
        files[sidecar.name] = digest(target / sidecar.name)
    manifest.update(request_count=count, includes_input_token_ids=False,
                    includes_output_token_ids=False, includes_output_hash_ids=True,
                    hash_ids_scope="input_output", hash_block_size=16,
                    files=files)
    manifest["compact_conversion"] = {
        "source_bundle": str(source), "source_manifest_sha256": digest(source / "manifest.json")["sha256"],
        "hash_algorithm": "sha256-chain-uint32le-truncated64",
        "input_tokens": input_total, "output_tokens": output_total,
        "sample_limit": args.max_requests,
        "semantics": "full-sequence block identity; output blocks become visible only after KV computation",
    }
    tmp = target / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(target / "manifest.json")
    print(f"Created {count:,} compact requests in {target}", flush=True)
    print(f"requests.jsonl: {files['requests.jsonl']['size'] / 1e9:.3f} GB", flush=True)


if __name__ == "__main__":
    main()
