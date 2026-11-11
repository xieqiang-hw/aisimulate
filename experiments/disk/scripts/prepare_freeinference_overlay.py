"""Fold chronological FreeInference segments onto one replay timeline."""
import argparse
from contextlib import ExitStack
import hashlib
import heapq
import json
from pathlib import Path
import re
import tempfile

TIMESTAMP = re.compile(rb'"timestamp"\s*:\s*(\d+)')


def overlay(source, output, factor):
    manifest_path = source / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    recipe = manifest["recipe"]
    span = recipe["score_end_offset_ms"]
    if recipe["warmup_seconds"] or recipe["score_start_offset_ms"] or span % factor:
        raise ValueError("Expected a zero-warmup trace with equally divisible duration")
    width = span // factor
    target = output / f"{source.name}-overlay-{factor}x"
    if target.exists():
        raise FileExistsError(target)
    names = ("requests.jsonl", "source_requests.jsonl")
    offsets = [None] * factor
    previous = -1
    count = 0
    with (source / names[0]).open("rb") as trace, (source / names[1]).open("rb") as side:
        while line := side.readline():
            row = json.loads(line)
            t = row["timestamp"]
            if not previous <= t < span or row["output_row"] != count:
                raise ValueError("Invalid source chronology or row ordinal")
            segment = t // width
            if offsets[segment] is None:
                offsets[segment] = (trace.tell(), side.tell() - len(line), count)
            if not trace.readline():
                raise ValueError("Missing request row")
            previous = t
            count += 1
        if trace.read(1) or count != manifest["request_count"]:
            raise ValueError("Source row count mismatch")

    def rows(trace, side, segment, offset):
        trace.seek(offset[0])
        side.seek(offset[1])
        for i, line in enumerate(side, offset[2]):
            row = json.loads(line)
            t = row["timestamp"]
            if t // width != segment:
                break
            raw = trace.readline()
            match = TIMESTAMP.search(raw)
            if not match or int(match[1]) != t or row["output_row"] != i:
                raise ValueError(f"Request/sidecar timestamp mismatch at row {i}")
            new_time = t - segment * width
            raw = raw[:match.start(1)] + str(new_time).encode() + raw[match.end(1):]
            row["parent_output_row"] = i
            row["timestamp"] = new_time
            yield new_time, i, raw, row

    with tempfile.TemporaryDirectory(dir=output, prefix=".overlay-") as temp:
        stage = Path(temp) / "bundle"
        stage.mkdir()
        hashes = {name: hashlib.sha256() for name in names}
        sizes = dict.fromkeys(names, 0)
        with ExitStack() as stack:
            streams = [rows(*(stack.enter_context((source / n).open("rb")) for n in names), seg, off)
                       for seg, off in enumerate(offsets) if off is not None]
            writers = [stack.enter_context((stage / n).open("wb")) for n in names]
            emitted = 0
            for _, _, raw, row in heapq.merge(*streams, key=lambda item: item[:2]):
                row["output_row"] = emitted
                payload = (raw, (json.dumps(row, separators=(",", ":")) + "\n").encode())
                for name, writer, data in zip(names, writers, payload):
                    writer.write(data)
                    hashes[name].update(data)
                    sizes[name] += len(data)
                emitted += 1
        if emitted != count:
            raise ValueError("Overlay row count mismatch")
        attribution = (source / "ATTRIBUTION.md").read_bytes() + (
            f"\nDerived overlay: {factor} equal segments folded onto the first segment; "
            "original source times and hash IDs preserved.\n"
        ).encode()
        (stage / "ATTRIBUTION.md").write_bytes(attribution)
        manifest["files"] = {n: {"sha256": hashes[n].hexdigest(), "size": sizes[n]} for n in names}
        manifest["files"]["ATTRIBUTION.md"] = {
            "sha256": hashlib.sha256(attribution).hexdigest(), "size": len(attribution)
        }
        manifest["overlay"] = dict(factor=factor, segment_ms=width, replay_speedup=1,
                                   parent_bundle=source.name,
                                   parent_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        manifest["hash_id_encoding"]["order"] = "inherited from parent bundle"
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stage.rename(target)
    print(f"Created bundle: {target} ({emitted:,} requests)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--factors", type=int, nargs="+", choices=(2, 4, 5, 6, 8, 16), default=(2, 4, 8))
    args = parser.parse_args()
    source = args.bundle.resolve()
    output = (args.output_root or source.parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for factor in dict.fromkeys(args.factors):
        overlay(source, output, factor)


if __name__ == "__main__":
    main()
