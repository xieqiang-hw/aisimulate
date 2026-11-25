"""Generate session-preserving overlays of compact scene-2 Mooncake data."""
import argparse
import json
from pathlib import Path

from prepare_swe_chat_overlay import SUPPORTED_FACTORS, main as prepare_overlay

DEFAULT_ROOT = Path("/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1")
DEFAULT_BUNDLE = DEFAULT_ROOT / "claude-code-main-glm52-128k-compact"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--factors", type=int, nargs="+", choices=SUPPORTED_FACTORS, default=[8])
    args = parser.parse_args(argv)
    bundle = args.bundle.expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        parser.error(f"completed compact manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("hash_ids_scope") != "input_output"
            or manifest.get("hash_block_size") != 16
            or manifest.get("trace_block_size") != 16):
        parser.error("expected a compact input_output hash bundle with 16-token hash blocks")
    if "overlay" in manifest:
        parser.error("generate overlays from the original 1x compact bundle")
    print(f"Source: {bundle}; {manifest['request_count']:,} requests; factors={args.factors}", flush=True)
    command = ["--bundle", str(bundle), "--factors", *map(str, args.factors)]
    if args.output_root:
        command += ["--output-root", str(args.output_root.expanduser())]
    return prepare_overlay(command)


if __name__ == "__main__":
    main()
