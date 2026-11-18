"""Wait for the verified-interleave bundle, build overlays, then replay them."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from run_experiment_suite import validate_bundle

SCRIPTS = Path(__file__).resolve().parent
BUNDLE = Path.home() / (
    "project/data/swe_chat/derived/verified_interleave_mooncake_v1/"
    "claude-code-main-glm52-128k"
)
OUTPUT = Path.home() / "project/results/swe-chat-glm52-verified-interleave"
FACTORS = (8, 16, 32)


def converter_running(bundle):
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = cmdline.read_bytes().decode(errors="surrogateescape").split("\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if not any(Path(arg).name == "prepare_swe_chat.py" for arg in args):
            continue
        for index, arg in enumerate(args[:-1]):
            if arg == "--output-dir" and Path(args[index + 1]).resolve() == bundle:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--plan", action="store_true", help="Preview cases without waiting")
    args = parser.parse_args()
    bundle = args.bundle.expanduser().resolve()
    factors = [str(factor) for factor in FACTORS]
    command = [
        sys.executable, "-u", str(SCRIPTS / "run_experiment_suite.py"),
        "--bundle", str(bundle), "--output-root", str(args.output_root.expanduser()),
        "--suites", "baseline", "overlay", "bandwidth", "--factors", *factors,
        "--read-bandwidths", "16", "32", "48", "64",
        "--jobs", "1", "--memory-limit-gb", "21",
    ]
    if args.plan:
        return subprocess.call([*command, "--plan"])

    manifest = bundle / "manifest.json"
    print(f"Waiting for {manifest}", flush=True)
    while not manifest.is_file():
        if not converter_running(bundle) and not manifest.is_file():
            parser.error(f"Converter has stopped without publishing {manifest}")
        time.sleep(20)

    try:
        count = validate_bundle(bundle, bundle)
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        missing = []
        for factor in FACTORS:
            target = bundle.with_name(f"{bundle.name}-overlay-{factor}x")
            if not target.exists():
                missing.append(str(factor))
                continue
            if validate_bundle(target, bundle) != count:
                raise ValueError(f"Unexpected request count: {target}")
            overlay = json.loads((target / "manifest.json").read_text())["overlay"]
            if overlay["parent_manifest_sha256"] != digest:
                raise ValueError(f"Overlay was generated from another manifest: {target}")
            print(f"Using verified overlay {target.name}", flush=True)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.error(f"Bundle verification failed: {error}")
    print(f"Base bundle ready: {count:,} requests", flush=True)

    if missing:
        subprocess.run([
            sys.executable, "-u", str(SCRIPTS / "prepare_swe_chat_overlay.py"),
            "--bundle", str(bundle), "--factors", *missing,
        ], check=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
