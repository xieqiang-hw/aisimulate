"""Run the full 8x H200 EP8 sweep after an existing sweep finishes."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from refresh_h200_ep8_report import process_identity
from run_h200_local_8ssd_ep8_sweep import EXTENSION_DRAM_GB, PARENT, READ_GBPS

SCRIPTS = Path(__file__).resolve().parent
BUNDLE = PARENT.with_name(f"{PARENT.name}-overlay-8x")
TERMINAL = {"completed", "failed", "interrupted"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pid", type=int, required=True)
    parser.add_argument("--after-sweep", type=Path, required=True)
    parser.add_argument("--after-watcher-pid", type=int, required=True)
    parser.add_argument("--include-sweep", type=Path, action="append", default=[])
    args = parser.parse_args()
    previous = args.after_sweep.resolve()
    output = previous.parent
    watcher_identity = process_identity(args.after_watcher_pid)
    runner_identity = process_identity(args.after_pid)
    print(f"Waiting for sweep {previous.name} (PID {args.after_pid})", flush=True)
    while runner_identity is not None and process_identity(args.after_pid) == runner_identity:
        time.sleep(15)
    plan = json.loads((previous / "sweep.json").read_text())
    if any(case["status"] not in TERMINAL for case in plan["cases"]):
        parser.error("Previous sweep exited before all cases reached a terminal state")

    command = [
        sys.executable, "-u", str(SCRIPTS / "run_h200_local_8ssd_ep8_sweep.py"),
        "--bundle", str(BUNDLE), "--output-root", str(output),
        "--p-dram-gb", *(str(gb) for gb in EXTENSION_DRAM_GB),
        "--read-bandwidths", "1", *(str(rate) for rate in READ_GBPS),
    ]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_log = output / f"sweep-8x-{stamp}.log"
    print(f"Starting 8x sweep; log {run_log}", flush=True)
    with run_log.open("xb") as log:
        runner = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    sweep = None
    while sweep is None:
        for line in run_log.read_text().splitlines():
            if line.startswith("Sweep directory: "):
                candidate = Path(line.removeprefix("Sweep directory: "))
                if (candidate / "sweep.json").is_file():
                    sweep = candidate
                break
        if sweep is None:
            if runner.poll() is not None:
                raise RuntimeError(f"8x sweep exited before initialization; see {run_log}")
            time.sleep(2)
    while watcher_identity is not None and process_identity(args.after_watcher_pid) == watcher_identity:
        time.sleep(5)
    refresh = [
        sys.executable, "-u", str(SCRIPTS / "refresh_h200_ep8_report.py"),
        "--sweep", str(sweep), "--watch-pid", str(runner.pid),
    ]
    for older in (*args.include_sweep, previous):
        refresh.extend(("--include-sweep", str(older.resolve())))
    watcher = subprocess.Popen(refresh)
    result = runner.wait()
    report_result = watcher.wait()
    return result or report_result


if __name__ == "__main__":
    raise SystemExit(main())
