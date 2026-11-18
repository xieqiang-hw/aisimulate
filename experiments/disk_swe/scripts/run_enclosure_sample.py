"""用精确 token 的 SWE-chat 小样本验证共享盘框。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path.home() / "project/data/swe_chat/derived/mooncake-v1/claude-code-main-glm52-128k-exact-sample-5")
    parser.add_argument("--output-root", type=Path, default=Path.home() / "project/results/swe-chat-enclosure-sample")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests 必须为非负整数")
    trace = args.bundle.expanduser() / "requests.jsonl"
    with trace.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            row = json.loads(line)
            for field, length in (("input_token_ids", "input_length"), ("output_token_ids", "output_length")):
                ids = row.get(field)
                if not isinstance(ids, list) or len(ids) != row[length]:
                    parser.error(f"第 {index} 条请求缺少完整的 {field}")
    command = [
        sys.executable, str(BASE.parent / "disk/scripts/run_experiment.py"),
        "--config", str(BASE / "configs/swe_enclosure_sample.yaml"),
        "--bundle", str(args.bundle.expanduser()),
        "--output-root", str(args.output_root.expanduser()),
        "--max-requests", str(args.max_requests),
    ]
    if args.dry_run:
        command.append("--dry-run")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
