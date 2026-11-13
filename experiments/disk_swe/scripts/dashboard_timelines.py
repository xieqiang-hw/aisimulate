"""Prepare bounded dashboard timelines from completed scheduler trace files."""
from pathlib import Path
import json
import base64
import gzip
import os
import re
import subprocess

BASE = Path(__file__).resolve().parents[1]
CACHE = BASE / "analysis/scheduler_timelines/assets"
TOOL = Path.home() / "build/trace-summary/release/aisim-trace-summary"


def latest_sweep():
    root = Path.home() / "project/results/swe-chat-glm52-1p1d"
    return max((p for p in root.iterdir() if p.is_dir() and (p / "sweep.json").exists()), key=lambda p: p.name)


def case_info(case):
    name = case["name"]
    gb = int(re.search(r"_p(\d+)gb", name)[1])
    factor = re.search(r"_overlay_(\d+)x", name)
    blocks = re.search(r"_hbm(\d+)", name)
    bandwidth = re.search(r"_read(\d+)", name)
    factor = int(factor[1]) if factor else 1
    hbm = int(blocks[1]) if blocks else "自动"
    bw = int(bandwidth[1]) if bandwidth else 8
    return {"id": name, "suite": case["suite"], "status": case["status"],
            "dram": gb, "factor": factor, "hbm": hbm, "bandwidth": bw,
            "label": f"{factor}× · DRAM {gb} GB · HBM {hbm} · SSD 读 {bw} GB/s"}


def report_dir(sweep, case):
    paths = sorted((sweep / case["suite"] / case["name"]).glob("*/run.json"))
    return paths[-1].parent / "prediction" if paths else None


def signature(directory):
    return [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(directory.glob("*"))
            if p.name.startswith(("iteration_metrics", "scheduler_states", "scheduling_events", "scheduler_trace_metadata"))]


def cache_path(sweep, case):
    return CACHE / sweep.name / (case["name"] + ".json")


def prepare_case(sweep, case):
    directory = report_dir(sweep, case)
    if case["status"] != "completed" or directory is None:
        return False
    path = cache_path(sweep, case)
    expected = json.loads(json.dumps(signature(directory)))
    if path.exists() and (cached := json.loads(path.read_text())).get("source_signature") == expected and cached.get("export_version") == 3:
        return False
    if not TOOL.exists():
        raise FileNotFoundError(f"Build trace_summary_tool first: {TOOL}")
    path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    command = ["nice", "-n", "10", str(TOOL), str(directory), str(path)]
    subprocess.run(command, env=env, check=True)
    data = json.loads(path.read_text())
    data.update(source_directory=str(directory), source_signature=expected, case=case_info(case), export_version=3)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)
    return True


def payload(sweep):
    plan = json.loads((sweep / "sweep.json").read_text())
    catalog, traces = [], {}
    for case in plan["cases"]:
        info = case_info(case)
        path = cache_path(sweep, case)
        cached = json.loads(path.read_text()) if path.exists() else None
        info["timeline_ready"] = bool(cached and case["status"] == "completed" and cached.get("export_version") == 3)
        info["full_bs_ready"] = bool(info["timeline_ready"] and cached.get("full_bs", {}).get("url") and (BASE / cached["full_bs"]["url"]).exists())
        if info["timeline_ready"]:
            traces[case["name"]] = cached
        catalog.append(info)
    return {"sweep": str(sweep), "catalog": catalog, "traces": traces}


def prepare_bs_pack(sweep, case):
    path = cache_path(sweep, case)
    if case["status"] != "completed" or not path.exists():
        return False
    data = json.loads(path.read_text())
    if data.get("export_version") != 3:
        return False
    asset = path.with_suffix(".bs.js")
    existing = data.get("full_bs", {})
    if existing.get("version") == 1 and asset.exists() and "b64" not in existing:
        return False
    if existing.get("version") == 1 and "b64" in existing:
        pack = existing
    else:
        binary = path.with_name(path.stem + ".bs.pack")
        result = subprocess.run(["nice", "-n", "10", str(TOOL), "--segments",
                                 data["source_directory"], str(binary)], check=True, capture_output=True, text=True)
        print(result.stdout.strip(), flush=True)
        raw = binary.read_bytes()
        if raw[:4] != b"BSR1":
            raise ValueError("Invalid BS segment pack")
        compressed = gzip.compress(raw, compresslevel=1, mtime=0)
        counts = json.loads(re.search(r"BS segments (\[[^\]]*\])", result.stdout)[1])
        pack = {
            "version": 1, "encoding": "gzip-varint-microseconds-v1",
            "b64": base64.b64encode(compressed).decode("ascii"),
            "segments": counts, "packed_bytes": len(raw), "compressed_bytes": len(compressed),
        }
        binary.unlink()
    temp_asset = asset.with_suffix(".js.tmp")
    temp_asset.write_text("window.aisimBsPackLoaded(" + json.dumps(case["name"]) + "," +
                          json.dumps(pack, separators=(",", ":")) + ");\n", encoding="utf-8")
    temp_asset.replace(asset)
    data["full_bs"] = {key: value for key, value in pack.items() if key != "b64"}
    data["full_bs"]["url"] = asset.relative_to(BASE).as_posix()
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temp.replace(path)
    return True
