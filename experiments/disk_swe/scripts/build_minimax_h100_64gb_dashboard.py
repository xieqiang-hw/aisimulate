"""Build the MiniMax H100/64GB dashboard using the shared chart layout."""
import argparse
from pathlib import Path
import shutil
import re
import tempfile
import build_results_dashboard as common

BASE = Path(__file__).resolve().parents[1]
DOCUMENT = BASE / "analysis/model_reports/source_documents/experiments_minimax_h100_64gb_8x.md"


def build(markdown_path=DOCUMENT, output=None):
    markdown_path = Path(markdown_path)
    source_archive = BASE / "analysis/model_reports/source_documents"
    if markdown_path.parent == BASE:
        markdown_path = source_archive / markdown_path.name
    if output is not None and Path(output).parent == BASE:
        output = source_archive / Path(output).name
    output = Path(output) if output else markdown_path.with_suffix(".html")
    source_parent = Path.home() / (
        "project/data/swe_chat/derived/verified_interleave_mooncake_v1/"
        "claude-code-main-glm52-128k"
    )
    markdown = markdown_path.read_text()
    factors = set(re.findall(r"^\| (\d+)× \|", markdown, re.M))
    if len(factors) != 1:
        raise ValueError("Expected a single overlay factor")
    factor = int(next(iter(factors)))
    bf16_kv = "| KV 精度 | BF16" in markdown
    gpu_blocks = 11193 if bf16_kv else 22386
    block_label = f"{gpu_blocks:,}"
    precision_label = "BF16" if bf16_kv else "FP8"

    # Shared builder reads the original shape manifest for scale and timestamps.
    shadow = "<!-- report:ep8-local-ssd -->\n" + markdown
    shadow += "\n数据目录 " + chr(96) + str(source_parent) + chr(96) + "\n"
    old_template = common.TEMPLATE
    template = old_template
    changes = {
        'const modes=meta.ep8?{1:"bandwidth"}':
            'const modes=meta.ep8?{5:"bandwidth"}',
        "原始模型标签统一映射到 GLM-5.2-FP8":
            "保留原 GLM tokenizer 的长度与前缀关系，使用 MiniMax 性能模型",
        "GLM-5.2-FP8": f"MiniMax-M2.5 · FP8 计算 / {precision_label} KV",
        "H200 SXM · 1P1D，P/D 各 8 卡，共 16 卡":
            "H100 SXM · 单卡总 HBM 限 64 GB<br>1P1D，P/D 各 8 卡，共 16 卡",
        "自动容量（memory_fraction=0.9）":
            f"自动 {block_label} 块/卡，{precision_label} KV 约 22.74 GB/卡；总 HBM 64 GB，使用率 90%",
        "引擎 KV 块 64 token；回放提供输出长度":
            "引擎 KV 块 64 token；保留原回放输出 token ID",
        '<title>SWE-chat · KV 缓存实验</title>':
            f'<title>MiniMax-M2.5 · H100 64 GB · {factor}× KV 缓存实验</title>',
    }
    for before, after in changes.items():
        template = template.replace(before, after)
    common.TEMPLATE = template
    workload_source = BASE / "analysis/model_reports/source_documents/experiments_h200_ep8_16x.workload.html"
    workload_target = output.with_suffix(".workload.html")
    if workload_source.exists() and not workload_target.exists():
        shutil.copyfile(workload_source, workload_target)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", encoding="utf-8",
                                         dir=Path.home() / "project/tmp", delete=False) as stream:
            stream.write(shadow)
            temporary = Path(stream.name)
        common.build(markdown_path=temporary, output=output)
        rendered = output.read_text()
        rendered = rendered.replace(temporary.name, markdown_path.name)
        rendered = rendered.replace(
            "H200 1P1D 本地缓存实验，比较 P 侧 DRAM 容量与 SSD 读带宽对前缀复用和请求性能的影响。",
            f"MiniMax-M2.5 / H100 64 GB / {factor}×，比较 DRAM 容量与 SSD 读带宽。"
            "沿用原负载形状，H100 计算能力保持不变。"
        )
        rendered = rendered.replace(
            "<h1>SWE-chat 分层 KV 缓存实验</h1>",
            "<h1>MiniMax-M2.5 分层 KV 缓存实验</h1>"
        )
        rendered = rendered.replace(
            "P、D 使用自动 HBM KV 容量。P 侧使用本地 DRAM 和 SSD，D 侧仅使用 HBM。",
            f"P/D 单卡总 HBM 64 GB，自动 {precision_label} KV {block_label} 块/卡。P 使用本地 DRAM 和 SSD，D 仅使用 HBM。"
        )
        # Do not replace configuration names inside the embedded evidence.
        temp = output.with_suffix(".html.tmp")
        temp.write_text(rendered, encoding="utf-8")
        temp.replace(output)
    finally:
        common.TEMPLATE = old_template
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    from refresh_model_reports import refresh
    refresh("minimax")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=DOCUMENT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    build(args.markdown, args.output)
