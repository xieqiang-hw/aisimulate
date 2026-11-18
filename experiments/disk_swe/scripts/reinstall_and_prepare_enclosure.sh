#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "用法: $0 [--max-sessions N] [--output-dir PATH]"
    echo "默认转换全部符合条件的会话。N=0 表示全量。"
}

max_sessions=0
output_dir=""
while (( $# > 0 )); do
    case "$1" in
        --max-sessions|--output-dir)
            if (( $# < 2 )) || [[ "$2" == --* || -z "$2" ]]; then
                usage >&2
                exit 2
            fi
            if [[ "$1" == --max-sessions ]]; then
                max_sessions="$2"
            else
                output_dir="$2"
            fi
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! "$max_sessions" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "--max-sessions 必须为非负整数" >&2
    exit 2
fi

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
venv="${AISIM_VENV:-$HOME/.venvs/aisim}"
data="${AISIM_DATA_DIR:-$HOME/project/data}"
uv="${UV_BIN:-$HOME/.local/bin/uv}"
index="${AISIM_PYPI_INDEX:-https://repo.huaweicloud.com/repository/pypi/simple}"
bundle_name=claude-code-main-glm52-128k-exact
if [[ "$max_sessions" != 0 ]]; then
    bundle_name="$bundle_name-sample-$max_sessions"
fi
if [[ -z "$output_dir" ]]; then
    output_dir="$data/swe_chat/derived/mooncake-v1/$bundle_name"
fi

for path in "$uv" "$venv/bin/python"; do
    if [[ ! -x "$path" ]]; then
        echo "缺少可执行文件 $path" >&2
        exit 1
    fi
done
for path in "$data/swe_chat/upstream" "$data/GLM_5.2_tokenizer"; do
    if [[ ! -d "$path" ]]; then
        echo "找不到数据目录 $path" >&2
        exit 1
    fi
done
if [[ -e "$output_dir" || -L "$output_dir" ]]; then
    echo "输出路径已存在，请指定新路径 $output_dir" >&2
    exit 1
fi
if [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env"
fi
command -v cargo >/dev/null
"$venv/bin/python" -c 'import transformers, tokenizers'

echo "从当前源码重装 AISim"
CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}" \
    "$uv" pip install --python "$venv/bin/python" --index-url "$index" \
    --reinstall-package aisim --no-deps -e "$repo/python/aisimulate"
"$venv/bin/python" -c 'import aisimulate._runtime as runtime; print("AISim 扩展:", runtime.__file__)'

echo "生成 SWE-chat 数据包 $output_dir"
"$venv/bin/python" -u "$repo/experiments/disk_swe/scripts/prepare_swe_chat.py" \
    --upstream "$data/swe_chat/upstream" --tokenizer "$data/GLM_5.2_tokenizer" \
    --output-dir "$output_dir" --context-limit 131072 \
    --max-sessions "$max_sessions" --include-input-token-ids
