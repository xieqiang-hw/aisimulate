# Chat trace 分层 KV 缓存实验

目录组织参考 `../disk_swe`：`scripts/` 放数据转换和验证入口，`configs/` 放后续实验配置，`analysis/` 放分析结果。已提供数据准备和 GLM-5.2/H200 baseline 配置与运行入口。BurstGPT 已按要求排除。

## 支持的本地数据

默认数据根目录 `/home/xieqiang/project/data`，默认转换 `qwen-chat` 和 `mooncake-chat`。`--dataset all` 转换下表所有文件，每个文件独立成包，不合并时间轴或 hash 命名空间。

| 数据集名称 | 根目录下的文件 | 源块大小 | 时间单位 |
|---|---|---:|---|
| qwen-chat | qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl | 16 token | 秒 |
| qwen-api | qwen-bailian-usagetraces-anon/qwen_traceB_blksz_16.jsonl | 16 token | 秒 |
| qwen-thinking | qwen-bailian-usagetraces-anon/qwen_thinking_blksz_16.jsonl | 16 token | 秒 |
| qwen-coder | qwen-bailian-usagetraces-anon/qwen_coder_blksz_16.jsonl | 16 token | 秒 |
| mooncake-chat | Mooncake-FAST25-release/conversation_trace.jsonl | 512 token | 毫秒 |
| mooncake-agent | Mooncake-FAST25-release/toolagent_trace.jsonl | 512 token | 毫秒 |
| mooncake-synthetic | Mooncake-FAST25-release/synthetic_trace.jsonl | 512 token | 毫秒，合成到达 |

Azure 和 WildChat 当前未在本地数据根目录中发现，因此此脚本不包含未经本地数据验证的适配器。

## 使用

在仓库根目录运行；转换只需要 Python 3.11+ 标准库：

```bash
python3 experiments/disk_chat/scripts/prepare_chat.py

# 转换所有已下载的 Qwen/Mooncake 文件，包括 API、agent 和 synthetic 对照组
python3 experiments/disk_chat/scripts/prepare_chat.py --dataset all

# 少量样本，单独输出，避免覆盖完整数据
python3 experiments/disk_chat/scripts/prepare_chat.py \
  --dataset all --max-requests 100 \
  --output-root /home/xieqiang/project/data/chat/derived/smoke

# Qwen 仅保留 text 请求；从源时间起点后 10 分钟选 30 分钟
python3 experiments/disk_chat/scripts/prepare_chat.py \
  --dataset qwen-chat --types text --start-seconds 600 --duration-seconds 1800 \
  --output-root /home/xieqiang/project/data/chat/derived/text-window

python3 experiments/disk_chat/scripts/validate_chat_bundle.py \
  /home/xieqiang/project/data/chat/derived/mooncake-v1/qwen-chat \
  /home/xieqiang/project/data/chat/derived/mooncake-v1/mooncake-chat

python3 -m unittest discover -s experiments/disk_chat/scripts/tests -v
```

输出默认放在 `/home/xieqiang/project/data/chat/derived/mooncake-v1/<dataset>/`。已有包拒绝覆盖；每包通过临时目录完成后原子发布，失败不会留下看似完整的数据包。`all` 是逐包发布，后续包失败时此前完成的包仍保留。

## AISim 输入协议

与 SWE 一样，每包包括：

- `requests.jsonl`：每行一次调用，只有 `timestamp`、`input_length`、`output_length`、`hash_ids`。时间为相对毫秒，按原始到达时间稳定排序，以第一条保留请求为零点。
- `source_requests.jsonl`：与请求逐行对齐，保留源行号、原始时间、请求类型。Qwen 还保留根会话 ID、父请求 ID、原始轮次。会话关联放在 sidecar，避免静态 loader 将重复 session ID 当作动态多轮输入拒绝。
- `manifest.json`：记录源文件 SHA256、Qwen 本地 checkout revision（若存在）、来源和许可、过滤计数、请求与会话数量、时间跨度、token 统计、目标模型标签和输出校验值。有本地源 LICENSE 时附带 `SOURCE_LICENSE`。

回放配置与 SWE 相同：

```yaml
traffic:
  source:
    type: trace
    format: mooncake
    paths:
      - /home/xieqiang/project/data/chat/derived/mooncake-v1/qwen-chat/requests.jsonl
    block_size: 16
  load:
    type: trace_timestamps
    speedup: 1.0
```

默认 `--target-model zai-org/GLM-5.2-FP8 --context-limit 131072`。MiniMax 实验应另建输出根目录，并设置 `--target-model MiniMaxAI/MiniMax-M2.5`。模型标签用于与 runner 核对；源数据没有明文 token，不会重新分词，所以这是保留源 token 长度分布的目标模型模拟负载。

可使用 `../disk/scripts/run_experiment.py --config <已有实验YAML> --bundle <包目录> --dry-run` 验证现有通用 runner。不要直接使用 `disk_swe/scripts/sweep_common.py` 的 SWE 专用验证器：它要求真实 `output_token_ids`，这些 chat 数据未提供。

## 前缀、时间与过滤口径

Qwen 的单块 hash 转成链式身份，避免不同历史中的同一内容块产生错误的 KV 复用。Mooncake 的每个完整 512-token 源块展开成 32 个虚拟 16-token 块：相同源前缀可共享，两个不同源块之间不推断更细粒度的共同前缀。任一不足源块大小的尾块全部私有化，包括其中的完整 16-token 子块。这是保守复用下界，无法补回源数据未披露的块内相同前缀。

全部数据仅保留源输出长度，不伪造输入、输出 token ID。可以评估 P 侧已观测输入前缀复用；无法精确重建 D 输出写回后下一轮 P 的命中。静态时间回放不会等待前一轮实际模拟完成，也不生成工具依赖或用户思考时间。

默认保留所有 Qwen 请求类型（含 image/search/file），仅按已发布 token 数模拟，未额外建模多模态编码。纯文本实验用 `--types text`，该参数只适用于 Qwen。`mooncake-synthetic` 在清单中明确标记为合成到达。

输入输出总长超过 context limit、输出长度为零的请求会过滤并计数；不截断，不把零输出改成一。缺字段、非法长度、hash 覆盖不足、重复请求 ID、断裂父链和循环父链直接报错。时间窗口为 `[start, start + duration)`；`max-requests` 是按时间排序后的前 N 条合格请求，会在边界截断会话，源会话标识仍保留。各筛选条件可能使原会话只剩部分请求。

转换需要临时 SQLite 磁盘空间，以稳定排序并限制内存使用。每次都扫描完整源文件以校验父链和统计过滤数量，即使指定了较小的 `--max-requests`。

源数据说明：[Qwen-Bailian](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon)、[Mooncake FAST25](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release)。转换代码独立编写；转换产物的源文件身份和许可记录在各包清单中。

## GLM-5.2 / H200 baseline

运行入口 `scripts/run_baselines.py`，模板 `configs/glm52_h200_ep8_baseline.yaml`；七份可直接回放的具体配置在 `configs/baseline/`。模板与具体配置通过以下命令同步生成，已有文件内容不同时拒绝覆盖：

```bash
~/.venvs/aisim/bin/python /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py \
  --write-configs /home/xieqiang/project/aisimulate/experiments/disk_chat/configs/baseline
```

参考 `../disk_swe/glm52_experiments.html`，baseline 参数如下：

| 项目 | 默认值 |
|---|---|
| 模型 / 性能模型 | zai-org/GLM-5.2-FP8；H200 SXM，vLLM 0.24.0，op_level，fallback=deny |
| 部署 / 并行 | 1P1D，P/D 各 8 卡；Attention TP8/DP1，MoE TP1/EP8 |
| 调度 | Chunked Prefill；P/D token 预算各 16,384，序列上限 32/256 |
| HBM | 自动 KV 容量，memory_fraction=0.9，64-token KV 块 |
| P 侧缓存 | DRAM 384 GB + worker_local SSD 约 56 TB |
| 缓存容量口径 | 沿用 SWE 的 GLM DSA：每 TP 分片每 token 60,216 字节；整实例每 KV 块 60,216 × 64 × 8 字节；DRAM 12,455 块，SSD 1,816,378 块 |
| SSD / 内存传输 | SSD 读写各 8 GB/s/TP；H2D/D2H 各 25 GB/s/TP |
| P→D | 50 GB/s/TP，destination_missing；D 仅 HBM |
| SLA | TTFT ≤ 1 秒，单请求平均 TPOT ≤ 30 ms |
| 负载 | 各文件完整原始时间轴，无叠加，speedup=1；静态时间回放 |
| 进程资源 | 默认框架自动管理；可设置 --memory-limit-gb / --reserve-memory-gb |

GB/TB 为十进制，DRAM 与 SSD 的容量指整 TP8 实例。上表与代码容量公式只适用于此模型和并行布局，脚本会拒绝不同身份的模板。目标模型标签不表示重新分词：chat 包仍使用源 token 长度和已观测的输入前缀，且无真实 output_token_ids；不能与 SWE 的 D 输出 token 级复用作完全等价解释。

### 命令

使用已安装 AISim 的 Python 环境（包括 PyYAML 和本项目）；以下命令可在任意目录执行：

```bash
# 只校验数据包、配置并打印计划：默认 7 个数据集，各一组
~/.venvs/aisim/bin/python /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py --plan

# 完成 bundle / YAML / runner 检查，不启动仿真
~/.venvs/aisim/bin/python /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py --dry-run

# 正式全量 baseline：七组依次执行，避免同时占用大量内存
~/.venvs/aisim/bin/python -u /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py

# 只运行两个普通 chat 数据集
~/.venvs/aisim/bin/python -u /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py \
  --datasets qwen-chat mooncake-chat

# 短回放检查；保留边界时间戳相同的全部请求，可能超过指定条数
~/.venvs/aisim/bin/python -u /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py \
  --datasets qwen-chat mooncake-chat --max-requests 10 \
  --output-root /home/xieqiang/project/results/chat-glm52-h200-smoke

# 继续补跑：只跳过输入、配置、请求上限、运行时身份等指纹相同的成功案例
~/.venvs/aisim/bin/python -u /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py \
  --skip-completed-from /home/xieqiang/project/results/chat-glm52-h200-baseline/<timestamp>/sweep.json

# baseline 完成后可扩展为与 SWE 相同的 DRAM × 读带宽扫描，仍不叠加会话
~/.venvs/aisim/bin/python -u /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py \
  --datasets qwen-chat mooncake-chat \
  --p-dram-gb 64 128 256 384 512 1000 --read-bandwidths 1 2 4 8 16 \
  --output-root /home/xieqiang/project/results/chat-glm52-h200-sweep
```

默认全部数据集包括 Qwen API/coder/thinking，以及 Mooncake agent/synthetic 对照组，不把它们统称为普通 chat。纯 chat 对比用 `--datasets qwen-chat mooncake-chat`。默认使用当前已转换数据，包括 Qwen 原文件中的多模态请求；如需只比较 text，可先转换带 `--types text` 的 Qwen 包，再用 `--bundle-root` 指向该根目录。

### 运行产物与恢复

baseline 默认结果根目录 `/home/xieqiang/project/results/chat-glm52-h200-baseline/<UTC timestamp>/`；`--sweep` 默认为 `/home/xieqiang/project/results/chat-glm52-h200-sweep/<UTC timestamp>/`：

```text
configs/<case>.yaml                  本次实际配置快照
sweep.json                          各案例状态、命令、指纹、耗时和子运行路径
cases/<case>/runner.log              包装层日志
cases/<case>/<UTC timestamp>/
  run.json                          通用 runner 的元数据及完成状态
  resolved.yaml                     CLI 实际读取的配置
  run.log                           CLI 执行日志
  prediction/prediction.json         结果汇总
  prediction/requests.jsonl          逐请求指标
```

`sweep.json` 的 `status=completed` 仅表示真实仿真完成且完成请求数等于选中请求数；dry-run 标记 `validated`。失败案例保留日志并继续后续案例，整个脚本最终返回非零；`--fail-fast` 可遇错即停。Ctrl+C 会停止当前 runner 及其子进程并保存中断状态。补跑时使用 `--skip-completed-from`；dry-run、失败、部分完成都不会被当作成功跳过。更改 speedup、容量、带宽、请求上限或运行时后会重新执行。

`--capture-scheduler-trace` 可额外导出 gzip 调度轨迹；`--export-html` 可导出框架自带的逐案例 HTML。默认只抓逐请求指标，减少产物体积。`--plan` 不创建结果目录；`--dry-run`、`--write-configs` 都不会启动性能仿真。

开发验证：

```bash
~/.venvs/aisim/bin/python -m unittest discover \
  -s /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/tests -v
```

## 实验结果记录

- [GLM-5.2 实验记录](glm52_experiments.md)：七组正式全量 baseline 的性能、复用与运行资源。
- [交互 HTML](glm52_experiments.html)：与 SWE 页面共用布局；按数据集切换 DRAM 容量扫描、SSD 读带宽扫描、带宽与容量对照，附分层复用和 SSD 读取量图，可导出 CSV。
- `analysis/model_reports/glm52_rows.json`：机器可读快照与原始结果路径。

更新命令（只读取结果，不启动仿真）：

```bash
~/.venvs/aisim/bin/python /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/refresh_experiments.py
```

默认同时扫描 `~/project/results/chat-glm52-h200-baseline/*/sweep.json` 与 `~/project/results/chat-glm52-h200-sweep/*/sweep.json`，排除 smoke、dry-run 和请求上限非零的运行；每组配置使用最新正式尝试。自定义结果根目录使用 `--results-root`，也可重复传入 `--sweep /absolute/path/sweep.json`。报告是离线快照，新增运行后重新执行刷新命令，再重载 HTML。

## 全数据集 DRAM × SSD 读带宽扫描（4 路并行）

已生成 `configs/sweep/` 下 210 份具体 YAML（7 个数据集 × 6 个 DRAM 档位 × 5 个 SSD 读带宽档位）。DRAM 为 64/128/256/384/512/1000 GB；SSD 读为 1/2/4/8/16 GB/s/TP，写带宽固定 8 GB/s/TP。模型、部署、调度预算、SLA 和原始时间轴 1× 均沿用 baseline。

```bash
# 计划预览，不启动实验
~/.venvs/aisim/bin/python /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py --sweep --jobs 4 --plan

# 正式运行：全部 210 组，最多同时运行 4 组
~/.venvs/aisim/bin/python -u /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/run_baselines.py --sweep --jobs 4

# 扫描完成或运行中刷新 MD/HTML；会同时读取 baseline 与 sweep
~/.venvs/aisim/bin/python /home/xieqiang/project/aisimulate/experiments/disk_chat/scripts/refresh_experiments.py
```

`--jobs` 控制独立实验的并发数，不改变被模拟的 P/D GPU 数。未传 `--jobs` 时仍默认为串行；上述扫描命令明确使用 4。执行器在空出槽位后派发下一组；每组独立配置、日志和结果目录，`sweep.json` 原子更新。运行内存预算仍默认由每个 AISim 实例自动确定，不是共享的四任务总预算；可用 `--memory-limit-gb` 为每组指定上限。

`--fail-fast` 在检测到失败后停止派发新组，已启动组正常结束并回收；Ctrl+C 会终止并回收全部活动组并保存中断状态。失败、待运行和中断案例可通过已有的 `--skip-completed-from /absolute/path/sweep.json` 补跑。自定义档位仍可配合 `--sweep --p-dram-gb ... --read-bandwidths ...` 使用。

HTML 由 `scripts/build_chat_dashboard.py` 直接适配 `../disk_swe/scripts/model_report_dashboard.html`，保留其 CSS、卡片、标签页、Canvas 图表和折叠表格。`scripts/chat_experiments_dashboard.html` 是自动生成的模板副本，不要手工修改；格式调整应修改适配器或共享源模板。数据集筛选作用于所有图表、对照和导出，避免将不同数据集连成同一条曲线。
