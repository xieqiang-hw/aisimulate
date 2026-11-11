# 实验索引

共同配置：GLM-5.2-FP8、H200、1P1D、每实例 TP8、vLLM 0.24.0 算子级模型、128K 上下文。FreeInference 全量 665,415 条，1倍原始到达时间、冷启动、无单独暖机。探索性 SLA：完整输出且 TTFT ≤ 5 秒、每请求平均 ITL ≤ 50 ms。G3 均关闭。

| 实验标识 | 目的与配置 | 运行入口 | 分析目录 |
| --- | --- | --- | --- |
| `hbm_only` | 两侧仅 HBM，建立基线 | `scripts/run_experiment.py --config configs/hbm_only.yaml` | `analysis/hbm_only/` |
| `p_g2_shared` | 仅 P 开 G2，考察前缀卸载收益；D 仅 HBM | 同一单组脚本，配置 `configs/p_g2_shared.yaml` | `analysis/p_g2_shared/` |
| `pd_g2_shared` | P/D 共用一个 G2 池，考察 D 接入的增量影响 | 同一单组脚本，配置 `configs/pd_g2_shared.yaml` | `analysis/pd_g2_shared/` |

`scripts/run_g2_experiments.py` 顺序运行两组 G2 配置。一组失败仍继续另一组，最终退出码非零。G2 两组池总容量都是 8192 块，不因 D 加入而翻倍。当前约 29.40 GiB/TP分片，TP8 分片合计约 235.22 GiB，不含额外开销。每分片搬运链路为32 GB/s、共享池各方向80 GB/s，首字节延迟为0，均为未实测校准的建模假设；共享 G2 不改变直接交接路径，也不保证保存 D 生成的全部 KV。当前设置与已完成的全量实验一致。

## 命令

以下均在仓库根目录执行：

```bash
source ~/.venvs/aisim/bin/activate
cd ~/project/aisimulate
BUNDLE="$HOME/project/data/freeinference_agentic_trace/derived/dynosim-openloop-v1/428f809eef5952012f9c"

# 单组；修改配置名可运行另外两组
python experiments/disk/scripts/run_experiment.py --bundle "$BUNDLE" \
  --config experiments/disk/configs/hbm_only.yaml

# 两组 G2 顺序运行
python experiments/disk/scripts/run_g2_experiments.py --bundle "$BUNDLE"
```

加 `--max-requests 1000` 可先做小规模运行；`--dry-run` 仅校验。`--output-root` 修改结果路径。双组脚本也支持同时覆盖两组的 G2 容量和带宽，详见 `--help`；通常直接维护各 YAML，并保持两组 G2 参数一致。

数据转换入口是 `scripts/prepare_freeinference.py`，产物保存在数据根目录的 `derived/dynosim-openloop-v1/<recipe-id>/`。

## 原始结果与分析产物

- 单组默认：`~/project/results/glm52-1p1d/<UTC时间>/`。
- 双组默认：`~/project/results/glm52-1p1d-g2/<UTC时间>/<实验标识>/<UTC时间>/`。
- 每次运行的 `resolved.yaml`、`run.json`、`run.log` 和 `prediction/` 为原始证据；双组还有 `suite.json`。
- 每个 Notebook 生成 `summary.json`、`metrics.csv`、`comparison.json`、`comparison.csv`、`comparison.png`。读取报告采用流式解析，缓存记录结果路径、文件大小和修改时间。
- 最大单次 token 间隔与 SLA 的每请求平均 ITL 不同；前缀 token 复用比例不等于 DRAM 命中率。当前汇总不能解释长停顿原因，也不能得出 SSD 带宽需求。

长停顿诊断见各组分析README及Notebook末尾单元：包含候选UTC时间、D侧重新准入、同一请求三组对照和分钟级负载。当前证据未保留逐token时间戳，不能精确定位最大间隔起止；详见证据边界。

分析目录内的数据和图统一放在各实验的 `assets/` 子目录；外层仅保留 README 和 Notebook，README 内嵌图表并链接数据。

## 实验4：P 本地 G3 功能验证

配置为 `configs/p_g3_local_validation.yaml`。P 使用本地 G2 与本地 SSD，D 仅 HBM。用 A → B → A 三条请求验证。B 挤出 A 在 G1/G2 中的前缀，G3 开启组再次请求 A 时经 G2 恢复。G2-only 对照要求零复用。两组都检查 P→D 交接和完整输出，并校验 SSD 读写完成及无跨 worker 读取。

```bash
cd ~/project/aisimulate
~/.venvs/aisim/bin/python experiments/disk/scripts/validate_p_g3_local.py
```

成功时在 `~/project/results/glm52-1p1d-g3-validation/` 下生成 `validation.json`。固定计算时延和小缓存用于强制淘汰，实验用于验证功能。

已通过合成功能验证。G2-only 的第二次 A 复用 0 token，G3 组通过 G2 恢复并复用 128 token，两组各完成 3 条请求。

## 实验5：P 本地 G3 的 FreeInference 全量回放

配置 `configs/p_g3_local.yaml`，入口 `scripts/run_p_g3_local.py`。沿用 GLM-5.2-FP8、H200、1P1D、每侧 TP8、原速冷启动，以及 TTFT ≤ 5 秒和平均 ITL ≤ 50 ms 的 SLA。P 使用 8192 块本地 G2 与 16384 块本地 G3，D 仅 HBM，P→D 直接交接。

G2 每方向链路 32 GB/s。G3 读写各 10 GB/s，首字节延迟 0.1 ms。G3 的容量和带宽采用引擎的 TP 分片口径，是首轮仿真假设。原仅 P 开 G2 的实验只有一个 DP rank，其 80 GB/s 池上限高于 32 GB/s 的链路上限。本实验使用 `dp_rank_local` 以接入 G3，保留该 rank 的容量与链路速度。

```bash
cd ~/project/aisimulate
BUNDLE="$HOME/project/data/freeinference_agentic_trace/derived/dynosim-openloop-v1/428f809eef5952012f9c"
~/.venvs/aisim/bin/python experiments/disk/scripts/run_p_g3_local.py --bundle "$BUNDLE"
```

加 `--max-requests 1000` 可先验证前 1000 条。默认结果目录为 `~/project/results/glm52-1p1d-p-g3-local/<UTC时间>/`。查看 `prediction/prediction.json` 的 `g3_offload`，其中包含命中、读写、淘汰及恢复回退统计。这个配置是后续容量和带宽扫描的一个起点。

## 实验6–9：P 本地 SSD 容量扫描

以实验5的 16,384 块作为 1× 对照。四组各有独立 YAML。SSD 读写各 10 GB/s、首字节延迟 0.1 ms，P 侧 G2 为 8192 块，D 侧仅 HBM。数据仍为同一份 FreeInference 16 周全量，665,415 条，冷启动原速回放，SLA 不变。

| 容量倍率 | SSD 块数 | 配置 |
| --- | ---: | --- |
| 4× | 65,536 | `configs/p_g3_local_4x.yaml` |
| 16× | 262,144 | `configs/p_g3_local_16x.yaml` |
| 64× | 1,048,576 | `configs/p_g3_local_64x.yaml` |
| 128× | 2,097,152 | `configs/p_g3_local_128x.yaml` |

使用实验5的 BUNDLE 路径，在仓库根目录运行：

```bash
~/.venvs/aisim/bin/python experiments/disk/scripts/run_p_g3_capacity_sweep.py --bundle "$BUNDLE"
```

按容量从小到大串行运行，任一组失败即停止。可用 `--multipliers 64 128` 选择部分组，用 `--max-requests 1000` 小规模验证。结果保存在 `~/project/results/glm52-1p1d-p-g3-capacity/<批次时间>/<实验标识>/<运行时间>/`。比较 SLA、TTFT、G3 命中和读写量时，使用相同数据的实验5结果作 1× 参照。

## 实验10–13：固定 SSD 128×，扫描 P 侧 DRAM

以 8192 块 DRAM 为 1×，测试 2048、4096、8192、32768 块。配置为
`configs/p_g3_local_dram_<块数>.yaml`。沿用 SSD 128× 的 2,097,152 块容量与全部其余配置，
包括 SSD 每方向 10 GB/s、G2 每方向 32 GB/s、D 仅 HBM、原 SLA 及 16 周同一 FreeInference 数据。
8192 块组与 SSD 扫描的 128× 组配置完全相同。

沿用实验5的 BUNDLE，在仓库根目录运行：

```bash
~/.venvs/aisim/bin/python experiments/disk/scripts/run_p_g3_dram_sweep.py --bundle "$BUNDLE"
```

默认串行运行四组，失败即停。128× 已完成且使用同一数据与配置时，可复用其 8192 块结果，
用 `--dram-blocks 2048 4096 32768` 只跑另外三组。支持 `--max-requests 1000` 和 `--dry-run`。
结果保存在 `~/project/results/glm52-1p1d-p-g3-dram/`。

DRAM 扫描结果见 [分析报告](analysis/p_g3_local_dram_sweep/README.md)。

## 实验14–17：SSD 读带宽扫描

4096、8192 块 DRAM 各跑 2.5、5 GB/s，复用各自 10 GB/s 对照。SSD 128×、写带宽 10 GB/s，其余配置沿用 DRAM 扫描。串行入口为 scripts/run_p_g3_read_bw_sweep.py，接收 --bundle，支持 --dram-blocks 与 --read-gbps 选择子集。结果在 ~/project/results/glm52-1p1d-p-g3-read-bw/。

## P 本地 G3 与 D 本地 G2 并存验证

P 使用独立 DRAM 和 SSD 池，D 使用自己的本地 DRAM 池。首次 P→D 仍通过 HBM 直接交接。

在仓库根目录执行：

    ~/.venvs/aisim/bin/python experiments/disk/scripts/validate_p_g3_d_g2_local.py

合成验证包括 P 的 A → B → A 前缀恢复，以及 D 生成中被抢占后的重新准入对照。P 经 SSD 恢复 128 token。D 开本地 G2 时重新准入复用 64 token，关掉时复用 0。各组完整输出。结果见 ~/project/results/glm52-1p1d-g3-d-g2-validation/。

## 场景 1 八盘方案：P DRAM × 到达倍率

同一份 16 周 FreeInference（665,415 条）、GLM-5.2-FP8、H200 1P1D、P/D 各 TP8、MoE TP8/EP1，原请求 SLA（完整输出、TTFT ≤ 5 秒、平均 ITL ≤ 50 ms）。每组从冷缓存开始，倍率只压缩到达间隔。

| 项目 | 设置 |
| --- | --- |
| P 本地 G2 | 384、256、128 GB，对应 12455、8303、4152 块 |
| D 本地 G2 | 256 GB，对应 8303 块，独立于 P |
| P 本地 G3 | 8 × 7 TB，约 56 TB，对应 1816378 块 |
| G3 带宽 | 读写各 8 GB/s/TP 分片，各方向节点等效 64 GB/s |
| G2 链路 | P/D 的 H2D 设 25 GB/s/TP 分片，D2H 沿用 32 GB/s/TP 分片 |
| 倍率 | 1×、2×、8×，共 9 组 |

容量按 30,830,592 字节/TP8 块取最近整数。384 GB 是全部划给 KV 的容量上限假设。SSD 读写独立限速。P→D 仍使用直接 HBM 交接。

配置位于 configs/p_local_8ssd_p<容量>gb_<倍率>x.yaml。运行入口：

    ~/.venvs/aisim/bin/python experiments/disk/scripts/run_p_local_8ssd_sweep.py --bundle "$BUNDLE"

BUNDLE 沿用实验5的路径。脚本默认串行运行九组，失败即停。支持 --p-dram-gb、--speedups、--max-requests 和 --dry-run。结果位于 ~/project/results/glm52-1p1d-local-8ssd/，逐请求报告保留 G2 直接复用和 G3 恢复复用计数。比较时在同一倍率横向对照 P 侧 DRAM，另看各容量从 1× 到 8× 的变化。

## 1× 本地 SSD 带宽扫描

固定 1× 全量回放。每 TP 分片 SSD 读写同档设置为 6、4、2 GB/s，分别组合 P 侧 DRAM 384、256、128 GB，共九组。沿用现有 8 盘配置的其余参数，使用原有 8 GB/s 的三组 1× 结果作对照。

入口为 `scripts/run_p_local_8ssd_bandwidth_sweep.py`。每组使用独立的 `configs/p_local_8ssd_p{容量}gb_1x_ssd{带宽}gbps.yaml`。比较联合 SLA、TTFT 和 SSD 读写流量。

## H100 80 GB 本地 SSD扫描

模型为 MiniMax-M2.5，使用 H100 SXM 80 GiB，1P1D 各 TP8、Attention DP1、MoE TP1 / EP8。入口为 `scripts/run_h100_local_8ssd_sweep.py`，运行时通过 `--bundle` 指定新转换的数据。

沿用 11 月 6 日的九组回放倍率与缓存块数，以及读写带宽和 SLA。容量档位名称沿用旧实验，MiniMax 的单块 KV 字节数可能不同。九组配置已通过结构校验。384 GB、1× 的 10 条请求小样本已使用 vLLM 0.24.0 算子数据跑通。

## 128K 时间叠加负载扫描

H200、GLM-5.2-FP8、1P1D，P/D 各 Attention TP8、MoE TP8／EP1。2×、4×、8× 叠加 trace 各跑三档 P 侧 DRAM，共九组。复用原 1× YAML，回放倍率固定为 1。原 1× 结果作为对照。先用 `prepare_freeinference_overlay.py` 生成数据，再用 `run_p_local_8ssd_overlay_sweep.py` 启动。

16× 为单独加跑的三组。把原 16 周按周叠到第一周，使用 `--factors 16` 生成，再用 `--overlays 16` 跑三档 DRAM。默认仍跑原九组。

## 8x overlay SSD read bandwidth sweep

P DRAM 128 GB, SSD read 12 and 16 GB/s per TP shard, write fixed at 8 GB/s. Reuse the 8 GB/s overlay baseline. Entry point: `scripts/run_p_local_8ssd_overlay_read_sweep.py`.
