# SWE-chat 分层 KV 缓存实验

## 原始需求

对于盘框这个场景建模分析：

1. 盘框拥有160GB/s的有效带宽，4个1823网卡，8个200GE的网口。24块盘，24 × 6.984TB NVMe SSD，单盘带宽8GB/s
2. P节点本地内存，24个通道DDR4，3200MHz，容量384GB，带宽614GB/s带宽
3. Ascend 910B3 × 8 / 节点，单卡 64 GB HBM，H2D带宽，单卡25GB，总带宽=200GB/s

在agent场景下，在GLM-5.2-FP8/minimax_m2.5模型下，1P和1D节点，共享一个盘框。 分析下910B下，

场景1：本地内存+本地SSD，采用LRU淘汰策略， 如果内存容量降低逐步降低、或增加，对于DRAM命中率逐步下降，频繁换入换出，需要SSD的加载带宽是多少？ 场景2：KV盘框，采用分流的策略， DRAM按照LRU保留前缀命中KV，盘框保存agent全量KV；分析下DRAM带宽和KV盘框带宽？

对比，预期结论。场景1当中，本地内存需要频繁换入换出，SSD的带宽比较高；场景2，相同性能，盘框带宽足够，DRAM容量需求低

## 本目录的实验范围

本目录使用 SWE-chat 评估场景 1 的本地 DRAM 和 SSD。我们固定模型与部署，观察 P 侧 DRAM 缩容后，前缀复用如何在 HBM、DRAM 与 SSD 间转移，以及提高 SSD 读带宽能追回多少联合 SLA 和尾延迟。项目背景与盘框需求见 [盘框实验 README](../disk/README.md)。

## 后续统一配置

| 项目 | 设置 |
|---|---|
| 模型与硬件 | GLM-5.2-FP8，H200 SXM，1P1D，P/D 各 8 卡 |
| 并行与性能模型 | P/D 各 Attention TP8／DP1、MoE TP8／EP1，沿用 vLLM 0.24.0 算子级性能模型 |
| P 侧 HBM | KV 池固定为 2,048 块，每块 64 token。只缩减 KV 容量，沿用 H200 计算耗时 |
| D 侧 | HBM KV 容量自动配置，本地独立 DRAM 约 256 GB |
| 场景 1 缓存 | P 侧 DRAM 128／256／384 GB，本地 SSD 约 56 TB，P→D 直接交接 |
| 带宽 | SSD 默认读写各 8 GB/s／TP 分片，扫描时只调整读带宽。H2D 为 25 GB/s／TP 分片 |
| 联合 SLA | 每条请求 TTFT ≤ 1 秒且平均 TPOT ≤ 30 ms，报告达标请求占比 |

原速基线包含自动 HBM 容量，HBM 容量扫描也包含其他块数。后续场景对照统一使用 P 侧 2,048 块的结果。

## 数据与结果

使用 Claude Code 主会话，输入与输出合计上限 128K，每组全量回放 174,438 条请求。Mooncake 数据包含输出长度和 token ID。叠加以完整会话为单位平移时间戳，保留会话内部节奏。结果以标称倍率标记，实际全程平均密度见各数据包的清单。默认数据位于 `~/project/data/swe_chat/derived/mooncake-v1/`。详见 [数据处理说明](data_process.md)。

- [实验记录](experiments.md)汇总原速基线、HBM 容量、DRAM 与负载叠加、SSD 读带宽四类实验。
- [HTML 汇总](experiments.html)提供 SLA、TTFT、TPOT、分层复用和 SSD 读取量对比。
- SWE 64× 下，128 GB DRAM 配置将读带宽从 8 提到 32 GB/s／TP 分片后，联合 SLA 从 84.029% 提高到 91.900%，P99 TTFT 从 136.37 秒降到 3.52 秒，接近 256 GB DRAM／8 GB/s 对照的 92.021% 和 3.33 秒。128× 下即使提高到 128 GB/s，P99 仍高于容量对照。两组负载可分别用于后续场景对照和尾延迟压力验证。

## 目录与入口

| 路径 | 用途 |
|---|---|
| `configs/` | 实验 YAML 配置 |
| `scripts/prepare_swe_chat.py` | 生成原速数据包 |
| `scripts/prepare_swe_chat_overlay.py` | 生成会话叠加数据包 |
| `scripts/run_experiment_suite.py` | 批量运行实验，使用 `--plan` 预览组合 |
| `scripts/build_results_dashboard.py` | 根据实验记录重建汇总 HTML |
| `analysis/` | 调度观测、时间线分析及验证材料 |
| `backup/` | 历史页面备份 |

原始回放结果保存在 `~/project/results/`，本机临时监视和填表脚本统一放在 `~/project/tmp/`。

在 WSL 中预览一组符合后续配置的扫描：

```bash
~/.venvs/aisim/bin/python \
  /home/xieqiang/project/aisimulate/experiments/disk_swe/scripts/run_experiment_suite.py \
  --suites bandwidth --factors 64 --read-bandwidths 32 --plan
```
