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

## 当前 GLM 场景 1 配置

| 项目 | 设置 |
|---|---|
| 模型与硬件 | GLM-5.2-FP8，H200 SXM，1P1D，P/D 各 8 卡 |
| 并行与性能模型 | Attention TP8／DP1，MoE TP1／EP8，vLLM 0.24.0 算子级性能模型 |
| HBM KV | P、D 自动容量，每块 64 token |
| 场景 1 缓存 | P 本地 DRAM＋约 56 TB SSD；D 仅 HBM；P→D 直接交接 |
| 扫描范围 | P DRAM 64／128／256／384／512／1,000 GB；读带宽 1／2／4／8／16 GB/s／TP 分片 |
| 传输与调度 | SSD 写 8、H2D／D2H 各 25 GB/s／TP；P/D 单轮预算均 16,384 token，序列上限 32／256 |
| 联合 SLA | 每条请求 TTFT ≤ 1 秒且平均 TPOT ≤ 30 ms |

具体组合和已完成状态以 [GLM-5.2 实验记录](glm52_experiments.md)为准；场景 2 的独有配置见 [盘框建模方案](disk_enclosure_modeling.md)。

## 数据与结果

当前 GLM 扫描使用 verified-interleave 版 SWE-chat Claude Code 主会话，输入与输出合计上限 128K，每组 607,190 条请求。Mooncake 数据包含输出 token ID。叠加以完整会话为单位平移时间戳，保留会话内部节奏；实际密度以数据包清单为准。数据位于 ~/project/data/swe_chat/derived/verified_interleave_mooncake_v1/，详见 [数据处理说明](data_process.md)。

- [GLM-5.2 实验记录](glm52_experiments.md)与 [GLM-5.2 HTML 汇总](glm52_experiments.html)：DRAM、带宽、叠加负载对比。
- [MiniMax 实验记录](minimax_experiments.md)与 [MiniMax HTML 汇总](minimax_experiments.html)：BF16 KV 的 DRAM、带宽及单步预算扫描。
- [逐请求指标分布](experiments_h200_ep8_distributions.html)：TTFT、TPOT、等待等分布。
- [吞吐与执行观测](throughput_8x.html)：原始 8× 数据的加压分析。
- 旧 EP1 两批实验的结果、文档、专用配置和入口已清理；历史备份独立保留。

## 目录与入口

| 路径 | 用途 |
|---|---|
| configs/ | 当前实验 YAML 配置 |
| scripts/prepare_swe_chat.py | 生成原速数据包 |
| scripts/prepare_swe_chat_overlay.py | 生成会话叠加数据包 |
| scripts/run_h200_local_8ssd_ep8_sweep.py | H200 EP8 扫描，支持 --plan 预览 |
| scripts/sweep_common.py | 当前各扫描入口复用的运行与数据检查函数 |
| scripts/refresh_model_reports.py | 按模型更新统一 MD 与独立 HTML，支持 --model glm52/minimax/all |
| analysis/model_reports/source_documents/ | 原分散报告留档，作为兼容填表的数据来源 |
| analysis/ | 分析、调度观测和验证材料 |
| backup/ | 历史页面备份 |

原始结果保存在 ~/project/results/ 或对应实验指定的 D 盘目录。临时监视和填表脚本统一放在 ~/project/tmp/。

预览一个当前配置，不启动实验：

```bash
~/.venvs/aisim/bin/python \
  /home/xieqiang/project/aisimulate/experiments/disk_swe/scripts/run_h200_local_8ssd_ep8_sweep.py \
  --bundle "$HOME/project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-overlay-8x" \
  --p-dram-gb 256 --read-bandwidths 2 --plan
```
