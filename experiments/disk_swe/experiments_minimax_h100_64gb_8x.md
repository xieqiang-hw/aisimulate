# SWE-chat 场景 1 · MiniMax-M2.5 / H100 64 GB / 8×

沿用 experiments_h200_ep8_16x.md 的 8× 负载形状与扫描范围，模型替换为 MiniMax-M2.5，GPU 替换为 H100 SXM，并将 P、D 每张卡的**总 HBM 容量限制为 64 GB**。H100 的计算、HBM 带宽与互联参数保持不变，因此这是容量约束对照，不是 910B 性能仿真。原实验配置和结果文档保持不变。

## 1. 实验配置

| 项目 | 配置 |
|---|---|
| 模型 | MiniMaxAI/MiniMax-M2.5；官方 FP8 权重，AISim 当前解析 KV 为 FP8 |
| GPU | H100 SXM；单卡总 HBM 为 64,000,000,000 字节，即 64 GB ≈ 59.60 GiB |
| HBM 预算 | P/D 使用率均为 90%，自动计算可用 KV 容量 |
| 部署 | 1P1D，各 8 卡；Attention TP8/DP1，MoE TP1/EP8，共 16 卡 |
| P 本地缓存 | HBM + G2 DRAM + worker_local G3 SSD |
| P DRAM | 1000／512／384／256／128／64 GB，按 MiniMax KV 布局重新换算 |
| P SSD | 8 × 7 TB ≈ 56 TB；读 1／2／4／8／16 GB/s/TP，写固定 8 GB/s/TP |
| D 本地缓存 | 仅 HBM，关闭 G2 和本地 G3 |
| H2D / D2H | 各 25 GB/s/TP 分片，TP8 均匀并行时折合节点各 200 GB/s |
| PD 交接 | 保留直通，50 GB/s/TP，destination_missing，不经过盘框 |
| 上下文上限 | 131,072 token（128K），沿用原输入与输出合计筛选条件 |
| 单步 token 预算 | P/D 均为 16,384 |
| 活跃序列上限 | P 32，D 256 |
| 回放 | 仅标称 8× 会话叠加，speedup=1，607,190 条请求 |
| SLA | TTFT ≤ 1 秒，且单请求平均 TPOT ≤ 30 ms |
| 执行 | 30 组串行，先按读带宽升序、同一带宽内按 DRAM 升序；主机预算默认自动 |
| 调度观测 | 默认开启 capture-scheduler-trace，scheduler-trace-compression=gzip |
| 性能估算 | op_level，SILICON，vLLM 0.24.0，禁止估算器回退 |

读带宽均按每 TP 分片计，节点等效为 8／16／32／64／128 GB/s；节点等效写带宽为 64 GB/s。读写方向沿用引擎的分别限速方式。

## 2. 单卡 HBM 容量核算

以下为 AISim 原生内存估算及回放容量解析实际返回值，P/D 在此配置下相同；GB 均为十进制。

| 用途 | 单卡容量（GB） |
|---|---:|
| 总 HBM | 64.000000 |
| 90% 使用预算 | 57.600000 |
| 模型权重（TP8/EP8 后） | 28.677538 |
| 激活与临时空间估计 | 2.013266 |
| 运行时预留 | 3.758096 |
| 通信预留 | 0.411042 |
| 剩余 KV 字节预算 | 22.740058 |
| 按整块取整后的 KV | 22.739878 |
| 未划入 90% 预算的余量 | 6.400000 |

自动 KV 容量：**P、D 每卡均为 22,386 块，覆盖 1,432,704 token 位置**。TP8 保存同一批 token 的不同 KV 分片，因此 token 容量不能再乘 8。

一条完整 128K 请求需要 2,048 块、单卡约 2.080 GB KV。仅按容量可放约 10 条完全独立、接近 128K 的全长 KV；实际并发还取决于前缀共享、请求长度、计算和调度，不能把这个数直接视为系统最大并发。

这里限制的是总 HBM，而不是把 KV 设为 64 GB。MiniMax 的权重和每 token KV 比 GLM 小，所以总 HBM 缩到 64 GB，也不意味着可缓存的 token 数比原 GLM/H200 更少。

## 3. G2 / G3 块数换算

MiniMax 为 62 层、8 个 KV heads、head_dim=128；TP8 每卡持有 1 个 KV head，FP8 KV 元素为 1 字节：

~~~text
每 token、每卡 = K/V 两份 × 62 层 × 1 KV head × 128 × 1 字节
              = 15,872 字节
每块、每卡    = 15,872 × 64 = 1,015,808 字节
每块、TP8节点 = 1,015,808 × 8 = 8,126,464 字节
缓存块数      = round(目标字节数 ÷ 8,126,464)
~~~

一块覆盖 64 个连续 token 的全部 62 层 KV；各卡存自己的 TP 分片。输入 trace 中 hash_ids 的 16 token 粒度与引擎 64 token KV 块不是同一单位。

| P DRAM 标称容量 | num_host_blocks | 实际容量（GB） |
|---:|---:|---:|
| 1000 GB | 123,055 | 1000.002028 |
| 512 GB | 63,004 | 511.999738 |
| 384 GB | 47,253 | 383.999803 |
| 256 GB | 31,502 | 255.999869 |
| 128 GB | 15,751 | 127.999934 |
| 64 GB | 7,876 | 64.004030 |

SSD：num_g3_blocks = **6,891,066**，实际约 **55.999999771 TB**。不沿用 GLM 的 G2/G3 块数。

## 4. 数据口径

本次采用已确认的“相同负载形状对照”：请求长度、前缀哈希、输出长度、回放 token ID 和到达时间沿用原 GLM tokenizer 生成的数据，没有用 MiniMax tokenizer 重新分词。MiniMax 提供性能模型及 KV 字节布局；结果不代表真实 MiniMax 输出文本或质量，也不能与重新分词后的真实 MiniMax 负载混称。

原包：/home/xieqiang/project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-overlay-8x\
新对照包：/home/xieqiang/project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-minimax-m25-shape128k-overlay-8x

新包单独记录模型映射与来源，requests.jsonl 为硬链接，不再复制约 18 GB 文件。请勿原地改写。原包 manifest 没有修改。仍沿用原先独立请求的时间戳回放，不额外加入 session_id/delay 的闭环约束。标称 8× 的全到达时段平均密度仍约为原始数据的 4.869×。

## 5. DRAM × SSD 读带宽结果

<!-- results:status:start -->
完成 1／30 组，运行 0 组，失败 0 组，待启动 28 组。最后更新 2026-11-19 20:19:59 +0800。
<!-- results:status:end -->

30 组配置已生成，当前全量进度如下。20 条真实请求的小样本只验证功能，不填入全量结果。

| 标称叠加倍率 | P 侧 DRAM | SSD 读带宽（GB/s／TP 分片） | 状态 | 完成请求 | 联合 SLA 达标率 | P99 TTFT（秒） | 平均 TTFT（秒） | TPOT（ms） | HBM 复用 | DRAM 直接复用 | SSD 恢复复用 | 总复用率 | SSD 累计读 | 实际耗时 | 扣排队后平均 TTFT（秒） | 扣排队后 P99 TTFT（秒） | 内存峰值（GB） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8× | 64 GB | 1 | 已完成 | 607,190 | 99.656% | 0.47 | 0.06 | 8.51 | 97.140% | 0.000% | 0.520% | 97.661% | 2.284 TB | 21 分 22 秒 | 0.06 | 0.44 | 12.838 |
| 8× | 128 GB | 1 | 已中断 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 256 GB | 1 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 384 GB | 1 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 512 GB | 1 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 1000 GB | 1 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 64 GB | 2 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 128 GB | 2 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 256 GB | 2 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 384 GB | 2 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 512 GB | 2 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 1000 GB | 2 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 64 GB | 4 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 128 GB | 4 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 256 GB | 4 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 384 GB | 4 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 512 GB | 4 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 1000 GB | 4 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 64 GB | 8 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 128 GB | 8 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 256 GB | 8 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 384 GB | 8 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 512 GB | 8 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 1000 GB | 8 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 64 GB | 16 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 128 GB | 16 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 256 GB | 16 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 384 GB | 16 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 512 GB | 16 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| 8× | 1000 GB | 16 | 待启动 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

联合 SLA 使用原始 TTFT；扣排队后两列只扣除 P 首次准入前调度等待，保留 KV 恢复等待。分层复用为首次准入时该层复用 token 占所有输入 token 的比例；SSD 累计读包含重复恢复。

## 6. 运行与验证

配置：configs/swe_minimax_h100_64gb_ep8_p*gb_read*_overlay_8x.yaml。\
入口：scripts/run_minimax_h100_64gb_8x_sweep.py。\
硬件覆盖：configs/systems_h100_64gb/h100_sxm.yaml。

hardware 仍为 h100_sxm，以复用对应算子数据库；独立 systems_paths 优先加载 64 GB 配置，不修改官方硬件文件。

查看计划：

~~~bash
cd ~/project/aisimulate
~/.venvs/aisim/bin/python experiments/disk_swe/scripts/run_minimax_h100_64gb_8x_sweep.py --plan
~~~

启动 30 组：先跑读带宽 1 GB/s 下的 DRAM 64→128→256→384→512→1000 GB，再依次跑 2、4、8、16 GB/s 的相同容量顺序。

~~~bash
cd ~/project/aisimulate
OUT="$HOME/project/results/swe-chat-minimax-h100-64gb-ep8-8x"
mkdir -p "$OUT"
LOG="$OUT/sweep-$(date +%Y%m%d-%H%M%S).log"
nohup ~/.venvs/aisim/bin/python -u \
  experiments/disk_swe/scripts/run_minimax_h100_64gb_8x_sweep.py \
  --capture-scheduler-trace \
  --scheduler-trace-compression gzip \
  --output-root "$OUT" > "$LOG" 2>&1 &
echo "PID=$! LOG=$LOG"
tail -F "$LOG"
~~~

默认记录 P/D 每轮执行与调度状态，生成 iteration_metrics.jsonl.gz、scheduler_states.jsonl.gz、scheduling_events.jsonl.gz 和 scheduler_trace_metadata.json，用于后续 BS、忙碌比例、计算 token 与等待原因分析。即使命令省略两个观测参数，本入口也默认开启 gzip 调度轨迹。

默认自动选择主机内存预算；如需固定每组预算，可添加 --memory-limit-gb 20。串行运行，单组失败后继续后续组，状态保存在 sweep.json。实验结果保存于 D 盘；后台程序在有新结果时同步更新本表和同名 HTML。

已验证：30 份 YAML 通过校验；P/D 实际解析均为 64 GB 总 HBM、15,872 字节/token/分片、22,386 GPU 块；20/20 条真实请求正常完成。未修改或重编译引擎，全量实验状态见第 5 节。

容量和验证证据：analysis/minimax_h100_64gb_setup/capacity.json、compiled_capacity.json、validation.json。

小样本结果：/home/xieqiang/project/results/swe-chat-minimax-h100-64gb-smoke/20261119T065944.250107Z。
