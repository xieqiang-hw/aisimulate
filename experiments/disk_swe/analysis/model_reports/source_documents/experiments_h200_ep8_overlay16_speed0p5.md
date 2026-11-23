# H200 EP8：16×会话叠加＋speedup 0.5

## 目的

与原8×叠加、speedup=1对照，增加会话间交错与同一会话的复用距离，同时使主要时段的原始请求到达强度大致相近，观察分层KV复用、SSD读量、执行状态和SLA的变化。

16×叠加沿用已有数据包；speedup=0.5由仿真入口执行，时间戳间隔放大两倍，不再复制一份放慢后的数据包。请求内容、输入输出长度和hash_ids保持不变。

本轮仍沿用原实验的独立请求开放式回放，没有新加session_id/delay依赖。总会话数并未翻倍，也不保证正在执行的请求数翻倍；增加的是会话叠加程度与缓存工作集压力的可能性。

## 配置与矩阵

| 项目 | 设置 |
|---|---|
| 模型、设备 | zai-org/GLM-5.2-FP8，H200 SXM |
| 部署 | 1P1D；P/D各Attention TP8/DP1，MoE TP1/EP8 |
| 最大上下文 | 131,072 token |
| 性能模型 | op_level，vLLM 0.24.0，沿用原配置 |
| HBM KV | 自动容量，memory_fraction=0.9，64 token/block |
| P DRAM | 512／384／256／128／64 GB |
| P 本地SSD | 约56 TB，worker_local |
| SSD读带宽 | 1／2／4／8／16 GB/s/TP |
| SSD写带宽 | 固定8 GB/s/TP |
| H2D / D2H | 各25 GB/s/TP |
| D 本地缓存 | 仅HBM |
| PD交接 | 原直通路径，50 GB/s，destination_missing |
| 调度 | P/D单轮16,384 token；P最多32、D最多256条活跃序列 |
| SLA | TTFT ≤ 1秒，单请求平均TPOT ≤ 30ms |
| 执行 | 共25组，串行；每组进程树预算18GB；失败继续后续组 |
| 观测 | 强制开启capture-scheduler-trace，gzip压缩 |
| 输出与临时文件 | D盘，默认 /mnt/d/aisim-results/swe-chat-glm52-h200-ep8-overlay16-speed0p5 |

这里“所有组合”按原报告16×矩阵定义：5档DRAM × 5档读带宽，包含新增16 GB/s；不新增1000 GB DRAM。

配置目录：configs/h200_overlay16_speed0p5/，每组一份YAML。脚本还会在结果目录保存本次使用的配置快照，包含命令行覆盖的内存预算。

## 时间范围与比较口径

| 指标 | 原8×，speedup=1 | 新16×，speedup=0.5 |
|---|---:|---:|
| 总请求数 | 607,190 | 607,190 |
| 主要负载窗口 | 28.97370天 | 28.97370天 |
| 主要窗口内请求数（据manifest） | 603,973 | 600,976 |
| 主要窗口平均到达请求/s | 0.241268 | 约0.240071 |
| 全到达跨度 | 47.60152天 | 66.22935天 |

主要窗口到达请求/s约差0.5%，并非严格相等；全程平均更不能直接看成一致。优先比较主要窗口及对齐时间段，保留突发曲线、长尾和排空时间。正式回放后再以逐请求结果核对统计。

即使原始输入请求速率相近，缓存命中变化仍可能增加实际Prefill计算或重算，不假设计算量固定。预计同一会话复用间隔变长，但是否真的降低HBM复用，应以结果验证。

## 运行

已有数据包：

/home/xieqiang/project/data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-overlay-16x

入口：scripts/run_h200_overlay16_speed0p5_sweep.py。

```bash
cd ~/project/aisimulate

OUT="/mnt/d/aisim-results/swe-chat-glm52-h200-ep8-overlay16-speed0p5"
mkdir -p "$OUT"
LOG="$OUT/sweep-$(date +%Y%m%d-%H%M%S).log"

nohup ~/.venvs/aisim/bin/python -u \
  experiments/disk_swe/scripts/run_h200_overlay16_speed0p5_sweep.py \
  --speedup 0.5 \
  --memory-limit-gb 18 \
  --capture-scheduler-trace \
  --scheduler-trace-compression gzip \
  --output-root "$OUT" \
  > "$LOG" 2>&1 &

echo "PID=$!  LOG=$LOG"
tail -F "$LOG"
```

这是独立的串行队列，不会自动等待其他扫描进程。若原20GB预算队列仍在运行，应先等它结束，再启动本队列；不要将两个大内存队列同时启动。

只列矩阵：上述脚本添加 --plan。保存并校验全部配置但不回放：添加 --dry-run。

筛选小范围可用 --p-dram-gb 256 128 --read-bandwidths 2；--max-requests仅用于小样本验证，正式默认0表示全部请求。

D盘未挂载或结果目录不在/mnt/d下时脚本会拒绝运行，避免意外写回WSL虚拟磁盘。报告、配置快照、日志、逐请求结果、scheduler trace和TMPDIR都位于D盘；源码、现有输入包与Python环境位置不变。

## 中断后续跑

```bash
~/.venvs/aisim/bin/python -u \
  ~/project/aisimulate/experiments/disk_swe/scripts/run_h200_overlay16_speed0p5_sweep.py \
  --capture-scheduler-trace --scheduler-trace-compression gzip \
  --resume "/mnt/d/aisim-results/swe-chat-glm52-h200-ep8-overlay16-speed0p5/实际时间戳/sweep.json"
```

默认25组可直接使用上述命令；若初次使用非默认选择或内存参数，续跑时重复相关参数。续跑核对输入、配置和引擎，跳过已完成且记录完整的组，重试失败或中断的组。禁止两个进程同时续跑同一计划。

## 验证情况

- 20份YAML通过AISim公开配置校验。
- 使用16×包前4条真实请求，以2GB小样本预算跑了256/128 GB DRAM、读2 GB/s两组，均完成。
- 到达时间准确等于源timestamp / 0.5。
- 每组P实际计算2,357 token、D生成438 token，与小样本报告及输出总量一致。
- P、D逐轮记录齐全，执行时间区间合法，gzip文件可读取。
- 报告与TMPDIR均在D盘，续跑正确跳过两组。
- 2GB只是小样本预算，不作为全量预算依据；正式默认18GB。
- 验证产物在 /mnt/d/aisim-results/overlay16-speed0p5-validation/。

新队列为25组；结果合入 experiments_h200_ep8_16x.md/html，以speedup区分。

本次启动时主机可用余量不足20 GB，运行预算调整为18 GB；失败启动记录保留在D盘，全部25组重新运行。

## 1000 GB DRAM补测

与原8×对齐，追加1000 GB DRAM × SSD读带宽1／2／4／8／16 GB/s/TP，共5组。等原25组全部结束后自动串行运行，仍为speedup=0.5、18 GB进程树预算、gzip调度轨迹，结果写D盘并合入同一报告。1000 GB约为32435块；此前默认25组的入口保持不变，补测通过--p-dram-gb 1000运行。
