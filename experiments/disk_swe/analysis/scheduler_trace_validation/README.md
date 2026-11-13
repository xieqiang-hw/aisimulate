# 逐轮观测验证与优化

本次只优化观测，不改变调度、缓存或性能模型。记录默认关闭，通过 `--capture-scheduler-trace` 开启。

## 优化内容

1. 固定 Rust 结构直接序列化成固定列数值数组，静态角色、worker、DP rank、容量和列定义只写入元数据。
2. 完整保留每轮 BS、token 数与执行时间；队列、KV 和恢复状态只在变化时记录。按 stream_id＋iteration_id 关联，避免同时间戳重试发生错位。
3. 增加压缩前缓冲，并实测 gzip／不压缩／关闭记录。未引入异步无界队列。

输出包括 iteration_metrics.jsonl.gz、scheduler_states.jsonl.gz、scheduling_events.jsonl.gz 和 scheduler_trace_metadata.json。使用 `scripts/scheduler_trace.py` 的 iter_iterations、iter_events 流式展开成具名字段；兼容旧版完整 JSON 对象。

## 正确性

- 239 项 vLLM 调度器测试通过，覆盖 chunked prefill、抢占、缓存和并行路径。
- gzip 与不压缩两种模式均完成 CLI 对照：P 本地 SSD 恢复、H2D 等待、D 抢占后本地 G2 恢复均能记录；开关前后的逐请求执行结果除 UUID 外完全一致。
- D 恢复夹具：1 次重新准入、复用 64 个 DRAM token，与 admission_history 一致。
- 状态读取器验证了同时间戳多次重试与 P/D 流交错。
- 在真实数据基准中，新旧记录的 **669,481 轮批次与状态逐字段完全一致**；所有基准运行的逐请求结果一致。没有因提速而丢弃迭代或状态。

## 性能对照

同一段 SWE-chat 3,000 请求；GLM-5.2-FP8／H200／1P1D，P DRAM 128 GB、HBM 2,048 块、2× 会话叠加。各方式重复两次，取中位数；计时为 run.json 的 wall_seconds，不包含 runner 前置数据校验。

| 方式 | 实际耗时 | 观测文件总大小 |
|---|---:|---:|
| 原版关闭记录 | 6.33 秒 | 0 |
| 原版逐轮 JSON＋gzip | 28.03 秒 | 38.93 MB |
| 优化版关闭记录 | 6.12 秒 | 0 |
| 优化版完整记录，不压缩 | 6.33 秒 | 67.97 MB |
| 优化版完整记录，gzip | 6.53 秒 | 14.47 MB |

新版 gzip 比旧记录方式快约 **4.29 倍**，文件缩小约 **62.8%**；对比优化版关闭记录，额外耗时约 **6.7%**。gzip 比不压缩多约 0.21 秒、文件缩小约 78.7%，因此全量重跑保留 gzip。

这些是此请求片段的单组实测，不能直接视为全部 40 组四并行运行的精确开销。

## 复现与证据

- 功能验证：`scripts/validate_scheduler_trace.py --compression gzip`，再用 `--compression none` 对照。
- 性能测试：`scripts/benchmark_scheduler_trace.py`。
- 逐请求与完整记录核对：`scripts/summarize_scheduler_trace_benchmark.py`。
- [功能验证 JSON](assets/validation.json)
- [性能与一致性核对 JSON](assets/performance.json)

脚本路径均相对于 disk_swe。完整字段及采样边界见仓库 docs/replay/engine/workers.md 的 Streaming scheduler observations。队列分类存在重叠，普通等待不代表 KV 已就绪，不能仅凭该计数判断算力瓶颈。
