# GLM-5.2 · SWE-chat 场景 2 共享盘框实验

更新时间：2026-11-25T18:42:40+08:00。共 9 组；已完成 7 组，失败 2 组。

## 1. 配置与数据

| 项目 | 内容 |
|---|---|
| 模型 | zai-org/GLM-5.2-FP8 |
| GPU／性能模型 | H200 SXM；算子级性能模型 |
| 精度 | GLM-5.2-FP8；KV FP8，容量使用已验证的 GLM DSA 布局 |
| 部署 | 1P1D；P/D 各 8 卡，共 16 卡 |
| 并行 | Attention TP8 / DP1；MoE TP1 / EP8 |
| HBM KV | P/D 自动 KV 容量，memory_fraction=0.9；64 token / KV 块 |
| 缓存层级 | P：HBM + 256／128／64 GB 本地 DRAM；D：HBM；P、D 共用盘框，无本地 SSD |
| 共享盘框 | 167.616 TB；整框读 80／40／20 / 写 80 GB/s；固定时延 0.1 ms |
| 传输 | H2D/D2H 各 25 GB/s/TP 分片；盘框直达 HBM；P→D 经盘框交接 |
| 调度 | P/D 每轮 16,384 token；序列上限 32/256；Chunked Prefill |
| 内存预算 | 进程树预算 20.0 GB |
| 联合 SLA | 原始 TTFT ≤ 1 秒，且单请求平均 TPOT ≤ 30 ms |
| 上下文 | 输入＋输出合计不超过 128K token |
| 数据来源 | SWE-chat Claude Code 主会话，verified-interleave 版 |
| 数据编码 | compact Mooncake；连续输入＋输出哈希；每段 16 token |
| 叠加时间 | 标称 8×；全程 47.60 天，全程平均密度 4.869× |
| 到达语义 | 按时间戳独立到达，未启用 session_id＋delay 完成依赖 |

每组 607,190 条请求，原始会话 10,962 个。叠加保留会话内相对间隔，speedup=1。

## 2. 实验结果

### 预算 16,384 token · 8× 叠加 · speedup=1

| P 侧 DRAM | 盘框读带宽（GB/s，整框） | 盘框写带宽（GB/s，整框） | 状态 | 完成请求 | 联合 SLA 达标率 | P99 TTFT（秒） | 平均 TTFT（秒） | TPOT（ms） | HBM 复用 | DRAM 直接复用 | 盘框恢复复用 | 总复用率 | 盘框累计读（TB） | 实际耗时 | 扣排队后平均 TTFT（秒） | 扣排队后 P99 TTFT（秒） | 内存峰值（GB） | 盘框累计写（TB） | D 生成块写回（块） | P 复用 D 生成块（块次） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 256 GB | 80 | 80 | 已完成 | 607,190 | 95.458% | 2.67 | 0.27 | 12.14 | 91.029% | 3.461% | 3.641% | 98.132% | 1,715.770 | 31 分 44 秒 | 0.24 | 2.10 | 11.781 | 324.923 | 2,212,249 | 5,242,963 |
| 256 GB | 40 | 80 | 已完成 | 607,190 | 93.701% | 3.30 | 0.31 | 12.14 | 91.019% | 3.485% | 3.628% | 98.132% | 1,723.214 | 30 分 32 秒 | 0.28 | 2.88 | 11.825 | 324.923 | 2,210,801 | 5,227,017 |
| 256 GB | 20 | 80 | 已完成 | 607,190 | 90.820% | 7.38 | 0.52 | 12.16 | 90.701% | 3.674% | 3.754% | 98.128% | 1,824.116 | 31 分 38 秒 | 0.43 | 6.34 | 11.777 | 324.923 | 2,204,792 | 5,342,616 |
| 128 GB | 80 | 80 | 已完成 | 607,190 | 94.481% | 2.81 | 0.28 | 12.14 | 90.350% | 0.081% | 7.700% | 98.131% | 2,259.633 | 38 分 49 秒 | 0.25 | 2.24 | 11.826 | 324.923 | 2,211,692 | 9,791,463 |
| 128 GB | 40 | 80 | 已完成 | 607,190 | 92.137% | 3.94 | 0.35 | 12.15 | 90.329% | 0.174% | 7.627% | 98.130% | 2,267.434 | 41 分 3 秒 | 0.32 | 3.41 | 11.775 | 324.923 | 2,209,362 | 9,680,972 |
| 64 GB | 80 | 80 | 已完成 | 607,190 | 94.346% | 2.83 | 0.29 | 12.14 | 89.857% | 0.007% | 8.267% | 98.131% | 2,335.579 | 62 分 2 秒 | 0.26 | 2.27 | 11.778 | 324.923 | 2,211,612 | 10,350,571 |
| 64 GB | 40 | 80 | 已完成 | 607,190 | 91.864% | 4.21 | 0.37 | 12.15 | 89.777% | 0.031% | 8.322% | 98.130% | 2,363.515 | 51 分 47 秒 | 0.33 | 3.59 | 11.831 | 324.923 | 2,208,990 | 10,393,487 |
| 64 GB | 20 | 80 | 失败 | — | — | — | — | — | — | — | — | — | — | 22 分 28 秒 | — | — | 20.003 | — | — | — |
| 128 GB | 20 | 80 | 失败 | — | — | — | — | — | — | — | — | — | — | 22 分 48 秒 | — | — | 20.005 | — | — | — |

## 3. 盘框路径与统计口径

- P→D 经共享盘框交接。P 写已计算的输入块，D 写已计算的完整生成块；后续 P 可命中 D 写回块。
- 分层复用率＝P 首次准入各层复用 token 总数／全部输入 token 总数。盘框恢复直达 HBM，不要求经过本地 DRAM。
- 累计读写为整框字节量，覆盖 P 恢复、P→D 交接和 D 写回等任务，不等同于首次准入实际复用量。D 写回与 D→P 复用计数不表示独立请求数。
- 扣排队后 TTFT 仅扣除 P 首次准入前调度等待，保留 KV 恢复、后续 chunk 等待、交接及 D 侧等待。
- SLA 使用原始 TTFT 与请求级平均 TPOT；单 token 输出跳过 TPOT 判定。原始分位取引擎报告，扣排队后分位取逐请求最近秩。
- 未完成项留空，不填 0，不绘制为已完成结果。GB/TB 为十进制；容量按已验证的 GLM 每分片 60,216 字节/token × 64 × TP8 换算。

## 4. 更新与来源

- HTML：[enclosure_experiments.html](enclosure_experiments.html)，单文件离线使用。
- 更新：仓库根目录运行 `~/.venvs/aisim/bin/python experiments/disk_swe/scripts/build_enclosure_report.py`；随后刷新浏览器。
- 数值与来源索引：`analysis/enclosure_reports/rows.json`。构建脚本不启动后台监视任务。
- 20261125T021230.161422Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T021230.161422Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
- 20261125T030237.183307Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T030237.183307Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
- 20261125T033734.358392Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T033734.358392Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
- 20261125T041339.276577Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T041339.276577Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
- 20261125T045648.626043Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T045648.626043Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
- 20261125T060932.189320Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T060932.189320Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
- 20261125T071602.157543Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T071602.157543Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
- 20261125T081219.593559Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T081219.593559Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
  - owned process tree exceeded the memory budget
- 20261125T083928.793082Z：`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T083928.793082Z`；数据：`/mnt/d/aisim-data/swe_chat/derived/verified_interleave_mooncake_v1/claude-code-main-glm52-128k-compact-overlay-8x`。
  - owned process tree exceeded the memory budget
  - 历史尝试 20261125T054209.554768Z：失败；`/home/xieqiang/project/results/swe-chat-enclosure-h200-ep8-compact/20261125T054209.554768Z`。
