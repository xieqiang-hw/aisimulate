# SWE-chat 数据处理计划

首版目标是将 SWE-chat 的 Agent 会话整理成一行一次模型调用的 Mooncake JSONL，用于 AISim 的 GLM-5.2-FP8 输入前缀复用实验。后续保留足够的中间数据，供 D 写入共享盘框、下一轮 P 读回复用的实验使用。

## 输入与产物

- 原始数据放在 `~/project/data/swe_chat/upstream/`，记录 Hugging Face 仓库 revision。用 `sessions.parquet` 筛选会话，用 `session_logs.parquet` 定位原始 transcript，`conversations.parquet` 辅助核对事件类型、响应 ID 和顺序。
- 转换脚本放在本目录的 `scripts/`。产物放在 `~/project/data/swe_chat/derived/mooncake-v1/<bundle-id>/`，包括 `requests.jsonl`、`source_requests.jsonl` 和 `manifest.json`。原始日志单独保留。

## 处理步骤

1. **筛选日志。** 首版用少量 Claude Code 主会话打通流程。优先读原始 transcript，避免结构化表中的工具结果截断。记录缺日志、无法判定调用边界和无法重建上下文的数量。
2. **归并模型调用。** 按会话内事件顺序处理，用 `response_id` 或原始日志中的可靠边界归并文本、思考和工具调用。纯工具调用也保留。工具结果在被送回模型时才计入下一次调用的输入。一条用户消息可能对应多次模型调用。
3. **重建输入与计划输出。** 根据日志还原每次调用可见的历史、工具定义、工具结果和上下文压缩状态。使用固定版本的 GLM-5.2 聊天模板与 tokenizer 生成 token 序列。对无法还原的区段标记或排除，记录原因。首版对照按输入与输出合计不超过 131,072 token 筛选。
4. **生成链式前缀块。** 每 16 个输入 token 生成一个链式 `hash_id`，身份由该块及此前缀共同决定。尾部不满一块时沿用 FreeInference 转换的私有尾块口径。AISim 的 trace 块为 16 token，引擎 KV 块为 64 token。
5. **生成时间轴并导出。** 优先用可确认的模型调用开始时间，换算成相对毫秒后稳定排序。只有响应时间时，单独标记时间代理来源。用代理时间构造的到达率单独解读。
6. **抽样验证，再扩大全量。** 对照原始日志检查纯工具调用、多个工具结果汇总、上下文压缩和分块边界。用短 trace 验证 AISim 能加载、请求能完整结束，以及预期的前缀块能复用。

## 导出字段与验证

| 文件 | 内容 |
|---|---|
| `requests.jsonl` | 每行一条调用，包含 `timestamp`（相对毫秒）、`input_length`、`output_length`、`hash_ids`。 |
| `source_requests.jsonl` | 记录输出行号、源会话和响应 ID、调用顺序、时间来源、重建质量标记。 |
| `manifest.json` | 记录数据 revision、目标模型、模板和 tokenizer 版本、上下文上限、trace 块大小、请求数、过滤计数和文件 SHA256。沿用现有回放入口所需的字段。 |

全量前检查时间单调、行数及哈希一致、块数与输入长度匹配。转换后再统计请求长度、同会话调用间隔和连续公共前缀，便于与 FreeInference 对比。最终结果按 **SWE-chat 日志重建的 GLM-5.2 负载**解释。

## 盘框实验的后续扩展

在中间产物中保留计划输出 token 和调用依赖。逐 token 比较上一轮的「完整输入＋计划输出」和下一轮的完整输入，统计共同前缀中由 D 计算补齐的完整 64-token KV 块。后续可导出带 `output_token_ids` 和 `wait_for` 的 `agentic_mooncake`。先用两轮构造样本验证 D 写入和 P 读回，再用重建负载测收益。

[官方字段说明](https://huggingface.co/datasets/SALT-NLP/SWE-chat#dataset-schema)
