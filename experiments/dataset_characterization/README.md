# 两种实验输入的数据画像

在 WSL 中运行（仅依赖 Python 标准库）：

```bash
cd /home/xieqiang/project/aisimulate/experiments/dataset_characterization
python3 characterize_datasets.py
~/.venvs/aisim/bin/python characterize_upstream.py
python3 -m unittest -v test_characterize_datasets
```

默认使用 disk / disk_swe 正式脚本中的基础数据包。可通过 `--freeinfer DIR`、`--swe DIR`、`--window-minutes 5`、`--output PATH.html` 覆盖。overlay 包可显式传入；报告时间统计将反映该包的重排后时间。

生成 `reports/dataset_characteristics.html` 与同名 JSON。HTML 内嵌 SVG，无网络依赖。JSON 保存完整窗口、联合直方图和来源元信息。脚本流式读取，FreeInference 使用带长度字段的 source_requests；SWE 逐行配对 trace 与 sidecar，仅提取 trace 标量而不解码大块 hash_ids。

原始数据脚本需要 PyArrow，生成 `reports/upstream_characteristics.html` 与同名 JSON。可用 `--free-root` 和 `--swe-root` 指定本地发布目录。两份报告均包含会话轮数分布及分轮段输入、输出长度，在各统计表中列出有效记录数。

验证范围：全部记录数、时间单调性、可用 SHA256、输入输出总量、时间窗请求数守恒。FreeInference 未读取的 requests.jsonl 及上游 parquet 不在本次校验范围内。session_id 不写入聚合报告。

数据口径：保留回放子集，非上游全量；FreeInference token 为 o200k 输入与 provider 输出，SWE 为 GLM 重建，不能将长度差异直接解释为硬件算量差异。SWE 到达使用首个响应事件代理，且缺失原始 system/tool schema。缓存 provider 计数不是模拟缓存命中率。FreeInference 上游来源、revision 与许可见数据包 ATTRIBUTION.md 和生成报告 manifest。
