# 输入输出连续哈希格式验证

新格式覆盖 input + output 的连续 token 序列，省略两个 token 数组，
通过 input_length / output_length 划分输入输出。hash_ids_scope 必须为
input_output，hash_block_size 必须与 traffic.source.block_size 一致。
当前转换脚本使用 16-token 哈希段，原生 KV 块仍为 64 token。

- Rust 加载与驱动回归：230 项通过。
- Rust 盘框回归：3 项通过，包括输入输出混合块、D→P 命中、不同前缀及 LRU 淘汰的双格式等价。
- Python 转换器测试：3 项通过。
- 实际 H200 GLM-5.2 TP8/EP8 盘框配置：同一批真实前 200 条请求，两组全部完成；
  逐请求报告和模拟汇总指标一致，详情见 validation.json。
- 比较排除随机 UUID、实际 wall clock 及其派生程序运行速度、性能模型来源描述；
  数值容差绝对 1e-9、相对 1e-12。
- 原 JSONL 29,158,769 字节，新 JSONL 6,042,409 字节，减少约 79.3%。
- 扩展已重新编译并安装至 /home/xieqiang/.venvs/aisim。

全量集成测试目标的额外编译检查仍遇到仓库原有夹具缺少 enclosure 字段的问题，
本次未更改这些夹具或 ReplayTopology 定义。上面的相关回归测试已通过。
本验证不代表全量数据已转换或已验证全量回放的峰值内存。

转换入口：../../scripts/compact_swe_chat_hashes.py
实际模型验证入口：../../scripts/validate_compact_hashes.py
转换格式与完整命令见 ../../data_process.md。
