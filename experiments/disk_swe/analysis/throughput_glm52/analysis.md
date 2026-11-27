# GLM-5.2 吞吐与执行观测

报告：[throughput_glm52.html](../../throughput_glm52.html)。

页面展示 8×／16×／32× 各一组：P DRAM 均为 256 GB、SSD 读 2 GB/s/TP、speedup=1。8× 的 64／128 GB 配置不再展示，原始结果保留。单轮预算 16,384 token，写带宽 8 GB/s/TP。

8× 基线快照来自原报告，经原始结果与调度轨迹对账。16×／32× 来自调度轨迹补测，各档 requests_*x.npz 保存逐请求时间与长度，execution_*x.json 保存原生执行聚合。每档独立确定主要负载、全部到达和高峰窗口，按 5 分钟聚合。P/D 忙碌比例不是 GPU 硬件利用率。

生成命令：

    ~/.venvs/aisim/bin/python experiments/disk_swe/scripts/prepare_throughput_glm52_16x.py
    ~/.venvs/aisim/bin/python experiments/disk_swe/scripts/prepare_throughput_glm52_32x.py
    ~/.venvs/aisim/bin/python experiments/disk_swe/scripts/build_throughput_dashboard.py

原 8× 上额外 speedup=2／4／8 的九组结果及相关缓存已按要求删除，原始叠加数据包不删除。

页面采用连续滚轮缩放和自由拖选，最小范围保留一个 5 分钟桶；悬停使用绘图缓存。第三节顺序为 Decode 生成速率、BS、P/D 执行忙碌占比。
