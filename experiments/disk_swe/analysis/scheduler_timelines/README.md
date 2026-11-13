# P/D 调度时间线

HTML 已覆盖 disk_swe/experiments.html，汇总数据来自 experiments.md。逐轮图表仅使用当前 轮次已完成的原始观测；旧实验数据不混入。

## 图表与口径

- 缓存保留每小时基础汇总，供辅助指标使用；BS 主图采用下文的自适应缩放。小时缓存保留 BS 峰值、执行期间按计算时长加权的平均 BS、计算繁忙时长、token 数、队列与 KV 的采样最大值，以及抢占和重新准入计数。
- 每个阶段选择累计计算时长最多的一分钟，提供同一窗口内 P/D 的原始逐轮 BS、Prefill 新计算 token 数和 Decode 耗时／KV 上下文。
- 逐轮 BS 仅画 compute_start 到 end 的计算段；空闲不延续上一次 BS。最忙分钟不一定是 BS 峰值所在分钟。
- 小时状态先按 iteration_id 还原每轮值，再取窗口最大值；状态未变化也会正确归入后续窗口，没有调度采样的桶留空。
- 队列、恢复和 HBM 是调度采样状态；普通等待不证明 KV 已就绪，各类等待存在重叠。计算繁忙占比不是实测 GPU 利用率。
- 为限制浏览器数据体积，单个原始窗口每个阶段最多展示 50,000 条；若触发上限，页面明确提示。完整原始记录不受影响。

可切换 Prefill／Decode BS、Prefill 新计算 token、Decode 每轮计算耗时、Decode KV 总长度／最大长度、等待队列、H2D／SSD 恢复、HBM KV 占用、计算繁忙占比、抢占与重新准入。支持时间范围输入和当前图表 CSV 导出。

## 文件与刷新

- assets/<轮次>/<实验名>.json：每组流式汇总缓存，记录源文件指纹及导出版本。
- scripts/trace_summary_tool：独立 Rust 流式数据汇总器，不修改引擎或仿真结果。
- scripts/dashboard_timelines.py：缓存、数据源与实验标签。
- scripts/dashboard_timeline.html / .css / .js：内嵌页面的时间线界面。
- scripts/dashboard_bs_zoom.js：按需加载精细数据、解码BS计算段和自适应聚合。
- scripts/build_results_dashboard.py：保留原有 SLA／时延／复用对比，并生成完整 HTML。
- `~/project/tmp/refresh_results_dashboard.py` refreshes the experiment summary HTML.

上述脚本路径相对于 disk_swe。浏览器刷新读取最新页面；仅导入 Markdown 只替换汇总表，不更新页面内嵌的时间线。

## 验证

使用既有小型 P/D 恢复夹具核对：11 轮记录、P 新计算 387 token、D 生成 6 token、P/D 峰值 BS 都为 1，与逐轮记录一致。另检查跨小时计算段仍计入没有新迭代开始的桶。浏览器核对覆盖所有指标、原始／聚合窗口、CSV 导出及桌面／手机布局。

使用下方 Notebook 可以查看缓存、重画小时 BS 峰值与执行期间均值。所有大规模原始记录仍保留在 results 目录。


## 全程缩放

BS 主图默认采用自适应繁忙占比面积图，分辨率按视窗自动选择；支持横向拖选、放大／缩小按钮、双击缩小与 P/D 同步。缩到 5 分钟以内改画实际 BS 计算段，空闲留白。右轴 BS 峰值和固定 0–100% 繁忙度轴均为可选。

精细 BS 按相邻且 BS 相同的计算区间合并，时间显示精度为 1 微秒。不是请求抽样；原始逐轮记录仍保留。核对夹具及真实基线的峰值、计算总时长和空闲间隔，见 assets/zoom_validation.json。

每组新增“实验名.bs.js”，保存压缩后的紧凑计算段，选择实验时从本地按需加载并解码。HTML 继续内嵌概览和预置分钟数据；完整精细缩放依赖其相对路径下的 analysis/scheduler_timelines/assets 数据包。移动 HTML 时需要同时保留该目录结构。所有数据都可离线加载，不使用网络服务。

辅助 token／时延／状态图仍使用小时概览或预置最忙分钟的完整逐轮数据；任意分钟缩放只保证 BS 主图及由其计算的繁忙度具有精细数据。低于辅助统计精度时页面会明确提示，不用小时峰值冒充瞬时值。
