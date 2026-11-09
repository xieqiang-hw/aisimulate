对于盘框这个场景建模分析：

1. 盘框拥有160GB/s的有效带宽，4个1823网卡，8个200GE的网口。24块盘，24 × 6.984TB NVMe SSD，单盘带宽8GB/s
2. P节点本地内存，24个通道DDR4，3200MHz，容量384GB，带宽614GB/s带宽
3. Ascend 910B3 × 8 / 节点，单卡 64 GB HBM，H2D带宽，单卡25GB，总带宽=200GB/s

在agent场景下，在GLM-5.2-FP8/minimax_m2.5模型下，1P和1D节点，共享一个盘框。 分析下910B下，

场景1：本地内存+本地SSD，采用LRU淘汰策略， 如果内存容量降低逐步降低、或增加，对于DRAM命中率逐步下降，频繁换入换出，需要SSD的加载带宽是多少？ 场景2：KV盘框，采用分流的策略， DRAM按照LRU保留前缀命中KV，盘框保存agent全量KV；分析下DRAM带宽和KV盘框带宽？

对比，预期结论。场景1当中，本地内存需要频繁换入换出，SSD的带宽比较高；场景2，相同性能，盘框带宽足够，DRAM容量需求低

---

# 盘框与分层 KV 缓存建模实验

本项目使用 AISimulate 分析 agent 负载下 KV 缓存容量、搬运带宽与服务时延的关系，验证上述预期结论。当前先研究场景1，采用 GLM-5.2-FP8、H200、1P1D、每实例 TP8 验证方法；尚未实现昇腾适配或远端盘框模型。P→D 保持直接交接。P 本地 G3 的功能实验见 [实验索引](experiments.md)。

## experiments 目录

```text
experiments/
└── disk/
    ├── README.md                    # 项目背景、范围及目录结构
    ├── experiments.md               # 实验目的、配置和运行入口
    ├── configs/                    # 每个实验一份完整 YAML
    │   ├── hbm_only.yaml
    │   ├── p_g2_shared.yaml
    │   ├── pd_g2_shared.yaml
    │   ├── p_g3_local_validation.yaml
    │   ├── p_g3_local.yaml
    │   └── p_g3_local_{4x,16x,64x,128x}.yaml
    ├── scripts/                    # 数据转换及仿真入口
    │   ├── prepare_freeinference.py
    │   ├── run_experiment.py
    │   ├── run_g2_experiments.py
    │   ├── validate_p_g3_local.py
    │   ├── run_p_g3_local.py
    │   └── run_p_g3_capacity_sweep.py
    └── analysis/                   # 按实验标识组织分析
        ├── hbm_only/
        ├── p_g2_shared/
        └── pd_g2_shared/
```

每个分析目录包含 `README.md`、一个 `analysis.ipynb`，以及 `assets/` 子目录（存放 Notebook 生成的汇总 JSON、CSV 和 PNG）。实验标识与 YAML、分析目录名称一致。原始大报告保留在 `~/project/results/`，原始及转换数据保留在 `~/project/data/`。

实验运行方式见 [experiments.md](experiments.md)。分析时用 VS Code 打开相应 `analysis.ipynb`，选择 `~/.venvs/aisim/bin/python` 内核并运行全部单元。若提示缺少内核，在该环境安装 `ipykernel`：

```bash
uv pip install --python ~/.venvs/aisim/bin/python \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple ipykernel
```

Notebook 自动选择最新完成的全量运行；可显式设置 `RUN_DIR`，或设置 `REFRESH=True` 强制重读。对比图使用其他实验已经生成的汇总，首次应将三个 Notebook 各运行一次，再运行绘图单元以包含全部对照。独立的实验结论写在各目录 README 中，重新运行不会覆盖手写说明。

## 待办（场景1）

- [x] 全量 HBM-only、仅 P 开 G2、P/D 共池 G2 对照。
- [ ] 定位极端生成停顿，确定正式 SLA 与目标达标比例。
- [x] 接入 P 本地 G3，验证写入 SSD、经 DRAM 读回及 P→D 交接。
- [ ] 核对 G3 容量与 LRU 在实际负载下的表现。
- [ ] 扫描 DRAM 容量与 SSD 读写带宽，分析 SLA、换入换出及回退重算。

分析目录内的数据和图统一放在各实验的 `assets/` 子目录；外层仅保留 README 和 Notebook，README 内嵌图表并链接数据。

缓存路径和各层行为见 [KV 缓存模式说明](kv_cache_modes.md)。
