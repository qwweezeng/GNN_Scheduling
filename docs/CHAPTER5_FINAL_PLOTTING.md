# Chapter V 最终检查与绘图清单

本清单只检查和绘制已经完成的 Chapter V 结果，不重新训练 GAT，也不重新运行 10000-slot 正式仿真。所有命令均从仓库根目录顺序执行。

## 1. 检查代码功能

```bash
uv run python -m unittest discover -s tests -v
```

该命令会运行 `tests/` 中的小型自动化例子。它不生成论文性能曲线，也不重新训练模型；它用于快速检查局部 marginal、邻域确认、endpoint feasibility、GAT block cache、observed/true CSI 边界以及队列守恒等代码规则是否仍然成立。

不要在仓库根目录直接运行裸 `pytest -q`：它会继续扫描归档审稿包和第三方 baseline 目录，把那些需要独立旧环境的 `*_test.py` 也收集进来。当前项目的规范入口是上面的 `unittest discover -s tests`，只运行主仓库维护的测试集。

预期：所有测试通过，命令退出码为 0。

## 2. 审计正式结果

```bash
uv run python -u experiments/audit_chapter5_results.py \
  --root results/paper/ch5_local_coordination_v1 \
  --output-prefix results/validation/chapter5_final_audit
```

预期：`status=PASS`，53/53 checks passed。该步骤检查正式 JSON/NPZ 的完整性、协议版本、文件哈希、局部效用、可行性、projection 和守恒误差。

## 3. 生成新版 Fig. 5

```bash
MPLCONFIGDIR=/private/tmp/gnn_mpl_cache \
uv run python -u paper_figures/fig05_dynamic_queue/plot.py \
  --profile deadline_2000ms \
  --nodes 50 \
  --raw-root results/paper/ch5_local_coordination_v1/fig05_merged_N50_seed42 \
  --output-root results/paper/tnse_current_plots \
  --metrics backlog,delay,goodput
```

输出位于：

```text
results/paper/tnse_current_plots/
├── figures/
│   ├── fig05_backlog_N50_deadline_2000ms.{eps,jpg}
│   ├── fig05_delay_N50_deadline_2000ms.{eps,jpg}
│   └── fig05_goodput_N50_deadline_2000ms.{eps,jpg}
└── data/plotting/fig05_deadline_2000ms_summary.csv
```

## 4. 生成完整 Fig. 7

该命令同时读取冻结的 Chapter IV JSON 和新版 Chapter V JSON，只作图，不训练或仿真。

```bash
MPLCONFIGDIR=/private/tmp/gnn_mpl_cache \
uv run python -u paper_figures/fig07_csi_feedback/plot.py \
  --nodes 20,30,40,50 \
  --chapters 4,5 \
  --chapter4-raw-root paper_figures/fig07_csi_feedback/data/raw \
  --chapter5-raw-root results/paper/ch5_local_coordination_v1/fig07_full \
  --output-root results/paper/tnse_current_plots
```

输出位于：

```text
results/paper/tnse_current_plots/
├── figures/
│   ├── fig07_chapter4_csi_rate.{eps,jpg}
│   └── fig07_chapter5_gat_mp_backlog.{eps,jpg}
└── data/plotting/
    ├── fig07_chapter4_csi_rate.csv
    └── fig07_chapter5_gat_mp_backlog.csv
```

## 5. 检查输出文件

```bash
find results/paper/tnse_current_plots \
  -type f \( -name '*.eps' -o -name '*.jpg' -o -name '*.csv' \) \
  -print | sort
```

应列出 Fig. 5 的 3 组 EPS/JPG、Fig. 7 的 2 组 EPS/JPG，以及三个 plotting CSV。由于使用了独立 `--output-root`，原有 `paper_figures/*/figures/` 不会被覆盖。
