# TNSE 全部论文图：四阶段顺序清单

这份清单把论文实验分成四个互斥阶段。先判断自己处在哪一阶段，只运行该阶段及其后续步骤；不要因为想画图而重新训练模型。

当前正式状态（2026-09-02）已经处于 **D：测试完成，只画图**。Chapter IV 的数据、模型和 checkpoint 冻结；Chapter V 使用 `conflict_graph_local_coordination_v1` 的正式结果。最新版统一出图目录为：

```text
results/paper/tnse_current_plots/
├── figures/       # Fig. 1--7 的 EPS/JPG
└── data/plotting/ # 每张图对应的 CSV
```

为避免 Matplotlib 和 uv 缓存写入用户目录，下面命令均可加此前缀：

```bash
MPLCONFIGDIR=/private/tmp/gnn_mpl_cache \
UV_CACHE_DIR=/private/tmp/gnn_scheduling_uv_cache \
uv run python -u <script> <arguments>
```

## A. 没有训练数据集

目的：生成并核对第四章静态 Rician 数据。第五章队列实验的拓扑、信道和到达轨迹由评测器按 seed 生成，不需要另建 queue pickle。

```bash
uv run python -u scripts/prepare_paper_datasets.py \
  --num-uavs 50,40,30,20 \
  --train-size 300 --val-size 128 --test-size 128
```

已有匹配数据时脚本只做 metadata/fingerprint 校验，不重写文件。不要使用 `--force`，除非明确决定生成一套新数据。

## B. 有数据集，但没有 checkpoint

目的：只训练第四章 edge-enhanced GAT。当前仓库已有正式 checkpoint，因此这组命令仅供缺失模型时恢复，不应在日常出图前运行。

Fig. 2/3 及 Fig. 1 共用的 300-snapshot、50-epoch、三 seed 模型：

```bash
uv run python -u paper_figures/fig02_training_curves/run.py \
  --stage train --nodes 20,30,40,50 --seeds 0,1,2 --epochs 50
```

Fig. 4 的不同训练样本量模型（正式图使用 n=10--300）：

```bash
uv run python -u paper_figures/fig04_data_efficiency/run.py \
  --stage train --nodes 20,30,40,50 \
  --sizes 10,20,30,40,60,80,100,200,300 \
  --seeds 0,1,2 --epochs 50
```

Fig. 7 Chapter IV 的 matched-CSI 模型：

```bash
uv run python -u paper_figures/fig07_csi_feedback/resume.py \
  --stage train --chapters 4 --nodes 20,30,40,50 --epochs 50
```

`--stage train` 不进行测试；Chapter V 没有额外训练过程，而是复用这些 matched-CSI checkpoint。

## C. 有 checkpoint，但没有测试结果

目的：保持模型冻结，只生成 JSON/NPZ 评测结果。正式结果已经存在；仅当文件缺失、fingerprint 不匹配或算法版本改变时运行。

```bash
# Fig. 1：满缓存物理层比较
uv run python -u paper_figures/fig01_full_buffer/run.py \
  --stage evaluate --nodes 20,30,40,50 --samples 128

# Fig. 2：已有 checkpoint 的六种评测产物
uv run python -u paper_figures/fig02_training_curves/run.py \
  --stage evaluate --nodes 20,30,40,50 --seeds 0,1,2 --epochs 50

# Fig. 3：只核对并同步 Fig. 2 的 training.json，不训练
uv run python -u paper_figures/fig03_activation_audit/run.py

# Fig. 4：已有 checkpoint 的 soft/hard/projected 评测
uv run python -u paper_figures/fig04_data_efficiency/run.py \
  --stage evaluate --nodes 20,30,40,50 \
  --sizes 10,20,30,40,60,80,100,200,300 \
  --seeds 0,1,2 --epochs 50

# Fig. 5：Chapter V 正式队列仿真（耗时，只有结果缺失时才运行）
uv run python -u paper_figures/fig05_dynamic_queue/run.py \
  --nodes 50 --slots 10000 --seeds 42 --deadline-ms 2000 \
  --output-root results/paper/ch5_local_coordination_v1/fig05_rerun

# Fig. 6：冻结 GAT 的跨规模测试
uv run python -u paper_figures/fig06_cross_n/run.py --train_size 300

# Fig. 7：不训练；Chapter IV 缺失结果和 Chapter V 正式结果评测
uv run python -u paper_figures/fig07_csi_feedback/resume.py \
  --stage evaluate --chapters 4,5 --nodes 20,30,40,50 \
  --slots 10000 --seeds 42 --arrival-rate 2.4 \
  --chapter5-output-root results/paper/ch5_local_coordination_v1/fig07_rerun
```

Fig. 5/7 的重跑目录故意与已经审计过的正式结果分开。完成后先运行 `experiments/audit_chapter5_results.py`，再决定是否替换正式 raw root。

## D. 测试完成，只画图

目的：只读取现有结果，把所有当前论文图写入统一隔离目录。下面是当前推荐直接顺序运行的命令。

```bash
PLOT_ROOT=results/paper/tnse_current_plots

MPLCONFIGDIR=/private/tmp/gnn_mpl_cache uv run python -u \
  paper_figures/fig01_full_buffer/plot.py \
  --raw-dir paper_figures/fig01_full_buffer/data/raw --output-root "$PLOT_ROOT"

MPLCONFIGDIR=/private/tmp/gnn_mpl_cache uv run python -u \
  paper_figures/fig02_training_curves/plot.py \
  --raw-root paper_figures/fig02_training_curves/data/raw --output-root "$PLOT_ROOT"

MPLCONFIGDIR=/private/tmp/gnn_mpl_cache uv run python -u \
  paper_figures/fig03_activation_audit/plot.py \
  --source paper_figures/fig02_training_curves/data/raw --output-root "$PLOT_ROOT"

MPLCONFIGDIR=/private/tmp/gnn_mpl_cache uv run python -u \
  paper_figures/fig04_data_efficiency/plot.py \
  --raw-root paper_figures/fig04_data_efficiency/data/raw \
  --sizes 10,20,30,40,60,80,100,200,300 --output-root "$PLOT_ROOT"

MPLCONFIGDIR=/private/tmp/gnn_mpl_cache uv run python -u \
  paper_figures/fig05_dynamic_queue/plot.py \
  --profile deadline_2000ms --nodes 50 \
  --raw-root results/paper/ch5_local_coordination_v1/fig05_merged_N50_seed42 \
  --output-root "$PLOT_ROOT" --metrics backlog,delay,goodput

MPLCONFIGDIR=/private/tmp/gnn_mpl_cache uv run python -u \
  paper_figures/fig06_cross_n/plot.py \
  --input paper_figures/fig06_cross_n/data/raw/cross_n_trainSize=300_seed0_epoch50.json \
  --output-root "$PLOT_ROOT"

MPLCONFIGDIR=/private/tmp/gnn_mpl_cache uv run python -u \
  paper_figures/fig07_csi_feedback/plot.py \
  --nodes 20,30,40,50 --chapters 4,5 \
  --chapter4-raw-root paper_figures/fig07_csi_feedback/data/raw \
  --chapter5-raw-root results/paper/ch5_local_coordination_v1/fig07_full \
  --output-root "$PLOT_ROOT"
```

这组命令不会训练、仿真或改写 `paper_figures/*/figures/` 中的历史图。论文实际引用的 EPS 需要从 `$PLOT_ROOT/figures/` 明确复制到论文的 `figures/` 目录，复制前应先备份旧图。

## 最后核验

```bash
uv run python -m unittest discover -s tests -v

uv run python -u experiments/audit_chapter5_results.py \
  --root results/paper/ch5_local_coordination_v1 \
  --output-prefix results/validation/chapter5_final_audit

find results/paper/tnse_current_plots -type f \
  \( -name '*.eps' -o -name '*.jpg' -o -name '*.csv' \) -print | sort
```

规范测试入口是 `unittest discover -s tests`，不是裸 `pytest`。后者会把归档审稿包和第三方 baseline 的旧环境测试一并收集，不能代表当前主仓库是否通过。
