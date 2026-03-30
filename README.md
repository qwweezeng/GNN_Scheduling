# GNN-Scheduling — 基于图神经网络的无人机密集集群链路调度

本项目是论文《GNN-based Dynamic Link Scheduling for Dense UAV Swarm Networks》的 PyTorch 仿真实现。

**研究背景：** 在大规模无人机密集集群自组网（FANET）中，大量链路并发争用无线信道资源，带来严重碰撞和干扰，使得网络资源难以高效利用。本项目提出基于图注意力网络（GAT）的分布式链路调度方案，目标是：**在每个时隙决策激活哪些链路，使全网效用（总吞吐量）最大化。**

**核心思路：** 将无人机网络中的通信链路建模为**冲突图（Conflict Graph）**：

- **节点** = 通信链路
- **边** = 链路间的干扰关系

通过 GAT 迭代地从局部节点特征（邻居干扰信息）学习整体网络态势，输出每条链路的激活概率。

---

## 安装与运行

### 环境准备

```bash
# 安装 uv（推荐，自动管理虚拟环境）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖（自动创建 .venv）
uv pip install -r pyproject.toml

# 或直接用 pip
pip install torch numpy matplotlib networkx tensorboard pyyaml pandas openpyxl
```

## 实验一览


| 实验           | 命令                                 | 主要用途                           |
| ---------------- | -------------------------------------- | ------------------------------------ |
| 单次训练与评估 | `bin/run_experiment.py`              | 训练 GAT 模型，输出 test utility   |
| 收敛性分析     | `experiments.convergence_analysis`   | 研究不同训练样本量对性能的影响     |
| 跨尺度扩展性   | `experiments.scaling_experiment`     | 研究模型在不同 numUAV 间的泛化能力 |
| CSI 不完美性   | `experiments.csi_imperfection_study` | 研究信道量化精度对调度性能的影响   |
| 超参数网格搜索 | `experiments.grid_search_lambda`     | 搜索稀疏性/冲突惩罚的最优权重      |
| 画图           | `utils.plot_paper`                   | 生成所有论文图表                   |
| TensorBoard    | `tensorboard --logdir=./logs/`       | 可视化训练曲线                     |

---

## 1. 单次训练与评估

### 目的

训练一个 GAT 调度模型，在指定 numUAV 网络拓扑上学习最大化全网效用，输出 test utility 并与基线算法比较。

### 常用参数


| 参数                | 说明                                              | 示例                       |
| --------------------- | --------------------------------------------------- | ---------------------------- |
| `--config`          | YAML 配置文件路径                                 | `configs/exp_example.yaml` |
| `--seed`            | 随机种子（用于复现）                              | `42`, `10086`              |
| `--dry-run`         | 小样本快速验证（不保存结果）                      | flag                       |
| `--gpu-id`          | 指定 GPU 编号                                     | `0`, `1`                   |
| `--csi-mode`        | 信道模式：`perfect` / `large_scale` / `quantized` | `perfect`                  |
| `--q-bits`          | 量化比特数（仅 quantized 模式生效）               | `4`, `8`                   |
| `--train-size`      | 训练样本数                                        | `500`, `1024`              |
| `--lambda-sparsity` | 稀疏性惩罚权重                                    | `0.1`                      |
| `--group`           | 实验组名称（决定 Excel sheet）                    | `csi_imperfection`         |
| `--notes`           | 实验备注                                          | `"baseline run"`           |

### 输出文件

- `results/models/`: 训练好的模型权重 `.pth`
- `results/metrics/`: 每次运行的指标 CSV
- `results/reports/`: 文本分析报告
- `logs/`: TensorBoard 日志、模型 checkpoint
- `results/all_experiments_summary.xlsx`: 所有实验的汇总 Excel

### 示例

```bash
# 快速验证配置（不保存，10 条样本）
uv run python bin/run_experiment.py --config configs/exp_example.yaml --seed 42 --dry-run

# 正式训练
uv run python bin/run_experiment.py --config configs/exp_example.yaml --seed 42

# 训练 numUAV=50，量化 CSI，4 bit
uv run python bin/run_experiment.py --config configs/exp_example.yaml \
    --num-uav 50 --csi-mode quantized --q-bits 4 --seed 42

# 指定实验组（结果写入 Excel Sheet3）
uv run python bin/run_experiment.py --config configs/exp_example.yaml \
    --group csi_imperfection --notes "quantized 4bit run" --seed 42
```

---

## 2. CSI 不完美性实验（`csi_imperfection_study`）

### 目的

研究信道状态信息（CSI）精度对调度性能的影响。分别测量：


| CSI 模式      | 说明                                     | 在图中的表现              |
| --------------- | ------------------------------------------ | --------------------------- |
| `perfect`     | 完美 CSI，含小尺度衰落                   | 最顶层虚线（upper bound） |
| `large_scale` | 仅大尺度衰落（路径损耗），无小尺度随机性 | 中层虚线                  |
| `quantized`   | 量化距离近似（大尺度 + 量化精度 loss）   | 实线，随 q_bits 变化      |

填充逻辑：对 numUAV = {20, 30, 40, 50} 的每条曲线，阴影填充范围为 `[test utility, min(next numUAV utility, upper bound)]`。

### 常用参数


| 参数               | 说明                   | 默认值                     |
| -------------------- | ------------------------ | ---------------------------- |
| `--num-uav`        | 基础无人机数量         | `20`                       |
| `--quantized-bits` | 要测试的量化比特数列表 | `2 4 6 8 10`               |
| `--config`         | 配置文件路径           | `configs/exp_example.yaml` |
| `--gpu-id`         | GPU 编号               | `0`                        |

### 输出文件

- `results/all_experiments_summary.xlsx`（Sheet3）：数值结果汇总
- `results/img/csi_imperfection.png / .eps`：论文图表

### 示例

```bash
# 默认跑 numUAV=20 的所有模式（perfect / large_scale / 各量化档位）
uv run python experiments/csi_imperfection_study.py --gpu-id 0

# 跑 numUAV=50，测试 2/4/6/8 bit 量化
uv run python experiments/csi_imperfection_study.py --num-uav 50 --quantized-bits 2 4 6 8

# 仅生成图表（数据已有时）
uv run python -m utils.plot_paper --plot csi
```

> **注意：** 脚本目前已实现 `perfect` 和 `large_scale` 的自动化跑实验；若要跑 `quantized` 模式，需在脚本中取消注释对应的 `for qb in args.quantized_bits` 循环。

---

## 3. 跨尺度扩展性实验（`scaling_experiment`）

### 目的

研究在 numUAV=X 上训练的模型能否泛化到 numUAV=Y；以及混合多尺度数据训练是否能提升跨尺度泛化能力。

### 常用参数


| 参数                    | 说明                                                       | 默认值           |
| ------------------------- | ------------------------------------------------------------ | ------------------ |
| `--mode`                | 实验模式：`paired`（配对）/ `mixed`（混合）/ `all`（全部） | 必选             |
| `--use-existing-models` | 复用已有模型（不重新训练）                                 | flag             |
| `--model-library`       | 模型库目录                                                 | `results/models` |
| `--quick`               | 减少训练 epoch / 样本量（快速调试）                        | flag             |
| `--gpu-id`              | GPU 编号                                                   | `0`              |

### 输出文件

- `results/img/scaling_paired.png`：配对实验图（train on X → test on Y）
- `results/img/scaling_mixed.png`：混合训练实验图

### 示例

```bash
# 配对泛化实验（已训练模型直接测）
uv run python -m experiments.scaling_experiment --mode paired --use-existing-models

# 混合训练实验
uv run python -m experiments.scaling_experiment --mode mixed

# 全部实验 + 快速模式
uv run python -m experiments.scaling_experiment --mode all --quick

# 指定模型库路径
uv run python -m experiments.scaling_experiment --mode paired \
    --model-library ./results/models --use-existing-models
```

---

## 4. 收敛性分析（`convergence_analysis`）

### 目的

研究不同训练样本量（train_size）对模型性能的影响，确定达到稳定性能所需的最少样本数。

### 常用参数


| 参数       | 说明     | 默认值 |
| ------------ | ---------- | -------- |
| `--gpu-id` | GPU 编号 | `0`    |

> 样本量列表在脚本内硬编码（`train_sizes = [20, 80]`），如需修改请直接编辑 `experiments/convergence_analysis.py` 第 138 行。

### 输出文件

- `results/all_experiments_summary.xlsx`（Sheet2）：不同 train_size 下的 test utility
- `results/img/convergence.png`：收敛曲线图

### 示例

```bash
uv run python -m experiments.convergence_analysis --gpu-id 0
```

---

## 5. 超参数网格搜索（`grid_search_lambda`）

### 目的

在 numUAV=50 上，对 `LAMBDA_SPARSITY` 和 `LAMBDA_CONFLICT` 进行网格搜索，找到最优的正则化权重组合。

### 常用参数


| 参数       | 说明                | 默认值                                 |
| ------------ | --------------------- | ---------------------------------------- |
| `--config` | 基础配置文件        | 必选                                   |
| `--out`    | 汇总 Excel 输出路径 | `experiments/grid_lambda_results.xlsx` |
| `--seed`   | 随机种子            | `42`                                   |
| `--gpu-id` | GPU 编号            | `0`                                    |

### 输出文件

- `experiments/grid_lambda_results.xlsx`：各 (λ_sparsity, λ_conflict) 组合的 test utility

### 示例

```bash
uv run python experiments/grid_search_lambda.py \
    --config configs/exp_example.yaml --seed 42 --gpu-id 0
```

---

## 6. 可视化（`utils.plot_paper`）

### 目的

根据已有实验结果数据，生成论文所需的高质量图表（EPS + PNG）。

### 常用参数


| 参数            | 说明                                                                 | 默认值         |
| ----------------- | ---------------------------------------------------------------------- | ---------------- |
| `--plot`        | 图表类型：`all` / `baseline` / `training` / `samples` / `csi`        | `all`          |
| `--results-dir` | 结果目录                                                             | `results`      |
| `--output-dir`  | 图片输出目录（覆盖 results-dir 默认值）                              | `results/img/` |
| `--format`      | 输出格式：`eps` / `png` / `both`                                     | `both`         |
| `--colors`      | 配色方案：`v1` / `v2` / `v3`（覆盖 `utils/plot_config.py` 全局配色） | 默认           |
| `--show`        | 显示图形窗口（需图形界面）                                           | 不显示         |

配色方案预览（`--colors v1/v2/v3` 均定义在 `utils/plot_paper.py` 第 771-791 行）：

- **v1**：深蓝 / 浅蓝 / 浅红 / 深红
- **v2**：蓝 / 紫 / 粉 / 红
- **v3**：淡蓝 / 深蓝 / 暗红 / 橙红

### 各图表说明


| `--plot`   | 内容                                        | 输出文件                          |
| ------------ | --------------------------------------------- | ----------------------------------- |
| `baseline` | GAT vs 各基线算法的吞吐量柱状图             | `baseline_comparison.png/.eps`    |
| `training` | train_loss 与 val_utility 随 epoch 变化曲线 | `training_*.png/.eps`             |
| `samples`  | test utility 随训练样本量变化曲线           | `performance_vs_samples.png/.eps` |
| `csi`      | CSI 不完美性：量化精度 vs test utility      | `csi_imperfection.png/.eps`       |

### 示例

```bash
# 生成所有图表（EPS + PNG）
uv run python -m utils.plot_paper --plot all

# 仅画 CSI 不完美性图（PNG 格式）
uv run python -m utils.plot_paper --plot csi --format png

# 单独画基线对比图
uv run python -m utils.plot_paper --plot baseline

# 使用 v2 配色方案
uv run python -m utils.plot_paper --plot all --colors v2

# 指定输出目录
uv run python -m utils.plot_paper --plot csi --output-dir ./figures/
```

---

## 项目结构

```
GNN_Scheduling/
├── bin/
│   └── run_experiment.py        # CLI 统一入口（推荐方式）
│
├── configs/
│   └── exp_example.yaml         # 所有超参数集中管理
│
├── models/
│   └── model.py                 # GATScheduler、calculate_achievable_rate
│
├── experiments/
│   ├── runner.py                # ExperimentRunner：实验编排器
│   ├── csi_imperfection_study.py # CSI 不完美性实验
│   ├── scaling_experiment.py    # 跨尺度扩展性实验
│   ├── convergence_analysis.py  # 收敛性分析
│   ├── grid_search_lambda.py     # 超参数网格搜索
│   └── compare_*.py             # 辅助对比脚本
│
├── train/
│   └── trainer.py               # 训练循环、梯度裁剪、早停
│
├── utils/
│   ├── baseline.py              # 基线算法（Random / Greedy / FPLinQ / Spatial DL）
│   ├── network_generate.py      # 数据生成器
│   ├── plot_paper.py            # 论文图表绘制
│   ├── plot_config.py           # 配色方案、全局画图风格
│   └── io.py                    # 文件 I/O、Excel 读写
│
├── results/
│   ├── models/                  # 训练好的 .pth 模型文件
│   ├── metrics/                 # 指标 CSV
│   ├── reports/                 # 文本报告
│   ├── img/                     # PNG / EPS 图表
│   └── all_experiments_summary.xlsx # 所有实验汇总
│
└── logs/
    └── tensorboard/             # TensorBoard 训练日志
```

---

## 关键模块说明

### models/model.py — GATScheduler


| 参数             | 说明                                |
| ------------------ | ------------------------------------- |
| `raw_node_dim`   | 原始节点特征维度（只用 SNR 时为 1） |
| `embed_node_dim` | 节点嵌入维度                        |
| `edge_dim`       | 边特征维度（干扰权重）              |
| `num_heads`      | 多头注意力头数                      |
| `T_iterations`   | GAT 消息传递轮数（层数）            |
| `gat_out_dim`    | GAT 层总输出维度                    |

**输入：** `[num_links, raw_node_dim]` 节点特征 + `[2, num_edges]` 边索引 + `[num_edges, 1]` 边特征
**输出：** `[num_links, 1]` 每条链路的激活概率 `pi_v ∈ [0,1]`

### utils/baseline.py — 基线算法


| 算法          | 说明                                |
| --------------- | ------------------------------------- |
| Random        | 随机选择链路                        |
| Simple Greedy | 贪心选 SNR 最高的链路               |
| Local Greedy  | 局部干扰感知的贪心调度              |
| FPLinQ        | 分数规划调度算法（Cui et al. 2019） |
| Spatial DL    | 空间深度学习方法（Cui et al. 2019） |

### configs/exp_example.yaml — 关键参数


| 参数                                | 说明                                     |
| ------------------------------------- | ------------------------------------------ |
| `NUM_UAV`                           | 无人机/链路数量                          |
| `CSI_MODE`                          | `perfect` / `large_scale` / `quantized`  |
| `Q_BITS`                            | 量化比特数（quantized 模式生效）         |
| `TRAIN_SIZE / VAL_SIZE / TEST_SIZE` | 数据集划分                               |
| `EPOCHS`                            | 训练轮数                                 |
| `LR`                                | 学习率                                   |
| `LAMBDA_SPARSITY`                   | 稀疏性惩罚（防止"全部激活"局部最优）     |
| `LAMBDA_CONFLICT`                   | 冲突惩罚（直接惩罚高干扰链路对同时激活） |
| `NUM_GAT_LAYERS`                    | GAT 层数（消息传递迭代次数）             |
| `SEED`                              | 随机种子（实验复现）                     |

---

## 训练监控

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs/
```

浏览器打开 `http://localhost:6006`，查看：

- `train_loss`：训练 loss 曲线
- `val_utility`：验证集效用曲线
- `pi_mean` / `pi_std`：链路平均激活概率及标准差

---

## 注意事项

1. **参数管理**：所有超参数集中于 `configs/exp_example.yaml`，不要直接改代码。
2. **实验复现**：务必使用 `--seed` 设置随机种子，相同 seed + 相同配置 = 相同结果。
3. **数据集缓存**：`DATASET_CACHE_PATH` 控制缓存路径；修改 `NUM_UAV`、`CSI_MODE` 或 `Q_BITS` 时应更新路径避免覆盖。
4. **GPU 选择**：多卡环境下用 `--gpu-id` 指定；`CSI_MODE=large_scale` 数据生成最快，`quantized` 次之，`perfect` 最慢（因为包含小尺度衰落）。
5. **Excel 汇总**：`ExperimentRunner` 自动将每次运行结果追加到 `results/all_experiments_summary.xlsx`；各实验组写入不同 sheet：
   - Sheet1：单次训练（`EXPERIMENT_GROUP=base`）
   - Sheet2：收敛性分析
   - Sheet3：CSI 不完美性实验
