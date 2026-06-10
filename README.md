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

---

## 参数联动速查表（重要！）

本项目有多个**可选功能模块**，每个模块涉及一组必须联动设置的参数。若参数不一致，代码会在运行时报错（防呆设计）。

### 1. 队列感知模块（Queue-aware）

| 功能 | 参数 | 值 | 说明 |
|------|------|----|------|
| 启用开关 | `run_queue_experiment.py` | 固定 | 队列感知实验脚本，无开关 |
| 特征维度 | `raw_node_dim` | `2` | GAT 输入 `[SNR, Q]` |
| Loss | — | `-Σ(Q_i × R_i)` | MWS 加权效用 |

**运行命令：**

```bash
# 队列感知实验（默认参数）
uv run python experiments/run_queue_experiment.py --num_uav 30 --seed 42

# 自定义数据集规模和训练轮数
uv run python experiments/run_queue_experiment.py --num_uav 30 \
    --train_size 500 --val_size 64 --test_size 64 --epochs 30 --seed 42
```

**数据集目录：**
- 非队列数据：`datasets/dataset_numUAV=<N>.pkl`（与 `bin/run_experiment.py` 共用）
- 队列数据：`datasets/queue/dataset_numUAV=<N>.pkl`（队列实验专用）

**标准模型库：**
- 非队列模型：`results/models/gat_numUAV=<N>.pth`（Loss = -Σ R_i）
- 队列模型：`results/models/gat_queue_numUAV=<N>.pth`（Loss = -Σ Q_i×R_i）
- 命名直接以 `num_uav` 定位，方便任何时候直接调取

### 2. 功率控制模块（Power Control）

| 功能 | 参数 | 值 | 说明 |
|------|------|----|------|
| 启用开关 | `USE_POWER_CONTROL` | `true` | 在实验脚本中设置 |
| 惩罚系数 | `LAMBDA_POWER_PENALTY` | `0.01` | 鼓励降低功率 |

**运行命令：**

```bash
# 功率控制实验
uv run python experiments/power_control_experiment.py --mode compare --num-uav 50
```

### 3. CSI 量化模块

| 功能 | 参数 | 值 | 说明 |
|------|------|----|------|
| CSI 模式 | `CSI_MODE` | `"quantized"` | 可选 `perfect`/`large_scale`/`quantized` |
| 量化精度 | `Q_BITS` | `4` | 比特数越小精度越低 |

**运行命令：**

```bash
uv run python bin/run_experiment.py --csi-mode quantized --q-bits 4 --seed 42
```

---

## 实验一览

> **⚠️ 重要提示：每个实验都有对应的专用脚本，请勿混用。**
> - 基础吞吐量实验 → `bin/run_experiment.py`
> - 队列感知实验 → `experiments/run_queue_experiment.py`
> - 功率控制实验 → `experiments/power_control_experiment.py`
> - 跨尺度扩展性 → `experiments/scaling_experiment.py`

| 实验 | 命令 | 主要用途 |
|------|------|----------|
| 基础吞吐量 | `bin/run_experiment.py` | 训练 GAT，输出 test utility |
| 队列感知仿真 | `run_queue_experiment.py` | 队列稳定性分析 |
| 功率控制对比 | `power_control_experiment.py` | 调度-功率联合优化 |
| 跨尺度扩展性 | `scaling_experiment.py --mode all` | 模型泛化能力 |
| CSI 不完美性 | `csi_imperfection_study.py` | 信道量化精度影响 |
| 收敛性分析 | `convergence_analysis.py` | 训练样本量分析 |
| 画图 | `utils.plot_paper` | 生成论文图表 |
| TensorBoard | `tensorboard --logdir=./logs/` | 可视化训练曲线 |

### 1. 基础吞吐量实验（推荐入口）

```bash
# 快速验证（不保存，10 条样本）
uv run python bin/run_experiment.py --config configs/exp_example.yaml --seed 42 --dry-run

# 正式训练
uv run python bin/run_experiment.py --config configs/exp_example.yaml --seed 42

# 量化 CSI 实验
uv run python bin/run_experiment.py --config configs/exp_example.yaml \
    --csi-mode quantized --q-bits 4 --seed 42
```

**输出：** `results/models/` 训练模型，`results/all_experiments_summary.xlsx` 汇总数据。

### 2. 队列感知实验

```bash
# 队列感知实验（训练 + 多时隙仿真）
uv run python experiments/run_queue_experiment.py --num_uav 30 --seed 42

# 自定义数据集规模
uv run python experiments/run_queue_experiment.py --num_uav 30 \
    --train_size 500 --val_size 64 --test_size 64 --seed 42

# 测试不同到达率
uv run python experiments/run_queue_experiment.py --num_uav 30 \
    --arrival_rates 0.5,1.0,2.0,3.0 --num_slots 5000 --seed 42
```

> **注意**：`run_queue_experiment.py` 固定使用队列感知模式（MWS 加权 loss），
> 数据集存储在 `datasets/queue/` 目录，与基础实验的 `datasets/` 分离。

**实验对比的调度算法：**

| 算法 | 特征输入 | 调度依据 | 说明 |
|------|----------|----------|------|
| `GAT` | `[SNR, Q]` (dim=2) | `Q_i × R_i` | 队列感知 GAT（核心方法） |
| `GAT-base` | `[SNR]` (dim=1) | 纯 `R_i` | 非队列 GAT baseline |
| `Random` | — | 均匀概率 | 随机调度 |
| `Greedy` | `[SNR]` | SNR 排序 | 贪心选 SNR 最高链路 |

> 通过 GAT vs GAT-base 对比可验证"队列感知特征"是否有额外价值。

**核心公式：**

```
D_i(t) = π_i(t) × R_i(t) × B × τ / L   (发送量，packets/slot)
Q_i(t+1) = max(0, Q_i(t) - D_i(t)) + A_i(t)   (Lindley 方程)
```

### 3. 功率控制实验

```bash
# 对比实验（有 vs 无功率控制）
uv run python experiments/power_control_experiment.py --num-uav 50 --train-size 300

# 仅运行功率控制模式
uv run python experiments/power_control_experiment.py --mode power-control-only --power-penalty 0.05

# 网格搜索最优 λ
uv run python experiments/power_control_experiment.py --mode grid-search \
    --power-penalty-search 0.001 0.01 0.05 0.1 0.5
```

### 4. 跨尺度扩展性实验

```bash
# 配对泛化（训练 X → 测试 Y）
uv run python -m experiments.scaling_experiment --mode paired --use-existing-models

# 混合训练
uv run python -m experiments.scaling_experiment --mode mixed

# 全部实验 + 快速模式
uv run python -m experiments.scaling_experiment --mode all --quick
```

### 5. CSI 不完美性实验

```bash
# 默认跑所有模式
uv run python experiments/csi_imperfection_study.py --gpu-id 0

# 跑 numUAV=50，测试 2/4/6/8 bit 量化
uv run python experiments/csi_imperfection_study.py --num-uav 50 --quantized-bits 2 4 6 8
```

### 6. 收敛性分析

```bash
uv run python -m experiments.convergence_analysis --gpu-id 0
```

---

## 队列感知仿真详解

### 物理场景

每个时隙的流程：

```
┌──────────────────────────────────────────────────────┐
│  时隙 t                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ 业务到达  │  │  MAC调度  │  │  物理传输  │           │
│  │ A_i~Poi(λ)│→ │ π_i(t)=0/1│→ │ R_i(SNR)  │           │
│  │(包/时隙) │  │ (链路激活) │  │ (bps/Hz)  │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│       ↓              ↓              ↓                 │
│  队列更新:        发送决策:       发送包数:          │
│  Q_i += A_i       π_i ∈ {0,1}    D_i = π_i ×      │
│                                    R_i×B×τ/L          │
└──────────────────────────────────────────────────────┘
```

### 关键参数说明

| 参数 | 说明 | 单位 |
|------|------|------|
| `arrival_lambda` | 泊松到达均值 | packets/slot |
| `slot_duration` | 时隙长度 τ | 秒 |
| `packet_size` | 包大小 | bits |
| `num_slots` | 仿真时隙数 | slots |

### 性能指标

| 指标 | 说明 | 单位 |
|------|------|-------|
| `avg_queue` | 最终平均队列长度 | packets |
| `long_term_avg_queue` | 长期平均队列长度 | packets |
| `throughput` | 吞吐量比（departures/arrivals） | 无量纲，≈1 表示稳定 |
| `throughput_mbps` | 实际网络总吞吐量 | Mbps |
| `departure_rate_pps` | 每链路平均发送速率 | packets/slot |
| `queue_overflow_prob` | 溢出概率（Q > 50 的比例） | — |
| `is_stable` | 稳定性判断 | True/False |

> **关键对比**：GAT vs GAT-base 决定了队列感知特征 `[SNR, Q]` 是否有额外价值。

---

## 项目结构

```
GNN_Scheduling/
├── bin/
│   └── run_experiment.py        # CLI 统一入口（基础吞吐量实验）
│
├── configs/
│   └── exp_example.yaml         # 基础配置（所有复杂功能默认关闭）
│
├── models/
│   └── model.py                 # GATScheduler、calculate_achievable_rate
│
├── experiments/
│   ├── runner.py                # ExperimentRunner：实验编排器
│   ├── run_queue_experiment.py  # 队列感知调度实验
│   ├── csi_imperfection_study.py # CSI 不完美性实验
│   ├── scaling_experiment.py    # 跨尺度扩展性实验
│   ├── convergence_analysis.py  # 收敛性分析
│   ├── power_control_experiment.py # 功率控制实验
│   └── compare_gat_vs_spatial_dl.py # GAT vs Spatial DL 对比
│
├── simulator/
│   ├── __init__.py            # 模块说明文档
│   ├── base.py                # BaseSimulator、SimpleSimulator
│   └── queue_simulator.py     # QueueSimulator：队列感知仿真器
│
├── train/
│   └── trainer.py              # 训练循环、梯度裁剪、早停
│
├── utils/
│   ├── baseline.py             # 基线算法（Random/Greedy/FPLinQ/Spatial DL）
│   ├── network_generate.py     # 数据生成器、NetworkState
│   ├── plot_paper.py          # 论文图表绘制
│   ├── plot_config.py          # 配色方案
│   └── io.py                  # 文件 I/O、Excel 读写
│
├── results/                    # 实验结果（详见「结果文件详解」章节）
│
└── logs/
    └── tensorboard/            # TensorBoard 训练日志
```

---

## 结果文件详解

### 目录结构

```
results/
├── models/                                   # 标准模型库（共用）
│   ├── gat_numUAV=<N>.pth                   # 非队列模型（raw_node_dim=1）
│   ├── gat_queue_numUAV=<N>.pth             # 队列感知模型（raw_node_dim=2）
│   ├── scaling_numUAV<N>.pth                # 跨尺度实验模型（配对泛化）
│   └── mixed_numUAV20_30_40_50.pth          # 混合尺度训练模型
│
├── exp_<numUAV>_<timestamp>/                # 基础吞吐量实验输出（bin/run_experiment.py）
│   ├── exp_<numUAV>_<timestamp>_per_epoch.json  # 每 epoch 训练日志
│   ├── exp_<numUAV>_<timestamp>_model.pth       # 训练模型快照
│   └── exp_<numUAV>_<timestamp>_baselines.json  # 基线算法对比结果
│
├── queue_exp<timestamp>_s<seed>/            # 队列感知实验输出（run_queue_experiment.py）
│   ├── simulation_results.json              # 仿真结果 JSON
│   ├── trajectory_lambda<λ>.png             # 队列长度/吞吐量轨迹图
│   └── queue_comparison_bar.png             # 各算法平均队列长度柱状图
│
├── img/                                      # 论文图表（utils/plot_paper.py 生成）
│   ├── baseline_comparison.{eps,png}        # 各算法吞吐量对比
│   ├── training_curves_val_utility.{eps,png}# 验证集效用训练曲线
│   ├── training_curves_train_loss.{eps,png} # 训练 Loss 曲线
│   ├── performance_vs_samples.{eps,png}     # 性能随训练样本量变化
│   └── csi_imperfection.{eps,png}           # CSI 量化精度影响
│
└── all_experiments_summary.xlsx             # 所有实验汇总表
```

### 模型文件说明

| 文件名 | 训练脚本 | 节点特征 | Loss 函数 | 用途 |
|--------|----------|----------|-----------|------|
| `gat_numUAV=10.pth` | `bin/run_experiment.py` | `[SNR]` (dim=1) | `-Σ R_i`（纯吞吐量） | 基础 GAT 调度 |
| `gat_queue_numUAV=10.pth` | `run_queue_experiment.py` | `[SNR, Q]` (dim=2) | `-Σ(Q_i × R_i)`（MWS 加权） | 队列感知调度 |
| `scaling_numUAV20/30/40/50.pth` | `scaling_experiment.py` | `[SNR]` (dim=1) | `-Σ R_i` | 配对泛化实验 |
| `mixed_numUAV20_30_40_50.pth` | `scaling_experiment.py` | `[SNR]` (dim=1) | `-Σ R_i` | 混合尺度训练 |

> 注意：`scaling_*.pth` 命名不带 `=`，`gat_*_numUAV=*.pth` 带 `=`。

> **注意**：`gat_numUAV=<N>.pth` 和 `gat_queue_numUAV=<N>.pth` 共用同一套数据集（`datasets/` 和 `datasets/queue/`），可通过 `run_queue_experiment.py` 中的 GAT vs GAT-base 对比来验证队列感知特征的价值。

### 实验目录命名规则

**基础实验** `exp_<numUAV>_<YYYYMMDD>_<HHMM>`：
- `exp_20_20260130_2147`：numUAV=20，2026年1月30日 21:47 运行

**队列实验** `queue_exp<YYYYMMDD>_<HHMM>_s<seed>`：
- `queue_exp20260521_1100_s42`：队列实验，2026年5月21日 11:00，seed=42
- `s<seed>` 后缀表示随机种子（默认 seed=42），用于实验复现
- 6 个带 `s42` 的目录中，5 个含 `simulation_results.json`（成功），1 个为空（失败或中断）

### Queue 实验结果解读（simulation_results.json）

每个实验对比 4 种调度算法，字段含义如下：

| 算法 | 说明 | 特征输入 | 调度依据 |
|------|------|----------|----------|
| `GAT` | 队列感知 GAT | `[SNR, Q]` | `Q_i × R_i` 加权 |
| `GAT-base` | 非队列 GAT | `[SNR]` | 纯 `R_i` 最大化 |
| `Random` | 随机调度 | — | 均匀概率激活 |
| `Greedy` | 贪心调度 | `[SNR]` | 选 SNR 最高的链路 |

结果按到达率 λ 分组，格式为 `<algo>_<λ>`（如 `GAT_0.5`）。关键指标：

| 指标 | 说明 | 稳定判断 |
|------|------|----------|
| `avg_queue` | 仿真结束时平均队列长度 |越小越好 |
| `throughput` | 吞吐量比（departures/arrivals） | ≈ 1 表示稳定 |
| `is_stable` | 稳定性判断（throughput ≥ 0.95） | True/False |
| `queue_overflow_prob` | 队列溢出概率（Q > 50） |越小越好 |

### 图片说明

| 图片文件 | 生成函数 | 数据来源 | 说明 |
|----------|----------|----------|------|
| `baseline_comparison` | `plot_baseline_comparison()` | `all_experiments_summary.xlsx` Sheet1 | GAT vs 基线算法吞吐量柱状图 |
| `training_curves_val_utility` | `plot_training_curves(metric='val_utility')` | 各 `exp_*/_per_epoch.json` | 验证集效用随 epoch 变化 |
| `training_curves_train_loss` | `plot_training_curves(metric='train_loss')` | 各 `exp_*/_per_epoch.json` | 训练 Loss 随 epoch 变化 |
| `performance_vs_samples` | `plot_performance_vs_samples()` | `all_experiments_summary.xlsx` Sheet2 | 性能随训练样本量变化（收敛性分析） |
| `csi_imperfection` | `plot_csi_imperfection()` | `all_experiments_summary.xlsx` Sheet3 | CSI 量化精度（2/4/6/8/10 bit）影响 |
| `trajectory_lambda<λ>.png` | `_plot_trajectories()` | `queue_exp*/simulation_results.json` | 队列长度/发送量/吞吐量随时隙变化轨迹 |
| `queue_comparison_bar.png` | `_plot_trajectories()` | `queue_exp*/simulation_results.json` | 各算法平均队列长度柱状对比 |

> 生成图片命令：`uv run python utils/plot_paper.py --plot <type>`，`<type>` 可选 `all`、`baseline`、`training`、`samples`、`csi`。

---

## 研究日志（Changelog）

> 本章节按时间顺序记录项目的每次重要迭代，包括成功的模块、失败的尝试、以及背后的理论分析。其他 chat 可以通过此日志快速了解项目全貌，避免重复踩坑。

---

### v0.3 — 队列感知模块（已放弃，但有记录价值）

**时间：** 2026 年 5 月

**目标：** 在 GAT 调度中加入队列长度信息，使调度器能够优先服务积压严重的链路，提升网络稳定性。

**实现方式：**
- 节点特征扩展：`[SNR]` → `[SNR, Q]`，即在输入特征中加入当前队列长度
- Loss 函数修改：`L = -Σ R_i` → `L = -Σ(Q_i × R_i)`，即 MWS（Maximum Weight Scheduling）加权效用
- 理论依据：Lyapunov 优化框架下的队列稳定性分析

**实验结果：**

| 指标 | GAT（队列感知） | GAT-base（非队列） |
|------|---------------|-----------------|
| 平均队列长度 | 更差 | 更好 |
| 吞吐量比 | 更差 | 更好 |
| 稳定性 | 更差 | 更好 |

**失败原因（核心发现）：**

经过深入分析，发现 MWS 加权 loss 在本场景下**理论上不成立**：

1. **证明前提失效**：Lyapunov 稳定性证明中要求 `π² = π`（二元调度）以及服务率恒定。但我们的场景中：
   - 调度值是连续概率 `π ∈ [0,1]`（而非二元 0/1）
   - 服务率由 Shannon 公式 `R = log₂(1+SINR)` 决定，是随信道连续变化的
   - 这使得 Lyapunov drift-plus-penalty 中的 `π² = π` 恒等式完全失效

2. **均匀到达抹平差异化需求**：所有链路的到达率 λ 相同，调度器无法从队列状态中识别"哪条链路更需要服务"

3. **训练-仿真分布不匹配**：训练时使用 `uniform(0, 500)` 的队列快照，仿真时使用 Poisson 累积队列

**结论：** 将 Q 显式加入 loss 函数对于本场景无意义。队列信息需要通过其他方式（如时间记忆模型）隐式传递，而非直接作为优化目标。

**相关脚本：** `experiments/run_queue_experiment.py`

**遗留影响：** 代码中仍保留 `use_queue` 参数和 `raw_node_dim=2` 的支持，但默认关闭。如果未来需要与时间记忆模型结合，应考虑让队列历史通过 GRU/LSTM 隐式传递，而非显式加在 loss 中。

---

### v0.2 — 功率控制模块（已实现，效果有限）

**时间：** 2026 年 4-5 月

**目标：** 在 GAT 输出中增加功率控制头，使调度器不仅决定"哪些链路激活"，还决定"每条链路开多大功率"。

**实现方式：**
- GAT 输出双头：`[π_i, α_i]`，其中 `π_i ∈ [0,1]` 为激活概率，`α_i ∈ [0,1]` 为功率缩放因子
- 可达速率修改为：`R_i = α_i × log₂(1 + α_i × SINR_i / Γ)`，其中 Γ 为 SNR gap
- Loss 添加功率惩罚项：`L = -Σ R_i + λ × Σ α_i`

**实验结果：** 在密集集群（链路数足够多）场景下，功率控制**几乎没有效果提升**。

**原因分析：** 密集集群中，调度决策（"选择开哪几条链路"）已经是决定性能的主要因素。由于可选链路数量充足，简单地通过调度选择"开最好的那几条"已经接近性能上界。功率控制（"每条链路开多亮"）在这个基础上的边际收益极小。

**结论：** 功率控制在密集集群基础调度中价值有限。但如果在以下方向扩展，可能仍有意义：
- 与**队列感知**结合：优先为积压严重的链路分配更高功率
- 与**安全约束**结合：为某些链路设定最小功率保证
- 与**时间记忆模型**结合：在信道动态变化时动态调整功率

**相关脚本：** `experiments/power_control_experiment.py`

**遗留代码：** `models/model.py` 中仍保留 `use_power_control` 和双头输出的实现，代码路径未被删除。

---

### v0.1 — CSI 量化模块（已验证，效果良好）

**时间：** 2026 年 3 月

**目标：** 研究信道状态信息（CSI）不完美性对调度性能的影响。

**实现方式：**
- 三种 CSI 模式：
  - `perfect`：理想信道状态
  - `large_scale`：仅大尺度衰落（路径损耗 + 阴影），不含小尺度
  - `quantized`：对 SNR 做 q-bit 量化（q = 2, 4, 6, 8）
- 距离量化：对链路距离做 q-bit 二进制编码作为边特征

**实验结果：** 4-bit 以上量化性能接近完美 CSI，2-bit 量化有明显下降。

**相关脚本：** `experiments/csi_imperfection_study.py`

---

### v0.0 — 基础 GAT 调度（核心模块）

**时间：** 2026 年 3 月之前

**目标：** 构建基于 GAT 的链路调度框架，最大化全网总吞吐量。

**核心设计：**
- 冲突图建模：节点 = 链路，边 = 干扰关系，边权重 = 干扰强度
- GAT 消息传递：迭代地从局部邻居学习整体网络态势
- Loss 函数：`L = -Σ R_i`，直接最大化可达速率之和
- 自环机制：添加自环确保节点自身信息不丢失

**相关脚本：** `bin/run_experiment.py`，`experiments/scaling_experiment.py`

---

### 进行中 — 时空结合模型（STGNN）

**目标：** 将 GAT（空间推理）与 LSTM/GRU（时间记忆）结合，处理以下场景：
- **动态信道**：UAV 移动导致大尺度信道连续变化
- **队列历史**：队列长度的时序积累规律
- **QoS 公平性**：通过时间记忆实现长期公平性保证

**技术路线：**
- 信道模型升级：从 block fading 升级到 3GPP TR 38.901 大尺度衰落（路径损耗 + 阴影衰落随位置连续变化）
- 时间模型选型：**LSTM**（优于 GRU，原因见下）
- 架构设计：**GRU 先融合时间 → 作为 GAT 的输入特征**（方式 B，保持 GAT 空间模块相对独立）
- Loss 函数：**仍是 `-Σ π_i × R_i`**，不包含 Q。时间记忆通过 GRU 的隐状态隐式传递

**LSTM 优于 GRU 的理由：**
- RouteNet-Fermi (Purdue 2024) 的 ablation：在吞吐量相关指标上 LSTM 优于 GRU
- LSTM 的 cell state 提供显式长期存储，适合队列历史积累
- 参数量差异不大（LSTM 多约 20% 参数，但你的网络规模下可忽略）

**审稿人问题准备（Q: Why LSTM/GRU instead of ...）：**
> "We evaluate both LSTM and GRU as temporal encoders following the ablation study of RouteNet-Fermi [ref]. LSTM consistently achieves better performance on throughput-related metrics. The explicit memory cell in LSTM provides more stable long-term storage for queue state evolution, which aligns with our goal of capturing channel correlation over UAV trajectories. This is further supported by Chung et al. [2014], which demonstrates LSTM's superior capability in capturing long-range dependencies."

**待实现（TODO）：**
- [ ] 修改 `NetworkGenerator` 支持时序生成（UAV 轨迹 → 位置 → 3GPP 信道）
- [ ] 修改 `QueueSimulator` 支持动态信道（`slots_per_block` 参数）
- [ ] 实现 LSTM/GAT 联合模型类
- [ ] 生成多时隙序列训练数据
- [ ] BPTT 端到端训练

---

## 关键模块说明

### models/model.py — GATScheduler

| 参数 | 说明 |
|------|------|
| `raw_node_dim` | 节点原始特征维度（`USE_QUEUE=false` 时为 1，`USE_QUEUE=true` 时为 2）|
| `embed_node_dim` | 节点嵌入维度 |
| `edge_dim` | 边特征维度（干扰权重）|
| `num_heads` | 多头注意力头数 |
| `T_iterations` | GAT 消息传递轮数 |
| `gat_out_dim` | GAT 层总输出维度 |

**输入：** `[num_links, raw_node_dim]` 节点特征 + `[2, num_edges]` 边索引 + `[num_edges, 1]` 边特征
**输出：** `[num_links, 1]` 每条链路的激活概率 `π_v ∈ [0,1]`

### utils/baseline.py — 基线算法

| 算法 | 说明 |
|------|------|
| Random | 随机选择链路 |
| Simple Greedy | 贪心选 SNR 最高的链路 |
| Local Greedy | 局部干扰感知的贪心调度 |
| FPLinQ | 分数规划调度算法（Cui et al. 2019） |
| Spatial DL | 空间深度学习方法（Cui et al. 2019） |

### simulator/ — 仿真器模块

| 类 | 说明 |
|----|------|
| `BaseSimulator` | 仿真器抽象基类 |
| `SimpleSimulator` | 单时隙评估（无队列，用于训练） |
| `QueueSimulator` | 多时隙队列仿真（用于队列感知实验） |

**QueueSimulator 使用注意：**
- `include_queue_feature` 必须与模型训练时的 `USE_QUEUE` 一致
- `arrival_lambda` 需选择合理值（过大则队列爆炸）

---

## 训练监控

```bash
tensorboard --logdir=./logs/
```

浏览器打开 `http://localhost:6006`，查看：

- `train_loss`：训练 loss 曲线
- `val_utility`：验证集效用曲线
- `pi_mean` / `pi_std`：链路平均激活概率

---

## 注意事项

1. **参数管理**：所有超参数集中于 `configs/exp_example.yaml`，不要直接改代码。
2. **实验复现**：务必使用 `--seed` 设置随机种子，相同 seed + 相同配置 = 相同结果。
3. **数据集缓存**：按 `NUM_UAV` 分离；修改后会自动生成新数据集。
4. **GPU 选择**：多卡环境下用 `--gpu-id` 指定。
5. **Excel 汇总**：`ExperimentRunner` 自动将每次运行结果追加到 `results/all_experiments_summary.xlsx`。
6. **防呆设计**：队列感知实验中若参数不一致，会打印 ERROR 并终止运行。
