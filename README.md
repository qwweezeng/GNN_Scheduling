# GNN_Scheduling — 基于GNN的无人机集群链路调度

本项目实现了论文《GNN-based Dynamic Link Scheduling for Dense UAV Swarm Networks》的仿真代码，使用图注意力网络 (GAT) 解决密集无人机集群中的分布式链路调度问题。

## 核心思想

将无人机网络中的通信链路建模为**冲突图 (Conflict Graph)**：
- **节点** = 通信链路
- **边** = 链路间的干扰关系

通过 GAT 学习调度策略，最大化网络总吞吐量/效用。

## 安装依赖

**使用 uv（推荐）:**
```bash
# 安装 uv (如果没有)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装依赖
uv pip install -r pyproject.toml
```

**使用 pip:**
```bash
pip install torch numpy matplotlib networkx tensorboard pyyaml pandas openpyxl
```

## 运行实验

### 1. 小规模测试 (Dry-run)
快速验证配置是否正确：
```bash
# 使用 uv
uv run python3 bin/run_experiment.py --config configs/exp_example.yaml --seed 42 --dry-run

# 或使用 python3
python3 bin/run_experiment.py --config configs/exp_example.yaml --seed 42 --dry-run
```

### 2. 正式训练与评估
训练 GAT 模型并评估性能：
```bash
# 使用 uv
uv run python3 bin/run_experiment.py --config configs/exp_example.yaml --seed 42

# 或使用 python3
python3 bin/run_experiment.py --config configs/exp_example.yaml --seed 42
```

### 3. 性能 vs 训练样本数量实验
分析模型在不同训练集大小下的收敛性能和泛化能力：
```bash
# 使用 uv
uv run python3 -m experiments.convergence_analysis

# 或使用 python3
python3 -m experiments.convergence_analysis
```

### 扩展性实验
```bash
uv run python -m experiments.scaling_experiment --mode paired --use-existing-models

uv run python -m experiments.scaling_experiment --mode mixed
```

### 4. 可视化结果
生成论文所需图表：
```bash
# 使用 uv
uv run python3 -m utils.plot_paper --plot all

# 或使用 python3
python3 -m utils.plot_paper --plot all
```

### 5. 监控训练过程
```bash
tensorboard --logdir=./logs/
```
浏览器打开 `http://localhost:6006` 查看训练曲线。

## 📊 项目架构

```
GNN_Scheduling/
├── 🎮 bin/
│   └── run_experiment.py          # CLI入口，统一实验执行
│
├── ⚙️ configs/
│   └── exp_example.yaml           # YAML配置，集中管理所有参数
│
├── 🧠 models/
│   └── model.py                   # GAT调度器核心算法
│
├── 🧪 experiments/
│   ├── runner.py                  # 实验编排器
│   └── convergence_analysis.py    # 收敛性与样本效率分析
│
├── 🏋️ train/
│   └── trainer.py                 # 训练循环
│
├── 🔧 utils/
│   ├── baseline.py                # 基线算法实现
│   ├── network_generate.py        # 数据生成器
│   ├── plot_paper.py              # 论文图表绘制
│   └── io.py                      # 文件I/O
│
├── 📊 results/
│   ├── models/                    # 模型权重
│   ├── metrics/                   # 性能指标
│   ├── reports/                   # 分析报告
│   └── img/                       # 生成图表
│
└── 📁 logs/
    └── tensorboard/               # 训练日志
```

## 关键模块

### models/model.py - GATScheduler
| 参数 | 说明 |
|-----|------|
| `raw_node_dim` | 原始节点特征维度 |
| `embed_node_dim` | 节点嵌入维度 |
| `edge_dim` | 边特征维度 |
| `num_heads` | 多头注意力数量 |
| `T_iterations` | GAT层数（消息传递轮数） |

**输入:** 节点特征、边索引、边特征  
**输出:** 每条链路的调度概率 `pi_v ∈ [0,1]`

### utils/baseline.py - 基线算法
- **Random**: 随机选择链路
- **Simple Greedy**: 简单贪心调度
- **Local Greedy**: 局部贪心调度
- **FPLinQ**: 分数规划调度算法
- **Spatial DL**: 空间深度学习 (Cui et al., 2019)

### utils/network_generate.py - 数据生成器
| 参数 | 说明 |
|-----|------|
| `num_uavs` | 无人机数量 |
| `area_width` | 模拟区域边长 (m) |
| `tx_power` | 发射功率 (dBm) |
| `bandwidth` | 系统带宽 (Hz) |
| `snr_thres` | 有效链路SNR阈值 (dB) |

## 配置说明

所有参数在 `configs/exp_example.yaml` 中配置：

| 参数 | 说明 |
|-----|------|
| `NUM_UAV` | 仿真无人机数量 |
| `NODE_RAW_DIM` | 节点特征维度 |
| `GAT_OUT_DIM` | GAT输出维度 |
| `NUM_GAT_LAYERS` | GAT层数 |
| `LR` | 学习率 |
| `EPOCHS` | 训练轮数 |
| `TRAIN_SIZE` / `VAL_SIZE` / `TEST_SIZE` | 数据集划分 |
| `LAMBDA_SPARSITY` | 稀疏性惩罚项 |

## 结果说明

| 目录 | 内容 |
|-----|------|
| `results/models/` | 训练好的模型权重 `.pth` |
| `results/metrics/` | 性能指标 CSV |
| `results/img/` | 生成的图表 (PNG/EPS) |
| `logs/tensorboard/` | TensorBoard 日志 |

## 注意事项

1. **参数管理**: 所有参数在 `configs/*.yaml` 中修改，不要直接改代码
2. **实验复现**: 使用 `--seed` 参数设置随机种子
3. **数据集缓存**: 设置 `DATASET_CACHE_PATH` 可缓存数据集，避免重复生成
