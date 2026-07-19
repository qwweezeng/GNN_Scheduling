# GNN_Scheduling：基于GNN的无人机集群链路调度

本项目是论文草稿《GNN-based Dynamic Link Scheduling for Dense UAV Swarm Networks》的 PyTorch 仿真实现。项目旨在模拟一个密集的无人机 (UAV) 自组网 (FANET)，并使用图神经网络 (GNN) 来解决动态、分布式的链路调度问题。

核心思想是将无人机网络中的通信链路建模为一个**冲突图 (Conflict Graph)**：

* 图中的**节点 (Vertex)** 代表一条**通信链路**。
* 图中的**边 (Edge)** 代表两条链路之间的**干扰关系**。

然后，我们使用图注意力网络 (GAT) 来学习一个调度策略 `pi` (即每条链路的激活概率)，目标是最大化网络的总吞吐量（或效用）。

## 项目结构

1. **`network_generate.py`**
   * `NetworkGenerator`: 模拟器。负责生成无人机集群的物理位置、计算信道增益 (SNR)、并构建冲突图。
   * `NetworkState`: 用于存储单个网络快照（即一个图样本）的 dataclass
   * `visualize_network`: 可视化工具，用于绘制无人机的物理连接图和抽象的冲突图。
1. **`model.py`**
   * `NodeEmbedding`: 节点（链路）的初始特征嵌入 MLP。
   * `GATLayer`: GAT 核心层，用于聚合邻居（干扰链路）的信息。
   * `GATScheduler`: 完整 GNN 调度模型。它堆叠了多层 GAT，并最终输出每条链路的调度概率 `pi`。
   * `calculate_achievable_rate`: 物理模型。根据 GNN 输出的 `pi`，计算在干扰下的 SINR 和可达速率 (Rate)。

1. **`main.py`** 
   * `generate_dataset`: 数据集生成函数。
   * `train`: 核心训练循环。
   * `evaluate`: 评估函数。
   * `if __name__ == '__main__':`: 主入口，包含所有超参数设置和模拟流程。

## 依赖环境

你需要安装以下 Python 库：
* `torch` (PyTorch)
* `numpy`
* `matplotlib`
* `networkx`
* `tensorboard`
```bash
pip install torch numpy matplotlib networkx tensorboard
```
## 如何使用

### 1. 训练模型

所有参数均在 `main.py`  中定义。
直接运行main.py 即可开始：
```bash
python main.py  
```

- 脚本会首先生成（或加载）数据集，保存到 `/datasets/dataset.pkl`。
- 训练过程中的日志（Loss, Utility, pi_mean 等）会保存到 `./logs` 目录。
- 一份网络的可视化样本会保存为 `./logs/uav_network_init.png`。
- 训练过程中验证集上表现最好的模型会保存为`./logs/best_model.pth`。
### 2. 监控训练
你可以使用 TensorBoard 来可视化训练过程：
```bash
tensorboard --logdir=./logs_train
```

在浏览器中打开`http://localhost:6006`，可以查看`train_loss`, `val_utility`, `pi_mean`(链路平均激活概率), `pi_std`(激活概率标准差) 等关键指标。
## 模块详解

### network_generate.py
#### NetworkGenerator
仿真环境类。

初始化参数 (`__init__`):
- `num_uavs` (int): 网络中的无人机总数。
- `area_width` (float): 模拟区域的边长 (米)。
- `height_min`, `height_max` (float, float): 无人机飞行的高度范围 (米)。
- `uav_min_distance` (float): 无人机之间的最小安全距离 (米)。
- `tx_power` (float): 发射功率 (dBm)。
- `Gt`, `Gr` (float): 收发天线增益 (dBi)。
- `noise_psd` (float): 噪声功率谱密度 (dBm/Hz)。
- `bandwidth` (float): 系统带宽 (Hz)。
- `snr_thres` (float): 用于确定是否存在有效链路的最小 SNR 阈值 (dB)。
#### generate_network()
`NetworkGenerator`的核心方法，用于生成一个网络快照。
- 功能: 生成`num_uavs`个无人机，计算它们之间的 SNR，构建冲突图。
- 返回值:`NetworkState`对象，包含一个完整的图样本。
#### NetworkState
存储图数据的容器。

### model.py
#### GATScheduler
GNN 调度器模型。
初始化参数 (`__init__`):
- `raw_node_dim` (int): 原始节点特征的维度（例如，只用 SNR 时为 1）。
- `mlp_hidden_dims` (list): 初始节点嵌入 MLP 的隐藏层维度。
- `embed_node_dim` (int): 节点嵌入后的维度。。
- `edge_dim (int)`: 边特征的维度（例如，干扰权重为 1）。
- `gat_out_dim` (int): GAT 层输出的特征维度。
- `num_heads` (int): GAT 使用的多头注意力数量。
- `T_iterations` (int): GNN 消息传递的轮数（即 GAT 层数）。

前向传播 (`forward`):
- 输入:
	- `x` (Tensor): [num_links, raw_node_dim] 节点特征。
	- `edge_index` (Tensor): [2, num_edges] 边索引。
	- `edge_attr` (Tensor): [num_edges, edge_dim] 边特征。
- 输出:
	- `pi_v` (Tensor): [num_links, 1]，每条链路的调度概率 (0 到 1 之间)。
	- `h_v_final` (Tensor): 最终的节点嵌入。

#### calculate_achievable_rate()
计算给定调度策略 pi 下的信道速率。
- 输入:
	- `pi` (Tensor): [num_links, 1]，来自 GNN 的调度概率。
	- `state` (NetworkState): 当前的网络状态。
	- `binary` (bool): True 表示评估时将 pi 四舍五入为 0/1 决策；False 表示训练时使用连续概率 pi 来计算 SINR（用于梯度传播）。
- 输出:
	- `achievable_rates` (Tensor): [num_links]，每条链路的可达速率 (bps)。    

### main.py (主程序块)
关键超参数：
- `NUM_UAV` (int): 仿真中的无人机数量。
- `NODE_RAW_DIM` (int): 节点特征维度，应与 GATScheduler 匹配。
- `NODE_EMBED_DIM` (int): 嵌入维度，应与 GAT_OUT_DIM 匹配。
- `GAT_OUT_DIM` (int): GAT 输出维度。
- `NUM_GAT_LAYERS` (int): GAT 迭代次数。
- `LR` (float): 学习率 (例如 1e-4)。
- `EPOCHS` (int): 训练轮数。
- `TRAIN_SIZE`, `VAL_SIZE` (int): 训练集和验证集的大小。
- `LAMBDA_SPARSITY` (float): (重要) 稀疏性惩罚项。用于平衡 `max(Rate)` 和 `min(pi)` 两个目标，防止模型陷入“全部激活”的局部最优。
	- `loss = -torch.mean(rates) + lambda_sparsity * torch.mean(pi)`
