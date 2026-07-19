# GNN Scheduling for Dense UAV Swarms

本项目研究密集无人机自组网中的分布式链路调度。会议版本使用 GAT 根据局部
信道与干扰信息最大化网络容量；当前期刊扩展进一步引入链路队列，以经典
Max-Weight Scheduling (MWS) 的逐时隙目标

$$
\max_{\mathbf a\in\{0,1\}^{L}}\sum_i Q_iR_i(\mathbf a)
$$

实现 QoS-aware 调度。

当前固定 perfect-CSI 主模型已经冻结为 **LocalPhysicalReuse-v2**。历史
Structure、Anchor-Join、ParallelMPNN、SetNegotiation 等探索路线不再位于活动
源码树；完整历史仍可从 `archives/` 恢复。

## 1. 当前结论与阅读入口

第一次阅读建议按以下顺序：

1. `docs/CSI_CHAT_HANDOFF_20260715.md`：当前冻结状态、最新结果和 CSI 两天止损任务；
2. `docs/FIXED_CSI_QUEUE_BENCHMARK_TUTORIAL.md`：模型、对照和固定 CSI 正式结果；
3. `docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`：IEEE 风格公式定义；
4. `docs/TWO_STAGE_LOCALMAX_REUSE_TUTORIAL.md`：为什么从历史路线走到当前模型；
5. `docs/NEXT_CHAT_CONTEXT.md`：移动 CSI 的较长历史交接；
6. `docs/archive/two_stage/`：只在需要追溯失败实验时阅读。

统一物理口径为 `TX_POWER=10 dBm / perfect CSI`。没有物理元数据的旧数据或
checkpoint 可能来自历史 22 dBm 设置，不进入当前正式对比。

## 2. 环境

本机正式实验使用 Conda 环境 `pytorch`：

```bash
conda activate pytorch
python <script.py> ...
```

Codex 或非交互 shell 中等价写法为：

```bash
conda run --no-capture-output -n pytorch python <script.py> ...
```

`--no-capture-output` 用于实时显示长训练和仿真日志。

## 3. 活动代码结构

```text
models/
  model.py                         基础 GAT 与物理速率计算
  local_physical_reuse.py          冻结 Proposed、Pure 和 LocalMax 消融

baseline/
  queue_schedulers.py              random、单链路、会议 GAT、集中式 MWS-Greedy
  distgcn_adapter.py               有向物理图到对称硬 MWIS 的显式适配层
  distgcn-main/                    Zhao 2023 原始第三方代码，尚未适配
  Spatial_Deep_Learning_.../       Spatial DL 原始第三方代码
  FPLinQ/                          FPLinQ 第三方代码（若本地目录存在）

simulator/
  queue_simulator.py               固定/block-fading 队列仿真与 FIFO 时延
  mobile_block_fading.py           移动拓扑和不完美 CSI 仿真

train/
  trainer.py                       通用 GAT 训练器
  train_end_to_end_queue_gat.py    直接 GAT-q 对照训练入口

experiments/
  exp_fixed_csi_snapshot_utility.py  同状态单时隙 sum(QR) 对比
  exp_fixed_csi_qos.py             固定 CSI 动态 QoS 正式入口
  evaluate_mobile_block_fading_csi.py 下一阶段移动 CSI 入口
  README.md                        实验入口索引

scripts/
  build_fixed_csi_trace_cache.py   数据预处理，不代表论文实验
  audit_dataset_distribution.py    旧数据物理来源审计

utils/
  scheduling_objective.py          只定义 sum(QR) 数学目标
  fixed_csi_benchmark.py           配对 trace、运行、指标聚合
  plot_queue_benchmark.py          正式 QoS 图
  project_setup.py                 统一配置、生成器和严格 checkpoint loader
  queue_features.py                队列特征

tests/                              可重复单元/回归测试
results/queue/fixed_csi/            固定 CSI 正式结果
archives/                           停止路线源码与旧 README 的可恢复备份
```

职责规则：

- `models/` 只放在线调度模型；
- `baseline/` 放对照算法，不放共享工具；
- `utils/` 放无论文结论的共享计算、聚合和画图；
- `scripts/` 放缓存、审计等预处理；
- `experiments/` 只放能对应论文图表或明确研究问题的入口；
- `tests/` 只放代码变化后应重复执行的正确性检查，不保存一次性科研诊断。

## 4. 冻结模型

LocalPhysicalReuse-v2 包含两个步骤：

1. 冻结会议 GAT 产生 channel-only 活动先验，将其转化为期望干扰和期望有效
   速率；当前队列与该速率相乘，多个局部最大链路并行形成 0/1 骨架；
2. 其余链路在这个真实骨架条件下计算局部 MWS 加入边际，正收益候选再做一轮
   并行局部竞争。

它不是全网选一个 anchor，也不是顺序贪心扫描所有链路。完整公式和分布式消息
需求见 `docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`。

## 5. 正式固定 CSI 流程

### 5.1 训练直接 GAT-q 对照

```bash
conda run --no-capture-output -n pytorch python \
  train/train_end_to_end_queue_gat.py --num_uav 10
```

输出：

```text
datasets/queue/dataset_numUAV=10_tx=10dBm_csi=perfect_formal.pkl
results/models/gat_queue_mws_v3_numUAV=10_tx=10dBm_csi=perfect.pth
```

### 5.2 生成配对动态 trace

```bash
conda run --no-capture-output -n pytorch python \
  scripts/build_fixed_csi_trace_cache.py \
  --num_uav 10 --num_slots 10000 \
  --arrival_rates 0.5,1,2,2.2,2.4,2.6,2.8,3,3.2 \
  --mc_seeds 42,43,44,45,46 \
  --output datasets/queue_traces/fixed_numUAV=10_tx=10dBm_csi=perfect_slots=10000_mc=5_boundary.pt
```

每个 seed 只存一个固定信道图和若干到达张量，不为每个时隙重复保存相同图。

### 5.3 单时隙动作质量

```bash
conda run --no-capture-output -n pytorch python \
  experiments/exp_fixed_csi_snapshot_utility.py \
  --num_uav 10 --num_snapshots 100 --seed 42
```

该实验让所有方法面对相同信道和队列，比较 `sum(QR)`；集中式 MWS-Greedy
是全局信息强参考，但不是精确最优上界。

### 5.4 动态 QoS 主图

```bash
conda run --no-capture-output -n pytorch python \
  experiments/exp_fixed_csi_qos.py \
  --num_uav 10 --num_slots 10000 \
  --arrival_rates 0.5,1,2,2.2,2.4,2.6,2.8,3,3.2 \
  --mc_seeds 42,43,44,45,46 \
  --trace_cache datasets/queue_traces/fixed_numUAV=10_tx=10dBm_csi=perfect_slots=10000_mc=5_boundary.pt \
  --output_dir results/queue/fixed_csi/main_N10_10000slots_5seeds_no_deadline_boundary
```

输出包含：

```text
config.json              完整设置、checkpoint 和拓扑规模
metrics.json             均值与 95% CI
per_seed_metrics.json    每个 seed 的标量证据
trajectories.npz         聚合轨迹
figures/*.png, *.pdf     throughput、backlog、delay、queue growth
```

当前入口默认无 deadline、无限缓存，只生成 delivered throughput、backlog 和完成包
平均时延三类主图。无 deadline 时仍未完成的慢包会造成右删失，因此平均时延只在
至少 99% 到达包已完整送达时绘制；过载区应查看 backlog 和 delivered throughput。

## 6. 测试

```bash
conda run --no-capture-output -n pytorch python \
  -m unittest discover -s tests -p 'test_*.py'
```

当前测试覆盖速率公式、队列特征、固定 trace 复现、FIFO 时延、
移动 block-fading 身份映射、冻结 LocalPhysicalReuse 和集中式 greedy 局部最优性。

## 7. 当前 baseline 边界

- DistGCN/Zhao 2023 使用对称硬冲突 MWIS。`baseline/distgcn_adapter.py` 已实现
  “有向连续干扰图对称化 + INR 阈值硬化 + 队列加权单链路价值”；尚未加载其
  TensorFlow/Spektral checkpoint，图例必须标注 hardened-conflict adapter；
- FPLinQ 直接使用完整信道/SINR，不应描述成硬冲突算法；
- Spatial Deep Learning 利用地理卷积估计聚合干扰，也允许弱干扰并发；
- DistGCN 的旧代码当前缺少 SciPy 依赖，不阻塞固定 CSI 主图。

## 8. 归档

`archives/two_stage_exploration_source_20260711_14.tar.gz` 保存本轮移除的 94 个
历史模型、训练、诊断和测试文件；同目录 manifest 与 SHA-256 已校验。

清理前的长版 README 保存在
`archives/README_pre_fixed_csi_cleanup_20260715.tar.gz`。归档不是活动代码入口。
