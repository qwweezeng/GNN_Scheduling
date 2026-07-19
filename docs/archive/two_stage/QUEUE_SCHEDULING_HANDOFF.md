# Queue-aware GAT 研究交接摘要

更新时间：2026-07-11

早期可复制上下文见同一归档目录的 `NEXT_CHAT_CONTEXT.md`；该快照现已过时，当前请使用根目录教学文档。

## 当前研究问题

目标是在干扰耦合的无人机链路网络中实现 link-level 队列感知调度。每时隙服务率由实际 SINR 和 Shannon 公式计算，希望通过近似求解

```text
max sum_i Q_i(t) * R_i(t)
```

获得队列稳定性和长期 QoS，同时保持在线分布式执行。

## 已完成的基础修复

1. 队列语义已统一为 link-level，并在每个时隙把动态 `Q(t)` 同步到模型输入。
2. 所有算法在同一 Monte Carlo run 中共享信道和到达轨迹。
3. 到达模型改为 link-level hotspot，`sum_i lambda_i = Lambda`。
4. 已加入 backlog、Little's-law delay、departures/slot 和集中式 MWS-Greedy baseline。
5. 速率代码采用 `R_i = pi_i * log2(1 + SINR_i)`；自身 `pi_i` 只乘一次，其他 `pi_j` 只进入干扰项。

## 端到端模型尝试

### v2

- 输入 `[SNR,Q]`，增加 local input skip，loss 使用 packet/slot 单位的 `sum Q*R`。
- 缩小训练队列范围到 `0..20` 后优于 GAT-base，但主要行为接近激活全部非空队列。
- 队列交换和信道交换敏感性很弱，与真正 MWS 仍有明显差距。

### v3

- 改用相对队列、stochastic STE、hard utility checkpoint 和阈值校准。
- 正式仿真中 GAT 在低负载优于 Random，但高负载接近 Random；MWS-Greedy 显著领先。
- 共享静态测试上，端到端 GAT 的 MWS 效用大约只有 MWS-Greedy 的 11%，且激活链路明显过多。
- 输出梯度分析显示：同一输入在不同随机联合调度下会收到相反监督；连续松弛倾向全开，STE 硬前向倾向全关。
- 原 checkpoint：`results/models/gat_queue_mws_v3_numUAV=10.pth`
- 存档副本：`results/models/archive/gat_queue_mws_v3_end_to_end_numUAV=10_20260711.pth`

## 当前递进诊断

脚本：`experiments/queue_model_diagnostics.py`

控制顺序：

1. `A_base`：SNR 输入和 `sum R`，确认基础 GAT 可学习。
2. `B_constant_q`：v3 双通道结构但 Q 恒为 1，仍优化 `sum R`。
3. `C_variable_q_unweighted`：变化 Q 进入模型但不进入 loss。
4. `D_weighted_continuous`：变化 Q 和连续 `sum Q*R`。
5. `D_weighted_ste`：变化 Q 和 stochastic STE 的 `sum Q*R`。

smoke 和正式 500/64/64、30 epoch、seeds 42/43/44 的诊断均已完成。正式报告：

```text
results/diagnostics/queue_learning_20260711_031525/diagnostic_report.json
```

正式测试中位结果：

| 控制 | Utility | 激活链路 | MWS-Greedy 比例 | 结论 |
|------|---------|----------|-----------------|------|
| A-base | 16.84 | 4.14 | - | 基础 GAT 可稳定学习 |
| B-constant-Q | 16.69 | 4.09 | - | 双通道和 local skip 不构成根本障碍 |
| C-variable-Q-unweighted | 16.73 | 4.09 | - | 变化 Q 输入本身不破坏信道学习 |
| D-weighted-continuous | 20.25 | 42.70 | 8.9% | 退化为几乎全部非空链路激活 |
| D-weighted-STE | 0 | 0 | 0% | 三个 seed 均退化为全关 |

## TwoStage-GAT-v1

若端到端 D 组的 MWS-Greedy 效用比例中位数低于 30%，或不能稳定超过 Random，则自动转向两阶段模型。

1. 冻结基础 GAT，使用 SNR 和干扰边生成信道嵌入 `h_i` 与信道分数 `c_i`。
2. 本地队列通过一跳邻居最大值归一化为 `q_rel`。
3. 显式构造 `priority_seed = q_rel * softplus(c_i)`。
4. 浅层 Queue-GAT 使用 `[h_i,c_i,q_rel,priority_seed]` 输出最终动作概率。
5. 离线集中式 MWS-Greedy 只提供教师标签；在线只运行邻居 GAT 交换、本地阈值和空队列屏蔽。

正式模型路径：

```text
results/models/gat_queue_two_stage_v1_numUAV=10.pth
```

规则诊断对照 `QxChannel-LocalMax` 使用同一冻结信道分数，但不使用 Queue-GAT。它用于判断学习式第二阶段是否优于简单局部排序，不是最终方案。

静态测试中 TwoStage-GAT-v1 达到 MWS-Greedy 效用的 35.9%，教师动作 accuracy=0.921、F1=0.384。交换正队列后平均概率变化约 6.37%，证明已经使用队列；仅降低单条链路 SNR 输入后概率变化约 0.011%，说明当前输入响应很弱，但该诊断没有同步修改物理信道矩阵和干扰边。静态快照中模型平均激活 8.06 条，而教师平均激活 2.34 条；动态 lambda=1 时模型又只激活约 0.3 条，说明它对队列占用密度的泛化不稳定。

## 首轮多时隙结果

结果目录：

```text
results/queue_exp20260711_0509_s42_mc3/
```

设置为 1000 slots、link-level hotspot、MC seeds 42/43/44：

| 方法 | lambda=1 | lambda=2 | lambda=5 | lambda=10 |
|------|----------|----------|----------|-----------|
| v3 GAT | 0.470 | 0.266 | 0.117 | 0.0645 |
| TwoStage-GAT-v1 | 0.308 | 0.276 | 0.216 | 0.121 |
| QxChannel-LocalMax | 0.991 | 0.615 | 0.209 | 0.0983 |
| MWS-Greedy | 0.998 | 0.986 | 0.614 | 0.352 |

TwoStage-GAT-v1 在高负载明显优于 v3，但低负载激活不足；平均激活数会随负载由约 0.3 增至 10.7。LocalMax 始终约 1 条激活，低负载几乎最优，但高负载受单赢家瓶颈限制。这个结果支持两阶段分解，但当前学习式第二阶段尚未学好局部并发度。

## 下一步判断

1. 不围绕 bias、temperature 或多惩罚项继续搜索。
2. 下一版把 Queue-GAT 的任务从独立二分类改为学习局部 priority 或可并发组，使其显式学习局部竞争和空间复用强度。
3. 保留一跳 LocalMax 作为结构有效性的下界验证，但避免最终方案退化为传统迭代 Local Greedy。
4. 改进信道敏感性测试和教师数据分布；当前模型对队列明显敏感，对单链路 SNR 变化不够敏感。
5. 教师训练数据需要按低/中/高非空队列比例分层，而不是让 `1..num_links` 的每个精确数量各占相同比例；后者会使动态低负载常见的稀疏状态在整体训练集中占比较低。
6. 结构确定后再运行 5000-slot 正式论文仿真；当前 1000-slot 结果用于方向验证。
