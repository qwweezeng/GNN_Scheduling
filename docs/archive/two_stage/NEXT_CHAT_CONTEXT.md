# 可直接转发给新 Chat 的项目上下文

下面整段可以直接作为新 Chat 的第一条消息使用。

---

我正在 `/Users/zqz/Documents/Codex/GNN_Scheduling` 中研究无人机集群网络的 link-level queue-aware 链路调度。请先阅读：

1. `README.md`
2. `docs/CHANGELOG.md`
3. `docs/archive/two_stage/QUEUE_SCHEDULING_HANDOFF.md`
4. `experiments/queue_model_diagnostics.py`
5. `models/two_stage.py`

## 研究目标与约束

每个时隙选择二元链路调度动作 `pi_i`，实际服务率由联合调度产生的 SINR 和 Shannon 公式计算，希望近似求解：

```text
max sum_i Q_i(t) * R_i(t)
```

目标是改善长期 queue backlog、delay、公平性和网络稳定性。在线部署必须保持分布式：链路只能通过 GAT 与冲突邻居交换信息，最终动作在本地完成；集中式 MWS-Greedy 只能用于离线教师和性能 baseline。

速率实现已经确认：

```text
R_i = pi_i * log2(1 + SINR_i)
```

自身 `pi_i` 只乘一次，其他链路的 `pi_j` 只进入干扰项；loss 中只再乘 `Q_i`。

## 已完成的仿真和代码修复

1. 队列统一为 link-level，并在每时隙把真实动态 `Q(t)` 同步到模型输入。
2. 同一 Monte Carlo run 中所有调度器共享信道和 arrival trace。
3. 到达模型为 link-level hotspot：20% 链路承担 70% 总流量，且 `sum_i lambda_i = Lambda`。
4. 已加入 backlog、Little's-law delay、departures/slot、平均激活链路数和集中式 MWS-Greedy。
5. GAT-base、v3 GAT-q、Random、Greedy、MWS-Greedy 使用不同模型/调度路径，已排除模型路径混用。

## 已失败的端到端路线

端到端 GAT-q 直接输入 `[SNR,Q]` 并最大化 `sum Q*R`。v2/v3 尝试过 local input skip、packet/slot 单位、相对队列、stochastic STE、hard checkpoint 和阈值校准。

正式递进控制使用共享 500/64/64 快照、30 epoch、训练 seeds 42/43/44：

| 控制 | 测试 Utility | 激活链路 | 相对 MWS-Greedy |
|------|-------------:|---------:|----------------:|
| A：基础 SNR GAT，目标 `sum R` | 16.84 | 4.14 | - |
| B：双通道但 Q 恒为 1，目标 `sum R` | 16.69 | 4.09 | - |
| C：变化 Q 输入但不加权，目标 `sum R` | 16.73 | 4.09 | - |
| D：变化 Q，连续 `sum Q*R` | 20.25 | 42.70 | 8.9% |
| D：变化 Q，STE `sum Q*R` | 0 | 0 | 0% |

结论：基础 GAT、双通道、local skip、变化 Q 输入都可以正常训练。真正失败的是 weighted-MWS 的联合动作信用分配：连续松弛稳定走向几乎全开，随机 STE 稳定走向全关。不要再把主要精力放在初始 bias、temperature 或堆叠惩罚项上。

正式诊断报告：

```text
results/diagnostics/queue_learning_20260711_031525/diagnostic_report.json
```

v3 checkpoint 已存档：

```text
results/models/archive/gat_queue_mws_v3_end_to_end_numUAV=10_20260711.pth
```

## 当前 TwoStage-GAT-v1

代码：`models/two_stage.py`

### Stage 1：冻结信道 GAT

- 加载 `results/models/gat_numUAV=10.pth`。
- 输入 SNR 和干扰边。
- 输出信道空间嵌入 `h_i` 和基础调度 logit `c_i`。
- 训练 Queue-GAT 时完全冻结，防止队列监督破坏已学到的信道结构。

### 显式队列融合

当前实现先计算：

```text
q_rel_i = Q_i / max(1, Q_i, max_{j in N_i} Q_j)
priority_seed_i = q_rel_i * softplus(c_i)
```

然后拼接：

```text
[h_i, c_i, q_rel_i, priority_seed_i]
```

### Stage 2：学习式 Queue-GAT

- 一个浅层 Queue-GAT 聚合邻居融合特征。
- 独立 sigmoid 头输出每条链路动作概率。
- 本地使用固定阈值 0.5 二元化。
- 强制 `Q_i=0` 时不激活。
- 使用集中式 MWS-Greedy 动作作为离线教师，loss 为固定类别权重的单一 BCE；没有额外效用惩罚项。

正式 checkpoint：

```text
results/models/gat_queue_two_stage_v1_numUAV=10.pth
```

## TwoStage-v1 的正式结果

静态测试：

- MWS-Greedy utility ratio：35.87%。
- action accuracy：92.10%，但这是严重类别不平衡下的指标，不能单独说明性能好。
- precision：24.81%，recall：85.33%，F1：38.44%。
- 队列交换后的概率变化：6.37%，说明模型确实使用了队列。
- 单链路 SNR 输入降低后的概率变化：约 0.011%，说明模型对当前 SNR 输入通道响应很弱。但该诊断没有同步修改物理信道矩阵和干扰边，不是完整的物理一致信道反事实。
- 空队列激活率：0。
- 静态测试中教师平均激活 2.34 条，模型平均激活 8.06 条。
- 验证集最佳 checkpoint 出现在 epoch 18，验证 MWS ratio=42.08%；epoch 29 已回落到 23.22%，训练后期并非持续改善。

多时隙方向验证使用 1000 slots、MC seeds 42/43/44、hotspot arrivals：

| 方法 | Lambda=1 | Lambda=2 | Lambda=5 | Lambda=10 |
|------|---------:|---------:|---------:|----------:|
| v3 GAT throughput | 0.470 | 0.266 | 0.117 | 0.0645 |
| TwoStage-GAT-v1 | 0.308 | 0.276 | 0.216 | 0.121 |
| QxChannel-LocalMax | 0.991 | 0.615 | 0.209 | 0.0983 |
| MWS-Greedy | 0.998 | 0.986 | 0.614 | 0.352 |

TwoStage 平均激活链路数随负载变化约为 0.31、0.59、6.08、10.72。它在低负载严重欠激活，在高负载又产生较多并发和干扰；不过 Lambda=5/10 时仍明显优于 v3 和 Random。

完整多时隙结果：

```text
results/queue_exp20260711_0509_s42_mc3/
```

## LocalMax 对照证明了什么

`QxChannel-LocalMax` 使用同一个冻结信道 GAT，计算 `q_rel*softplus(c_i)`，然后每条链路与一跳邻居比较，局部最大者激活。

它在 Lambda=1 时 throughput=0.991，几乎追平 MWS-Greedy；这证明“两阶段分解 + 显式局部竞争”方向是有效的，也说明基础信道 GAT 的输出并非完全没用。

但它始终只激活约 1 条链路，Lambda 增大后吞吐很快饱和，因此不能作为最终方案。它只是结构验证 baseline，不应把最终论文方案退化成传统迭代 Local Greedy。

## 当前 TwoStage-v1 的主要问题

1. **独立 BCE 没有学习调度集合结构。** 每条链路只判断自己是否超过 0.5，没有显式学习邻居间谁让谁、应允许多少条并发。
2. **类别极不平衡。** 教师每个快照只激活约 2.34 条，而网络约有 80 条链路；92% accuracy 很大程度来自预测负类。
3. **队列占用密度分布错配。** 训练时精确非空数量 `K` 在 `1..N` 均匀采样，导致动态低负载常见的 `K<=5` 状态作为一个整体只占约 `5/N`；动态 Lambda=1 因而欠激活。
4. **信道特征可能被 Queue-GAT 忽略。** queue swap 明显改变输出，但仅修改 SNR 输入的 perturbation 几乎不改变输出。需要 channel-only、queue-only、shuffle-channel，以及同步修改 direct gain/干扰边/速率的物理一致消融确认。
5. **全局教师与局部可观测性存在差距。** MWS-Greedy 使用全局 CSI/Q，局部有限层 GAT 不一定能确定性复现同一个标签。
6. **当前 `q_rel` 仍有一次显式邻居 max 聚合。** 它是分布式的一跳操作，但严格说不是“只通过 GAT 交换邻居信息”。下一版应考虑输入本地原始/对数 Q，让相对紧迫程度由 Queue-GAT 学习。
7. **固定 0.5 阈值没有并发度语义。** 模型随队列密度从 0.3 条跳到 10.7 条激活，说明概率并没有稳定校准成调度集合。

## 希望新 Chat 优先完成的工作

请先分析和设计 TwoStage-GAT-v2，不要立即加入 GRU/LSTM，也不要优先做 bias、temperature 或多惩罚项搜索。

第一优先级：完善诊断。

1. 按 sparse/medium/dense 三类非空队列比例分层生成教师数据，而不是简单 `K~Uniform(1,N)`。
2. 统计每个密度分组的教师激活数、模型激活数、precision/recall、MWS ratio。
3. 做 `queue-only`、`channel-only`、`shuffle channel embedding` 消融，确认 Stage 1 是否真的贡献决策。
4. 检查不同固定阈值只作为诊断，不把阈值调优包装成根本改进。

第二优先级：设计简洁的结构化局部竞争。

建议让 Queue-GAT 输出局部 priority 和竞争/兼容信息，而不是直接做独立二分类。可以先研究一种仅通过 GAT 消息传递实现的 learned inhibition/competition layer，使本链路输出显式依赖邻居 priority，并能够允许多个弱冲突链路并发。训练仍尽量保持一个主监督目标，不堆叠大量互相制衡的 penalty。

必须同时保留：

- `MWS-Greedy`：集中式性能参照和离线教师。
- `QxChannel-LocalMax`：证明显式局部竞争有效的规则下界。
- `TwoStage-GAT-v1`：当前学习式基线。
- `v3 GAT`：端到端 weighted-MWS 失败基线。

新版本先做静态 MWS ratio、动作集合和敏感性验证，确认明显优于 v1 后，再运行 1000-slot 三 seed；结构稳定后才运行 5000-slot 正式论文仿真。

代码风格要求：这是研究生论文实验代码，优先简洁、可读、少参数，不要继续堆企业式接口或大规模包装。每次修改都同步维护 `README.md` 和 `docs/CHANGELOG.md`，并详细解释修改目的、参数含义、预期作用和需要观察的指标。

---
