# 直接交给下一 Chat：CSI 验证与十天投稿闸门

> **文档状态：2026-07-15 交接快照。** 其中“冻结模型只做止损”的计划已被后续
> 物理信道修正和接口根因分析取代。当前唯一入口为
> [`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`](MOBILE_BLOCK_FADING_CSI_TUTORIAL.md)。

> 用途：把本文直接发给负责移动/block-fading、不完美 CSI 验证的新 Chat。
> 更新时间：2026-07-15。用户报告距离投稿截止约十天，因此必须先做能够快速
> 给出“继续或停止”结论的实验，不能重新展开长期模型搜索。

## 0. 下一 Chat 先做什么

先读以下四个入口，不需要重新翻全部历史聊天：

1. `docs/CSI_CHAT_HANDOFF_20260715.md`：本文件，当前任务和止损条件；
2. `docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`：移动 CSI 仿真定义和已有 E9--E12；
3. `docs/FIXED_CSI_QUEUE_BENCHMARK_TUTORIAL.md`：固定 CSI 最新 5-seed 结果；
4. `docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`：当前算法公式与分布式接口。

完成上述事实核对后，按
`docs/MOBILE_CSI_GAT_VS_PURE_PLAN_20260715.md` 执行当前两天配对闸门。
该计划把本文件的要求具体化为可检验的逐 seed 联合胜利标准，并列出了现有移动
脚本需要的最小修改；若它与更早移动教程中的“下一步”冲突，以该 dated plan 为准。

2026-07-15 第一层 Frozen-GAT/Pure 正式配对已经完成，结果和逐实验解释见
`docs/MOBILE_CSI_FROZEN_GAT_VS_PURE_TUTORIAL.md`。高负载三种 imperfect CSI
均通过天然 resilience 闸门，但低负载未通过，且相对 latest 仍有20%--31%损失；
因此随后完成同架构 CSI-augmented Spatial GAT，而不是停止或直接跳到复杂 STGNN。

2026-07-15 第二、三层也已完成：非坍缩的 teacher-prior 单帧增强版降低了 imperfect
CSI 的 prior reconstruction error，但没有稳定超过 Frozen-GAT；一层永久链路 GRU
残差在独立验证上未超过零残差起点，最终40-run结果与单帧增强版逐项相同。详见
`docs/CSI_AUGMENTED_SPATIAL_GAT_TUTORIAL.md` 和
`docs/TEMPORAL_RESIDUAL_GAT_TUTORIAL.md`。当前已按止损规则停止扩大 temporal GNN
搜索，下一优先级回到论文写作和正式图表。

**后续重要纠正：** channel-only审计发现Frozen-GAT阈值动作在1200个正式状态中全部
全开；强LocalMax/物理加入阶段遮蔽了该问题。当前纯Rayleigh模型也显著放大stale误差；
Rician 5 dB将stale dB误差从5.04降至3.42 dB，但stale-only GRU仍未超过零残差。
因此停止条件仅适用于Frozen-teacher logit重建，不应扩大成所有物理信道预测均无效。
新的事实入口为 `docs/CHANNEL_MODEL_AND_STAGE_MASKING_TUTORIAL.md`。

下一 Chat 的第一优先级不是设计预测网络，而是回答：

> 在相同 imperfect/stale CSI 观测下，GAT-assisted LocalPhysicalReuse 是否比
> 不使用 GAT 的 PurePhysicalReuse 更稳健？

要求两天内给出冻结会议 GAT 的零改造基线结论。若答案是否定或高度依赖 seed，
不能把它误写成“图学习路线失败”；它只说明原 checkpoint 没有跨 CSI 分布的天然
resilience。此时进入一次同架构 CSI 增强重训和一个最小时间残差 GAT。只有这两项
也失败，才停止扩大时空模型并回到论文写作。

## 1. 用户、论文和时间约束

- 用户是通信专业博士生，learning 术语需要先解释再使用；
- 会议基础：`docs/GNN-based link scheduling for dense UAV swarm.pdf`；
- 会议版 GAT-base 在固定信道、不含队列时最大化网络容量；
- 目标：约 17 页 IEEE TNSE 专刊论文，主题为 AI-driven low-altitude networks；
- 期刊扩展：动态链路队列、MWS、分布式条件协商、空间复用和后续移动 CSI；
- 用户报告只剩约十天，当前需要“足够成立且能写完”的版本，不追求无止境刷新性能。

研究记录必须使用“动机—定义—实现—结果—分析—下一步”，不能只列脚本和数字。
不要用 bias、temperature、STE、额外惩罚项等补丁掩盖结构问题。

Python 环境统一为：

```bash
conda run --no-capture-output -n pytorch python -u <script.py> ...
```

## 2. 当前暂时冻结的算法

正式调用类是：

`models/local_physical_reuse.py::GATExpectedRateLocalPhysicalReuseScheduler`

当前版本仍是 LocalPhysicalReuse-v2，结构为：

1. 会议版 channel-only GAT 输出旧调度 logit；
2. `sigmoid(logit)` 被解释为邻居活动先验；
3. 用活动先验和有向干扰增益计算期望有效速率 `C_tilde_i`；
4. 每时隙以 `Q_i * C_tilde_i` 做多个局部 LocalMax，形成真实 0/1 骨架；
5. 其余链路以骨架为条件，解析计算加入收益；
6. 正收益候选再做一次局部最大竞争，并行加入。

最后两步是 Physics-Informed Message Passing：消息包含动作、有向干扰、队列和
局部边际收益；速率和损害使用真实 SINR 公式，不用黑盒替代。

关键消融是：

`models/local_physical_reuse.py::PurePhysicalReuseScheduler`

Pure 与正式方法的物理协商完全相同，只把骨架权重改为
`Q_i * C_i_solo`，不使用 GAT。CSI 阶段必须同时运行它，不能只和 random 比较。

## 3. 今天对代码接口做了什么

### 3.1 队列仿真器

`simulator/queue_simulator.py` 现在支持：

- 精确 FIFO 包完成时延；
- 可选 deadline 和有限缓存；
- deadline drop、overflow drop、未决包分别统计；
- 区分 physical service 与完整成功包 goodput；
- `successful_delivery_ratio` 表示有限窗口内完整成功包比例。

当前模型筛选默认关闭 deadline 和有限缓存。deadline 接口没有删除，只留待后期
真实性实验使用。

### 3.2 正式固定 CSI 实验入口

`experiments/exp_fixed_csi_qos.py` 当前默认值为：

- 10000 slots；
- seeds `42,43,44,45,46`；
- 无 deadline、无限缓存；
- 到达率 `0.5,1,2,2.2,2.4,2.6,2.8,3,3.2`。

默认主图只保留：

1. delivered throughput；
2. average queue backlog；
3. 无 deadline 完成包平均时延。

无 deadline 时，慢包可能在仿真结束仍未完成。`utils/plot_queue_benchmark.py`
只有在 `successful_delivery_ratio >= 0.99` 时才画平均时延点；不稳定点靠 backlog
和 delivered throughput 判断，不能用完成包条件时延比较。

### 3.3 数据和结果目录

最新 trace：

`datasets/queue_traces/fixed_numUAV=10_tx=10dBm_csi=perfect_slots=10000_mc=5_boundary.pt`

最新主结果：

`results/queue/fixed_csi/main_N10_10000slots_5seeds_no_deadline_boundary/`

其中包含 `config.json`、`metrics.json`、`per_seed_metrics.json`、轨迹和三组 PNG/PDF。

20000-slot deadline 结果仍保存在：

`results/queue/fixed_csi/main_N10_20000slots_qos_deadline/`

它不是当前筛选口径，不要和无 deadline 主结果混成同一张图。

## 4. 固定 rho 到底试没试

今天的正式动态结果仍使用旧会议 GAT 标量接口，没有换成训练后的 rho-GAT。

但固定 channel-only 速率保留系数已经做了比训练网络更强的 target-sufficiency
审计。定义真实保留系数为：

`rho_i(A) = R_i(A) / C_i_solo`。

对每张图使用 128 组尺度变化队列，通过 PurePhysicalReuse 的真实活动集合直接
估计离线 oracle `bar_rho_i`。该 oracle 使用部署时不可获得的大量队列和真实结果，
因此性能上限高于任何存在预测误差的 rho-GAT。

30 张 N=10 图的结果：

| 骨架优先级 | 相对集中式 greedy 快照效用 |
|---|---:|
| Pure: `Q*C_solo` | 87.28% |
| 旧 GAT 期望速率 | 82.53% |
| Oracle: `Q*C_solo*bar_rho` | 87.69% |

Oracle 只提高 0.41 个百分点，并且仅 16.7% 未见队列状态严格胜过 Pure。结论是：

> 固定单标量 rho 本身不是充分状态，不值得再训练一个上限更低的 rho-GAT。

原始证据：

`results/queue/fixed_csi/channel_rho_target_sufficiency_N10.json`

入口：

`experiments/diagnose_channel_rho_prior.py`

完整解释：

`docs/GAT_CHANNEL_PRIOR_REDESIGN_TUTORIAL.md`

## 5. 另外两次 GAT 读出尝试及止损

为了确认是否必须保留丰富 channel embedding，今天还做了两个诊断版本。它们均
不是当前 proposed，不要在 CSI 主线中加载。

### ChannelContext-v1

- 冻结会议 GAT encoder；
- 每条链路读取自己的 48 维 channel embedding、局部相对队列和 solo rate；
- 浅读出头输出条件 rho；
- 用真实物理反事实二选一标签训练，不模仿集中式最终动作。

结果：Pure 为集中式 greedy 的 88.68%，v1 为 87.88%。失败。

### ChannelContext-v2

- 在 v1 前增加一轮队列条件 GAT 消息，看到具体竞争邻居；
- 反事实判断准确率略有提高；
- 200 个快照最终效用仍从 Pure 的 89.04% 降到 87.87%。

因此已经按预设规则停止，不再增加层数、hidden dimension 或 loss 项。

诊断代码：

- `models/channel_context_reuse.py`；
- `train/train_channel_context_readout.py`；
- `train/train_channel_context_negotiation.py`。

结果：

- `results/queue/fixed_csi/channel_context_v1/`；
- `results/queue/fixed_csi/channel_context_v2/`。

## 6. 最新无 deadline、5-seed 固定 CSI 结果

N=10、10000 slots、5 个配对 seeds：

| Lambda | GAT 完成率 | Pure 完成率 | GAT backlog | Pure backlog | GAT/Pure 稳定 seeds |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 99.75% | 99.82% | 0.0115 | 0.0081 | 5/5, 5/5 |
| 2.2 | 98.79% | 98.34% | 0.0603 | 0.0706 | 5/5, 4/5 |
| 2.4 | 94.71% | 93.91% | 0.2649 | 0.2884 | 2/5, 2/5 |
| 2.6 | 89.43% | 89.24% | 0.5547 | 0.5617 | 0/5, 1/5 |
| 2.8 | 84.66% | 84.82% | 0.8571 | 0.8399 | 0/5, 1/5 |
| 3.0 | 80.23% | 80.16% | 1.1837 | 1.1797 | 0/5, 0/5 |

Backlog 单位为 Mbit/link。

解释边界：

- Lambda=2 时两者稳定，Pure backlog/时延更低；
- Lambda=2.2 时 GAT 修复了一个 Pure 失败的困难 seed，但平均优势仍小；
- Lambda=2.4 以后两者总体都越过边界；
- GAT 没有一致、显著地把全体稳定边界向右推移；优劣方向随拓扑 seed 变化。

因此固定 CSI 的 GAT 相对 Pure 问题先搁置，不再消耗投稿前时间。

## 7. CSI Chat 必须做的最小实验矩阵

已有移动仿真在 `experiments/evaluate_mobile_block_fading_csi.py`，状态与定义见
`docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`。已经实现 permanent directed-link
identity、真实信道与观测 CSI 分离、latest/stale/large-scale/quantized 条件。

下一 Chat 必须在同一个真实移动、到达和观测轨迹上做 2×4 消融：

| 调度骨架 | latest | one-block stale | current large-scale | quantized large-scale |
|---|---:|---:|---:|---:|
| GAT-assisted LocalPhysicalReuse-v2 | 必跑 | 必跑 | 必跑 | 先用 4-bit |
| PurePhysicalReuse | 必跑 | 必跑 | 必跑 | 同样 4-bit |

所有方法实际服务必须在当前真实信道上结算；只能改变调度器看到的观测 CSI。

快速设置建议：

- N=10；
- 5 个配对 seeds；
- 先用 1000 slots；
- 无 deadline、无限缓存；
- 至少测试边界负载 Lambda=2.2 和压力负载 Lambda=4；
- 先固定一个具有中等相关性的移动场景，例如 5 m/s、5 ms block；
- 指标先保留 delivered throughput、backlog、physical service；时延只有完成率足够高
  才解释。

闸门问题不是“stale 是否比 latest 差”——这一点已经证明；而是：

`Delta_GAT = metric(GAT-assisted) - metric(Pure)`

是否在 imperfect CSI 下比 perfect CSI 更稳定、更一致。

## 8. 两层继续/止损判据

### 8.1 第一层：冻结会议 GAT 的天然鲁棒性

Frozen-GAT 相对 Pure 的正向证据为：

1. GAT-assisted 相对 Pure 在至少一种 imperfect CSI 条件下出现方向一致的增益；
2. 增益不能只由一个 seed 驱动，优先要求至少 4/5 配对 seeds 同方向；
3. 同时改善 delivered throughput/backlog，不能只提高 physical service；
4. 若声称预测有效，必须至少优于 naive stale 和 current large-scale 两个参考。

若第一层未通过：

- 不再声称原会议 checkpoint 天然对抗 CSI 变化；
- 但不立即停止 GAT 路线；
- 先用相同会议 GAT 结构、移动/扰动 CSI 输入和当前真实信道物理损失做一次
  CSI-augmented fine-tuning；
- 再测试一个以会议空间 GAT 为骨架、沿永久 link identity 增加一层本地 GRU
  时间状态和零初始化 logit 残差的最小 STGNN-like 候选；
- 两者均不改变 LocalMax、条件物理复用或最终队列动作定义。

### 8.2 第二层：时间建模是否真的有独立价值

Temporal-Residual GAT 至少需要：

1. 相对 Pure 在至少 4/5 seeds 同时改善 delivered throughput 和 backlog；
2. 相对 Frozen-GAT 也满足同方向配对优势；
3. 相对仅增加训练数据的 CSI-augmented Spatial GAT 仍有增益，否则只能归因于
   训练分布覆盖，不能声称时间维度贡献；
4. 若把模块表述为信道预测/恢复，还要优于 naive stale 和 current large-scale。

只有第二层也未通过时：

- 不再训练复杂 temporal GNN、RNN 或多步预测器；
- 不继续调整 fixed-CSI GAT 骨架；
- 将 LocalPhysicalReuse-v2 作为 GAT-assisted physics-informed hybrid 投稿；
- Pure 以 `analytic-prior ablation without learned GAT prior` 放入一个消融表，不必
  出现在每张主图，但不能完全隐去；
- CSI 风险可作为敏感性诊断或 future work，不阻塞主稿。

## 9. 论文可以怎样写，不等待 CSI 成功

建议主线：

1. 会议 GAT-base 学习密集 UAV 有向干扰图；
2. 会议模型不理解动态队列，直接 GAT-q 又不能可靠形成联合 0/1 集合；
3. 期刊提出 queue-aware physics-informed graph negotiation；
4. GAT 提供慢信道先验，队列提供快速紧急性，确定性物理消息形成条件空间复用；
5. 相比 GAT-base、EndToEnd-GAT-q、random、硬冲突方案和集中式参考，期刊方案
   在动态队列和规模扩展上有明确贡献；
6. Pure 消融表明物理协商是可靠地基，GAT 的增量收益具有负载/CSI 条件性。

不要声称当前模型具有严格 MWS 稳定域定理，也不要把集中式 MWS-Greedy 称为
数学最优或上界。当前算法更准确的名称是 queue-aware，而非 deadline-guaranteed。

## 10. 活动文件、归档和不要误用的内容

活动核心：

- `models/local_physical_reuse.py`：Proposed、Pure、LocalMax 消融；
- `simulator/queue_simulator.py`：固定 CSI 队列与 QoS 接口；
- `utils/fixed_csi_benchmark.py`：配对 trace 和聚合；
- `utils/plot_queue_benchmark.py`：正式画图；
- `experiments/exp_fixed_csi_qos.py`：固定 CSI 正式入口；
- `experiments/evaluate_mobile_block_fading_csi.py`：移动 CSI 入口；
- `simulator/mobile_block_fading.py`：移动、persistent identity、观测/真实状态。

历史 two-stage 源码已归档：

`archives/two_stage_exploration_source_20260711_14.tar.gz`

今天的 rho/v1/v2 是诊断失败路线，代码暂时保留用于审计，但不得被误认为当前模型。
其独立可恢复归档为：

`archives/gat_prior_redesign_diagnostics_20260715.tar.gz`

对应 manifest 和 SHA-256 位于同目录。活动副本没有删除，便于复现实验。
没有删除用户文件，也不要执行工作区级清理、reset 或批量删除。

## 11. 验证状态

2026-07-15 当前检查：

- 23 个 `unittest` 全部通过；
- 关键 Python 文件 `py_compile` 通过；
- `git diff --check` 通过；
- 20000-slot deadline 结果逐 seed 满足包守恒；
- 最新 5-seed 无 deadline 结果和 trace 已落盘。

CSI Chat 修改后至少重新运行：

```bash
conda run -n pytorch python -m unittest discover -s tests -v
conda run -n pytorch python -m py_compile \
  simulator/mobile_block_fading.py \
  experiments/evaluate_mobile_block_fading_csi.py
git diff --check
```
