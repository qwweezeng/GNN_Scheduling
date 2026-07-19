# 固定已知信道版本：冻结、清理与正式仿真计划

> 状态：**待用户确认，尚未删除任何文件。**
>
> 本文只处理 2026-07-11 以后新增的 queue-aware / two-stage 探索代码。会议版原始代码、旧数据集、其他 baseline、功率控制与 CSI 研究文件均不在删除范围内。

## 0. 一页结论

当前冻结方法为 **LocalPhysicalReuse-v2（GAT Expected-Rate）**。现有仓库的问题不是缺少实现，而是最终方法、历史失败模型、诊断 oracle、训练脚本和画图逻辑互相引用，导致无法一眼看出“论文方法究竟是哪几份代码”。

建议按以下顺序收尾：

1. 先把冻结方法提取到独立文件，并建立一个只做正式对比的固定 CSI 入口；
2. 运行单元测试和小规模结果复现，确认提取前后动作完全一致；
3. 将历史探索源码打成一个带清单和校验值的压缩归档；
4. 只有在用户确认后，删除下文列出的中间模型、训练脚本和对应测试；
5. 运行 N=10/20/50 的正式固定 CSI 仿真，并把结果统一放到 `results/queue/fixed_csi/`；
6. 正式结果完成后，再确认是否删除约 400 MB 的旧诊断结果。

这样做的原因是：当前很多新增文件尚未进入 Git，直接删除后不能依靠 Git 恢复。应先完成最终代码提取和轻量源码归档，再做物理删除。

## 1. 当前冻结模型的真实调用链

当前 LocalPhysicalReuse-v2 实际分散在以下文件中：

| 文件 | 当前作用 | 问题 |
|---|---|---|
| `models/model.py` | 会议版 `GATScheduler` 和真实 SINR/rate 计算 | 应保留，是会议版与期刊版共同基础 |
| `models/two_stage.py` | `QueueChannelLocalMaxScheduler`；其中 `raw_queue_gat_rate` 构造 v2 骨架 | 同一文件还混有已停止的 TwoStage-v1 |
| `models/parallel_reuse.py` | `LocalPhysicalReuseScheduler` 与推荐类 `GATExpectedRateLocalPhysicalReuseScheduler` | 同一文件还混有失败的 LearnedReuse-v1/v2 |
| `utils/parallel_reuse.py` | 稀疏边物理增益、局部加入收益、局部正收益赢家 | 同一文件还含 centralized exact/top-k 诊断函数 |
| `utils/mws.py` | MWS 效用与集中式 MWS-Greedy | 文件较小，可直接保留 |
| `simulator/queue_simulator.py` | 动态队列、真实 departures、backlog 和基础时延估计 | 应保留，但正式 packet delay 仍需增强 |
| `experiments/evaluate_localmax_priority.py` | 当前 v2 动态评测入口 | 是审计脚本，不是干净的论文评测入口 |
| `experiments/run_queue_experiment.py` | 数据、训练、baseline、仿真、聚合、画图全混合入口 | 可复现 GAT-q，但职责过多且包含大量旧模型分支 |
| `experiments/queue_model_diagnostics.py` | 加载 10 dBm channel-only GAT，并训练/诊断 TwoStage-v1 | 最终只需要其中的可靠 checkpoint 加载功能 |

### 1.1 建议提取后的最终核心结构

以下为计划结构，尚未执行：

```text
models/
├── model.py                         # 会议版 GAT 与物理速率计算
└── local_physical_reuse.py          # 仅含冻结 v2 及纯物理消融

simulator/
└── queue_simulator.py               # 共享动态队列仿真器

utils/
├── local_physical_reuse.py          # v2 所需 O(L+E) 局部物理操作
├── mws.py                           # 集中式全知 MWS-Greedy 参考
├── queue_benchmark.py               # paired trace、指标聚合、checkpoint 加载
└── plot_queue_benchmark.py          # 只读正式结果并画图

experiments/
├── fixed_csi_queue_benchmark.py     # 固定已知信道正式仿真唯一入口
├── train_end_to_end_queue_gat.py    # 端到端 GAT-q 失败基线的训练/加载入口
├── cache_queue_benchmark.py         # N=50 trace cache
├── benchmark_local_physical_scale.py# 复杂度实验
└── oracle_reuse_headroom.py         # 仅用于小规模结构上限诊断

tests/
├── test_local_physical_reuse.py     # 冻结算法公式与动作回归
├── test_queue_metrics.py            # departures/backlog/delay 指标
├── test_hotspot_queue_sampling.py   # 到达模型
├── test_queue_trace_reproducibility.py
├── test_rate_and_queue_features.py
└── test_artifact_physics.py

results/queue/fixed_csi/<run_id>/
├── config.json                      # 完整参数和 checkpoint 身份
├── metrics.json                     # MC 均值、标准差、95% CI
├── per_seed_metrics.json            # 每个 seed 的标量结果
├── trajectories.npz                 # 绘图所需轨迹，不把大数组塞入 JSON
└── figures/                         # PNG + PDF
```

`tests/` 只放“代码是否正确”的自动检查；正式仿真入口放 `experiments/`；画图逻辑放 `utils/plot_queue_benchmark.py`。画图不是 test，因为图的数值变化不代表代码必然错误。

## 2. 建议保留的方案与代码

### 2.1 正式算法和必要消融

| 名称 | 定义 | 论文作用 |
|---|---|---|
| **LocalPhysicalReuse-v2** | GAT 活动先验转期望有效速率；`Q × expected-rate` 形成 LocalMax 骨架；一轮局部物理正收益复用 | 期刊正式方法 |
| **PurePhysicalReuse** | 用 `Q × solo-rate` 形成同样骨架和同样复用步骤，不使用 GAT 先验 | 判断 GAT 信道学习是否真的贡献性能 |
| **LocalMax-only** | 只形成局部骨架，不做第二步空间复用 | 判断条件复用步骤的独立贡献 |

### 2.2 公平对比基线

| 名称 | 是否用 Q | 信息范围 | 保留原因 |
|---|---:|---|---|
| Conference GAT | 只用“队列是否非空”，不看队列大小 | 固定轮 GAT 邻居消息 | 会议版原方法；必须展示期刊扩展相对基础版本的收益 |
| End-to-End GAT-q | 是 | 固定轮 GAT 邻居消息 | 直接把 Q 输入并优化 `sum(QR)` 的自然基线；也是本文两阶段设计的失败动机 |
| Random-nonempty | 只用非空 mask | 无 | 下界与正确性 sanity check |
| SNR-SingleLink | 只用非空 mask | 全局选择最高 SNR 一条 | 单发保守基线；当前代码名 `Greedy` 容易误导，正式图中应改名 |
| Queue×SoloRate LocalMax | 是 | 一跳冲突邻居 | 不使用 learning 的强分布式解析基线 |
| Centralized MWS-Greedy | 是 | 全局 Q、信道和重复全局效用计算 | 集中式全知参考，不是严格数学上界 |

### 2.3 关于“性能上界”的严格命名

当前 `CentralizedMWSGreedyScheduler` 只是用全局信息逐条尝试加入，再做删除改进。它可能陷入局部最优，因此不能写成 `optimal` 或 `upper bound`。

论文中建议称为：

> Centralized full-information MWS-Greedy reference.

若确实需要严格上界，只能补充以下之一：

1. 在很小的裁剪冲突子图上穷举或整数规划求精确 MWS；
2. 使用“所有链路均按无干扰 solo rate 服务”的松上界，但该上界通常很松，不能代表可实现调度性能。

主图优先使用集中式 MWS-Greedy，不在名称上过度声称。

## 3. DistGCN 是否现在接入

论文 `Zhao et al., Link Scheduling Using Graph Neural Networks, TWC 2023` 的分布式方法为：

1. GCN 根据冲突图学习一个拓扑缩放量 `z_i`；
2. 将原始链路效用 `u_i` 改写为 `z_i u_i`；
3. 再用 distributed local greedy solver 生成最大权独立集近似。

它很适合作为相关工作的思想对比，因为它同样是“learning 提供结构信息，可靠规则产生集合”，而不是独立概率直接阈值化。

但它不能未经说明直接接到本仿真中：

- DistGCN 把冲突边视为硬排斥，边两端绝不能并发；我们的 SINR 模型允许弱冲突链路条件复用；
- 作者代码输入是无向 MWIS adjacency 和节点效用；我们的图含有向物理干扰、共享 UAV 约束和连续 Shannon rate；
- 原代码是旧 TensorFlow 管线。当前 `pytorch` Conda 环境有 TensorFlow 2.20，但缺少 `scipy`，直接 import 已失败；此外还需核对 `networkx` API 与 checkpoint 兼容性；
- 作者随仓库提供了很多 MWIS checkpoint，但训练图分布与本 UAV 冲突图是否匹配尚未验证。

因此建议设一个低成本接入闸门，而不是立即承诺加入正式主图：

1. 把我们冲突图对称化为 hard-conflict adjacency；
2. 使用作者建议的 `u_i=Q_i r_i`，其中 `r_i` 暂取 solo rate；
3. 原样加载作者的 1-layer GCN-LGS checkpoint，不重新定义其模型；
4. 在 N=10 的 20 个共同静态快照上测试是否能正常输出独立集，以及是否优于作者自己的 LGS；
5. 只有通过以上闸门，再进入动态队列主图。

若失败，应在论文 related work 中比较思想，不把一个经过大幅改造的复现结果冒充原作者方法。当前正式固定 CSI 仿真不应等待 DistGCN。

## 4. 正式仿真图：哪些合理，哪些需要修正

### 4.1 图 A：共同快照上的瞬时 MWS 决策质量

目的：直接回答每个方法在同一个 `(Q,H)` 状态下，选择的联合集合有多好。

做法：从独立验证轨迹中采样共同的信道与队列快照，对所有方法输入完全相同的状态，计算

$$
U_{\mathrm{alg}}=\sum_i Q_i\mu_i(\mathbf a_{\mathrm{alg}}),
$$

并报告相对集中式参考

$$
\rho_U=\frac{\sum_s U_{\mathrm{alg}}(s)}{\sum_s U_{\mathrm{MWS\text{-}Greedy}}(s)}.
$$

这里使用“先汇总再相除”，避免空队列快照造成逐样本比值不稳定。建议按 sparse/medium/dense 非空队列比例分组，并报告均值与 95% 置信区间。

**重要修正：** 不应直接比较各算法动态运行后的原始 `sum(QR)` 曲线，因为各算法的 Q 已经不同；失败方法积压更大，反而可能得到更大的 `sum(QR)`。动态阶段应看真实服务和 QoS。

### 4.2 图 B：真实吞吐或 departure 随 Lambda 变化

横轴：全网总到达率 `Lambda`（packets/slot）。

纵轴建议同时提供：

- `departure_total_rate_pps` 或换算后的 Mbps：真实服务能力；
- `throughput_ratio=departures/arrivals`：是否跟得上到达，只作为辅助稳定性指标。

不建议只画 throughput ratio，因为过载时所有方法都会自然下降，容易被误读为算法准确率。

### 4.3 图 C：平均队列积压随 Lambda 变化

对应 MAB Fig. 4。使用时间平均、链路平均 backlog：

$$
\bar Q=\frac{1}{TL}\sum_{t=1}^{T}\sum_{i=1}^{L}Q_i(t).
$$

同时建议补一条 hotspot backlog，因为当前 20% 热点链路承担 70% 业务；只看全链路平均会被大量低业务链路稀释。

### 4.4 图 D：平均包时延随 Lambda 变化

对应 MAB Fig. 5。但当前代码的 `average_delay_ms` 是 Little 定律估计：

$$
\bar W\approx \bar Q/\bar D,
$$

不是对每个成功发送包逐个记录的真实排队时延。

正式论文建议增加 FIFO packet/cohort 记账，至少报告：

- 成功发送包的平均排队时延；
- 95% 分位时延，用来体现 QoS 尾部风险。

MAB 的 200 Mbit buffer 和 2000 ms deadline 不能机械照搬：我们当前正式仿真只有 1000 slots × 1 ms，2000 ms deadline 根本不会触发。若画丢包率，必须先定义与 UAV 业务相符的有限 buffer 和 deadline，并把仿真时长延长到足以观测丢包；否则先不画丢包率。

### 4.5 建议额外加入的两张图

1. **稳定边界/队列增长率图：** 展示 `queue_growth_slope` 随 Lambda 变化，比单一阈值稳定判定更透明；
2. **规模与开销图：** N=10/20/50 下报告链路数 L、冲突边数 E、每时隙 CPU 时间和相对 MWS 性能，支撑“密集且可扩展”的论文主张。

## 5. 正式实验设置建议

第一轮正式固定 CSI 结果建议：

| 项目 | 建议 |
|---|---|
| UAV 数 | 10、20、50 |
| TX power / CSI | 10 dBm / perfect CSI |
| channel | 每个 seed 固定一个信道图，固定 CSI 主线不混入 block fading |
| arrivals | 现有 link-hotspot Poisson；20% links 承担 70% traffic |
| Lambda | 先用短跑扫描找到各 N 的稳定转折区，再为正式图选择 6--8 个点；不能盲目对所有 N 使用完全相同的负载区间 |
| MC seeds | 诊断至少 3；正式图建议 10，若 N=50 过慢则先 5 并报告 95% CI |
| slots | 预热 1000 slots + 统计 5000--10000 slots；时延/丢包实验应更长 |
| paired fairness | 同一 N、Lambda、seed 下所有方法共享完全相同的信道图、热点集合和到达 trace |
| outputs | 标量 JSON、per-seed JSON、压缩轨迹 NPZ、PNG/PDF 图 |

不同 N 下候选链路数和容量不同，因此相同 `Lambda` 不一定表示相同负载强度。论文可以同时展示：

- 绝对 `Lambda`，便于复现实验；
- 归一化 offered load `Lambda / Lambda_ref`，其中 `Lambda_ref` 可由集中式 MWS-Greedy 的稳定边界估计。

## 6. 待删除源码审批清单

以下文件均是 2026-07-11 后的历史探索路线。建议先打包到 `archives/two_stage_exploration_source_20260711_14.tar.gz`，生成 manifest 和 SHA-256，再从活动源码树删除。

### 6.1 历史模型文件

| 文件 | 内容/删除理由 |
|---|---|
| `models/anchor_join.py` | 全局 anchor 后顺序 join；已停止 |
| `models/anchor_one_join.py` | anchor + 单次 join 变体；已停止 |
| `models/parallel_mpnn.py` | ParallelMPNN-v1；动态队列失败路线 |
| `models/parallel_mpnn_v2.py` | ParallelMPNN-v2；历史补丁版本 |
| `models/parallel_mpnn_v3.py` | ParallelMPNN-v3；local best-response 变体 |
| `models/parallel_mpnn_v4.py` | ParallelMPNN-v4；local best-response 变体 |
| `models/physical_channel_estimator.py` | 旧 PhysicalNegotiation 的信道估计器，不是未来移动 CSI predictor |
| `models/physical_negotiation.py` | PhysicalNegotiation-v1 |
| `models/physical_negotiation_v2.py` | PhysicalNegotiation-v2 |
| `models/physical_negotiation_v3.py` | PhysicalNegotiation-v3 |
| `models/sequential_set_policy.py` | 顺序集合策略；规模不适合主线 |
| `models/set_negotiation.py` | SetNegotiation-v1/v2/v3；好集合依赖随机命中 |
| `models/two_stage_marginal.py` | Marginal 路线模型 |
| `models/two_stage_physical.py` | 旧 physical estimator + decoder 总模型 |
| `models/two_stage_v2.py` | Structure-v2/v2.1/v2.2 历史模型 |

特殊处理，不立即直接删：

- `models/two_stage.py`：先把冻结 LocalMax 骨架提取到 `models/local_physical_reuse.py`，再归档/删除其中的 TwoStage-v1；
- `models/parallel_reuse.py`：先把冻结 LocalPhysicalReuse-v2 提取，再删除 LearnedReuse-v1/v2 和旧容器文件。

### 6.2 历史训练/评测/诊断脚本

以下脚本的结论已写入 `docs/TWO_STAGE_RESEARCH_OVERVIEW.md` 和教程文档，正式论文入口不再调用：

```text
experiments/audit_anchor_join_scalability.py
experiments/diagnose_analytic_rho.py
experiments/diagnose_anchor_join_opportunities.py
experiments/diagnose_anchor_value_context.py
experiments/diagnose_anchor_value_of_learning.py
experiments/diagnose_candidate_portfolio.py
experiments/diagnose_dynamic_queue_shift.py
experiments/diagnose_exact_set_greedy.py
experiments/diagnose_pairwise_value_context.py
experiments/diagnose_parallel_mpnn_local_trap.py
experiments/diagnose_parallel_mpnn_pair_traps.py
experiments/diagnose_parallel_mpnn_scenarios.py
experiments/diagnose_physical_rho_sufficiency.py
experiments/diagnose_queue_gat_policy.py
experiments/diagnose_random_set_dynamic.py
experiments/diagnose_reuse_safety_neighborhood.py
experiments/diagnose_sequential_policy_context.py
experiments/diagnose_set_exploration_coverage.py
experiments/diagnose_set_negotiation_multistart.py
experiments/diagnose_set_queue_scale.py
experiments/evaluate_oracle_marginal.py
experiments/evaluate_oracle_physical_negotiation.py
experiments/evaluate_parallel_reuse.py
experiments/evaluate_static_baselines_tx10.py
experiments/evaluate_two_stage_marginal.py
experiments/evaluate_two_stage_physical.py
experiments/evaluate_two_stage_v2.py
experiments/parallel_mpnn_scenarios.py
experiments/run_parallel_mpnn_queue_experiment.py
experiments/teacher_label_ambiguity.py
experiments/train_anchor_exact_set_utility.py
experiments/train_anchor_join.py
experiments/train_anchor_one_join.py
experiments/train_anchor_value_model.py
experiments/train_oracle_physical_negotiation.py
experiments/train_parallel_mpnn.py
experiments/train_parallel_mpnn_rollout_calibration.py
experiments/train_parallel_reuse.py
experiments/train_physical_channel_estimator.py
experiments/train_sequential_set_policy.py
experiments/train_set_negotiation.py
experiments/train_two_stage_marginal.py
experiments/train_two_stage_stratified.py
experiments/train_two_stage_v2.py
experiments/two_stage_v1_diagnostics.py
```

以下脚本在正式入口建立后再替换/归档：

```text
experiments/diagnose_local_margin_sufficiency.py
experiments/diagnose_localmax_priority.py
experiments/evaluate_localmax_priority.py
experiments/diagnose_localmax_reuse_headroom.py
experiments/queue_model_diagnostics.py
experiments/run_queue_experiment.py
```

它们分别包含冻结版本的来源证据、checkpoint loader、GAT-q 训练与公共仿真函数，因此不能在新入口完成前先删。

### 6.3 与历史模型一一对应的测试

```text
tests/test_anchor_join.py
tests/test_anchor_one_join.py
tests/test_candidate_portfolio.py
tests/test_parallel_mpnn.py
tests/test_parallel_mpnn_v2.py
tests/test_parallel_mpnn_v3.py
tests/test_parallel_mpnn_v4.py
tests/test_parallel_mpnn_scenarios.py
tests/test_physical_channel_estimator.py
tests/test_physical_negotiation.py
tests/test_physical_negotiation_v2.py
tests/test_physical_negotiation_v3.py
tests/test_sequential_set_policy.py
tests/test_set_negotiation.py
tests/test_set_negotiation_v2.py
tests/test_set_negotiation_v3.py
tests/test_set_negotiation_multistart.py
tests/test_two_stage_marginal.py
tests/test_two_stage_physical.py
tests/test_two_stage_v2.py
```

`tests/test_parallel_reuse.py` 不直接删除：先改写为冻结 v2 的 `tests/test_local_physical_reuse.py`，验证动作等价后再删除旧文件名。

### 6.4 文档处理

本轮建议**不删除任何 `docs/` 下的历史文档**。原因是它们已经集中在 `docs/archive/two_stage/`，不会干扰新读者，并保存了失败路线的完整逻辑。

活动入口只保留并维护：

```text
docs/TWO_STAGE_RESEARCH_OVERVIEW.md
docs/TWO_STAGE_LOCALMAX_REUSE_TUTORIAL.md
docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md
docs/FIXED_CSI_FREEZE_AND_CLEANUP_PLAN.md
docs/NEXT_CHAT_CONTEXT.md
```

`experiments/TWO_STAGE_EXPERIMENT_INDEX.md` 将在清理后改写成简短代码入口，不再维护历史脚本清单。

## 7. 旧结果文件如何处理

当前 `results/diagnostics/` 有 164 个 JSON，约 353 MB；此外旧 `queue_exp*`、`parallel_queue*` 和 `random_set_dynamic*` 目录约几十 MB。它们包括 smoke、失败训练、重复 seed 和已被修正口径替代的结果。

建议不要在正式结果生成前删除。正式仿真完成后再进行第二次确认：

### 7.1 建议保留的历史证据

```text
results/diagnostics/local_margin_sufficiency_numUAV=10_tx10_20260714_154940.json
results/diagnostics/local_margin_sufficiency_numUAV=20_tx10_20260714_155001.json
results/diagnostics/local_physical_scale_benchmark_numUAV=10_tx10_20260714_180851.json
results/diagnostics/local_physical_scale_benchmark_numUAV=20_tx10_20260714_180856.json
results/diagnostics/local_physical_scale_benchmark_numUAV=50_tx10_20260714_182524.json
results/diagnostics/localmax_priority_dynamic_numUAV=10_tx10_20260714_181130.json
results/diagnostics/localmax_priority_dynamic_numUAV=20_tx10_20260714_181749.json
results/diagnostics/localmax_priority_dynamic_numUAV=50_tx10_20260714_182346.json
results/diagnostics/mobile_block_fading_csi_20260714_194642.json
results/diagnostics/README.md
```

这些文件随后移到 `results/archive/two_stage_evidence/`，而不是继续混在正式结果目录中。

### 7.2 建议删除的结果组（正式结果生成后另行确认）

```text
results/queue_exp20260710_*
results/queue_exp20260711_*
results/queue_exp20260712_*
results/parallel_queue_tx10_*
results/random_set_dynamic_tx10_*
results/diagnostics/anchor_*.json
results/diagnostics/candidate_*.json
results/diagnostics/exact_set_*.json
results/diagnostics/parallel_mpnn_*.json
results/diagnostics/parallel_reuse_*.json
results/diagnostics/physical_*.json
results/diagnostics/sequential_*.json
results/diagnostics/set_*.json
results/diagnostics/teacher_*.json
results/diagnostics/two_stage_*.json
results/diagnostics/*smoke*.json
```

由于这一组包含大量文件且 `results/` 已被 `.gitignore` 忽略，删除前会生成一份**展开后的精确文件名 manifest**，再次提交用户确认，不会只按通配符直接执行。

## 8. 用户确认与执行状态（2026-07-15）

本轮请求用户只确认源码清理方向，不立即确认旧结果删除：

- [x] 已提取冻结 v2、新建正式 fixed-CSI benchmark，并完成动作等价测试；
- [x] 第 6.1--6.3 节及用户确认的旧缓存/规模诊断共 94 个文件已先压缩归档、
  校验 SHA-256，再从活动源码树删除；
- [x] 历史 `docs/archive/two_stage/` 未删除；
- [x] DistGCN 只做低成本接入闸门，不阻塞正式主图；
- [x] 旧结果尚未删除，仍需用展开后的精确 manifest 单独确认。

归档位置：`archives/two_stage_exploration_source_20260711_14.tar.gz`。
正式实现和结果说明转入 `docs/FIXED_CSI_QUEUE_BENCHMARK_TUTORIAL.md`。

用户可以直接回复：

> 同意第 8 节全部方案，可以开始清理和固定 CSI 正式仿真。

也可以指出某个具体文件或历史路线希望额外保留。
