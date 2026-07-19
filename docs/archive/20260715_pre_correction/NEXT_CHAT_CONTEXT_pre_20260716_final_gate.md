# 下一 Chat 交接：TNSE 队列感知 UAV 分布式调度

> **2026-07-16 入口更新：** 本文件以下正文是旧固定/移动 CSI 交接快照，只用于追溯。
> 当前任务请按顺序阅读：
>
> 1. `docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`：CSI、GRU、正确 checkpoint 和常数归因；
> 2. `docs/GAT_MWS_TOPOLOGY_SCALING_TUTORIAL.md`：当前 GAT-MWS 结构重设计与一日闸门；
> 3. `docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`：现有两阶段系统公式。
>
> 正式口径冻结为 N=20、Rician K=5 dB、训练/测试均为 10 dBm。旧 N=10、13 dBm
> 和 Rayleigh 再生成数据不再作为当前模型输入。

> 更新时间：2026-07-15。固定 CSI 阶段已经完成源码冻结、正式入口和第一轮
> 5000-slot 主结果。新 Chat 应先读
> `docs/FIXED_CSI_QUEUE_BENCHMARK_TUTORIAL.md`；本文仍负责论文背景与移动 CSI
> 交接。本文后半部保留的 2026-07-14 数值和旧命令只用于历史追溯，若与新教程
> 冲突，以新教程和根目录 README 为准。

## 2026-07-15 固定 CSI 收尾快照

- 活动模型只保留 `models/local_physical_reuse.py` 中的
  LocalPhysicalReuse-v2 及必要消融；
- 94 个历史探索源码已归档到
  `archives/two_stage_exploration_source_20260711_14.tar.gz`，manifest 和 SHA-256
  已验证；
- 正式动态入口为 `experiments/exp_fixed_csi_qos.py`，单时隙入口为
  `experiments/exp_fixed_csi_snapshot_utility.py`；旧
  `run_queue_experiment.py`、`evaluate_localmax_priority.py` 等已归档；
- N=10、5000 slots、seeds 42/43/44 主结果位于
  `results/queue/fixed_csi/main_N10_5000slots/`；
- Proposed 在 `Lambda=8` 的服务率为 2.761 packet/slot，Pure 为 2.617，
  Centralized-MWS-Greedy 为 2.946，即 Proposed 达到集中式 greedy 的 93.7%；
- N=10/20/50 的 5000-slot 配对固定信道 traces 已保存在
  `datasets/queue_traces/`；
- EndToEnd-GAT-q 已在严格 10 dBm/perfect 元数据下重训，但动态性能很差，
  作为失败对照保留；
- 时延图只显示稳定率至少 80% 的点，避免不稳定系统的成功包条件时延产生幸存者
  偏差；
- 下一步先扩充正式 seeds 与 N=20/50 长跑，再进入移动 CSI；DistGCN 仅作为
  不阻塞主线的 hardened-conflict adapter。

## 0. 给下一个 Codex 的工作方式要求

用户是一名通信专业博士生，但对 learning 术语不熟悉。回答和实验记录必须采用“动机—定义—实现—结果—分析—下一步”的连续逻辑；新名词先解释，不能突然堆缩写。模型设计应尽量简洁，每个模块都要说明为什么必要，不能依靠 bias、temperature、STE 或互相制衡的惩罚项修补根本学习问题。

用户希望模型真正通过局部消息理解信道竞争与队列紧急性，而不是单纯模仿集中式教师。MWS-Greedy 只可作为集中式强参考或误差诊断，不能称为数学上界，也不能成为整个 learning 方案的唯一目标。

运行 Python 使用本机 Conda 环境：

```bash
conda run --no-capture-output -n pytorch python -u <script.py> ...
```

代码树可能包含用户原有修改和本轮研究文件。禁止清理整个工作区、重置用户改动或随意删除历史结果。

## 1. 论文背景、基础与期限

- 项目：密集无人机集群中的 GAT 分布式通信链路调度；
- 会议基础：`docs/GNN-based link scheduling for dense UAV swarm.pdf`，解决“不含队列时，以最大化瞬时网络容量为目标的 GAT 分布式调度”；
- 期刊目标：在会议版基础上扩展为约 17 页的 TNSE 论文；
- 专刊：**AI-Driven Low-Altitude Intelligent Networks: From Modeling to Real-World Deployment**；
- 用户当前计划的投稿截止时间：2026-08-01；
- 本轮期刊扩展的核心贡献方向：加入 link-level 动态队列，在密集、分布式、有限轮交互条件下形成 QoS-aware 联合调度；随后再考虑 UAV 移动、block fading 和不完美 CSI。

官方 CFP 特别强调三维时变拓扑、高移动性、间歇连接、机载算力/能量限制，以及不完整、延迟或含噪网络状态；动态干扰管理和轻量分布式执行与本项目直接相关。官方页面：<https://www.comsoc.org/publications/journals/ieee-tnse/cfp/ai-driven-low-altitude-intelligent-networks-modeling-real-world>。

当前统一物理口径是 **10 dBm / perfect CSI / 5 MHz / 1 ms slot / 12000-bit packet**。旧的仅按 `numUAV` 命名的数据和模型可能来自 22 dBm，不得与当前正式结果混用。

## 2. 当前问题的数学目标

冲突图的每个节点是一条有向通信链路。链路 $i$ 的二值发送动作为 $a_i(t)\in\{0,1\}$，队列为 $Q_i(t)$，在联合动作 $\mathbf a(t)$ 下的实际速率为 $R_i(\mathbf a(t))$。

经典 Max-Weight Scheduling（MWS）的逐时隙目标为

$$
\max_{\mathbf a(t)\in\{0,1\}^{L(t)}}
\sum_i Q_i(t)R_i(\mathbf a(t)).
$$

MWS 的意义是：长期队列稳定问题可以转化为每时隙根据当前队列和信道选择较高的队列加权服务，不必给调度器增加长时间记忆。但我们的固定轮局部近似并没有自动继承精确 MWS 的 throughput-optimal 定理，论文不能过度声称。

真正困难不是“每条链路独立判断自己好不好”，而是 $R_i$ 随整个发送集合变化。当前共识是：

1. 要真实联合集合条件，不要彼此独立的一次性概率；
2. 要多个局部区域并行形成空间复用，不要唯一全局 anchor；
3. 要固定轮局部协商，不要随激活数增长的顺序 greedy 或 Best-K 搜索；
4. 队列只负责快速紧急性，信道模块负责较慢的竞争结构；
5. learning 必须通过物理可解释接口参与，而不是任意把 GAT logit 当速率。

## 3. 已走过的路线与纠偏结论

最初的端到端 GAT-q 诊断、MWS 理论边界和 TwoStage-GAT-v1 起点来自上一任务 `019ebadf-f775-7371-92ab-c32156b320ad`；本任务在此基础上继续完成了 Structure、集合协商、LocalMax + ParallelReuse、50-UAV 闸门与 v2 物理接口。新 Chat 不需要重新读取聊天记录，以下表格和本文件已经吸收两轮结论。

| 路线 | 得到的认识 | 为什么不再作为主线 |
|---|---|---|
| 端到端 GAT-q | 仅把 Q 加入输入、直接优化 $\sum QR$ 不足以学会联合集合 | 连续动作易全开/全关，独立概率缺少集合条件 |
| Structure v1--v2.2 | 两阶段分离比端到端更可学 | 仍接近逐链路打分，高负载联合竞争较弱 |
| Marginal / PhysicalNegotiation | 有向干扰、真实 SINR、条件边际收益都很重要 | 从任意坏集合单链路修补容易陷入局部最优 |
| Anchor--Join | 决策必须以当前集合为条件 | 唯一 anchor、不可撤销和顺序轮数不适合大规模分布式场景 |
| ParallelMPNN | 固定轮 GNN 具有静态集合表达能力 | 动态高负载显著下降，连续意图不等于可靠 0/1 集合 |
| Best-K | 多个完整候选确实能提高上限 | K 份生成、评价和一致选择难以扩展 |
| SetNegotiation-v3 | 真实 0/1 集合可作为可加入/退出的协商状态 | 好集合存在但单次命中依赖运气，联合信用分配仍失败 |
| LocalPhysicalReuse-v1 | 可靠局部骨架 + 一轮条件复用在 10/20/50 UAV 均有效 | `q_rel × softplus(GAT logit)` 没有速率量纲 |
| **LocalPhysicalReuse-v2** | GAT 活动先验转成期望有效速率，再与 Q 结合 | **当前推荐学习版**；移动 CSI/消息开销已做第一轮诊断，尚缺最终规模与真实 MAC |

详细历史见 `docs/TWO_STAGE_RESEARCH_OVERVIEW.md`，不要从归档代码中继续挑一个旧 v2/v3 当作当前模型。

## 4. 当前最新版：LocalPhysicalReuse-v2

### 4.1 总体结构

模型分为“慢信道理解”和“快队列调度”两种时间尺度：

1. 每个信道相干块运行一次冻结的会议版 channel-only GAT；
2. 每个时隙读取当前队列，用缓存的信道结果形成多个局部胜者；
3. 其余链路以真实局部胜者集合为条件，做一次并行空间复用补充。

它没有唯一 anchor、没有 Best-K、没有循环加入十轮，也没有训练一个网络模仿 MWS-Greedy 的最终动作。

### 4.2 阶段一：channel-only GAT

会议版 GAT 输入本链路 SNR、冲突图邻接关系和有向干扰 edge feature，经固定 $T$ 层消息传递输出 logit $c_i^{\mathrm{ch}}$。定义活动先验

$$
p_i^{\mathrm{ch}}=\operatorname{sigmoid}(c_i^{\mathrm{ch}}).
$$

对链路 $i$，利用入邻居的先验和有向物理增益计算

$$
\widetilde I_i=\sum_{j:(j,i)\in\mathcal E}
p_j^{\mathrm{ch}}g_{ji},
$$

$$
\widetilde C_i=
B\log_2\!\left(1+
\frac{g_{ii}}{\Gamma(\sigma^2+\widetilde I_i)}\right).
$$

这里 GAT 不直接决定本时隙动作；它提供“在当前信道竞争图案下的期望有效速率”。这是 v2 相比 v1 唯一最重要的理论修正。

### 4.3 阶段二 A：队列感知局部骨架

每个时隙定义局部竞争权重

$$
W_i(t)=Q_i(t)\widetilde C_i.
$$

非空链路只与冲突邻居比较 $W$，局部最大者进入真实二值骨架 $\mathcal A_0(t)$。相距较远的邻域会同时产生多个胜者，因此 $\mathcal A_0$ 是分布式局部胜者集合，不是全网头节点。

### 4.4 阶段二 B：一轮条件空间复用

对未进入 $\mathcal A_0$ 的非空链路 $i$，根据当前活动冲突邻居计算局部加入收益：

$$
\Delta_i^{\mathrm{loc}}(\mathcal A_0)
=\underbrace{Q_iC_i(\mathcal A_0\cup\{i\})}_{\text{自己的收益}}
-\underbrace{\sum_{k\in\mathcal A_0:(i,k)\in\mathcal E}
Q_k\left[C_k(\mathcal A_0)-C_k(\mathcal A_0\cup\{i\})\right]}_{\text{给活动邻居造成的损失}}.
$$

正收益候选再与相互冲突的正收益候选比较一次；局部最大者并行加入，得到

$$
\mathcal A(t)=\mathcal A_0(t)\cup\mathcal J(t).
$$

该动作是确定的 0/1 集合，且整个过程固定轮数。

完整公式、假设和不能声称的内容见 `docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`。

## 5. 模型如何训练、多久训练一次、在线输入什么

### 5.1 训练不是在线周期任务

当前仍是 **集中训练、分布式执行**：

- channel-only GAT 在离线阶段用许多信道/拓扑快照集中训练；训练 loss 可以访问完整网络速率，训练完成后保存一套共享参数；
- 当前 v2 直接冻结并复用会议版 GAT checkpoint，没有再训练队列决策网络；
- 部署后 UAV 不需要每个时隙、每个信道块重新训练。只有当信道分布发生长期变化且泛化失败时，才考虑离线重训或未来的域适配，这不是当前模型的一部分。

### 5.2 在线推理周期

| 时间尺度 | 输入 | 计算/交换 | 输出 |
|---|---|---|---|
| 每个信道相干块一次 | 本链路 SNR、冲突邻接、有向边特征 | $T$ 层邻居 GAT 消息；缓存邻居活动先验和 $\widetilde C_i$ | 慢变信道先验 |
| 每个时隙一次 | 当前 $Q_i(t)$、缓存 $\widetilde C_i$ | 交换 $W_i$，得到 $\mathcal A_0$ | 多个局部胜者 |
| 同一时隙固定后续轮 | 邻居骨架动作、队列、边增益 | 计算并交换 $\Delta_i^{\mathrm{loc}}$，局部竞争 | 最终 0/1 动作 |

“GAT 有 $T$ 层”在数学上对应 $T$ 轮冲突邻居隐藏状态传播；它不是 UAV 把整个全网图发给中心。但当前 PyTorch 实现仍在一台机器上用完整张量模拟这些并行局部运算，尚未实现真实 MAC 控制包、异步时钟和丢包。因此只能称为**逻辑分布式/可分布式实现**，不能称为已完成实网分布式部署。

同理，“固定轮”不等于“控制开销很小”。密集冲突图中每轮涉及的邻接关系很多，后续论文必须报告消息维度、轮数和可能的广播/聚合实现。

## 6. 动态仿真指标到底是什么

表中的 0.3、0.6、0.99 都是

$$
\eta_{\mathrm{srv}}
=\frac{\text{1000 个时隙内实际发送的数据包总数}}
{\text{1000 个时隙内到达的数据包总数}},
$$

不是“本算法除以 MWS-Greedy”的相对性能。所有方法在同一 Monte Carlo seed 和同一 $\Lambda$ 下共享完全相同的信道图和到达轨迹，因此可以横向公平比较；MWS-Greedy 是同一物理参数下单独列出的一列。

当前 $\Lambda$ 是**全网总到达率**，不是每条链路到达率。20% 热点链路承担 70% 到达业务。$\Lambda=8$ packets/slot、12000 bit/packet、1 ms/slot 对应约 96 Mbit/s 的 offered load。20-UAV 结果中：

- LocalPhysicalReuse-v2：服务比 0.3865，即约 37.23 Mbit/s；
- MWS-Greedy：服务比 0.4254，即约 40.97 Mbit/s；
- 两者均判定为不稳定，队列持续增长。

所以降到 0.3--0.4 是正常的过载现象，不是模型突然退化到 MWS 的 30%。若只看同一负载下相对 MWS 服务能力，则 v2 约为 $0.3865/0.4254=90.9\%$。

正式论文不应只画服务比。稳定区应同时画平均 backlog/时延；过载区可画实际吞吐量和相对 MWS 性能，并明确它是压力测试，不把服务比小于 1 误写成算法准确率。

## 7. 当前可信实验结果

设置：10 dBm、perfect CSI、固定信道、1000 时隙、共享 seed 42/43/44。

| 规模 / $\Lambda$ | v1 | **v2 GAT Expected-Rate** | 纯物理骨架 | MWS-Greedy |
|---|---:|---:|---:|---:|
| 10 UAV / 2 | 0.991 | 0.987 | 0.991 | 0.992 |
| 10 UAV / 4 | 0.566 | **0.631** | 0.615 | 0.655 |
| 20 UAV / 2 | 0.976 | 0.972 | **0.981** | 0.988 |
| 20 UAV / 4 | 0.632 | 0.638 | **0.654** | 0.736 |
| 20 UAV / 8 | 0.340 | **0.387** | 0.367 | 0.425 |
| 50 UAV / 1 | 0.999 | 0.999 | 0.999 | 未运行 |
| 50 UAV / 2 | 0.960 | 0.965 | **0.970** | 未运行 |
| 50 UAV / 4 | **0.634** | 0.627 | 0.634 | 未运行 |
| 50 UAV / 8 | 0.353 | **0.380** | 0.361 | 未运行 |

结论必须保持克制：v2 不是每个点都最强，但它在 10/20/50 UAV 没有性能断崖，在高负载明显优于旧 v1，且 GAT 有清楚物理接口。纯物理 $Q\times$solo-rate 在部分中负载点更高，是必须保留的强消融。

50-UAV 每个图约 835--1075 条通信链路；代表图有 363456 条有向冲突边。CPU 暖时隙约 9.2 ms，尚未达到 1-ms 单 CPU 实时性。

### 7.1 第一轮移动 block-fading / CSI 风险诊断

已新增永久有向 UAV 对 link ID、跨块队列、暂时不可用动作屏蔽，以及真实服务信道/调度观测 CSI 分离。10 UAV、$\Lambda=4$、300 时隙、seed 42/43/44 下扫描 5/20 m/s 与 5/20-slot 块：

- latest perfect CSI 服务比为 0.674--0.752；
- stale one-block CSI 相对 latest 下降 16.9%--28.1%，最坏 backlog 增长 68.0%；
- large-scale 与 stale 大体同层；
- 2-bit 距离量化明显不足；4-bit、8-bit 与未量化 large-scale 在当前三个 seed 下接近，但尚不能证明最终量化位数。

因此 one-block stale CSI 失效判定为严重，足以支持下一轮预测可行性实验；它没有证明学习式预测模块必要。该结果只是 10-UAV、0.3-s 风险诊断，轨迹尚未触发链路可用性切换。完整教学记录见 `docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`。

## 8. 50-UAV 冲突边与数据集的正确理解

50 个 UAV 最多产生 $50\times49=2450$ 条有向通信链路。冲突图把通信链路作为节点，因此冲突边最坏可到 $L(L-1)$，百万级边在密集网络中是自然结果，不应因为数字大就武断稀疏化。

分解“共享 UAV 端点约束 / 强物理干扰 / 弱干扰”的目的只应是：

1. 解释 CPU 时间和真实控制消息成本；
2. 判断哪些关系可由 UAV 端点一次广播或聚合，而不必逐条 link-link 发消息；
3. 为复杂度/部署章节提供诚实证据。

它不是当前模型正确性的前置条件，也不是把密集网络改成稀疏网络。若做边压缩，必须同时报告效用损失，不能只追求省边。

当前 v2 不训练队列模块，因此没有必要生成“每个时隙复制一张大图”的 50-UAV 队列训练集。为复用当前固定信道评测，本轮新增了紧凑 trace cache：每个 MC seed 只保存一张信道图，再分别保存各 $\Lambda$ 的轻量到达张量。生成脚本为：

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/cache_queue_benchmark.py \
  --num_uav 50 --num_slots 1000 \
  --arrival_rates 1,2,4,8 --mc_seeds 42,43,44
```

默认输出到 `datasets/queue_traces/`。这只是可重复评测轨迹，不是未来信道预测模块的训练集。未来 block-fading 数据应按“信道块”存拓扑/CSI，队列和到达轨迹单独存，避免每时隙重复整张图。

复用缓存运行当前 v2 的示例：

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/evaluate_localmax_priority.py \
  --num_uav 50 --num_slots 1000 \
  --arrival_rates 1,2,4,8 --mc_seeds 42,43,44 \
  --modes raw_queue_gat_rate \
  --trace_cache datasets/queue_traces/fixed_numUAV=50_tx=10dBm_csi=perfect_slots=1000_mc=3.pt
```

旧 `datasets/dataset_numUAV=50.pkl` 约 16 GB，缺少 10-dBm/perfect-CSI 标签，没有被当前正式实验读取或覆盖。

## 9. 当前有用文件

新 Chat 只需按以下顺序阅读：

1. `docs/NEXT_CHAT_CONTEXT.md`：本交接；
2. `docs/TWO_STAGE_RESEARCH_OVERVIEW.md`：所有历史路线的地图；
3. `docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`：当前模型完整公式；
4. `docs/TWO_STAGE_LOCALMAX_REUSE_TUTORIAL.md`：每轮实验的目的、结果与因果演进；
5. `docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`：移动、link identity 与四类 CSI 的从零教学记录；
6. `experiments/TWO_STAGE_EXPERIMENT_INDEX.md`：代码与历史方案索引；
7. `results/diagnostics/README.md`：代表性 JSON 分类。

当前实现核心：

- `models/parallel_reuse.py`：LocalPhysicalReuse-v1/v2 调度器；
- `models/two_stage.py`：GAT Expected-Rate LocalMax 骨架；
- `utils/parallel_reuse.py`：按冲突边计算局部条件收益；
- `simulator/queue_simulator.py`：队列演化与服务比指标；
- `simulator/mobile_block_fading.py`：永久 link ID、移动 block fading、真实/观测 CSI 分离；
- `experiments/evaluate_localmax_priority.py`：10/20/50 UAV 动态比较；
- `experiments/evaluate_mobile_block_fading_csi.py`：latest/stale/large-scale/quantized 共享轨迹诊断；
- `experiments/benchmark_local_physical_scale.py`：运行时间；
- `experiments/cache_queue_benchmark.py`：固定信道基准缓存；
- `experiments/run_queue_experiment.py`：共享信道/到达轨迹的仿真框架；
- `utils/mws.py`：集中式 MWS-Greedy 参考。

`docs/archive/two_stage/` 与索引中列出的旧模型只用于追溯，不能在未解释原因时恢复为主线。

## 10. 下一阶段研究计划

固定信道结构和第一轮移动 CSI 风险诊断已经完成。下一步不再修改局部骨架，而是验证最小块级信道预测是否能修复已经量化的 stale-CSI 损失。

### 阶段 A：冻结当前固定信道基线

- 用紧凑 trace cache 复现 10/20/50 UAV 关键点；
- 确认服务比、backlog、实际 Mbps 和相对 MWS 四种口径；
- 保留 v1、纯物理骨架、Random、MWS-Greedy 作为消融/参考；
- 不再为提高某一个点继续修改 v2。

### 阶段 B：block-fading 状态定义（已完成）

已用有向 UAV 对 $(s_i,d_i)$ 定义持久队列 ID；链路暂时不可用时保留队列但动作置零。调度器共享同一移动、真实信道和到达轨迹，观测 CSI 与真实服务信道分开保存。

当前最小模型每 $T_c$ 个时隙更新 UAV 位置、大尺度信道和相关小尺度衰落；块内缓存信道先验，队列每时隙更新。第一轮没有混入预测器或额外估计网络。

### 阶段 C：回答“缓存多久会失效”（第一轮已完成）

第一轮已扫描相干块长度和移动速度，并比较：

1. 每块使用最新 perfect CSI；
2. 使用过期一块的 CSI；
3. large-scale CSI；
4. distance-quantized large-scale CSI。

结果已同时报告服务比、backlog/时延、CSI 误差和信令频率；stale 损失明确越过预设闸门，因此已决定进入最小预测验证。

### 阶段 D：预测可行性诊断（当前下一步）

过期 CSI 的损失已经明显越过 5% 服务/20% backlog 风险闸门，但这只证明值得继续诊断。先解耦真实信道块长与 CSI 更新周期，加入纯物理骨架排除 frozen-GAT 输入分布变化，再测试无训练的 Gauss--Markov 一步解析预测。只有解析预测稳定优于 stale/large-scale，才考虑下一块局部物理量/channel-prior 的最小学习头。队列仍按当前 $Q(t)$ 决策，绝不预测最终队列动作。

### 阶段 E：复杂度与真实分布式审计（第一轮已完成）

当前已核算每块 5 轮信道消息、每时隙 3 轮队列消息，以及逐边单播上界/发射 UAV 聚合 payload 下界。下一轮需要加入 CSI 获取、包头、分片、MAC 竞争和 CPU/GPU 延迟；冲突边来源分解只用于讨论聚合实现，不为了随意删边。

## 11. 明确不做或暂缓的内容

- 截稿前不重新打开连续功率控制主线。固定功率下的链路开/关本身已是一种粗粒度功率分配；此前连续功率控制收益不明显。安全传输、beamforming 或能耗约束可作为未来工作再引入；
- 不用时间 RNN 记忆长期队列历史。MWS 当前只需要当前 $Q(t)$；若加入时间模型，应服务于信道预测，而不是安排未来队列；
- 不把 MWS-Greedy 教师动作作为唯一监督标签；
- 不为了让公式看起来智能而增加无必要 MLP、阈值、温度或惩罚项；
- 不宣称当前算法已证明 throughput-optimal、满足 1-ms 实时性或完成真实 MAC 部署。

## 12. 新 Chat 可直接使用的首条指令

> 请先完整阅读 `docs/NEXT_CHAT_CONTEXT.md`、`docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md` 和 `docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`。不要从归档版本继续迭代。我们已完成 link-identity 一致的移动 block-fading 与 latest/stale/large-scale/distance-quantized large-scale CSI 第一轮诊断：stale 相对 latest 的服务比损失 16.9%--28.1%，但这只证明预测实验值得做，并未证明学习式预测器必要。下一步先解耦真实信道块长和 CSI 更新周期，加入纯物理骨架消融，再测试无训练的 Gauss--Markov 一步解析预测；只有它稳定优于 last-CSI/large-scale，才考虑最小学习预测头。不要预测最终队列动作。随后用更长轨迹触发链路可用性切换并扩展到 20 UAV，新增消息和计算开销必须单独核算。
