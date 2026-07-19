# GAT 如何补足 Pure 的拓扑盲区：MWS-aligned 重设计教程

> **架构状态更新（2026-07-18）**：本文记录的是MarginalMP定型前对GAT/Pure骨架接口
> 的负结果与止损依据，不是当前系统架构。当前后端为
> `Quality → LocalMax → K轮MarginalMP`（$K=4$，无Join）；GAT只保留为可替换Quality
> 分支或未来CSI归因接口。

> **本文是已完成的GAT结构重设计历史。** 移动信道、stale/large-scale/quantized
> CSI、旧 GRU 和 CSI-augmented GAT 的诊断统一保留在
> `MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`。本文只讨论一个新问题：在
> LocalPhysicalReuse 的 `sum(QR)` 目标下，GAT 应向局部胜者骨架输出什么。

## 0. 一页结论

会议 GAT 没有坏。它在原始 channel-only 总容量任务中能从数百条候选链路中选择约
11 条兼容链路，并把 N=30 Rician 容量从全发的 3.40 提高到 34.64。问题发生在期刊
两阶段接口：会议输出 $\pi_i$ 是旧 sum-rate 目标下的最终激活倾向，却被重新解释成
邻居独立活动概率，代入 expected interference 后形成队列骨架。

同均值常数对照已经证明，这个接口中观察到的高负载或 stale-CSI 增益主要来自数值
校准，不来自 GAT 学到的链路差异。因此本轮没有继续修补会议 $\pi$，而是从 Pure 的
缺口出发，依次检查两种新接口。

第一种接口是给每条链路输出 MWS 局部 greedy 缺失的**拓扑机会成本修正**：

$$
W_i(t)=z_i(t)Q_i(t)C_i^{\rm solo}(t).
$$

$z_i$ 不是概率，也不是最终动作。若所有 $z_i$ 相同，排序严格退化为 Pure。GAT 只有
学会“哪些高自利链路会排挤更多高价值邻居”，才能超过 Pure。

无训练 headroom 闸门通过：N=20、Rician、10 dBm 的 36 个未见快照中，
集中式一次骨架替换 oracle 相对 Pure 平均提升最终 $\sum_iQ_iR_i$ 12.52%，在 83.3%
快照严格改善；异构和饱和队列分别提升 16.23% 和 8.29%。这证明骨架排序存在足够大
的优化空间，但冻结 encoder 的 scalar-$z$ 只比 Pure 排序准确率提高 0.48 个百分点，
完整调度闸门失败。因此“每条链路预先输出一个标量”已经停止。

第二种接口是在 Pure 骨架已经形成后，让 inactive challenger 针对当前 winner 集合
提出一次替换。这里的 **set-conditioned** 指“同一 challenger 的好坏必须结合当前
究竟有哪些 winner”判断。诊断发现：固定 degree、邻居效用占比和一跳干扰惩罚都不
稳定；一跳局部最终收益虽与真实收益高度相关，却仍会选出使全网效用下降的替换；两跳
之所以完全复现 oracle，是因为 98.8% 的候选两跳范围已经覆盖全网，不能算低开销局部
方案。

最后只训练了一次严格受限的 set-conditioned challenger GAT：会议 encoder 冻结，
每时隙新增两轮 GAT，仅使用可本地形成的骨架信息和一跳物理量，并与同输入、零消息的
MLP 对照。它在验证和未见测试上都选择“不替换”，相对 Pure 增益为 0，闸门明确失败。

用户随后提出了一个更直接的检查：让 $Q$ 真正进入图消息，并让 GAT 同时输出完整
骨架的 keep/add/drop。零训练审计首先确认这个动机成立——同一信道图只改变队列后，
12/12 图的最优修改发生变化，三种队列状态两两比较时 94.4% 的 oracle challenger
不同。但小规模整骨架监督闸门中，Q-aware GAT 仍完整保留 Pure，未见图增益为 0；
因此“Q 必须进入”是必要条件，却不是解决联合骨架信用分配的充分条件。

随后又澄清并补测了会议 $\pi$ 的两种物理语义。连续 $\pi$ 进入 expected-SINR 正是
LocalPhysicalReuse-v2 已有公式；将其 round 成会议联合动作再做复用也已在正确 N=20
下失败。新补的 nonempty-queue mask 能移除空队列假干扰，但没有稳定收益。最后把
Pure/GAT 当成两个候选生成器做集中式选择，heterogeneous 有 +3.05% 上界，却只在
3/12 图选择 GAT，且同均值 Constant 候选库能复制或超过该收益。因此该候选组合仍未
证明 GAT 图特异性价值。

最后按“块级生成多个骨架、队列先选一个、再只协商一次”测试了 $K\le8$ 骨架码本。
更强的离线校准教师在 heterogeneous 未见队列上，即使违规完成全部8个候选再挑最好
也只有 Pure 的1.0147；在 saturated 虽有1.0596的候选覆盖上界，合法的骨架预选只
得到0.9849，并且仅2/12次选中完成后的最佳骨架。瓶颈同时包括异质队列覆盖不足和
协商前后候选排序反转，因此没有进入多头 GAT 训练。

**该历史分支的最终结论：停止模型迭代。** 当时证据支持“Pure 骨架确有 8%--16% 的集中式替换
空间”，但不支持“该空间能由目前低开销、逻辑分布式的 GAT 接口稳定取得”。不再调
层数、学习率、阈值或辅助 loss，也不进入动态队列和 CSI 鲁棒性验证。保底主干采用
Pure + 解析条件物理复用；会议 GAT 保留其原始 channel-only 贡献和前作衔接，不宣称
它已提高队列稳定域。

## 1. 为什么会议 $\pi$ 与队列骨架目标不一致

### 1.1 会议任务

会议模型不看队列，目标是最大化单时隙网络总 Shannon 容量：

$$
\max_{\mathbf a}\sum_iR_i(\mathbf a,H).
$$

最终 $\pi_i$ 因而偏好直达信道好、干扰较低、容易与其他链路共存的链路。这种偏好在
channel-only 任务中是正确的。

### 1.2 期刊队列任务

队列调度希望近似求解：

$$
\max_{\mathbf a}\sum_iQ_i(t)R_i(\mathbf a,H_t).
$$

若不同链路队列差异很大，$Q_i$ 必须保留积压业务的紧迫性；若所有队列都很大且近似
相同，目标才近似回到总容量。这解释了旧会议 GAT 为什么只在极高负载偶尔有利。

### 1.3 Pure 做对了什么，又缺了什么

Pure 第一阶段使用

$$U_i^{\rm pure}=Q_iC_i^{\rm solo}.$$

它明确保留队列和自身理想速率，不会把旧 sum-rate 偏好直接压过队列。但它只知道
“我有多重要”，不知道“我获胜会挡住多少邻居、这些邻居有多重要”。因此 Pure 是
queue-faithful 的分布式 greedy，不是精确的全局 MWS。

GAT 应补足的是后一个缺口，而不是重新预测一个最终发送概率。

## 2. 新接口：拓扑缩放因子 $z_i$

### 2.1 定义

会议 GAT 的多层 node/edge attention 继续编码信道图。新的读出头输出正值 $z_i$：

$$
z_i=\operatorname{TopologyReadout}
(h_i^{\rm ch},Q_i,\{Q_j,h_j^{\rm ch},e_{ji}\}_{j\in\mathcal N_i}).
$$

局部骨架权重为

$$W_i=z_iQ_iC_i^{\rm solo}.$$

随后仍执行原有 LocalMax 和一轮条件物理复用。

### 2.2 $z_i$ 代表什么

- $z_i$ 小：链路自身看起来很强，但会占用一个代价很高的局部位置；
- $z_i$ 大：链路不仅自身有价值，也容易与周围候选共同复用；
- $z_i$ 相同：排序完全等于 Pure；
- $z_i$ 没有“发送概率”语义，最终动作仍由确定性局部竞争产生。

### 2.3 为什么有研究依据

Zhao 等人的 GCN-LGS 分布式 MWIS 方法同样使用

$$w_i=z_iu_i,$$

让 GNN 学习 topology-aware scaling，再调用分布式 local greedy。该路线的核心目的
正是补足普通 greedy 忽略的图拓扑机会成本，并保持集中训练、分布式执行。

这给出结构依据，但不能替代我们自己的 SINR、队列和物理复用验证。

## 3. 为什么旧 ChannelContext-v1/v2 不能直接复用结论

旧 v1/v2 也使用过 `Q*C_solo*rho`，所以本轮不能通过改名重复失败路线。

旧实验的问题是：

1. 使用后来判废的 N=10 基线；
2. 训练目标是脱离最终整体骨架的局部二选一反事实；
3. v2 虽提高约 1 个百分点的局部判断准确率，完整调度效用却低于 Pure；
4. 这说明局部标签更准不保证多个局部选择能组成更好的全局骨架。

本轮的实质区别是：每个训练偏好都必须通过**完整两阶段动作的最终全局
$\sum_iQ_iR_i$**结算，而不是只看一对链路自身。

## 4. 实验 R0：训练前先测骨架排序有没有足够上限

### 4.1 动机

如果改变 Pure 骨架中的局部胜者几乎不能提高最终效用，那么继续训练 GAT 只是在错误
接口上调参。R0 在训练前测量一个保守的、不可部署的下界 oracle。

### 4.2 方法

对每个固定 N=20、Rician $K=5$ dB、10 dBm 信道和队列快照：

1. 用 Pure 形成局部胜者骨架；
2. 枚举一个尚未获胜的非空链路 $c$；
3. 暂时提升 $c$，移除所有与它冲突的 Pure 胜者；
4. 运行完全相同的第二阶段条件物理复用；
5. 用真实 SINR 结算最终 $\sum_iQ_iR_i$；
6. 只保留相对 Pure 最好的单次替换。

这个 oracle 使用集中枚举和真实结果，只测“骨架标量排序是否有改进空间”，不是候选
算法，也不能写成可部署性能。

队列覆盖三种状态：

- sparse：大部分队列为空，检验低竞争边界；
- heterogeneous：普通队列与热点积压并存；
- saturated：所有链路都有较大积压。

预先闸门要求 heterogeneous 和 saturated 各自满足：平均提升至少 2%，且至少 70%
快照严格改善。sparse 不要求正收益，因为没有足够竞争时 Pure 本来就可能接近正确。

### 4.3 结果

| 队列状态 | 平均相对提升 | 严格改善快照 |
|---|---:|---:|
| sparse | 13.03% | 8/12 |
| heterogeneous | **16.23%** | **12/12** |
| saturated | **8.29%** | **10/12** |
| 全部 | **12.52%** | **30/36** |

95% CI 半宽分别约为 8.39、4.57、4.49 和 3.58 个百分点。异构与饱和队列都超过
预设闸门，因此决定进入一次最小 GAT 训练。

### 4.4 说明了什么，不说明什么

R0 证明：Pure 的第一阶段确实会选出“自身 $Q C^{\rm solo}$ 很高，但最终阻塞代价
更大”的局部胜者；只改变骨架排序，不改第二阶段公式，就存在明显效用上升空间。

R0 没有证明：

- GAT 能从局部消息预测正确替换；
- 动态 throughput/backlog 一定改善；
- oracle 的集中枚举可以部署。

原始结果：
`results/formal/mws_topology_scaling_headroom_N20_rician_20260716.json`。

## 5. 实验 R1：冻结会议 encoder 的最小训练闸门

### 5.1 模型

1. 冻结正式 N=20、Rician、10 dBm 会议 GAT encoder；
2. 保留其每信道块产生的 $h_i^{\rm ch}$；
3. 每时隙增加一轮队列条件图读出；
4. 输出 $z_i\in(0,1)$；
5. 使用 $W_i=Q_iC_i^{\rm solo}z_i$ 形成骨架；
6. 第二阶段物理复用完全不变。

输出头初始为所有链路相同的 $z_i=0.5$。共同常数不改变排序，因此训练起点严格等于
Pure，不会先人为偏向某组链路。

### 5.2 训练标签

从 Pure 骨架出发，枚举 challenger 替换冲突 winner，并执行完整第二阶段：

- 若最终全局 $\sum Q_iR_i$ 提高，训练 challenger 的最终权重高于被替换 winner；
- 若没有提高，训练 challenger 不应压过原最高权重 winner；
- 训练样本平衡正负替换，不按链路 ID 学习。

### 5.3 预设停止条件

未见 fixed/latest 快照中：

- heterogeneous 与 saturated 均须相对 Pure、Constant 平均至少提高 1%；
- 对两个参考的严格胜率均须至少 60%；
- sparse 相对 Pure 的平均效用不得低于 99%；
- 任一条件失败，则停止当前 frozen-encoder scalar-$z$ 路线，不增加层数或 loss。

### 5.4 结果

训练集 12 张图、验证集 4 张图、未见评估 12 张图；每张图均包含 sparse、
heterogeneous、saturated 三种队列。训练 20 epoch，会议 encoder 始终冻结。

排序标签上的结果为：

| 指标 | Pure 原排序 | 学习后 |
|---|---:|---:|
| 未见验证 pair accuracy | 61.34% | 61.81% |

只提高 0.48 个百分点，说明新头几乎没有学会稳定的 winner/challenger 修正。最终完整
两阶段效用更重要：

| 队列状态 | Candidate / Pure 平均比 | 严格胜过 Pure | Candidate / Constant 平均比 | 严格胜过 Constant |
|---|---:|---:|---:|---:|
| sparse | 98.23% | 1/12 | 122.67% | 7/12 |
| heterogeneous | 99.48% | 1/12 | 104.03% | 7/12 |
| saturated | 102.31% | 3/12 | 102.02% | 6/12 |

heterogeneous 与 saturated 均未达到“相对 Pure 和 Constant 平均至少 1%、胜率至少
60%”；sparse 也低于 99% safeguard。因此闸门失败。

### 5.5 为什么 R0 有 8%--16% headroom，R1 却学不到

R0 的 oracle 能在看到整个 Pure 骨架后，逐一尝试 challenger，并用最终真实
$\sum Q_iR_i$ 选择是否替换。R1 必须在替换发生前为每条链路压缩出一个标量 $z_i$。
两者的信息接口不同：

1. 一条链路的机会成本取决于此时究竟有哪些邻居已经成为胜者；
2. 同一 challenger 在不同 winner 集合下可能应该升权，也可能应该降权；
3. 单个 $z_i$ 无法完整表达这种 set-conditioned 决策；
4. 冻结的会议 embedding 又是为 channel-only sum-rate 训练的，不保证保留队列加权
   替换所需的全部信息。

所以 R0 证明的是“Pure 骨架有改进空间”，不是“每链路 scalar scaling 是充分接口”。
R1 与旧 fixed-rho、ChannelContext-v1/v2 的失败共同指向同一根因：集合条件信息不能
稳定压成一个预先计算的链路标量。

### 5.6 决策

按照预设规则，当前 frozen-encoder scalar-$z$ 路线正式停止：

- 不解冻 encoder；
- 不增加 GAT 层数；
- 不调 temperature、STE、bias 或辅助惩罚；
- 不进入动态队列和多 CSI 训练；
- 不把 saturated 的少量均值提升包装成成功，因为胜率只有 3/12。

原始结果位于
`results/formal/mws_topology_scaling_gate_N20_rician_20260716/report.json`。

## 6. 实验 R2：固定拓扑公式能否直接补足 Pure

### 6.1 动机

R1 失败可能有两种原因：一是神经网络没有训练好；二是“在骨架形成前给每条链路一个
固定标量”本身就不够。为避免把结构问题误判为训练问题，R2 完全不训练模型，直接测
四种可解释公式。

这里的**静态拓扑分数**是指：分数由当前链路及其一跳邻居的固定公式一次算出，不随
之后真正选中了哪些 winner 改变。四种规则分别是：

1. degree penalty：用冲突邻居数惩罚拥挤链路；
2. neighbor share：比较本链路 $QC^{\rm solo}$ 在邻域总效用中的占比；
3. physical net benefit：自身理想效用减去它对一跳邻居造成的估计损失；
4. benefit/cost ratio：自身理想效用除以估计干扰代价。

所有规则仍使用相同 LocalMax 和相同条件物理复用，区别只有第一阶段分数。

### 6.2 结果

| 固定规则 | sparse 平均变化 | heterogeneous 平均变化 | saturated 平均变化 |
|---|---:|---:|---:|
| degree penalty | +0.52% | +1.72% | +0.48% |
| neighbor share | -2.61% | +2.67% | +0.35% |
| physical net benefit | -3.49% | **-12.76%** | **-11.80%** |
| benefit/cost ratio | +3.82% | -1.28% | -0.44% |

没有规则通过“heterogeneous 和 saturated 平均至少 +2%、胜率至少 70%，且 sparse
不低于 99%”的闸门。均值偶尔为正也不稳定：例如 neighbor share 在 heterogeneous
只赢 4/12 个快照；degree penalty 分别只赢 2/12 和 3/12。

R2 还测了一个诊断上界：Pure 的完整动作形成后，允许 inactive challenger 撤掉其
所有 active 冲突邻居并替换一次。它在三种队列下平均分别提高 5.85%、5.05% 和
6.20%，但 heterogeneous 和 saturated 都只在 8/12 快照改善，仍低于预设 70%
胜率。这个结果不能包装成通过。

### 6.3 分析

固定惩罚失败说明“冲突多”不等于“此时应该让步”。一个高度链路可能正好拥有最大
队列和最好直达信道，统一降权会损失多用户分集；另一个高度链路也可能挡住几个更有
价值的邻居，应该让步。决定因素不是静态 degree，而是当前 winner 集合。

成形后替换弱于 R0 的原因是时点不同。R0 在骨架阶段替换，然后允许第二阶段重新补充
复用；成形后粗暴撤掉 active 邻居却不重新组织整个复用集合，会丢失已有的有益 joiner。
因此下一步只检查“骨架条件替换是否能由局部信息可靠判断”。

原始结果：
`results/formal/static_topology_and_replacement_probe_N20_rician_20260716.json`。

## 7. 实验 R3：局部消息能否识别有益 challenger

### 7.1 新概念：反事实替换收益

“反事实”不是新信道模型，而是问一个假设问题：如果让 challenger $c$ 替换当前冲突
winner，并重新执行第二阶段，最终 $\sum QR$ 会变好还是变坏？真实差值记为

$$
\Delta_c^{\rm true}=U(\text{replace }c\text{ then complete})-U(\text{Pure}).
$$

集中式 oracle 能逐一试完再看 $\Delta_c^{\rm true}$，分布式节点做不到。R3 比较四种
无需训练的近似：自身理想效用交换、骨架全局收益参考、一跳骨架收益、以及第二阶段后
的一跳局部收益。只有后一种直接观察“替换怎样改变附近 active 链路的真实速率”。

### 7.2 结果与现象

在 3512 个候选替换上，一跳最终局部收益与真实全网收益的 Pearson 相关系数为
**0.968**，符号判断准确率为 **96.13%**。按该分数选择一次替换，三种队列的平均
效用变化为：

| 队列状态 | 平均变化 | 严格改善 | 严格下降 |
|---|---:|---:|---:|
| sparse | +10.72% | 7/12 | 2/12 |
| heterogeneous | +12.82% | 9/12 | 3/12 |
| saturated | +6.96% | 8/12 | 3/12 |

均值很大，但存在 8/36 次下降，违反“不能用不稳定损失换平均增益”的停止条件，所以
一跳规则没有通过。

把收益汇总范围扩到两跳后，结果逐样本等于精确 oracle：heterogeneous +16.23%，
saturated +8.29%，无下降。可是消息范围审计发现：

| 影响范围 | 平均覆盖全网链路比例 | 中位数 | 完全覆盖全网的候选比例 |
|---|---:|---:|---:|
| 一跳联合影响域 | 90.86% | 97.32% | — |
| 两跳联合影响域 | **98.80%** | **100%** | **98.80%** |

这里的范围是一次替换所改变的链路及其邻域的并集，不是单个节点的 degree。由于链路
冲突图很密，多个被撤换/补充链路的邻域迅速覆盖全网。因此“两跳成功”本质上是近全网
计算，不是低开销分布式算法。

### 7.3 说明了什么

R3 找到了真正的科学矛盾：

- 多用户分集由 $QC^{\rm solo}$ 保留“谁更紧急、谁自身更强”；
- 空间复用要求知道一次替换对当前整个 active set 的累计外部性；
- 一跳信息已经接近正确，但少量遗漏会造成不可接受的反向决策；
- 两跳能消除误差，却几乎等于把全网信息都收集回来。

这给 GAT 的最后一次合理机会不是预测新概率，而是用固定轮数消息对一跳替换收益做
远端安全修正。原始结果：
`results/formal/local_replacement_realizability_N20_rician_20260716.json`。

## 8. 实验 R4：set-conditioned challenger GAT 最终闸门

### 8.1 模型定义

R4 是唯一一次后续训练，没有调参分支。流程为：

1. 冻结正式会议 channel encoder；
2. Pure 先形成骨架并执行原条件复用；
3. 每个 inactive challenger 用一跳信息描述它会替换哪些 winner；
4. 新增两轮队列条件 GAT，同时给所有 challenger 打分；
5. 与固定的“保持 Pure 不变”选项竞争；
6. 训练标签是最终真实 $\sum QR$ 最好的单次替换，若所有替换都更差则选择不变。

输入只包含冻结信道 embedding、相对队列、归一化单链路速率、Pure 骨架/最终动作标记、
候选标记、理想效用交换、一跳骨架物理收益和撤换 winner 比例。**没有把一跳最终收益、
两跳收益或真实全网标签作为输入**，否则等于泄漏待预测答案。

为了判断图消息是否真的有用，同步训练一个使用完全相同逐链路输入、但不做任何消息
传递的 MLP。只有 GAT 同时超过 Pure 和 MLP，才能把增益归因于图结构。

### 8.2 数据和停止条件

- 训练：12 张图、36 个队列快照、3801 个候选；
- 验证：4 张图、12 个快照、1204 个候选；
- 未见评估：12 张图、36 个快照、3689 个候选；
- 每组都包含 sparse、heterogeneous、saturated；
- 训练 20 epoch，会议 encoder 始终冻结。

预设条件要求 heterogeneous 和 saturated 中，GAT/Pure 均值至少 1.02、胜率至少
70%、损失率不超过 10%；GAT/MLP 均值至少 1.01 且胜率至少 70%；sparse/GAT-Pure
不得低于 0.99。

### 8.3 结果

| 数据/模型 | 最优候选选择准确率 | 相对 Pure 平均比 |
|---|---:|---:|
| 验证 MLP | 16.67% | 1.0023 |
| 验证两轮 GAT | **0%** | **1.0000** |

两轮 GAT 在未见评估的三种队列上都选择“不替换”，因此相对 Pure 的均值严格为 1、
胜率为 0。MLP 虽偶尔发起挑战，但 sparse 平均下降 6.73%，heterogeneous 平均下降
4.81%，只有 saturated 平均提高 0.78%。它同样不可靠，也没有值得保留的 1% 以上
稳定收益。

### 8.4 为什么不是再训练几轮

GAT 不是因为某一个阈值略偏而只差 1%；它在验证集最优候选准确率为 0，并退回
no-change。更深层原因是：最终标签取决于“每个 challenger 替换后，第二阶段会重新
选出哪些 joiner，以及这些 joiner 的累计干扰”。本地骨架特征没有显式包含这个反事实
active set；而把它完整算出来又需要近全网范围，已经失去低交互优势。

增加 epoch、层数、辅助分类 loss 或手工降低 no-change 阈值，只会迫使模型更频繁地
猜测挑战，无法补回缺失的反事实集合。按照止损规则，本路线到此结束。

原始结果：
`results/formal/set_conditioned_challenger_gate_N20_rician_20260716/report.json`。
其中 `gat_model.pth` 是**被闸门拒绝的诊断 checkpoint**，不得作为正式模型加载。

### 8.5 消息轮数和开销边界

R0--R3 的 oracle/枚举是集中式诊断，不得把它们的计算次数隐藏后称为固定轮算法。
R4 若继续部署，会在已有 Pure 每时隙 3 轮之外新增两轮 queue-conditioned GAT；冻结
会议 encoder 还需要每信道块 4 轮图传播，若分发其输出再保守计 1 轮。也就是说，在
尚未计 challenger 一致仲裁和替换确认的情况下，最低逻辑深度已经是：

$$
3\ \text{轮/slot}+2\ \text{轮/slot}+5/T_c\ \text{轮/slot}.
$$

当 $T_c=5$ slots 时至少为 6 轮/slot，且还缺候选仲裁轮数。更重要的是，R3 的候选
联合影响域一跳平均覆盖 90.86% 链路、两跳平均覆盖 98.80%，所以不能用“只有两层
GAT”掩盖实际 payload 可能接近全网扩散。由于性能闸门已经失败，本轮不再虚构 MAC
打包方案或优化该 rejected 模型的带宽。当时保底 Pure 仍按已有审计计 3 轮/slot；
其逐边单播上界与发射 UAV 聚合下界见
`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md` 的 O.4 节。

## 9. 实验 R5：Q-aware GAT 直接输出完整骨架

### 9.1 用户提出的问题

此前失败会不会是因为会议 encoder 只学习 channel-only sum-rate，而 GAT 不知道当前
队列，所以无法主动让骨架朝更大的 $\sum QR$ 调整？进一步，能否让 GAT 不再只输出
一个缩放因子或 challenger，而是直接输出新骨架中每条链路的 keep/add/drop？

这个问题包含两个可分开的判断：

1. **必要性：** 同一信道图的正确骨架修改是否真的随 $Q$ 改变？
2. **可学习性：** 给 GAT 提供 $Q$ 和 Pure 骨架后，它能否在未见图上学会联合修改？

### 9.2 零训练必要性检查

R0 的 12 张信道图各自都有 sparse、heterogeneous、saturated 三种队列。保持信道
完全不变，只比较 oracle 选择的最佳 challenger：

- 12/12 张图在三种队列下出现不同的最佳修改；
- 三种队列两两比较，94.44% 的最佳 challenger 不同；
- 每张图平均出现 2.83 种不同修改。

因此用户的第一点判断成立：骨架修正不是 channel-only 属性。$Q$ 不只应在最后乘一下
权重，而应参与“谁和谁竞争、谁需要退出”的图条件判断。

### 9.3 最小模型定义

为了不立刻重回难以归因的 direct-$\sum QR$ 训练，本轮先做更便宜的监督可学习性闸门：

1. 冻结健康会议 channel encoder，保留其信道 embedding；
2. 新的两轮 GAT 同时读取相对队列、单链路速率、Pure 骨架、Pure 完成动作和冲突度；
3. 对每条链路直接输出“属于新骨架/不属于新骨架”的 logit；
4. 输出以 Pure 骨架作为 residual identity 起点；零 residual 严格复现 Pure；
5. 监督目标是 R0 中最终 $\sum QR$ 最好的单次替换骨架；
6. 输出冲突时仍用统一的一跳 LocalMax 做确定性消冲突；
7. 同时训练完全相同输入、但不交换消息的 MLP。

这里的 class-balanced membership BCE 只让稀疏骨架中的 active/inactive 两类具有相等
训练权重，不添加 sum-rate、稀疏度或 flip 数量惩罚。

### 9.4 与旧模型的区别

- 与早期 end-to-end GAT-q 不同：不从空白猜完整动作，不先用难归因的 $\sum QR$
  梯度；从可靠 Pure 骨架开始，只学习残差修改。
- 与 R1 scalar-$z$ 不同：输出是直接的骨架 membership，不是预先缩放每条链路。
- 与 R4 challenger GAT 不同：不是在“一个 challenger/不变”之间分类，而是所有链路
  同时输出 keep/add/drop，理论上允许联合修改。

训练规模按低成本闸门冻结：8 张训练图、2 张验证图、8 张未见评估图，每张图三种
队列，10 epochs，不做第二组超参数。

### 9.5 结果

训练 24 个快照中 22 个 oracle 要求修改 Pure；验证 6/6 要求修改；未见评估 24 个
快照中 19 个要求修改。因此模型没有因为“数据里根本不需要改”而选择 identity。

| 模型 | 验证 changed-node recall | 未见 changed-node recall | 未见效用/Pure |
|---|---:|---:|---:|
| 同输入 MLP | 0 | 0 | 1.000 |
| 两轮 Q-aware GAT | **0** | **0** | **1.000** |

分队列结果如下：

| 队列状态 | GAT/Pure | GAT 严格改善 | Oracle/Pure | Oracle 严格改善 |
|---|---:|---:|---:|---:|
| sparse | 1.000 | 0/8 | 1.0554 | 4/8 |
| heterogeneous | 1.000 | 0/8 | **1.1353** | **8/8** |
| saturated | 1.000 | 0/8 | **1.1167** | **7/8** |

最低验证 membership loss 选择的 GAT/MLP checkpoint 都完整保留 Pure。MLP 在最后
一个 epoch 的小验证集曾短暂出现 +3.39%，但同一 epoch 训练集反而 -3.07%、验证 loss
变差，而且它不是 GAT；按照预先按验证监督 loss 选模型的规则，不能把这个偶然点当作
成功或继续调参的理由。

### 9.6 原因分析

$Q$ 缺失确实是旧 channel-only 接口的问题之一，但加入 $Q$ 后仍存在更根本的联合信用
分配：一次正确修改通常是在上百条链路中加入 1 条 challenger，同时撤掉 1--2 条
winner。单个节点仅知道“我的 membership 标签变了”，却不知道它必须与哪几个邻居
同步翻转，才能在第二阶段产生有益的新 active set。

因此逐节点 membership loss 最安全的解仍是复现 Pure。直接换回 $\sum QR$ 梯度只会
让这个稀疏联合翻转问题重新变得更模糊，正是早期 end-to-end GAT-q 已暴露的困难。

### 9.7 决策

预设闸门要求 heterogeneous 和 saturated 中 GAT/Pure 均值至少 1.02、胜率至少
5/8、损失不超过 1/8，并至少比同输入 MLP 高 1%。实际增益和胜率均为 0，明确失败。

因此停止在监督可学习性阶段：

- 不进入 direct-$\sum QR$ fine-tune；
- 不增加 epoch、层数、类别权重或 flip 辅助 loss；
- 不降低输出阈值强迫模型修改骨架；
- 结论是“Q-aware 必要但不充分”，不是“队列不重要”。

原始结果：
`results/formal/queue_aware_skeleton_residual_gate_N20_rician_20260717/report.json`。
其中 `gat_probe.pth` 已标记 `status=rejected`，不得正式加载。

## 10. 实验 R6：连续会议 $\pi$ 进入 SINR 是否遗漏

### 10.1 用户的新想法与代码核对

用户提出：会议 GAT 已经输出每条链路的连续调度概率 $\widetilde\pi_j$，是否可以把它
当作邻居计划发送活动度，直接代入 SINR，再用新速率形成队列骨架？

代码核对确认，这正是 LocalPhysicalReuse-v2 已经实现并正式测试的公式：

$$
\widehat I_i=\sum_{j\ne i}\widetilde\pi_jg_{ji},\qquad
\widehat R_i=\log_2\!\left(1+\frac{g_{ii}}{\sigma^2+\widehat I_i}\right),
\qquad W_i=Q_i\widehat R_i.
$$

`sigmoid(logit)` 后的 $\widetilde\pi_j$ 保持连续，没有 round。随后只用 $W_i$ 做
LocalMax，第二阶段不变。因此这不是遗漏方案，而是 v2 名称中 `GATExpectedRate` 的
准确含义。

### 10.2 为什么连续 $\pi$ 不是“一张实际发送图”

$\widetilde\pi_j$ 是每个节点的边缘活动倾向。把各个概率分别代入
$\sum_j\widetilde\pi_jg_{ji}$，相当于只保留平均干扰，丢失“哪些链路作为一个联合
集合同时发送”的相关性。会议真正的联合图是按阈值得到的二元动作；但它是 channel-only
固定偏好的联合图，不知道当前谁的队列更紧急。

这两种语义都已经测试：

- soft expected activity：当前 v2；
- hard conference action + 相同物理复用：`GAT-HardSeed-Reuse`。

正确 N=20 动态结果中，hard seed 在 $\Lambda=2.2/4$ 的吞吐为 18.776/20.836 Mbps，
明显低于 Pure 的 24.883/28.649 Mbps。它长期服务同一小组 channel-efficient 链路，
其他持久队列被饿死。

### 10.3 新发现：空队列假干扰

当前 v2 的 expected interference 只依赖信道，不检查干扰链路是否有包。因此即使
$Q_j=0$、链路 $j$ 当前绝不可能发送，$\widetilde\pi_jg_{ji}$ 仍被计入。物理一致的
最小修正是

$$
\widehat I_i^{\rm nonempty}
=\sum_{j\ne i}\widetilde\pi_j\mathbf1[Q_j>0]g_{ji}.
$$

没有重新训练模型，只比较 current v2、nonempty-mask v2、Pure，以及相同均值
Constant 的 masked/unmasked 版本。12 张图、三种队列与 R0 共享设置。

### 10.4 queue-mask 结果

| 队列状态 | 平均非空比例 | Masked GAT / 当前 v2 | Masked GAT / Pure | Masked GAT / Masked Constant |
|---|---:|---:|---:|---:|
| sparse | 27.08% | 1.0351 | 0.9880 | 0.9841 |
| heterogeneous | 89.45% | 0.9842 | 0.8898 | 0.8924 |
| saturated | 100% | 1.0000 | 0.9829 | 0.9587 |

sparse 平均 +3.51%，但只在 4/12 图改善、2/12 图下降；heterogeneous 平均反而
下降 1.58%；saturated 因全部队列非空而与当前 v2 严格相同。预设的 1% 均值、60%
胜率闸门失败。

queue mask 是正确的物理语义澄清，却没有形成稳定性能价值；而 Constant mask 在 sparse
平均 +6.59%，进一步说明变化主要来自清除空队列假干扰，不是 GAT 学到的链路差异。

原始结果：
`results/formal/queue_masked_expected_rate_probe_N20_rician_20260717.json`。

## 11. 实验 R7：让 GAT 与 Pure 各自产生一个候选

### 11.1 动机

与其要求一个骨架公式同时兼顾空间复用和队列紧急性，可以把它们视为两个专家：

- Pure 产生 queue-urgent 候选，强调 $QC^{\rm solo}$；
- conference GAT 产生 channel-efficient 候选，强调兼容发送集合；
- 当前时隙再选择真实 $\sum QR$ 最大的候选。

这是连接会议与期刊最自然的“候选码本”思路。R7 先用集中式精确选择测上界；如果连
上界都没有稳定 GAT 特异收益，就不开发分布式 selector。

GAT 候选库包含 Pure、连续-$\pi$ expected-rate v2、会议 hard-seed reuse；Constant
对照库包含 Pure 和同均值 Constant expected-rate。所有候选执行相同第二阶段。

### 11.2 结果

| 队列状态 | GAT 候选库 / Pure | 选择 GAT 次数 | Constant 候选库 / Pure | GAT 库 / Constant 库 |
|---|---:|---:|---:|---:|
| sparse | 1.0530 | 4/12 | 1.0561 | 0.9986 |
| heterogeneous | 1.0305 | 3/12 | 1.0273 | 1.0030 |
| saturated | 1.0184 | 4/12 | **1.0565** | **0.9659** |

由于选择器总能保留 Pure，候选库不会下降。但 heterogeneous/saturated 的 GAT 选择
胜率只有 3/12、4/12，低于预设 60%；相对 Constant 候选库也没有 1% 优势。尤其在
saturated，Constant 候选库明显更好。

会议 hard-seed 只在一个 sparse 快照成为最佳候选，在 heterogeneous 平均只有 Pure
的 70.44%，12/12 都更差；真正偶尔产生互补性的仍是连续-$\pi$ v2，而其收益可被
Constant 的数值校准复制。

### 11.3 决策

R7 证明 GAT 候选在个别拓扑不是完全无用，但没有证明值得为它增加第二套候选的生成、
评价、全网选择和一致执行消息。停止开发分布式二候选 selector。

若未来有充足研究周期，真正更有逻辑的版本不是重复同一个会议输出，而是让 GAT 每个
信道块生成多个**不同但物理兼容的联合调度码字**，队列时隙只在小码本中做 Max-Weight
选择。这保留联合相关性，也把 channel learning 与 queue selection 分工清楚；但需要
解决多样化训练、码本覆盖率和分布式一致选择，已经超出当前截稿和额度范围。

原始结果：
`results/formal/gat_candidate_portfolio_probe_N20_rician_20260717.json`。

## 12. 实验 R8：信道块级骨架码本可行性闸门

### 12.1 为什么做这个实验

R7 只让会议 GAT 产生一个额外候选，可能低估了图表示的价值。新的设想是：对同一
信道图产生 $K$ 个不同的二元骨架，队列时隙按当前 $Q$ 选择一个骨架，随后只对被选
骨架执行一次现有条件物理复用。它试图把两种能力分开：

- GAT 在信道块尺度提供多种空间兼容模式；
- 快队列阶段在这些模式中实现多用户分集；
- 后半段物理协商仍只运行一次，不把网络训练成最终完整调度器。

码本写作

$$
\mathcal C(H)=\{\boldsymbol a_0^{(1)},\ldots,
\boldsymbol a_0^{(K)}\},\qquad a_{0,i}^{(k)}\in\{0,1\}.
$$

各码字只依赖块信道 $H$，评估队列到来后先删除空队列链路，再用骨架自身真实 SINR
下的 $\sum_iQ_iR_i(\boldsymbol a_0^{(k)})$ 选择一个。最后只调用一次
`complete_physical_reuse_from_base`。

### 12.2 两层上界为什么都需要

第一层 `physics-multistart` 完全不使用队列。它从 solo rate、全开干扰下速率和有向
损害分数出发，用固定扰动生成多组互不冲突骨架，再保留高 channel-only sum-rate 且
彼此不同的候选。

该生成器若失败，不能立即证明所有码本都失败，因为它可能只是候选不够好。因此又
加入更强的 `calibration teacher`：对与正式评估完全分开的12个校准队列，收集 Pure
骨架和一次最佳替换骨架，然后贪心压缩成固定 $K$ 个码字。校准队列只相当于未来训练
多头网络时的离线教师；码本冻结后，正式评估时仍不能读取当前队列。

每个码本同时报告两种结果：

1. **合法单次协商结果**：先根据未完成骨架的真实 $QR$ 选择，再完成一次；
2. **all-completions oracle**：把全部 $K$ 个骨架都完成后挑最好，只用于区分“码本中
   没有好骨架”和“选择器选错”，不能作为部署算法。

正式设置仍为12张未见 N=20、Rician K=5 dB、10 dBm 图，三种队列快照，
$K\in\{2,4,8\}$。不训练模型、不跑动态队列。

### 12.3 无队列物理码本结果

先看 $K=8$：

| 队列状态 | 合法选择后一次协商 / Pure | all-completions oracle / Pure |
|---|---:|---:|
| sparse | 0.9877 | 1.0999 |
| heterogeneous | 0.8502 | 0.9305 |
| saturated | 0.9773 | 1.0474 |

无队列物理扰动确实生成了不同骨架，但异质队列连违规 oracle 都只有 Pure 的0.9305。
这说明“channel sum-rate 高且彼此不同”不等于能覆盖队列紧急性的变化。

### 12.4 更强 calibration-teacher 结果

| 队列状态 | $K$ | 合法选择后一次协商 / Pure | all-completions oracle / Pure | 选择器命中 oracle |
|---|---:|---:|---:|---:|
| sparse | 2 | 0.9769 | 1.0472 | 7/12 |
| sparse | 4 | 0.8923 | 1.0638 | 3/12 |
| sparse | 8 | 0.9399 | 1.0659 | 2/12 |
| heterogeneous | 2 | 0.8925 | 0.9864 | 5/12 |
| heterogeneous | 4 | 0.8526 | 1.0038 | 3/12 |
| heterogeneous | 8 | 0.8807 | 1.0147 | 4/12 |
| saturated | 2 | 1.0290 | 1.0361 | 11/12 |
| saturated | 4 | 0.9974 | 1.0421 | 7/12 |
| saturated | 8 | 0.9849 | 1.0596 | 2/12 |

预设闸门要求 heterogeneous 与 saturated 同时达到平均 +5%、至少60%图改善。
`calibration teacher` 在两种状态都未通过，因此没有继续运行 GAT/shuffled-pi/
Solo-SNR 第二层，更没有训练多头模型。

### 12.5 结果说明了什么

首先，异质队列是码本覆盖瓶颈。即使教师先在另一批队列上观察了多种 Pure 和 oracle
骨架，8码字在未见异质队列上全部完成后的上限也只有 +1.47%，且只在5/12图改善。
固定小码本很难覆盖“此时究竟哪几条链路最紧急”的高维组合。

其次，saturated 中不是完全没有候选价值。8码字 all-completions oracle 达到 +5.96%，
9/12图改善；但合法选择只有 -1.51%。候选增加时 oracle 上升、真实选择反而下降，原因
是后半段加入者和受害链路损失会改变最终排序。骨架自身 $QR$ 不能可靠预测“一轮条件
协商以后”的 $QR$。

若为了解决这个问题而把 $K$ 个骨架全部协商再选最好，就重新引入 $K$ 份物理计算、
候选评价和一致选择；若训练一个网络预测每个骨架完成后的价值，又回到了此前失败的
集合条件信用分配。因此这不是值得在截稿期继续修补的接口。

### 12.6 消息开销与停止边界

即使只完成一个骨架，每时隙仍要为 $K$ 个码字计算本地 $Q_iR_i^{(k)}$，并做一次全网
$K$ 维效用聚合和统一索引决定。它不是当前冲突邻域内固定一轮 LocalMax；真实邻居
实现至少需要随图直径传播的一致性过程。选定后，还需要现有的骨架动作/物理量交换和
候选收益竞争两轮局部消息。

因此 R8 的停止结论是 `STOP_CODEBOOK_CEILING_NOT_MATERIAL`：不训练多头 GAT，不增加
候选数，不把 saturated 的不可部署 oracle 当作模型成功。该结论只否定当前
“$K\le8$、块级固定骨架、先选一个、再完成一次”的路线；它不是数学上证明所有可能
的 Q-aware 联合模型都不可能，但后者会回到已经证实高风险的端到端联合动作学习。

原始结果：
`results/formal/skeleton_codebook_gate_N20_rician_20260717.json`。

## 13. 最终保底策略与研究边界

该阶段当时进入保底：Pure 骨架 + 解析条件物理复用作为队列调度主干；会议 GAT 保留为前作
和 channel-only 容量模型。论文可以诚实表述：会议 GAT 提供信道图学习起点，期刊扩展
发现直接迁移其 $\pi$ 会产生 sum-rate 偏置，因此采用 queue-faithful 的物理两阶段
调度，并通过消融揭示图先验接口的适用边界。

不能表述为：新 GAT 已扩大队列稳定域、两跳 challenger 是低开销分布式算法、或 CSI
鲁棒 GAT 已被验证。若投稿时间允许，后续价值更高的是把“空间复用收益与消息覆盖范围”
作为敏感性/复杂度分析，而不是继续训练模型。

## 14. 复现入口

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/diagnose_mws_topology_scaling_headroom.py

conda run --no-capture-output -n pytorch python -u \
  train/train_mws_topology_scaling.py

conda run --no-capture-output -n pytorch python -u \
  experiments/diagnose_static_topology_and_replacement.py

conda run --no-capture-output -n pytorch python -u \
  experiments/diagnose_local_replacement_realizability.py

conda run --no-capture-output -n pytorch python -u \
  train/train_set_conditioned_challenger.py

conda run --no-capture-output -n pytorch python -u \
  train/train_queue_aware_skeleton_residual.py

conda run --no-capture-output -n pytorch python -u \
  experiments/diagnose_queue_masked_expected_rate.py

conda run --no-capture-output -n pytorch python -u \
  experiments/diagnose_gat_candidate_portfolio.py

conda run --no-capture-output -n pytorch python -u \
  experiments/diagnose_skeleton_codebook_gate.py
```
