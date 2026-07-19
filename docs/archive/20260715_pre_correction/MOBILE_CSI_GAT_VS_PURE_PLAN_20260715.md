# 移动与不完美 CSI：冻结基线、CSI 增强重训与最小时空 GAT 计划

> **文档状态：2026-07-15 已执行计划，不再作为当前计划。** 最新物理修正、根因分析
> 与十天路线统一见
> [`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`](MOBILE_BLOCK_FADING_CSI_TUTORIAL.md)。

> 文档定位：本文是 2026-07-15 固定 CSI 结果更新后的当前执行计划。
> `docs/CSI_CHAT_HANDOFF_20260715.md` 是事实基线；
> `docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md` 中 E9--E12 的结果保留为历史风险诊断，
> 但其中“下一步先做解析预测”的顺序已被本文重新安排。

> 时间约束：距离投稿约十天。第一轮先用冻结会议 GAT 建立“零改造鲁棒性基线”；
> 若它不能抵抗 CSI 变化，不立即否定图学习，而是继续检查“增加移动 CSI 训练数据”
> 和“在会议 GAT 上增加最小时间维度”能否恢复性能。停止条件放在这两类最小改造之后，
> 而不是放在冻结模型诊断之后。

> **Phase A--F 已完成：** 正式80-run结果表明，Frozen-GAT 在 `Lambda=4` 的
> stale、large-scale 和4-bit distance条件分别取得5/5、4/5、5/5联合胜利；
> `Lambda=2.2`没有条件通过。随后同架构单帧 CSI 增强虽降低 teacher-prior 重建
> 误差，却未稳定超过 Frozen；最小 Temporal-Residual GAT 的独立验证最终选择
> 零残差起点，40-run 队列结果与单帧增强版逐项相同。完整记录见
> `docs/MOBILE_CSI_FROZEN_GAT_VS_PURE_TUTORIAL.md`、
> `docs/CSI_AUGMENTED_SPATIAL_GAT_TUTORIAL.md` 和
> `docs/TEMPORAL_RESIDUAL_GAT_TUTORIAL.md`。当前已触发 Phase-F 止损，不再扩大
> 时间模型搜索。
>
> **后续审计更新：** Frozen teacher在1200个channel-only状态中阈值动作全部全开，
> 因而Phase-F只否定teacher-logit residual。Rayleigh/Rician/LoS及骨架--协商反事实
> 证明信道模型显著影响可预测性，并为“物理gain/rate预测”保留一次重新定义目标的最小
> 验证资格。见 `docs/CHANNEL_MODEL_AND_STAGE_MASKING_TUTORIAL.md`。

---

## 0. 一页总结

### 0.1 现在真正要回答的问题

旧移动实验已经证明：对 **GAT-assisted LocalPhysicalReuse-v2** 而言，
one-block stale CSI 会显著降低服务、增加积压。

但是，这个结果还不能回答：

> 问题究竟来自 GAT 对错误 CSI 特别敏感，还是任何使用相同错误 CSI 的物理调度器
> 都会受到类似损害？

因此，当前首要任务不再是立即增加信道预测器，而是把同一个物理调度框架拆成：

1. 使用会议版 GAT 信道先验的 **GAT-assisted**；
2. 不使用 GAT、只使用解析单链路速率的 **Pure**。

然后让两者面对完全相同的移动轨迹、到达轨迹和 CSI 观测，直接比较谁在不完美
CSI 下更稳健。

### 0.2 第一阶段最小正式矩阵

| 维度 | 取值 |
|---|---|
| 调度骨架 | GAT-assisted / Pure |
| CSI 条件 | latest / one-block stale / current large-scale / 4-bit distance-quantized large-scale |
| 全网总到达率 | `Lambda=2.2` / `Lambda=4` |
| 随机种子 | 42、43、44、45、46 |

因此正式矩阵共有：

$$
2\text{ 个骨架}\times4\text{ 个 CSI}\times2\text{ 个负载}\times5\text{ 个 seed}
=80\text{ 次运行}.
$$

共同设置为 `N=10`、1000 slots、5 m/s、每个真实信道块 5 slots、无 deadline、
无限缓存。

### 0.3 第一层闸门：冻结模型的天然鲁棒性

对每个 imperfect-CSI 条件和每个负载，逐 seed 定义：

$$
\Delta D_s=D_s^{\mathrm{GAT}}-D_s^{\mathrm{Pure}},
$$

$$
\Delta Q_s=Q_s^{\mathrm{Pure}}-Q_s^{\mathrm{GAT}}.
$$

其中 $D$ 是完整包 delivered throughput，$Q$ 是平均 backlog。这样二者为正都表示
GAT 更好。

一个 seed 只有在

$$
\Delta D_s>0\quad\text{且}\quad\Delta Q_s>0
$$

时才记为一次 **联合胜利**。

若至少一种 imperfect-CSI 条件满足：

1. 至少 4/5 seeds 联合胜利；
2. 五个 seed 的平均 delivered throughput 更高；
3. 五个 seed 的平均 backlog 更低；
4. 优势不只是 physical service 更高；

说明会议 GAT 本身已经保留一定天然鲁棒性，并给后续时间建模提供了正向基线。

若没有任何条件满足，结论只能是：

> **原始会议 GAT 在未见过的移动/不完美 CSI 分布上没有天然 resilience。**

这不是整个 GAT 路线的停止结论。它会触发第二层实验：先用同一空间 GAT 做 CSI
增强重训，再增加一个最小时间模块。只有二者都不能相对 Frozen-GAT 和 Pure 形成
一致优势，才停止 CSI 模型开发。

---

## 1. 当前模型是什么

两个方法共享同一个第二阶段，只在“谁先形成局部骨架”上不同。

### 1.1 GAT-assisted LocalPhysicalReuse-v2

正式类：

`GATExpectedRateLocalPhysicalReuseScheduler`

每个信道块，冻结的会议版 channel-only GAT 根据观测 CSI 产生链路活动先验。
该先验被转换为考虑期望邻居干扰的有效速率 $\widetilde C_i$。

每个时隙使用

$$
W_i^{\mathrm{GAT}}(t)=Q_i(t)\widetilde C_i(t)
$$

进行局部最大竞争，形成多个局部胜者组成的二值骨架。

### 1.2 PurePhysicalReuse

关键消融类：

`PurePhysicalReuseScheduler`

Pure 不运行 GAT，而使用观测 CSI 下的无干扰单链路速率 $C_i^{\mathrm{solo}}$：

$$
W_i^{\mathrm{Pure}}(t)=Q_i(t)C_i^{\mathrm{solo}}(t).
$$

### 1.3 两者后续步骤完全相同

骨架形成后，两者都：

1. 以当前骨架为条件计算候选链路自己的收益；
2. 计算候选加入会对已激活邻居造成的 SINR 损害；
3. 只保留局部边际收益为正的候选；
4. 再做一轮并行局部竞争，补充空间复用链路。

因此，GAT 与 Pure 的差异能够归因到 **GAT 信道先验是否改善了初始局部骨架**，
而不是两者使用了不同的物理复用算法。

---

## 2. 为什么固定 CSI 的新结果改变了旧计划

### 2.1 固定 CSI 已经得到的结果

最新 N=10、10000 slots、5-seed、无 deadline 结果表明：

- `Lambda=2.0` 时两者稳定，Pure 的 backlog 更低；
- `Lambda=2.2` 时 GAT 修复了一个 Pure 失败的困难 seed，但并非多数 seed 都更好；
- `Lambda>=2.4` 后两者总体接近，优劣方向随 seed 改变；
- 离线 oracle 固定 $\bar\rho_i$ 相对 Pure 的快照效用只提高约 0.41 个百分点。

这说明在 perfect fixed CSI 下，GAT 尚未表现出一致的稳定域提升。继续训练简单
rho-GAT 或扩大固定 CSI 结构搜索，预期收益很低。

### 2.2 旧计划为什么需要后移

旧移动教程原计划先做：

1. 信道块与 CSI 更新周期解耦；
2. 解析一步预测；
3. 必要时再做学习预测。

现在必须先插入 Frozen-GAT/Pure 对照。它的作用是建立零改造基线和定位问题，
不是提前否定时间图学习。若 Frozen-GAT 在 stale/large-scale CSI 下与 Pure 一样
或更差，需要继续区分两种原因：

1. **训练分布不匹配：** 会议 GAT 只见过固定 perfect CSI，换成移动或粗糙 CSI 后
   输入统计变化；仅增加对应训练数据可能已经足够；
2. **缺少时间状态：** 单个观测帧无法表达信道变化趋势，需要沿永久 link identity
   连接相邻信道块的表示。

只有把这两个原因分别实验后，才能判断时空 GAT 是否必要。

### 2.3 新的研究顺序

```text
会议 GAT checkpoint 作为空间骨架
        ↓
Frozen-GAT / Pure 做 2×4 零改造配对
        ↓
Frozen-GAT 是否已有天然 resilience？
        ├── 是：得到正向基线，检查时间模块是否还有明确增量
        └── 否：不能直接止损，进入最小 CSI 适配
                         ↓
      同架构 CSI-augmented GAT：只换移动/扰动训练数据
                         ↓
      最小 Temporal-Residual GAT：会议空间 GAT + 一层时间记忆
                         ↓
时间模型是否同时胜过 Frozen-GAT、CSI-augmented GAT 和 Pure？
        ├── 是：保留为期刊时空扩展候选
        └── 否：停止扩大 STGNN，CSI 作为敏感性/未来工作
```

---

## 3. 四种 CSI 条件分别是什么意思

四种条件只改变调度器看到的信道。所有方法的实际服务始终使用当前真实瞬时信道。

### 3.1 Latest instantaneous CSI

调度块 $b$ 时观察块 $b$ 的真实直接功率和交叉干扰功率。

它是移动环境的参考条件，不表示 CSI 可以零成本获得。

### 3.2 One-block stale instantaneous CSI

调度块 $b$ 时仍观察块 $b-1$ 的瞬时功率；块 0 使用 latest 暖启动。

它模拟 CSI 晚一个真实信道块到达。当前 5-slot 块对应 5 ms 延迟。

### 3.3 Current large-scale CSI

调度器知道当前 UAV 距离和路径损耗，但看不到当前 small-scale fading。

它回答：当前几何信息能否替代快速变化的瞬时 CSI？

### 3.4 4-bit distance-quantized large-scale CSI

先把当前 UAV 距离量化到 16 个区间，再由区间代表距离重建路径损耗。

它不是瞬时复信道的 4-bit 幅相量化，也不是 GAT logit 量化。正式图表必须使用完整
名称，不能只写模糊的 `4-bit CSI`。

---

## 4. 配对实验怎样保证公平

### 4.1 什么叫配对 seed

seed 不是只控制一个随机数。每个 seed 固定：

- 初始 UAV 位置；
- 移动方向；
- 每个块的小尺度衰落创新；
- 永久有向链路目录；
- 热点业务链路；
- 每个时隙的到达包数。

同一个 seed、同一个负载下，8 个“骨架 × CSI”方法共享这些随机量。

### 4.2 真实信道与观测 CSI 必须分离

每个运行包含两套状态：

1. **观测状态**：用于 GAT、Pure 骨架、冲突图和条件复用决策；
2. **真实状态**：只用于计算最终动作实际得到的 SINR 和服务。

如果 stale 方法既用旧 CSI 决策，又用旧 CSI 结算服务，错误动作会在错误信道里
显得正确。当前 `mobile_block_fading.py` 已经正确分离两者，本轮不得重写。

### 4.3 Link identity 为什么不需要重做

永久链路身份是有向 UAV 对 $(s,d)$。链路暂时不可用时，队列仍保留，只把动作
强制为零；恢复可用后继续服务原队列。

该接口已有单元测试。本轮只增加 GAT/Pure 和包级指标，不重新实现流目录。

---

## 5. 正式实验设置及每个参数的理由

| 参数 | 设置 | 理由 |
|---|---:|---|
| UAV 数 | 10 | 与固定 CSI 主结果和会议 GAT checkpoint 对齐，先做快速归因 |
| slots | 1000 | 比旧 300-slot 风险诊断更可靠，同时能在两天内完成 |
| seeds | 42--46 | 与固定 CSI 的五个配对拓扑一致 |
| 速度 | 5 m/s | 先避免同时扫描大量移动条件 |
| 真实信道块 | 5 slots = 5 ms | 相邻块 $\rho\approx0.643$，历史 CSI 仍有中等信息，适合判断鲁棒性 |
| `Lambda=2.2` | 边界负载 | 固定 CSI 下最容易暴露困难 seed 的位置 |
| `Lambda=4` | 压力负载 | 队列长期非空，容易暴露并发集合的物理服务能力 |
| deadline | 无 | 当前任务是稳定性/服务能力，不引入额外丢包机制 |
| buffer | 无限 | 避免 overflow 把调度损失截断成丢包 |
| quantization | 4-bit distance | 旧诊断显示 4-bit 已接近连续 large-scale，先不重复 2/8-bit 搜索 |

需要注意：`Lambda=2.2` 只是固定 CSI 的边界负载。移动 CSI 可能降低容量，使它也
变成过载点。若 latest CSI 下所有方法的完成率都明显低于 99%，则不解释 packet
delay；只有为了获得一个稳定区参考时，才补一个 `Lambda=2.0`，不扩大成完整负载扫描。

---

## 6. 三类主指标分别测量什么

### 6.1 Delivered throughput

Delivered throughput 只统计已经完整完成服务的数据包：

$$
D=\frac{N_{\mathrm{completed}}S_{\mathrm p}}
{T_{\mathrm{sim}}}.
$$

它回答“用户在仿真窗口内真正收到了多少完整数据”。

### 6.2 Physical service

Physical service 统计每个时隙实际用于队列的连续服务量。一个尚未完成的包可能已经
接受了部分 physical service，因此它可能大于 delivered throughput。

它用于诊断无线并发集合是否更有效，但不能单独作为 GAT 通过闸门的证据。

### 6.3 Average backlog

对每个时隙的所有永久链路队列取平均，并换算为 Mbit/link：

$$
\overline Q=\frac{1}{TL}\sum_{t=1}^{T}\sum_{i=1}^{L}Q_i(t)
\frac{S_{\mathrm p}}{10^6}.
$$

在相同到达轨迹下，backlog 越低，说明业务被更及时地服务。

### 6.4 Packet delay 什么时候才解释

无 deadline 的有限窗口会留下未完成慢包。若只平均已经完成的包，过载方法可能因为
最慢包尚未完成而得到虚假的较低时延。

因此只有 `successful_delivery_ratio >= 0.99` 时才解释平均包时延和 P95 时延。
否则只报告完成率、delivered throughput、physical service 和 backlog。

---

## 7. 主比较与辅助比较

### 7.1 主比较：同一 CSI 下 GAT 对 Pure

对每个 `CSI × Lambda × seed`，直接计算：

- delivered throughput 差；
- average backlog 差；
- physical service 差。

这是止损闸门真正使用的比较。

### 7.2 辅助比较：各骨架相对自身 latest 的退化

可以额外计算：

$$
L_D^{m}(c)=
\frac{D^{m}(\mathrm{latest})-D^{m}(c)}
{D^{m}(\mathrm{latest})},
$$

其中 $m$ 为 GAT 或 Pure，$c$ 为 imperfect CSI。

它说明哪个骨架从 latest 退化得更少，但不单独作为闸门。原因是一个 latest 基线本来
较低的方法也可能表现出较小的“相对损失”。

### 7.3 为什么不把 MWS utility 当作主指标

不同调度器运行后会形成不同队列。积压更大的方法可能因为 $Q$ 更大而得到更高的
$\sum_iQ_iR_i$，所以动态轨迹中的 utility 不适合跨方法直接判断优劣。

---

## 8. 继续与停止标准的精确定义

### 8.1 Frozen-GAT 天然 resilience 的通过条件

对 `stale`、`large-scale`、`4-bit distance-quantized large-scale` 的每个负载
分别检查。

某一条件通过，必须同时满足：

1. 5 个 seed 中至少 4 个在 delivered throughput 和 backlog 上同时胜过 Pure；
2. 五个 seed 聚合后，GAT 的平均 delivered throughput 更高；
3. 五个 seed 聚合后，GAT 的平均 backlog 更低；
4. 报告配对差值和 95% CI，确认结果不是只由一个异常 seed 驱动；
5. physical service 只作机制解释，不能替代前两项 QoS 结果。

五个 seed 的 CI 很宽时，可以说“方向一致但效应大小不确定”，不能说已经得到强统计
显著性。闸门以预先规定的 4/5 联合方向为主，不在看到结果后改变阈值。

### 8.2 Frozen-GAT 未通过时

若没有任何 imperfect-CSI 条件通过：

- 只说明会议 checkpoint 对新分布没有天然鲁棒性；
- 进入同架构 CSI 增强重训，检查问题是否仅来自训练分布；
- 再进入一个预先定义的最小时间扩展，检查历史信道是否有可学习价值；
- 不重新打开 fixed-CSI rho、ChannelContext 或复杂骨架搜索。

### 8.3 最小基线一：CSI-augmented Spatial GAT

该模型的网络结构与会议 `GATScheduler` 完全相同，并从会议 checkpoint 初始化。
唯一变化是训练样本从固定 perfect CSI 改为成对的：

```text
输入：latest / stale / large-scale / 4-bit large-scale 观测图
结算：对应时刻的当前真实瞬时信道
```

训练仍使用会议版 channel-only 总速率目标：

$$
\mathcal L_{\mathrm{spatial}}
=-\sum_i R_i^{\mathrm{true}}
\left(\boldsymbol\pi(\mathcal G^{\mathrm{obs}})\right).
$$

它不读取队列，也不学习最终 LocalPhysicalReuse 动作。这个实验回答：

> 只要让原 GAT 见过 CSI 抖动，是否已经可以获得鲁棒性？

如果它成功，说明主要问题是训练分布，而不是必须增加时间结构。

### 8.4 最小基线二：Temporal-Residual GAT

这是当前允许开发的最小 STGNN-like 候选，不从零设计新的空间网络。

每个块先用共享参数的会议 GAT 处理当前观测图，得到每条链路的 48 维空间表示
$\mathbf h_i(b)$。永久 link identity 把同一有向链路在相邻块的表示对齐。然后每条
链路本地维护一层 GRU 状态：

$$
\mathbf z_i(b)=
\operatorname{GRUCell}\!\left(\mathbf h_i(b),\mathbf z_i(b-1)\right).
$$

GRU 是一种带门控的时间记忆单元：它学习“保留多少旧信息、接受多少新观测”。这里
时间更新只发生在同一条永久链路上，不需要把历史状态上传到中心。

最终只对会议 GAT 的 channel logit 做一个时间残差：

$$
c_i^{\mathrm{temp}}(b)
=c_i^{\mathrm{conference}}(b)+
\delta_i\!\left(\mathbf z_i(b)\right).
$$

残差头初始化为零，因此训练起点严格等于会议 GAT，而不是随机重新开始。模型仍只
输出 channel-only 活动先验，队列 LocalMax 和条件物理复用完全不变。

训练使用与会议模型相同形式的单一物理目标，但速率在当前真实信道上计算：

$$
\mathcal L_{\mathrm{temporal}}
=-\sum_i R_i^{\mathrm{true}}
\left(\boldsymbol\pi(
\mathcal G^{\mathrm{obs}}_{1:b})\right).
$$

第一步冻结会议空间 GAT，只训练 GRU 和残差读出；只有验证集明确改善时，才允许一次
小学习率联合微调。不搜索多层 GRU、hidden size、attention head 或辅助 loss。

### 8.5 时间维度是否真正有价值的判据

Temporal-Residual GAT 必须在至少一种 imperfect CSI 条件下：

1. 相对 Pure 至少 4/5 seeds 在 delivered throughput/backlog 上联合胜利；
2. 相对 Frozen-GAT 至少 4/5 seeds 联合胜利；
3. 相对 CSI-augmented Spatial GAT 也形成一致增益，否则只能说明数据增强有效，
   不能声称时间维度有贡献；
4. 不以明显损害 latest-CSI 性能为代价；
5. 若把它解释为信道预测/恢复模块，还必须优于 naive stale 和 current large-scale
   两个非学习参考。

只有这一层也失败，才执行真正的模型止损：

- 不增加多层 STGNN、Transformer、复杂 temporal attention 或多任务损失；
- 保留 Frozen-GAT-assisted hybrid 和 Pure 消融；
- CSI 鲁棒性写成负结果、敏感性分析或 future work。

### 8.6 第一轮训练集和优化设置预先固定

为避免看到 test 结果后反复调模型，第一轮直接固定：

| 用途 | seeds | 每条轨迹 | 说明 |
|---|---|---:|---|
| train | 1000--1019 | 1000 slots | 20 条独立移动/衰落轨迹 |
| validation | 2000--2004 | 1000 slots | 只用于 checkpoint 选择和 early stopping |
| final test | 42--46 | 1000 slots | 与 2×4 配对闸门完全相同，训练期间不可见 |

每条训练轨迹按 5-slot block 形成 200 个时间块，四类观测条件平衡使用。数据按完整
seed 划分，不能把同一轨迹的不同 block 随机分到 train/test。

第一轮只用两组固定优化设置：

1. CSI-augmented Spatial GAT：会议 checkpoint 初始化，全部空间参数以 `1e-4`
   fine-tune，最多 30 epochs，validation 连续 5 次无改善即停止；
2. Temporal-Residual GAT：从最佳 CSI-augmented 空间 checkpoint 初始化并冻结空间
   编码器，只训练单层 GRUCell 和残差读出，学习率 `1e-3`，同样最多 30 epochs。

第一轮不搜索历史长度、hidden size、层数和 loss 权重。若训练数值不稳定，先记录并
检查实现/归一化，不通过 temperature、额外 penalty 或多目标 loss 修补。

---

## 9. 现有代码核对结果与最小修改范围

### 9.1 可以直接复用

`simulator/mobile_block_fading.py` 已经实现：

- persistent directed-link identity；
- 暂时不可用链路保留队列；
- 共享真实移动/block-fading 轨迹；
- 真实信道与观测 CSI 分离；
- latest/stale/large-scale/distance-quantized large-scale 接口；
- 最终动作在当前真实信道结算。

这些部分不重写。

### 9.2 当前实验脚本缺少什么

`experiments/evaluate_mobile_block_fading_csi.py` 当前：

1. 只实例化 `GATExpectedRateLocalPhysicalReuseScheduler`；
2. 只接受一个 `arrival_rate`；
3. 默认重复 2/4/8-bit 和四个速度/块长场景；
4. 用连续 departures/arrivals 的比例命名为 throughput；
5. 闸门仍判断“stale 是否严重”，没有判断 GAT 是否优于 Pure；
6. 对 GAT 和 Pure 尚无不同的消息轮数核算。

### 9.3 第一阶段允许的最小修改

只做以下改动：

1. 增加 GAT/Pure scheduler 维度；
2. 增加 `arrival_rates=2.2,4`；
3. 正式默认缩小为 5 m/s、5-slot block、4-bit；
4. 为移动永久队列增加与固定 CSI 一致的 FIFO 完整包完成记账；
5. 输出每 seed 的配对差值、联合胜利数和聚合 CI；
6. 把旧 prediction gate 改为 GAT-vs-Pure evidence gate；
7. 分开核算 GAT 与 Pure 的算法消息轮数和 payload。

第一阶段不修改 `models/local_physical_reuse.py` 的决策公式，也不训练 checkpoint；
它只建立 Frozen-GAT/Pure 的零改造结果。

若第一阶段显示 Frozen-GAT 缺少 resilience，第二阶段允许新增：

1. 独立训练/验证/test seeds 的移动 CSI 序列数据集；
2. 同架构 CSI-augmented GAT checkpoint；
3. 一个 `conference spatial GAT + single GRUCell + residual readout` 模型；
4. 对应训练、加载、测试和消融脚本。

第二阶段仍不得改变 LocalMax、条件物理复用、队列消息轮数或最终动作定义。

---

## 10. 实际执行顺序

### Phase A：接口和守恒测试

目的：先证明实验比较本身没有错。

新增或更新测试：

1. GAT/Pure 在同一 seed 下收到完全相同的 trace 和 arrivals；
2. 八个方法的真实服务 state 都来自当前 latest block；
3. stale block 0 暖启动、block 1 映射到 block 0；
4. 完整包数 + 未完成包数与总到达包数守恒；
5. 暂时不可用链路的 FIFO 包和流体队列都不丢失；
6. 闸门只在 throughput/backlog 同时改善时计为联合胜利。

### Phase B：小规模 smoke test

先用 2 个 seeds、100 slots 跑完整 2×4×2 接口：

- 检查报告层级和标签；
- 检查所有方案 arrivals 完全相同；
- 检查 physical departures 不超过到达与已有队列；
- 检查 latest 的 CSI error 为 0；
- 检查 GAT/Pure 消息轮数不同。

smoke test 只验证实现，不解释性能。

### Phase C：正式 80-run 快速闸门

运行 N=10、1000 slots、5 seeds、两个负载、四种 CSI、两个骨架。

输出至少包括：

- 完整配置和 checkpoint；
- 每 seed 原始指标；
- 每个条件的 GAT-Pure 配对差；
- 联合胜利 seed 数；
- 均值及 95% CI；
- relative-to-latest 退化；
- 方法特定的消息轮数与 payload 审计；
- 自动生成的 continue/stop 结论。

### Phase D：记录冻结模型诊断

按以下顺序记录每一组结果：

1. 动机：该比较为了验证什么；
2. 定义：两个方法和 CSI 条件分别是什么；
3. 实现：共享了哪些轨迹，只改变了什么；
4. 结果：均值、CI 和逐 seed 方向；
5. 与 fixed/latest 的区别；
6. 说明了什么、不能说明什么；
7. 可能原因；
8. 下一步：Frozen-GAT 已足够，或进入最小 CSI 适配。

### Phase E：CSI 增强重训

**执行状态：已完成，未通过单帧适配闸门。** teacher-prior E2 保持 CSI sensitivity，
但没有任何 imperfect 条件同时以4/5胜过 Frozen 和 Pure。完整记录见
`docs/CSI_AUGMENTED_SPATIAL_GAT_TUTORIAL.md`。

使用与 test seeds 42--46 完全分离的移动序列：

- 训练集、验证集、测试集按 seed 划分，不能按 block 随机打散造成轨迹泄漏；
- 四种 CSI 观测在训练中平衡采样；
- 输入使用观测图，训练物理速率使用配对的当前真实信道；
- 模型结构、层数和 hidden dimension 与会议 GAT 相同；
- 只比较 Frozen-GAT、CSI-augmented GAT 和 Pure，不调队列阶段。

### Phase F：最小时间扩展

**执行状态：已完成，未通过时间闸门。** 五次验证均未超过零残差初始化，最终
checkpoint 等于 E2 空间基线；正式40-run goodput逐项差为0。完整记录见
`docs/TEMPORAL_RESIDUAL_GAT_TUTORIAL.md`。

若单帧 CSI 增强仍不足，或需要验证时间维度的独立贡献：

1. 复用会议 GAT `encode()` 作为空间编码器；
2. 用 persistent link identity 维护每条链路的一层 GRU 状态；
3. 只预测 channel logit 残差；
4. 先冻结空间编码器训练时间模块；
5. 在相同 2×4、负载、seed 和真实轨迹上比较四种方法；
6. 按第 8.5 节判断时间维度是否保留。

### 投稿前时间预算

- 第 1 天：完成移动 runner、FIFO 指标和 Frozen-GAT/Pure smoke test；
- 第 2 天：完成 80-run 零改造基线并给出天然 resilience 诊断；
- 第 3 天：生成独立训练/验证序列，完成 CSI-augmented Spatial GAT；
- 第 4 天：完成 Temporal-Residual GAT 和统一 test；
- 第 4 天结束必须作最终模型止损，不再扩大结构搜索。

这样最多使用四天完成 CSI 主线，剩余时间保留给论文写作、绘图和必要基线。

---

## 11. 消息轮数与开销怎样单独核算

### 11.1 GAT-assisted 的逻辑轮数

当前会议 GAT 有 4 层，每层需要一轮邻居隐藏状态传播；再计一轮活动先验交换。
因此每个信道块 5 轮。

队列阶段每时隙固定 3 轮：

1. 交换骨架权重；
2. 交换骨架动作与队列；
3. 交换候选标志与局部边际收益。

5-slot block 下，摊销顺序深度为：

$$
3+\frac{5}{5}=4\text{ rounds/slot}.
$$

### 11.2 Pure 的逻辑轮数

Pure 不运行 GAT，也不交换 GAT 活动先验。它只保留每时隙的三轮队列/物理协商：

$$
3\text{ rounds/slot}.
$$

因此不能继续把 GAT 的 block-message 开销同时记在 Pure 上。

### 11.3 当前能核算和不能核算的边界

本轮分别报告：

- 每个算法的逻辑顺序轮数；
- 逐 link-conflict-edge 单播 payload 上界；
- 按物理发射 UAV 聚合广播的 payload 下界；
- GAT 相对 Pure 新增的 block-level 开销。

CSI 导频、信道估计、反馈包头、竞争、重传和半双工时序尚无协议定义，不能伪装成
一个精确的“真实空口 Mbit/s”。在同一 CSI 条件内，GAT/Pure 使用相同观测，CSI
获取成本在性能配对中相同；但论文的绝对部署开销仍需单列为未完成项。

### 11.4 最小时间模块会不会增加空口消息轮数

Temporal-Residual GAT 的 GRU 状态属于同一永久链路自己的本地历史。它不向邻居发送
额外时间消息，因此理想同步实现中：

- 空间 GAT 仍是每块 5 轮；
- 队列/物理阶段仍是每时隙 3 轮；
- 时间更新增加本地存储和计算，但不增加邻居通信轮数。

若以后改成跨链路 temporal attention，才会产生新的消息和同步开销；本轮不允许该
扩展。

---

## 12. 本轮明确不做什么

- 不训练 rho-GAT；
- 不加载 ChannelContext-v1/v2 失败路线；
- 不从零设计新的空间 GNN；
- 不搜索多层 GRU、Transformer、temporal attention 或辅助 loss；
- 不扫描 2/8-bit 距离量化；
- 不同时扫描 5/20 m/s 和 5/20-slot block；
- 不在本轮先解耦真实块长与 CSI 更新周期；
- 不修改最终 LocalMax 和一轮条件空间复用结构；
- 不把 1000-slot、5-seed 结果包装成严格稳定域定理。

允许的学习改造只有两项：同架构 CSI 增强重训，以及会议 GAT 后的一层本地时间
记忆。这样既能真正检查图学习能否对抗 CSI 抖动，也避免重新开始大范围模型搜索。

---

## 13. 计划执行后的三种终点

### 终点 A：Frozen-GAT 已有天然 resilience

若原会议 checkpoint 在至少一种 imperfect CSI 条件出现 4/5 联合胜利：

1. 记录其天然泛化能力；
2. 仍可测试最小时间模块是否带来额外、可归因的增益；
3. 若没有额外增益，不为了 STGNN 名称强行增加模块。

### 终点 B：Frozen-GAT 失败，但最小适配成功

若 CSI-augmented 或 Temporal-Residual GAT 通过对应闸门：

1. 把 Frozen-GAT 作为训练分布消融；
2. 把 CSI-augmented GAT 用来说明数据覆盖的作用；
3. 只有 Temporal 相对 CSI-augmented 仍有一致增益时，才声称时间建模贡献；
4. LocalPhysicalReuse 的队列和物理阶段保持不变。

### 终点 C：两类最小适配都失败

只有到这里才停止扩大 STGNN：

1. 如实记录 Frozen、数据增强和时间模块均未形成一致优势；
2. 保留现有 GAT-assisted hybrid 和 Pure；
3. CSI 鲁棒性作为敏感性分析或 future work；
4. 不让负结果阻塞期刊主稿。

因此第一轮 80-run 实验是诊断起点，不再被错误地当成整个时空图学习路线的最终
生死闸门。
