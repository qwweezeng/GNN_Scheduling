# CSI-Augmented Spatial GAT：单帧重训为什么仍然不够

> **文档状态：历史失败审计。** 本文不再承担当前路线规划；学习目标与接口的最新
> 根因分析已集中到
> [`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`](MOBILE_BLOCK_FADING_CSI_TUTORIAL.md)。

> 日期：2026-07-15  
> 当前阶段：移动与不完美 CSI 主线的 Phase E  
> 本文目的：判断“只增加移动/扰动 CSI 训练数据，但不增加时间维度”能否让会议版
> GAT 更好地应对 CSI 不准确。
>
> **后续状态：** 本文计划的最小 Temporal-Residual GAT 已完成，但独立验证选择
> 零残差起点，未形成新增时间收益。完整记录见
> `docs/TEMPORAL_RESIDUAL_GAT_TUTORIAL.md`。

---

## 0. 先用一句话说明这一步在做什么

上一阶段已经发现：原会议 GAT 在高压力负载下，对 stale、large-scale 和 4-bit
large-scale CSI 有一定天然鲁棒性，但优势并不覆盖低负载，也仍然存在明显的
relative-to-latest 性能下降。

因此本阶段先做最小改造：

> 不改变会议版 GAT 的层数、隐藏维数和空间消息传递结构，只让它在移动和四种 CSI
> 观测上重新训练，检查“训练数据覆盖”本身是否足以提高鲁棒性。

如果这一版本成功，就说明主要问题是训练分布不匹配，不一定需要时间模型；如果失败，
才有充分理由加入每条永久链路的历史状态。

本阶段最终得到的结论是：

> 合理的单帧重训没有发生表示坍缩，也更准确地重建了会议 GAT 在真实当前 CSI 下的
> channel prior；但它在真实队列仿真中几乎复现 Frozen-GAT，没有任何 imperfect CSI
> 条件同时稳定胜过 Frozen-GAT 和 Pure。因此，只增加单帧训练数据不足以补回观测中
> 已经丢失的瞬时信道信息，下一步进入最小时间残差模型是有依据的。

这里还保留了一个失败的训练目标。它非常重要，因为它暴露了“GAT 输出在系统中的角色”
如果定义错误，即使物理速率损失看起来合理，也会把模型训练成没有输入辨识能力的
“全链路激活先验”。

---

## 1. 本阶段在整条研究路线中的位置

目前的在线调度器仍然是 `LocalPhysicalReuse-v2`，由两部分组成：

1. **信道阶段**：会议版 GAT 每个信道块读取信道图，为每条链路产生一个连续的
   channel prior；
2. **队列阶段**：每个时隙用当前队列形成多个局部胜者，再做一轮具有 SINR 和 MWS
   物理意义的条件空间复用。

本阶段只训练第 1 部分，不修改第 2 部分。这样最终性能变化才能归因于 channel prior，
而不是队列规则或复用规则也跟着变化。

四个连续阶段的逻辑关系是：

1. `Frozen-GAT vs Pure`：原会议 checkpoint 有没有天然 CSI resilience？
2. `CSI-Augmented Spatial GAT`：只改变训练分布是否已经足够？
3. `Temporal-Residual GAT`：在同一空间骨架上增加本地历史是否提供独立增益？
4. 统一配对测试：时间模型是否同时胜过 Frozen、单帧增强和 Pure？

本文件记录第 2 步。

---

## 2. 先解释本阶段所有新概念

### 2.1 什么是 Frozen-GAT

`Frozen-GAT` 是会议论文已经训练好的 channel-only GAT checkpoint。它训练完成后参数
不再改变，直接放到移动、不完美 CSI 环境中测试。

它回答的问题是：原模型有没有天然泛化能力？

### 2.2 什么是 CSI-Augmented Spatial GAT

`CSI-Augmented Spatial GAT` 与 Frozen-GAT 使用完全相同的网络结构：

- 相同 GAT 层数；
- 相同 hidden dimension；
- 相同节点和冲突边定义；
- 相同输出 channel logit；
- 相同 LocalPhysicalReuse-v2 队列阶段。

唯一变化是训练样本不再只有原来的静态、perfect CSI，而是包含移动轨迹以及四种观测：

- latest CSI；
- one-block stale CSI；
- current large-scale CSI；
- 4-bit distance-quantized large-scale CSI。

它仍然只看当前一个观测图，不保存历史。因此它是“单帧空间模型”，不是时间模型。

### 2.3 什么是单帧

“帧”在这里指一个信道块对应的观测图。单帧模型计算第 $b$ 个块的 prior 时，只能使用
第 $b$ 个块交给它的 CSI：

$$
z_b=f_{\mathrm{GAT}}(\widehat{G}_b).
$$

它不知道 $\widehat{G}_{b-1}$、$\widehat{G}_{b-2}$ 等历史。若当前观测删除了小尺度
衰落，或者只提供量化距离，单帧模型不能凭空知道被删除的真实瞬时值。

### 2.4 什么是 channel logit 和 channel prior

GAT 为每条链路输出一个实数 $z_i$，称为 **channel logit**。经过 sigmoid：

$$
p_i=\sigma(z_i),
$$

得到 $0$ 到 $1$ 之间的 **channel prior**。

这里的 $p_i$ 不是最终的“发送/不发送”动作。它只是队列阶段使用的连续信道偏好。最终
动作还要结合当前队列、局部胜者骨架和条件复用产生。

这个角色区别是本阶段第一个失败实验的核心。

### 2.5 什么是 teacher-prior reconstruction

在训练样本的同一个真实信道块中：

- 冻结的会议 GAT 看到 current true CSI，输出目标 logit $z_i^{\mathrm{teacher}}$；
- 待训练 GAT 只看到指定的不完美观测，输出 $z_i^{\mathrm{student}}$。

训练目标是：

$$
\mathcal{L}_{\mathrm{teacher}}
=\frac{1}{|\mathcal{L}_b|}
\sum_i\left(z_i^{\mathrm{student}}-z_i^{\mathrm{teacher}}\right)^2.
$$

通俗地说，teacher 告诉 student：如果能看到当前真实 CSI，会议 GAT 会给出怎样的
信道偏好；student 尝试仅凭不完美观测重建这个偏好。

这不是增加一个随意的辅助 loss。它是把训练目标与部署接口对齐：训练的对象就是队列
阶段实际接收的 channel prior，而不是把 prior 错当成最终调度动作。

### 2.6 什么是表示坍缩

如果不同链路、不同 CSI 输入经过模型后几乎都得到同一个输出，模型就失去了区分输入的
能力。本文用每个状态内不同链路 prior 的标准差检查这一现象。

若所有 $p_i=1$，标准差为 0，含义不是“所有信道都很好”，而是模型不再使用 CSI，
只给队列阶段一个全激活常数。

### 2.7 什么是联合胜利

对同一个 seed，候选模型只有同时满足下面两项，才记作一次联合胜利：

1. 完整包 delivered throughput 更高；
2. 平均 backlog 更低。

正式闸门要求在同一 imperfect CSI 条件下：

- 相对 Frozen-GAT 至少 4/5 seeds 联合胜利；
- 相对 Pure 也至少 4/5 seeds 联合胜利；
- 候选输出仍对 CSI 敏感。

只提高 physical service、只胜过 Pure、或者输出已经坍缩，都不能称为单帧训练成功。

---

## 3. 统一实验环境

为了使变化可归因，本阶段沿用 Frozen-GAT/Pure 正式基线：

| 项目 | 设置 |
|---|---:|
| UAV 数量 | 10 |
| 每条仿真长度 | 1000 slots |
| slot 时长 | 1 ms |
| 信道块长度 | 5 slots |
| UAV 速度 | 5 m/s |
| 测试负载 | $\Lambda=2.2,4$ |
| 测试 seeds | 42--46 |
| 缓存 | 无限缓存、无 deadline |
| 真实服务结算 | current true instantaneous CSI |

训练、验证、测试按整条轨迹的 seed 隔离：

| 数据用途 | seeds |
|---|---|
| 训练 | 1000--1019 |
| 验证 | 2000--2004 |
| 正式测试 | 42--46 |

不能把同一条移动轨迹中的不同 block 随机拆到训练集和测试集，否则相邻位置与相关衰落会
造成轨迹泄漏。训练中每个 epoch 访问每个真实 block 一次，并以确定性的轮换方式平衡四种
CSI 观测。

所有候选方案与 Frozen/Pure 共享相同的：

- 真实 UAV 移动轨迹；
- 真实小尺度衰落轨迹；
- 到达包轨迹；
- CSI 观测条件；
- LocalPhysicalReuse-v2 队列和条件复用规则。

唯一改变是 channel GAT checkpoint。

---

## 4. 实验 E1：直接最大化真实 soft sum-rate

### 4.1 动机：为什么先测试这个目标

一个直观想法是：student 看不完美 CSI，调度效果却用 current true CSI 结算；如果直接
最大化真实物理速率，模型或许能学到在观测误差下更保守、更鲁棒的 channel prior。

因此第一版训练使用：

$$
\mathcal{L}_{\mathrm{physical}}
=-\sum_i R_i(\mathbf{p};G_b^{\mathrm{true}}),
$$

其中 $\mathbf p$ 是 GAT 输出的连续概率，$G_b^{\mathrm{true}}$ 是配对的当前真实信道。

### 4.2 具体怎样训练

- 从会议 checkpoint 初始化；
- 不改变模型结构；
- 学习率 $10^{-4}$；
- 每个 epoch 使用 4000 个独立 block 样本；
- 四种 CSI 观测平衡；
- validation 连续 5 次不改善则提前停止。

### 4.3 实验现象

验证 soft sum-rate 从 3.4564 增至 3.4753，表面上看有改善；但最佳模型的四种 CSI 验证
值全部精确相同：

| CSI | 最终验证 soft sum-rate |
|---|---:|
| latest | 3.475316 |
| stale | 3.475316 |
| large-scale | 3.475316 |
| 4-bit large-scale | 3.475316 |

进一步检查输出发现：

- mean prior = 1.0000；
- 每个状态内链路 prior 标准差 = 0；
- 不同 CSI 输入不再改变输出。

也就是说，模型退化成“所有链路 prior 都为 1”。

### 4.4 为什么它在队列指标上偶尔看起来还不错

这个 checkpoint 在 $\Lambda=4$、large-scale 条件下，数值上相对 Frozen 和 Pure 都达到
4/5 以上联合胜利。但这不能算作学习到 CSI 鲁棒性。

原因是后续 LocalPhysicalReuse-v2 仍然会使用队列和显式物理规则做最终决策。即使 GAT
退化成一个常数，后续规则仍可能在某些轨迹上工作。因此最终队列指标好，不等于前面的
GAT 学会了对抗 CSI 变化。

### 4.5 和 Frozen-GAT 的根本区别

Frozen-GAT 的平均 prior 约为 0.9177，状态内链路标准差约为 0.00364；虽然变化幅度不大，
但仍会根据链路和 CSI 改变输出。E1 模型则完全不区分输入。

### 4.6 这个失败说明了什么

它说明训练目标和部署接口错位：

- loss 把 $p_i$ 当成最终并发调度概率；
- 实际系统把 $p_i$ 当成后续队列/物理模块的连续先验。

在 soft-rate 公式里，提高所有 $p_i$ 能直接提高期望发送项，优化器因此把输出推向 1；
但部署时最终动作并不是这些 $p_i$。所以这个目标没有训练我们真正需要的接口。

### 4.7 下一步为什么不是调 temperature 或加惩罚项

这里的问题不是缺一个防坍缩超参数，而是目标对象定义错了。继续加 bias、temperature、
熵惩罚或稀疏惩罚只会掩盖接口错误。因此下一实验直接训练正确的 channel-logit 接口。

---

## 5. 实验 E2：重建 true-CSI teacher 的 channel prior

### 5.1 动机：这一实验要验证什么

E2 只回答一个干净的问题：

> 在不增加历史的前提下，移动和不完美 CSI 数据增强能否让同一个空间 GAT，从受损观测
> 中恢复出会议 GAT 在 current true CSI 下本来会产生的 channel prior？

如果能恢复，并进一步稳定改善真实队列指标，说明单帧训练已经足够；如果只能近似模仿
Frozen-GAT，却不能在下游形成增益，就说明当前单帧观测的信息上限已经成为主要限制。

### 5.2 具体方法

对每个训练 block：

1. Frozen teacher 读取 current true CSI；
2. student 读取 latest、stale、large-scale 或 4-bit large-scale 之一；
3. 按永久 link identity 对齐同一条链路；
4. 最小化 student logit 与 teacher logit 的均方误差；
5. 测试时把 student 放回完全相同的 LocalPhysicalReuse-v2 中。

teacher 不更新，student 从会议 checkpoint 初始化。网络结构、hidden size 和消息轮数不变。

验证报告中的数值记为负 MSE，越接近 0 越好。

### 5.3 训练结果

| CSI | 训练前负 MSE | 最佳 checkpoint 负 MSE | 变化 |
|---|---:|---:|---:|
| latest | 0.00000 | -0.00286 | 略有退化 |
| stale | -0.03782 | -0.03417 | 改善 |
| large-scale | -0.03451 | -0.03009 | 改善 |
| 4-bit large-scale | -0.08540 | -0.06863 | 明显改善 |
| 四条件平均 | -0.03943 | -0.03394 | 改善 |

最佳 checkpoint 出现在 epoch 4，之后 5 次验证未改善，于 epoch 9 后提前停止。

这个现象符合多条件训练的正常权衡：为了从受损观测恢复 teacher 输出，student 牺牲了
一点 latest 的完全复现能力，但降低了三种 imperfect CSI 的 reconstruction error。

### 5.4 输出有没有坍缩

没有。

| 输出诊断 | Frozen-GAT | E2 student |
|---|---:|---:|
| mean prior | 0.917734 | 0.918484 |
| 平均状态内 prior 标准差 | 0.003640 | 0.001566 |
| 最小状态内 prior 标准差 | 0.002401 | 0.000968 |

student 的输出比 Frozen 更平滑，但标准差非零，并且不同 CSI 状态会产生不同输出。因此
E2 是有效的 CSI-sensitive 空间基线，不能与 E1 的全 1 坍缩混为一谈。

### 5.5 正式队列结果：相对 Frozen-GAT

表中“联合胜利数”指 5 个配对 seeds 中，同时提高 delivered throughput 并降低 backlog
的次数；$\Delta T>0$ 和 $\Delta Q>0$ 才表示 E2 更好。

| 负载 | CSI | 联合胜利 | $\Delta T$ (Mbps) | $\Delta Q$ (Mbit/link) |
|---:|---|---:|---:|---:|
| 2.2 | latest | 1/5 | -0.0096 | -0.000006 |
| 2.2 | stale | 1/5 | -0.0336 | -0.000017 |
| 2.2 | large-scale | 1/5 | -0.0024 | -0.000147 |
| 2.2 | 4-bit large-scale | 2/5 | +0.0288 | +0.000698 |
| 4 | latest | 1/5 | +0.0072 | +0.000179 |
| 4 | stale | 0/5 | -0.0240 | -0.000045 |
| 4 | large-scale | 3/5 | +0.0456 | +0.000799 |
| 4 | 4-bit large-scale | 3/5 | -0.0096 | +0.000325 |

没有一个 imperfect CSI 条件达到相对 Frozen-GAT 的 4/5 联合胜利。

### 5.6 正式队列结果：相对 Pure

| 负载 | CSI | 联合胜利 | $\Delta T$ (Mbps) | $\Delta Q$ (Mbit/link) |
|---:|---|---:|---:|---:|
| 2.2 | latest | 0/5 | -0.2016 | -0.003371 |
| 2.2 | stale | 2/5 | -0.2280 | -0.003842 |
| 2.2 | large-scale | 2/5 | +0.3936 | +0.001138 |
| 2.2 | 4-bit large-scale | 3/5 | +0.6096 | +0.003550 |
| 4 | latest | 3/5 | +0.3696 | +0.004432 |
| 4 | stale | 5/5 | +1.2456 | +0.018937 |
| 4 | large-scale | 4/5 | +2.1672 | +0.024612 |
| 4 | 4-bit large-scale | 5/5 | +2.5656 | +0.033915 |

E2 在高压力 imperfect CSI 下仍明显优于 Pure。这不是 E2 新创造出的优势，而是基本保留
了上一阶段 Frozen-GAT 已经存在的天然鲁棒性。因为它没有稳定胜过 Frozen-GAT，所以不能
把这些数值重新归因于 CSI-augmented training。

### 5.7 和上一版相比发生了什么

与 Frozen-GAT 相比，E2 的下游变化通常只有百分之几 Mbps 或 $10^{-4}$ Mbit/link 量级，
且 seed 方向不一致。最接近成功的是 $\Lambda=4$ 下的 large-scale：

- 相对 Frozen 为 3/5 联合胜利，而不是要求的 4/5；
- 平均 goodput 只增加 0.0456 Mbps；
- 平均 backlog 只改善 0.000799 Mbit/link。

因此它更像一个平滑后的 Frozen prior，而不是形成了新的鲁棒调度机制。

### 5.8 说明了什么

本实验支持三点：

1. 使用正确接口后，同架构移动 CSI 重训不会必然坍缩；
2. 单帧数据增强能降低 imperfect observation 到 true-CSI teacher prior 的误差；
3. 这种重建改善没有转化为稳定的完整包 goodput/backlog 增益。

### 5.9 不能说明什么

本实验不能说明：

- GAT 没有应对信道变化的能力；Frozen-GAT 在高压力下已经优于 Pure；
- 时间模型一定成功；目前只证明测试时间建模是合理的下一步；
- large-scale CSI 能恢复瞬时小尺度衰落；单帧输入本身没有这些信息；
- 已经得到稳定域定理；1000-slot、5-seed 只是配对经验结果。

### 5.10 可能归因

最主要的原因不是模型容量太小，而是信息缺失：

- stale CSI 提供过去一个块的瞬时信道，却没有当前创新项；
- large-scale CSI 删除了当前小尺度衰落；
- 4-bit large-scale 又增加距离量化误差；
- 单帧 GAT 无论怎样重训，都只能从当前受损观测建立静态映射。

另外，LocalPhysicalReuse-v2 后续的显式队列和物理规则本身很强，channel prior 的小幅 MSE
改善未必足以改变离散局部胜者和复用集合。这也解释了为何训练指标改善、队列指标却几乎
不动。

---

## 6. Phase E 的正式判断

CSI-Augmented Spatial GAT 的通过条件是：至少一种 imperfect CSI 同时满足

- 相对 Frozen-GAT 至少 4/5 联合胜利；
- 相对 Pure 至少 4/5 联合胜利；
- mean delivered throughput 与 mean backlog 同时改善；
- 输出保持 CSI sensitivity。

E2 保持了 CSI sensitivity，但没有任何条件通过前两项共同闸门。

因此正式结论是：

> 同架构、单帧 CSI 数据增强没有形成可归因且一致的下游增益。它作为“只有训练分布变化、
> 没有时间状态”的必要空间基线保留；研究继续进入最小 Temporal-Residual GAT。

E1 的全激活模型只作为失败目标审计保留，不作为时间模型的有效对照，也不作为初始化。

---

## 7. 下一步：最小 Temporal-Residual GAT 到底增加什么

下一模型不重新设计空间图网络，只在 E2 的空间骨架后加入每条永久链路自己的本地时间
状态：

$$
h_{i,b}^{\mathrm{time}}
=\mathrm{GRU}(h_{i,b}^{\mathrm{space}},h_{i,b-1}^{\mathrm{time}}),
$$

$$
z_{i,b}^{\mathrm{temporal}}
=z_{i,b}^{\mathrm{spatial}}+\Delta z_{i,b}^{\mathrm{time}}.
$$

这里：

- $h_{i,b}^{\mathrm{space}}$ 仍由会议 GAT 的空间消息传递产生；
- $h_{i,b-1}^{\mathrm{time}}$ 是同一永久链路上一个 block 的本地记忆；
- GRU 只输出一个 logit 残差，不替换原空间 prior；
- 残差头从 0 初始化，因此训练开始时严格等于 E2 空间基线；
- 时间状态不在链路之间交换，不增加空口消息轮数。

它要验证的不是“GRU 是否更复杂”，而是：

> 当当前 CSI 不完整时，同一链路的历史能否提供单帧训练无法获得的信息，并最终改变
> LocalPhysicalReuse-v2 的完整包 goodput 和 backlog。

时间模型必须在相同测试轨迹上同时比较 Frozen-GAT、Pure 和 E2 空间模型。只有它相对 E2
仍形成一致增益，才可以把改善归因于时间维度。

---

## 8. 文件与可复现入口

### E1：失败的 physical soft-sum-rate 目标

- 训练脚本：`train/train_csi_augmented_spatial_gat.py --objective physical_soft_sum_rate`
- checkpoint：`results/models/gat_csi_augmented_spatial_v1_numUAV=10_tx=10dBm.pth`
- 训练报告：`results/diagnostics/csi_augmented_spatial_train_N10_20260715.json`
- 正式测试：`results/diagnostics/csi_augmented_spatial_eval_N10_20260715.json`

### E2：有效但未通过队列闸门的 teacher-prior 目标

- 训练脚本：`train/train_csi_augmented_spatial_gat.py --objective teacher_logit_mse`
- checkpoint：`results/models/gat_csi_augmented_spatial_teacher_v2_numUAV=10_tx=10dBm.pth`
- 训练报告：`results/diagnostics/csi_augmented_spatial_teacher_train_N10_20260715.json`
- 正式测试：`results/diagnostics/csi_augmented_spatial_teacher_eval_N10_20260715.json`

测试脚本统一为：`experiments/evaluate_csi_augmented_spatial_gat.py`。

---

## 9. 给论文写作保留的严格表述

当前可以写：

> CSI-augmented single-frame training reduced reconstruction error under stale,
> large-scale, and quantized observations, while preserving CSI-sensitive outputs.
> However, it did not yield consistent paired gains over the frozen spatial GAT in
> packet goodput and backlog, motivating an explicit test of link-wise temporal state.

当前不能写：

- “数据增强提高了稳定域”；
- “时间模型已经被证明必要”；
- “E1 的 all-active prior 学会了鲁棒调度”；
- “只要训练更多数据，GAT 就能恢复真实瞬时 CSI”。

下一阶段只有在 Temporal-Residual GAT 相对 E2 空间基线也通过配对闸门后，才能声称时间
建模带来了独立贡献。
