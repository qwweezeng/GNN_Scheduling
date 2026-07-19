# Temporal-Residual GAT：最小时间维度实验与止损结论

> **文档状态：历史失败审计。** 本文只证明“模仿旧 Frozen-GAT logit 的 GRU”没有
> 新增价值，不证明所有 STGNN 无效。当前统一解释与候选方向见
> [`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`](MOBILE_BLOCK_FADING_CSI_TUTORIAL.md)。

> 日期：2026-07-15  
> 当前阶段：移动与不完美 CSI 主线的 Phase F  
> 本文目的：在会议版空间 GAT 骨架上只增加一层永久链路本地记忆，严格检查时间历史
> 能否带来单帧 CSI 增强无法提供的独立增益。
>
> **2026-07-15 后续审计纠正：** channel-only 检查发现 Frozen-GAT 在正式抽样的
> 1200个状态中阈值动作全部为全链路激活，teacher prior 只有约0.003--0.004的链路间
> 标准差。因此本文负结果只否定“重建 Frozen teacher logit 的时间残差”，不能扩展成
> “所有物理信道预测或STGNN均无价值”。Rayleigh/Rician/LoS和骨架--协商分解见
> `docs/CHANNEL_MODEL_AND_STAGE_MASKING_TUTORIAL.md`。

---

## 0. 如果只读一页，先读这里

上一阶段的 `CSI-Augmented Spatial GAT` 已经证明：

- 同架构移动/扰动数据重训可以降低 imperfect CSI 到 true-CSI teacher prior 的误差；
- 合理目标不会产生全链路激活坍缩；
- 但真实队列 goodput/backlog 几乎复现 Frozen-GAT，没有形成新的 4/5 配对优势。

因此，本阶段按照预定计划加入最小时间维度：

1. 空间编码器仍是会议 GAT；
2. 使用上一阶段非坍缩 E2 checkpoint 作为单帧空间起点；
3. 每条有向链路按 `(发送 UAV, 接收 UAV)` 保存一份 48 维 GRU 状态；
4. GRU 只预测 channel logit 的残差；
5. 空间 GAT 完全冻结，队列和物理复用阶段完全不改；
6. 训练目标仍是恢复 Frozen teacher 在 current true CSI 下的 channel logit。

正式结果是一个清楚的负结果：

> 5 个验证 epoch 都没有超过“时间残差恒为 0”的初始化。早停因此选择 epoch = -1，
> 即保留训练前的单帧空间 checkpoint。统一 40-run 队列测试与 E2 的 40 个 goodput
> 数值逐项完全相同，最大绝对差为 0；时间历史对 logit 的贡献也为 0。因此最小时间模型
> 没有通过继续闸门。

这不等于“所有 STGNN 永远无效”，也不等于“GAT 对 CSI 变化完全没有韧性”。它准确说明：

> 在当前 5 m/s、5 ms block、当前 CSI 生成过程和 1000-slot/5-seed 证据下，冻结会议
> GAT 已在高压力 imperfect CSI 中保留条件性优势，但同架构单帧重训和一层本地 GRU
> 都没有带来可归因的额外队列收益。按投稿前止损规则，不继续搜索更深 RNN、temporal
> attention 或 Transformer。

---

## 1. 为什么这个实验现在才出现

如果一开始直接加入 GRU，最终性能改善可能来自两种不同原因：

1. 模型看到了更合适的移动/扰动训练数据；
2. 模型真正利用了同一链路的历史。

上一阶段先训练完全相同的单帧空间结构，就是为了建立第一种原因的对照。只有时间模型
继续胜过这个对照，才能把增益归因于时间维度。

因此当前比较链是：

```text
Pure
  ↑ 学习式信道先验是否有价值
Frozen-GAT
  ↑ 移动/扰动训练数据是否有价值
CSI-Augmented Spatial GAT (E2)
  ↑ 永久链路时间历史是否有独立价值
Temporal-Residual GAT
```

本阶段不重新打开空间 GAT 搜索，也不改 LocalPhysicalReuse-v2 的队列阶段。

---

## 2. 使用前先解释全部新概念

### 2.1 什么是时间状态

单帧 GAT 在第 $b$ 个信道块只读取当前交给它的观测图 $\widehat G_b$。时间模型额外保留
同一条链路过去处理观测后形成的内部向量：

$$
s_{i,b-1}\in\mathbb R^{48}.
$$

这个向量叫作链路 $i$ 的 **时间状态**。它不是队列长度，也不是额外 CSI 反馈，而是模型
对该链路过去信道 embedding 的压缩记忆。

### 2.2 什么是 GRU

GRU 是一种带“保留多少旧信息、写入多少新信息”门控的循环神经网络单元。本实验使用：

$$
s_{i,b}=\operatorname{GRU}(h_{i,b}^{\mathrm{space}},s_{i,b-1}),
$$

其中 $h_{i,b}^{\mathrm{space}}$ 是会议 GAT 在当前 block 为链路 $i$ 产生的 48 维空间
embedding。

本实验只有一层 48 维 GRU，没有堆叠、多头时间注意力或 Transformer。

### 2.3 为什么叫 Temporal-Residual

模型不让 GRU 从零重写 channel prior，而是保留空间模型的 logit，再加一个小修正：

$$
z_{i,b}^{\mathrm{temporal}}
=z_{i,b}^{\mathrm{spatial}}+\Delta z_{i,b}^{\mathrm{time}}.
$$

$\Delta z$ 称为 **残差**。这种设计的物理含义是：会议空间 GAT 仍负责当前干扰图解释，
时间模块只在历史确实有用时修正它。

### 2.4 什么是零初始化残差

残差输出层的权重和 bias 全部初始化为 0。因此开始训练前：

$$
\Delta z_{i,b}^{\mathrm{time}}=0,
$$

时间模型与 E2 空间模型严格相同。若验证集认为任何时间修正都更差，早停可以安全回退到
原空间模型，而不是保存一个被训练破坏的版本。

### 2.5 为什么 permanent link identity 仍然重要

图中数组第 7 行不保证每个 block 都是同一条链路。若直接按行号传 GRU 状态，UAV 2→5
的历史可能错误交给 UAV 1→4。

本实验用有向 UAV 对 `(tx, rx)` 作为 key：

```text
(2, 5) 的上一块状态 → 只能更新 (2, 5) 的当前状态
```

即使某条链路在图中的行位置变化，状态仍按物理身份对齐。若链路暂时不可用，状态保留但
不更新；重新可用时仍取回同一 key。

### 2.6 什么是截断式反向传播

一条训练轨迹有 200 个信道块。若把 200 个 GRU 步骤一次性反向传播，内存和数值稳定性
都会变差。因此每连续 10 个 block 计算一次梯度，然后切断更早的梯度链。

关键区别是：

- 隐藏状态数值继续传给下一个 10-block 窗口；
- 只有梯度不再回到更早窗口；
- 在线推理不会每 10 块清空状态。

### 2.7 什么是 early stopping 的 epoch = -1

训练前的零残差模型也参加验证，编号记作 `-1`。如果所有训练 epoch 都比它差，最佳
checkpoint 就是 `best_epoch=-1`。

这表示验证数据明确选择“不做时间修正”，不是缺少 checkpoint，也不是脚本失败。

### 2.8 什么是 history contribution

一个 GRU 即使不使用旧状态，也可以对当前 embedding 做一次非线性变换。为了避免把这种
静态变换误称为时间贡献，本实验比较：

- **stateful logit**：连续保留同一链路历史；
- **memoryless logit**：每个 block 都把旧状态清零。

两者 logit 的平均绝对差称为 history contribution。只有它非零，才能说明旧状态真的
进入了当前 prior。

---

## 3. 模型定义与原版本的区别

### 3.1 保持不变的部分

- 会议版四层空间 GAT；
- E2 单帧空间 checkpoint；
- 当前观测图和冲突边定义；
- channel prior 到 expected rate 的转换；
- queue-aware LocalMax 局部胜者骨架；
- 一轮条件物理空间复用；
- current true CSI 服务结算；
- FIFO 完整包统计。

### 3.2 唯一增加的部分

- 一层 `GRUCell(48, 48)`；
- 一个 `Linear(48, 1)` logit 残差头；
- 每条永久链路一份 48 维 float32 本地状态。

新增可训练参数共 14,161 个；冻结空间 GAT 有 22,401 个参数。

### 3.3 与“重新设计 STGNN”的区别

本实验没有：

- 改空间 GAT 层数或注意力结构；
- 做跨链路 temporal attention；
- 增加多步信道预测头；
- 增加辅助 loss、稀疏惩罚或 temperature；
- 联合修改队列调度器。

因此它是对“时间维度是否值得继续”的最小检验，而不是从零开始的新模型。

---

## 4. 集中训练与逻辑分布式执行

### 4.1 训练为什么是集中式

训练时需要 current true CSI teacher 输出作为监督目标，并统一反向传播 GRU 参数。这些
信息可以在离线仿真器或集中训练服务器中获得，不要求在线 UAV 知道真实当前信道。

### 4.2 在线为什么仍是逻辑分布式

在线第 $b$ 个信道块中：

1. 链路按会议 GAT 与冲突邻居交换空间隐藏状态；
2. 每条链路只用自己的空间 embedding 和自己的旧 GRU 状态做本地更新；
3. 只交换原有 channel prior；
4. 队列阶段继续执行固定三轮局部协商。

GRU 状态不发给邻居，所以没有新的空口消息轮次。

---

## 5. 训练与测试怎样保证公平

### 5.1 数据划分

| 用途 | seeds |
|---|---|
| 训练 | 1000--1019 |
| 验证 | 2000--2004 |
| 正式测试 | 42--46 |

按整条移动轨迹划分，不能把相邻 block 分散到 train/test。

### 5.2 环境

| 项目 | 设置 |
|---|---:|
| UAV | 10 |
| 轨迹 | 1000 slots |
| 信道块 | 5 slots = 5 ms |
| 速度 | 5 m/s |
| CSI | latest / stale / large-scale / 4-bit large-scale |
| 负载 | $\Lambda=2.2,4$ |
| 缓存 | 无限、无 deadline |
| 服务结算 | current true instantaneous CSI |

### 5.3 训练目标

Frozen teacher 看到 current true CSI，时间 student 看到指定观测序列：

$$
\mathcal L
=\frac{1}{|\mathcal L_b|}\sum_i
\left(z_{i,b}^{\mathrm{temporal}}-z_{i,b}^{\mathrm{teacher,true}}\right)^2.
$$

四种 CSI 在每个 epoch 中各 1000 个 block，完全平衡。

### 5.4 为什么预计算 frozen feature cache

空间 GAT 和 teacher 都冻结，重复计算相同 block 的 embedding 不会提供新梯度。因此正式
训练先一次性缓存：

- E2 空间 embedding；
- E2 空间 base logit；
- true-CSI Frozen teacher target；
- 对齐后的永久 link key。

随后每个 epoch 只训练 GRU。这只是等价计算加速，不改变任何输入或目标。

---

## 6. 实验 T0：学习率稳定性审计

### 6.1 动机

零残差初始化的意义是保护空间基线。因此第一件事不是看训练 loss 是否下降，而是检查
一次 epoch 会不会用过大的步长直接破坏这个起点。

### 6.2 方法

最初按计划尝试 Adam、学习率 $3\times10^{-4}$，每 10 个 block 更新一次参数；一个
epoch 共 400 次更新。

### 6.3 现象

| 指标 | 训练前 | epoch 0 后 |
|---|---:|---:|
| 验证平均负 MSE（越大越好） | -0.03394 | -0.11080 |
| 平均绝对 logit 残差 | 0 | 0.27876 |

一步就使验证误差扩大三倍以上，说明步长与每 epoch 的更新次数不匹配。

### 6.4 决定

该运行被主动中止，没有保存候选 checkpoint，也不作为模型性能结果。正式训练保持同一
结构、数据和 loss，只将学习率降低为 $10^{-5}$。

这是优化稳定性修正，不是对多个模型/损失做性能搜索。审计原始记录保存在
`results/diagnostics/temporal_residual_learning_rate_audit_20260715.json`。

---

## 7. 实验 T1：正式 Temporal-Residual 训练

### 7.1 动机

检查同一链路的历史能否在验证轨迹上稳定降低 true-CSI teacher logit MSE，同时不破坏
latest CSI 和其他观测条件。

### 7.2 具体方法

- 空间 E2 checkpoint 冻结；
- 仅训练 14,161 个时间参数；
- Adam，学习率 $10^{-5}$；
- 10-block 截断梯度；
- 四种 CSI 每 epoch 各 1000 blocks；
- 连续 5 个 epoch 未超过历史最佳则早停；
- 训练前零残差模型也作为候选。

### 7.3 结果

训练前四条件平均负 MSE 为 -0.0339408。五个 epoch 的结果为：

| epoch | 四条件平均负 MSE | 平均绝对残差 | 是否胜过训练前 |
|---:|---:|---:|---|
| -1（训练前） | **-0.0339408** | 0 | **是，最终最佳** |
| 0 | -0.0356799 | 0.04376 | 否 |
| 1 | -0.0350855 | 0.03587 | 否 |
| 2 | -0.0354689 | 0.04107 | 否 |
| 3 | -0.0340124 | 0.00640 | 否 |
| 4 | -0.0344888 | 0.02550 | 否 |

epoch 3 最接近起点。它在 large-scale 上从 -0.0300939 轻微改善到 -0.0299939，但：

- latest 变为 -0.0029661，差于 -0.0028647；
- stale 变为 -0.0342902，差于 -0.0341697；
- 4-bit 变为 -0.0687994，差于 -0.0686349；
- 四条件平均仍低于训练前。

因此 early stopping 保存 `best_epoch=-1`，时间残差保持为 0。

### 7.4 与 E2 空间模型的区别

最终 checkpoint 在结构上包含 GRU，但其残差头仍是零初始化状态。于是：

$$
z^{\mathrm{temporal}}=z^{\mathrm{E2\ spatial}}
$$

对所有测试输入严格成立。它不是一个“略有提升或略有下降”的模型，而是验证选择后与 E2
完全相同的安全回退结果。

### 7.5 说明了什么

在独立验证轨迹上，当前最小本地历史残差没有形成可泛化的 teacher-prior 恢复能力。训练
loss 中出现的时间修正不能同时改善四种观测，因此不应强行保存最后 epoch。

### 7.6 可能原因

1. **stale CSI 的历史方向有限**：当前输入已经是一块以前的瞬时 CSI，GRU 保存的更早
   历史主要帮助外推；Gauss--Markov 创新项仍不可观测。
2. **large-scale 输入从未包含小尺度衰落**：连续 large-scale 历史可以跟踪位置变化，
   但不能重建每块随机小尺度创新。
3. **4-bit 又丢失距离精度**：时间平滑可能减少量化抖动，却无法恢复被量化和小尺度删除
   的全部信息。
4. **E2 已经很接近 Frozen prior**：剩余 logit MSE 小，且 LocalPhysicalReuse-v2 后续
   规则较强，小改动不容易改变最终离散动作。
5. **统一模型面对四种不同误差机制**：本轮故意不增加 CSI-mode 条件头，避免在投稿前
   展开多模型搜索；结果说明一个无条件最小 GRU 没有普适改善。

第五点是未来可能研究的方向，但不是继续打补丁的理由。

---

## 8. 实验 T2：统一 40-run 队列验证

### 8.1 动机

训练目标是 channel logit MSE，而论文真正关心完整包 goodput 和 backlog。因此即使最佳
checkpoint 回退到零残差，仍需通过完整模拟核对它与 E2 是否逐项等价。

### 8.2 方法

测试 5 seeds × 2 loads × 4 CSI = 40 个 Temporal runs。每个 run 与 Frozen、Pure、E2
共享真实移动、衰落和到达轨迹，所有服务都由 current true CSI 结算。

### 8.3 结果

Temporal 相对 E2 的所有 8 个条件均为：

| 条件范围 | 联合胜利 | mean $\Delta$ goodput | mean backlog improvement |
|---|---:|---:|---:|
| $\Lambda=2.2$ 的四种 CSI | 全部 0/5 | 全部 0 | 全部 0 |
| $\Lambda=4$ 的四种 CSI | 全部 0/5 | 全部 0 | 全部 0 |

进一步对 40 个 run 逐项核对：

$$
\max |T_{\mathrm{Temporal}}-T_{\mathrm{E2}}|=0\ \mathrm{Mbps}.
$$

backlog 和 physical service 同样逐项相同。

### 8.4 历史是否真的改变 logit

测试序列上：

- prior 的链路间标准差非零，所以空间输出仍对 CSI 敏感；
- mean absolute temporal residual = 0；
- mean absolute history contribution = 0。

因此失败不是表示坍缩，而是最佳验证 checkpoint 没有采用时间修正。

### 8.5 为什么不再跑 reset-every-block 完整队列消融

预先规定：只有 stateful Temporal 先通过 Frozen、Pure、E2 的联合闸门，才需要额外跑
“每块清零历史”的 40-run 队列消融。

本结果中 stateful logit 与 E2 已经严格相同，history contribution 也为 0。此时再运行
memory-reset 队列实验必然得到完全相同动作，只会重复 40 个无信息实验。因此按条件计划
不再运行。

---

## 9. 与 Frozen-GAT 和 Pure 的关系

Temporal 最终等于 E2，所以它保留了 E2/Frozen 已有的现象：

- 高压力 imperfect CSI 下仍能明显胜过 Pure；
- 低负载不形成一致优势；
- 它没有额外胜过 Frozen-GAT；
- 它也没有缓解 relative-to-latest 的绝对 CSI 损失。

不能把原本属于 Frozen-GAT 的条件性优势重新记到 GRU 名下。

---

## 10. 消息轮数、存储与计算开销

### 10.1 空口消息轮数

时间状态只在同一链路本地更新，不发送给邻居。因此与 GAT-assisted E2 相同：

- 每信道块：4 轮 GAT 隐藏状态 + 1 轮 prior 交换；
- 每时隙：3 轮队列/物理协商；
- 5-slot block 下：$3+5/5=4$ rounds/slot。

Pure 仍是 3 rounds/slot。

### 10.2 新增本地存储

每条可用永久链路保存 48 个 float32：

$$
48\times32=1536\ \mathrm{bits/link}.
$$

正式 40-run 中按平均可用链路数核算：

- 平均约 55.67 kbit；
- 最小约 47.95 kbit；
- 最大约 70.66 kbit。

这些是模型本地状态存储，不是空口 payload。

### 10.3 新增计算

每个信道块每条链路运行一次 48×48 GRUCell 和一个 48→1 线性头。当前报告记录了参数和
存储，但尚未用目标 UAV 硬件测量能耗或墙钟延迟，因此不能把参数量伪装成真实部署时延。

---

## 11. Phase F 正式闸门

时间模型至少要在一种 imperfect CSI 条件中同时满足：

1. 相对 Pure 至少 4/5 seeds 联合改善 delivered throughput 和 backlog；
2. 相对 Frozen-GAT 也至少 4/5；
3. 相对 E2 CSI-Augmented Spatial GAT 也至少 4/5；
4. 输出仍对 CSI 敏感；
5. 旧链路状态确实改变当前 logit。

本模型只满足第 4 项，其他均未满足。因此：

> `temporal_residual_gate_passed = false`。

---

## 12. 止损之后下一步做什么

按照投稿前预定规则，本轮在这里停止扩大 **Frozen-teacher-prior residual**：

- 不训练更深 GRU/LSTM；
- 不增加 temporal attention 或 Transformer；
- 不给该近常数teacher增加 CSI-mode head、辅助 loss 或预测 horizon 搜索；
- 不重新打开 fixed-CSI GAT 架构搜索；
- 不修改 LocalPhysicalReuse-v2 队列和物理阶段。

论文主线保留：

1. `LocalPhysicalReuse-v2` 作为 GAT-assisted physics-informed hybrid；
2. Pure 作为必要的 analytic-prior 消融；
3. Frozen-GAT 高压力 imperfect CSI 的条件性 resilience 作为敏感性结果；
4. E2 说明单帧数据增强可以改善 prior reconstruction、但未转化为队列增益；
5. T1/T2 作为最小时间扩展负结果，决定不扩大模型搜索。

后续channel-only审计已经重新打开一个更窄且物理定义不同的方向：预测current物理
direct/cross gain或物理rate，并同时服务骨架和加入协商。它不是继续给失败teacher打
补丁；是否实施以 `CHANNEL_MODEL_AND_STAGE_MASKING_TUTORIAL.md` 的闸门为准。

---

## 13. 当前证据可以和不可以怎样表述

### 可以表述

> The frozen spatial GAT retained conditional robustness over the analytic-prior
> ablation under high-load imperfect CSI. However, neither same-frame CSI-augmented
> retraining nor a minimal persistent-link GRU residual produced consistent gains
> over the frozen/spatial baselines. The temporal candidate therefore reverted to
> its zero-residual initialization under independent validation.

### 不可以表述

- “STGNN 在 UAV 信道预测中普遍无效”；
- “历史与当前信道统计独立”；
- “GRU 已经过充分结构和超参数搜索”；
- “Frozen-GAT 在所有负载和 CSI 下都更鲁棒”；
- “1000-slot、5-seed 结果证明了严格稳定域”。

---

## 14. 文件与复现入口

- 模型：`models/temporal_residual_gat.py`
- 训练：`train/train_temporal_residual_gat.py`
- 统一测试：`experiments/evaluate_temporal_residual_gat.py`
- 单元测试：`tests/test_temporal_residual_gat.py`
- 正式 checkpoint：
  `results/models/gat_temporal_residual_v1_numUAV=10_tx=10dBm.pth`
- 正式训练报告：
  `results/diagnostics/temporal_residual_train_N10_20260715.json`
- 正式队列报告：
  `results/diagnostics/temporal_residual_eval_N10_20260715.json`
- 初始学习率审计：
  `results/diagnostics/temporal_residual_learning_rate_audit_20260715.json`

全套 29 个单元测试通过；此外已通过 `py_compile` 和 `git diff --check`。
