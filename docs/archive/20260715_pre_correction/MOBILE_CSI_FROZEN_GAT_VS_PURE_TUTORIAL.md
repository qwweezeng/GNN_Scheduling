# 移动与不完美 CSI：Frozen-GAT 对 Pure 的第一层鲁棒性实验教程

> **文档状态：历史实验快照。** 本文结果来自旧 Rayleigh 口径，只保留作结构诊断。
> 当前唯一入口为
> [`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`](MOBILE_BLOCK_FADING_CSI_TUTORIAL.md)。

> 更新时间：2026-07-15。
>
> 本文记录已经完成的第一阶段：不修改会议 GAT、不重新训练，只判断原会议
> checkpoint 在移动和不完美 CSI 下是否具有“天然鲁棒性”。
>
> 本文不是最终时空 GAT 结果。它的作用是建立后续 CSI 增强重训和时间建模必须超过
> 的基线。

---

## 0. 如果只读一页，先读这里

### 0.1 为什么要做这个实验

旧 E9--E12 已经证明：使用 one-block stale CSI 会显著降低
LocalPhysicalReuse-v2 的服务能力。

但旧实验只运行了 GAT-assisted 调度器，没有运行不使用 GAT 的 Pure 调度器。因此
旧结果不能区分：

1. 所有物理调度器都会被错误 CSI 破坏；
2. 会议 GAT 对训练外 CSI 特别敏感；
3. GAT 学到的空间干扰先验反而能在部分不完美 CSI 下帮助调度。

本轮把 Frozen-GAT 和 Pure 放到完全相同的真实移动、衰落、到达和 CSI 观测上，
直接回答这个归因问题。

### 0.2 两个方法是什么

**Frozen-GAT**：会议版 channel-only GAT checkpoint 完全冻结。它根据观测 CSI
产生活动先验，再由 LocalPhysicalReuse-v2 形成队列骨架和条件空间复用。

**Pure**：不使用 GAT，只用观测 CSI 下的单链路无干扰速率形成队列骨架；后续条件
空间复用与 Frozen-GAT 完全相同。

所以二者的区别只在 GAT 信道先验，不在队列算法和 SINR 物理计算。

### 0.3 最重要的结果

N=10、1000 slots、5 个配对 seeds、5 m/s、5 ms block：

| 负载 | CSI | GAT/Pure 联合胜利 | GAT-Pure delivered 差值 | GAT backlog 改善 | 闸门 |
|---:|---|---:|---:|---:|---|
| 2.2 | latest | 0/5 | -0.192 Mbps | -0.00336 Mbit/link | 未通过 |
| 2.2 | stale | 2/5 | -0.194 Mbps | -0.00383 Mbit/link | 未通过 |
| 2.2 | large-scale | 2/5 | +0.396 Mbps | +0.00129 Mbit/link | 未通过 |
| 2.2 | 4-bit distance | 3/5 | +0.581 Mbps | +0.00285 Mbit/link | 未通过 |
| 4 | latest | 3/5 | +0.362 Mbps | +0.00425 Mbit/link | 未通过 |
| 4 | stale | **5/5** | **+1.270 Mbps** | **+0.01898 Mbit/link** | 通过 |
| 4 | large-scale | **4/5** | **+2.122 Mbps** | **+0.02381 Mbit/link** | 通过 |
| 4 | 4-bit distance | **5/5** | **+2.575 Mbps** | **+0.03359 Mbit/link** | 通过 |

正的 delivered 差值和正的 backlog 改善都表示 Frozen-GAT 更好。

### 0.4 一句话结论

> 原会议 GAT 没有在所有负载下普遍优于 Pure，但在 `Lambda=4` 的持续高竞争状态下，
> 它在 stale、large-scale 和 4-bit distance-quantized large-scale CSI 中形成了
> 4/5 或 5/5 seeds 的一致优势。

因此可以说：

> 会议 GAT 学到的空间干扰先验在“队列持续非空 + CSI 不完美”的条件下表现出天然
> resilience。

但不能说：

> Frozen-GAT 已经解决不完美 CSI。

因为在 `Lambda=4` 下，它相对自身 latest CSI 仍损失约 20%--31% delivered
throughput。这个明显剩余损失正是继续做 CSI 增强重训和最小时间建模的动机。

---

## 1. 本轮在整条研究线中的位置

```text
固定 perfect CSI 下确定 LocalPhysicalReuse-v2
        ↓
移动 E9--E12 证明 stale CSI 会损害 GAT-assisted 调度
        ↓
本轮：Frozen-GAT / Pure 同轨迹配对
        ↓
判断 GAT 是否在不完美 CSI 下仍有可重复价值
        ↓
结果：高负载通过，低负载未通过，而且绝对 CSI 损失仍很大
        ↓
下一步：CSI-augmented Spatial GAT
        ↓
再下一步：Temporal-Residual GAT
```

本轮不是为了把冻结模型包装成最终方案，而是为了确认：把会议 GAT 作为后续时空
模型的空间骨架，是否仍然有实验依据。

结果表明有依据，但这种价值具有明显的负载条件。

---

## 2. 新概念逐一解释

### 2.1 Frozen-GAT 是什么

Frozen 表示“冻结”。本轮直接加载：

`results/models/gat_numUAV=10_tx=10dBm_csi=perfect.pth`

所有参数保持不变，没有训练、fine-tune 或加入时间模块。

因此它测量的是原会议模型的自然跨分布泛化，不是经过适配后的最好性能。

### 2.2 Pure 是什么

PurePhysicalReuse 不使用学习先验。骨架权重为：

$$
W_i^{\mathrm{Pure}}(t)=Q_i(t)C_i^{\mathrm{solo}}(t),
$$

其中 $C_i^{\mathrm{solo}}$ 是调度器观测 CSI 下的单链路无干扰速率。

Pure 仍然使用同一套：

- 局部胜者骨架；
- 条件 SINR；
- 候选加入收益；
- 一轮并行空间复用。

因此 Pure 不是“简单随机基线”，而是去掉 GAT 后的关键解析消融。

### 2.3 天然 resilience 是什么

这里的 resilience 指：

> 模型没有针对移动或不完美 CSI 重新训练，却仍能比 Pure 更稳定地形成有用的发送
> 集合。

它不等于“性能不下降”。一个方法可以比 Pure 更鲁棒，但仍明显弱于自己的 latest
CSI 上限。

所以本文同时报告两类比较：

1. Frozen-GAT 对 Pure：判断 GAT 是否有增量价值；
2. 每个方法对自己的 latest：判断 CSI 不完美还造成多少绝对损失。

### 2.4 Delivered throughput 是什么

无线速率可以在一个时隙中只发送 0.3 个包对应的数据量。这个包尚未完整交付，不能
立即算成用户收到的完整数据包。

因此 delivered throughput 只统计在仿真窗口内已经完成全部服务的 FIFO 包：

$$
D=\frac{N_{\mathrm{completed}}S_p}{T_{\mathrm{sim}}}.
$$

它的单位是 Mbps。

### 2.5 Physical service 是什么

Physical service 是实际信道已经提供的连续服务量，包括尚未完成包已接受的部分服务。

它回答“调度集合的物理传输效率如何”，但不能单独代表完整包交付。因此本轮闸门要求：

- delivered throughput 改善；
- backlog 同时改善；
- physical service 只负责解释机制。

### 2.6 Average backlog 是什么

Backlog 是仍留在队列中的数据。本文使用时间平均、链路平均的 Mbit/link：

$$
\overline Q=\frac{1}{TL}\sum_{t,i}Q_i(t)\frac{S_p}{10^6}.
$$

在共享到达轨迹下，backlog 更低表示业务被更及时地服务。

### 2.7 联合胜利是什么

对同一个 seed，定义：

$$
\Delta D_s=D_s^{\mathrm{GAT}}-D_s^{\mathrm{Pure}},
$$

$$
\Delta Q_s=Q_s^{\mathrm{Pure}}-Q_s^{\mathrm{GAT}}.
$$

只有 $\Delta D_s>0$ 且 $\Delta Q_s>0$，才计为一次联合胜利。

这个定义避免以下错误：

- 只提高物理服务，却没有完成更多包；
- 只提高吞吐，却让积压更严重；
- 平均值由一个特别有利的 seed 驱动。

### 2.8 4/5 闸门是什么

一种 imperfect CSI 条件至少需要 4 个 seed 联合胜利，并且五个 seed 的平均吞吐差
和 backlog 改善都为正，才算 Frozen-GAT 的天然 resilience 通过。

它是预先设定的方向一致性规则，不是看到结果以后选择的阈值。

### 2.9 95% CI 是什么

CI 是基于五个 seed 计算的均值不确定性范围。五个 seed 仍然不够支持非常强的统计
显著性结论，因此本文同时强调逐 seed 方向和效应大小。

---

## 3. 实验前先做了哪些实现修正

### 3.1 动机

旧移动 runner 只运行 Frozen-GAT，而且把连续 physical departures/arrivals 称为
throughput。它不能完成本轮需要的完整包 GAT/Pure 配对。

### 3.2 具体修改

1. 增加 `PurePhysicalReuseScheduler`；
2. 支持 `Lambda=2.2,4` 多负载；
3. 默认只保留 5 m/s、5-slot block 和 4-bit distance quantization；
4. 增加与固定 CSI 一致的 FIFO 完整包完成记账；
5. 分开输出 delivered throughput 和 physical service；
6. 自动计算每个 seed 的 GAT-Pure 联合胜利；
7. 把旧“stale 是否严重”闸门改为 Frozen-GAT/Pure 天然 resilience 闸门；
8. 分别核算 GAT 和 Pure 的算法消息轮数。

### 3.3 没有修改什么

- 没有修改会议 GAT 参数；
- 没有修改 LocalPhysicalReuse 公式；
- 没有修改真实/观测信道分离；
- 没有修改永久 link identity；
- 没有引入预测器或时间模块。

### 3.4 新增测试

新增测试验证：

- 完整包数 + 未完成包数 = 总到达包数；
- 暂时不可用链路仍保留 FIFO 包；
- physical service 的流体工作量守恒；
- 只有吞吐/backlog 同 seed 同时改善才计联合胜利；
- Pure 不承担 GAT 的 block-level 消息轮数。

最终全量 26 个 `unittest` 全部通过。

---

## 4. Smoke test：为什么不能直接跑正式结果

### 4.1 目的

Smoke test 是小规模实现检查，不是性能实验。它回答：

> 32 个组合能否完整运行，报告层级、配对、守恒和消息字段是否正确？

### 4.2 设置

- 2 seeds：42、43；
- 100 slots；
- 2 个负载；
- 2 个调度器；
- 4 种 CSI；
- 共 32 次运行。

### 4.3 结果

- 32 次运行全部完成；
- 每个 `Lambda × seed` 的八个方法具有完全相同的总到达包数；
- packet count conservation error 为 0；
- 最大流体工作量误差约 $3.05\times10^{-5}$ packet，属于浮点误差；
- latest CSI error 为 0；
- GAT 为 4.0 rounds/slot，Pure 为 3.0 rounds/slot。

### 4.4 观察到的现象

seed 42 多数条件 Pure 略好；seed 43 的 large-scale/4-bit 条件 GAT 明显更好。

这不能解释为正式性能，但说明优劣具有拓扑依赖，5-seed 配对不可省略。

### 4.5 下一步

保持代码、checkpoint、移动设置不变，将时间延长到1000 slots并使用五个 seeds。

---

## 5. 正式实验的共同设置

| 参数 | 值 | 含义 |
|---|---:|---|
| UAV 数 | 10 | 与会议 checkpoint 和固定 CSI 主结果一致 |
| seeds | 42--46 | 五条共享移动/衰落/到达轨迹 |
| slots | 1000 | 每条轨迹 1 秒 |
| 速度 | 5 m/s | 第一轮固定一个中等移动速度 |
| block | 5 slots | 真实信道每 5 ms 更新，$\rho\approx0.643$ |
| 负载 | 2.2 / 4 packets/slot | 边界负载与压力负载 |
| CSI | latest/stale/LS/4-bit distance | 四种观测条件 |
| deadline | 无 | 不引入超时丢包 |
| buffer | 无限 | 不让溢出截断 backlog |

总运行数：

$$
2\times4\times2\times5=80.
$$

所有方法的真实服务均使用当前瞬时信道，只改变调度器看到的 CSI。

---

## 6. 实验 F1：Latest CSI——移动环境控制组

### 6.1 动机

Latest 条件不是不完美 CSI 闸门，而是检查移动环境中两种骨架的基础关系。后续 stale、
large-scale 和 4-bit 都要和各自方法的 latest 比较。

### 6.2 定义

每个信道块，调度器看到当前真实直接和交叉功率，动作也在同一当前信道结算。

### 6.3 与固定 CSI 的区别

模型不变，但位置和 small-scale fading 每个 block 更新，GAT 每块重新执行。

### 6.4 结果

| Lambda | Frozen-GAT goodput | Pure goodput | GAT-Pure | 联合胜利 | GAT/Pure backlog |
|---:|---:|---:|---:|---:|---:|
| 2.2 | 25.438 | 25.630 | -0.192 | 0/5 | 0.01499 / 0.01162 |
| 4 | 36.310 | 35.947 | +0.362 | 3/5 | 0.17840 / 0.18265 |

Goodput 单位 Mbps，backlog 单位 Mbit/link。

### 6.5 现象

- 在 `Lambda=2.2`，Pure 五个 seed 全部更好；
- 在 `Lambda=4`，GAT 平均略高，但只有3/5联合胜利；
- 两个 latest 点都没有通过4/5闸门。

### 6.6 说明了什么

这与固定 CSI 结论一致：GAT 信道先验并非所有状态都比 Pure 更好，其价值更可能在
持续高竞争状态出现。

### 6.7 下一步

把观测延迟一个 block，检查错误瞬时 CSI 是否改变 GAT/Pure 的相对关系。

---

## 7. 实验 F2：One-block stale CSI

### 7.1 动机

这个实验验证：当调度器拿到5 ms以前的瞬时信道时，GAT 学到的空间先验是否比 Pure
的单链路速率骨架更能抵抗错误干扰关系。

### 7.2 定义

块 $b$ 使用块 $b-1$ 的直接和交叉功率决策；当前队列仍是最新；最终服务使用块 $b$
真实信道。

### 7.3 与 F1 的唯一区别

观测 CSI 晚一个 block。模型、队列、到达和真实服务轨迹全部不变。

### 7.4 结果

| Lambda | GAT goodput | Pure goodput | GAT-Pure | 联合胜利 | backlog 改善 | 闸门 |
|---:|---:|---:|---:|---:|---:|---|
| 2.2 | 22.421 | 22.615 | -0.194 | 2/5 | -0.00383 | 未通过 |
| 4 | **28.882** | 27.612 | **+1.270** | **5/5** | **+0.01898** | 通过 |

### 7.5 现象

`Lambda=2.2` 时 GAT 没有优势；`Lambda=4` 时五个 seed 的 delivered throughput 和
backlog 全部同方向优于 Pure。

物理服务差为 +1.279 Mbps，与完整包 goodput 的 +1.270 Mbps 基本一致，说明优势
不是有限窗口的整包取整偶然，而是实际服务集合更有效。

### 7.6 可能原因

在高负载下，大量链路长期非空，骨架选择主要受信道空间兼容性支配。会议 GAT 的邻域
干扰表示可能比 Pure 的单链路 solo-rate 更能保留“哪些链路适合共同出现”的结构。

这仍是基于结果的机制推断，尚未通过 embedding 或动作集合分析直接证明。

### 7.7 说明了什么

冻结会议 GAT 在高竞争 stale CSI 下具有可重复的相对鲁棒性。

### 7.8 不能说明什么

它仍比自身 latest goodput 低约20.5%，所以不能说 stale 问题已经解决。

### 7.9 下一步

移除全部 small-scale 观测，只使用当前 large-scale CSI，检查 GAT 空间表示是否仍有
价值。

---

## 8. 实验 F3：Current large-scale CSI

### 8.1 动机

若瞬时 CSI 难以及时获得，系统可能只能知道当前位置和平均路径损耗。该实验验证 GAT
是否能从粗糙的当前几何干扰图中提取比 Pure 更有用的空间模式。

### 8.2 定义

调度器使用当前距离/path loss 计算直接与交叉功率，看不到 small-scale fading；真实
服务仍使用当前瞬时信道。

### 8.3 与 F2 的区别

F2 是“完整但过期的细地图”，F3 是“当前但缺少小尺度衰落的粗地图”。

### 8.4 结果

| Lambda | GAT goodput | Pure goodput | GAT-Pure | 联合胜利 | backlog 改善 | 闸门 |
|---:|---:|---:|---:|---:|---:|---|
| 2.2 | 20.194 | 19.798 | +0.396 | 2/5 | +0.00129 | 未通过 |
| 4 | **25.020** | 22.898 | **+2.122** | **4/5** | **+0.02381** | 通过 |

### 8.5 现象

- 低负载平均值虽然偏向 GAT，但方向只在2/5 seeds成立，不能叫稳定优势；
- 高负载4/5 seeds联合胜利，平均 delivered 增益比 stale 更大；
- 高负载物理服务增益 +2.112 Mbps，与 goodput 增益一致。

### 8.6 可能原因

Pure 只根据 large-scale solo rate 排序，无法表达“邻居们同时活动时的空间干扰图案”。
GAT 的多层邻域聚合即使输入变粗，仍可能保留相对空间结构，因此在所有队列都很活跃时
更有帮助。

另一方面，Frozen-GAT 原本没有用 large-scale 输入训练，能够取得4/5优势是正向
证据，但不是输入已经校准的证明。

### 8.7 下一步

在相同 large-scale 信息上只把距离压缩为4 bit，检查有限距离精度是否改变结论。

---

## 9. 实验 F4：4-bit distance-quantized large-scale CSI

### 9.1 动机

真实系统交换位置或距离时精度有限。旧实验已经显示2-bit过粗、4-bit接近连续
large-scale，因此本轮只保留4-bit，避免重复搜索。

### 9.2 定义

当前距离被映射到16个均匀区间，再用代表距离重建 path loss。它不是复信道幅度/相位
的4-bit量化。

### 9.3 与 F3 的唯一区别

F3 使用连续距离，F4 使用16级距离。二者都看不到 small-scale fading。

### 9.4 结果

| Lambda | GAT goodput | Pure goodput | GAT-Pure | 联合胜利 | backlog 改善 | 闸门 |
|---:|---:|---:|---:|---:|---:|---|
| 2.2 | 20.304 | 19.723 | +0.581 | 3/5 | +0.00285 | 未通过 |
| 4 | **25.435** | 22.860 | **+2.575** | **5/5** | **+0.03359** | 通过 |

### 9.5 现象

高负载下五个 seed 全部联合胜利，而且这是三种 imperfect CSI 中平均相对 Pure 增益
最大的条件。

### 9.6 如何正确解释

不能说“量化提高了 GAT 性能”。4-bit 与连续 large-scale 的差异处于 seed 波动内，
而且量化也会改变 Pure。更稳妥的解释是：

> 在当前距离精度达到4 bit后，GAT 相对 Pure 的高负载空间先验优势仍然保留，没有被
> 距离量化破坏。

### 9.7 下一步

把四个条件合起来判断天然 resilience，同时检查相对 latest 的绝对损失是否还值得
训练新模型。

---

## 10. 把四个实验合起来后的结论

### 10.1 结论 A：GAT 的价值具有负载条件

`Lambda=2.2` 没有任何 CSI 条件通过。队列较轻时，许多链路不是持续非空，Pure 的
简单 solo-rate 骨架已经有效；GAT 先验有时反而改变了本来足够好的排序。

`Lambda=4` 的三种 imperfect CSI 全部通过。此时队列持续非空、并发竞争强，GAT
学习到的邻域干扰结构更容易发挥作用。

### 10.2 结论 B：Frozen-GAT 有相对 resilience，但绝对性能仍明显下降

`Lambda=4` 相对自身 latest 的 delivered throughput 损失：

| 方法 | stale | large-scale | 4-bit distance |
|---|---:|---:|---:|
| Frozen-GAT | 20.5% | 31.1% | 30.0% |
| Pure | 23.2% | 36.4% | 36.5% |

Frozen-GAT 的退化确实更小，但30%左右的 absolute loss 仍然很大。

所以当前最准确的结论是：

> Frozen-GAT 比 Pure 更能利用高负载下的不完美空间 CSI，但它仍没有恢复 latest CSI
> 能达到的服务水平。

### 10.3 结论 C：继续开发最小 CSI 适配具有依据

继续的依据同时来自两方面：

1. GAT 相对 Pure 已经在高负载形成4/5或5/5优势，说明空间图学习不是无效部件；
2. Frozen-GAT 相对 latest 仍有20%--31%损失，说明存在明确改进空间。

因此下一步不是重新设计队列模型，而是训练 GAT 如何处理 CSI 抖动和时间变化。

---

## 11. 为什么本轮不解释 packet delay

完成率如下：

| Lambda | CSI | Frozen-GAT | Pure |
|---:|---|---:|---:|
| 2.2 | latest | 96.5% | 97.2% |
| 2.2 | stale | 85.0% | 85.8% |
| 2.2 | large-scale | 76.6% | 75.1% |
| 2.2 | 4-bit | 77.0% | 74.8% |
| 4 | latest | 75.2% | 74.5% |
| 4 | stale | 59.8% | 57.2% |
| 4 | large-scale | 51.9% | 47.4% |
| 4 | 4-bit | 52.7% | 47.4% |

所有点都低于预设的99%完成率门槛。很多慢包在仿真结束时仍未完成，只平均已完成包会
产生幸存者偏差。

所以本轮只解释 delivered throughput、physical service 和 backlog，不解释平均包
时延或P95时延。

---

## 12. 消息轮数与开销

### 12.1 轮数

| 方法 | 每块信道阶段 | 每时隙队列阶段 | 5-slot块摊销 |
|---|---:|---:|---:|
| Frozen-GAT | 5轮 | 3轮 | 4.0 rounds/slot |
| Pure | 0轮 | 3轮 | 3.0 rounds/slot |

Frozen-GAT 多出的是4层 GAT 传播和1轮活动先验交换。

### 12.2 Payload 审计

| 方法 | 逐冲突边单播上界 | 按发射UAV聚合payload下界 |
|---|---:|---:|
| Frozen-GAT | 1327.9 Mbit/s | 48.32 Mbit/s |
| Pure | 97.6 Mbit/s | 3.55 Mbit/s |

这些不是实际空口开销，因为尚未加入 CSI 获取、包头、竞争、重传和半双工时序。

它们说明 Frozen-GAT 的高负载鲁棒性不是免费得到的。后续 Temporal-Residual GAT 的
时间状态保存在本地，不应增加邻居消息轮数；若实现需要新增跨链路时间消息，则必须
重新核算并说明必要性。

---

## 13. 当前能写和不能写的论文表述

### 13.1 可以写

- Frozen conference GAT 在高负载 imperfect-CSI 条件下相对 Pure 形成一致优势；
- 这种优势同时出现在完整包 goodput、physical service 和 backlog；
- GAT 的相对退化小于 Pure，说明 learned spatial prior 具有条件性 resilience；
- 低负载没有一致优势，GAT 增益依赖竞争强度；
- 不完美 CSI 相对 latest 仍造成明显 absolute loss，需要专门适配。

### 13.2 不能写

- GAT 在所有负载和 CSI 下普遍优于 Pure；
- Frozen-GAT 已解决 stale CSI；
- 4-bit 距离量化优于连续 large-scale；
- 现有结果证明时间模型一定有效；
- 当前 payload 数字就是实际 MAC 开销；
- 1000-slot、5-seed 结果构成吞吐最优或稳定域定理。

---

## 14. 下一阶段：为什么先做 CSI-augmented Spatial GAT

Frozen-GAT 从未在移动、不完美 CSI 上训练。它的剩余损失可能来自：

1. 输入分布改变；
2. 单帧输入缺少时间信息。

如果直接加入 GRU 后性能提高，我们无法知道提升来自“看过移动数据”还是“时间记忆”。

因此下一步先保持会议 GAT 结构完全不变，只增加移动/扰动 CSI 训练数据。输入为观测
图，训练速率在当前真实信道上结算，目标仍是 channel-only 网络总速率。

它回答：

> 只改变训练数据和训练分布，能否已经恢复 Frozen-GAT 的 CSI 性能？

只有再加入时间状态并超过这个同架构重训基线，才能证明时间维度本身有贡献。

---

## 15. 文件与复现入口

### 15.1 实现

- `simulator/mobile_block_fading.py`：永久队列、真实/观测 CSI、FIFO 完整包指标；
- `experiments/evaluate_mobile_block_fading_csi.py`：2×4×2×5 配对实验和自动闸门；
- `tests/test_mobile_block_fading.py`：link identity、FIFO守恒和消息轮数；
- `tests/test_mobile_csi_experiment.py`：联合胜利和闸门逻辑。

### 15.2 Smoke 结果

`results/diagnostics/mobile_csi_frozen_gat_vs_pure_smoke_20260715.json`

### 15.3 正式结果

`results/diagnostics/mobile_csi_frozen_gat_vs_pure_N10_1000slots_5seeds_20260715.json`

### 15.4 正式命令

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/evaluate_mobile_block_fading_csi.py \
  --num_uav 10 --num_slots 1000 \
  --arrival_rates 2.2,4 \
  --mc_seeds 42,43,44,45,46 \
  --speeds_mps 5 --slots_per_blocks 5 \
  --quantization_bits 4
```

---

## 16. 本阶段最终总结

本阶段完成了四件事：

1. 建立了 Frozen-GAT/Pure 的严格同轨迹配对；
2. 分开核算完整包 delivered throughput 与连续 physical service；
3. 证明会议 GAT 在高负载 imperfect CSI 下具有4/5或5/5 seeds的一致相对优势；
4. 同时确认 Frozen-GAT 相对 latest 仍损失20%--31%，不能作为最终 CSI 方案。

因此第一层结论是：

> 会议 GAT 作为后续时空模型的空间骨架具有实验依据；下一步应先用移动 CSI 数据做
> 同架构重训，再验证最小时间记忆能否带来独立增益。
