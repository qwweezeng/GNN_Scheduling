# 信道模型、GAT Prior 与两阶段调度遮蔽效应：从零诊断教程

> **文档状态：历史实验快照。** 本文保留 2026-07-15 的完整诊断证据，不再单独
> 维护后续计划。当前结论、物理模型修正和统一研究路线请从
> [`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`](MOBILE_BLOCK_FADING_CSI_TUTORIAL.md) 进入。

> 日期：2026-07-15  
> 任务定位：在继续设计时间 GAT 之前，先回答三个更基础的问题：当前信道是否缺少
> LoS；会议 GAT 的 channel-only 输出是否仍有调度辨识力；LocalMax 骨架与后续加入
> 协商分别怎样放大或修复 CSI 误差。

---

## 0. 一页结论

本轮得到四个必须改变后续研究判断的结论。

### 结论一：当前确实是纯 Rayleigh，但不是代码偏离会议论文

会议论文第 II-A 节定义：

$$
h_{ij}=a_{ij}g_{ij},\qquad g_{ij}\sim\mathcal{CN}(0,1).
$$

当前固定网络生成器和移动 block-fading 生成器都忠实实现了这个公式。因此代码没有
“忘记实现论文里的 LoS”；更准确的问题是：**会议论文自身采用了零均值 Rayleigh
小尺度衰落，与低空 A2A 常见的显著 LoS 分量存在模型张力。**

### 结论二：Rayleigh 显著放大不可预测性，但不是 GRU 失败的唯一原因

在相同 UAV 轨迹和 diffuse 随机数下：

| 信道 | current-stale dB相关 | stale误差 | large-scale误差 |
|---|---:|---:|---:|
| Rayleigh | 0.548 | 5.04 dB | 4.42 dB |
| Rician $K=5$ dB | 0.664 | 3.42 dB | 2.83 dB |
| pure LoS | 约1.000 | 0.0008 dB | 约0 dB |

Rician stale-only teacher-logit MSE 也比 Rayleigh 低约55%。但是，在 Rayleigh 和
Rician 5 dB 下分别单独训练同一 GRU，最佳 checkpoint 都仍是零残差 `epoch=-1`。

因此 Rayleigh 是“问题更难”的重要原因，但不能单独解释 GRU 无用。

### 结论三：冻结会议 GAT 的直接二值输出已经全开

正式实验包含3种信道×4种 CSI×5 seeds×20 blocks，共1200个状态。在全部状态中：

- prior 均值约0.91--0.925；
- 链路间 prior 标准差只有0.0029--0.0044；
- 最小 prior 仍约0.904--0.909；
- 所有链路都高于0.5阈值；
- `Conference-GAT binary` 在1200/1200状态中都是全链路激活。

这说明用 Frozen-GAT teacher logit 作为 GRU 预测目标，本身缺少足够的调度辨识力。

### 结论四：后半段物理机制确实遮蔽了 prior 问题，但骨架和加入协商都重要

虽然会议 GAT 直接阈值后全开，LocalPhysicalReuse-v2 并不直接采用这个二值动作。
它用连续 prior 构造 expected rate，然后重新执行 LocalMax 和条件加入。最终平均只激活
约4.6--5.4条链路，并达到 full-information greedy 的约91%--94%。

在 latest CSI 下：

- 骨架贡献约66%--72%的最终频谱效率；
- 一轮加入协商补充约28%--34%；
- Pure 在本次统一队列权重 channel-only 实验中仍略高于 GAT-assisted。

所以后半段确实“救回”了近常数 prior，但不能说只有后半段决定性能。Rayleigh stale 下，
骨架相对 latest 的 Jaccard 只有0.387，最终集合也只有0.413，骨架选择本身明显受 CSI
影响。

---

## 1. 为什么要在队列实验之前做这个诊断

动态队列实验把多种机制叠加在一起：

1. GAT channel prior；
2. 当前队列；
3. LocalMax 骨架；
4. 一轮条件加入；
5. 多时隙队列反馈。

若最终 goodput 变化不大，可能有两种完全不同的解释：

- prior 真的对 CSI 误差鲁棒；
- prior 已经接近常数，但后续确定性物理规则替它完成了大部分工作。

这两种解释对论文贡献和后续 STGNN 设计完全不同。因此本轮暂时取消动态队列，把所有
链路权重统一设为1，只看 Shannon 频谱效率、动作集合和模块反事实。

---

## 2. 新概念完整解释

### 2.1 Rayleigh fading

Rayleigh 模型令小尺度复系数为零均值复高斯：

$$
g\sim\mathcal{CN}(0,1).
$$

其功率 $|g|^2$ 服从指数分布。它没有确定性直射分量，容易出现深衰落。移动模型用
Gauss--Markov/Jakes 相关保存一部分历史，同时每块注入新的随机 innovation。

### 2.2 LoS 分量

LoS 是发送端与接收端之间的确定性直射径。窄带复基带可写成：

$$
g_{\mathrm{LoS}}(d)=e^{-j2\pi d/\lambda}.
$$

当前调度器只使用接收功率 $|h|^2$，不做多径相干合并。在 pure-LoS 单径极限下，
$|g_{\mathrm{LoS}}|^2=1$，所以功率主要随距离路径损耗平滑变化。

### 2.3 Rician fading 与 K-factor

Rician 模型同时包含 LoS 和 diffuse 多径：

$$
g_K=sqrt{\frac{K}{K+1}}g_{\mathrm{LoS}}
+\sqrt{\frac{1}{K+1}}g_{\mathrm{diffuse}}.
$$

$K$ 是直射功率与散射功率之比。$K=0$ 的线性极限趋向 Rayleigh，$K\to\infty$
趋向 pure LoS。

本轮使用 $K=5$ dB，不把它宣称为所有 UAV 场景的固定真值。依据是2026年的低空
A2A/A2G同步测量在其校园连续LoS场景中报告A2A中位K-factor约5 dB。它只是一个具有
测量依据的中间敏感性点。

### 2.4 为什么三种信道保持平均功率公平

有限K的两个系数满足：

$$
\frac{K}{K+1}+\frac{1}{K+1}=1.
$$

因此平均小尺度功率仍为1。Rayleigh、Rician和LoS共享：

- 相同位置；
- 相同速度；
- 相同路径损耗；
- 相同 diffuse 随机数；
- 相同永久链路集合。

只改变 LoS/diffuse 功率比例。

### 2.5 interference-free Shannon ceiling

对每条链路移除全部互扰，并假设所有链路都能各自获得同一资源，得到：

$$
C_{\mathrm{IF}}=\sum_i\log_2\left(1+\frac{S_i}{N_0}\right).
$$

任何实际并发动作中，每条链路 SINR 都不会超过 solo SNR，因此这是严格数学 ceiling。
但它忽略资源共享、半双工和互扰，数值会很松，不能当成可实现调度器。

### 2.6 centralized true-CSI greedy reference

该参考从空集合开始，使用 current true CSI 逐次加入能提高全网 Shannon 和速率的链路，
并允许有益删除。它是一个可实现的集中式强参考，但只是局部贪心解，不是数学最优或
upper bound。

### 2.7 GAT prior 坍缩与“全1”

GAT 输出连续概率 $p_i$。会议算法按 $p_i\ge0.5$ 形成最终二值动作。若所有链路都满足
这个条件，动作就是全1。

当前 prior 并非浮点值精确等于1，所以严格说不是“数值全1”；但它们都在0.9以上且全部
越过阈值，**从会议二值决策语义看确实是全链路激活坍缩。**

### 2.8 Jaccard similarity

比较两个激活集合 $A,B$：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

1表示完全相同，0表示没有共同链路。本文用它比较 imperfect CSI 与 latest CSI 产生的
骨架和最终集合。

### 2.9 两个模块反事实

为了分开骨架错误和加入协商错误，构造：

1. **observed skeleton + true join**：骨架仍由错误CSI产生，但加入阶段使用current true
   CSI；
2. **latest skeleton + observed join**：骨架固定为latest结果，只有加入阶段使用错误CSI。

最终全部动作仍在 current true CSI 下结算。

---

## 3. 实验环境

| 项目 | 设置 |
|---|---:|
| UAV | 10 |
| 轨迹 | 1000 slots |
| block | 5 slots = 5 ms |
| 速度 | 5 m/s |
| seeds | 42--46 |
| 抽样 | 每10个block取1个，跳过block 0 |
| 每种信道快照数 | 100 |
| CSI | latest / stale / large-scale / 4-bit large-scale |
| 队列权重 | 全部固定为1 |
| 服务结算 | current true CSI |

共 $3\times4\times100=1200$ 个 channel-mode 状态。

---

## 4. 实验 C1：会议论文与代码信道审计

### 4.1 动机

确认“当前是纯 Rayleigh”究竟来自实现遗漏，还是会议论文原始假设。

### 4.2 核对结果

会议论文第2页公式(1)明确给出 $g_{ij}\sim\mathcal{CN}(0,1)$，并直接写明
“small-scale factor follows Rayleigh fading”。

当前代码：

```python
real = randn / sqrt(2)
imag = randn / sqrt(2)
h = large_scale * (real + 1j * imag)
```

移动版本只是在block之间加入：

$$
g_b=\rho g_{b-1}+\sqrt{1-\rho^2}\epsilon_b,
$$

仍然是零均值Rayleigh边缘分布。

### 4.3 说明了什么

- 代码实现与会议论文一致；
- 不能把问题描述成实现bug；
- 期刊扩展有理由加入LoS/Rician敏感性，因为论文引言强调高LoS概率，而系统模型却用
  纯Rayleigh。

---

## 5. 实验 C2：信道可预测性是否受 LoS 影响

### 5.1 动机

验证GRU失败是否可能因为Rayleigh innovation过强，使运动方向无法可靠预测接收功率。

### 5.2 指标

- current与上一块stale功率的dB Pearson相关；
- stale相对current的平均绝对dB误差；
- current large-scale相对current true power的误差；
- current-stale创新量的标准差。

### 5.3 结果

| 信道 | current-stale相关 | stale MAE | LS MAE | 创新标准差 |
|---|---:|---:|---:|---:|
| Rayleigh | 0.548 | 5.04 dB | 4.42 dB | 6.79 dB |
| Rician 5 dB | 0.664 | 3.42 dB | 2.83 dB | 4.75 dB |
| pure LoS | 约1.000 | 0.0008 dB | 约0 dB | 0.001 dB |

### 5.4 分析

LoS占比提高后，真实功率更接近由几何距离决定的large-scale分量，stale也更能代表当前
信道。因此用户提出的机制是成立的：纯Rayleigh使运动轨迹对SNR的解释力大幅下降。

但Rician相关性仍只有0.664，因为5 dB K-factor下diffuse功率仍存在，LoS和diffuse复
系数还会相干叠加。不能期待只靠位置得到完美瞬时功率。

---

## 6. 实验 C3：Shannon ceiling 与 channel-only prior 审计

### 6.1 动机

在队列和两阶段物理规则介入前，检查：

1. 信道模型本身怎样改变理论容量边界；
2. perfect/latest CSI 下集中式参考能取得多少；
3. 会议GAT直接阈值动作是否仍然有效。

### 6.2 latest CSI 结果

单位为全网和频谱效率 bit/s/Hz。当前带宽5 MHz时，乘5即可得到数值上的 Mbps。

| 信道 | 无干扰Shannon ceiling | true-CSI greedy | Conference-GAT binary | GAT两阶段 | Pure两阶段 |
|---|---:|---:|---:|---:|---:|
| Rayleigh | 112.43 | 17.11 | 3.45 | 15.57 | 15.70 |
| Rician 5 dB | 121.46 | 16.74 | 3.52 | 15.20 | 15.44 |
| pure LoS | 132.08 | 16.46 | 3.63 | 15.40 | 15.71 |

### 6.3 为什么ceiling升高，greedy反而略降

Rayleigh会产生深衰落，也会偶尔产生很强的有利衰落。无干扰ceiling把所有链路solo
capacity相加，深衰落会显著拉低总和；LoS消除深衰落，因此ceiling升高。

集中式并发greedy则可以选择当前恰好很强、且互扰较小的链路，利用Rayleigh的多用户
分集。消除随机衰落后，这种机会性选择减少，所以greedy从17.11轻微降到16.46。

这两个指标并不矛盾，也说明不能只看一个“上界”判断调度难度。

### 6.4 GAT prior 诊断

| 信道 | mean prior | mean链路std | mean最小prior | 全开状态比例 |
|---|---:|---:|---:|---:|
| Rayleigh latest | 0.9118 | 0.00437 | 0.9066 | 100% |
| Rician latest | 0.9082 | 0.00406 | 0.9037 | 100% |
| pure LoS latest | 0.9145 | 0.00290 | 0.9088 | 100% |

四种CSI和三种信道的1200个状态全部全开。Conference-GAT binary的频谱效率只有
3.45--3.63，远低于true-CSI greedy和两阶段模型。

### 6.5 说明了什么

此前直接用Frozen-GAT true-CSI logit作为temporal teacher存在根本问题：teacher输出
虽然有小数变化，但会议二值语义不区分任何链路。GRU没有一个强、非坍缩的调度目标。

因此旧Temporal-Residual负结果只能否定“重建该teacher prior”这条路线，不能否定
预测物理信道、物理速率或非坍缩oracle score的时间模型。

---

## 7. 实验 C4：imperfect CSI 对纯 Shannon 调度参考的影响

### 7.1 动机

排除“GAT看似不受影响只是因为后半段太强”的混杂，先让同一个集中式greedy只改变其
观察CSI，再用真实当前信道结算。

### 7.2 结果

表中为 observed-CSI greedy 相对同快照 true-CSI greedy 的均值比例。

| 信道 | stale | large-scale | 4-bit large-scale |
|---|---:|---:|---:|
| Rayleigh | 79.0% | 85.4% | 87.3% |
| Rician 5 dB | 91.4% | 93.5% | 94.6% |
| pure LoS | 100.0% | 100.0% | 102.1%* |

`*` 4-bit结果略高于100%不表示超过数学最优。true-CSI greedy只是局部贪心，量化扰动
偶尔把它引向另一个更好的局部解。这再次说明不能把greedy叫作upper bound。

### 7.3 分析

即使完全不使用GAT和队列，Rayleigh下CSI观察误差也会造成13%--21%的可实现频谱效率
损失。换成Rician后降为5%--9%，pure LoS下latest/stale/large-scale几乎等价。

因此此前CSI失效严重不只是GAT结构问题，信道模型本身贡献很大。

---

## 8. 实验 C5：骨架与加入协商分别贡献多少

### 8.1 latest条件的结构贡献

| 信道 | 骨架SE | 最终SE | 加入增益 | 骨架占最终比例 |
|---|---:|---:|---:|---:|
| Rayleigh | 11.01 | 15.57 | +4.56 | 约72% |
| Rician 5 dB | 10.72 | 15.20 | +4.48 | 约72% |
| pure LoS | 9.97 | 15.40 | +5.43 | 约66% |

平均骨架约2.7--2.8条链路，加入协商再补约1.9--2.6条。

所以：

- 骨架提供最终性能的大多数基础；
- 一轮协商贡献并不小，约28%--34%；
- “全部由后半段决定”和“后半段可以忽略”都不准确。

### 8.2 集合是否对CSI不敏感

| 信道/CSI | 骨架Jaccard | 最终Jaccard |
|---|---:|---:|
| Rayleigh stale | 0.387 | 0.413 |
| Rayleigh large-scale | 0.398 | 0.493 |
| Rayleigh 4-bit | 0.377 | 0.459 |
| Rician stale | 0.453 | 0.529 |
| Rician large-scale | 0.461 | 0.620 |
| Rician 4-bit | 0.444 | 0.530 |
| pure LoS stale/LS | 1.000 | 1.000 |

Rayleigh下骨架变化很大，不能说CSI不准对骨架影响不大。Rician提高了一致性，但仍未接近
完全相同。

### 8.3 两个反事实怎样解释损失

表中数值均相对latest最终动作。

| 信道/CSI | 实际错误骨架+错误协商 | 错误骨架+true协商 | latest骨架+错误协商 |
|---|---:|---:|---:|
| Rayleigh stale | 79.1% | 91.2% | 88.6% |
| Rayleigh LS | 88.7% | 92.1% | 94.7% |
| Rayleigh 4-bit | 86.6% | 91.1% | 93.8% |
| Rician stale | 90.0% | 97.1% | 98.5% |
| Rician LS | 95.7% | 94.7% | 98.6% |
| Rician 4-bit | 94.1% | 93.9% | 99.4% |

注意两个单独反事实不能简单线性相加，因为加入集合取决于骨架，骨架改变后局部竞争邻域
也随之改变。它们的作用存在交互。

总体结论：

- Rayleigh stale中，纠正任一阶段都能恢复一部分性能；
- 只预测骨架不能完全解决问题；
- 只让加入阶段看到正确CSI也不能完全解决问题；
- 若以后做物理信道预测，最好让预测CSI同时服务骨架和加入协商，而不是只修GAT prior。

---

## 9. 实验 C6：Rayleigh/Rician stale-only GRU 因果探针

### 9.1 为什么只测试stale

- stale输入包含过去的瞬时diffuse样本，时间模型有可能外推；
- large-scale从未观察diffuse分量，历史无法恢复每块创新；
- pure LoS下stale已经近乎current，无需GRU。

因此用stale单独测试最能回答“Rician后GRU会不会有用”。

### 9.2 公平设置

- 同一E2空间checkpoint；
- 同一48维GRU和零初始化残差；
- 同一train seeds 1000--1019；
- 同一validation seeds 2000--2004；
- 同一learning rate和early stopping；
- 唯一改变Rayleigh或Rician 5 dB。

### 9.3 结果

| 信道 | 零残差初始MSE | 最佳epoch | 最终残差 |
|---|---:|---:|---:|
| Rayleigh stale | 0.03417 | -1 | 0 |
| Rician 5 dB stale | 0.01535 | -1 | 0 |

Rician使初始预测问题容易约55.1%，但五个非零残差epoch全部不如起点。

### 9.4 说明了什么

支持：

- Rayleigh是CSI失效严重的重要原因；
- Rician显著减小stale与current差异。

不支持：

- 只要换Rician，当前GRU就会自动有效；
- 当前teacher-prior residual能够利用增加的可预测性。

更可能的直接限制是：teacher GAT prior本身近常数，时间模型被要求重建一个缺少调度
辨识力的目标。

---

## 10. 对三个原始想法的逐项回答

### 想法1：后半段会不会掩盖CSI和prior问题

会，而且已经实证。会议binary动作全开，但两阶段规则仍达到greedy的91%--94%。

不过骨架并非不重要：它贡献最终SE的大约三分之二到四分之三；Rayleigh stale还会把骨架
Jaccard降到0.387。最准确表述是：

> 强物理后处理遮蔽了GAT prior坍缩，同时骨架错误与加入错误共同决定剩余CSI损失。

### 想法2：应先做无队列Shannon验证

完全正确。本轮已经补齐，并发现了队列结果中看不出的全开prior。以后任何新channel
prior或时间模型必须先通过channel-only闸门，再进入动态队列。

### 想法3：缺少LoS是否导致GRU无用

部分正确：Rayleigh使功率创新更大、stale误差更大、CSI调度损失更严重；Rician明显改善。

但它不是充分原因：Rician stale-only GRU仍然回退到零残差。当前更直接的问题是时间模型
的teacher target缺少辨识力。

---

## 11. 后续研究顺序怎样修正

此前的“停止所有时间建模”需要收窄为：

> 停止继续优化 Frozen-GAT teacher-logit reconstruction；不再给这个近常数target增加
> 更深GRU、attention或辅助loss。

仍然合理的最小后续路线是：

1. 将Rician 5 dB作为LoS主导敏感性模型，Rayleigh保留为会议一致性/严重衰落对照；
2. 新候选先预测具有物理意义且非坍缩的量，例如current log-power、direct/cross gain或
   true-CSI局部物理rate，而不是Frozen-GAT logit；
3. 预测CSI必须同时供LocalMax骨架和条件加入使用；
4. 先比较naive stale、current large-scale和简单解析相关预测；
5. 只有channel-only Shannon、动作集合和prediction error均改善，才进入动态队列；
6. 仍保持会议GAT空间编码器、一层永久链路时间状态和固定消息轮数，不展开复杂STGNN
   搜索。

下一模型在实现前还需明确：是预测完整direct/cross功率图，还是只预测一个链路rate。
本轮反事实提示只修骨架不够，因此完整但稀疏的节点/冲突边物理预测更有解释力。

---

## 12. 结论边界

本轮可以证明：

- 当前会议与代码都是纯Rayleigh；
- 加入LoS显著提高跨块功率可预测性；
- Frozen-GAT直接阈值动作在正式状态中全部全开；
- 两阶段物理规则遮蔽了该问题；
- 骨架和加入阶段均受CSI误差影响；
- Rician降低了stale难度，但没有让当前teacher-prior GRU成功。

本轮不能证明：

- 某个固定K=5 dB适用于全部UAV场景；
- pure LoS是完整真实A2A模型；
- 所有STGNN或信道预测器都无效；
- interference-free ceiling可由实际调度达到；
- 100个抽样快照构成严格容量或稳定域定理。

---

## 13. 文件与复现入口

- Rician/LoS移动信道：`simulator/mobile_block_fading.py`
- channel-only和阶段分解：
  `experiments/diagnose_channel_model_and_stage_masking.py`
- 正式结果：
  `results/diagnostics/channel_model_and_stage_masking_N10_5seeds_20260715.json`
- Rayleigh stale-only训练：
  `results/diagnostics/temporal_residual_stale_rayleigh_probe_N10_20260715.json`
- Rician stale-only训练：
  `results/diagnostics/temporal_residual_stale_ricianK5_probe_N10_20260715.json`

测试新增了Rician跨模型link identity配对和pure-LoS功率一致性检查。
