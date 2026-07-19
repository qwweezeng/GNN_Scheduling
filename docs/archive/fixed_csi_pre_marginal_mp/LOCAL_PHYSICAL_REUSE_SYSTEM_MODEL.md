# LocalPhysicalReuse-v2：期刊式 System Model 与算法定义

> **历史归档（2026-07-18）**：本文仅保留旧GAT-Assisted Join方案的物理公式来源。
> 它不是当前系统模型，也不得作为下一轮研究入口。

> **2026-07-16 结论勘误：** 本文保留 v2 公式，用于复现已完成的
> GAT-assisted/Pure 消融；但下文“当前推荐 GAT Expected-Rate”已被正确 N=20
> Rician checkpoint 下的同均值 Constant 归因，以及 scalar/set-conditioned 两次
> 训练闸门否决。当前队列调度保底主干是 `PurePhysicalReuseScheduler`：
> $Q_iC_i^{\rm solo}$ LocalMax 骨架 + 相同解析条件物理复用。完整证据和停止决定见
> `GAT_MWS_TOPOLOGY_SCALING_TUTORIAL.md`。

> 文档定位：这是当前最有前景模型的正式数学定义稿，写法尽量接近 IEEE 期刊的 System Model / Problem Formulation / Proposed Method。它不是已经排版完成的论文正文。凡是尚未验证的主张均显式标出。

## 1. 模型状态与版本结论

以下定义的是已完成评估的 **LocalPhysicalReuse-v2（GAT Expected-Rate）**，不是
当前推荐保底。它由三个部分组成：

1. **信道 GAT：** 从信道图中产生会议定义的 channel-only 软调度概率 $\widetilde\pi_i$；
2. **队列感知局部骨架：** 把 $\widetilde\pi_i$ 作为期刊扩展中的期望活动度近似，转换为具有速率单位的有效速率，再与瞬时队列相乘；
3. **条件空间复用：** 以真实骨架集合为条件，计算候选加入的局部 MWS 增益，并通过一次并行局部竞争补充复用链路。

与 v1 相比，v2 不再使用缺乏量纲解释的

$$
q_i^{\mathrm{rel}}\operatorname{softplus}(c_i).
$$

必须澄清术语：$\widetilde\pi_i$ 不是 `prior_i` 的缩写，而是会议论文中的软调度
概率；会议算法原本直接用
$\pi_i=\mathbb 1[\widetilde\pi_i\geq0.5]$ 得到 channel-only 二元动作。v2 不直接
执行该旧动作，而是新增一个接口近似：把 $\widetilde\pi_i$ 当作邻居期望活动度，用于
构造期望干扰和期望有效速率。这个转换具有明确物理计算过程，但
$\widetilde\pi_i$ 是否是校准良好的独立活动概率并没有由会议训练自动保证，必须通过
消融实验验证，不能把它写成已证明事实。

当前允许使用的 GAT checkpoint 是 10 dBm、Rician $K=5$ dB、N=20 密集场景训练的
会议结构模型。其 N=20 测试硬激活比例约 5.05%，硬容量 25.36；跨规模部署到 N=10
后硬激活比例约 9.34%，硬容量 12.32。两个后期 N=10 全发 checkpoint 已判定为训练
坍缩，不得继续用于 GAT-vs-Pure 或 CSI 鲁棒性结论。

## 2. UAV 网络与链路集合

考虑由 $N$ 个 UAV 组成的单信道网络。时隙 $t$ 的候选有向通信链路集合写为

$$
\mathcal L(t)=\{1,\ldots,L(t)\}.
$$

链路 $i$ 的发射端和接收端分别记为 $s_i$ 和 $d_i$。二值调度变量为

$$
a_i(t)\in\{0,1\},
$$

其中 $a_i(t)=1$ 表示链路 $i$ 在时隙 $t$ 发射。联合动作记为

$$
\mathbf a(t)=[a_1(t),\ldots,a_{L(t)}(t)]^\mathsf T.
$$

所有当前实验使用固定发射功率 $P=10\,\mathrm{dBm}$。固定信道正式基线使用 perfect CSI；第一轮移动风险诊断在同一真实瞬时服务信道上分别给调度器 latest、one-block stale、large-scale 和 distance-quantized large-scale CSI。旧的无物理标签 50-UAV 数据集没有用于当前正式实验。

## 3. 信道、干扰与可达速率

令

$$
g_{ji}(t)=P\left|h_{s_j,d_i}(t)\right|^2
$$

表示链路 $j$ 的发射端在链路 $i$ 的接收端产生的接收功率。于是 $g_{ii}$ 是链路 $i$ 的期望信号功率，$g_{ji},j\neq i$ 是有向交叉干扰功率。

给定联合动作 $\mathbf a(t)$，链路 $i$ 的 SINR 为

$$
\gamma_i(\mathbf a(t))=
\frac{g_{ii}(t)}
{\sigma^2+\sum_{j\neq i}a_j(t)g_{ji}(t)},
$$

其中 $\sigma^2$ 是接收噪声功率。引入实现损失对应的 SNR gap $\Gamma\geq1$，链路 $i$ 的瞬时可达速率定义为

$$
R_i(\mathbf a(t))=
a_i(t)B\log_2\!\left(1+\frac{\gamma_i(\mathbf a(t))}{\Gamma}\right),
$$

其中 $B$ 是系统带宽。这里的速率显式依赖整个发送集合，而不是单链路独立属性。

## 4. 有向冲突图

把每条通信链路映射为一个图节点，得到有向冲突图

$$
\mathcal G(t)=(\mathcal L(t),\mathcal E(t)).
$$

若链路 $j$ 的发送会对链路 $i$ 造成显著干扰，或两条链路共享不能同时收发的 UAV，则建立有向边

$$
(j,i)\in\mathcal E(t).
$$

方向 $j\rightarrow i$ 表示“$j$ 干扰 $i$”。物理增益计算保留该方向；局部胜者竞争使用其对称邻域

$$
\mathcal N_i=\{j:(i,j)\in\mathcal E\ \text{or}\ (j,i)\in\mathcal E\}.
$$

必须区分两个规模量：UAV 数量 $N$ 和冲突边数量 $E=|\mathcal E|$。在固定区域内增加 UAV 会同时增加候选链路数与图密度，因此实际复杂度主要由 $L+E$ 决定。

## 5. 队列演化与 QoS 状态

链路 $i$ 的队列积压为 $Q_i(t)$，新到达数据包数为 $A_i(t)$。若数据包大小为 $S_{\mathrm p}$，时隙长度为 $T_{\mathrm s}$，则可服务包数为

$$
\mu_i(t)=\frac{T_{\mathrm s}}{S_{\mathrm p}}R_i(\mathbf a(t)).
$$

队列按下式更新：

$$
Q_i(t+1)=\left[Q_i(t)-\mu_i(t)\right]^+ + A_i(t).
$$

较大的 $Q_i(t)$ 表示该链路有更紧急的发送需求。当前算法不保存更早时隙的队列记忆，因为 MWS/Lyapunov 思路要求调度器在每个时隙根据当前 $Q(t)$ 和当前信道作决策，而不是预测未来队列。

## 6. 每时隙 MWS 目标

经典 Max-Weight Scheduling 在每个时隙求解

$$
\mathbf a^{\star}(t)
=\arg\max_{\mathbf a\in\{0,1\}^{L(t)}}
\sum_{i=1}^{L(t)}Q_i(t)R_i(\mathbf a).
$$

记队列加权速率效用为

$$
U(\mathbf a;t)=\sum_i Q_i(t)R_i(\mathbf a).
$$

穷举动作空间需要检查 $2^{L}$ 个集合。集中式 MWS-Greedy 仍需要全局信道和多次完整效用计算，因而只作为小规模参考。LocalPhysicalReuse 的目标是用固定轮局部消息得到一个有效但不保证全局最优的近似动作。

必须严格说明：MWS 理论解释了为什么 $Q_iR_i$ 是正确的逐时隙目标，但当前近似策略并未自动继承精确 MWS 的吞吐最优性或稳定域定理。

## 7. 阶段一：会议 GAT 软调度概率及其期刊接口

### 7.1 输入与消息传递

会议版 channel-only GAT 读取信道冲突图。链路节点特征为本链路 SNR，边特征描述有向干扰强度。第 $\ell$ 层消息传递可概括为

$$
\mathbf h_i^{(\ell+1)}=
\mathop{\Vert}_{k=1}^{K}
\sigma\!\left(
\sum_{j\in\mathcal N_i^{\mathrm{in}}\cup\{i\}}
\alpha_{ji}^{(\ell,k)}
\left(\mathbf W_k\mathbf h_j^{(\ell)}+
\mathbf U_k\mathbf e_{ji}\right)
\right),
$$

其中 $\alpha_{ji}^{(\ell,k)}$ 是在接收节点 $i$ 的入邻居上归一化的注意力权重。经过 $T$ 层后，GAT 输出 channel-only logit $c_i^{\mathrm{ch}}$，对应软调度概率

$$
\widetilde\pi_i^{\mathrm{GAT}}
=\operatorname{sigmoid}(c_i^{\mathrm{ch}}).
$$

在原会议任务中，最终动作是

$$
\pi_i^{\mathrm{GAT}}
=\mathbb 1[\widetilde\pi_i^{\mathrm{GAT}}\geq0.5].
$$

因此 $\widetilde\pi_i^{\mathrm{GAT}}\approx1$ 表示强烈倾向发送，而不是“保守地假设
邻居可能发送”。若所有链路均接近 1，就是全发坍缩，必须拒绝该 checkpoint。

### 7.2 从 GAT logit 到物理有效速率

v2 不直接执行会议的硬动作，也不把 logit 当成速率。它采用一个新增的
GAT-guided expected-interference 接口：用邻居的软调度概率近似邻居活动度，构造
局部期望干扰

$$
\widetilde I_i=
\sum_{j:(j,i)\in\mathcal E}
\widetilde\pi_j^{\mathrm{GAT}}g_{ji},
$$

并得到具有 bit/s 量纲的 GAT 条件有效速率

$$
\widetilde C_i=
B\log_2\!\left(
1+\frac{g_{ii}}
{\Gamma(\sigma^2+\widetilde I_i)}
\right).
$$

因此 GAT 的链路区分度通过 $\widetilde\pi_j^{\mathrm{GAT}}$ 诱导的空间干扰图案进入
队列阶段，后续模块使用具有速率量纲的 $\widetilde C_i$。需要注意，联合调度动作之间
高度耦合，把各链路软概率直接放入期望干扰并不等价于精确计算联合动作分布；它是
LocalPhysicalReuse-v2 的低交互近似，也是 GAT-vs-Pure 消融需要检验的关键接口。

在一个信道相干块内，$\widetilde\pi_i^{\mathrm{GAT}}$ 和 $\widetilde C_i$ 只计算一次；
每时隙只更新快速变化的队列权重。

## 8. 阶段二第一步：队列感知局部骨架

链路 $i$ 的局部竞争权重定义为

$$
W_i(t)=Q_i(t)\widetilde C_i(t).
$$

链路 $i$ 在队列非空且不低于所有冲突邻居时成为局部胜者：

$$
a_i^0(t)=
\mathbb 1\!\left[
Q_i(t)>0,
W_i(t)>W_j(t),\ \forall j\in\mathcal N_i
\right].
$$

相等时使用固定链路 ID 打破平局。所有局部胜者并行构成真实二值骨架

$$
\mathcal A_0(t)=\{i:a_i^0(t)=1\}.
$$

$\mathcal A_0$ 不是全网唯一 anchor。互不相邻的多个区域会同时产生自己的局部胜者，因此该过程对应“多个分布式小社会同时选举”，而不是集中式选头。

## 9. 阶段二第二步：条件空间复用

### 9.1 骨架条件下的局部速率

在骨架 $\mathcal A_0$ 下，链路 $k$ 观察到的冲突图内干扰为

$$
I_k^0=
\sum_{j\in\mathcal A_0:(j,k)\in\mathcal E}g_{jk}.
$$

其局部估计速率为

$$
C_k(\mathcal A_0)=
B\log_2\!\left(
1+\frac{g_{kk}}{\Gamma(\sigma^2+I_k^0)}
\right).
$$

### 9.2 候选加入收益

对任意 $i\notin\mathcal A_0$ 且 $Q_i>0$，若 $i$ 加入，自己的队列加权收益为

$$
B_i(\mathcal A_0)=
Q_i C_i(\mathcal A_0\cup\{i\}).
$$

链路 $i$ 会降低其当前活动受害邻居的速率，因此定义局部损害

$$
H_i(\mathcal A_0)=
\sum_{k\in\mathcal A_0:(i,k)\in\mathcal E}
Q_k\left[
C_k(\mathcal A_0)-C_k(\mathcal A_0\cup\{i\})
\right].
$$

条件局部边际收益为

$$
\Delta_i^{\mathrm{loc}}(\mathcal A_0)
=B_i(\mathcal A_0)-H_i(\mathcal A_0).
$$

这一步明确回答“在当前已经准备发送的链路集合下，我加入是否有利”，而不是让所有链路独立输出发送概率。

### 9.3 一轮并行加入

正收益候选还要与相互冲突的其他候选比较。并行加入集合定义为

$$
\mathcal J=
\left\{
i:\Delta_i^{\mathrm{loc}}>0,
\Delta_i^{\mathrm{loc}}>\Delta_j^{\mathrm{loc}},
\forall j\in\mathcal N_i\cap\mathcal C^+
\right\},
$$

其中 $\mathcal C^+$ 是所有正收益候选集合，平局仍由链路 ID 决定。最终调度集合为

$$
\mathcal A=\mathcal A_0\cup\mathcal J.
$$

所有候选只计算一次条件收益并同时加入，不随最终激活链路数重复执行。因此它不是顺序 greedy，也不是 Best-K 搜索。

## 10. 分布式实现所需信息

每条链路只需维护或交换以下局部信息：

1. 本链路队列 $Q_i$、直接增益 $g_{ii}$ 和固定 ID；
2. 冲突邻居的 channel-only 活动先验 $p_j^{\mathrm{ch}}$；
3. 冲突边有向增益 $g_{ji}$；
4. 骨架动作 $a_j^0$、邻居队列和局部条件收益。

逻辑消息流程为：

1. 每个信道块执行固定 $T$ 层 GAT 消息传递并缓存 $p_i^{\mathrm{ch}}$；
2. 每个时隙交换 $W_i$，并行形成 $\mathcal A_0$；
3. 交换骨架动作与局部物理量，计算 $\Delta_i^{\mathrm{loc}}$；
4. 交换候选收益，一轮局部竞争形成 $\mathcal J$。

当前代码用张量集中模拟上述并行运算，但公式只依赖冲突邻居。真正的 MAC 消息格式、同步误差与控制开销尚未实现，论文中不能把“逻辑分布式”直接写成“已经实网部署”。

这里的训练与部署应明确区分：会议版 channel-only GAT 在离线阶段集中训练，训练完成后所有链路共享同一组网络参数；部署时不按时隙重新训练。每个信道相干块执行一次 $T$ 层邻居消息传递，块内缓存 $p_i^{\mathrm{ch}}$ 和 $\widetilde C_i$；快速队列阶段则每个时隙按当前 $Q_i(t)$ 执行固定轮局部交换。

$T$ 层 GAT 在算法图上对应 $T$ 轮邻居隐藏状态传播，不表示每个 UAV 把全网状态上传中心。不过，当前 PyTorch 程序仍在单机内持有完整 `NetworkState` 并用张量实现并行运算，所以目前只能称为**集中训练、逻辑分布式执行**。真实空口中的广播聚合、消息丢失、异步和控制包比特数仍需单独验证。

## 11. 结构性质

### 11.1 队列尺度不变性

若所有队列共同放大为 $Q_i'=\kappa Q_i,\kappa>0$，则

$$
W_i'=\kappa W_i,
\qquad
\Delta_i^{\mathrm{loc}\prime}=\kappa\Delta_i^{\mathrm{loc}}.
$$

局部排序、正负号及最终动作均不变。因而 v2 不需要 `log(1+Q)` 或逐节点 `q_rel` 归一化。

### 11.2 分量可组合性

若冲突图由多个互不连接的分量组成，每个分量独立形成骨架和加入集合，全网结果等于各分量结果的并集。算法不会强行选举一个全局 anchor。

### 11.3 Work-conserving 性质

只要某个连通分量内至少有一条非空队列链路，有限集合中必然存在一个最大 $W_i$，因此该分量至少产生一个骨架胜者。这里 work-conserving 的含义是“有可服务业务时不输出全零”，不等价于全局效用最优。

### 11.4 条件性与确定性

复用决策显式以真实集合 $\mathcal A_0$ 为条件，并输出确定的 0/1 动作；不存在采样碰运气、temperature 或 STE 阈值坍缩。

## 12. 复杂度与规模边界

设 GAT 隐藏维数为 $d$、消息层数为 $T$。每个信道块的 GAT 计算复杂度约为

$$
\mathcal O(TEd),
$$

每个时隙的骨架比较、局部干扰、条件增益和并行候选竞争均可按稀疏边实现为

$$
\mathcal O(L+E),
$$

存储复杂度同样为 $\mathcal O(L+E)$。当前实现已从稠密 $L\times L$ 增益矩阵改成等价的边稀疏计算。

但 $E$ 本身可能接近 $L^2$。在当前固定 1 km 区域的 50-UAV 样本中，约有 835 条候选链路和 363456 条有向冲突边。CPU 基准为：

- 首个信道块：约 239 ms；
- 相同信道块内暖启动时隙：平均约 9.2 ms；
- 纯物理骨架暖时隙：约 9.0 ms。

因此 v2 已证明能运行到 50 UAV，但尚未满足 1-ms 时隙的单 CPU 实时性。数十万到百万级冲突边在密集链路图中是预期现象；后续重点不是武断判断“图太密”，而是区分 GPU/并行实现收益、真实控制消息开销和可聚合的端点广播。

## 13. 当前实验结论

1000 时隙、三个共享随机轨迹下的服务满足率为：

$$
\eta_{\mathrm{srv}}=
\frac{\sum_t\sum_i D_i^{\mathrm{actual}}(t)}
{\sum_t\sum_i A_i(t)}.
$$

该数值表示“总到达数据包中有多少在仿真窗口内被实际发送”，不是“本算法效用除以 MWS-Greedy”。MWS-Greedy 在相同信道、到达轨迹和物理参数下作为单独一列。接近 1 表示服务能力覆盖当前负载；明显小于 1 通常表示过载且队列持续增长。

| 规模 / 负载 | v1：GAT-logit 乘子 | v2：GAT Expected-Rate | 纯物理骨架 | MWS-Greedy |
|---|---:|---:|---:|---:|
| 10 UAV / Lambda=2 | 0.991 | 0.987 | 0.991 | 0.992 |
| 10 UAV / Lambda=4 | 0.566 | **0.631** | 0.615 | 0.655 |
| 20 UAV / Lambda=2 | 0.976 | 0.972 | **0.981** | 0.988 |
| 20 UAV / Lambda=4 | 0.632 | 0.638 | **0.654** | 0.736 |
| 20 UAV / Lambda=8 | 0.340 | **0.387** | 0.367 | 0.425 |
| 50 UAV / Lambda=1 | 0.999 | 0.999 | 0.999 | 未运行 |
| 50 UAV / Lambda=2 | 0.960 | 0.965 | **0.970** | 未运行 |
| 50 UAV / Lambda=4 | **0.634** | 0.627 | 0.634 | 未运行 |
| 50 UAV / Lambda=8 | 0.353 | **0.380** | 0.361 | 未运行 |

例如 20 UAV、$\Lambda=8$ 时，全网 offered load 约为 96 Mbit/s。v2 的服务满足率为 0.3865（约 37.23 Mbit/s），MWS-Greedy 为 0.4254（约 40.97 Mbit/s），两者均不稳定。若只计算此过载点下相对 MWS 的实际服务能力，v2 约为 90.9%，而不是 38.7%。

v2 并非在所有负载下逐点最优，但它同时满足：

- 中负载可稳定；
- 高负载明显优于原 v1；
- 50-UAV 不发生性能断崖；
- GAT 曾被赋予明确的 expected-rate 物理接口，但后续 Constant 归因未证明该接口
  的图特异性价值；
- 不需要重新训练或模仿 MWS-Greedy。

因此这一段只能作为 v2 的设计动机，不能再作为当前推荐结论。当前正式保底是纯物理
$Q_iC_i^{\mathrm{solo}}$ 骨架；v2 GAT-assisted 作为必要消融和前作接口保留。

## 14. 当前不能宣称的内容

1. 不能宣称 v2 已证明 throughput-optimal；
2. 第一轮移动/block-fading 只有 10 UAV、300 时隙和三个 seed，只能作为 CSI 风险诊断，不能冒充最终论文统计；
3. 不能宣称 50-UAV 已满足 1-ms 实时部署；
4. 不能把 perfect-CSI 下的精确信道增益当成不完美 CSI 下自动可得；
5. 不能把当前逻辑分布式张量实现等同于已经完成 MAC 协议；
6. 不能宣称会议版 GAT 概率已经严格校准，后续仍需做概率校准和跨信道模型检查。

## 15. 下一步研究接口

模型闸门已经停止后，剩余工作优先用于论文保底和证据边界，不自动启动新预测模型：

1. **block-fading / UAV 移动：** 已用永久有向 UAV 对定义 link identity；不可用链路
   保留队列并屏蔽动作。当前只保留已有结果作为 LAIN 场景敏感性，不追加高风险模型；
2. **不完美 CSI：** latest、过期、large-scale 与量化 large-scale 的第一轮共享轨迹
   诊断已完成；过期损失明显，但尚未证明预测器必要，截稿前不再自动开启 GRU/STGNN；
3. **复杂度与协议：** 百万级冲突边在密集网络中是预期现象。区分共享节点约束、强干扰边和弱干扰边只用于核算 CPU/消息开销及广播聚合可能性，不以强行稀疏化为目标。

任何新增模块都必须对应上述一种明确误差来源；若不能在相同随机轨迹上改善性能、鲁棒性或复杂度，则不进入正式模型。
