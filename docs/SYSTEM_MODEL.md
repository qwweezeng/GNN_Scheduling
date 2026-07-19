# TNSE投稿版系统模型与算法说明

> **文档定位。** 这是论文第2--4章和投稿代码之间的唯一方法入口。它说明“研究什么、
> 每个量是什么、代码实际上做什么、论文可以声称什么”。历史尝试与已经停止的路线只保留
> 在`docs/archive/`，不能覆盖本文档中的当前定义。

## 0. 先用一句话说明整篇论文

本文研究低空密集无人机网络中的**有向链路调度**：第3章先在所有链路始终有数据的
物理层条件下，用Channel-GAT学习具有全局干扰意识的稀疏并发倾向；第4章再把该倾向
转换成当前队列状态下的初始竞争骨架，并用固定轮数的、具有明确SINR边际意义的消息传递
修正骨架，最终在空间复用能力与积压链路服务优先级之间取得平衡。

论文主线统一为：

1. **第2章：系统模型与问题定义**；
2. **第3章：满缓存物理层总速率调度（Channel-GAT）**；
3. **第4章：GAT-assisted队列感知边际效用消息传递（MarginalMP）**；
4. **可选支线：CG-CSI**，仅作为信道预测敏感性案例，默认关闭。

这里必须提前说明证据边界：`GAT-Assisted MarginalMP`是为了保持第3章与第4章的结构
连续性而选定的**论文主方案**；`Direct-Link MarginalMP`是去掉GAT的必要消融。现有开发
实验没有证明GAT初始化稳定优于Direct-Link初始化，因此不能把整个MarginalMP的收益
因果归因给GAT，也不能隐去Direct-Link这一消融。可以准确地称为“所提出的GAT-assisted
框架”，但只有正式配对实验显示显著差异后，才能进一步声称“GAT带来性能增益”。

---

## 1. 网络、链路与控制时标

### 1.1 无人机与候选有向链路

无人机集合记为

$$
\mathcal U=\{1,\ldots,N\}.
$$

一条调度对象不是单个无人机，而是有方向的发送--接收对

$$
i=(s_i,r_i),\qquad s_i,r_i\in\mathcal U,\quad s_i\ne r_i.
$$

论文默认区域为$1000\,{\rm m}\times1000\,{\rm m}$，高度范围为$50$--$150$ m，初始
无人机间距至少20 m。载波频率为2.4 GHz，带宽为5 MHz，所有发送端固定使用10 dBm
功率；本文不研究功率控制。

静态快照中，候选链路由当前接收SNR门限产生：

$$
\mathcal L=\{(u,v):u\ne v,\;{P_{uv}}/{N_0}\ge\beta\},
$$

默认$\beta=5$ dB。链路$i$既是GAT图中的一个节点，也是一个独立业务队列。

### 1.2 为什么移动仿真必须保留link identity

若每个信道块重新生成并排序链路，第7号队列在下一块可能会错误地映射到另一对无人机。
因此移动仿真在第0块建立一个按$(s_i,r_i)$字典序排列的**persistent link catalog**：

- catalog中的每一项始终表示同一发送--接收对；
- 每条catalog链路拥有独立且持续存在的队列；
- 后续某块中链路低于大尺度SNR门限时只标记为`unavailable`，该块动作强制为0；
- 链路恢复可用时继续服务原来的队列，不创建新队列，也不交换队列身份。

这就是本文所说的“link identity一致”。它保证不同CSI方法共享完全相同的流、运动轨迹和
真实信道。

### 1.3 时隙与信道块

队列每$\tau=1$ ms更新一次。信道块包含$S_b$个时隙，因此

$$
T_b=S_b\tau.
$$

例如`slots_per_block=5`表示5 ms一个信道块。调度器仍然每个时隙读取当前队列并作出
动作；信道、冲突图和Channel-GAT信道先验只在块边界更新。

### 1.4 有向冲突图如何产生

冲突图中的边$j\rightarrow i$表示“链路$j$的发送会显著影响链路$i$的接收”，方向不能
随意对称化。对观测功率矩阵$\widehat P$，代码在以下任一条件成立时建立该边：

1. 两条链路共享任意发送端或接收端UAV；
2. $j$单独造成的干扰足以使$i$低于链路SNR门限$\beta$：

$$
\widehat P_{s_jr_i}\ge
\frac{\widehat P_{s_ir_i}}{\beta}-N_0.
$$

边特征由干扰功率与上述可容忍干扰余量之比经过$\log_{10}(1+x)$和图内归一化得到；共享
端点边用很大的原始权重标记。该图服务于GAT聚合和局部仲裁，物理干扰边本身不是硬互斥
约束，实际服务用连续SINR结算。

**共享UAV约束的实现审计。** 当前代码也把共享发送端/接收端的链路写成冲突边，因此
LocalMax骨架不会同时选择相邻的共享端点链路；但MarginalMP后端只在“本轮同时提出翻转
的候选”之间仲裁，尚未对最终活动集合执行严格的single-radio/half-duplex可行性投影。
若论文假设每架UAV只有一个射频链并且不能同时收发，下一项正式仿真任务必须先补充并测试
这个hard constraint。若不补，则系统模型只能明确假设链路级并发传输能力，不能宣称代码
已经严格保证每架UAV至多参与一条活动链路。

噪声功率由噪声功率谱密度与带宽换算：

$$
N_0=10^{N_{\rm PSD}/10}\times10^{-3}\times B,
$$

其中$N_{\rm PSD}$以dBm/Hz表示，正式配置为-169 dBm/Hz。

---

## 2. 空对空LoS主导Rician信道

### 2.1 大尺度衰落

代码采用距离幂律幅度模型

$$
\ell_{uv}[b]=\sqrt{\lambda_0}\,d_{uv}[b]^{-\xi/2},\qquad
\lambda_0=\left(\frac{c}{4\pi f_c}\right)^2,
$$

其中路径损耗指数$\xi=2.5$。因此只含大尺度衰落时的接收功率为

$$
P^{\rm LS}_{uv}[b]=P_{\rm tx}\lambda_0d_{uv}[b]^{-\xi}.
$$

当前正式模型没有额外叠加对数正态阴影衰落、天线方向图、机身遮挡或双径地面反射；这些
是模型边界，不应在论文中暗示已经模拟。

### 2.2 Rician小尺度衰落

复信道写为

$$
h_{uv}[b]=\ell_{uv}[b]\left[
\sqrt{\frac{K}{K+1}}e^{-j2\pi d_{uv}[b]/\lambda_c}
+\sqrt{\frac{1}{K+1}}g_{uv}[b]\right],
$$

其中$g_{uv}[b]\sim\mathcal{CN}(0,1)$为散射分量。真实接收功率为

$$
P_{uv}[b]=P_{\rm tx}|h_{uv}[b]|^2.
$$

上述归一化保证$\mathbb E[|h_{uv}|^2]=\ell_{uv}^2$，因此改变$K$改变的是LoS与散射功率
比例，而不是偷偷改变平均接收功率。

### 2.3 $K=5$ dB是否太小

正式设置为

$$
K_{\rm dB}=5\ {\rm dB},\qquad K=10^{5/10}\approx3.162.
$$

这意味着归一化小尺度功率中，LoS项占

$$
\frac{K}{K+1}\approx0.760,
$$

散射项占约0.240。因此它仍然是**LoS主导**，不是Rayleigh/NLoS信道，只是LoS并未强到
完全压倒反射和散射。

该数值有现实依据：一项低空校园场景的同步A2A/A2G测量报告A2A链路的Rician因子
中位数约5 dB，而A2G约13 dB；作者将A2A较低的$K$部分归因于较强地面反射提高了其他
多径分量功率。测量结论具有场景依赖性，但足以说明5 dB是合理、偏保守的A2A主设置，
不是明显错误的参数。来源：
[A Simultaneous Low-Altitude A2A/A2G Channel Measurement and Analysis](https://www.mdpi.com/2504-446X/10/2/104)。

因此投稿版不应仅为了让stale CSI损失变小而把$K$提高到10或15 dB。$K$越大，当前模型
越接近仅随距离平滑变化的pure-LoS功率，信道预测和过期CSI问题通常也越容易，这会使
鲁棒性实验变得更“好看”却更不保守。建议口径是：

- **主结果固定$K=5$ dB**，所有方法在相同真实信道上公平比较；
- 如版面允许，在CSI敏感性小节补充$K\in\{5,10,15,+\infty\}$的轻量敏感性检查；
- 该检查只回答结论是否依赖LoS强度，不用来重新选择一个最有利的主参数；
- Rayleigh数据只保留作历史复现，不进入正式A2A主结果。

---

## 3. 无人机运动与相关block fading

### 3.1 当前采用的运动模型

第0块中，无人机位置在三维区域内均匀采样并满足最小间距。每架无人机以相同速度$v$
水平直线飞行，但初始航向$\phi_u$独立均匀采样：

$$
\mathbf v_u=v[\cos\phi_u,\sin\phi_u,0]^\mathsf T,
$$

$$
\mathbf p_u[b+1]=\mathbf p_u[b]+T_b\mathbf v_u[b].
$$

当飞行轨迹越过水平区域边界时采用镜面反射：越界坐标关于边界折返，对应速度分量反号。
初始垂直速度为0，因此当前实验中高度保持不变。

这个模型应准确称为**有边界反射的恒速直线飞行模型**。它适合受控地扫描速度、块长和
CSI年龄，但不等价于真实编队任务中的随机转弯、加速度、垂直机动、避碰或任务航迹。

### 3.2 块间相关散射分量

在每个块内部，位置、LoS相位、散射分量和整个功率矩阵保持不变；到达下一块时统一更新。
散射分量按复Gauss--Markov过程演化：

$$
g_{uv}[b]=\rho g_{uv}[b-1]+\sqrt{1-\rho^2}\,\epsilon_{uv}[b],
\quad \epsilon_{uv}[b]\sim\mathcal{CN}(0,1),
$$

$$
\rho=J_0(2\pi f_DT_b),\qquad f_D=\frac{vf_c}{c}.
$$

因此当前模型确实是**时间相关的block-fading模型**，不是每个时隙独立抽样，也不是每块
独立Rayleigh/Rician抽样。`v=0`时$\rho=1$，散射实现保持不变；速度或块长增大时，相邻
块相关性通常降低。

当前$f_D=vf_c/c$采用单架无人机速度形成的代表性最大Doppler，而不是针对每个UAV pair
根据相对径向速度单独计算。这是一个明确的简化假设，正式写作不能称为精确轨道级Doppler
模型。

### 3.3 四种调度器可见的CSI

所有方法最终都用当前真实功率$P[b]$结算，只改变调度器观察到的矩阵$\widehat P[b]$：

1. **latest CSI**：$\widehat P[b]=P[b]$；
2. **one-block stale CSI**：$\widehat P[b]=P[b-1]$，第0块退化为当前CSI；
3. **current large-scale CSI**：$\widehat P[b]=P^{\rm LS}[b]$，知道当前几何但不知道
   当前Rician小尺度实现；
4. **$q$-bit quantized large-scale CSI**：先把距离在可行距离区间均匀量化为$2^q$级，
   再重构$P^{\rm LS}$，正式设置$q=4$。

因此“4-bit CSI”不是直接量化复信道I/Q，也不是量化瞬时SINR，而是**量化距离所重构的
大尺度功率**。

当前移动代码用**当前真实大尺度功率**判断catalog链路在本块是否可用，然后只在可用链路
上比较不同观测CSI。这意味着实验研究的是“已完成链路发现后的信道数值误差”，没有同时
模拟stale链路发现或控制面遗漏。该假设应在图注中说明。

### 3.4 静态数据与移动trace的差别

静态训练/测试集中的每个网络快照独立生成位置和Rician散射实现；它没有跨样本时间身份，
适合训练第3章的瞬时Channel-GAT。移动trace则先固定persistent link catalog，再让位置、
LoS相位和相关散射分量按块演化，适合队列与CSI-age实验。不能把静态数据集中相邻样本
解释为同一条链路的时间序列。

---

## 4. 连续物理干扰下的调度与服务

### 4.1 二元并发动作

时隙$t$的动作是

$$
\mathbf a(t)=[a_1(t),\ldots,a_L(t)]^\mathsf T,\qquad a_i(t)\in\{0,1\}.
$$

给定当前真实功率矩阵，链路$i$的SINR为

$$
\gamma_i(\mathbf a,H)=
\frac{P_{s_i r_i}}
{N_0+\sum_{j\ne i}a_jP_{s_jr_i}},
$$

谱效率为

$$
R_i(\mathbf a,H)=a_i\log_2\left(1+\frac{\gamma_i}{\Gamma}\right)
\quad [{\rm bit/s/Hz}].
$$

$a_i$写在对数外确保未激活链路服务为0。冲突图只决定哪些链路必须交换消息和哪些候选
不能在同一仲裁轮同时修改动作；最终速率始终由上面的连续SINR结算，不能把一条冲突边
解释为“二者绝对不能同时发送”。

### 4.2 为什么第3章不是标准MWIS

标准最大权独立集（MWIS）先给每个节点一个固定权重，再从硬冲突图中选互不相邻节点。
本文物理模型中，$R_i$取决于整个活动集合$\mathbf a$，加入一个链路会连续降低多个接收端
的速率；因此节点权重并非固定，弱干扰链路也可能同时发送。

第3章的准确名称是：

> **连续SINR物理干扰下的满缓存总速率链路调度**

或英文`full-buffer physical-layer sum-rate link scheduling under continuous
SINR interference`。只有把干扰阈值化成硬冲突图后，才得到MWIS近似；DistGCN等MWIS
方法在本文中属于这种抽象下的baseline。

### 4.3 bit、packet与“流体服务”之间的关系

物理层先产生bit服务能力。一个时隙内链路$i$的容量为

$$
B\tau R_i(t)\quad [{\rm bit/slot}].
$$

代码令每个数据包大小$L_p=12000$ bit（1500 byte），因此换算为

$$
S_i(t)=\frac{B\tau R_i(t)}{L_p}\quad[{\rm packet\ equivalents/slot}].
$$

队列内部用浮点数记录packet equivalents，实际服务为

$$
D_i(t)=\min\{Q_i(t),S_i(t)\}.
$$

这允许一个时隙发送0.3个包的“工作量”，但不代表仿真把0.3个真实包记为成功送达。代码
同时维护FIFO整数包队列和每个队首包的剩余工作量：只有累计完成整个12000 bit后，才增加
`delivered_packet_count`并记录该包时延。因此：

- `physical service`衡量完成的bit/packet-equivalent工作量；
- `goodput`只统计完整送达的数据包；
- packet delay只对已经完整送达的包定义；
- 到达发生在本时隙服务之后，新到包最早从下一时隙开始发送。

---

## 5. 异质业务到达与队列演化

### 5.1 当前异质到达模型

$\Lambda$表示**全网所有persistent links合计的平均到达包数/时隙**，不是每条链路都以
$\Lambda$到达。每个Monte Carlo seed固定一组hotspot链路：

- 20%的链路被选为热点，合计承载70%的到达业务；
- 其余80%的链路合计承载30%的业务；
- 每条链路的时隙到达独立服从Poisson分布。

令热点集合为$\mathcal H$，则

$$
\lambda_i=
\begin{cases}
0.7\Lambda/|\mathcal H|, & i\in\mathcal H,\\
0.3\Lambda/(L-|\mathcal H|), & i\notin\mathcal H.
\end{cases}
$$

并有$A_i(t)\sim\operatorname{Poisson}(\lambda_i)$。配对实验对所有算法复用相同的热点集合、
到达轨迹、真实信道和CSI观测，以免把随机差异误判为算法收益。

### 5.2 队列时序

每个时隙按以下顺序执行：

```text
读取块级观测CSI和时隙开始时队列 Q(t)
    -> 调度器产生二元动作 a(t)
    -> 当前真实信道计算R(t)
    -> 服务 min{Q(t), S(t)}
    -> 本时隙新包到达
    -> 得到 Q(t+1)
```

无限缓存、无deadline时的队列更新为

$$
Q_i(t+1)=\max\{Q_i(t)-D_i(t),0\}+A_i(t).
$$

正式主实验默认无限缓存且无deadline，避免丢包策略或截止期机制改变核心问题。代码支持
有限缓存和deadline，但只有在论文明确新增相应实验时才能启用。

### 5.3 “queue-aware”与“QoS-aware”的准确口径

第4章逐时隙采用最大权思想：

$$
U(\mathbf a;t)=\sum_iQ_i(t)R_i(\mathbf a,H(t)).
$$

队列长度是拥塞和等待紧迫性的代理，因此该方法可以称为`queue-aware`或
`QoS-oriented queue-aware scheduling`，并用backlog、goodput和packet delay评价。

但是仅输入队列长度、且没有显式deadline、可靠性约束、优先级类别或确定性时延保证时，
不宜直接宣称已经提供“严格QoS保证”。此外，动态轨迹上平均$\sum QR$不是公平的最终性能
指标：服务差的方法会积累更大的$Q$，反而可能得到更大的数值。正式动态指标应是goodput、
平均backlog、完成率和在完成率足够高时的完整包时延。

---

## 6. 第3章：Channel-GAT满缓存物理层调度

### 6.1 图输入、输出和物理含义

图中一个节点代表一条候选有向链路。节点输入是直达SNR，边特征描述一条发送链路对另一
接收端造成的有向干扰强度。四层多头GAT通过冲突邻域聚合，输出logit $z_i$和

$$
\pi_i=\sigma(z_i)\in(0,1).
$$

$\pi_i$沿用会议论文中的“调度概率”符号：在满缓存信道优化目标下，它表示链路的连续
激活倾向。部署到第3章二元调度时使用

$$
a_i=\mathbb 1[\pi_i\ge0.5].
$$

训练时用连续$\pi$构造可微代理：

$$
\widetilde R_i(\boldsymbol\pi,H)=
\pi_i\log_2\left(1+
\frac{P_{s_ir_i}}
{\Gamma\left(N_0+\sum_{j\ne i}\pi_jP_{s_jr_i}\right)}
\right),
$$

$$
\min_\theta-\sum_i\widetilde R_i(\boldsymbol\pi_\theta,H).
$$

这一目标不包含队列，回答的是“如果所有链路一直有数据，哪些链路并发能提高总物理层
速率”。正式训练必须审计连续$\pi$、二元激活比例、全发效用与模型效用，防止全0/全1
坍缩。

### 6.2 集中训练、逻辑分布式执行

离线训练可以批量读取完整网络快照并由全网总速率提供损失。在线推理时，每层GAT只聚合
冲突邻居消息，同一组参数在所有链路复用。这属于**集中训练、逻辑分布式执行**。

“逻辑分布式”不等于已经实现完整MAC协议。论文仍需单独报告每层广播次数、payload、
链路身份和CSI获得开销。

---

## 7. 第4章：GAT-Assisted MarginalMP队列调度

### 7.1 为什么第4章仍使用第3章GAT

Direct-Link初始化只知道链路独占信道时的速率

$$
C_i^{\rm solo}=\log_2\left(1+\frac{P_{s_ir_i}}{\Gamma N_0}\right),
$$

它看不到“邻居也可能发送”时的干扰。第3章GAT的$\pi_j$编码了在满缓存物理优化中邻居
$j$的激活倾向，因此可构造一个具有干扰意识的链路能力先验。对代码中冲突邻域
$\mathcal N_i$，定义

$$
\widehat I_i^{\rm GAT}=
\sum_{j\in\mathcal N_i}\pi_jP_{s_jr_i},
$$

$$
C_i^{\rm GAT}=\log_2\left(1+
\frac{P_{s_ir_i}}{\Gamma(N_0+\widehat I_i^{\rm GAT})}
\right).
$$

随后把真实队列紧迫性显式乘回：

$$
w_i^{\rm GAT}(t)=Q_i(t)C_i^{\rm GAT}(t).
$$

这个接口的直观含义是：GAT不替队列作最终决定，而是告诉骨架阶段“在邻居有不同发送
倾向时，这条链路大致还有多少竞争能力”；队列$Q_i$再决定此刻服务紧迫性。

必须诚实限定：会议$\pi$是总速率代理目标学出的连续调度倾向，并未通过概率校准实验
证明它等于邻居真实发送概率。因此$\widehat I_i^{\rm GAT}$应称为**GAT-induced expected-
interference score（GAT诱导的期望干扰分数）**，不能宣称为统计上无偏的干扰估计。

### 7.2 Part I：一轮LocalMax形成非空骨架

每条非空链路把$w_i^{\rm GAT}$与冲突邻居比较；局部最大者进入初始活动集合
$\mathcal A^{(0)}$，链路ID用于确定性平局消解。该步骤保证初始骨架非空且稀疏。

去掉GAT的必要消融使用

$$
w_i^{\rm direct}(t)=Q_i(t)C_i^{\rm solo}(t),
$$

其余流程完全相同，名称为`Direct-Link MarginalMP`。比较二者才能回答GAT先验本身是否
有贡献。

### 7.3 Part II：条件加入/退出边际

骨架不是最终调度。对当前活动集合$\mathcal A^{(k)}$，每条链路只提出翻转自身状态。
若$i$未活动：

$$
\Delta_i^+=U(\mathcal A^{(k)}\cup\{i\})-U(\mathcal A^{(k)}).
$$

若$i$已活动：

$$
\Delta_i^-=U(\mathcal A^{(k)}\setminus\{i\})-U(\mathcal A^{(k)}).
$$

该边际同时包括：

1. 链路$i$加入后获得的服务，或退出后失去的服务；
2. 它对当前所有活动接收端造成或解除的干扰外部性。

因此边际不是独立链路速率，也不是神经网络猜出的分数，而是由当前观测功率矩阵和当前
联合活动集合直接计算的物理量。

### 7.4 并行仲裁与固定$K$轮

所有$\Delta_i>0$的链路提出翻转，冲突邻域中只保留最大正边际提案，胜者集合
$\mathcal W^{(k)}$并行翻转：

$$
\mathcal A^{(k+1)}=\mathcal A^{(k)}\triangle\mathcal W^{(k)}.
$$

正式参数固定$K=4$。论文正文可以称为$K$轮MarginalMP，并在参数表给出$K=4$；轮数
曲线只作为可选消融或附录，不必占用主图。$K$是受限通信/计算预算，不是全局最优保证。

当前精确边际对活动接收端求和。部署时需要这些接收端回报速率变化；仿真中向量化的
$L\times|\mathcal A|$计算不能被写成“零通信”或“只需本链路CSI”。

### 7.5 完整算法

```text
输入：当前观测功率矩阵、冲突图、队列Q(t)、冻结Channel-GAT
块级：GAT输出连续pi，并缓存GAT-induced expected-rate C^GAT
时隙级：
  1. 计算 w_i = Q_i(t) C_i^GAT
  2. 一轮LocalMax得到初始骨架 A(0)
  3. 重复 k=0,...,K-1
     a. 基于当前A(k)计算每条链路精确加入/退出边际
     b. 广播正边际提案
     c. 冲突邻域保留最大正提案
     d. 胜者并行翻转
  4. 输出二元动作 A(K)
  5. 当前真实信道结算服务并更新队列
```

---

## 8. CSI不完美时究竟在验证谁

在latest/stale/large-scale/quantized实验中，Channel-GAT的输入、GAT诱导先验、LocalMax
骨架和后续物理边际都基于同一观测CSI；最终服务永远用当前真实CSI结算。

论文可以报告“所提出GAT-assisted框架在不完美CSI下的性能和退化”。但若要进一步说
“GAT带来了鲁棒性”，必须在同一轨迹上比较：

$$
\text{GAT-Assisted MarginalMP}
\quad\text{vs.}\quad
\text{Direct-Link MarginalMP},
$$

并考察两者从latest到stale/large-scale/quantized的**退化差值**。如果二者同样退化，
证据只能说明MarginalMP或场景本身有一定容忍度，不能归因给GAT。把“所提框架在该设置
下仍工作”写成“GAT增强了鲁棒性”会越过证据边界。

---

## 9. 可选支线：CG-CSI

CG-CSI默认关闭。显式开启后：

```text
aged复CSI + 当前/历史几何 + CSI年龄
    -> 共享小型MLP
    -> 预测当前完整功率矩阵
    -> 同一MarginalMP调度器
    -> 当前真实信道结算
```

它不包含GAT或GRU，也不是论文核心贡献。代码必须同时保留启用和禁用接口；主结果以不使用
CG-CSI为准。它只适合说明“简单、强先验条件下的功率补全能否缓解某类CSI延迟”，不能
泛化为普遍的无人机信道预测方案。

---

## 10. 消息轮数、payload与计算开销

主算法的协议预算分开核算：

1. Channel-GAT每个信道块进行4层邻居特征传播，并额外输出一次活动倾向；
2. 每个时隙LocalMax进行1轮分数竞争；
3. 每轮MarginalMP包含1次物理外部性反馈和1次正提案仲裁；
4. 执行$r\le K$轮时，队列阶段逻辑交换次数为$1+2r$，最坏$K=4$时为9次；
5. GAT块级开销按$S_b$个时隙摊销，不能与每时隙队列消息混为一谈；
6. CSI导频、估计和反馈单独核算，不包含在上述算法消息中。

每轮还应报告：活动状态1 bit、边际/速率变化标量、必要的link ID、GAT embedding维数、
活动接收端外部性报告数、冲突边仲裁消息数。本代码给出的是逻辑消息/标量等价开销；真实
控制信道的广播合并、分片、碰撞、重传与时钟同步不在当前实现范围内。

---

## 11. 方法、消融与代码接口

| 论文角色 | 名称 | 初始化 | 后端 | 是否训练 |
|---|---|---|---|---|
| 第3章主方法 | Channel-GAT | 不适用 | 阈值化二元调度 | 需要 |
| 第4章主方法 | GAT-Assisted MarginalMP | $QC^{\rm GAT}$ LocalMax | $K$轮精确边际 | 复用第3章GAT，不另训 |
| 必要消融 | Direct-Link MarginalMP | $QC^{\rm solo}$ LocalMax | 同一$K$轮精确边际 | 不需要 |
| 结构消融 | Direct-Link LocalMax | $QC^{\rm solo}$ | 无后端 | 不需要 |
| 结构消融 | Direct-Link Join | $QC^{\rm solo}$ | 一轮只加入 | 不需要 |
| 学习失败机制消融 | End-to-End GAT-Q | SNR与Q直接出动作 | 无物理边际 | 需要重新训练 |
| 可选支线 | CG-CSI + MarginalMP | 预测功率后进入同一主流程 | 不改变 | 预测器需checkpoint |

对应代码：

- `models/model.py`：第3章Channel-GAT、统一Shannon速率；
- `train/train_channel_gat_rician.py`：Rician训练与坍缩审计；
- `models/initial_schedule_construction.py`：GAT/Direct-Link骨架接口；
- `models/marginal_mp.py`：第4章精确翻转边际与两种初始化；
- `simulator/queue_simulator.py`：固定CSI队列、FIFO完整包统计；
- `simulator/mobile_block_fading.py`：移动、persistent identity、真实/观测CSI分离；
- `utils/fixed_csi_benchmark.py`：异质hotspot到达和配对trace；
- `models/complex_geometry_csi.py`：默认关闭的CG-CSI支线。

**实现对齐提醒。** 当前`GATAssistedMarginalMP`类和固定CSI选择接口已经存在，但部分
实验脚本的无参数默认值仍沿用上一版`Direct-Link MarginalMP`，移动脚本也尚未接入
Channel-GAT。下一项“正式baseline与作图任务”必须先按`docs/EXPERIMENTS.md`的冻结矩阵
统一这些入口；在完成接口测试前，不应直接把旧默认命令当作投稿结果。

---

## 12. 论文可以与不可以声称什么

### 可以声称

- Channel-GAT在满缓存物理层目标下学习拓扑与有向干扰相关的稀疏调度倾向；
- 第4章把该GAT倾向转换为队列加权的初始骨架信息，再由物理边际消息修正；
- MarginalMP的每个单链路提案具有明确的$\sum QR$加入/退出边际含义；
- 固定$K$轮提供可控的交互与计算预算；
- 移动仿真保持link identity，并严格分离观测CSI与真实服务CSI；
- “所提出的GAT-assisted框架”可与外部baseline比较整体性能。

### 在没有新证据前不可以声称

- 第3章问题等价于标准MWIS；
- GAT-assisted初始化稳定优于Direct-Link初始化；
- MarginalMP的增益由GAT造成；
- GAT使系统对CSI不完美更鲁棒；
- $K=4$达到全局最优或算法具有吞吐最优性证明；
- 每轮并行翻转必然单调提高全局效用；
- Centralized Greedy是数学上界；
- 当前queue-aware方法提供严格deadline/可靠性QoS保证；
- CG-CSI解决了普遍的真实无人机信道预测问题；
- 逻辑分布式接口等价于已经实现的无线MAC协议。
