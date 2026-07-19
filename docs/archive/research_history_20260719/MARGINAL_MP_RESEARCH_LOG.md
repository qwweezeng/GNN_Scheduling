# MarginalMP：队列感知消息传递调度与移动CSI验证

> 正式方法：**Marginal-Utility Message-Passing Scheduling (MarginalMP)**  
> 当前主实例：**Direct-Link MarginalMP**  
> 学习辅助实例：**GAT-Assisted MarginalMP**  
> 正式实现：`models/marginal_mp.py`  
> 冻结日期：2026-07-18

## 0. 阅读指引与核心结论

本文档按照期刊论文的方法章节逻辑展开，而不是按实验发生顺序记录。第一次阅读时，
建议依次阅读第1--8节；第9节以后用于实验写作、代码复现和CSI任务交接。

MarginalMP解决的问题是：在每个时隙已知链路队列和信道状态时，分布式决定哪些链路
同时发送，使队列加权服务效用尽可能高。它不是让所有链路彼此独立地一次输出发送
概率，也不要求学习模型从空集合中猜出一个完整联合动作。

算法包含前后串联的两个部分：

1. **Part I — Initial schedule construction**：选择一种链路质量来源，并通过`LocalMax`
   产生可靠、非空的初始骨架；
2. **Part II — MarginalMP refinement**：以当前二元活动集合为状态，通过消息传递精确计算每条
   链路加入或退出造成的边际效用变化，并执行固定预算的并行修正。算法一般写为
   $K$轮，当前实验固定$K=4$。

当前固定、完美CSI主实例是`Direct-Link MarginalMP`。它不含训练参数。会议版GAT可
替换初始链路质量估计，形成`GAT-Assisted MarginalMP`，但已有固定CSI证据尚未证明
GAT辅助版本优于Direct-Link版本。因此，固定CSI部分只能把GAT版本作为消融或后续CSI
扩展接口，不能声称学习模块已经带来性能提升。第16节证明power-only Link-GRU和一层
GAT提案筛选均无效；第17节进一步定位到“功率输入丢失Rician复相位”这一信息瓶颈，
并接受了897参数的`Complex-Geometry CSI Predictor`。因此perfect/latest-CSI主方法仍是
无学习的`Direct-Link MarginalMP`；当CSI存在年龄时，正式鲁棒实例为
`CG-CSI-Assisted Direct-Link MarginalMP`。

### 0.1 先区分主流程和历史Join方案

为了避免再次把模块名和完整方案名混淆，先把当前主流程拆成三个按时间排列的计算模块：

| 模块 | 输入 | 操作 | 输出 |
|---|---|---|---|
| Q：Quality | $Q,H$ | 二选一计算Direct-Link或GAT-Assisted竞争质量 | 分数$w_i$ |
| L：LocalMax | $w_i,\mathcal G_c$ | 一轮冲突邻域局部胜者竞争 | 骨架$\mathcal A^{\mathrm{LM}}$ |
| M：MarginalMP | $\mathcal A^{\mathrm{LM}},Q,H$ | $K$轮精确加入/退出边际消息传递 | 最终集合$\mathcal A^{(K)}$ |

论文主方法只有下列两种实例，区别仅在Quality分支：

| 完整方案名称 | Quality分支 | LocalMax | $K$轮MarginalMP |
|---|---|---:|---:|
| Direct-Link MarginalMP（固定CSI主实例） | Direct-Link | 是 | 是 |
| GAT-Assisted MarginalMP（学习辅助接口） | GAT-Assisted | 是 | 是 |

历史`PurePhysicalReuse`是完整的`Direct-Link Join`，即`Q → L → J`。它既不是只有
LocalMax，也不是只有Join。历史`LocalPhysicalReuse-v2`是完整的
`GAT-Assisted Join`。

结构消融已经回答了“为什么会有两个加入过程”：Join与MarginalMP都能加入链路，但
Join只允许加入且使用一跳近似，功能存在重叠。去掉Join后，静态和初步动态性能均未
下降，模型还更简洁。因此正式MarginalMP直接执行`Q → L → M`；Join只保留为一个关键
内部baseline，用于说明可退出、可反复修正的边际消息传递相对一次add-only补充的价值。

---

## 1. 从会议版信道调度到队列感知调度

### 1.1 会议版解决了什么

会议版GAT-base将每条候选通信链路表示成冲突图中的一个节点，通过GAT聚合邻居的信道
与干扰信息，输出链路调度概率$\pi_i$。训练目标主要是提高当前信道快照下的网络总
速率。该设计说明GAT能够在连续、非对称干扰图上学习稀疏并发结构，而不必把每条冲突
边都解释成严格的二元排斥。

但是，信道感知调度并不等于队列感知调度。当不同链路的待发送业务量差异很大时，
只根据瞬时信道质量选择链路可能反复服务少数优质链路，使其他链路长期积压。

### 1.2 队列感知扩展的目标

队列感知版本需要同时回答两个问题：

- **空间复用问题**：哪些链路在当前信道下可以同时发送而不过度互相干扰？
- **业务紧迫性问题**：在多个可行发送组合之间，哪些队列更值得在当前时隙获得服务？

经典最大权调度理论提供了逐时隙目标$\sum_i Q_iR_i$，但并没有自动给出一个低开销、
分布式且适用于连续干扰信道的组合求解器。MarginalMP的贡献位于这一求解过程，而
不是重新提出最大权目标本身。

### 1.3 为什么不直接让GAT一次输出所有动作

端到端GAT-Q曾把信道和队列同时输入GAT，并直接优化队列加权速率。实验暴露出以下
结构困难：

1. 所有链路近似同时输出概率，缺少一个显式的当前联合活动集合；
2. 一条链路是否应发送取决于其他链路是否发送，而不是只取决于自己的节点表示；
3. 一个动作既改变自己的服务，也改变许多其他接收端的速率，梯度信用难以分配；
4. 连续概率经过干扰函数和二元化后，容易出现全发、全不发或对队列不敏感；
5. 从空集合随机生成高质量联合动作，需要在指数规模的动作空间中命中协调组合。

因此，当前路线不再要求学习模型从零生成完整动作，而是先形成可靠集合，再通过物理
意义明确的边际效用消息对集合进行修正。

---

## 2. System Model

### 2.1 无人机与有向通信链路

考虑由$N$架无人机组成的自组网。候选有向通信链路集合记为

$$
\mathcal L=\{1,2,\ldots,L\}.
$$

链路$i$由发送节点$s_i$和接收节点$r_i$组成。时隙$t$中，链路$i$的二元调度动作是

$$
a_i(t)\in\{0,1\},
$$

其中$a_i(t)=1$表示链路$i$在该时隙发送。联合动作和活动集合分别为

$$
\mathbf a(t)=[a_1(t),\ldots,a_L(t)]^{\mathsf T},
$$

$$
\mathcal A(t)=\{i\in\mathcal L\mid a_i(t)=1\}.
$$

### 2.2 连续干扰信道

令$S_i(t)$表示链路$i$的期望接收信号功率，$G_{j,i}(t)$表示链路$j$的发送端对链路
$i$接收端造成的干扰功率。噪声功率为$\sigma^2$。给定活动集合$\mathcal A(t)$，链路
$i$的SINR为

$$
\gamma_i(\mathcal A,t)=
\frac{S_i(t)}
{\sigma^2+\sum_{j\in\mathcal A\setminus\{i\}}G_{j,i}(t)}.
$$

考虑实现损失对应的SNR gap $\Gamma$，链路频谱效率为

$$
C_i(\mathcal A,t)=
\log_2\!\left(1+\frac{\gamma_i(\mathcal A,t)}{\Gamma}\right).
$$

若系统带宽为$B$，则物理速率为$R_i=BC_i$。代码在调度比较中使用$C_i$，因为所有
链路共享的正常数$B$不会改变边际效用的符号；动态队列仿真在把服务换算成bit/s或
packet/slot时再乘带宽。

### 2.3 冲突图的作用

冲突图写为

$$
\mathcal G_c=(\mathcal L,\mathcal E_c),
$$

其中图节点是通信链路，图边表示两条链路存在需要协调的显著干扰关系。与硬冲突MWIS
不同，MarginalMP不把$(i,j)\in\mathcal E_c$解释成“两条链路绝对不能同时发送”。
冲突边主要承担两项功能：

1. LocalMax阶段限定初始局部竞争邻域；
2. MarginalMP阶段避免相邻正收益候选在同一轮同时修改集合。

最终速率仍由连续SINR计算，因此弱干扰链路即使存在冲突边，也可能在不同协商轮后
共同保持活动。

### 2.4 队列演化

令$Q_i(t)$为时隙$t$开始时链路$i$的队列长度，$A_i(t)$为新到达业务，$D_i(t)$为真实
离队业务，则

$$
Q_i(t+1)=\max\{Q_i(t)-D_i(t),0\}+A_i(t).
$$

$D_i(t)$由二元调度动作、实际可达速率、时隙长度、包长和当前剩余队列共同决定，不能
简单等同于物理层理论速率。

### 2.5 逐时隙效用

当前时隙的队列加权效用定义为

$$
U(\mathcal A,t)=
\sum_{i\in\mathcal A}Q_i(t)C_i(\mathcal A,t).
$$

最大权调度希望在每个时隙选择高效用活动集合。MarginalMP不是全局最优求解器，而是
利用固定轮分布式消息传递，从一个可靠初始集合出发提高该效用。

为简化后续符号，下文省略$t$。

---

## 3. 设计要求与总体框架

一个适合密集无人机网络的调度器至少应满足：

1. **集合依赖**：链路决策必须以当前哪些链路活动为条件；
2. **连续干扰感知**：不能把所有冲突边都处理为严格互斥；
3. **队列感知**：紧急队列应获得更高的当前服务价值；
4. **可撤销**：初始骨架中的错误胜者必须有退出机会；
5. **并行分布式**：不能依赖集中式逐链路枚举；
6. **固定交互轮数**：快变网络中不能等待不确定次数的全网收敛；
7. **信息含义明确**：交换量应能解释为链路质量、速率变化或效用变化。

基于这些要求，完整调度器分为前后串联的两个**部分（Part）**。这里使用“Part”而不
继续使用“第几阶段”，是为了避免与历史模型中“LocalMax第一阶段、Reuse第二阶段”的
旧说法混淆。

| 完整调度器中的部分 | 内部结构 | 输出 |
|---|---|---|
| Part I: Initial schedule construction | 二选一质量估计 $\rightarrow$ LocalMax | 初始二元集合$\mathcal A^{(0)}$ |
| Part II: MarginalMP refinement | $K$轮加入/退出边际计算、局部仲裁和并行更新 | 最终集合$\mathcal A^{(K)}$ |

两部分之间是严格的时间顺序：Part I先完成，Part II再开始。Part I内部则同时包含
“二选一”和“顺序执行”两种关系：Direct-Link与GAT-Assisted是两种互斥的质量估计
分支；选择其中一个后执行相同的LocalMax骨架构造。

---

## 4. Part I：Initial Schedule Construction

### 4.1 本部分的层级和时间顺序

Part I不是四个并列模块。它的真实数据流是：

```text
                    二选一的链路质量来源

  Direct-Link quality ──┐
                        ├──> LocalMax skeleton ──> A^(0)
  GAT-Assisted quality ─┘

                    顺序执行 ------------------------------->
```

因此：

1. 第4.2节先说明如何得到局部竞争分数$w_i$，其中4.2.1和4.2.2是二选一关系；
2. 第4.3节把选定的$w_i$送入LocalMax，生成骨架$\mathcal A^{\mathrm{LM}}$；
3. 第4.4节明确Part I的两个可选输出实例；
4. 第5章进入Part II，即$K$轮MarginalMP修正。

换句话说，Direct-Link/GAT-Assisted回答“**用什么信息打分**”，LocalMax回答“**谁先
成为局部胜者**”，MarginalMP才回答“**在当前集合条件下，谁应加入或退出**”。

### 4.2 链路质量估计：两个可替换分支

LocalMax需要每条链路先产生一个队列加权竞争分数$w_i$。当前保留两种互斥的构造
方式。一次调度只选择其中一种，不会先运行Direct-Link再运行GAT-Assisted。

#### 4.2.1 Direct-Link质量分支

若链路$i$独自发送，其无干扰频谱效率为

$$
C_i^{\mathrm{DL}}=
\log_2\!\left(1+\frac{S_i}{\Gamma\sigma^2}\right).
$$

Direct-Link LocalMax使用

$$
w_i^{\mathrm{DL}}=Q_iC_i^{\mathrm{DL}}
$$

作为局部竞争分数。这里的Direct-Link只描述**初始链路质量来源**，并不表示后续
MarginalMP忽略交叉链路干扰。

#### 4.2.2 GAT-Assisted质量分支

会议版GAT输出链路活动概率$\pi_j$。注意，$\pi$是会议论文中的调度概率符号，不是
英文prior的缩写。GAT-Assisted版本用这些概率形成冲突邻域内的预期干扰

$$
\widehat I_i^{\mathrm{GAT}}
=\sum_{j:(j,i)\in\mathcal E_c}\pi_jG_{j,i},
$$

以及GAT辅助链路质量

$$
\widehat C_i^{\mathrm{GAT}}
=\log_2\!\left(
1+\frac{S_i}{\Gamma(\sigma^2+\widehat I_i^{\mathrm{GAT}})}
\right).
$$

对应局部竞争分数为

$$
w_i^{\mathrm{GAT}}=Q_i\widehat C_i^{\mathrm{GAT}}.
$$

之所以称为GAT-Assisted而不是GAT-Based，是因为GAT只辅助初始集合构造；后续边际
消息和集合更新仍由确定物理公式完成。

### 4.3 顺序步骤A：LocalMax生成骨架

完成4.2.1或4.2.2中的一个质量分支后，得到统一形式的分数$w_i$。所有非空队列链路
与冲突邻居交换该分数。链路$i$在满足

$$
w_i>w_j,\quad \forall j\in\mathcal N_i
$$

时成为局部胜者。相同分数通过链路编号确定性打破平局。所有局部胜者同时进入骨架
$\mathcal A^{\mathrm{LM}}$。

LocalMax的优点是低成本、非空且分布式；局限是一次局部竞争不能判断弱干扰邻居是否
仍可安全并发，也不能在集合形成后撤销错误胜者。

### 4.4 Part I的两个可选输出实例

Part I的运算结构固定为`quality branch → LocalMax`，区别只在最开始选择哪种
质量分支：

| Part I实例 | 质量分支 | 顺序操作 | 输出 |
|---|---|---|---|
| Direct-Link LocalMax | Direct-Link | LocalMax | $\mathcal A_{\mathrm{DL}}^{(0)}$ |
| GAT-Assisted LocalMax | GAT-Assisted | LocalMax | $\mathcal A_{\mathrm{GAT}}^{(0)}$ |

二者都可以作为第5章MarginalMP的输入。当前主方法选择Direct-Link LocalMax，GAT-Assisted
LocalMax用于学习辅助消融与CSI扩展。

---

## 5. Part II：MarginalMP Refinement

### 5.1 Part II的输入以及为什么要修正LocalMax骨架

Part II只接收Part I已经完成的LocalMax二元集合$\mathcal A^{(0)}$。它不再重新选择
Direct-Link或GAT-Assisted质量分支，也不重新运行LocalMax；它只根据当前活动集合计算
精确加入/退出边际并修正该集合。

LocalMax只比较单链路竞争分数，不能直接判断弱干扰邻居能否共同发送，也不能在集合
形成后撤销错误胜者。因此，MarginalMP同时允许骨架外链路加入和骨架内链路退出，
使初始骨架只是可靠起点，而不是不可改变的最终答案。

### 5.2 未活动链路的加入边际

在第$k$轮，当前活动集合为$\mathcal A^{(k)}$。对未活动链路$i\notin\mathcal A^{(k)}$，
加入边际定义为

$$
\Delta_i^{+}(\mathcal A^{(k)})
=U(\mathcal A^{(k)}\cup\{i\})-U(\mathcal A^{(k)}).
$$

展开为

$$
\Delta_i^{+}
=Q_iC_i(\mathcal A^{(k)}\cup\{i\})
+\sum_{j\in\mathcal A^{(k)}}Q_j
\left[
C_j(\mathcal A^{(k)}\cup\{i\})
-C_j(\mathcal A^{(k)})
\right].
$$

第一项是候选自己的服务收益；第二项是它对当前所有活动接收链路造成的外部影响，通常
为负。

### 5.3 活动链路的退出边际

对活动链路$i\in\mathcal A^{(k)}$，退出边际为

$$
\Delta_i^{-}(\mathcal A^{(k)})
=U(\mathcal A^{(k)}\setminus\{i\})-U(\mathcal A^{(k)}),
$$

即

$$
\Delta_i^{-}
=-Q_iC_i(\mathcal A^{(k)})
+\sum_{j\in\mathcal A^{(k)}\setminus\{i\}}Q_j
\left[
C_j(\mathcal A^{(k)}\setminus\{i\})
-C_j(\mathcal A^{(k)})
\right].
$$

第一项是链路$i$退出后失去的服务；第二项是其他活动链路因干扰释放而获得的收益。

统一定义

$$
\Delta_i(\mathcal A^{(k)})=
\begin{cases}
\Delta_i^{+}(\mathcal A^{(k)}), & i\notin\mathcal A^{(k)},\\
\Delta_i^{-}(\mathcal A^{(k)}), & i\in\mathcal A^{(k)}.
\end{cases}
$$

若$\Delta_i>0$，则单独翻转链路$i$能够提高当前队列加权效用。

### 5.4 边际效用消息

对未活动候选$i$，活动接收链路$j$产生加入消息

$$
m_{j\rightarrow i}^{+}
=Q_j\left[
C_j(D_j+G_{i,j})-C_j(D_j)
\right],
$$

其中$D_j$是接收端$j$当前的干扰加噪声。候选聚合所有活动接收端消息后得到

$$
\Delta_i^{+}=Q_iC_i(\mathcal A\cup\{i\})
+\sum_{j\in\mathcal A}m_{j\rightarrow i}^{+}.
$$

退出消息同理：

$$
m_{j\rightarrow i}^{-}
=Q_j\left[
C_j(D_j-G_{i,j})-C_j(D_j)
\right].
$$

因此MarginalMP符合一般消息传递计算形式

$$
m_{j\rightarrow i}=\phi(Q_j,S_j,D_j,G_{i,j},a_i,a_j),
$$

$$
M_i=\sum_{j\in\mathcal A}m_{j\rightarrow i},
$$

$$
\Delta_i=\psi(Q_i,S_i,D_i,a_i,M_i).
$$

但$\phi$和$\psi$均由SINR和边际效用公式确定，没有可训练参数。因此它是model-driven
message passing，而不是一个新训练的MPNN。

### 5.5 冲突邻域仲裁

所有正边际候选形成提案集合

$$
\mathcal P^{(k)}=
\{i\mid Q_i>0,\Delta_i(\mathcal A^{(k)})>0\}.
$$

候选与冲突邻居交换$\Delta_i$，冲突局部最大者进入本轮更新集合$\mathcal S^{(k)}$。
所有选中链路并行翻转：

$$
a_i^{(k+1)}
=a_i^{(k)}\oplus\mathbf 1\{i\in\mathcal S^{(k)}\}.
$$

局部仲裁避免有显著相互干扰的候选依据同一个旧集合同时更新。由于冲突图可能忽略弱
干扰边，不相邻候选仍可能产生残余交互，所以并行更新不具有严格单调上升保证。这也是
当前N=20快照中曾出现一次轻微退化的原因之一。

### 5.6 固定预算的$K$轮消息传递

MarginalMP一般定义为$K$轮。每轮都在最新活动集合上重新计算边际，因此前一轮加入或
退出后新出现的联合机会，可以在下一轮被发现。若某轮没有正边际候选，则提前停止。

当前正式仿真固定$K=4$。这个数值不是理论最优，也不是通过不断尝试直到性能最高得到
的超参数；它只是固定在线交互预算。本文只做一次$K=1,2,3,4$的简短消融，说明增加
协商轮数带来的效用与开销变化，不继续搜索更大的轮数。论文方法名始终写成
MarginalMP，而不把轮数写进名称。

---

## 6. Algorithms

### Algorithm 1：Initial Schedule Construction

```text
Input: queue vector Q, observed channel H, conflict graph Gc,
       quality mode in {Direct-Link, GAT-Assisted}

1: Compute link quality c_i
       Direct-Link: c_i <- interference-free direct-link quality
       GAT-Assisted: c_i <- GAT-assisted expected link quality
2: w_i <- Q_i c_i
3: Exchange w_i with conflict neighbors
4: A^(0) <- all nonempty conflict-local maxima
5: Return A^(0)
```

### Algorithm 2：MarginalMP

```text
Input: queue vector Q, observed channel H, conflict graph Gc,
       initial schedule constructor I, rounds K

1: A^(0) <- I(Q,H,Gc)
2: for k = 0,...,K-1 do
3:     Every active receiver measures its current local state
           (Q_j, S_j, D_j, a_j)
4:     Active receivers form join/leave externality messages
5:     Every link i aggregates messages and computes exact Delta_i(A^(k))
6:     P^(k) <- {i | Q_i>0 and Delta_i>0}
7:     Candidates exchange Delta_i with conflict neighbors
8:     S^(k) <- conflict-local positive maxima in P^(k)
9:     if S^(k) is empty then break
10:    A^(k+1) <- A^(k) symmetric-difference S^(k)
11: end for
12: return A^(K)
```

当前实验令$K=4$。主方法`Direct-Link MarginalMP`令$I$为Direct-Link LocalMax；
`GAT-Assisted MarginalMP`令$I$为GAT-Assisted LocalMax。

---

## 7. 与集中式贪心和学习型MPNN的区别

### 7.1 不是集中式逐链路greedy

集中式Greedy在每一步获得所有候选的全局边际，选择唯一最佳链路加入或退出，然后重新
计算下一步，直到没有正收益动作。MarginalMP则：

- 每条链路根据消息并行计算自己的边际；
- 每个冲突局部社会可以同时产生一个胜者；
- 每轮允许多个互不冲突的链路同时加入或退出；
- 只运行固定$K$轮，不等待全局局部最优。

因此它借用了边际改进思想，但计算和仲裁结构是分布式并行的。

### 7.2 不是学习型MPNN

MPNN是“消息函数—聚合函数—节点更新”的一般计算框架。GAT和GraphSAGE都是具有可
训练消息或聚合规则的MPNN实例。MarginalMP也具有消息传递计算图，但消息由已知物理
公式确定。准确表述应是：

> a model-driven marginal-utility message-passing scheduler

如果CSI模块未来学习边际残差或链路质量，它应被称为learning-assisted MarginalMP，
而不是把整个确定性后端重新包装成神经网络。

---

## 8. 分布式实现、交互和复杂度

### 8.1 GNN内部传播与真实无线交互不是一回事

仿真代码中的矩阵运算可以一次完成所有消息计算，但真实部署需要控制信道或导频支持。
每轮至少涉及：

1. 活动状态广播；
2. 活动接收端的队列、当前信号和干扰状态；
3. 候选发送端到活动接收端的交叉增益或相应速率变化反馈；
4. 正边际候选与冲突邻居之间的仲裁分数。

可能的获得方式包括导频测量、信道互易、接收端反馈或局部控制信道。论文不能把这些
信息默认为零开销；后续应分别统计物理状态消息和邻域仲裁消息。

### 8.2 活动集合稀疏化后的实现

当前实现不显式构造所有链路之间的$L\times L$翻转矩阵，只计算：

- 活动发送链路到所有接收端：$|\mathcal A|\times L$；
- 所有候选发送链路到活动接收端：$L\times|\mathcal A|$。

因此每轮精确边际的主要计算与存储量为

$$
\mathcal O(L|\mathcal A|),
$$

冲突邻域仲裁为

$$
\mathcal O(|\mathcal E_c|).
$$

固定轮数$K$时，总量级为

$$
\mathcal O\!\left(K[L|\mathcal A|+|\mathcal E_c|]\right).
$$

密集网络的活动集合通常远小于候选链路集合，但$|\mathcal A|\ll L$仍是需要用规模实验
验证的经验性质，不应直接写成无条件复杂度保证。

---

## 9. 方法命名与Baseline的严格定义

### 9.1 本文内部消融

#### Direct-Link LocalMax

- 输入：队列、直达链路信号、噪声、冲突邻域；
- 分数：$Q_iC_i^{\mathrm{DL}}$；
- 操作：只执行一次局部最大竞争；
- 不执行Join，不执行MarginalMP。

#### Direct-Link Join

- 先执行Direct-Link LocalMax；
- 再执行一次一跳近似、只允许加入的Join；
- 对应历史`PurePhysicalReuse`；
- 是本文保留的唯一关键内部结构baseline。

其近似加入边际为

$$
\widetilde\Delta_i^{\mathrm{join}}
=Q_iC_i(\mathcal A^{\mathrm{LM}}\cup\{i\})
+\sum_{j\in\mathcal A^{\mathrm{LM}}\cap\mathcal N_i}
Q_j\left[C_j(\mathcal A^{\mathrm{LM}}\cup\{i\})-C_j(\mathcal A^{\mathrm{LM}})\right].
$$

正边际局部胜者只加入一次，已有LocalMax链路不能退出。它与主方法的对比直接回答：
一次add-only补充是否足够，还是需要可加入、可退出的反复条件修正。

#### Direct-Link MarginalMP

- 当前固定CSI主方法；
- 直接以Direct-Link LocalMax为初始集合，不执行Join；
- 执行$K$轮精确加入/退出边际消息传递，当前固定$K=4$；
- 不含学习参数。

#### GAT-Assisted LocalMax

- GAT输出会议版活动概率$\pi$；
- 通过预期干扰形成GAT辅助链路质量；
- 使用$Q_i\widehat C_i^{\mathrm{GAT}}$执行一次LocalMax。

#### GAT-Assisted MarginalMP

- 直接以GAT-Assisted LocalMax为初始集合，不执行Join；
- 后端使用与Direct-Link MarginalMP完全相同的$K$轮边际消息传递；
- 只有初始化信息来源不同，因此是最公平的GAT消融。

### 9.2 论文中保留哪些方法

为了避免内部版本过多造成混淆，正式性能图只保留下列核心角色：

1. `Direct-Link MarginalMP`：本文固定CSI主方法；
2. `Direct-Link Join`：唯一关键内部baseline；
3. `Hard-Conflict LGS`：外部分布式硬冲突baseline；
4. `Centralized Greedy`：使用全局信息的强参考，而非数学上界。

`Direct-Link LocalMax`只在一次结构消融中出现；$K=1,2,3,4$只属于轮数—开销消融，
不当作四个独立算法。GAT-Assisted版本用于GAT作用归因或不完美CSI研究，不与上述方法
一起堆入每一张主性能图。

### 9.3 外部与强参考方法

#### Hard-Conflict LGS

Zhao式LGS把项目冲突图对称化并作为硬排斥图。每次迭代：

1. 剩余节点使用$Q_iC_i^{\mathrm{DL}}$作为权重；
2. 所有剩余冲突局部最大节点同时获胜；
3. 删除胜者及其全部硬冲突邻居；
4. 在剩余图上重复，直到没有节点。

它是分布式算法，最终产生一个极大硬冲突独立集，但不保证是最大权独立集。它不能量化干扰强弱，也不允许存在
硬冲突边的弱干扰链路并发。当前适配使用项目相同快照和最终SINR评价，但不是对Zhao
原ER图实验的逐项复现。

#### Centralized Greedy

集中获得所有队列和交叉链路信道，逐步计算每个候选的精确全网加入边际，选择唯一最佳
正收益链路，并允许后续精确删除。它是强集中参考，但仍是贪心局部最优，不是数学上界。

### 9.4 能力对照

| 方法 | 分布式 | 连续SINR评价 | 弱干扰并发 | 可加入 | 可退出 | 学习模块 |
|---|---:|---:|---:|---:|---:|---:|
| Direct-Link LocalMax | 是 | 仅初始直链质量 | 有限 | 否 | 否 | 否 |
| Direct-Link Join | 是 | 一跳近似 | 是 | 是 | 否 | 否 |
| Direct-Link MarginalMP | 是，协议待量化 | 精确活动集合边际 | 是 | 是 | 是 | 否 |
| GAT-Assisted MarginalMP | 是，协议待量化 | 同上 | 是 | 是 | 是 | GAT辅助初始化 |
| Hard-Conflict LGS | 是 | 否 | 否 | 迭代选择 | 否 | 可选GCN剪枝 |
| Centralized Greedy | 否 | 精确 | 是 | 是 | 是 | 否 |

会议GAT、端到端GAT、Random和SNR Single-Link属于过去的诊断或会议扩展，不再作为
本节固定CSI MarginalMP主对比代码。需要追溯时只能从归档材料读取，不能重新混入当前
默认实验入口。

---

## 10. 公平评价原则

### 10.1 相同$(Q,H)$快照

静态比较必须让所有方法观察同一队列和信道快照，最终动作统一用真实SINR计算：

- $\sum_iQ_iC_i$；
- 相对Direct-Link Join的收益；
- 相对Centralized Greedy的比值；
- 活动链路数；
- 交互轮数与消息量。

### 10.2 各自动态队列轨迹

动态运行后，不同算法会产生不同的$Q(t)$。此时应比较：

- average queue backlog；
- mean与95-percentile packet delay；
- delivered throughput/goodput；
- unfinished packet fraction；
- queue growth slope与稳定负载区域。

不能直接横向比较各自轨迹上的平均$\sum Q_iR_i$，因为服务较差的方法会积累更大队列，
反而可能得到更大的数值。

---

## 11. 当前固定CSI性能证据

### 11.1 N=20相同快照

12张Rician图、三类队列，共36个快照：

| 方法 | 相对Direct-Link Join | 相对Centralized Greedy |
|---|---:|---:|
| Hard-Conflict LGS | 89.61% | 76.11% |
| Direct-Link Join | 100% | 85.50% |
| Direct-Link MarginalMP ($K=4$) | **116.40%** | **97.91%** |

Direct-Link MarginalMP在88.89%快照中优于Direct-Link Join，5.56%快照受损。

### 11.2 N=50相同快照

5张图、三类队列，共15个快照，通信链路数875--1102：

| 方法 | 相对Direct-Link Join | 相对Centralized Greedy |
|---|---:|---:|
| Hard-Conflict LGS | 94.17% | 72.78% |
| Direct-Link Join | 100% | 77.67% |
| Direct-Link MarginalMP ($K=4$) | **130.06%** | **99.76%** |

15个快照全部优于Direct-Link Join。Hard-Conflict LGS平均竞争轮数从N=20的2.17增加到
N=50的2.8，而MarginalMP固定最多四轮；这只是当前样本中的观测，不能替代完整复杂度实验。

### 11.3 轮数消融：$K$是预算，不是新算法名称

| 规模 | 指标 | $K=1$ | $K=2$ | $K=3$ | $K=4$ |
|---|---|---:|---:|---:|---:|
| $N=20$ | 相对Centralized Greedy | 94.56% | 97.22% | 98.55% | 97.91% |
| $N=50$ | 相对Centralized Greedy | 92.80% | 96.61% | 98.22% | 99.76% |

该表只说明在当前样本中，多轮更新确实能利用前一轮改变后新出现的机会，但收益并非
逐轮严格单调。本文选择$K=4$作为统一固定预算，不声称它在所有网络规模和信道上最优，
也不继续搜索更大轮数来追求更高测试集数值。每增加一轮，都线性增加一次状态消息、边际计算
和局部仲裁开销。

### 11.4 N=20动态门

2000时隙、5个seed、无deadline的初步门结果：

| 总到达率 | 方法 | backlog (Mb) | delay (ms) | goodput (Mb/s) |
|---:|---|---:|---:|---:|
| 2 | Direct-Link MarginalMP ($K=4$) | 0.00273 | 40.87 | 23.27 |
| 2 | Direct-Link Join | 0.00484 | 54.34 | 23.05 |
| 2 | Hard-Conflict LGS | 0.02129 | 105.15 | 20.51 |
| 3 | Direct-Link MarginalMP ($K=4$) | 0.02974 | 119.13 | 31.09 |
| 3 | Direct-Link Join | 0.05052 | 150.01 | 27.69 |
| 3 | Hard-Conflict LGS | 0.07376 | 181.25 | 23.96 |

$\lambda=3$时所有方法均未判稳，因此该点只能用于性能边界诊断，不能直接作为最终稳定
时延曲线。

---

## 12. 当前可声称与不可声称的内容

### 可以声称

1. MarginalMP使用当前活动集合下的加入/退出边际，而不是链路独立概率；
2. 它允许连续干扰下的弱干扰并发，不受硬独立集限制；
3. 它采用冲突邻域并行仲裁和固定预算的$K$轮更新，当前实验取$K=4$；
4. 当前N=20/N=50静态证据稳定优于Direct-Link Join和Hard-Conflict LGS；
5. 初步动态门中，backlog、delay和goodput均表现出正向改善。

### 不能声称

1. 四轮MarginalMP达到全局最优，或$K=4$具有理论最优性；
2. 每次并行更新都严格单调提高效用；
3. Centralized Greedy是数学上界；
4. 当前分布式控制消息是零开销；
5. 固定CSI下GAT-Assisted版本已经优于Direct-Link版本；
6. MarginalMP本身是一个训练得到的GNN或MPNN。

---

## 13. 不完美CSI研究使用的稳定接口

CSI任务不需要重新设计队列调度后端。可以从两个接口接入。

### 13.1 替换初始集合构造器

实现

```python
schedule_with_state(state, queues)
```

并传入

```python
MarginalMPScheduler(initializer, rounds=4)
```

可研究利用过时CSI预测、时序GAT或不确定性估计产生更可靠的初始集合。

### 13.2 筛除不可信的正边际提案

先根据观测CSI计算

$$
\Delta_i^{\mathrm{obs}},
$$

若$Delta_i^{\mathrm{obs}}>0$，链路$i$会成为本轮加入或退出的候选。低成本学习接口不
重新预测最终调度，也不直接回归一个难以校准的完整边际值，而只估计

$$
r_i=\Pr\!\left(\Delta_i^{\mathrm{true}}>0\mid
\text{本地观测、队列、当前动作和冲突邻域}\right).
$$

随后仅保留

$$
\mathcal P_{\rm keep}
=\{i:\Delta_i^{\mathrm{obs}}>0,\ r_i\ge\tau\},
$$

再让保留下来的候选执行原来的局部最大值竞争。该接口保留了“物理公式负责计算边际，
学习只负责处理CSI不确定性”的职责边界。必须同时比较无学习观测边际、相同输入的
本地MLP、一层GAT和真实符号oracle。第16节的正式闸门已经失败，因此本截稿周期不再
训练更深GAT、GRU、STGNN或边际残差模型。

---

## 14. 正式代码与复现实验

### 14.1 模型

```text
models/initial_schedule_construction.py
    DirectLinkLocalMaxScheduler
    DirectLinkJoinScheduler
    GATAssistedLocalMaxScheduler

models/marginal_mp.py
    MarginalMPScheduler
    DirectLinkMarginalMP
    GATAssistedMarginalMP
    exact_link_toggle_marginals
```

历史文件`models/local_physical_reuse.py`因CSI实验仍依赖其低层物理计算而保留，但不是
正式模型入口。旧MWS-CGN兼容代码已移入`archives/fixed_csi_pre_marginal_mp_code_20260718/`。

### 14.2 Baselines

```text
baseline/queue_schedulers.py
    HardConflictLocalGreedyScheduler
    CentralizedMWSGreedyScheduler
```

### 14.3 静态相同快照

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/compare_marginal_mp_vs_lgs.py
```

### 14.4 动态队列

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/exp_fixed_csi_qos.py \
  --methods "Direct-Link MarginalMP,Direct-Link Join,Hard-Conflict LGS,Centralized Greedy"
```

旧命令中的`MWS-CGN`、`PurePhysicalReuse`等名称不再由正式入口接受；需要复现实验时
使用归档源码，不得把旧显示名写入新论文结果。

---

## 15. 下一步固定CSI工作

1. 对N=20少量受损快照分析非邻接候选并行交互；
2. 正式统计Direct-Link MarginalMP、Hard-Conflict LGS和Centralized Greedy的运行
   时间、轮数和消息量；
3. 运行完整负载范围、10000时隙、5-seed动态实验；
4. 输出backlog、mean/95-percentile delay、goodput和稳定区域图；
5. 将第2--8节压缩改写为IEEE论文System Model与Proposed Method正文。

---

## 16. 移动与不完美CSI：最后一轮学习价值闸门

本节只围绕一个问题展开：**当观测CSI不再等于当前真实CSI时，学习模块能否为
Direct-Link MarginalMP带来至少5%的、跨seed一致的实际调度收益？** 所有方案都在同一
真实UAV轨迹、同一队列快照和同一CSI观测下生成动作，最终效用统一用当前真实CSI结算。
本节没有把学习器当作最终调度器，也没有修改四轮MarginalMP后端。

### 16.1 新概念解释

**当前真实CSI**是本时刻真实的直达和交叉信道功率，只用于结算服务或构造oracle；实际
调度器不能偷看。

**观测CSI**是调度器可用的信息。本节检查三种不完美形式：上一信道块的`stale CSI`、
当前几何位置决定的`large-scale CSI`以及4-bit量化大尺度CSI。

**错误正提案**是满足$\Delta_i^{\mathrm{obs}}>0$，但实际上
$\Delta_i^{\mathrm{true}}\le0$的加入/退出建议。它说明观测CSI把一次有害动作误判成了有益
动作。

**真实符号筛选oracle**保留观测边际的大小和排序，只删除错误正提案。它不是可部署
算法；它回答的是“即使存在完美分类器，只改提案的正负判断最多能获得多少收益”。

**Link-GRU**是已有的逐链路时间预测器。它从历史大尺度状态预测下一块物理信道，再把
预测信道交给完全相同的Direct-Link MarginalMP；本节不重新训练它。

### 16.2 实验A：损失来自骨架还是四轮边际修正

#### 动机

如果主要损失发生在初始LocalMax骨架，学习器应当修正Part I；如果主要损失发生在后续
边际计算，则继续修改会议GAT骨架没有抓住问题。先做这一分解可以避免训练错误模块。

#### 定义与具体实现

固定$N=20$、10 dBm、Rician $K=5$ dB、5 m/s、5 ms信道块、5个配对seed、每个seed
11个有效块、$K=4$。分别使用异质队列和饱和队列。比较：

1. 观测骨架 + 观测边际；
2. 真实骨架 + 观测边际；
3. 观测骨架 + 真实边际；
4. 观测边际 + 真实符号筛选；
5. 真实骨架 + 真实边际。

#### 与上一版区别

旧实验只比较最终调度器在不同CSI输入下的结果，无法判断错误发生在哪一阶段。本实验
逐项替换骨架CSI和边际CSI，因而可以做因果定位。

#### 结果

| CSI / 队列 | 观测方案占真实CSI方案 | 换真实骨架后/观测方案 | 换真实边际后/观测方案 | 真实符号筛选/观测方案 |
|---|---:|---:|---:|---:|
| stale / 异质 | 89.01% | 102.83% | 110.22% | 104.10% |
| stale / 饱和 | 91.06% | 102.88% | 108.27% | 101.67% |
| large-scale / 异质 | 92.01% | 101.99% | 107.99% | 102.22% |
| large-scale / 饱和 | 93.69% | 103.24% | 106.42% | 102.00% |
| 4-bit / 异质 | 88.43% | 103.67% | 112.47% | 105.12% |
| 4-bit / 饱和 | 86.97% | 106.74% | 114.22% | 103.20% |

观测正提案中约25.9%--33.1%在真实CSI下为负；最终被局部竞争选中的动作中约
17.0%--29.3%实际有害。

#### 分析：证明了什么

第一，CSI不完美确实严重，造成约6.3%--13.0%的真实效用损失。第二，主要损失位于四轮
边际修正，而不是初始骨架：换成真实边际可追回6.4%--14.2%，换真实骨架多数只有约
2%--3%。因此继续只修改会议GAT的初始$pi$接口不是主攻方向。

第三，虽然错误正提案很多，但“只识别正负号”的性能上限很低。真实符号oracle只有
4-bit异质队列达到预设5%门槛，其他五个单元只有约1.7%--4.1%。原因是一次错误提案即使
被删除，后续局部竞争和其他轮次仍会改变活动集合；反过来，一些真正需要的是修正边际
大小与排序，而不只是删除负号错误。

#### 下一步

不进入高成本的完整边际残差回归。只允许进行两个更便宜的闸门：先直接复用已有
Link-GRU；再用一个本地MLP和一层GAT检查错误正提案是否至少具有可学习性。

### 16.3 实验B：现成Link-GRU能否转化为MarginalMP收益

#### 动机与方法

旧Link-GRU曾在物理信道预测指标上有微弱改善，但预测误差更小不等于队列调度更好。
因此不重新训练，直接把同一N=10 Rician checkpoint产生的预测功率矩阵送入
`Direct-Link MarginalMP`，与stale、current large-scale和简单affine-AR预测比较；最终
效用仍用真实CSI计算。这个N=10实验只用于判断现成模型是否值得迁移到正式N=20，不是
论文正式规模结果。

#### 结果

| 队列 | 最强非学习预测 | Link-GRU / 最强非学习预测 | 配对seed胜率 |追回latest差距|
|---|---|---:|---:|---:|
| 异质 | current large-scale | 101.05% ± 0.80% | 4/5 | 14.13% |
| 饱和 | affine-AR | 101.34% ± 1.25% | 5/5 | 14.79% |

#### 分析与下一步

Link-GRU方向多数一致，但平均收益仅1.05%和1.34%，远低于5%继续门槛，而且只追回约
14%--15%的latest-CSI差距。该现象与物理推理一致：逐链路大尺度预测只能修正一部分
缓慢变化，无法可靠恢复同时受许多交叉增益影响的联合边际排序。因此停止N=20重训，
不增加GRU层数或STGNN时间模块。

### 16.4 实验C：本地MLP与一层GAT的提案可靠性探测

#### 动机

实验A发现错误正提案客观存在。最后需要回答：这些错误是否包含一种只有图邻居聚合
才能识别的规律？若一层GAT明显超过同输入MLP，才说明“图理解”在CSI不确定性下具有
独立价值。

该接口借鉴Zhao等人的“GNN先做稀疏化、传统算法再完成调度”分工，但没有机械照搬：
GAT只判断哪些MarginalMP提案值得进入局部竞争，最终动作仍由物理边际和传统仲裁决定。
这样可以直接检验GNN是否真的降低竞争或提高鲁棒性，而不是把两部分简单拼接。

#### 定义与具体实现

学习目标是第13.2节的$r_i$，而不是最终动作。两种模型共享本地输入：对应CSI模式下的
观测直达SNR、冲突度、CSI类型、观测边际、归一化队列和当前二元动作。stale模型不偷看
当前large-scale CSI，4-bit模型也不偷看未量化large-scale CSI。
本地MLP不交换消息；一层GAT额外聚合一次冲突邻域，得到的信道嵌入在同一信道块的四轮
MarginalMP中缓存复用。只使用2个训练seed、1个验证seed、6 epochs；5个正式测试seed
不参与训练。阈值在验证集上选择，并要求至少保留95%的真实有益提案。

#### 与上一版区别

它不是让GAT替代物理调度，也不是用GAT直接输出骨架。物理公式先产生正提案，学习器只
承担不完美CSI下的可靠性筛选；这已经是比旧$pi$骨架接口更贴近损失来源的最小改造。

#### 结果

验证集上，MLP与GAT的平衡准确率分别只有50.68%和50.51%，几乎等于随机猜测。为保留
99.65%的真实有益提案，MLP与GAT分别只能删除1.71%和1.37%的错误提案。

在线四轮调度中：

- 本地MLP在六个单元中的效用比全部为100.00%，没有任何seed级可归因改善；最多只在
  stale/heterogeneous减少1.97%的提案通知；
- 一层GAT在六个单元中的效用比也全部为100.00%，没有超过同输入MLP；最多只在
  stale/heterogeneous减少1.69%的提案通知；
- 两个模型在其余大多数单元几乎不筛选任何提案；
- 若从调度器自身核算开销，GAT每信道块新增1轮邻居聚合。按冲突边逐次嵌入传输计数，
  即使暂不考虑嵌入向量比标量提案更大，GAT后的总节点广播数也达到原MarginalMP的
  9.86--19.81倍，总邻居接收数达到11.19--21.43倍。因此不能声称总消息开销下降。

#### 分析：为什么失败

真正困难的量不是“这条链路自己的CSI是否偏了”，而是一次翻转对当前完整活动集合造成
的联合外部性。它取决于许多交叉增益误差共同作用后的边际大小和局部排序。一个浅层
分类器看到了局部信号，却无法在保证不误删有益动作的同时识别足够多的有害动作。
更重要的是，真实符号oracle本身在五个单元都不到5%；即使继续堆深模型，性能天花板也
不支持投入。

### 16.5 最终决定与论文边界

本轮对power-only时间模型和提案筛选接口的决定为
`STOP_LEARNING_ASSISTED_MARGINAL_MP`：

1. 不训练更深GAT、GRU、STGNN、RNN或完整边际残差模型；
2. 不为不足1%的变化继续调阈值、loss、temperature或训练轮数；
3. 固定CSI和移动CSI下的队列调度主方法均保持`Direct-Link MarginalMP`；
4. 会议GAT章节仍作为channel-only全网香农速率最大化方法保留；期刊队列章节将其作为
   动机来源和严格消融，但不虚构GAT已经改善队列调度；
5. LAIN相关性由LoS主导Rician移动信道、persistent link identity、真实/观测CSI分离，
   以及latest/stale/large-scale/quantized敏感性分析支撑；学习鲁棒化作为有明确负证据的
   边界或未来工作，而不是当前贡献。

该决定只覆盖当时已验证的power-only与proposal-filter接口。第17节后来根据用户提出的
“规律运动应当保留复相位可预测性”假设，使用不同的信息接口重新开启了一次最终闸门；
它不推翻本节对GRU/GAT的负结论。

复现实验与正式结果：

```text
experiments/diagnose_marginal_mp_mobile_csi_headroom.py
results/formal/marginal_mp_mobile_csi_headroom_N20_rician_20260718.json

experiments/evaluate_link_gru_marginal_mp_gate.py
results/formal/link_gru_direct_link_marginal_mp_gate_N10_rician_20260718.json

experiments/probe_marginal_mp_proposal_reliability.py
results/formal/marginal_mp_proposal_reliability_probe_N20_rician_20260718.json
```

---

## 17. 重新审视“规律运动是否能使信道预测有用”

第16节否定的是现有power-only Link-GRU和提案筛选接口，不等于否定所有轨迹辅助信道
预测。用户进一步提出：如果UAV在一段时间内按规划轨迹规律运动，Rician直射径是否应当
具有更强的可预测性？本节先澄清当前模型，再记录一次不训练的初步区域扫描。

### 17.1 当前运动其实已经很规律

当前每架UAV初始化一个航向，随后匀速直线运动，只有到达区域边界时才镜像反射。因此
此前结果不是由随机游走或每块随机改变方向造成的。真正的随机性来自每个有向UAV pair
各自独立的Gauss--Markov散射创新。

当前Link-GRU输入为上一块功率残差、当前大尺度功率、当前距离和距离变化，但不包含复
信道的实部/虚部。Rician直射径的相位按$2\pi d/\lambda$变化，瞬时功率来自直射径与
散射分量的相干叠加。仅知道上一块功率无法唯一恢复散射复相位，因此即使距离轨迹完全
规律，power-only GRU仍丢失了预测下一块相干叠加所需的状态。

此外，当前不同UAV pair的散射创新独立。邻居链路不能提供本链路下一次随机创新，所以
空间GAT没有超过Link-GRU并非意外。若未来要证明空间GAT预测有用，必须先有物理依据地
引入共享散射簇、空间相关阴影或编队运动产生的跨链路相关性，不能只为获得增益而人为
制造相关数据。

### 17.2 初步实验D：K-factor与速度区域扫描

#### 动机与方法

预测只有在“CSI损失足够大”且“历史相关性仍足够强”时才可能产生调度收益。固定
$N=20$、10 dBm、5 ms信道块、Direct-Link MarginalMP和3个配对seed，不训练任何模型，
扫描$K\in\{5,10,15\}$ dB与速度$\{2,5,10,20\}$ m/s。下表给出one-block stale相对
latest的真实效用损失；括号内为当前模型的$\rho^2$。

| K-factor | 2 m/s（0.880） | 5 m/s（0.413） | 10 m/s（0.003） | 20 m/s（0.029） |
|---|---:|---:|---:|---:|
| 5 dB / 异质队列 | 5.14% | 11.76% | 15.75% | 16.30% |
| 5 dB / 饱和队列 | 5.05% | 10.30% | 14.60% | 13.07% |
| 10 dB / 异质队列 | 1.95% | 6.90% | 7.88% | 7.45% |
| 10 dB / 饱和队列 | 1.80% | 2.50% | 4.82% | 4.80% |
| 15 dB / 异质队列 | -0.85% | -0.83% | 1.31% | 1.18% |
| 15 dB / 饱和队列 | 0.52% | 0.55% | 2.04% | 0.82% |

负损失只表示有限样本中stale动作偶然优于latest动作，不表示过期CSI具有普遍优势。

#### 现象与分析

单纯提高速度会扩大$K=5$ dB下的CSI损失，但也把可预测相关性降到接近零，形成“损失
大但主要由新创新造成”的区域。提高到$K=15$ dB后，信道更可预测，却只剩约0%--2%的
调度损失，缺少值得恢复的空间。$K=10$ dB处于中间区域，但多数单元的5%上限仍偏紧。

因此，下一步不应继续盲扫更高速度。更合理的变量是把**物理信道块长度**与**CSI反馈
年龄**分开：保持2--5 m/s和5 ms物理块，通过1/2/4块的观测延迟形成5/10/20 ms CSI
年龄。这样可能在保留时间相关性的同时积累足够的调度损失。

### 17.3 唯一值得继续的最小闸门

先不训练网络，只增加两个诊断：

1. 扫描$K$、速度和独立CSI年龄，画出
   $1-U_{\rm stale}/U_{\rm latest}$的容忍区域；
2. 让trace保留复信道，构造使用上一块复CSI、当前位置/相对速度和已知Rician状态转移
   的条件均值预测器。它是当前仿真模型下的物理预测上限，不是学习模型。

只有某个单元同时满足“stale损失至少8%”“物理条件预测相对stale改善至少5%”和
“至少3/3 seed同方向”，才训练一个最小complex Link-GRU。其输入应包括上一块复CSI、
相对位置/速度、CSI年龄和K-factor；仍先做link-local时间模型，不先加空间GAT。

若这个物理参考都不能通过，立即停止。如果它通过，只训练同输入Complex MLP与最小
Complex Link-GRU。学习预测器必须在5个未见seed上同时超过stale和large-scale并转化为
MarginalMP效用；要声称时间记忆有用，GRU还必须比MLP高至少2个百分点。空间GAT只有在
后续采用具有共享散射或空间相关性的、文献可辩护的信道模型后才重开。

### 17.4 实验E：复CSI条件预测器是否存在真实调度恢复空间

#### 动机与定义

第17.2节只测量stale损失，没有证明这些损失可预测。本实验让trace额外只读保留复信道
$h$，并使用上一份复CSI、当前/历史几何、已知Rician $K$和已知Gauss--Markov $\rho$
计算当前功率的条件期望。该预测器不读取延迟期间产生的新创新，也不训练网络；它是
当前仿真模型下的强物理参考，而不是可部署学习模型或数学上的调度上界。

固定$N=20$、10 dBm、$K\in\{5,10\}$ dB、速度$\{2,5\}$ m/s、CSI年龄
$\{5,10,20\}$ ms、3个配对seed、异质/饱和队列和四轮Direct-Link MarginalMP。所有动作
用当前真实CSI结算。继续闸门要求：stale损失至少8%、条件预测相对stale改善至少5%、
3/3 seed调度效用同方向且3/3 seed功率MAE改善。

#### 结果

$K=5$ dB的12个单元中有10个通过；仅2 m/s、5 ms年龄的两个单元因stale损失不足8%
没有通过。代表结果如下：

| 速度 | CSI年龄 | 队列 | stale损失 | 条件预测/stale | stale MAE | 条件预测MAE |
|---:|---:|---|---:|---:|---:|---:|
| 2 m/s | 10 ms | 异质 | 8.44% | +7.82% | 2.621 dB | 1.394 dB |
| 2 m/s | 20 ms | 饱和 | 15.52% | +15.42% | 3.604 dB | 1.805 dB |
| 5 m/s | 5 ms | 异质 | 10.67% | +7.82% | 3.281 dB | 2.152 dB |
| 5 m/s | 10 ms | 饱和 | 14.43% | +12.14% | 3.817 dB | 2.506 dB |
| 5 m/s | 20 ms | 异质 | 17.54% | +13.76% | 3.816 dB | 2.677 dB |

$K=10$ dB的单元全部未通过，主要因为stale损失只有约2%--7%，不是因为条件预测完全
失效。该结果证明之前power-only Link-GRU确实丢失了可用于调度的复相位状态，允许进入
唯一一次最小学习闸门。

### 17.5 实验F：complex Link-GRU是否真的需要时间记忆

#### 方法

只使用通过物理闸门的$K=5$ dB区域，训练一个跨2/5 m/s和1/2/4块年龄共享的预测器。
两种模型输入完全相同：

1. aged复CSI经大尺度归一化后的实部和虚部；
2. aged/current LoS相位的正弦和余弦；
3. $\rho^D$、以波长归一化的距离变化、CSI年龄和速度。

输出是当前真实功率相对当前large-scale功率的dB残差。`Complex MLP`无隐藏状态；
`Complex Link-GRU`额外保留每个永久有向UAV pair的16维本地状态。使用4个训练seed、
1个验证seed、5 epochs，正式测试5个未见seed；不做超参数搜索。

#### 结果

| 模型 | 参数量 | 验证MAE | 测试MAE | 测试90%误差 |
|---|---:|---:|---:|---:|
| Complex MLP | 897 | 2.252 dB | 2.281 dB | 4.709 dB |
| Complex Link-GRU | 2569 | 2.248 dB | 2.275 dB | 4.558 dB |

GRU在多个静态调度单元相对stale获得5%以上改善，但没有任何单元比同输入MLP高2个
百分点。两者MAE也只差0.006 dB。因此预设决定为
`STOP_COMPLEX_LINK_GRU_NO_SCHEDULING_VALUE`：成功来自复CSI和显式几何，不来自循环
时间记忆。不得通过增加epoch、hidden dimension或GRU层数追逐该微小差异。

相反，Complex MLP在5 m/s的5/10/20 ms年龄下，相对stale分别取得约6.0%--11.6%的
静态效用改善，并追回物理条件预测器约80%--105%的调度收益。它同时优于current
large-scale，因此允许进入一次不重训的短动态队列闸门。

### 17.6 实验G：静态改善能否转化为吞吐和backlog

#### 动机与设置

静态$\sum QR$提高不等于长期队列一定改善。冻结Complex MLP，选择最小且现实的
$K=5$ dB、5 m/s、5 ms CSI年龄，固定$N=20$、1000 slots、5个配对seed和热点到达，比较
latest、stale、current large-scale、物理条件预测和Complex MLP。Direct-Link MarginalMP
完全不变，所有真实服务由当前CSI结算。

#### 结果

| 负载 | 方法 | delivered throughput | physical service | 平均总backlog | 平均packet delay |
|---:|---|---:|---:|---:|---:|
| 2.2 | latest | 24.888 Mbps | 25.691 Mbps | 0.550 Mbit | 38.57 ms |
| 2.2 | stale | 23.378 Mbps | 24.327 Mbps | 1.308 Mbit | 61.52 ms |
| 2.2 | complex conditional | 24.185 Mbps | 25.056 Mbps | 0.962 Mbit | 50.79 ms |
| 2.2 | Complex MLP | 23.971 Mbps | 24.831 Mbps | 1.094 Mbit | 55.29 ms |
| 4 | latest | 37.169 Mbps | 38.091 Mbps | 5.231 Mbit | 94.54 ms |
| 4 | stale | 28.927 Mbps | 29.689 Mbps | 9.005 Mbit | 143.73 ms |
| 4 | complex conditional | 32.522 Mbps | 33.314 Mbps | 7.437 Mbit | 117.54 ms |
| 4 | Complex MLP | 31.651 Mbps | 32.403 Mbps | 7.858 Mbit | 120.62 ms |

相对两个非预测基线中更强的stale CSI：

- $\Lambda=2.2$：delivered throughput提高2.59%，physical service提高2.12%，backlog
  降低12.94%，4/5 seed同时满足吞吐不降、服务不降和backlog至少降低5%；
- $\Lambda=4$：delivered throughput提高9.64%，physical service提高9.35%，backlog
  降低12.50%，5/5 seed联合改善。

因此动态决定为`ACCEPT_COMPLEX_GEOMETRY_MLP_CSI_MODULE`。

### 17.7 最终模型、物理意义与止损边界

正式模块命名为：

- **Complex-Geometry CSI Predictor（CG-CSI）**；
- 完整鲁棒实例：**CG-CSI-Assisted Direct-Link MarginalMP**。

CG-CSI不是最终调度器。每个接收端利用本地保存的aged复CSI和已知/规划几何，独立预测
相关入射链路的当前功率；各链路拼成一张自洽功率矩阵后，仍由Direct-Link MarginalMP
完成队列感知调度。模型集中训练，但执行时每个pair使用共享的897参数MLP，无隐藏状态、
无GAT、无新增邻居消息轮。

开销必须分成三类报告：

1. **算法消息轮**：相对Direct-Link MarginalMP新增0轮；
2. **本地计算**：每pair每信道块约848次乘加。$N=20$全部380个有向pair的保守上限为
   322240次乘加/块，即5 ms块长下约64.4 M MAC/s；
3. **CSI观测表示**：power-only stale CSI每pair保存1个实数，CG-CSI需要复CSI的I/Q两个
   实数。若接收端本地保留导频得到的复估计，不增加空口消息；若必须集中反馈，float32
   口径下相对power-only多32 bit/pair/block，380 pair、5 ms时上限约2.432 Mbit/s。

这项CSI获取/反馈开销不能伪装成“零开销”，但也不能与MarginalMP邻居协商轮数混为
一谈。当前论文应同时给出本地执行和集中反馈两种口径。

最终科学结论是：

1. 原power-only GRU无效不是因为Rician信道绝对不可预测，而是输入丢失复相位；
2. 显式恢复复CSI和几何后，最简单MLP已稳定修复CSI aging损失；
3. GRU没有额外价值，因为当前一阶Markov信道的充分状态已由aged复CSI和几何显式给出；
4. GAT仍无预测价值，因为不同pair的散射创新独立，不存在可利用的跨链路随机信息；
5. 本任务到此冻结，不再增加GRU/GAT/STGNN、训练轮数或损失补丁。

正式实现与结果：

```text
models/complex_geometry_csi.py
experiments/diagnose_complex_csi_prediction_gate.py
experiments/train_complex_link_gru_gate.py
experiments/evaluate_complex_mlp_dynamic_gate.py

results/formal/complex_csi_prediction_gate_N20_rician_20260718.json
results/formal/complex_link_gru_gate_N20_ricianK5_20260718.json
results/formal/complex_mlp_dynamic_gate_N20_ricianK5_20260718.json
results/models/complex_geometry_csi_N20_ricianK5_20260718.pth
```
