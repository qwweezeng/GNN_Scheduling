# TNSE投稿版仿真设计、baseline适配与任务交接

> **文档定位。** 本文档集中说明仿真章为什么分成三部分、每幅图验证什么、比较哪些
> 方法、第三方baseline需要怎样改造，以及下一任务应按什么顺序执行。算法公式统一见
> `docs/SYSTEM_MODEL.md`。本文档不是结果报告；在正式实验完成前，所有“预期说明”都不能
> 写成已经证明的结论。

## 0. 下一任务必须先知道的五件事

1. 正式物理设置固定为N从20起、10 dBm、Rician $K=5$ dB；不再把N=10或Rayleigh作为
   投稿主场景。
2. 第3章主方法是满缓存Channel-GAT；第4章主方法改为`GAT-Assisted MarginalMP`。
3. `Direct-Link MarginalMP`必须保留为去掉GAT的关键消融，不能因结果不利而省略。
4. 前两个仿真小节默认perfect CSI；跨规模、量化、移动、stale CSI统一放入第三小节。
5. 当前只冻结了实验设计。部分脚本默认值仍是上一版Direct-Link主线；正式运行前必须先
   完成第8节的接口对齐，不能把旧默认输出直接用于论文。

---

## 1. 仿真章的三段逻辑

### Subsection A：满缓存物理层Channel-GAT

目标是证明第3章GAT在其**原本训练目标**上确实有效：所有候选链路一直有数据时，模型能
根据有向干扰拓扑产生非坍缩的稀疏并发调度，并与经典满缓存baseline竞争。

本小节只使用perfect CSI，不讨论队列，也不把其结果解释成QoS结论。

### Subsection B：perfect CSI下的队列感知调度

目标是证明第4章的两阶段结构有效：GAT先验与当前队列形成骨架，MarginalMP再通过加入/
退出边际修正骨架，最终改善goodput、backlog和完整包时延。

本小节默认perfect CSI，以免把“调度结构好坏”和“CSI估计误差”混在一起。这里必须同时
报告GAT-assisted主方案和Direct-Link必要消融。若二者接近，论文只能把收益归因给
MarginalMP后端，不能归因给GAT。

### Subsection C：CSI敏感性与模型泛化

这一小节集中回答两个部署问题：

1. GAT的共享参数能否跨网络规模使用；
2. 静态量化、无人机移动和CSI过期会使第3章与第4章各损失多少。

这样安排后，CSI内容不会在两条主线中来回跳，也不会给读者造成“前面默认完美CSI，后面
突然引入一个新预测模型”的割裂感。CG-CSI默认不进入主图。

为避免旧讨论中的图号造成误解，重排关系如下：

| 上一版内容 | 本文档新位置 |
|---|---|
| 原图1：满缓存容量 | 新图1，Subsection A |
| 原图2：训练与数据量 | 新图2，Subsection A |
| 原图5(a)：固定CSI机制消融 | 新图3，Subsection B |
| 原图6：动态队列 | 新图4，Subsection B；删除队列斜率 |
| 原图3：跨规模泛化 | 新图5，Subsection C |
| 原图4：满缓存CSI误差 | 新图6，Subsection C |
| 原图7：移动/stale CSI | 新图7，Subsection C |
| 原图5(b)：$K$轮折中 | 不再是必选主图，只保留附录候选 |

---

## 2. 统一物理、训练与统计设置

| 项目 | 正式设置 | 含义 |
|---|---:|---|
| UAV数 | 20/30/40/50 | N=20为第4章主规模 |
| 区域 | $1000\times1000$ m | 高度50--150 m |
| 最小初始间距 | 20 m | 仅初始位置生成约束 |
| 发射功率 | 10 dBm | 训练与测试一致 |
| 载频/带宽 | 2.4 GHz / 5 MHz | 所有方法一致 |
| 噪声PSD | -169 dBm/Hz | 与`configs/paper.yaml`一致 |
| 链路SNR门限 | 5 dB | 生成候选有向链路 |
| 信道 | Rician | 主设置$K=5$ dB |
| 时隙 | 1 ms | 队列每时隙更新 |
| 包大小 | 12000 bit | 1500 byte |
| 静态数据划分 | 500/128/128 | train/validation/test |
| MarginalMP轮数 | $K=4$ | 正文参数表固定；轮数图可选 |
| 缓存/deadline | 无限/无 | 主结果不引入丢弃策略 |
| 到达 | 异质Poisson hotspot | 20%链路承担70%业务 |
| 配对seeds | 至少5个 | 方法共享信道与到达trace |

移动默认采用恒速水平直线飞行、边界镜面反射和时间相关Rician block fading。建议主点为
$v=5$ m/s、每块5个时隙（5 ms），再扫描速度、块长或CSI年龄。$K$因子主结果保持5 dB；
若版面允许，只在第三小节补充5/10/15 dB或pure-LoS敏感性。

### 2.1 公平性硬规则

- 同一数据点的所有方法使用相同拓扑、真实信道、到达序列和seed；
- 所有二元动作使用同一连续SINR/Shannon函数结算；
- 不完美CSI实验只改变调度器观察到的CSI，真实服务仍用当前真实信道；
- 训练方法只能读取训练/验证集，所有调参结束后再读取固定test set；
- 缺失第三方包或checkpoint时必须明确报错，不能静默换成另一个算法；
- 任何裁剪链路、2D投影、硬冲突化或集中CSI假设都必须写进表格/图注；
- 正式作图重新训练本项目模型，并把配置、数据fingerprint和checkpoint metadata绑定；
- 结果报告均值、95%置信区间和逐seed配对差值，不只报告最好的seed。

### 2.2 动态队列正式指标

主指标只保留：

1. `successful_goodput_mbps`：完整送达包形成的goodput；
2. `average_total_queue_backlog_mbit`：全网平均积压；
3. `successful_delivery_ratio`或`unfinished_packet_fraction`；
4. `average_packet_delay_ms`与`p95_packet_delay_ms`：只在完成率足够高时解释；
5. `average_active_links`：辅助说明空间复用程度；
6. `physical_service_mbps`：只作诊断，不能替代完整包goodput。

按用户决定，**队列斜率不进入主图**。代码可保留该诊断字段，但正文不依赖它判断稳定性。

---

## 3. 建议主图：Subsection A（Channel-GAT）

### 图1：满缓存总速率与激活稀疏性

**动机。** 复现会议版最核心结论，并在正确Rician信道和10 dBm设置下重新确认GAT没有
全发/全不发坍缩。

**实验定义。** 横轴为N=20/30/40/50；纵轴主图为测试集平均总谱效率，副图或表格报告
平均激活比例。每个规模使用同规模Rician训练集重新训练Channel-GAT。

**比较方法。** Channel-GAT、Random、Top-direct-SNR、Local Greedy、FPLinQ、Spatial
DL；若Graph-Embedding适配成本可控，再作为可选补充。

**说明什么。** 只说明GAT在满缓存物理总速率任务上的调度质量、稀疏性和规模趋势。

**不能说明什么。** 不能据此证明队列稳定性、时延改善或GAT对CSI误差的鲁棒性。

### 图2：训练收敛、数据需求与坍缩审计

建议四个子图：

- (a) train surrogate utility和validation binary utility随epoch；
- (b) 训练样本数50/100/200/300/500对应同一test set的二元总速率；
- (c) 连续$\pi$的均值和标准差随epoch；
- (d) 二元激活比例、模型效用、全发效用和全不发检查。

不同样本量必须取正式500训练样本池的前缀，并共享同一validation/test set。该图回答
“训练是否真的学到东西、需要多少数据”，不能只画loss下降。

---

## 4. 建议主图：Subsection B（perfect-CSI队列调度）

### 图3：相同$(Q,H)$快照上的决策质量与机制消融

**动机。** 动态仿真中不同算法会形成不同队列，无法直接用各自的$\sum QR$判断单次动作
好坏。本图固定完全相同的队列与信道，隔离调度器本身。

**实验定义。** 使用多种队列形状（均匀、hotspot、饱和）和固定Rician信道快照，计算
最终二元动作在真实CSI下的$\sum_iQ_iR_i$。报告相对Centralized Greedy的归一化效用和
真实消息开销。

**最低必须比较。**

- GAT-Assisted MarginalMP（主方案）；
- Direct-Link MarginalMP（去GAT必要消融）；
- GAT-Assisted LocalMax（只保留GAT骨架，如接口补齐）；
- Direct-Link LocalMax；
- Direct-Link Join；
- Hard-Conflict LGS；
- Centralized Greedy（强参考，不是上界）。

**判读方式。**

- GAT-assisted与Direct-Link MarginalMP之差回答GAT初始化是否有价值；
- MarginalMP与相同初始化的LocalMax之差回答后半段边际协商是否有价值；
- 与Hard-Conflict LGS之差回答连续物理干扰是否值得保留。

若GAT-assisted没有超过Direct-Link，应如实报告“性能相当，GAT用于跨章节结构复用且未
损害后端性能”，不能写成GAT带来增益。

### 图4：随网络总负载变化的goodput、backlog和完整包时延

**动机。** 回答真实随机到达下，各算法能承载多少业务，以及空间复用和多用户队列分集
是否转化为更低积压和时延。

**实验定义。** N=20、固定perfect Rician CSI、异质hotspot Poisson到达；横轴为全网
总到达率$\Lambda$。建议三个或四个子图：

- delivered goodput；
- average total backlog；
- successful delivery ratio；
- average/p95 completed-packet delay（完成率足够高的点才解释）。

**主比较。** GAT-Assisted MarginalMP、Direct-Link MarginalMP、Direct-Link LocalMax、
Direct-Link Join、Hard-Conflict LGS、Random、Weighted FPLinQ和Centralized Greedy。
End-to-End GAT-Q可作为“直接从SNR+Q输出动作”的机制消融。

**明确删除。** 不再把后半程队列斜率放进主图；不以平均$\sum QR$替代QoS指标。

### $K$轮数是否需要单独作图

正文统一描述为“$K$轮MarginalMP”，参数表固定$K=4$。只要低成本检查确认$K=2$--4已
达到大部分效用且继续增加收益很小，主文无需专门占一幅图。可保留一张小型轮数--效用--
消息开销图作为附录或审稿回复素材；它不是当前主图的硬要求。

---

## 5. 建议主图：Subsection C（泛化与CSI敏感性）

### 图5：Channel-GAT跨网络规模泛化

**动机。** 验证共享参数和局部聚合能否在不同UAV数上零微调工作。

**实验定义。** 训练规模与测试规模均取20/30/40/50，形成4×4热图。每个单元报告相对
“该测试规模专用模型”的总速率百分比，并同时给出激活比例。

**说明什么。** 这是模型尺寸泛化，不是CSI鲁棒性，也不是运行时间可扩展性。

### 图6：第3章Channel-GAT的静态CSI表示敏感性

**动机。** 判断GAT的满缓存调度需要多精确的瞬时信道，以及低反馈大尺度信息造成多少
容量损失。

**实验定义。** 比较perfect、current large-scale及2/3/4/6/8-bit quantized
large-scale CSI。仅改变GAT输入，所有二元动作都用同一perfect真实信道结算。

**比较方式。** 至少同时报告Channel-GAT和一个非学习参考（如FPLinQ或Local Greedy）
在相同CSI变化下的绝对性能和相对退化。若所有算法同比退化，只能说明CSI误差的普遍影响。

### 图7：第4章完整调度器在移动与过期CSI下的敏感性

**动机。** 量化低空移动、块长和CSI年龄对队列调度的损害，并严格判断GAT是否提供了
额外鲁棒性。

**实验定义。** 使用persistent link identity、相关Rician block fading和相同到达轨迹。
比较latest、one-block stale、current large-scale和4-bit quantized large-scale；
服务统一用当前真实信道结算。横轴可分别取：

- 速度$v\in\{0,5,10,20\}$ m/s；
- block length；
- stale age；
- 边界/压力负载$\Lambda$。

**最低二维配对。** 每种CSI条件下都运行：

- GAT-Assisted MarginalMP；
- Direct-Link MarginalMP。

**因果判读。**

- 可写：“所提出的GAT-assisted框架在某CSI条件下保持了多少goodput”；
- 只有GAT-assisted相对Direct-Link表现出更小的配对退化，才能写“GAT提高鲁棒性”；
- 如果二者相似，应写“MarginalMP对该误差具有系统级容忍度”，不能归因给GAT。

CG-CSI默认不进入主图；如保留，只放附录虚线，且明确是额外信道预测支线。

### 可选：Rician因子敏感性

若审稿前有少量算力，可固定N=20和少数负载，仅检查$K=5/10/15$ dB及pure-LoS。目的
不是寻找最优$K$，而是说明主结论是否依赖散射强度。主表和主训练仍以有实测支持的5 dB
为准。

---

## 6. Baseline总表：采用目的与公平性

### 6.1 第3章满缓存baseline

| 方法 | 原始能力 | 在本文中的角色 | 所需输入 | 训练 |
|---|---|---|---|---|
| Random / Top-SNR | 简单启发式 | 最弱参考/无干扰感知参考 | 非空链路或direct SNR | 否 |
| Local Greedy | 冲突邻域局部竞争 | 分布式规则参考 | 链路权重+硬冲突图 | 否 |
| FPLinQ | 连续干扰下集中迭代优化 | 强物理优化参考 | 完整功率矩阵 | 否 |
| Spatial DL | 2D空间卷积调度 | 外部深度学习参考 | 固定数量2D链路布局 | 是 |
| Graph Embedding | structure2vec链路调度 | 可选学习参考 | 图与FPLinQ标签 | 是 |
| Channel-GAT | 有向图干扰学习 | 第3章主方法 | SNR、有向边特征 | 是 |

### 6.2 第4章队列baseline与消融

| 方法 | 角色 | 对队列的处理 | 物理干扰处理 |
|---|---|---|---|
| Random Nonempty | 最弱参考 | 只过滤空队列 | 最终连续SINR结算 |
| Direct-Link LocalMax | 去后端结构消融 | $QC^{solo}$ | 初始化不计并发干扰 |
| GAT-Assisted LocalMax | 去后端结构消融 | $QC^{GAT}$ | GAT诱导期望干扰 |
| Direct-Link Join | 旧两阶段消融 | $QC^{solo}$ | 一轮只允许加入 |
| End-to-End GAT-Q | 机制消融 | SNR+Q直接输出 | 训练代理目标 |
| Hard-Conflict LGS | 外部分布式规则参考 | 节点权重$QC^{solo}$ | 对称硬冲突图 |
| DistGCN | 外部GNN+传统调度参考 | 节点权重可换成$QC^{solo}$ | 原问题是硬图MWIS |
| Weighted FPLinQ | 集中迭代强参考 | 直接用Q作utility weight | 完整连续功率矩阵 |
| Centralized Greedy | 强参考，非上界 | 全局$QR$精确边际 | 完整连续功率矩阵 |
| Direct-Link MarginalMP | 去GAT必要消融 | $QC^{solo}$骨架+$QR$边际 | 连续SINR |
| GAT-Assisted MarginalMP | 第4章主方法 | $QC^{GAT}$骨架+$QR$边际 | 连续SINR |

---

## 7. 第三方baseline的改造、依赖与重训练计划

下表是下一项专门baseline任务的工作清单。`低/中/高`描述在当前代码基础上的工程成本，
不是算法复杂度。

| Baseline | 满缓存如何接入 | 队列版如何接入 | 需要的包/外部工具 | checkpoint与重训练 | 成本与投稿建议 |
|---|---|---|---|---|---|
| FPLinQ | 用本项目完整$L\times L$功率矩阵、权重全1，输出二元动作后统一结算 | 把utility weight改为当前$Q_i$；每时隙重新迭代 | NumPy、SciPy；保留源码当前与Spatial-DL目录耦合，建议拆成纯NumPy适配器 | 无checkpoint、无需训练 | **低，必做。** 但它是集中迭代方法，必须报告迭代数与全局CSI |
| Spatial DL | 把3D端点映射到其2D网格输入；必须pad/mask并保留完整链路，或让所有方法使用相同裁剪集合 | 原模型无Q输入；需增加queue/weight通道并以加权物理目标重新训练 | TensorFlow `compat.v1`、NumPy、SciPy、Matplotlib；建议独立legacy环境 | 原checkpoint只适用于其原分布；零样本测试只能称distribution transfer。公平Rician对比宜重新训练 | **满缓存中到高，队列版高。** 满缓存保留；十天期限内不建议强做QoS版 |
| DistGCN | 先把有向连续干扰阈值化为对称硬图，再运行其GCN+distributed greedy | 节点权重换为$Q_iC_i^{solo}$；每时隙权重变化，拓扑块级变化 | TensorFlow、Spektral、NetworkX 2.6.2、SciPy、pandas、dwave-networkx、cnfgen、PuLP、python-igraph、natsort；宜独立环境 | 现有checkpoint训练图分布未必匹配UAV硬图。可先零样本审计；正式公平结果优先在UAV硬图及相应权重分布上重训/微调 | **高。** 可做但必须标注“硬冲突MWIS适配”；当前adapter只生成输入，尚未完成端到端推理 |
| Graph Embedding | 用同一UAV功率快照生成FPLinQ标签，再训练structure2vec | 需生成Q-weighted FPLinQ标签并重新训练 | 旧RDKit、Boost、C++ `s2v_lib`编译、PyTorch；原数据生成还依赖MATLAB | 必须针对Rician UAV数据重新生成标签和训练 | **很高。** 只作备选，不应阻塞投稿 |
| Hard-Conflict LGS | 对称硬图上的局部分布式贪心 | 使用$Q_iC_i^{solo}$，当前已实现 | PyTorch，项目内置 | 无需训练 | **低，必做。** 不能把它命名为DistGCN |
| Centralized Greedy | 全局逐项加入/退出 | 直接最大化当前$\sum QR$边际，当前已实现 | PyTorch，项目内置 | 无需训练 | **低，必做。** 是强greedy reference，不是全局上界 |
| End-to-End GAT-Q | 不属于第3章外部baseline | SNR+Q直接输出动作 | 当前PyTorch环境 | 必须在正式N=20 Rician数据与队列快照上重训 | **中，建议做一次。** 用来证明model-based两阶段结构必要性，不做大范围调参 |

### 7.1 Spatial DL最容易产生的不公平

保留源码当前默认固定50条链路，并把3D坐标投影到2D。如果N=20产生的候选有向链路超过
50，只裁剪Spatial DL而让其他方法使用全部链路，会改变问题本身。下一任务只能选择：

1. 为Spatial DL增加可变长度pad/mask并验证所有真实链路都参与结算；或
2. 建立明确的共同50-link子问题，让所有方法在同一子集上比较。

前者更公平但需要重新训练；后者只能作为受限子问题实验，不能替代完整网络主结果。

### 7.2 DistGCN到底改了什么

DistGCN原始目标是硬冲突图上的MWIS。当前`baseline/distgcn_adapter.py`只完成：

1. 用INR门限把有向干扰关系硬化；
2. 将两方向取OR得到无向邻接；
3. 强制共享UAV的链路冲突；
4. 构造节点权重$Q_iC_i^{solo}$。

它尚未完成TensorFlow checkpoint推理、输出link-index逆映射、二元动作统一SINR结算和
消息开销审计。因此当前论文结果中不能出现标为“DistGCN”的数字。完成后也必须在图注
声明它求解的是硬图近似，而不是本文的连续干扰目标。

### 7.3 哪些baseline值得在截止期前投入

优先级建议：

1. **必须完成：** Random、LocalMax/Join、Hard-Conflict LGS、FPLinQ、Centralized
   Greedy、Direct-Link消融；
2. **应完成：** 满缓存Spatial DL（先解决完整链路公平性）、一次End-to-End GAT-Q；
3. **有时间再做：** DistGCN端到端适配；
4. **不阻塞投稿：** Graph Embedding队列版、Spatial DL队列版。

该排序避免为了增加一个名字而引入比主模型更大的新训练工程。

### 7.4 当前项目依赖清单与legacy依赖的差距

主`pyproject.toml`已经声明PyTorch、TensorFlow 2.20、NumPy、SciPy、pandas、Matplotlib等
常用包，因此FPLinQ的纯数值适配原则上不需要新学习框架。需要注意：**配置文件中声明了
TensorFlow，不等于旧baseline已经兼容当前环境。**

- Spatial DL使用`tensorflow.compat.v1`，仍需验证旧checkpoint、变量名和算子；
- DistGCN额外需要Spektral、固定旧版NetworkX、dwave-networkx、cnfgen、PuLP、
  python-igraph和natsort，这些没有全部进入主项目依赖；
- Graph Embedding还需要C++编译链、Boost、旧RDKit和MATLAB数据生成流程；
- 旧baseline很可能与Python 3.12、NumPy 2.x或TensorFlow 2.20存在版本冲突。

所以下一任务应先记录可复现的独立环境文件，再安装和测试；不要直接在主环境中反复降级
NetworkX、TensorFlow或NumPy。

---

## 8. 下一“正式仿真与baseline”任务的执行闸门

### Gate 0：先对齐代码主线，尚未运行

当前必须完成但本次文档任务没有改动的代码项：

1. 固定CSI正式默认从`Direct-Link MarginalMP`切到`GAT-Assisted MarginalMP`；
2. 无参数默认必须加载同规模、10 dBm、Rician K=5的已审计Channel-GAT checkpoint；
3. 移动脚本增加`GAT-Assisted/Direct-Link × CSI mode`二维配对接口；
4. 每个块只计算一次GAT信道先验并缓存，每时隙只乘当前Q；
5. 结果JSON保存GAT checkpoint路径、训练数据fingerprint、K因子、功率和模型variant；
6. 若checkpoint与N、功率、信道族或K因子不匹配，必须fail closed；
7. 补充GAT-assisted与Direct-Link动作、消息轮数、true/observed CSI分离的单元测试。
8. 明确无线电约束：若采用单射频/半双工假设，必须对MarginalMP每轮和最终动作施加
   shared-UAV hard feasibility，并增加“每架UAV至多参与一条活动链路”的测试；若不采用，
   必须在系统模型中说明链路级并发与功率预算假设。

在Gate 0完成前，`experiments/exp_fixed_csi_qos.py`与
`experiments/evaluate_mobile_block_fading_csi.py`的旧默认命令不是投稿主线命令。

### Gate 1：本地参考算法

先跑小样本smoke test，确认所有方法使用相同link order、功率矩阵、队列和速率函数。
任何一个方法若需要删链路或改变冲突定义，必须在结果metadata中显式记录。

### Gate 2：第三方环境隔离

Spatial DL和DistGCN都依赖旧TensorFlow生态。不要污染主`pytorch`环境；分别建立可冻结的
legacy环境，并先完成“载入原checkpoint+单样本推理+link mapping回环”测试，再决定是否
重训。

### Gate 3：低成本配对预实验

正式长仿真前先用N=20、2个seeds、少量slots检查：

- GAT-assisted与Direct-Link是否都能运行；
- action不全0/全1；
- packet与work守恒误差接近0；
- 完成率足够时delay才有解释空间；
- 各方法结果JSON配置完全一致。

### Gate 4：正式作图

通过以上闸门后，再按图1--图7顺序运行。外部baseline失败时如实留空并记录原因，不允许
用Hard-Conflict LGS、旧checkpoint或不同链路子集冒名替代。

---

## 9. 数据、模型与结果的可追溯目录

正式静态数据保持：

```text
datasets/paper/rician_K5_tx10/perfect/N{20,30,40,50}/
```

每个数据目录必须包含split size、生成配置、seed和fingerprint。正式Channel-GAT模型必须
与训练数据fingerprint及配置绑定。移动轨迹按物理参数和seed即时生成或缓存，但不能只按
“N20.pkl”命名。

建议正式结果JSON至少保存：

- git commit与工作区diff标识；
- 完整`configs/paper.yaml`快照；
- 数据fingerprint；
- 模型checkpoint及其metadata；
- 方法全名与variant；
- true CSI、observed CSI和结算模式；
- Monte Carlo seed、到达trace seed、hotspot集合；
- 逻辑轮数、实际执行轮数和消息开销字段。

---

## 10. 论文写作的统一措辞

推荐：

- “full-buffer physical-layer sum-rate scheduling”；
- “queue-aware / QoS-oriented scheduling”；
- “GAT-induced expected-interference score”；
- “GAT-Assisted MarginalMP”；
- “Direct-Link MarginalMP ablation”；
- “robustness/sensitivity of the proposed framework under imperfect CSI”。

谨慎或禁止：

- 不把第3章直接称为标准MWIS；
- 没有deadline/可靠性证明时不写“strict QoS guarantee”；
- 没有GAT-vs-Direct配对退化证据时不写“GAT improves CSI robustness”；
- 不把Centralized Greedy称为oracle或upper bound；
- 不把只生成硬图的adapter结果标为DistGCN；
- 不因Direct-Link结果不利于叙事而省略该必要消融。

最稳妥的主线表述是：第3章的Channel-GAT为第4章提供有向干扰感知的信道先验；第4章
再用显式队列权重和物理边际协商完成最终决策。GAT先验的独立贡献由Direct-Link消融
检验，MarginalMP整体性能由外部baseline检验，两类证据不能互相替代。
