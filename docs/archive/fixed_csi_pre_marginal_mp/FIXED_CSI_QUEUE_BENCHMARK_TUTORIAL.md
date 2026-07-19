# 固定已知 CSI：冻结模型、正式对照与第一轮结果

> **历史归档（2026-07-18）**：本文中的LocalPhysicalReuse-v2/PurePhysicalReuse是旧
> Join流程及其早期仿真，不是当前主方法。当前架构为
> `Quality → LocalMax → K轮MarginalMP`，正式设置$K=4$。

> **2026-07-16 有效性更正。** 本文原 N=10 GAT-assisted 曲线加载的是后来审计判废的
> 全发 checkpoint，因此本文关于“GAT 在部分中高负载优于 Pure”的数值不得继续用于
> 论文。使用正确稀疏 $\pi$、N=20 Rician matched checkpoint 的替代结果已集中写入
> `MOBILE_BLOCK_FADING_CSI_TUTORIAL.md` 的 C8--C12：GAT 在 $\Lambda=2.2,4$ 较差，
> 在 $\Lambda=6,8$ 曾相对 Pure 达到 4/5、5/5 配对联合优势；但同均值常数对照在
> 两个负载都进一步优于 GAT，说明该增益主要来自 expected-interference 数值校准，
> 尚不能归因于 GAT 的链路特异拓扑信息。本文以下内容只保留历史实验定义和旧结果
> 追溯价值，不是当前结论入口。

> 文档定位：这是固定 perfect-CSI 阶段的当前入口。读者不需要先了解
> Structure、Anchor-Join、ParallelMPNN 等历史路线；这些失败探索已经归档。

> 2026-07-15 第二次更新：当前性能边界主线已经改回**无 deadline、无限缓存、
> 5 个配对 seeds**。2000 ms deadline 结果保留为后期 QoS 真实性实验，不再用于
> 当前模型筛选。正式入口默认生成 delivered throughput、backlog 和无 deadline
> 完成包时延三类主图。

## 0. 一页总结

本阶段已经完成三件事：

1. 冻结期刊候选模型 **LocalPhysicalReuse-v2**；
2. 用统一的 `10 dBm / perfect CSI` 物理口径建立正式对照和配对队列仿真；
3. 清理 94 个停止路线文件，并保留可恢复源码压缩包。

初始 N=10、5000 slots、3 个 Monte Carlo seeds 的主要现象是：

- 在 `Lambda <= 2 packet/slot` 的稳定区，Proposed、PurePhysicalReuse 和
  Centralized-MWS-Greedy 都能基本送完到达业务；
- 在 `Lambda=3--8` 的过载区，LocalPhysicalReuse-v2 的饱和服务率持续高于
  PurePhysicalReuse；
- 在 `Lambda=8` 时，Proposed、Pure 和集中式 greedy 的服务率分别为
  `2.761`、`2.617` 和 `2.946 packet/slot`，Proposed 达到 greedy 的
  `93.7%`；
- 单时隙同状态对比中，Proposed 达到 greedy 的 `83.8% +/- 2.7%`，而
  Pure 达到 `88.4% +/- 2.4%`。这说明学习先验不是在所有状态上都优于
  纯物理规则，但它在持续高负载下提高了可复用服务能力；
- EndToEnd-GAT-q、原会议 GAT 和 random 明显落后。直接 GAT-q 在本次动态
  扫描的全部负载点均未稳定。

因此当前最诚实的论文结论是：**LocalMax + 条件物理复用的结构贡献已经被
验证；GAT 信道先验的收益主要出现在持续高负载，而不是每个孤立快照。**

后续 20000-slot deadline QoS 审计进一步修正了最后一句：若吞吐严格定义为完整
成功包 goodput，旧 GAT 在 `Lambda=3--4` 优于 Pure，但在 `Lambda=6--8` 反而
略差。因此 GAT 收益不能概括为“持续高负载普遍更强”，只能说它在部分中高负载
区间改善骨架；固定 CSI 下稳健成立的是共同的物理图协商结构。

## 1. 当前模型究竟是什么

LocalPhysicalReuse-v2 每个时隙分两步：

1. 信道 GAT 在一个固定信道块内产生 channel-only 活动先验，并将其转换为
   每条链路的期望有效速率；当前队列乘以该速率，多个互不冲突的局部最大值
   并行形成真实 0/1 骨架；
2. 其余链路以这个真实骨架为条件，计算“我加入后自己的收益减去对已激活
   邻居造成的损失”，正收益候选再做一轮局部竞争并行加入。

它不是全网唯一 anchor，也不是顺序尝试所有链路。完整公式见
`docs/LOCAL_PHYSICAL_REUSE_SYSTEM_MODEL.md`。

## 2. 为什么要同时保留 Proposed 和 Pure

两者第二步完全相同，区别只在骨架优先级：

- `LocalPhysicalReuse-v2`：`Q_i` 乘 GAT 先验诱导的期望有效速率；
- `PurePhysicalReuse`：`Q_i` 乘无干扰单链路速率，不使用学习得到的信道先验。

Pure 不是另一个 proposed model，而是最关键的消融：

- Proposed 优于 Pure，才能说明 GAT 信道先验带来额外价值；
- 两者都优于简单 baseline，只能证明共同的局部骨架和条件复用结构有效。

当前证据显示 Pure 在孤立快照和部分低负载状态更强，Proposed 在高负载动态
服务率上更强。论文必须同时展示这两个事实，不能只选有利结果。

## 3. 正式对照方法

| 名称 | 输入/视野 | 目的 |
|---|---|---|
| LocalPhysicalReuse-v2 | 局部信道图 + 当前队列 | Proposed |
| PurePhysicalReuse | 局部物理量 + 当前队列 | 去学习消融 |
| EndToEnd-GAT-q | `[SNR, Q/max(Q)]` | 直接端到端 `sum(QR)` 学习对照 |
| Conference-GAT-nonempty | 会议版 channel-only GAT + 非空掩码 | 完全不利用队列大小的原方案 |
| Random-nonempty | 随机激活非空队列 | 下限/健全性检查 |
| SNR-SingleLink | 只发最高 SNR 非空链路 | 保守单链路对照 |
| Centralized-MWS-Greedy | 全局 CSI、队列和候选边际 | 集中式强参考 |

Centralized-MWS-Greedy 不是数学上界。它每轮用全局信息计算所有未激活链路
加入后的真实 `sum(QR)` 边际，选择最大正增益，并允许有利退出；但它仍可能
陷入局部最优。

## 4. 为什么需要两类实验

### 4.1 单时隙同状态效用

所有方法面对完全相同的信道和队列快照，比较

$$
U(\mathbf a)=\sum_i Q_iR_i(\mathbf a).
$$

这回答“在同一问题上，谁选出的联合动作更好”。它不能回答长期队列是否稳定。

100 个 N=10 快照的平均效用相对集中式 greedy 为：

| 方法 | `sum(QR)` 比例 |
|---|---:|
| PurePhysicalReuse | 88.4% |
| LocalPhysicalReuse-v2 | 83.8% |
| EndToEnd-GAT-q | 29.8% |
| SNR-SingleLink | 25.9% |
| Random-nonempty | 17.5% |
| Conference-GAT-nonempty | 14.6% |

### 4.2 多时隙 QoS

每个 `N, Lambda, seed` 下所有方法共享同一个固定信道图、热点链路集合和到达
序列，比较成功包 goodput、物理服务率、平均 backlog、FIFO 包时延和丢包率。

成功 goodput 只统计完整送达包；物理服务率还包含最终可能超时包在丢弃前已经
接受的部分服务。论文主吞吐图应使用前者，后者只作为物理层诊断。

不能把动态运行中的 `sum(QR)` 直接跨方法比较：不同算法很快形成不同的队列，
更差的算法可能因为积压更大反而得到更大的权重。动态实验应以真实服务和 QoS
为主，单时隙实验负责隔离动作质量。

## 5. N=10、5000-slot 第一轮正式结果

### 5.1 稳定区

在本次到达模型下，Proposed、Pure 和集中式 greedy 在 `Lambda <= 2` 时
3/3 seeds 稳定；到 `Lambda=3` 时均出现正队列增长。因此第一轮观测到的
稳定转折位于 `2` 与 `3 packet/slot` 之间，尚未做更细的二分搜索。

在 `Lambda=2`：

| 方法 | departure (packet/slot) | backlog (Mbit/link) |
|---|---:|---:|
| Centralized-MWS-Greedy | 1.993 | 0.0063 |
| PurePhysicalReuse | 1.993 | 0.0091 |
| LocalPhysicalReuse-v2 | 1.992 | 0.0134 |
| Random-nonempty | 1.273 | 0.6865 |
| EndToEnd-GAT-q | 0.387 | 1.5646 |

低中负载下 Pure 的 backlog 比 Proposed 更低，不能隐去。

### 5.2 过载后的服务上限

| Lambda | Proposed | Pure | Centralized greedy | Proposed/greedy |
|---:|---:|---:|---:|---:|
| 3 | 2.421 | 2.402 | 2.596 | 93.3% |
| 4 | 2.513 | 2.453 | 2.716 | 92.5% |
| 6 | 2.688 | 2.543 | 2.844 | 94.5% |
| 8 | 2.761 | 2.617 | 2.946 | 93.7% |

这组结果支持“GAT 先验在持续高竞争状态下帮助空间复用”，但还需要更多 seeds
和 N=20/50 的正式长跑确认置信区间。

### 5.3 时延、丢包与未决包必须一起看

旧图曾经只保留 `stable_fraction >= 0.8` 的时延点。这能暂时避免过载状态下的
“幸存者偏差”，但稳定阈值是人为诊断规则，截断后的曲线也不适合作为论文主图。

2026-07-15 起，正式仿真改为同时记录：

1. 成功完成服务数据包的 FIFO 平均时延和 95% 分位时延；
2. 超过可配置截止时间的丢包与有限缓存溢出丢包；
3. 仿真结束时尚未成功、也尚未超时的未决包比例。

曾参考 `mab.pdf` 测试 2000 ms 截止时间和每链路 200 Mbit 缓存，但该配置现在
只作为后期 QoS 真实性接口保留，不再是实验入口默认值。期刊正文仍需根据无人机
业务类型解释 QoS 门限，或补充门限敏感性实验。`mab.pdf` 图 5 的纵轴也是成功送达包的
平均时延；它并没有把全部未完成包直接赋值为 2000 ms，而是把超时和溢出放入
丢包率指标，并用远长于截止时间的仿真减小末端观察盲区。

因此最终应联合展示完整时延曲线和丢包率，不能只凭成功包时延判断过载算法。
图中的不确定性默认使用均值的 95% 置信区间阴影；它与旧“工状”误差棒表达
相同统计量，不是重复实验的最大值和最小值。

### 5.4 当前无 deadline、5-seed 稳定边界结果

当前主结果使用 N=10、10000 slots、seeds 42--46，并细扫
`Lambda=2.0--3.2`。无 deadline 时，成功包平均时延只在至少 99% 到达包已经
完成时画出；越过该条件后，慢包仍在队列中，完成包条件均值不再用于比较。

| Lambda | GAT 完成率 | Pure 完成率 | GAT backlog | Pure backlog | GAT/Pure 稳定 seeds |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 99.75% | 99.82% | 0.0115 | 0.0081 | 5/5, 5/5 |
| 2.2 | 98.79% | 98.34% | 0.0603 | 0.0706 | 5/5, 4/5 |
| 2.4 | 94.71% | 93.91% | 0.2649 | 0.2884 | 2/5, 2/5 |
| 2.6 | 89.43% | 89.24% | 0.5547 | 0.5617 | 0/5, 1/5 |
| 2.8 | 84.66% | 84.82% | 0.8571 | 0.8399 | 0/5, 1/5 |

Backlog 单位为 Mbit/link。GAT 在 2.2 的一个困难拓扑上明显优于 Pure，但没有
一致地把总体稳定边界显著右移；2.4 以后两者差异很小且随 seed 改变方向。因此
固定 rho 或继续调骨架优先级不再是当前高优先级任务。

结果位于
`results/queue/fixed_csi/main_N10_10000slots_5seeds_no_deadline_boundary/`。

## 6. 规模闸门，不冒充正式大规模结论

seed 42 的固定信道图规模为：

| UAV | 候选链路 L | 有向冲突边 E | 冲突密度 |
|---:|---:|---:|---:|
| 10 | 27 | 450 | 0.641 |
| 20 | 148 | 13,908 | 0.639 |
| 50 | 835 | 363,456 | 0.522 |

N=50 的 200-slot 短闸门没有爆内存或失效；在 `Lambda=8`、seed 42 下，
Proposed、Pure、Random 的服务率分别为 3.219、3.123、1.258 packet/slot。
这只证明实现可运行且明显优于 random，不能替代 5000-slot、多 seed 的正式
规模图。

## 7. 代码结构与调用方式

```text
models/local_physical_reuse.py       冻结 Proposed + Pure/LocalMax 消融
baseline/queue_schedulers.py         会议 GAT、random、单链路、集中式 greedy
utils/scheduling_objective.py        只定义 sum(QR) 数学目标
utils/fixed_csi_benchmark.py         配对 trace、运行和聚合
utils/plot_queue_benchmark.py        正式队列图
scripts/build_fixed_csi_trace_cache.py 只做数据缓存，不代表论文实验
train/train_end_to_end_queue_gat.py  训练直接 GAT-q 对照
experiments/exp_fixed_csi_snapshot_utility.py  单时隙论文实验
experiments/exp_fixed_csi_qos.py     动态 QoS 论文实验
tests/                               可重复的公式、接口和回归测试
```

环境统一使用：

```bash
conda run --no-capture-output -n pytorch python <script> ...
```

本节旧 N=10 命令不再保留为可执行入口。2026-07-16 已统一停止 N=10 正式实验并删除
其训练集和 trace 缓存；当前 N=20 Rician 命令与结果见
`MOBILE_BLOCK_FADING_CSI_TUTORIAL.md` 的 C7--C12。

## 8. 数据与归档

- 旧 N=10 GAT-q 训练集和队列 traces 已删除，结果 JSON/checkpoint 仅作失败历史；
- N=20/50 的历史配对 traces 仍在 `datasets/queue_traces/`；N=50 文件约
  406 MB，它只存 3 个信道图和到达张量，不重复存 5000 份相同图；
- 正式 GAT-q checkpoint：
  `results/models/gat_queue_mws_v3_numUAV=10_tx=10dBm_csi=perfect.pth`；
- 当前 N=10 无 deadline、5-seed 主结果：
  `results/queue/fixed_csi/main_N10_10000slots_5seeds_no_deadline_boundary/`；
- 早期 N=10、5000-slot/3-seed 结果仍保存在
  `results/queue/fixed_csi/main_N10_5000slots/`，只用于历史核对；
- 94 个历史探索源码：
  `archives/two_stage_exploration_source_20260711_14.tar.gz`，manifest 与 SHA-256
  在同目录。

## 9. DistGCN、FPLinQ 与 Spatial DL

DistGCN/Zhao 2023 的核心问题是对称硬冲突图上的 MWIS。低成本接入方式是：

1. 把当前有向、带连续干扰强度的冲突关系对称化；
2. 选择一个明确阈值将其硬化为 0/1 排斥边；
3. 节点权重使用当前队列加权的单链路价值；
4. 按原代码训练/推理，并在图例中写清 `hardened-conflict adapter`。

前三步已经在 `baseline/distgcn_adapter.py` 实现：以 interference-to-noise
ratio (INR) 阈值硬化每条有向关系，再取双向 OR 得到无向图，并把共享 UAV 的
链路强制互斥。当前未完成的是加载其旧 TensorFlow/Spektral 模型并运行 GCN。

这种“强行硬化再喂给 baseline”是公平且能突出本文贡献的，因为损失恰恰来自
baseline 自身不能表达有向干扰强度和弱干扰并发。当前 Conda 环境缺少其旧代码
依赖的 SciPy，因此它作为低成本接入闸门，不阻塞主图。

需要纠正两个分类：FPLinQ 直接使用完整信道/SINR，不是简单硬冲突算法；Spatial
Deep Learning 也利用地理卷积估计聚合干扰并允许弱干扰并发。它们仍可作为
physical-interference baselines，但不能被论文错误描述成 MWIS 硬排斥方法。

## 10. 下一步

1. 当前 5-seed 边界结果先冻结，不继续为固定 CSI 调 GAT；
2. CSI 阶段必须在相同观测条件下直接比较 GAT-assisted 与 Pure，判断学习先验在
   imperfect/stale CSI 下是否获得一致增益；
3. 若 CSI 快速闸门仍无增益，停止结构修改并进入论文写作；
4. 时间允许再补 N=20/50、更多 seeds 和 DistGCN，而不阻塞主稿。
