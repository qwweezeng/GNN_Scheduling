# Chapter V 代码同步与分阶段仿真执行报告

日期：2026-08-11

本报告记录本次已经实际执行的修改、测试、正式实验和停止条件。算法原理和逐段代码说明见
`docs/CHAPTER5_GAT_MP_CODE_GUIDE_ZH.md`。

## 1. 本次边界

- 只修改和验证 Chapter V 的 GAT-MP / Direct-MP 及其实验接口。
- 没有修改 Chapter IV 的 GAT 结构、训练代码或 decoder。
- 没有重新训练 GAT；所有实验均加载已有 matched-condition checkpoints。
- 没有修改论文 TeX。
- 新结果写入 `results/validation/ch5_local_coordination_v1/` 和
  `results/paper/ch5_local_coordination_v1/`，没有覆盖旧论文结果。

## 2. toggle 是什么

toggle 就是“把一条链路当前的活动状态翻转一次”：

- `0 -> 1`：链路申请加入当前活动集合，称为 joining toggle；
- `1 -> 0`：链路申请退出当前活动集合，称为 leaving toggle。

代码使用 XOR 执行已经确认的 toggles。它不是随意开关：加入和退出都必须有正的
conflict-graph-local marginal；同轮候选的协调邻域必须互不重叠；加入候选还必须获得两个
endpoint UAV 的确认。

## 3. 已完成的功能修改

主要实现位于 `models/marginal_mp.py`：

1. marginal 只读取直达功率和 conflict edges 上的干扰功率；非冲突物理功率不会进入计算。
2. 同时支持 joining 和 leaving marginal。
3. 每个协调邻域成员只为 marginal 最大的 proposal 投票；完全相同时选链路编号较小者。
4. proposal 只有获得整个闭协调邻域的一致确认才被接受，因此同轮 accepted neighborhoods
   两两不相交。
5. joining proposal 还必须获得两个空闲 endpoint 的确认；GAT-MP 不再依赖事后集中投影。
6. 每轮记录 local utility、joins/leaves、消息数量、邻域互斥和 endpoint feasibility。

固定 block 缓存位于 `models/local_physical_reuse.py` 和 `models/marginal_mp.py`：

- GAT-derived rate prior 每个 state/block 只计算一次；队列权重和 MP 每个时隙更新。
- 固定冲突图的协调邻域和 membership 映射每个 block 只构建一次。

为降低 N=50 密图的运行成本，邻域投票由逐 proposal 的 Python 循环改为等价的张量投票：

1. 每个成员取得覆盖它的 proposals 中的最大 marginal；
2. marginal 相同时取得最小链路编号；
3. 候选所得票数等于自己的邻域大小时才被确认。

该优化没有更改论文机制。N=50、200-slot 的 8 个对照任务在 goodput、backlog、delay、
active links、accepted toggles 和 minimum local utility delta 上与优化前的差值全部为 0；
运行时间从约 2 分 58 秒降到约 47 秒。

## 4. 代码测试结果

最终单元测试命令：

```bash
MPLCONFIGDIR=/private/tmp/gnn_mpl_cache \
.venv/bin/python -m unittest discover -s tests -v
```

结果：63/63 通过。

测试覆盖：

- 非冲突功率变化不改变 local marginal；
- marginal 等于单候选 conflict-graph utility 变化；
- joining 和 leaving 都能执行；
- 有重叠协调邻域的候选不能同时接受；
- 并行 accepted toggles 的联合效用变化具有可加性；
- endpoint 占用和 endpoint 竞争能够阻止加入；
- GAT prior 每个 block 只计算一次；
- observed CSI 改变 marginal，隐藏 true CSI 不进入调度；
- 向量化确认与直接投票定义在全部 32 种测试 proposal 组合上完全一致；
- 固定 CSI benchmark 能将每时隙诊断汇总为标量指标。

## 5. Fig. 5 正式重跑结果

输出目录：

```text
results/paper/ch5_local_coordination_v1/fig05_N50_seed42
```

配置：N=50、10000 slots、seed=42、J=4、Rician K=5 dB、deadline=2000 ms。

| lambda | Direct goodput | GAT goodput | Direct backlog (Mbit) | GAT backlog (Mbit) | Direct delay (ms) | GAT delay (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.491 | 0.491 | 0.007 | 0.007 | 41.32 | 41.50 |
| 1.0 | 0.980 | 0.978 | 0.021 | 0.022 | 55.08 | 59.40 |
| 1.5 | 1.470 | 1.470 | 0.102 | 0.135 | 83.75 | 94.63 |
| 2.0 | 1.881 | 1.866 | 2.004 | 2.762 | 181.72 | 197.60 |
| 2.2 | 2.016 | 1.981 | 4.680 | 5.912 | 227.37 | 240.79 |
| 2.4 | 2.115 | 2.091 | 9.082 | 10.008 | 277.14 | 284.73 |
| 2.6 | 2.188 | 2.166 | 13.097 | 14.165 | 322.15 | 340.28 |
| 2.8 | 2.257 | 2.236 | 16.336 | 17.497 | 343.66 | 374.37 |
| 3.0 | 2.316 | 2.305 | 21.625 | 22.739 | 405.52 | 442.17 |
| 3.2 | 2.342 | 2.356 | 27.918 | 28.095 | 492.69 | 497.55 |
| 3.5 | 2.380 | 2.429 | 36.603 | 34.741 | 621.67 | 566.75 |
| 4.0 | 2.494 | 2.571 | 49.336 | 45.070 | 747.90 | 623.96 |
| 4.5 | 2.621 | 2.693 | 60.873 | 55.613 | 818.02 | 663.36 |
| 5.0 | 2.719 | 2.784 | 71.724 | 66.332 | 864.13 | 709.56 |

这里的 backlog 是所有链路的平均总 backlog，不是控制台显示的每链路 backlog。

协议诊断：

- 28/28 method-load tasks 完成；
- minimum local utility delta 的全局最小值为 `9.5367431640625e-7`；
- accepted neighborhoods 全部互不重叠；
- 全部时隙满足 shared-UAV feasibility；
- hard projection rejection 最大值为 0；
- 协议版本均为 `conflict_graph_local_coordination_v1`。

### 如何解释性能

GAT-MP 不是在所有负载都严格优于 Direct-MP：

- 低负载时非空队列和可竞争链路少，两种 initializer 经 LocalMax 和 MP 后常得到相同或很接近的活动集。
- 2.0--3.0 的中负载区，GAT prior 有小幅劣势；这应在论文消融中如实报告。
- 约 3.2 开始出现 goodput 交叉；3.5--5.0 时 GAT-MP 在 goodput、backlog 和 delay 上均改善。
- 一个合理解释是：拥塞时有更多非空链路和空间复用竞争，GAT prior 更容易把初始活动集放入更好的局部协调起点；随后四轮 MP 在局部范围内改进。该解释是实验推断，不是理论保证。

当前目录只包含发生变化的 Direct-MP 和 GAT-MP。生成最终七方法 Fig. 5 前，必须先核对
旧 baselines 与新结果的 trace/topology fingerprints 完全一致，再合并；不能直接将当前目录
作为最终完整图的数据目录。

## 6. Fig. 7 代表 CSI 门控

### N=20，10000 slots

输出：`results/paper/ch5_local_coordination_v1/fig07_gate_N20.json`

| observed CSI | goodput (Mbps) | backlog (Mbit) | delay (ms) | CSI error (dB) |
|---|---:|---:|---:|---:|
| latest | 18.966 | 50.070 | 835.56 | 0.000 |
| large-scale | 17.791 | 56.902 | 991.92 | 2.922 |
| 4-bit | 17.827 | 57.064 | 983.96 | 3.197 |

### N=50，2000 slots

输出：`results/paper/ch5_local_coordination_v1/fig07_gate_N50.json`

| observed CSI | goodput (Mbps) | backlog (Mbit) | delay (ms) | CSI error (dB) |
|---|---:|---:|---:|---:|
| latest | 20.664 | 6.620 | 170.21 | 0.000 |
| 4-bit | 13.062 | 14.247 | 214.94 | 3.109 |

两组门控均满足：

- matched-condition checkpoint 被正确加载；
- latest CSI error 为 0，其他 observed representations 的 error 非零；
- shared-UAV feasibility 全时隙为 true；
- hard projection rejection 为 0；
- packet count conservation error 为 0；
- 服务使用 realized true channel 结算。

N=50 的 4-bit 性能损失较大，但不是新协议独有异常。旧 matched-CSI、10000-slot 结果中
latest/4-bit goodput 为 19.584/11.737 Mbps，损失约 40.1%；本次门控损失约 36.8%，
方向和幅度一致。

## 7. 当前停止位置和原因

Fig. 5 已完成全部正式负载。Fig. 7 已完成 T6a，但尚未执行 N=20/30/40/50 的全部
7 个 CSI 条件。

资源门控的实测结果：N=50、2000 slots、两个 CSI 条件约需 1 小时。线性估计 N=50、
10000 slots、7 条件约需 15--20 小时；加上 N=20/30/40 后更长。因此没有在无人确认时
直接启动全量 T6b。

## 8. 下一步分批命令

`resume.py` 会在每个条件后写 progress，并只运行 Chapter V。以下命令不会训练 GAT，
因为显式使用 `--chapters 5`。

先运行 N=20，并在结束后检查趋势：

```bash
MPLCONFIGDIR=/private/tmp/gnn_mpl_cache \
UV_CACHE_DIR=/private/tmp/gnn_scheduling_uv_cache \
uv run python -u paper_figures/fig07_csi_feedback/resume.py \
  --nodes 20 --slots 10000 --seeds 42 --arrival-rate 2.4 \
  --chapters 5 --device auto \
  --chapter5-output-root results/paper/ch5_local_coordination_v1/fig07_full
```

N=20 通过后依次将 `--nodes` 改为 30、40、50。建议 N=40 和 N=50 在服务器或独立长时
窗口运行。若中途停止，重复同一命令即可从 progress 继续。

每个 N 完成后检查：

```bash
jq '{
  protocol:.marginal_protocol_version,
  records:(.records|length),
  all_feasible:([.records[].shared_uav_feasible_all_slots]|all),
  max_projection_rejections:([.records[].average_hard_projection_rejections]|max),
  max_count_error:([.records[].packet_count_conservation_error]|map(abs)|max),
  csi:[.records[]|{method,delivered_throughput_mbps,
    average_total_queue_backlog_mbit,mean_csi_absolute_error_db}]
}' results/paper/ch5_local_coordination_v1/fig07_full/chapter5_N20_v2_matched_csi.json
```

预期目的不是要求量化 bit 数严格单调，而是确认：全部条件可行、没有隐藏 projection、
latest 无 CSI 误差、各 observed representation 确实改变调度、服务仍使用 true channel，
并且总体趋势能支持论文中有限度的 imperfect-CSI 结论。
