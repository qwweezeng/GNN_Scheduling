# Changelog

## 2026-07-09 - 队列仿真改为 link-level hotspot 非均匀到达

### 背景

此前队列仿真中所有链路使用相同的泊松到达率，长期看各链路业务需求近似均匀，不利于体现重点中继链路、热点区域链路等高业务压力场景，也不利于检验 QoS-aware 调度是否会主动照顾拥塞链路。

### 修改的问题

1. **到达率口径改为全网总到达率**
   - 修改：`run_queue_experiment.py` 中 `--arrival_rates` 现在表示全网总业务到达率 `Lambda`，单位 packets/slot。
   - 说明：脚本内部生成每条链路的 `lambda_i`，并满足 `sum_i lambda_i = Lambda`。

2. **默认使用 link-level hotspot 模型**
   - 修改：预生成 arrival trace 时随机选择约 20% 链路作为热点链路，让它们承担 70% 总业务；其余链路承担 30% 背景业务。
   - 说明：同一个 Monte Carlo seed 下，不同 `Lambda` 复用同一组热点链路，只改变总业务强度，避免负载扫描时热点集合变化带来额外干扰。

3. **补充热点/普通链路分组指标**
   - 修改：每轮结果新增 `hotspot_avg_queue_mbit`、`normal_avg_queue_mbit` 等指标，用于观察热点链路 backlog 是否被有效控制。
   - 修改：终端汇总表新增 `Arr/slot` 和 `Dep/slot`，直接展示全网平均到达/发送包数。

4. **更新 README**
   - 修改：README 中 `--arrival_rates` 说明、推荐扫描范围和结果指标说明同步改为 hotspot 总到达率口径。

### 对已有数据和模型的影响

1. 不需要重新训练模型。
2. 旧的队列仿真结果仍是均匀 per-link 到达口径，不能直接和本版本 hotspot 到达结果混在同一张图中比较。
3. 后续正式 QoS-aware 对比应重新运行队列仿真，并优先观察 `average_queue_backlog_mbit`、`hotspot_avg_queue_mbit`、`normal_avg_queue_mbit`、`average_delay_ms` 和 `Dep/slot`。

## 2026-07-08 - 队列实验新增平均 backlog 与平均 delay 指标

### 背景

参考 `docs/mab.pdf` 中 Fig.4/Fig.5 的仿真展示方式，队列稳定性结果不仅需要展示最终队列或吞吐量，还需要直接对比平均队列积压和平均时延随业务到达率变化的趋势。

### 修改的问题

1. **新增平均队列积压指标**
   - 修改：`QueueSimulator.compute_metrics()` 新增 `average_queue_backlog`、`average_queue_backlog_mbit`、`average_total_queue_backlog` 和 `average_total_queue_backlog_mbit`。
   - 说明：`average_queue_backlog` 与 `long_term_avg_queue` 口径一致，但命名更贴近论文图表。

2. **新增平均时延指标**
   - 修改：新增 `average_delay_slots` 和 `average_delay_ms`。
   - 说明：当前采用 Little's law 估计，`average_delay_slots = average_queue_backlog / departure_rate_pps`，不引入逐包 FIFO age 记录。
   - 边界：若某个方法在某轮仿真中没有成功发送任何包，则 delay 无定义并记为 `None`，Monte Carlo 汇总时跳过该无效样本。

3. **新增 backlog/delay-vs-lambda 图**
   - 修改：`run_queue_experiment.py` 新增 `queue_backlog_delay_vs_lambda.png`，用 Monte Carlo 汇总结果画平均队列积压和平均时延随 λ 变化的曲线，并在可用时显示 95% CI error bar。

4. **更新终端汇总表和 README**
   - 修改：终端汇总表新增 `Backlog(Mb)` 和 `Delay(ms)`；README 补充指标含义、delay 估计口径和新图文件说明。

### 对已有数据和模型的影响

1. 不需要重新训练模型。
2. 旧的 `simulation_results.json` 不包含这些新字段，需要重新运行队列仿真后才会生成。
3. 当前 delay 是 Little's law 估计值；若论文后续需要严格统计 successfully delivered packets 的真实等待时间，需要单独实现 packet-level FIFO delay bookkeeping。

## 2026-07-07 - 队列仿真加入 paired Monte Carlo 公平比较

### 背景

在 `lambda=0.5` 的单次仿真中，GAT-q 的累计吞吐量略高于 GAT-base，但最终平均队列长度反而略大。检查结果发现，不同调度算法虽然共享了信道轨迹，但业务到达过程仍由各自的 `QueueSimulator` 独立采样。因此当两个算法性能非常接近时，某个算法额外承受的随机到达包数可能盖过其吞吐优势，造成单次结果看似反常。

### 修改的问题

1. **同一轮仿真内共享 arrival trace**
   - 问题：此前所有算法共享 `channel_realizations`，但不共享泊松到达过程。
   - 修改：`QueueSimulator.run()` 新增 `arrival_realizations` 参数；若传入 shape 为 `[num_slots, num_links]` 的到达矩阵，则每个时隙直接使用该 trace，而不是重新采样。

2. **Phase 2 支持 Monte Carlo 多 seed 平均**
   - 问题：单次 seed 下的最终队列长度容易受随机到达波动影响。
   - 修改：`run_queue_experiment.py` 新增 `--mc_runs` 和 `--mc_seeds`。每个 Monte Carlo seed 内，所有调度器共享同一套信道轨迹和到达轨迹；不同 seed 之间重新生成随机过程，最终汇总均值、标准差和 95% 置信区间。

3. **结果表和图改用汇总口径**
   - 问题：旧表格中的 `AvgQ` 实际是最后时刻平均队列，容易被误解为长期平均队列。
   - 修改：汇总表同时输出 `FinalQ` 和 `LTAvgQ`，并为 Monte Carlo 结果显示 `mean ± 95% CI`；轨迹图使用多 seed 的平均轨迹，并在可用时显示置信区间阴影。

4. **结果文件拆分**
   - 问题：多 seed 时保存完整 `[num_slots, num_links]` 队列轨迹会让 JSON 过大。
   - 修改：`simulation_results.json` 保存 Monte Carlo 汇总结果；`simulation_runs.json` 保存每个 seed 的紧凑明细，并去掉完整二维 `queue_trajectory`，保留标量和一维绘图轨迹。

### 对已有数据和模型的影响

1. 本次只改变多时隙仿真的随机过程控制、统计汇总和展示方式。
2. 不需要重新生成训练数据。
3. 不需要重新训练 GAT-base 或 GAT-q 模型。
4. 后续比较 GAT-q 与 GAT-base 时，建议使用 `--mc_runs` 或显式 `--mc_seeds`，并优先观察 paired Monte Carlo 下的 `LTAvgQ`、`FinalQ`、`throughput_ratio` 及其置信区间。

## 2026-07-07 - 队列语义统一为 link-level

### 背景

队列感知调度中，GAT 冲突图的每个节点表示一条通信链路，因此队列也应与链路节点一一对应。此前 `NetworkGenerator.generate_network()` 在训练数据生成时先生成 UAV-level 队列，再按发送端映射到链路，这会导致同一 UAV 发出的多条链路共享同一个队列值，不符合当前论文设定中的 link-level flow/queue 语义。

### 修复的问题

1. **训练数据队列从 UAV-level 映射改为 link-level 生成**
   - 问题：旧逻辑在链路集合生成前先生成长度为 `num_uavs` 的队列，再用链路发送端索引得到链路队列。
   - 修改：先生成链路集合，再按 `num_links` 为每条链路独立生成队列；`state.queue_states`、`state.vertex_features[:, 1]`、GAT 节点数量保持一致。

2. **外部传入队列只接受 link-level**
   - 问题：上一版临时兼容 UAV-level 和 link-level，语义仍不够清晰。
   - 修改：`generate_network(queue_states=...)` 现在要求 `queue_states` 长度必须等于当前链路数，否则直接报错。

3. **队列实验缓存路径升级**
   - 问题：旧缓存文件可能包含 UAV-level 映射生成的队列数据，继续复用会污染 GAT-q 训练结果。
   - 修改：队列实验缓存路径改为 `datasets/queue/dataset_numUAV=<N>_linklevel.pkl`，避免静默加载旧缓存。

4. **缓存安全检查增强**
   - 问题：不同实验入口可能复用旧缓存。
   - 修改：`run_queue_experiment.py`、`experiments/runner.py`、`power_control_experiment.py` 在队列模式下检查 `queue_states` 长度是否等于 `num_vertices`，并要求节点特征至少包含 `[SNR, queue]`。

### 对已有数据和模型的影响

1. 旧的队列训练数据需要重新生成，尤其是 `datasets/queue/dataset_numUAV=<N>.pkl` 这类旧文件不应继续用于 GAT-q。
2. 已基于旧队列数据训练出的 `gat_queue_numUAV=<N>.pth` 需要重新训练。
3. 非队列数据 `datasets/dataset_numUAV=<N>.pkl` 和非队列模型 `gat_numUAV=<N>.pth` 不受本次队列语义修正影响。
4. 若曾在 2026-07-02 之后、2026-07-07 之前重新生成过队列数据，虽然其 `queue_states` 长度可能已经等于链路数，但来源仍可能是 UAV-level 映射，建议也重新生成。

## 2026-07-02 - Queue-aware GAT 仿真修复

### 背景

当前需要验证 GAT-q 在队列稳定性上表现不佳，究竟是方法本身无效，还是仿真/实现设置导致队列信息没有正确进入模型。因此本次先修复确定的实现问题，不改变 snapshot 训练分布设计，也暂不加入异质到达率。

### 修复的问题

1. **动态队列没有同步到 GAT-q 输入特征**
   - 问题：多时隙仿真中真实队列保存在 `QueueSimulator.queue_states`，但 GAT-q 前向传播读取的是 `NetworkState.vertex_features[:, 1]`。队列更新后只更新了 `state.queue_states`，没有同步更新节点特征第二维。
   - 修改：在 `QueueSimulator` 中新增 `_sync_queue_features()`，在初始化、每次调度前、每次队列更新后，将真实 `Q(t)` 写入 `vertex_features[:, 1]`。

2. **预生成信道中的随机队列特征可能残留到第一时隙输入**
   - 问题：公平比较用的预生成信道状态包含队列特征；此前第一块信道在 `queue_states=None` 时会随机生成队列特征。仿真器随后把真实队列初始化为 0，但如果不覆盖 `vertex_features[:, 1]`，模型看到的不是 0 队列。
   - 修改：预生成信道时使用零队列作为占位；真实队列仍由 `QueueSimulator` 在每个时隙调度前覆盖。

3. **队列粒度语义不清**
   - 问题：训练数据生成默认使用 UAV-level 队列并映射到链路，而多时隙仿真维护的是 link-level 队列。原代码在 `generate_network()` 中默认按 UAV 索引处理传入队列，容易造成语义错位。
   - 修改：当时临时支持两类输入以避免仿真崩溃；该兼容逻辑已在 2026-07-07 被移除，当前统一采用 link-level 队列。

4. **`include_queue_feature` 元数据没有写入 `NetworkState`**
   - 问题：生成队列特征后，`NetworkState.include_queue_feature` 仍可能保持默认值，影响缓存检查和实验可追踪性。
   - 修改：创建 `NetworkState` 时传入真实的 `include_queue_feature`。

5. **训练器内部队列开关使用实例状态而非调用参数**
   - 问题：`Trainer.train(..., use_queue=...)` 传入了队列开关，但 loss 分支使用的是 `self.use_queue`。当前调用通常一致，但这是潜在模式错位。
   - 修改：训练 loss 分支改为使用本次调用解析后的 `use_queue`。

### 暂不修改但列入后续计划

1. 重新运行 GAT-q vs GAT-noQ，观察修复后 GAT-q 是否能改善队列稳定性。
2. 加入异质到达率，例如固定总负载 `sum(lambda_i)=lambda_total`，再按均匀/Zipf/热点链路分配到不同链路。
3. 如果 GAT-q 仍不生效，再尝试 “spatial GAT 学习空间干扰和速率结构 + 队列感知 MWS/轻量排序调度” 的 model-based 方案。
