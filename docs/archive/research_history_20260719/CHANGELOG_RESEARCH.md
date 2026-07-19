# Changelog

## 2026-07-18 - MarginalMP固定CSI架构冻结与入口清理

- 正式架构冻结为`Quality → LocalMax → K轮MarginalMP`，统一实验预算$K=4$；
- 结构消融删除主流程中重复的add-only Join；Direct-Link Join仅保留为关键内部baseline；
- 固定CSI主方法为`Direct-Link MarginalMP`；`GAT-Assisted MarginalMP`仅保留为GAT/CSI
  归因接口，后端与主方法完全相同；
- 正式baseline收敛为Direct-Link Join、Hard-Conflict LGS和Centralized Greedy；
- 当前文档只保留`MARGINAL_MP_SYSTEM_MODEL_AND_HANDOFF.md`与`NEXT_CHAT_CONTEXT.md`为入口，
  旧LocalPhysicalReuse、MWS-CGN和two-stage教程移动到归档；
- 旧兼容模型、Join-initialized MarginalMP和历史fixed-CSI实验源码移动到
  `archives/fixed_csi_pre_marginal_mp_code_20260718/`，不再由正式入口导入。

## 2026-07-15 - LoS/Rician、channel-only与骨架--协商遮蔽诊断

- 核对会议论文和代码：两者均采用`g~CN(0,1)`纯Rayleigh，而非实现遗漏；
- 为移动生成器增加均值功率归一化的Rician K-factor和pure-LoS敏感性边界，保持位置、
  diffuse随机数、永久link identity和large-scale预算配对；
- 新增`experiments/diagnose_channel_model_and_stage_masking.py`，在无动态队列条件下单列
  interference-free Shannon ceiling、true-CSI centralized greedy、会议GAT二值动作、
  LocalMax骨架、条件加入和两个阶段反事实；
- 5-seed正式结果发现Frozen-GAT在1200/1200状态中阈值后全部全开；LocalPhysicalReuse
  后半段仍恢复到greedy的约91%--94%，证实最终队列指标会遮蔽prior问题；
- Rayleigh/Rician/pure-LoS的current-stale相关分别约0.548/0.664/1.000，stale误差分别
  5.04/3.42/0.0008 dB；Rayleigh stale骨架/最终Jaccard仅0.387/0.413；
- 完成Rayleigh和Rician 5 dB stale-only同架构GRU探针。Rician将初始teacher-logit
  MSE降低约55%，但两者均早停在零残差，说明Rayleigh放大难度但不是当前GRU失败的
  充分原因；
- 新增后归档至`docs/archive/20260715_pre_correction/CHANNEL_MODEL_AND_STAGE_MASKING_TUTORIAL.md`，并收窄旧Phase-F止损：停止
  Frozen-teacher logit重建，不外推为所有物理信道预测/STGNN无效。

## 2026-07-15 - 单帧 CSI 增强与最小 Temporal-Residual 止损

- 新增 `train/train_csi_augmented_spatial_gat.py` 和统一配对评估：直接 physical
  soft-sum-rate 目标退化成全1、输入不敏感 prior，因此即使个别队列条件数值通过也不
  计为学习式鲁棒性；纠正为 true-CSI Frozen teacher logit 重建后，输出保持 CSI
  sensitivity，但没有任何 imperfect 条件同时以4/5胜过 Frozen 和 Pure；
- 新增 `models/temporal_residual_gat.py`：冻结会议空间 GAT，每条有向永久链路保存
  一层48维 GRU 状态，只输出零初始化 logit 残差，不新增邻居消息轮次；
- 使用 train seeds 1000--1019、validation seeds 2000--2004、test seeds 42--46。
  正式时间训练连续5个 epoch 未超过零残差初始化，最佳 `epoch=-1`；
- 新增 `experiments/evaluate_temporal_residual_gat.py`。40个正式队列 run 与非坍缩
  单帧增强版的 delivered throughput 逐项完全相同，最大绝对差为0，时间历史 logit
  贡献为0，因此触发 Phase-F 止损；
- 新增后归档 `docs/archive/20260715_pre_correction/CSI_AUGMENTED_SPATIAL_GAT_TUTORIAL.md` 和
  `docs/archive/20260715_pre_correction/TEMPORAL_RESIDUAL_GAT_TUTORIAL.md`，完整记录动机、定义、失败目标、训练现象、
  配对结果、因果边界、消息/存储开销和下一步；
- 新增3个时间身份/零残差测试，全套29个测试通过；`py_compile` 与
  `git diff --check` 通过。

## 2026-07-15 - Frozen-GAT / Pure 移动 CSI 第一层配对闸门

- 扩展 `simulator/mobile_block_fading.py`：在永久 link identity 队列上增加 FIFO
  完整包完成记账，分开 delivered throughput 与 physical service；
- 重写 `experiments/evaluate_mobile_block_fading_csi.py`：支持 Frozen-GAT/Pure、
  `Lambda=2.2,4`、5-seed配对差、4/5联合胜利闸门和方法特定消息开销；
- 新增 `tests/test_mobile_csi_experiment.py`，全量26个测试通过；
- 正式 N=10、1000-slot、5-seed结果中，`Lambda=4` 的 stale、large-scale、4-bit
  distance条件分别为5/5、4/5、5/5联合胜利；`Lambda=2.2`均未通过；
- 新增后归档 `docs/archive/20260715_pre_correction/MOBILE_CSI_FROZEN_GAT_VS_PURE_TUTORIAL.md`，下一步进入同架构
  CSI-augmented Spatial GAT，用于隔离训练分布与时间维度贡献。

## 2026-07-14 - 移动 block-fading、CSI 风险与消息开销诊断

新增 `simulator/mobile_block_fading.py` 和 `experiments/evaluate_mobile_block_fading_csi.py`：永久 link ID 为有向 UAV 对，队列跨块保留，不可用链路动作置零；真实服务信道与调度观测 CSI 分离。小尺度信道采用块级 Gauss--Markov/Jakes 相关，UAV 采用恒速反射移动。所有 CSI 方法共享真实移动、衰落和到达轨迹。

10 UAV、$\Lambda=4$、300 slots、seed 42/43/44、速度 5/20 m/s、块长 5/20 slots 的诊断显示：one-block stale CSI 相对 latest 的服务比下降 16.9%--28.1%，最坏长期 backlog 增长 68.0%，因此预测可行性实验具有必要性，但尚未证明学习式预测模块必要。large-scale 与 stale 大体同层；2-bit 距离量化明显不足，4/8-bit 与未量化 large-scale 在当前三个 seed 下接近。当前轨迹只有 0.3 s，尚未触发可用链路切换，结果只作为风险诊断。

根据可读性和结论边界审查，新增 `docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`，从当前任务动机、全部新概念、统一实验环境、E9--E12 逐项因果记录、可证明边界、混杂因素和下一步止损条件重新说明本轮工作。同时纠正旧措辞：当前只支持“stale 风险严重，值得测试预测”，不支持“预测器已经被证明必要”；当前 quantized 实验明确命名为 distance-quantized large-scale CSI。

消息审计区分顺序轮数和 payload：每块 5 轮信道消息，每时隙 3 轮队列消息，摊销为 $3+5/T_c$ 轮/slot。逐冲突边单播上界约 0.48--1.56 Gbit/s，按发射 UAV 聚合且不含头部/MAC 的净载荷下界约 16.0--52.4 Mbit/s；固定轮不等于低开销。原始结果为 `results/diagnostics/mobile_block_fading_csi_20260714_194642.json`。

## 2026-07-14 - LocalPhysicalReuse-v2、50-UAV 闸门与下一 Chat 交接

### 当前模型

将 `LocalPhysicalReuse-v2 (GAT Expected-Rate)` 确定为当前推荐学习版。会议版 channel-only GAT 的活动先验先形成期望局部干扰和有效速率，再与瞬时队列组成 LocalMax 权重；真实局部胜者集合之后只做一轮冲突邻域条件复用。旧 `q_rel * softplus(GAT logit)` 降为 v1 历史基线，纯物理 `Q * solo-rate` 保留为强消融。

### 规模与指标结论

完成 10/20/50 UAV、10 dBm、perfect CSI、三个共享轨迹的动态扫描。表中数值明确为 `实际发送包数 / 总到达包数`，不是相对 MWS 比值。20 UAV、Lambda=8 时 v2/MWS-Greedy 分别为 0.3865/0.4254，对应约 37.23/40.97 Mbit/s，二者均处于过载。50 UAV 没有性能断崖，但约 835 条通信链路、36 万条冲突边时 CPU 暖时隙约 9.2 ms，尚不满足 1-ms 实时性。

### 数据与复现

新增 `experiments/cache_queue_benchmark.py` 和 `--trace_cache` 接口。固定信道动态基准不再按时隙重复保存大图，而是每个 MC seed 保存一张图，并为各 Lambda 单独保存到达张量。已生成 75.4-MiB 的 50-UAV 三 seed 缓存；缓存重跑 seed 42、Lambda=2 与原报告的 v2 服务比逐位一致（0.9749838618908898）。旧 16-GB、无物理标签的 `dataset_numUAV=50.pkl` 未读取、未覆盖。

### 文档与下一阶段

新增 `docs/NEXT_CHAT_CONTEXT.md` 作为唯一新 Chat 入口，补充集中训练/逻辑分布式执行、在线输入和推理周期。现行 two-stage 文档的公式分隔符由 `\[...\]` 改为 Codex 预览兼容的 `$$...$$`。下一阶段先定义 link identity 一致的 block-fading/UAV 移动仿真，再依次验证最新、过期、large-scale 和 quantized CSI；只有证明确有明显过期损失时才引入最小信道预测模块。

## 2026-07-13 - Two-stage 路线审查、10 dBm 统一与文档归档

### 路线结论

单 Anchor 后逐条 greedy Join 不再作为默认最终方向。它保留为条件化集合决策的结构诊断，但其不可撤销局部最优、随激活数增长的顺序轮数和大规模全图选举不符合密集、快变、交互敏感场景。下一步先审查“稀疏图 + 固定轮并行可撤销 GAT/MPNN 协商”，在复杂度、队列因果作用和固定轮收敛得到证据前不训练正式新版本。

### 10 dBm 物理口径

1. 当前正式默认统一为 `TX_POWER=10 dBm`；修正队列实验、通用 runner、功率控制和网络生成示例中遗留的 22 dBm fallback。
2. 新数据集文件名加入 `tx=<...>dBm_csi=<...>`，避免只按 UAV 数量静默加载旧 22 dBm 缓存。
3. 新模型路径和 checkpoint 元数据保存相同物理标识；加载基础信道模型时校验功率和 CSI。
4. 旧 22 dBm 数据、模型和结果不删除、不覆盖，只作为历史结构诊断。此前 two-stage 探索数字需要在 10 dBm 下重新训练后才能成为正式比较。

### 规模审计

新增无训练审计 `experiments/audit_anchor_join_scalability.py`。10 dBm 下，10/20/50 UAV 平均或代表性样本分别产生约 31/138/961 个有效链路节点和 668/11,303/517,718 条有向冲突边。50 UAV 时固定三轮 GAT约需 155 万次逻辑边更新；按图独立集大小逐条选举的结构代理约为 1,657 万次，证实顺序 Anchor–Join 的扩展风险。

### 文档整理

根目录只保留：

- `docs/TWO_STAGE_FROM_ZERO_TUTORIAL.md`：从零教学与完整时间线；
- `docs/TWO_STAGE_DIRECTION_REVIEW.md`：当前路线审查与决策。

其余 Structured、Marginal、PhysicalNegotiation、Anchor–Join、顺序学习、原始流水和旧 Chat 交接文件全部移动到 `docs/archive/two_stage/`，并由归档 README 统一索引。

## 2026-07-11 - 端到端递进诊断与 TwoStage-GAT-v1

### 方向

不再继续搜索初始 bias、temperature 或附加惩罚项。使用共享快照和三个训练 seed，依次验证基础 GAT、恒定 Q、变化 Q 但不加权、连续 weighted-MWS 和 STE weighted-MWS，并在确认失败后转向冻结信道 GAT + 显式队列融合 + 学习式 Queue-GAT。

### 修改

1. 固化 `R_i=pi_i*log2(1+SINR_i)` 的单次 `pi_i` 语义并增加数值测试；loss 只计算 `sum Q_i*R_i`，不再额外乘 `pi_i`。
2. 新增 `queue_model_diagnostics.py`，记录 A-D 的激活数、输入敏感性、梯度符号一致性和相对 MWS-Greedy 效用。
3. 存档 v3 checkpoint，新增 `TwoStage-GAT-v1`：冻结基础信道 GAT，显式构造一跳相对队列和 `q_rel*softplus(channel_score)`，浅层 Queue-GAT 模仿离线 MWS-Greedy 教师动作。
4. 新增 `QxChannel-LocalMax` 规则诊断 baseline；在线只使用邻居信息和本地决策，不访问集中式 MWS 教师。
5. 主队列实验在正式两阶段 checkpoint 存在时自动加入两个新方法；README 和交接文档补充运行方式与模型路径。

### 结果

A/B/C 都稳定学习到约 4.1 条激活和约 16.7 的测试总速率，说明变化 Q 输入本身不破坏基础 GAT。连续 weighted-MWS 退化为约 42.7 条激活、仅达到 MWS-Greedy 效用的 8.9%；STE 三个 seed 均退化为全关。因此失败集中在 weighted-MWS 全局信用分配，而不是 GAT、双通道或 local skip 本身。

TwoStage-GAT-v1 静态测试达到 MWS-Greedy 效用的 35.9%，动作 F1=0.384，队列交换概率变化约 6.37%，但 SNR 降低敏感性仍很弱。1000-slot、MC seeds 42/43/44 中，其 throughput 在 lambda=1/2/5/10 时约为 0.308/0.276/0.216/0.121；高负载明显优于旧 GAT，低负载仍存在激活不足。规则 LocalMax 对应约 0.991/0.615/0.209/0.098，说明显式局部竞争在低负载有效，但单赢家限制了高负载空间复用。模型在静态训练分布平均激活 8.06 条、教师为 2.34 条，而动态低负载只激活约 0.3 条，还暴露了队列占用密度的分布泛化问题。下一步应学习局部并发度或结构化竞争，并对教师数据按稀疏/中等/密集占用分层，不继续围绕初始化和惩罚项修补。

## 2026-07-10 - GAT-q MWS v3 与集中式 MWS-Greedy 尝试

### 方向

针对 v2 在仿真中退化为“几乎激活全部非空链路”的问题，结合本地 GAT 论文中关于 all-ones collapse、梯度稳定性和连续概率训练的讨论，尝试让训练前向直接看到二元调度，并增加一个可解释的全局 MWS 参照。

### 修改

1. 队列特征改为相对尺度 `Q_i/max(Q)`，降低对绝对队列范围和 λ 的依赖。
2. 队列训练启用 stochastic STE；验证阶段校准全局硬决策阈值，并只按 hard MWS utility 选择 checkpoint。
3. 队列模型路径升级为 `gat_queue_mws_v3_numUAV=<N>.pth`，加载时检查版本、特征方式、训练决策方式和 `queue_range`，不匹配则重新训练。
4. 增加 `MWS-Greedy (global CSI)`：全局获取 Q 和信道增益，逐步加入候选链路并每次重算实际 SINR，再做删除改进；这是 MWS 的贪心近似，不是精确穷举最优。
5. 增加平均激活链路数、非空链路命中率、空队列误激活率、SNR/队列反事实敏感性和平均 `ΣQ×service` 等记录。
6. 训练 loss 的梯度尺度改回决策无关的 `-Σ(Q×μ)/ΣQ`，避免人为使用 `num_links×queue_max` 压小 STE 梯度。

### 当前认识

v2 在 λ=1、2 时能超过 Random，但受控测试显示其 SNR/队列交换后的硬决策变化均为 0，主要学成了 work-conserving 的非空队列激活。静态测试中 all-nonempty 的 `ΣQR` 明显低于 MWS-Greedy，说明仍有很大空间复用优化空间。v3 用于验证 STE、相对队列和阈值校准是否能恢复真正的信道/队列选择能力。

### 影响

信道/拓扑数据集可复用；需要重新训练并生成 v3 队列模型。旧 v2 模型保留作为失败对照，不会自动加载。本轮仍属于实验性尝试，后续需重点观察 `queue_hard_decision_change`、`snr_hard_decision_change`、`average_active_links` 及 GAT 与 MWS-Greedy 的差距。

smoke 产生的临时模型已保留在 `results/models/smoke/gat_queue_mws_v3_numUAV=10_smoke30x3.pth`，标准模型路径已清空，避免小规模 smoke 模型被正式实验误加载。

### Smoke 验证

30 个训练快照、3 epoch、30 slot 的小实验运行通过。λ=1 时 MWS-Greedy throughput ratio=0.955、GAT=0.250、Random=0.060；MWS-Greedy 平均激活约 1.2 条链路，GAT 约 0.2 条。v3 受控测试中队列交换使约 4.2% 的硬决策改变，SNR 打乱使约 8.8% 改变，说明 STE 后已经出现输入敏感性，但尚不能说明接近 MWS。

## 2026-07-10 - GAT-q MWS v2 训练路径

### 背景

修正队列同步、link-level 队列和 paired Monte Carlo 后，旧 GAT-q 的调度仍与 GAT-base 基本一致。定量检查显示：训练数据中队列几乎全部非空，SNR 输入尺度远大于队列特征，且旧模型 logit 严重饱和，对队列变化几乎不响应。

### 修改

1. **保留原 GAT 主体，增加最小 local input skip**
   - 不改变 4 层 GAT、消息传递和 sigmoid 调度头，不加正系数约束、STE 或额外惩罚项。
   - 最终调度 MLP 输入改为 `[GAT embedding, normalized local SNR, normalized local Q]`，让调度头同时使用邻居聚合信息和本链路状态。
   - 小型对照中，1/2/4 层无 skip 模型均未改变队列硬决策；加 skip 后，交换正队列大小使约 8.18% 链路改变二值决策，最佳硬验证 MWS utility 由约 79.88 提高到 207.77。

2. **动态稀疏队列训练**
   - 每次前向为当前信道快照重新生成 link-level 队列。
   - 非空链路数在 `1..num_links` 中均匀采样，非空队列长度由 `--queue_range` 控制，同时覆盖低负载稀疏状态和拥塞状态。

3. **输入标准化**
   - 对 `[SNR_dB, log1p(Q)/10]` 使用训练分布 mean/std 分通道标准化。
   - 标准化统计与模型权重一起保存，确保训练和多时隙仿真使用同一口径。

4. **packet/slot MWS loss**
   - 服务量改为 `mu_i = R_i * bandwidth * slot_duration / packet_size`。
   - Loss 改为 `-sum(Q_i * mu_i) / (sum(Q_i) + eps)`；分母不依赖调度决策，不改变单快照 MWS 最优解，只稳定跨快照梯度尺度。
   - 仿真 `utility_trajectory` 改用调度决策时的 `Q(t)` 与当时隙服务量计算，不再错用队列更新后的 `Q(t+1)`。

5. **空队列物理约束**
   - 训练和仿真均对 `Q_i=0` 执行 `pi_i=0`，避免空队列链路产生无效干扰。

6. **验证和模型选择**
   - 验证队列使用固定 seed，按硬二值调度 MWS utility 保存并恢复最好 checkpoint。
   - 新增 soft/hard gap、mask 前空队列激活率、队列交换敏感性和 logit 分位数诊断。

### 数据和模型影响

1. 队列 checkpoint 升级为 `results/models/gat_queue_mws_v2_numUAV=<N>.pth`，旧 `gat_queue_numUAV=<N>.pth` 保留但不再自动加载。
2. GAT-q 必须重新训练；GAT-base 模型不受影响。
3. 现有 link-level 队列数据集可继续作为信道/拓扑快照来源，其内固定队列不再直接用于训练 Loss，因此不需要为本次修改重新生成数据集。

### 下一步验证

小型对照已观察到 `queue_hard_decision_change > 0`。下一步用正式训练规模重新训练 GAT-q，再运行 paired Monte Carlo 对比 backlog、delay 和 throughput。若队列敏感性已存在但长期性能仍无改善，再转向两阶段调度。

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
