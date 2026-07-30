# TNSE 当前实验设计与执行顺序

本文档定义投稿版 Fig. 1--8。算法和符号见 [`SYSTEM_MODEL.md`](SYSTEM_MODEL.md)；每幅图的可执行命令见 [`../paper_figures/README.md`](../paper_figures/README.md)。不要使用 `archive_pipeline/` 的历史图或历史数值。

## 1. 全部实验的固定条件

- N=20/30/40/50；队列投稿主图固定 N=50；如不做特殊声明，令epoch=50.
- 10 dBm、2.4 GHz、5 MHz、Rician $K=5$ dB；
- train/validation/test=300/128/128，slot=1 ms，packet=12000 bit；
- 异质 Poisson 到达：20% hotspot links 承担 70% 总业务；
- GAT-MP 的 $K=4$；至少五个配对 Monte-Carlo seed；Fig. 2--4 的训练审计当前采用三个模型初始化 seed，只有标准误过大时才补充 seed；
- 所有方法共享拓扑、真实信道、arrival trace、link set 和 seed；只改变调度器可见 CSI；
- 所有最终动作必须是二元，并通过 single-radio/half-duplex hard feasibility；
- 正式图只用 `figXX_*/run.py` 写原始数据，再用 `plot.py` 画独立 EPS。

## 2. 名称、内部实现与显示名称

| 显示名称 | 内部实现/JSON 名称 | 角色 |
|---|---|---|
| Proposed-GAT | `GATScheduler` | 第四章主方案 |
| GAT-MP | `GATAssistedMarginalMP`；`GAT-Assisted MarginalMP` | 第五章主方案 |
| Direct-MP | `DirectLinkMarginalMP`；`Direct-Link MarginalMP` | 必要去 GAT 消融 |
| GAT-LocalMax | `GATAssistedLocalMaxScheduler` | 结构消融 |
| Direct-Link LocalMax | `DirectLinkLocalMaxScheduler` | 无 GAT 骨架参考 |
| End-to-End GAT-Q | queue-input GAT checkpoint | 重要学习 baseline |

## 3. Fig. 1--8

### Fig. 1：满缓存总物理层速率（第四章）

横轴 N=20/30/40/50，纵轴 total physical-layer rate (Mbit/s)。比较 Random、Top-SNR (2%)、Local Greedy、Spatial DL、Weighted FPLinQ 与 Proposed-GAT。该指标是共同 test snapshots 上的平均 hard-feasible binary aggregate physical-layer rate：Proposed-GAT 输出先 `pi>=0.5`，再 endpoint mutual matching/shared-UAV hard feasibility，最后在真实 test channel 上结算；不是 soft rate、queue utility 或 goodput。Spatial DL 只能作为满缓存学习 baseline，必须完整 pad/mask，不可只裁剪它自己的链路。

### Fig. 2：Proposed-GAT 训练收敛

两个独立子图：训练 loss--epoch、validation binary rate--epoch。N=20/30/40/50，每条曲线平均 3 个训练 seed。这里的 validation binary rate 是 `pi>=0.5` 后的二元调度速率，按训练日志的 binary validation 口径记录并换算为 Mbit/s；不使用连续 `pi` 的 soft utility 或“surrogate rate”作为图标签。训练日志另外保存 soft 和 projected hard-feasible 诊断指标，但不替代主图 binary 曲线。

### Fig. 3：Proposed-GAT 输出审计

两个独立子图：连续 $\pi$ 的均值与标准差--epoch；二元激活链路比例--epoch。复用 Fig. 2 的完全相同 N=20/30/40/50、seed=0/1/2 共 12 条训练轨迹，不另行训练。该图回答是否发生全开/全关或输出坍缩。

### Fig. 4：Proposed-GAT 数据需求

横轴训练样本数 10/20/30/40/60/80/100/200/300/500，存储训练后所有epoch中效果最好的一个checkpoint；纵轴同一规范 test set 在这个最好checkpoint上的 hard-feasible projected binary rate。N=20/30/40/50 每格 3 个 seed。每个小样本集必须是同一 500-sample pool 的前缀；模型输出先 `pi>=0.5` 二值化，再经过 endpoint mutual matching/shared-UAV hard feasibility，最后在真实 test channel 上结算。`n=300` 复用 Fig. 2 的相同 `(N, seed)` checkpoint，不重复训练。同时存储每种情况下训练过程中的 best_val_util、best checkpoint 和 last checkpoint 作用于 test 集后得到的 soft test utility（pi 没有经过二元化）、hard test utility（pi 经过二元化但不做 hard feasibility）、以及 projected hard-feasible binary rate，这些数值和诊断图不得混为同一指标。

### Fig. 5：动态队列主结果（第五章）

投稿主图仅使用 N=50、2000 ms deadline，横轴 traffic arrival rate $\lambda$ 取 0.5/1.0/1.5/2.0/2.2/2.4/2.6/2.8/3.0/3.2/3.5/4.0/4.5/5.0。分别绘制 backlog、resolved-packet deadline-capped latency、goodput 三张独立 EPS。backlog 是所有链路 queue 的时间平均总 backlog（Mbit）；goodput 是仿真结束前真正完成交付的 packet bits/仿真时长，不是 physical service 或 $\sum_iQ_iR_i$ utility。时延图对成功完成包使用实际 FIFO 时延，对 deadline/buffer drop 记为 2000 ms，并仅以已完成或已丢弃的 resolved packets 为分母；未决包通过 backlog 与 goodput 呈现。方法固定为：Random、Centralized Greedy、Weighted FPLinQ、Direct-LocalMax、GAT-LocalMax、DirectMP、GAT-MP。当前 README 中的 N=50 无 deadline 数据仅作内部诊断，不纳入投稿图。

必须保留两套口径：

1. **无限缓存、无 deadline（内部诊断）**：只对仿真结束前已完成的包统计 FIFO completion delay；未完成包为 right-censored，因此高负载下不可将该条件均值作为 QoS 主结论。
2. **2000 ms deadline（投稿 Fig. 5）**：delay 面板报告 resolved-packet deadline-capped latency。对每个已完成包 $p$ 使用其实际 FIFO completion delay $D_p$；对 deadline drop 和 buffer-overflow drop 赋固定惩罚 $D_{\max}=2000$ ms。令 $\mathcal{C}$、$\mathcal{D}$ 分别为已完成包和已丢弃包集合，则

   $$\bar D_{\rm cap}=\frac{\sum_{p\in\mathcal{C}}D_p+|\mathcal{D}|D_{\max}}{|\mathcal{C}|+|\mathcal{D}|}.$$

   分母只包含已经完成或已经丢弃的 resolved packets；仿真结束时仍在队列中的未决包不被当作成功包或丢包，其拥塞影响由 backlog 和 goodput 两个主图共同体现。该定义避免了 completed-packet delay 的幸存者偏差：过载方法若提前丢弃长等待包，不能因只保留少量快速完成包而获得虚假的低时延。纵轴固定为 $[0,2000]$ ms。completed-packet delay 继续存储在原始 JSON 与 `_completed_only` 内部诊断图中，但不用于投稿主图或性能排序。

动态主图不以 $\sum_iQ_iR_i$ 或 queue slope 作为性能结论；它们最多是内部机制诊断。

### Fig. 6：Proposed-GAT 跨规模泛化

训练 N=20/30/40/50/混合20、30、40、50数据 的 Proposed-GAT 分别在测试 N=20/30/40/50 上评测。train_sample=500。用训练来源曲线表示 average hard-feasible binary aggregate sum rate；若 JSON 含 mixed checkpoint，实际包含 N=20/30/40/50/mixed 五条曲线，不用热图、不画 active-link fraction。每个单元格是测试样本上的平均 `mean_aggregate_phy_rate_mbps`：`pi>=0.5` 后 endpoint mutual matching，确认 shared-UAV feasibility，再在 test-N 真实 Rician channel 上结算。出现 0 rate 时先作为 checkpoint/CSI/decoder gate 排查，不可直接用于论文。

### Fig. 7：CSI 反馈精度

两个独立子图，均含 N=20/30/40/50：

- 第四章 Proposed-GAT 的 test sum rate；
- 第五章 GAT-MP 的 average queue backlog。

量化横轴为 2/4/6/8/10-bit，均画为每个 N 的实线；perfect CSI 同色点线、large-scale CSI 同色虚线，均为参考线而不连入量化曲线。Chapter 4 的 rate 是 hard-feasible binary aggregate physical-layer sum rate，不是 soft rate，perfect CSI 是上界参考。Chapter 5 的 backlog 是 `average_total_queue_backlog_mbit`，即固定 lambda=2.4、speed=0、block=1 下的时间平均总队列积压；backlog 越低越好，因此 perfect CSI 是低 backlog 参考，而非 upper bound。large-scale CSI 不是有限 Monte Carlo 下的严格数学上界/下界。非 perfect CSI 出现 0 rate 是实验 gate 失败，不是可解释的结果。

### Fig. 8：移动性与 CSI 年龄

2张独立 EPS，均含 N=20/30/40/50 和 latest(perfect)/one-block stale：

- GAT-MP backlog 随 UAV speed；
- GAT-MP backlog 随 block length。

每个信道块只计算一次 Proposed-GAT 并缓存 $\pi/C^{\rm GAT}$；GAT-MP 每时隙只结合当前 $Q$。两张图的纵轴均为 `average_total_queue_backlog_mbit`（所有链路 queue 的时间平均总 backlog），不报告 sum rate、goodput 或 soft utility。服务始终在当前真实移动 Rician 信道上结算。

## 4. baseline 支持范围

| 方法 | Fig. 1 | Fig. 5 | Fig. 7/8 | 备注 |
|---|---:|---:|---:|---|
| Random / LocalMax / Local Greedy | 是 | Random 是 | 否 | 本项目参考方法 |
| Weighted FPLinQ | 是 | 是 | 可选 | 集中式全 CSI，100 iterations |
| Spatial DL | 是 | 否 | 否 | legacy 50-link checkpoint，pad/mask 零样本迁移 |
| DistGCN-LGS | 可选 | 可选 | 否 | zero-shot hard-graph adaptation，必须同时有 Hard-Conflict LGS |
| Graph Embedding | 否 | 否 | 否 | 已移除，不阻塞投稿 |
| End-to-End GAT-Q | 否 | 否 | 否 | 当前投稿 Fig. 1--8 不使用；保留实现与 checkpoint 加载逻辑，不补充实验数据 |

## 5. Gate 0：运行长实验前的最小检查

先在 N=20、两个 seed、短 slot 数下运行相应 `run.py`，确认：

1. packet/work 守恒；
2. 最终二元动作；
3. link identity 一致；
4. shared-UAV hard feasibility；
5. JSON 中有 config、checkpoint、fingerprint、seed、method variant；
6. 所有方法使用同一真实信道和到达轨迹。

通过后再启动五 seed、长 slots 或 Fig. 2/4 的十 seed 训练。第三方依赖异常必须如实停止并记录，不能替换为名称相似的自写算法。

## 6. 当前状态与下一步

- 主模型已冻结：不再做 GAT 结构搜索、GRU、CG-CSI、码本或新的 teacher 分支；
- Fig. 1--8 的新 `run.py/plot.py` 已建立，旧 pipeline 与旧图已移至 `paper_figures/archive_pipeline/`；
- 下一步是按 Fig. 1 至 Fig. 8 的顺序运行 Gate 0、审计原始 JSON、再绘制 EPS；
- 若必须改 GAT 输入或数据生成方式，先在 N=20 最小成本验证，并同步更新本文件、`SYSTEM_MODEL.md`、`README.md` 和 JSON metadata，不能静默改口径。
