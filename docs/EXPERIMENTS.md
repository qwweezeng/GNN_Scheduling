# TNSE 当前实验设计与执行顺序

本文档定义投稿版 Fig. 1--8。算法和符号见 [`SYSTEM_MODEL.md`](SYSTEM_MODEL.md)；每幅图的可执行命令见 [`../paper_figures/README.md`](../paper_figures/README.md)。不要使用 `archive_pipeline/` 的历史图或历史数值。

## 1. 全部实验的固定条件

- N=20/30/40/50；队列主图额外明确 N=20、50；
- 10 dBm、2.4 GHz、5 MHz、Rician $K=5$ dB；
- train/validation/test=500/128/128，slot=1 ms，packet=12000 bit；
- 异质 Poisson 到达：20% hotspot links 承担 70% 总业务；
- GAT-MP 的 $K=4$；至少五个配对 Monte-Carlo seed（Fig. 2--4 的训练审计为十个 seed）；
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

横轴 N=20/30/40/50，纵轴 total physical-layer rate (Mbit/s)。比较 Random、Top-SNR (3%)、LocalMax、Local Greedy、Spatial DL、Weighted FPLinQ 与 Proposed-GAT。Spatial DL 只能作为满缓存学习 baseline，必须完整 pad/mask，不可只裁剪它自己的链路。该图不报告 queue utility 或 goodput。

### Fig. 2：Proposed-GAT 训练收敛

两个独立子图：训练 loss--epoch、validation binary rate--epoch。N=20/30/40/50，每条曲线平均十个训练 seed。这里的 validation binary rate 是同一网络的硬二元调度速率；不使用“surrogate rate”作为图标签。

### Fig. 3：Proposed-GAT 输出审计

两个独立子图：连续 $\pi$ 的均值与标准差--epoch；二元激活链路比例--epoch。复用 Fig. 2 的完全相同十个 seed 训练轨迹，不另行训练。该图回答是否发生全开/全关或输出坍缩。

### Fig. 4：Proposed-GAT 数据需求

横轴训练样本数 10/20/30/40/50/60/80/100/200/300/500；纵轴同一规范 test set 的 test binary rate。N=20/30/40/50 每格十个 seed。每个小样本集必须是同一 500-sample pool 的前缀；模型输出一律经 hard feasibility 后再结算。

### Fig. 5：动态队列主结果（第五章）

N=20 和 N=50 分开出图，横轴 traffic arrival rate $\lambda$。每个 N 各有 backlog、completed-packet delay、goodput 三张独立 EPS。方法固定为：random、GAT-MP、Direct-MP、GAT-LocalMax、Direct-Link LocalMax、End-to-End GAT-Q、Hard-Conflict LGS、Centralized Greedy、Weighted FPLinQ。

必须保留两套口径：

1. 无限缓存、无 deadline：未完成包 delay right-censored；
2. 2000 ms deadline：报告 deadline drop/loss；completed-packet delay 只对成功包统计，坐标范围至 2000 ms。

动态主图不以 $\sum_iQ_iR_i$ 或 queue slope 作为性能结论；它们最多是内部机制诊断。

### Fig. 6：Proposed-GAT 跨规模泛化

训练 N=20/30/40/50 的 Proposed-GAT 分别在测试 N=20/30/40/50 上评测。用四条“训练规模”曲线表示 average sum rate，不用热图、不画 active-link fraction。出现 0 rate 时先作为 checkpoint/CSI/decoder gate 排查，不可直接用于论文。

### Fig. 7：CSI 反馈精度

两个独立子图，均含 N=20/30/40/50：

- 第四章 Proposed-GAT 的 test sum rate；
- 第五章 GAT-MP 的 average queue backlog。

量化横轴为 2/4/5/8-bit；perfect CSI 与 large-scale CSI 分别画为边界线，不能误连为量化曲线。非 perfect CSI 出现 0 rate 是实验 gate 失败，不是可解释的结果。

### Fig. 8：移动性与 CSI 年龄

四张独立 EPS，均含 N=20/30/40/50 和 latest(perfect)/one-block stale：

- Proposed-GAT sum rate 随 UAV speed；
- Proposed-GAT sum rate 随 block length；
- GAT-MP backlog 随 UAV speed；
- GAT-MP backlog 随 block length。

每个信道块只计算一次 Proposed-GAT 并缓存 $\pi/C^{\rm GAT}$；GAT-MP 每时隙只结合当前 $Q$。服务始终在当前真实移动 Rician 信道上结算。

## 4. baseline 支持范围

| 方法 | Fig. 1 | Fig. 5 | Fig. 7/8 | 备注 |
|---|---:|---:|---:|---|
| Random / LocalMax / Local Greedy | 是 | Random 是 | 否 | 本项目参考方法 |
| Weighted FPLinQ | 是 | 是 | 可选 | 集中式全 CSI，100 iterations |
| Spatial DL | 是 | 否 | 否 | legacy 50-link checkpoint，pad/mask 零样本迁移 |
| DistGCN-LGS | 可选 | 可选 | 否 | zero-shot hard-graph adaptation，必须同时有 Hard-Conflict LGS |
| Graph Embedding | 否 | 否 | 否 | 已移除，不阻塞投稿 |
| End-to-End GAT-Q | 否 | 是 | 否 | 必须有 N 对应 checkpoint，否则 fail closed |

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
