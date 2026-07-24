# 服务器新 Chat 交接：TNSE UAV 链路调度仿真

## 任务边界

你正在维护 `/Users/zqz/Documents/Codex/GNN_Scheduling` 的 TNSE 投稿仿真代码。只做仿真、baseline 对齐、结果核验、作图和 README 复现命令；不要扩展新模型主线。冻结基准为 `ad2ee67d76e05effa4953db18ab677502ddd3ba1`，但当前工作树已经为投稿整理而修改；以当前 `README.md`、`docs/SYSTEM_MODEL.md`、`docs/EXPERIMENTS.md` 为活动定义。

## 当前两条主线

1. **第四章 Proposed-GAT**：`models/model.py` 的 `GATScheduler`。满缓存信道优先级学习，输入直达 SNR 与有向干扰图，输出 $\pi$，最终一轮端点互选保证 matching。训练脚本：`train/train_channel_gat_rician.py`。
2. **第五章 GAT-MP**：`models/marginal_mp.py` 的 `GATAssistedMarginalMP`。每个信道块缓存 Proposed-GAT 的 $\pi/C^{\rm GAT}$；每时隙结合当前队列，用 LocalMax 骨架和 K=4 边际效用消息传递。固定/移动队列入口分别为 `experiments/exp_fixed_csi_qos.py` 和 `experiments/evaluate_mobile_block_fading_csi.py`。

论文显示名称与内部名不同：

| 显示名 | 内部名 |
|---|---|
| Proposed-GAT | `GATScheduler` |
| GAT-MP | `GATAssistedMarginalMP` / `GAT-Assisted MarginalMP` |
| Direct-MP | `DirectLinkMarginalMP` / `Direct-Link MarginalMP` |
| End-to-End GAT-Q | queue-input GAT baseline |

## 不要做的事

- 不重新开启 GRU、CG-CSI、Graph Embedding、码本、teacher、边编码 GAT 或复杂 GAT 结构搜索；
- 不把 Centralized Greedy 称为上界；
- 不把 FPLinQ/Spatial DL 称为分布式方法；
- 不因为 GAT-MP 有 GAT 先验就宣称全部性能收益来自 GAT；
- 不让任何方法跳过 shared-UAV hard feasibility；
- 不使用 `paper_figures/archive_pipeline/` 的历史结果当论文证据。

## 正式设置与公平性

N=20/30/40/50，10 dBm、2.4 GHz、5 MHz、Rician K=5 dB，500/128/128，slot=1 ms，packet=12000 bit，20% hotspot links 承担 70% Poisson 业务。所有方法共享拓扑、真实信道、arrival trace、link set 和配对 seed。调度器的 CSI 可变，但服务始终用当前真实信道结算。最终动作必须经 `shared_uav_action_audit`。

## 图与入口

图的阶段命令统一在 `paper_figures/README.md`：A 数据、B 训练、C 仿真结果、D 仅绘图。Fig. 1--8 使用各自 `run.py` 与 `plot.py`；每个子图独立 EPS。历史 pipeline 和每图一张历史双栏 EPS 已隔离到 `paper_figures/archive_pipeline/`。

下一步按 Fig. 1 至 Fig. 8 执行：先 N=20、2 seed、短 slots 的 Gate 0，确认守恒、二元动作、link identity、metadata、checkpoint 与 hard feasibility；再扩展到正式规模和 seed 数。

## 必要 baseline

- FPLinQ：`baseline/fplinq_numpy.py`，独立 `gnn-fplinq`；
- Spatial DL：仅 Fig. 1，独立 `gnn-spatial-dl-legacy`，pad/mask 完整 link set；
- DistGCN-LGS：可选队列学习参考，独立 `gnn-distgcn-legacy`，必须与 Hard-Conflict LGS 成对解释；
- Graph Embedding 已移除，不阻塞投稿。

## 当前需要优先核验的点

1. Fig. 5 的 End-to-End GAT-Q 需要 N=20、N=50 匹配 checkpoint；没有时必须 fail closed，不得替代；
2. Fig. 7 的非 perfect CSI 曾出现 0 rate，先检查 CSI adapter、图构建和 decoder，再出正式图；0 不是可解释的正式数据点；
3. Fig. 2/3 需要四个 N、十个 seed 的同一训练轨迹；Fig. 4 是 4×11×10 数据量网格，计算成本高；
4. 如确有必要改 GAT 输入或数据生成方式，只允许先 N=20 最小成本验证，并同步更新 `SYSTEM_MODEL.md`、`EXPERIMENTS.md`、`README.md` 和所有结果 metadata。
