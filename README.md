# Dense-UAV Link Scheduling：TNSE 投稿仿真仓库

本仓库只服务于一篇 TNSE 期刊稿的两条模型主线与其必要比较：

- **第四章 Proposed-GAT**：满缓存、瞬时信道条件下的链路优先级学习器。它输入链路直达 SNR 与有向干扰图，输出连续优先级 $\pi_i$；最终二元动作通过端点互选保证单射频/半双工可行性。
- **第五章 GAT-MP**：队列感知主方案，即代码类 `GATAssistedMarginalMP`。它每个信道块缓存 Proposed-GAT 的 $\pi$ 与 $C^{\rm GAT}$，每个时隙再结合当前队列 $Q$，以固定 $K=4$ 轮边际效用消息传递决定动作。

不是主模型的内容只保留为比较方法：Random、Top-SNR (3%)、LocalMax、Local Greedy、Hard-Conflict LGS、Centralized Greedy、Weighted FPLinQ、Spatial DL、DistGCN-LGS、Direct-MP、GAT-LocalMax、Direct-Link LocalMax 与 End-to-End GAT-Q。仓库中不存在 GRU、CG-CSI、Graph Embedding 或任何 GAT 结构搜索支线。

## 先读什么

1. [`docs/SYSTEM_MODEL.md`](docs/SYSTEM_MODEL.md)：统一符号、物理模型、两条主线和硬可行性边界；
2. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)：Fig. 1--8 的目的、方法集合、数据/模型/结果/作图四阶段；
3. [`paper_figures/README.md`](paper_figures/README.md)：每一幅图可直接粘贴的运行命令；
4. [`docs/NEXT_CHAT_HANDOFF.md`](docs/NEXT_CHAT_HANDOFF.md)：交给服务器新 chat 的任务交接；
5. [`configs/paper.yaml`](configs/paper.yaml)：唯一正式物理和训练配置。

## 术语与代码对应（必须统一使用）

| 论文/图中名称 | 代码对象或原始 JSON 名称 | 用途 |
|---|---|---|
| Proposed-GAT | `models.model.GATScheduler` | 第四章满缓存主模型 |
| GAT-MP | `models.marginal_mp.GATAssistedMarginalMP`；运行器内部名 `GAT-Assisted MarginalMP` | 第五章队列主方案 |
| Direct-MP | `DirectLinkMarginalMP`；内部名 `Direct-Link MarginalMP` | 去除 GAT 的必要消融 |
| GAT-LocalMax | `GATAssistedLocalMaxScheduler` | 只保留 GAT 骨架的结构消融 |
| Direct-LocalMax | `DirectLinkLocalMaxScheduler` | 无 GAT 的骨架参考 |
| End-to-End GAT-Q | `train/train_end_to_end_queue_gat.py` 训练的队列输入 GAT | 重要学习 baseline，不是主方案 |

当前运行器和绘图脚本都直接写入上表的论文名称；历史 archive 中的旧内部名不参与正式图。

## 目录结构

```text
GNN_Scheduling/
├── configs/paper.yaml                 # 唯一正式配置
├── datasets/paper/                    # 规范 Rician K=5、10 dBm 数据集
├── results/models/paper/              # 仅 N=20/30/40/50 正式 Proposed-GAT checkpoint
├── models/                            # GAT、GAT-MP、LocalMax/物理复用
├── train/                             # Proposed-GAT 与 End-to-End GAT-Q 训练
├── simulator/                         # 静态/移动 Rician、队列、真实 CSI 服务结算
├── baseline/                          # FPLinQ、Spatial DL、DistGCN 与独立 legacy 环境
├── experiments/                       # 满缓存、固定队列、CSI、移动与跨 N 的底层运行器
├── paper_figures/                     # 投稿 Fig. 1--8 的 run.py/plot.py
│   └── archive_pipeline/              # 上一版 pipeline 源码与每图一张历史双栏 EPS，仅追溯
├── scripts/                           # 数据准备、trace cache、artifact 核验
├── tests/                             # 当前主线与 baseline 适配测试
└── docs/                              # 当前方法、实验设计与交接文档
```

`paper_figures/archive_pipeline/` 不读取、不训练、不作为论文证据；它只避免上一版代码和图完全丢失。正式图只使用 `paper_figures/fig01_full_buffer/` 至 `fig08_mobility/` 中的 `run.py` 与 `plot.py`。

## 正式物理口径

- N=20/50 为队列主规模；满缓存/泛化使用 N=20/30/40/50；
- 10 dBm、2.4 GHz、5 MHz、Rician $K=5$ dB；
- 500/128/128 train/validation/test，slot=1 ms，packet=12000 bit；
- 20% hotspot 链路承载 70% Poisson 到达，无限缓存；另提供 2000 ms deadline 版本并报告丢包；
- 所有方法共享拓扑、真实信道、arrival trace、link set 和配对 seed；调度器只可看到其指定 CSI，服务永远以当前真实信道结算；
- 每个最终二元动作都必须通过 `shared_uav_action_audit`。

## 最短入口

先看 [`paper_figures/README.md`](paper_figures/README.md)。它将每幅图严格分为：

1. 没有数据：生成数据；
2. 有数据、没有模型：训练模型；
3. 有数据和模型、没有结果：运行 baseline/仿真；
4. 结果已在：只重画 EPS。

例如 Fig. 1 的四阶段分别为：

```bash
conda run --no-capture-output -n pytorch python -u paper_figures/fig01_full_buffer/run.py --stage data
conda run --no-capture-output -n pytorch python -u paper_figures/fig01_full_buffer/run.py --stage train
conda run --no-capture-output -n pytorch python -u paper_figures/fig01_full_buffer/run.py --stage evaluate --force
conda run --no-capture-output -n pytorch python -u paper_figures/fig01_full_buffer/plot.py
```

## baseline 环境

主 PyTorch 环境不得降级。FPLinQ 使用 NumPy/SciPy 适配器；Spatial DL 与 DistGCN 使用独立 legacy conda 环境。详细环境文件、smoke 命令和比较边界见 [`baseline/README.md`](baseline/README.md)。

## 证据边界

Proposed-GAT 是第四章的信道优先级学习模型；GAT-MP 是第五章主方案。Direct-MP 是必须保留的去 GAT 消融。不要把 Centralized Greedy 写成上界，不要把 FPLinQ 或 Spatial DL 写成分布式方法，也不要仅因 GAT-MP 使用 GAT 先验就把全部队列收益因果归因给 GAT。
