# GNN Scheduling：TNSE 论文代码与仿真仓库

本仓库维护论文第四章、第五章的代码和仿真证据。仓库内的 `TNSEmanuscript0730/` 是与代码一起审计、编译的论文镜像；独立 Git 仓库 `/Users/zqz/Documents/Codex/TNSE_Submission_2026/manuscript/` 是最终投稿稿件的规范维护位置。两者同步时，以经过本仓库结果核验的新稿为来源，并保留投稿仓库自身的版本历史。

当前论文的主线分为两个阶段：

- **第四章，edge-enhanced GAT**：在信道块尺度上，根据链路特征和有向干扰图学习 interference-aware spatial reuse。模型输出连续激活倾向，论文中将这些输出及其导出的容量估计统称为 **GAT-derived interference prior**。
- **第五章，GAT-MP**：在每个时隙把上述先验与当前队列状态结合，先形成初始调度骨架，再通过 **queue-aware marginal coordination** 调整活动链路集合。

代码中仍保留了一些历史名称，例如 `GATScheduler`、`Proposed-GAT` 和 `GAT-Assisted MarginalMP`。阅读代码时应按下表理解，不要因为内部名称不同而把它们当成额外算法。

| 论文中的名称 | 主要代码对象 | 作用 |
|---|---|---|
| Edge-enhanced GAT | `models.model.GATScheduler` | 第四章的信道驱动调度模型 |
| GAT-derived interference prior | GAT 输出 `pi` 及由其形成的 expected-interference/capacity score | 第五章初始骨架的学习先验 |
| GAT-MP | `models.marginal_mp.GATAssistedMarginalMP` | 第五章的两阶段队列感知方案 |
| GAT-LocalMax | `models.initial_schedule_construction.GATAssistedLocalMaxScheduler` | 只保留学习先验和初始骨架的消融 |
| Direct-MP | `models.marginal_mp.DirectLinkMarginalMP` | 使用 direct-link score、保留相同协调过程的消融 |
| Direct-Link LocalMax | `models.initial_schedule_construction.DirectLinkLocalMaxScheduler` | 不使用 GAT 和边际协调的局部参考方法 |

## 当前最重要的代码同步状态

论文第五章已经把边际协调整理为一种基于冲突图邻域的协议：每条链路是一个 state-flip candidate，局部接收机返回 queue-weighted rate variation，候选更新需要协调邻域确认，互不影响的候选可以并行翻转。

**当前代码已经完成这次同步。** `models/marginal_mp.py` 采用 conflict-graph-local marginal、协调邻域确认和加入前 endpoint confirmation；GAT-MP 每轮更新不再依赖事后集中式 projection。对应实现提交为 `6180c4d`，协议版本为 `conflict_graph_local_coordination_v1`。

新版 Fig. 5 和 Fig. 7 Chapter V 正式结果保存在 `results/paper/ch5_local_coordination_v1/`。最终审计 53/53 通过，包括局部效用不下降、accepted neighborhoods 两两不相交、所有时隙 shared-UAV feasible、零事后 projection，以及 Fig. 7 的 packet/work conservation 检查。第四章训练代码、GAT checkpoint 和训练数据在本次同步中保持冻结，没有重新训练或覆盖。

绘图时不要默认读取 `paper_figures/*/data/raw/` 中的历史 Chapter V 结果。应显式传入新版 raw result root 和独立 output root。当前 Fig. 1--7 的统一最新版位于 `results/paper/tnse_current_plots/`；从“没有数据”到“只画图”的完整命令见 `docs/PAPER_FIGURE_PIPELINE.md`，Chapter V 专项核验见 `docs/CHAPTER5_FINAL_PLOTTING.md`。

## 运行环境

主环境由 `pyproject.toml` 和 `uv.lock` 管理，要求 Python 3.12 或更高版本。推荐从仓库根目录执行：

```bash
uv sync --locked
uv run python -c "import torch, numpy, scipy, matplotlib, yaml; print('environment: pass')"
uv run python -m unittest discover -s tests
```

训练和长时队列仿真可使用 CUDA：

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

所有正式命令都应通过 `uv run python ...` 在仓库根目录运行。CPU 可以完成模型推理和小规模检查，但大规模训练与长时队列仿真会明显更慢。

第三方 baseline 不应迫使主环境降级：

- Weighted FPLinQ 的主代码使用 NumPy/SciPy 适配器；
- Spatial DL 和 DistGCN 依赖旧版 TensorFlow，应使用 `baseline/environments/` 中的独立环境；
- 具体环境文件、smoke test 和比较边界见 `baseline/README.md`。

## 论文与代码的对应关系

| 论文部分 | 主要实现 | 主要验证 |
|---|---|---|
| Section III：系统、信道、干扰图与队列模型 | `utils/network_generate.py`、`simulator/`、`utils/fixed_csi_benchmark.py` | 数据 metadata、信道/速率测试、队列守恒测试 |
| Section IV：edge-enhanced GAT | `models/model.py`、`train/train_channel_gat_rician.py`、`utils/feasibility.py` | Fig. 1--4、cross-scale table、Fig. 7 Chapter IV |
| Section V-A/V-B：两阶段结构和初始骨架 | `models/initial_schedule_construction.py`、`models/local_physical_reuse.py` | Fig. 5 中 GAT-LocalMax、Direct-Link LocalMax 及两个 MP 方案 |
| Section V-C/V-D：边际协调与分布式执行 | `models/marginal_mp.py` | 局部 marginal、邻域确认、endpoint confirmation；Fig. 5、Fig. 7 Chapter V |
| Section VI：数值结果 | `experiments/` 负责底层评测，`paper_figures/` 负责逐图编排和绘图 | 论文中的全部正式结果；`fig08_mobility` 仅作诊断 |

## 代码架构

```text
GNN_Scheduling/
├── configs/
│   └── paper.yaml                    # 论文的规范物理、模型和训练参数
├── datasets/
│   ├── paper/                        # TNSE 正式 Rician 数据及 queue baseline 数据
│   └── conference_rayleigh/          # 会议版历史 Rayleigh 数据，不作期刊主证据
├── checkpoints/paper/proposed_gat/   # Fig. 2/4 使用的 seed-resolved 双 checkpoint
├── results/models/paper/             # 标准路径和 CSI 条件模型；也含历史模型文件
├── models/                            # GAT、初始骨架、GAT-MP 和局部复用逻辑
├── train/                             # Edge-enhanced GAT 与保留 baseline 的训练入口
├── simulator/                         # 固定/移动 block fading、真实信道服务与队列仿真
├── baseline/                          # Random、Greedy、FPLinQ、Spatial DL、DistGCN 等
├── experiments/                       # 可复用的底层评测器，详见 experiments/README.md
├── paper_figures/                     # 每个投稿结果的 run/resume/plot 封装和原始结果
├── scripts/                           # 数据准备、artifact 审计和辅助脚本
├── tests/                             # 物理口径、可行性、队列和实验接口测试
├── docs/                              # 系统模型、实验设计、代码审查和交接材料
└── TNSEmanuscript0730/                # 与代码同步审计的论文镜像；最终稿维护在姐妹投稿仓库
```

### 关键模块的职责

- `models/model.py`：edge-enhanced GAT 的网络结构、连续输出和可微分速率目标。
- `utils/feasibility.py`：endpoint-mutual feasibility 和 shared-UAV 可行性审计。最终调度必须满足单射频/半双工端点约束。
- `models/local_physical_reuse.py`：根据 direct-link score 或 GAT-derived score 形成 LocalMax 初始集合。
- `models/initial_schedule_construction.py`：给第五章使用的、命名更清楚的初始骨架接口。
- `models/marginal_mp.py`：第五章边际协调。该文件是下一轮代码修改的中心。
- `simulator/mobile_block_fading.py`：生成移动 block-fading trace。调度器看到 observed CSI，服务结算使用同一时隙的 true CSI。
- `utils/fixed_csi_benchmark.py`：生成固定信道、异质 Poisson 到达和 paired trace，并统计 backlog、delay、goodput 等指标。
- `experiments/*.py`：将模型、数据、baseline 和 simulator 组合成可复用评测任务。
- `paper_figures/figXX_*/run.py`：为论文某一结果固定参数并调用底层评测器；`plot.py` 只读取结果作图。

## 数据、模型与结果存放位置

### 正式静态数据

TNSE 主实验使用：

```text
datasets/paper/rician_K5_tx10/{perfect,large_scale,quantized_2bit,...}/N{20,30,40,50}/
```

典型目录包含 train/validation pickle、test pickle 和 `manifest.json`。加载器会检查物理参数、数据规模、seed 和 fingerprint；不能只根据文件名判断某个数据集可复用。

`datasets/conference_rayleigh/` 是会议版历史数据。它可以用于追溯旧实验，但不能作为当前 TNSE Rician 结果的数据来源。

固定队列 trace 和移动 trace 不依赖一个统一的大型离线数据集：前者由 `utils/fixed_csi_benchmark.py` 按实验参数生成或加载，后者由 `simulator/mobile_block_fading.py` 在线生成。同一次比较中，各方法必须共享 topology、link set、channel trace、arrival trace 和 seed。

### 模型 checkpoint

第四章的可复现训练保存在：

```text
checkpoints/paper/proposed_gat/rician_K5_tx10/
└── N{N}/seed{seed}/{full_300,prefix_nXX}/
    ├── best_model.pth
    └── final_model.pth
```

其中 `full_300` 是 Fig. 2 的规范训练，Fig. 1 直接复用 seed 0 的 best checkpoint；`prefix_nXX` 用于 Fig. 4 的数据量实验。

`results/models/paper/` 保存按网络规模和 CSI 条件命名的标准 checkpoint，例如 perfect、large-scale 和不同量化 bit 的模型，也保留部分历史兼容路径。使用前必须读取 checkpoint metadata，确认训练规模、CSI 条件、信道和数据 fingerprint，而不能把同名 `.pth` 当成可无条件互换的模型。

### 实验结果和图片

论文结果按图存放在：

```text
paper_figures/figXX_name/
├── data/raw/       # 仿真或评测产生的 JSON/NPZ；判断是否需要重跑的主要依据
├── data/plotting/  # 可选的绘图中间表格
├── figures/        # EPS 等可直接放入论文的图片
├── run.py          # 数据、训练或评测编排
├── resume.py       # 长任务的断点续跑入口，若该目录提供
└── plot.py         # 只读已有结果并绘图
```

`results/paper/` 和实验脚本的默认输出路径主要用于通用运行；投稿图应以各 `paper_figures/figXX_*/data/raw/` 中、且 provenance 与当前代码相符的结果为准。

## 四阶段工作流

任何实验开始前，先判断目前位于哪一个阶段。不要因为看见一条完整命令就从头重新生成数据。

| 阶段 | 已有资产 | 缺少资产 | 正确动作 |
|---|---|---|---|
| A：没有数据集 | 无匹配 dataset | dataset、checkpoint、result | 只生成并验证数据 |
| B：有数据，未训练 | dataset | 匹配 checkpoint、result | 训练模型并保存 metadata |
| C：有 checkpoint，未评测 | dataset、checkpoint | result JSON/NPZ | 运行底层评测，不重复训练 |
| D：已有有效结果 | dataset、checkpoint、与当前实现匹配的 result | figure | 只运行 `plot.py` |

“文件存在”不等于“阶段已经完成”。至少要检查：

- dataset 的物理参数、seed、split size 和 fingerprint；
- checkpoint 的训练规模、CSI 条件、数据 fingerprint 和权重类型；
- result 的方法集合、网络规模、arrival rate、slot 数、seed 和 checkpoint provenance；
- 产生该 result 的算法实现是否已经变化。

只改坐标、图例、颜色或排版时，停留在阶段 D。修改模型推理、解码、边际协调、信道生成、队列更新或指标定义后，旧结果通常失效，应从阶段 C、B 或 A 中相应位置重新开始。

## 修改代码后的验证顺序

1. 先运行与修改模块直接相关的 `tests/`，确认速率、队列守恒和 endpoint feasibility 没有破坏。
2. 用很小的 N、slot 数和单个 seed 做 smoke run，只检查执行路径和 JSON schema，不用它支撑论文结论。
3. 在正式 dataset/checkpoint 上运行对应的 `experiments/*.py` 或 `paper_figures/figXX_*/run.py`。
4. 将新结果与修改前结果比较，解释变化来自算法修正、随机误差还是口径变化。
5. 确认结果文件的 provenance 后再运行 `plot.py`，最后核对论文中的数值和文字。

修改影响范围可按下面判断：

| 修改内容 | 至少应复查的正式结果 |
|---|---|
| GAT 结构、输入特征或训练损失 | Fig. 1--4、cross-scale table、Fig. 7 Chapter IV；第五章若复用新 prior 也应重跑 |
| endpoint-mutual feasibility 或二值解码 | Fig. 1、Fig. 4 诊断、cross-scale table、Fig. 7 Chapter IV，以及所有使用该动作的队列结果 |
| GAT-MP 初始骨架或边际协调 | Fig. 5、Fig. 7 Chapter V；Fig. 8 作为可选诊断 |
| 固定队列、deadline、delay 或 goodput 统计 | Fig. 5 |
| observed/true CSI、移动信道或 CSI 量化 | Fig. 7；必要时用 Fig. 8 辅助定位 |
| 纯绘图样式 | 只需重新运行相应 `plot.py` |

## 论文口径与证据边界

- 第四章最终二元动作必须经过 endpoint-mutual feasibility，不能把未投影的 threshold 输出当成可部署物理速率。
- 第五章的 Lyapunov 推导提供设计指导，论文没有声称 GAT-MP 具有 throughput optimality 或严格的长期队列稳定性保证。代码实验支持的是 backlog、deadline-capped delay 和 goodput 所反映的 QoS 改善。
- Weighted FPLinQ 和 Centralized Greedy 是集中式参考，不是数学上界；Spatial DL 的运行环境和输入域也必须如实保留。
- imperfect-CSI 实验中，调度决策使用指定的 observed CSI，数据传输服务应在 realized true CSI 上结算。
- cross-scale 结果支持的是所测试规模之间的 scale generalization，不应扩展成任意网络规模或任意 CSI 分布下的保证。
- `paper_figures/fig08_mobility/` 当前没有形成稳定、清晰的论文结论，因此不进入投稿稿件，只保留为移动性和陈旧 CSI 的诊断实验。

## 推荐阅读顺序

1. `/Users/zqz/Documents/Codex/TNSE_Submission_2026/manuscript/tnse_uav_scheduling.tex`：最终投稿稿件的规范维护位置；
2. `TNSEmanuscript0730/tnse_uav_scheduling.tex`：与本仓库代码和结果同步的审计镜像；
3. 本 README：项目边界、目录和资产流；
4. `docs/PAPER_FIGURE_PIPELINE.md`：四阶段顺序运行清单；
5. `experiments/README.md`：每个正式实验的设置、目标、输入和输出；
6. `configs/paper.yaml` 与具体 `paper_figures/figXX_*/run.py`：实际运行参数；
7. `PAPER_CODE_REVIEW_REPORT.md`：已发现问题及其证据；
8. 具体模型、simulator 和测试文件。

详细的逐图命令可继续参考 `paper_figures/README.md`。若 README、论文和运行脚本出现冲突，应先以当前论文明确的算法定义作为目标，再以源代码和结果 metadata 判断“现在实际实现了什么”，不要静默地把二者混在一起。
