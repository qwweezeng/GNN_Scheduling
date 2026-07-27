# TNSE 投稿仿真图：中文复现总说明

这里是投稿版 Fig. 1--8 的唯一入口。每张图都严格分成四个阶段：

| 阶段 | 你手里有什么 | 应做什么 | 是否会训练/仿真 |
|---|---|---|---|
| A. 数据 | 没有对应 Rician 数据集 | 生成并校验规范数据集 | 只生成数据，不训练 |
| B. 模型 | 有数据，没有匹配 checkpoint | 训练缺失模型 | 会训练 |
| C. 结果 | 有数据和模型，没有本图原始结果 JSON | 用模型/基线运行本图实验 | 会仿真，不训练 |
| D. 作图 | 原始 JSON/CSV 已完整 | 只读取保存的数据生成 EPS | 不训练、不仿真 |

所有 `plot.py` 都只做阶段 D，且每个子图单独保存为 EPS。改配色、字体、坐标轴或图例时，只需重新运行 `plot.py`。全局字体和颜色在 [`common/plot_style.py`](common/plot_style.py)；某一张图的坐标轴和布局在该图自己的 `plot.py`。

> **重要：不要使用旧的 `pipeline.py`。** 它们是先前工作流的遗留文件。投稿版只使用下表的 `run.py` 和 `plot.py`。

## Fig. 1：满缓存物理层总速率

目录：[`fig01_full_buffer/`](fig01_full_buffer/)

方法：Random、**Top-SNR（固定选全局前 2%）**、Local Greedy、Spatial DL、Weighted FPLinQ、Proposed-GAT。所有方法在同一 Rician 测试快照上结算真实物理层速率并统一执行硬可行性检查。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A：没有规范数据集 | `uv run python -u paper_figures/fig01_full_buffer/run.py --stage data` | 只生成/核验 N=20/30/40/50 的 300/128/128 数据集；不会训练，也不会出结果。 |
| B：有数据、没有 GAT checkpoint | `uv run python -u paper_figures/fig01_full_buffer/run.py --stage train` | 只训练 N=20/30/40/50 的冻结 Proposed-GAT checkpoint；不运行 baseline，不生成本图结果。 |
| C：有数据和模型、缺少本图结果 | `SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" uv run python -u paper_figures/fig01_full_buffer/run.py --stage evaluate` | 只计算缺失的 N 对应 JSON，在 128 个共同测试快照上评测六种方法；不训练，也不改写已有 JSON。 |
| C：明确需要重算某个或全部 N | `SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" uv run python -u paper_figures/fig01_full_buffer/run.py --stage evaluate --nodes {N 或 20,30,40,50} --force` | 每个将被重写的 JSON 会先自动创建带时间戳的 `before_rerun` 备份；然后只评测指定 N，不训练。Top-SNR 固定选全局前 2%。 |
| D：已有 Fig. 1 原始结果 | `uv run python -u paper_figures/fig01_full_buffer/plot.py` | 仅读取 `data/raw/benchmark_N*.json`，写入 CSV 和 `figures/fig01_total_rate.eps`。 |

如果你既缺数据又缺模型并希望一次跑完 A--C，才使用：

```bash
SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" uv run python -u paper_figures/fig01_full_buffer/run.py --stage all
```

其中 `--stage data/train/evaluate/all` 分别就是上面的 A/B/C/A--C，不是任何模型超参数或算法设置。当前仓库的 Spatial-DL 使用同一个 `uv` 环境加载原始预训练 checkpoint；每次包含 Spatial-DL 的命令均需设置 `SPATIAL_DL_PYTHON="$PWD/.venv/bin/python"`。先用下面两条命令验证环境：

```bash
SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" uv run python baseline/smoke_tests/smoke_spatial_dl_bridge.py
SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" uv run python baseline/smoke_tests/smoke_spatial_dl_paper_states.py
```

两条输出均为 `"status": "pass"` 后，再运行 Fig. 1 的阶段 C。

## Fig. 2：训练收敛曲线

目录：[`fig02_training_curves/`](fig02_training_curves/)

两张独立 EPS：训练损失、验证二元速率。N=20/30/40/50，每个 N 使用 10 个**模型初始化/训练**种子取均值；10 次训练共享该 N 的正式 300/128/128 数据集，因此曲线差异不会混入数据集变化。本图需要 40 次冻结 GAT 训练。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A/B/C：缺少任意训练轨迹 | `uv run python -u paper_figures/fig02_training_curves/run.py` | 对缺失的 N×seed 训练运行；每个结果保存在 `data/raw/N*/seed*/training.json` 和对应 checkpoint。已有结果默认复用。 |
| D：12 个 `training.json` 均已存在 | `uv run python -u paper_figures/fig02_training_curves/plot.py` | 仅聚合 3 个 seed，生成两张 EPS。 |

若明确要覆盖重训，才加 `--force`；它会覆盖该图的训练报告和 checkpoint。

## Fig. 3：连续输出与二元激活审计

目录：[`fig03_activation_audit/`](fig03_activation_audit/)

Fig. 3 不是另一批训练，而是**审计 Fig. 2 完全相同的 40 条训练轨迹**，避免“同一模型却用两套随机结果”。输出为：连续 $\pi$ 的均值/标准差，以及二元激活链路比例。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A--C：Fig. 2 训练轨迹还不存在 | 先运行 Fig. 2 的 `run.py` | Fig. 3 不重复训练。 |
| 已有 Fig. 2 轨迹，想核对来源 | `uv run python -u paper_figures/fig03_activation_audit/run.py` | 只检查 40 个输入 JSON 是否齐全。 |
| D：输入 JSON 已齐全 | `uv run python -u paper_figures/fig03_activation_audit/plot.py` | 只生成两张独立 EPS。 |

## Fig. 4：数据需求

目录：[`fig04_data_efficiency/`](fig04_data_efficiency/)

训练样本数为 10/20/30/40/50/60/80/100/200/300；每个 N、每个数据量采用同一 300-sample 正式训练池的前缀，并以 10 个模型初始化/训练 seed 重复，在同一个规范测试集上计算经过硬可行性解码后的 **test binary rate**。完整网格为 4×10×10 次训练。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A--C：缺少任意 `(N, 数据量, seed)` 模型或结果 | `uv run python -u paper_figures/fig04_data_efficiency/run.py` | 缺模型则训练；有模型但缺 `evaluation.json` 则仅运行测试；已有两者则复用。 |
| D：所有 `evaluation.json` 已存在 | `uv run python -u paper_figures/fig04_data_efficiency/plot.py` | 只生成 `fig04_data_efficiency.eps`。 |

## Fig. 5：动态队列主结果

目录：[`fig05_dynamic_queue/`](fig05_dynamic_queue/)

比较方法：random、GAT-MP（主方案）、Direct-MP、GAT-LocalMax、Direct-Link LocalMax、End-to-End GAT-Q、Hard-Conflict LGS、Centralized Greedy、Weighted FPLinQ。N=20 和 N=50，五个配对 seed。每种 deadline 口径分别保存，因此不能混用。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| B：没有匹配的 End-to-End GAT-Q checkpoint | 先训练该 baseline checkpoint，再运行下方命令 | 本脚本会**报错停止**，不会用别的方法冒充 End-to-End GAT-Q。Proposed-GAT/GAT-MP 不重新训练。 |
| C1：无限等待、无限缓存口径，没有原始队列结果 | `uv run python -u paper_figures/fig05_dynamic_queue/run.py --deadline-ms none` | 运行 paired trace；未完成包是 right-censored。 |
| C2：2000 ms deadline 口径，没有原始队列结果 | `uv run python -u paper_figures/fig05_dynamic_queue/run.py --deadline-ms 2000` | 相同设置下执行 2000 ms 丢包；JSON 额外记录 deadline drop/loss。 |
| D1：已有无限 deadline 结果 | `uv run python -u paper_figures/fig05_dynamic_queue/plot.py --profile no_deadline` | 仅生成 N=20/N=50 的 backlog、delay、goodput 独立 EPS。 |
| D2：已有 2000 ms deadline 结果 | `uv run python -u paper_figures/fig05_dynamic_queue/plot.py --profile deadline_2000ms` | 仅生成有限 deadline 口径的独立 EPS。 |

## Fig. 6：跨网络规模泛化

目录：[`fig06_cross_n/`](fig06_cross_n/)

该图不训练新模型：读取各 N 的冻结 Proposed-GAT checkpoint，在 N=20/30/40/50 测试集互测。纵轴只有平均 sum rate，不再画激活比例或热图。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| C：没有 `cross_n.json` | `uv run python -u paper_figures/fig06_cross_n/run.py` | 不训练；执行 4×4 的 checkpoint 互测。 |
| D：已有 `cross_n.json` | `uv run python -u paper_figures/fig06_cross_n/plot.py` | 只生成一张独立 EPS。 |

## Fig. 7：CSI 反馈精度

目录：[`fig07_csi_feedback/`](fig07_csi_feedback/)

两个独立 EPS：第四章冻结 GAT 的 test sum rate；第五章 **GAT-MP** 的 average queue backlog。横轴为 2/4/5/8-bit；perfect CSI 与 large-scale CSI 是同色水平边界，不被错误连成量化曲线。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| C：没有各 N 的 CSI 原始 JSON | `uv run python -u paper_figures/fig07_csi_feedback/run.py` | 不训练；重跑 N=20/30/40/50 的 CSI 评估。 |
| D：已有所有 CSI JSON | `uv run python -u paper_figures/fig07_csi_feedback/plot.py` | 仅生成两张 EPS。 |

> 旧结果中“非 perfect CSI 时 GAT rate 为 0”是 CSI 适配故障，不是可用于论文的性能点。任何 Fig. 7 正式图都必须先通过该非零速率核验。

## Fig. 8：移动信道影响

目录：[`fig08_mobility/`](fig08_mobility/)

四张独立 EPS：第四章 GAT 的 sum rate 随速度/块长度变化，以及第五章 **GAT-MP** 的 backlog 随速度/块长度变化。每张都包含 N=20/30/40/50、latest（perfect）与 one-block stale 两组曲线；所有服务由当前真实移动 Rician 信道结算。

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| C：没有移动轨迹原始结果 | `uv run python -u paper_figures/fig08_mobility/run.py` | 不训练；生成共享移动轨迹上的全缓存 GAT 与队列 GAT-MP 结果。 |
| D：已有移动 JSON | `uv run python -u paper_figures/fig08_mobility/plot.py` | 仅生成四张独立 EPS。 |

## 统一原则

1. Fig. 1、2、4 的 `run.py` 会复用已有结果；Fig. 5--8 当前会直接重算并写入固定路径。投稿前只在确认缺少结果时运行相应 `run.py`；Fig. 1 若只需恢复某个 N，按上表调用底层评测器，勿使用全局 `--force`。
2. 不同方法必须共享拓扑、真实信道、到达轨迹、link set 和配对 seed；第三方 baseline 失败必须报错，不能替换为自写方法。
3. `data/raw/` 保存原始 JSON；`data/plotting/` 保存可直接二次绘图的 CSV；`figures/` 只保存 EPS。
4. 从服务器回来后，只要原始 JSON 在，就永远先运行相应 `plot.py`，不要重新训练或重新仿真。
