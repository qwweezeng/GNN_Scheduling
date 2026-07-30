# TNSE 投稿仿真图：复现指南

这里是投稿版 Fig. 1--8 的唯一入口，**所有绘图数据均由代码自动生成**。拿到本项目后，按下面的阶段顺序执行即可复现全部图片。

## 核心概念：四阶段工作流

每张图都严格分为四个阶段：

| 阶段 | 你手里有什么 | 应做什么 | 会训练吗 |
|---|---|---|---|
| **A. 数据集** | 没有该图的 Rician 数据 | 生成并校验规范数据集 | 只生成数据，不训练 |
| **B. 模型** | 有数据，没有匹配 checkpoint | 训练缺失模型 | 会训练 |
| **C. 结果** | 有数据和模型，没有本图原始 JSON | 运行实验得到原始数据 | 会仿真，不训练 |
| **D. 作图** | 原始 JSON 已完整 | 读取数据生成 EPS | 不训练、不仿真 |

> **原则：永远优先阶段 D。** 从服务器回来后，只要原始 JSON 在，就永远直接运行相应 `plot.py`，不要重新训练或重新仿真。

### 哪些指令会覆盖已有数据

- `fig01/run.py --stage evaluate --force`：会覆盖 `benchmark_N*.json`，但会先自动在同目录创建 `*.before_rerun_时间戳.json` 备份。
- `fig02/run.py --force`：会覆盖 `data/raw/N*/seed*/training.json` 和对应 checkpoint。
- `fig04/run.py --force`：会覆盖所有 `training.json`、`eval_*.json` 和 checkpoint。
- `fig05/run.py`：会直接覆盖 `data/raw/{profile}/N*/{metrics.json,per_seed_metrics.json,...}`。
- `fig06/run.py`：会直接重写其固定输出 `data/raw/cross_n_trainSize={train_size}.json`；不要用它覆盖不同训练样本数的结果。
- `fig07/resume.py`：Chapter 4 仅在 checkpoint 不存在时训练；已有 checkpoint 和已有 per-mode JSON 都跳过。Chapter 5 每完成一个条件会重写同一个 `chapter5_N*_v2_matched_csi.progress.json`，全部完成后写入 `.json`；同一 N 不得并发运行两个进程。
- `fig08/run.py`：会调用底层断点逻辑并重写相应最终 JSON；需要补缺或中断恢复时，使用 `fig08/resume.py`，不要假设 `run.py` 有 `--force`。

所有 `plot.py` 只读数据，不写原始 JSON，永远安全。

## 第三方依赖与环境

本仓库不是纯 Python 标准库项目。主实验环境由根目录 `pyproject.toml` 管理，建议使用 `uv sync` 创建/同步环境；Python 版本要求为 `>=3.12`。主环境直接依赖以下第三方包：

| 依赖 | 用途 |
|---|---|
| PyTorch、torchvision | GAT 训练、模型推理、张量计算 |
| NumPy、SciPy | 信道/队列仿真、统计、FPLinQ 数值计算 |
| Matplotlib | EPS/JPG 绘图 |
| PyYAML | YAML 配置读取 |
| TensorBoard | 训练日志支持 |
| pandas、openpyxl | 数据分析和表格相关辅助功能 |
| argparse | 命令行解析（虽随现代 Python 发行版提供，仍在项目依赖声明中） |
| TensorFlow | legacy baseline 适配；不是 Proposed-GAT 主模型 |

主环境安装：

```bash
uv sync
uv run python -c "import torch, numpy, scipy, matplotlib, yaml; print('main environment: pass')"
```

Fig. 1 的 Spatial DL 和可选的 DistGCN 不是主环境中的纯 PyTorch 实现，它们分别依赖独立的 legacy conda 环境。环境文件和 smoke test 位于 [`../baseline/README.md`](../baseline/README.md)：

```bash
conda env create -f baseline/environments/fplinq.yml
conda env create -f baseline/environments/spatial_dl_legacy.yml
conda env create -f baseline/environments/distgcn_legacy.yml
```

- `gnn-fplinq`：仅用于 legacy/隔离的 FPLinQ 依赖验证；主代码也有 NumPy/SciPy FPLinQ 适配器。
- `gnn-spatial-dl-legacy`：Fig. 1 的 Spatial DL 外部 checkpoint 和 TensorFlow legacy 运行环境。
- `gnn-distgcn-legacy`：可选 DistGCN-LGS baseline；不参与当前 Fig. 1--8 默认主图。
- CUDA GPU、兼容 NVIDIA 驱动和 CUDA-enabled PyTorch 是 `--device cuda` 的运行前提；没有 GPU 时可改成 `--device cpu`，但长实验会明显变慢。
- Graph Embedding 已从当前仓库移除，不是复现 Fig. 1--8 的依赖，也不能用其他 baseline 冒充它。

## 指标口径总表

除非特别说明，`binary` 表示最终动作由 `pi >= 0.5` 得到 0/1，再经过 endpoint mutual matching 或 shared-UAV hard-feasibility projection；`soft` 表示直接使用连续的 `pi`，不二值化。所有最终论文性能图均使用真实物理信道结算，不把训练 surrogate 当作物理速率。

| 名称 | 代码字段/含义 | 是否主图指标 |
|---|---|---|
| soft utility/rate | 连续 `pi` 直接参与的 utility/rate，不经过二值化；主要用于 Fig. 2/4 诊断 | Fig. 4 诊断，不是主曲线 |
| binary rate | `pi >= 0.5` 的二元动作，但需结合对应评测脚本的 decoder；训练日志中的 `val_utility` 对应这一路径 | Fig. 2 validation 主曲线 |
| hard-unprojected rate | 二元 threshold 后不做 endpoint/shared-UAV projection；用于区分动作阈值和可行性解码的影响 | Fig. 4 诊断 |
| hard-feasible/projected binary rate | 二元 threshold 后执行 endpoint mutual matching/shared-UAV feasibility，再用真实信道 `calculate_achievable_rate(..., binary=True)` 结算 | Fig. 1、Fig. 4 主曲线、Fig. 6、Fig. 7 Chapter 4 |
| physical service | 队列中实际被服务/离队的总工作量；可能包含仿真结束时尚未交付给用户的业务 | Fig. 5 不作为 goodput 主指标 |
| delivered goodput | 在仿真结束前真正完成交付的 packet bits / 仿真时长 | Fig. 5 主指标 |
| average total queue backlog | 所有 link queue 长度的时间平均总和，按 `packet_size` 转为 Mbit | Fig. 5、Fig. 7 Chapter 5、Fig. 8 |

---

## 统一前提

- 实验固定条件：N=20/30/40/50、10 dBm、Rician K=5 dB、train/val/test=300/128/128、slot=1 ms、packet=12000 bit。
- 所有 `run.py/plot.py` 均在项目根目录运行（`cd /workspace/GNN_Scheduling_codex`）。
- 环境变量示例：
  ```bash
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=32 \
  MKL_NUM_THREADS=32 \
  OPENBLAS_NUM_THREADS=32 \
  uv run python -u <命令>
  ```
- Spatial-DL 验证（如 Fig. 1 包含该方法）：
  ```bash
  SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" \
  uv run python baseline/smoke_tests/smoke_spatial_dl_bridge.py
  SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" \
  uv run python baseline/smoke_tests/smoke_spatial_dl_paper_states.py
  ```
  两条均输出 `"status": "pass"` 后再运行 Fig. 1 阶段 C。

---

## Fig. 1：满缓存物理层总速率（第四章）

目录：[`fig01_full_buffer/`](fig01_full_buffer/)

横轴 N=20/30/40/50，纵轴 total physical-layer rate (Mbit/s)。
比较方法：Random、Top-SNR（前 2%）、Local Greedy、Spatial DL、Weighted FPLinQ、Proposed-GAT。

**指标口径：** 这是 128 个共同 test snapshot 上的平均聚合物理层速率，不是 soft rate，也不是 queue utility/goodput。对于 Proposed-GAT，模型输出先做 `pi >= 0.5` 二值化，再做 endpoint mutual matching/shared-UAV hard-feasibility；之后在真实测试信道上计算所有被激活链路的 achievable rate 并求和。Random、Top-SNR、Local Greedy、Weighted FPLinQ 和 Spatial DL 也在同一真实信道和同一完整 link set 上执行 hard feasibility。JSON 字段为 `mean_aggregate_phy_rate_mbps`，误差字段为 `standard_error_aggregate_phy_rate_mbps`。

**文件布局：** `data/raw/benchmark_N{20,30,40,50}.json`

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A：无数据集 | `uv run python -u paper_figures/fig01_full_buffer/run.py --stage data` | 生成 N=20/30/40/50 的 300/128/128 规范数据集；不训练、不出结果 |
| B：有数据、无 checkpoint | `uv run python -u paper_figures/fig01_full_buffer/run.py --stage train` | 验证 N=20/30/40/50 的 `full_300` checkpoint 是否存在（N=300 的正式 checkpoint 复用 Fig. 2 的训练结果）；不重复训练 |
| C：有数据和模型、缺结果 JSON | `SPATIAL_DL_PYTHON="$PWD/.venv/bin/python" uv run python -u paper_figures/fig01_full_buffer/run.py --stage evaluate` | 在 128 个共同测试快照上评测六种方法；已有 JSON 跳过，可用 `--force` 强制重算 |
| C：重算指定 N | 同上，加 `--nodes 20,50` | 只评测指定 N |
| D：已有 JSON | `uv run python -u paper_figures/fig01_full_buffer/plot.py` | 读取 `benchmark_N*.json`，生成 `figures/fig01_total_rate.eps` |

**注意：** `--stage train` 不训练新模型，只复用 Fig. 2 已经保存的 `best_model.pth/final_model.pth`（位置：`checkpoints/paper/proposed_gat/rician_K5_tx10/N{num_uav}/seed{model_seed}/full_300/`）。

---

## Fig. 2：Proposed-GAT 训练收敛（第四章）

目录：[`fig02_training_curves/`](fig02_training_curves/)

两张独立 EPS：训练 loss -- epoch、validation binary rate -- epoch。
N=20/30/40/50，每条曲线平均 3 个训练 seed（`--seeds 0,1,2`）。

**指标口径：** `training_loss` 是训练目标的负 utility/损失记录，不应解释成物理层速率。图中的 `validation_binary_rate_mbps` 来自训练日志的 `val_utility`：模型输出经过 `pi >= 0.5` 二值化，并按训练期间的 hard binary scheduling 口径结算；单位换算为 5 MHz × bit/s/Hz = Mbit/s。日志还保存 `validation_soft_rate_mbps`（连续 `pi`，不二值化）和 `validation_projected_rate_mbps`（二值化后 hard-feasibility projection），但当前 Fig. 2 主图只画 binary validation rate，不画 soft/surrogate 曲线。

**文件布局：** `data/raw/N{20,30,40,50}/seed{0,1,2}/training.json`（含 best/final checkpoint 路径）

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A/B/C：缺少任意 N×seed 训练 | `uv run python -u paper_figures/fig02_training_curves/run.py --nodes 20,30,40,50 --seeds 0,1,2 --epochs 50` | 缺训练则训练；已有 `training.json` + 双 checkpoint 则跳过；训练完成后自动在规范测试集上运行 soft/hard/projected 三种评测 |
| D：12 个 `training.json` 均存在 | `uv run python -u paper_figures/fig02_training_curves/plot.py --nodes 20,30,40,50 --seeds 0,1,2` | 生成两张 EPS：`fig02_training_loss.eps` 和 `fig02_validation_rate.eps` |

**`--force` 会覆盖已有训练报告和 checkpoint**，慎用。

---

## Fig. 3：Proposed-GAT 输出审计（第四章）

目录：[`fig03_activation_audit/`](fig03_activation_audit/)

Fig. 3 不训练新模型，**复用 Fig. 2 完全相同的 4×3=12 条训练轨迹**。
两个子图：连续 π 的均值与标准差 -- epoch；二元激活链路比例 -- epoch。

**指标口径：** 第一个子图统计每个 epoch、每个 test/validation batch 中模型连续输出 `pi` 的均值和标准差，完全不代表物理速率。第二个子图统计 `pi >= 0.5` 后的 0/1 激活链路比例，用于检查全开、全关或输出坍缩；它不是可行性投影后的 throughput。

**文件布局：** `data/raw/N{20,30,40,50}/seed{0,1,2}/training.json`（从 Fig. 2 复制）

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A--C：Fig. 2 轨迹不存在 | 先运行 Fig. 2 的 `run.py` | Fig. 3 不重复训练 |
| 已有 Fig. 2 轨迹 | `uv run python -u paper_figures/fig03_activation_audit/run.py` | 将 Fig. 2 的 `training.json` 同步到 Fig. 3 的 `data/raw/` 目录（--source 可指定其他路径） |
| D：输入 JSON 齐全 | `uv run python -u paper_figures/fig03_activation_audit/plot.py` | 生成两张独立 EPS |

---

## Fig. 4：Proposed-GAT 数据需求（第四章）

目录：[`fig04_data_efficiency/`](fig04_data_efficiency/)

横轴训练样本数 10/20/30/40/60/80/100/200/300/500，N=20/30/40/50，每格 3 个 seed。
每个小样本集必须是同一 500-sample pool 的前缀；n=300 复用 Fig. 2 的 `(N, seed)` checkpoint，不重复训练。

**指标口径：** 主曲线 `fig04_best_projected_rate` 是 best-validation checkpoint 在同一规范 128-sample test set 上的 **hard-feasible projected binary rate**：`pi >= 0.5`，再做 endpoint mutual matching/shared-UAV feasibility，最后用真实 test channel 结算。它不是 soft utility，也不是仅 threshold 后未经 projection 的 binary utility。代码同时保存并绘制 audit 指标：best/final checkpoint 的 soft rate、hard-unprojected rate、projected rate，以及训练阶段 `best_val_utility`；这些指标不能互相替代。

**文件布局：** `data/raw/N{20,30,40,50}/n{10,20,...,500}/seed{0,1,2}/training.json` + `eval_best_*.json` + `eval_final_*.json`

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| A/B/C：缺少任意 (N, size, seed) 组合 | `uv run python -u paper_figures/fig04_data_efficiency/run.py --nodes 20,30,40,50 --epochs 50` | 缺模型则训练；n=300 自动复用 Fig. 2 checkpoint；训练后自动运行 best/final checkpoint 的 soft/hard/projected 三种评测 |
| C：只跑 n=500 | `uv run python -u paper_figures/fig04_data_efficiency/run.py --train_size 500 --nodes 20,30,40,50 --epochs 50` | 单独运行 n=500 训练（不使用 Fig. 2 的 n=300 checkpoint） |
| D：所有 `eval_*.json` 已存在 | `uv run python -u paper_figures/fig04_data_efficiency/plot.py` | 生成 `fig04_best_projected_rate.eps`（主曲线）及其他诊断 EPS |

---

## Fig. 5：动态队列主结果（第五章）

目录：[`fig05_dynamic_queue/`](fig05_dynamic_queue/)

投稿主图：N=50、2000 ms deadline，横轴 λ=0.5/1.0/1.5/2.0/2.2/2.4/2.6/2.8/3.0/3.2/3.5/4.0/4.5/5.0。
三张独立 EPS：backlog、deadline-capped latency、goodput。

**指标口径：**

- **Backlog**：`average_total_queue_backlog_mbit`，所有 link queue 的总 backlog 在所有 slot 上的时间平均值，按 12000 bit/packet 换算为 Mbit。它包括仿真结束时仍未解决的 packet，因此是拥塞程度指标。
- **Goodput**：`successful_goodput_mbps`，在仿真结束前真正完成交付的 packet 数乘以 12000 bit，再除以仿真总时长；即 `delivered_packet_count * packet_size / duration / 1e6`。它不是 `physical_service_mbps`，后者统计总离队/服务工作量，也不是 `sum_i Q_i R_i` utility。
- **Deadline-capped latency**：对 resolved packets（已完成或已因 deadline/buffer 丢弃）统计。完成 packet 使用实际 FIFO completion delay；deadline drop 和 buffer-overflow drop 使用 2000 ms 惩罚；仿真结束仍在队列中的 unfinished packet 不进入该分母。代码计算为
  `(完成包数 × 平均完成 delay + [deadline drop + overflow drop] × 2000 ms) / [完成包数 + deadline drop + overflow drop]`。
  字段为 `resolved_deadline_capped_delay_ms`，纵轴固定 `[0, 2000] ms`。
- **Completed-only delay**：`average_packet_delay_ms` 只对已完成 packet 统计，可能存在 right-censoring survivor bias；仅作为 no-deadline/内部诊断，不作为投稿主图结论。

每个方法的最终动作由对应 scheduler 产生并经过 single-radio/half-duplex hard feasibility；服务在当前真实 Rician channel 上结算。所有方法共享 paired topology、arrival trace、channel trace 和 seed。

方法（全部方法共享同一拓扑、信道、到达轨迹）：
Random、Centralized Greedy、Weighted FPLinQ、Direct-LocalMax、GAT-LocalMax、DirectMP、GAT-MP。

> Fig. 5 的 wrapper 显式通过 `--methods` 固定传入七种主图方法，其中 `GAT-Assisted MarginalMP` 就是图中的 GAT-MP。因此，底层 `exp_fixed_csi_qos.py` 在**未传入** `--methods` 时采用的默认集合不会影响 Fig. 5。当前投稿图不纳入 End-to-End GAT-Q，且不应为它重新训练或补充数据。

**文件布局：**
```
data/raw/no_deadline/N50/{metrics.json,per_seed_metrics.json,config.json,trajectories.npz,tasks/}
data/raw/deadline_2000ms/N50/{metrics.json,per_seed_metrics.json,config.json,trajectories.npz,tasks/}
```

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| C1：无限缓存、无 deadline | `uv run python -u paper_figures/fig05_dynamic_queue/run.py --nodes 50 --slots 10000 --seeds 42 --deadline-ms none` | 运行无 deadline 口径；结果写入 `data/raw/no_deadline/N50/` |
| C2：2000 ms deadline | `uv run python -u paper_figures/fig05_dynamic_queue/run.py --nodes 50 --slots 10000 --seeds 42 --deadline-ms 2000` | 运行 deadline 口径；结果写入 `data/raw/deadline_2000ms/N50/` |
| D1：已有无限 deadline 结果 | `uv run python -u paper_figures/fig05_dynamic_queue/plot.py --profile no_deadline` | 读取 `no_deadline/N50/` 生成 backlog/delay/goodput EPS |
| D2：已有 2000 ms 结果 | `uv run python -u paper_figures/fig05_dynamic_queue/plot.py --profile deadline_2000ms` | 读取 `deadline_2000ms/N50/` 生成 backlog/delay/goodput EPS |

> **注意：** `--profile no_deadline` 和 `--profile deadline_2000ms` 不可混用；两者生成独立的 EPS。两套口径的 delay 定义不同：无限缓存口径统计已完成的包（高负载下 right-censored）；2000 ms 口径对完成包用实际 FIFO delay，对 deadline/buffer drop 记 2000 ms 惩罚，分母只含已解决包。

---

## Fig. 6：Proposed-GAT 跨规模泛化（第四章）

目录：[`fig06_cross_n/`](fig06_cross_n/)

训练 N=20/30/40/50/mixed 的 Proposed-GAT 在 N=20/30/40/50 测试集上互测。
train_size=500。若 JSON 中存在 mixed checkpoint，实际图包含五条训练来源曲线（N=20、30、40、50、mixed），不是热图。

**指标口径：** 每个 train-N/test-N 单元格的指标是 128 个测试样本上的平均 **hard-feasible binary aggregate sum rate**，字段为 `mean_aggregate_phy_rate_mbps`。模型输出先 `pi >= 0.5`，再用 endpoint mutual matching 解码，确认 shared-UAV feasibility，最后在 test-N 的真实 Rician channel 上计算所有激活链路 achievable rate 之和。它不是 soft sum rate、active-link fraction 或 queue utility。`export_table.py` 额外输出 train-N ≤ test-N 单元格相对于同一 test-N matched-scale checkpoint 的 rate 百分比；该百分比不是新的仿真指标。

**文件布局：** `data/raw/cross_n_trainSize=500.json`（主文件）+ `cross_n_raw.json`（详细调试数据）

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| C：无 `cross_n_trainSize=500.json` | `uv run python -u paper_figures/fig06_cross_n/run.py --device cuda --train_size 500` | 训练缺失 scale 的 checkpoint（若已有则跳过），然后执行 4×4 互测 |
| D：已有 JSON | `uv run python -u paper_figures/fig06_cross_n/plot.py` | 读取 JSON 生成 `fig06_cross_network_generalization.eps` |
| D：生成投稿表格 | `uv run python -u paper_figures/fig06_cross_n/export_table.py --input paper_figures/fig06_cross_n/data/raw/cross_n_trainSize=500.json` | 从正式 500-sample JSON 生成 `data/plotting/fig06_small_to_large_table.csv`（只含 train ≤ test 单元格） |

---

## Fig. 7：CSI 反馈精度（第四/五章）

目录：[`fig07_csi_feedback/`](fig07_csi_feedback/)

两个独立 EPS：
- **第四章**：Proposed-GAT test sum rate vs CSI 量化 bit
- **第五章**：GAT-MP average queue backlog vs CSI 量化 bit

CSI 条件：perfect、large-scale、quantized 2/4/6/8/10-bit（共 7 种）。
所有 CSI 条件共享同一真实 Rician 移动轨迹和到达业务。

**指标口径：** Chapter 4 的纵轴是 `mean_aggregate_phy_rate_mbps`，即 test samples 上经过 `pi >= 0.5`、endpoint mutual matching 和 hard feasibility 后的二元动作，在真实当前 Rician test channel 上结算的平均 sum rate；不是 soft `pi` rate。Chapter 5 的纵轴是 `average_total_queue_backlog_mbit`，即固定 `lambda=2.4 packet/slot`、speed=0、block length=1、seed=42、10000 slots 下所有链路 queue 的时间平均总 backlog；不是 throughput、goodput 或训练 utility。两张图都以 2/4/6/8/10-bit 为横轴实线；每个 N 的 perfect CSI 用同色点线、large-scale CSI 用同色虚线作参考线，不把它们误连为量化曲线。对于 Chapter 4（rate 越大越好），perfect CSI 是上界参考；对于 Chapter 5（backlog 越小越好），perfect CSI 是低 backlog 参考。large-scale CSI 是低开销反馈参考，而非有限 Monte Carlo 下的严格数学上界/下界。Chapter 5 只改变 scheduler 看到的 CSI，服务仍在当前真实信道上结算。

### 数据文件布局

**第四章（每 N 每 CSI 条件一个文件）：**
```
data/raw/chapter4_N{20,30,40,50}_v2_{perfect,large_scale,quantized_2bit,...,quantized_10bit}.json
```

**第五章（每 N 一个文件）：**
```
data/raw/chapter5_N{20,30,40,50}_v2_matched_csi.json        ← 完整结果（7 个 CSI 条件均完成）
chapter5_N{20,30,40,50}_v2_matched_csi.progress.json        ← 断点续跑中间文件
```

### 阶段命令

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| C：缺任何第四章 v2 JSON | `uv run python -u paper_figures/fig07_csi_feedback/resume.py --nodes 20,30,40,50 --chapters 4 --epochs 50 --device cuda` | 遍历 N×7 种 CSI 条件，缺 checkpoint 则训练，然后执行评测；已有 JSON 跳过 |
| C：N=50 先行两个条件 | `uv run python -u paper_figures/fig07_csi_feedback/resume.py --nodes 50 --chapters 4 --epochs 50 --device cuda` | 仅处理 N=50 |
| C：缺第五章任何 JSON | `uv run python -u paper_figures/fig07_csi_feedback/resume.py --nodes 20,30,40,50 --chapters 5 --slots 10000 --seeds 42 --arrival-rate 2.4 --device cuda` | 缺 checkpoint 则报错退出；否则对每个 N 载入全部 7 个匹配 GAT 模型，在同一移动轨迹上比较所有 CSI 条件 |
| C：N=50 分两段续跑（支持断点） | 第一段：`--csi-methods latest,large_scale`；等 checkpoint 齐全后第二段：`--csi-methods latest,large_scale,quantized_2bit,quantized_4bit,quantized_6bit,quantized_8bit,quantized_10bit` | 两次 `--csi-methods` 参数不同但输出路径相同；脚本自动读取已有的 `.progress.json` 跳过已完成条件，最终合并为完整 7 条记录的 `.json` |
| D：N=20/30/40 Chapter 5 JSON 完整 | `uv run python -u paper_figures/fig07_csi_feedback/plot.py --nodes 20,30,40 --chapters 5` | 生成 Chapter 5 折线图：量化 bit 为实线，perfect/large-scale 为同色参考线 |
| D：所有 Chapter 4/5 JSON 完整 | `uv run python -u paper_figures/fig07_csi_feedback/plot.py --nodes 20,30,40,50 --chapters 4,5` | 生成两张 EPS：`fig07_chapter4_csi_rate.eps`（第四章）和 `fig07_chapter5_gat_mp_backlog.eps`（第五章） |

> 第五章的 `resume.py` 使用 `experiments/evaluate_mobile_block_fading_csi.py`，每次运行固定写入同一个 `chapter5_N{num}_v2_matched_csi.progress.json`，完成后写入 `.json`。同一 N 的多次续跑必须使用**相同** `--seeds`、`--speeds-mps`、`--slots-per-block`、`--stale-blocks`、`--arrival-rates`、`--scheduler-variants` 参数，否则断点续跑逻辑可能跳过或重复。

---

## Fig. 8：移动性与 CSI 年龄（第五章）

目录：[`fig08_mobility/`](fig08_mobility/)

两张独立 EPS：GAT-MP backlog 随 UAV speed（固定 block=5）；GAT-MP backlog 随 block length（固定 v=10 m/s）。
N=20/30/40/50，CSI 方法：latest（perfect）和 one-block stale。

**指标口径：** 两张图的纵轴都为 `average_total_queue_backlog_mbit`，即所有链路 queue 在仿真时隙上的时间平均总 backlog，单位 Mbit；横轴分别是 UAV speed (m/s) 和 channel block length (slots)。每个 CSI 条件使用 `gat_assisted` GAT-MP，GAT 输出作为每个信道块缓存的 link priority，MP 每时隙结合当前 queue 做调度；最终服务在当前真实移动 Rician channel 上结算。这里不画 sum rate、goodput、soft utility 或 active-link fraction。

**文件布局：**
```
data/raw/gatmp_speed_N{20,30,40,50}.json + .progress.json
data/raw/gatmp_block_N{20,30,40,50}.json + .progress.json
```

| 你当前状态 | 命令 | 实际行为 |
|---|---|---|
| C：无移动 JSON | `uv run python -u paper_figures/fig08_mobility/run.py --nodes 20,30,40,50 --slots 5000 --seeds 42 --device cuda` | 对每个 N 依次运行 speed sweep 和 block sweep；底层依赖 `.progress.json` 去重，但该 wrapper 没有 `--force` |
| C：只补缺失 N | `uv run python -u paper_figures/fig08_mobility/run.py --nodes 40,50 ...` | 只跑指定 N |
| C：断点续跑（推荐） | `uv run python -u paper_figures/fig08_mobility/resume.py --nodes 20,30,40,50 --slots 5000 --seeds 42 --device cuda` | 读取已有 JSON 跳过已完成条件，只补缺失组合 |
| D：已有全部 JSON | `uv run python -u paper_figures/fig08_mobility/plot.py` | 生成 `fig08_gatmp_speed.eps` 和 `fig08_gatmp_block_length.eps` |

---

## 统一原则

1. **复用优先，但以各图表格为准**：Fig. 1、2、4、7 的 runner 明确跳过已有 artifact；Fig. 5、6、8 会重写部分固定 aggregate JSON 或依赖 progress 去重。拿到项目后先运行 `plot.py`，再根据缺失文件和本图的覆盖说明决定是否运行 `run.py`。
2. **断点续跑**：`fig07` 和 `fig08` 支持分批续跑，只要参数不变即可安全合并。
3. **方法一致性**：不同方法必须共享拓扑、真实信道、到达轨迹、link set 和配对 seed；只改变调度器可见 CSI。
4. **第三方 baseline**：Spatial-DL 等外部依赖失败必须报错，不能替换为自写方法。
5. **目录结构**：`data/raw/` 保存原始 JSON；`data/plotting/` 保存可直接二次绘图的 CSV；`figures/` 只保存 EPS。
6. **数据冻结**：所有 `run.py` 输出 JSON 中必须包含 `config`、`checkpoint`、`fingerprint`、`seed`、`method variant` 等 provenance 字段。

---

## 当前已有数据状态（2026-07-30）

| 图 | N | 数据状态 | 备注 |
|---|---|---|---|
| Fig. 1 | 20/30/40/50 | ✅ `benchmark_N*.json` 完整 | 可直接运行 `plot.py` |
| Fig. 2 | 20/30/40/50 | ✅ `training.json` × 3 seed 完整 | 可直接运行 `plot.py` |
| Fig. 3 | 20/30/40/50 | ✅ 复用 Fig. 2 轨迹 | 可直接运行 `plot.py` |
| Fig. 4 | 20/30/40/50 | ✅ n=10~300 完整，缺 n=500 | 需单独运行 `--train_size 500` |
| Fig. 5 (2000ms) | 50 | ✅ 完整 7 方法 | 可直接运行 `plot.py --profile deadline_2000ms` |
| Fig. 5 (no deadline) | 50 | ✅ 完整 | 可直接运行 `plot.py --profile no_deadline` |
| Fig. 6 | — | ⚠️ 仅有旧名 `cross_n.json`/`cross_n_raw.json`，缺正式 `cross_n_trainSize=500.json` | 运行 `fig06_cross_n/run.py --train_size 500` 后再作图 |
| Fig. 7 Ch4 | 20/30/40 | ✅ 7 种 CSI 条件完整 | 可直接生成 Chapter 4 EPS |
| Fig. 7 Ch4 | 50 | ⚠️ 只有 `perfect`、`large_scale` | 还需量化 2/4/6/8/10-bit |
| Fig. 7 Ch5 | 20/30 | ✅ `chapter5_N*_v2_matched_csi.json` 完整 | 可直接生成 Chapter 5 EPS |
| Fig. 7 Ch5 | 40 | ⚠️ 只有 `latest`、`large_scale` 两条 progress | 需续跑补量化 2/4/6/8/10-bit |
| Fig. 7 Ch5 | 50 | ⚠️ 已有 `latest`、`large_scale` 两条记录；量化条件正在分批补齐 | 最后一次续跑必须传入完整 7 个 `--csi-methods`，使最终 JSON 带完整 matched-model provenance |
| Fig. 8 | 20/30/40/50 | ✅ `gatmp_speed/block_N*.json` 完整 | 可直接运行 `plot.py` |
