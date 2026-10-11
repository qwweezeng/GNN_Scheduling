# `experiments/`：论文实验的底层评测说明

这个目录负责回答“某个模型或调度器在一组确定的输入上表现如何”。它连接数据集、checkpoint、baseline 和 simulator，并把可复查的指标写成 JSON/NPZ。论文逐图的参数编排和绘图入口位于 `paper_figures/figXX_*/`：通常由其中的 `run.py` 或 `resume.py` 调用本目录脚本，再由 `plot.py` 读取结果作图。

因此，两层目录的职责不同：

- `experiments/*.py`：底层评测器，可改变 N、seed、slot 数、CSI 条件和输出路径；
- `paper_figures/figXX_*/run.py`：把底层评测器固定成论文对应的设置；
- `paper_figures/figXX_*/data/raw/`：论文结果的原始证据；
- `paper_figures/figXX_*/plot.py`：只读取结果并生成 EPS，不训练、不仿真。

修改代码后，应先用这里的实验判断行为是否符合论文逻辑，再决定是否更新论文图。不要直接根据训练 loss 或一条临时输出判断第五章协议已经正确实现。

全部图的最新四阶段顺序命令集中维护在 `docs/PAPER_FIGURE_PIPELINE.md`。当前统一只画图输出目录是 `results/paper/tnse_current_plots/`；下面各节保留每个实验的物理含义和单独入口。

## 先判断当前处于哪个阶段

每个实验都按以下四个阶段理解：

| 阶段 | 当前已有内容 | 当前缺少内容 | 下一步 |
|---|---|---|---|
| A：没有数据集 | 配置和代码 | 匹配的 dataset | 生成并核对 dataset metadata |
| B：有数据集，未训练 | dataset | 匹配的 checkpoint | 训练并保存 checkpoint provenance |
| C：有 checkpoint，未跑实验 | dataset、checkpoint | result JSON/NPZ | 运行评测器，不重复训练 |
| D：已有实验结果 | 与当前实现匹配的 result | figure | 只运行 `plot.py` |

这里的“匹配”非常重要。仅仅存在一个 `.pkl`、`.pth` 或 `.json` 文件，并不能说明阶段已经完成。至少应核对：

- dataset：N、信道、发射功率、CSI 类型、train/validation/test seed 和 fingerprint；
- checkpoint：训练 N、训练样本数、CSI 条件、模型 seed、best/final 类型和 dataset fingerprint；
- result：方法集合、N、arrival rate、slot 数、Monte Carlo seed、checkpoint 路径和算法版本；
- code：模型、解码、信道、队列更新或指标定义在结果生成后是否发生过变化。

### 动态队列实验怎样对应四阶段

Fig. 5、Fig. 7 Chapter V 和 Fig. 8 的 topology、channel trace 和 arrival trace 由运行器按 seed 生成，不要求预先准备一个统一的大型 queue dataset。对这类实验：

- 阶段 A 表示确认 trace 生成规则和 seed，而不是先生成静态 pickle；
- 阶段 B 只涉及其所需的 channel-GAT checkpoint；GAT-MP 本身不再训练；
- 阶段 C 才生成 paired trace、运行各调度器并写结果；
- 阶段 D 只读取 JSON/NPZ 作图。

## 当前论文结果与入口总表

| 论文中的结果 | `paper_figures` 入口 | 底层实现 | 核心问题 |
|---|---|---|---|
| Fig. 1：物理层总速率 | `fig01_full_buffer/run.py` | `compare_gat_vs_spatial_dl.py` | Edge-enhanced GAT 能否学习有效的空间复用 |
| Fig. 2：训练 loss 与 thresholded validation rate | `fig02_training_curves/run.py` | 训练脚本和 common evaluator | 训练是否收敛，连续输出能否形成稳定二元决策 |
| Fig. 3：输出审计 | `fig03_activation_audit/run.py` | 复用 Fig. 2 日志 | 输出是否发生全开、全关或无区分度坍缩 |
| Fig. 4：数据量与 soft utility | `fig04_data_efficiency/run.py` | 训练脚本和 common evaluator | 有限训练 snapshot 下的 sample efficiency |
| Fig. 5：动态异质业务 | `fig05_dynamic_queue/run.py` | `exp_fixed_csi_qos.py` | GAT prior 与 marginal coordination 对 QoS 的互补作用 |
| Cross-scale table | `fig06_cross_n/run.py` | `evaluate_channel_gat_cross_n.py` | 小规模训练模型能否迁移到更大规模网络 |
| Fig. 7 Chapter IV | `fig07_csi_feedback/resume.py --chapters 4` | `evaluate_channel_gat_csi.py` | Edge-enhanced GAT 对低精度 CSI 的表现 |
| Fig. 7 Chapter V | `fig07_csi_feedback/resume.py --chapters 5` | `evaluate_mobile_block_fading_csi.py` | GAT-MP 在不同 CSI 条件下的 backlog 表现 |
| `fig08_mobility` | `fig08_mobility/run.py` / `resume.py` | `evaluate_mobile_block_fading_csi.py` | 移动速度和 block length 的诊断；当前不进入论文 |

Fig. 2--4 没有单独的 `experiments/*.py` 主入口，因为它们直接围绕训练日志和 checkpoint 展开。它们仍属于同一实验链条，并由 `paper_figures` 中的 wrapper 管理。

## 公共实验口径

除非某一节特别说明，正式实验使用 `configs/paper.yaml`：N 为 20/30/40/50，10 dBm 发射功率，2.4 GHz 载频，5 MHz 带宽，Rician K=5 dB。静态 channel-GAT 的规范 split seed 为 10086/10087/10088，常规 train/validation/test 数量为 300/128/128。

调度结果还必须区分以下指标：

| 指标 | 含义 | 主要用途 |
|---|---|---|
| Soft utility/rate | 连续 `pi` 直接进入 surrogate，不做二值化 | Fig. 4 和训练诊断 |
| Thresholded binary rate | `pi >= 0.5` 后的二元动作；是否做 feasibility 要看具体字段 | Fig. 2 训练行为 |
| Hard-feasible physical rate | 二值动作通过 endpoint/shared-UAV feasibility 后，在物理信道上结算 | Fig. 1、cross-scale、Fig. 7 Chapter IV |
| Average total queue backlog | 所有链路缓存业务量的时均总和 | Fig. 5、Fig. 7 Chapter V、Fig. 8 |
| Deadline-capped delay | 完成包使用实际 delay，deadline/buffer drop 记为 Dmax | Fig. 5 |
| Goodput | 仿真结束前成功交付的 bits / 仿真时长 | Fig. 5 |

训练 surrogate、未经过可行性处理的 binary rate、physical service 和 delivered goodput 不能互相替代。

---

## 1. 满缓存物理层比较

### 脚本与设置

底层脚本：`experiments/compare_gat_vs_spatial_dl.py`

论文入口：

```bash
uv run python -u paper_figures/fig01_full_buffer/run.py --stage data
uv run python -u paper_figures/fig01_full_buffer/run.py --stage train
uv run python -u paper_figures/fig01_full_buffer/run.py --stage evaluate
uv run python -u paper_figures/fig01_full_buffer/plot.py \
  --raw-dir paper_figures/fig01_full_buffer/data/raw \
  --output-root results/paper/tnse_current_plots
```

正式设置为 N=20/30/40/50，每个规模使用共同的 128 个 Rician test snapshots。比较 Random、Top-SNR 2%、Local Greedy、Spatial DL、Weighted FPLinQ 和 edge-enhanced GAT。各方法返回的动作都要满足 shared-UAV feasibility，最后在同一真实 test channel 和完整 link set 上计算 aggregate physical-layer rate。

### 它验证什么

该实验验证第四章模型是否能从有向、带权的干扰图中学习有效的 spatial reuse。它只评价满缓存物理层 sum rate，不包含 queue、delay 或 goodput，也不能单独说明第五章的 queue-aware coordination 有效。

### 四阶段资产

| 阶段 | 检查位置 | 说明 |
|---|---|---|
| A | `datasets/paper/rician_K5_tx10/perfect/N*/` | 需要规范 test set 和 matching manifest |
| B | `checkpoints/paper/proposed_gat/rician_K5_tx10/N*/seed0/full_300/` | Fig. 1 复用 Fig. 2 的 best checkpoint；`--stage train` 只验证，不另行训练 |
| C | `paper_figures/fig01_full_buffer/data/raw/benchmark_N*.json` | `--stage evaluate` 调用本节底层脚本；已有文件默认跳过 |
| D | `results/paper/tnse_current_plots/figures/fig01_total_rate.eps` | `plot.py` 只读取 benchmark JSON，并写入统一隔离目录 |

若必须重算，wrapper 的 `--force` 会先备份旧 JSON。Spatial DL 还需要独立 legacy 环境；先按 `baseline/README.md` 完成 smoke test。

---

## 2. Edge-enhanced GAT 训练曲线

### 脚本与设置

入口：

```bash
uv run python -u paper_figures/fig02_training_curves/run.py \
  --stage evaluate --nodes 20,30,40,50 --seeds 0,1,2 --epochs 50
uv run python -u paper_figures/fig02_training_curves/plot.py \
  --nodes 20,30,40,50 --seeds 0,1,2 \
  --raw-root paper_figures/fig02_training_curves/data/raw \
  --output-root results/paper/tnse_current_plots
```

每个 N 使用 3 个 model seed。训练数据 seed 固定为 10086，validation/test seed 固定为 10087/10088。每次训练保存 best-validation 和 final-epoch 两个 checkpoint，并额外在规范 test set 上保存 soft、hard-unprojected 和 projected 三种诊断结果。

### 它验证什么

论文中的两张图分别展示 training loss 和 thresholded validation rate 随 epoch 的变化。训练 loss 表示 surrogate objective，validation binary rate 用于观察 threshold 后的决策是否逐渐稳定。曲线早期的波动不是“平滑收敛”，需要结合 threshold 附近输出变化解释。

### 四阶段资产

| 阶段 | 检查位置 | 说明 |
|---|---|---|
| A | perfect-CSI `train/val/test` dataset | 缺失时训练脚本可生成，但正式复现应先核对 metadata |
| B | `checkpoints/.../N*/seed*/full_300/{best_model,final_model}.pth` | 缺失组合才训练；`--force` 会覆盖训练报告和 checkpoint |
| C | `fig02_training_curves/data/raw/N*/seed*/training.json` 和 `eval_*.json` | wrapper 在训练后自动评估两种 checkpoint |
| D | `fig02_training_curves/figures/` | 正式稿使用 `fig02_training_loss.eps` 和 `fig02_validation_binary_rate.eps` |

---

## 3. 连续输出与激活比例审计

### 脚本与设置

入口：

```bash
uv run python -u paper_figures/fig03_activation_audit/run.py
uv run python -u paper_figures/fig03_activation_audit/plot.py \
  --source paper_figures/fig02_training_curves/data/raw \
  --output-root results/paper/tnse_current_plots
```

Fig. 3 不训练模型，也不重新推理。它复制并读取 Fig. 2 中相同 N、相同 model seed 的 `training.json`，统计连续 `pi` 的均值/标准差和 `pi >= 0.5` 后的激活比例。

### 它验证什么

这是对训练行为的 sanity check：确认模型没有退化成所有链路给出相同分数、全部激活或全部关闭。它不能替代 endpoint feasibility，也不能把稀疏激活比例直接解释成较高物理层速率。

### 四阶段资产

Fig. 3 的 A--C 阶段完全依赖 Fig. 2。若 Fig. 2 的 `training.json` 不存在或与当前模型不匹配，应先完成 Fig. 2；只有同步后的 Fig. 3 JSON 存在时，才进入阶段 D 运行 `plot.py`。

---

## 4. 训练数据量实验

### 脚本与设置

入口：

```bash
uv run python -u paper_figures/fig04_data_efficiency/run.py \
  --stage evaluate --nodes 20,30,40,50 \
  --sizes 10,20,30,40,60,80,100,200,300 \
  --seeds 0,1,2 --epochs 50
uv run python -u paper_figures/fig04_data_efficiency/plot.py \
  --raw-root paper_figures/fig04_data_efficiency/data/raw \
  --sizes 10,20,30,40,60,80,100,200,300 \
  --output-root results/paper/tnse_current_plots
```

当前正式图的训练样本数为 10/20/30/40/60/80/100/200/300。小数据集来自同一个 500-snapshot prefix pool 的前缀，避免把不同随机数据集造成的差别误认为样本数效应；n=300 直接复用 Fig. 2 的 `(N, seed)` 双 checkpoint。n=500 尚未形成完整正式结果，因此不进入当前图。

### 它验证什么

最终论文使用 `fig04_best_soft_utility.eps`，观察 best-validation checkpoint 的 soft utility 随训练样本数的变化，用于支持 edge-enhanced GAT 的 sample efficiency。目录中同时保存 hard-unprojected 和 projected physical-rate 诊断，它们用于检查结论是否依赖某一种解码口径，不应与论文当前 soft-utility 图混为一谈。

### 四阶段资产

| 阶段 | 检查位置 | 说明 |
|---|---|---|
| A | 每个 N 的 500-snapshot prefix pool 和规范 test set | 所有较小训练集从同一 pool 截取 |
| B | `checkpoints/.../N*/seed*/prefix_nXX/` | 每个 size/seed 保存 best 和 final；n=300 指向 `full_300` |
| C | `fig04_data_efficiency/data/raw/N*/n*/seed*/training.json` 与 `eval_*.json` | 每个 checkpoint 做 soft/hard/projected 评测 |
| D | `fig04_data_efficiency/figures/fig04_best_soft_utility.eps` | 这是当前论文使用的图；其余 EPS 是诊断 |

---

## 5. 固定 Rician 信道下的动态异质业务

### 脚本与设置

底层脚本：`experiments/exp_fixed_csi_qos.py`

论文入口：

```bash
uv run python -u paper_figures/fig05_dynamic_queue/run.py \
  --nodes 50 --slots 10000 --seeds 42 --deadline-ms 2000
uv run python -u paper_figures/fig05_dynamic_queue/plot.py \
  --profile deadline_2000ms --nodes 50 \
  --raw-root results/paper/ch5_local_coordination_v1/fig05_merged_N50_seed42 \
  --output-root results/paper/tnse_current_plots \
  --metrics backlog,delay,goodput
```

正式稿使用 N=50、10000 slots、1 ms/slot、12000 bit/packet、Dmax=2000 ms。总 arrival rate 为 0.5 到 5.0 packet/slot 的预设序列，20% hotspot links 承担 70% 流量。每个 offered load 下所有方法共享固定 Rician channel、topology、link set、arrival trace 和 seed。

正式比较七种方法：Random、Centralized Greedy、Weighted FPLinQ、Direct-Link LocalMax、GAT-LocalMax、Direct-MP 和 GAT-MP。三个论文指标是 average total queue backlog、resolved-packet deadline-capped delay 和 goodput。

### 它验证什么

该实验是第五章最重要的数值证据。比较 GAT-MP 与 GAT-LocalMax，观察在相同 GAT prior 后加入 marginal coordination 的变化；比较 GAT-MP 与 Direct-MP，观察在相同协调轮数下 GAT-derived prior 的作用。它支持的是有限 horizon、异质业务下的 QoS 改善，不是 throughput optimality 或长期队列稳定性证明。

### 四阶段资产

| 阶段 | 检查位置 | 说明 |
|---|---|---|
| A | 无独立 queue pickle；trace 在评测时按 seed 生成 | 先确认 topology/channel/arrival 的 paired 规则 |
| B | 对应 N=50、perfect CSI 的 channel-GAT checkpoint | GAT-MP 和 GAT-LocalMax 需要；Direct 系方法不需要训练 |
| C | `results/paper/ch5_local_coordination_v1/fig05_merged_N50_seed42/deadline_2000ms/N50/` | 新协议正式 `config.json`、`metrics.json`、`per_seed_metrics.json` 和 `trajectories.npz` |
| D | `results/paper/tnse_current_plots/` | `plot.py` 从新版 raw root 生成图和 plotting CSV，不覆盖历史图 |

`no_deadline` 目录是内部对照，delay 只统计完成包并可能受 right censoring 影响，不应与论文中的 deadline-capped delay 混用。

### 修改 GAT-MP 后必须注意缓存

`exp_fixed_csi_qos.py` 的 MP task fingerprint 现在包含 `marginal_protocol_version` 和 `scheduler_model_version`。因此新旧 MP 协议不会使用相同 fingerprint；但是，正式实验仍应使用独立结果目录，使 provenance、任务缓存和绘图输入保持清楚。

因此，修改第五章实现后应先保留旧目录作为“旧实现基线”，再使用新的输出目录运行底层脚本；若仍使用 Fig. 5 wrapper 的固定目录，也必须先将旧结果目录整体归档。不要只覆盖 `metrics.json` 而留下旧 `tasks/`，也不要在没有备份和 provenance 记录的情况下盲目删除缓存。

---

## 6. 跨网络规模评测

### 脚本与设置

底层脚本：`experiments/evaluate_channel_gat_cross_n.py`

论文入口：

```bash
uv run python -u paper_figures/fig06_cross_n/run.py \
  --device auto
uv run python -u paper_figures/fig06_cross_n/plot.py \
  --input paper_figures/fig06_cross_n/data/raw/cross_n_trainSize=300_seed0_epoch50.json \
  --output-root results/paper/tnse_current_plots
uv run python -u paper_figures/fig06_cross_n/export_table.py
```

模型分别在 N=20/30/40/50 上训练，然后在四种 test N 上交叉评测。每个单元格都将模型输出二值化，经 endpoint-mutual matching 后在 test-N 的真实 Rician channel 上计算 aggregate rate。最终论文表格只报告 train N 不大于 test N 的方向，用于讨论小规模训练向更大规模部署的迁移。

### 它验证什么

该实验验证所测试 N 范围内的 scale generalization，重点是局部图结构和共享参数能否从较小网络迁移到更密集网络。它不证明任意规模、任意链路分布或任意信道条件下都具有同样性能。

### 四阶段资产

| 阶段 | 检查位置 | 说明 |
|---|---|---|
| A | N=20/30/40/50 的规范 test set 和相应 training pool | 所有模型在相同 test-N 数据上比较 |
| B | `checkpoints/paper/proposed_gat/rician_K5_tx10/N*/seed0/full_300/best_model.pth` | 复用 Fig. 2 的 300-sample、50-epoch、seed-0 checkpoint |
| C | `fig06_cross_n/data/raw/cross_n_trainSize=300_seed0_epoch50.json` | 保存 train-N/test-N 矩阵和 checkpoint provenance |
| D | cross-scale EPS 和导出的 CSV/LaTeX 表格 | 当前论文使用表格而不是把整张矩阵图作为主结果 |

`evaluate_channel_gat_cross_n_raw.py` 和 `investigate_cross_n_generalization.py` 用于排查 seed、输入分布或 checkpoint 异常，不是投稿表格的正式入口。它们的结果不能替换上述正式 JSON。

---

## 7. 第四章的 imperfect-CSI 物理层评测

### 脚本与设置

底层脚本：`experiments/evaluate_channel_gat_csi.py`

论文入口：

```bash
uv run python -u paper_figures/fig07_csi_feedback/resume.py \
  --nodes 20,30,40,50 --chapters 4 --epochs 50 --device auto
uv run python -u paper_figures/fig07_csi_feedback/plot.py \
  --nodes 20,30,40,50 --chapters 4 \
  --chapter4-raw-root paper_figures/fig07_csi_feedback/data/raw \
  --output-root results/paper/tnse_current_plots
```

CSI 条件包括 perfect、large-scale 和 quantized 2/4/6/8/10 bit。每个条件使用在相同 CSI 表示下训练的 checkpoint。调度器根据 observed CSI 构图、推理和选择动作，最终 physical rate 在数据样本所携带的 realized true Rician channel 上结算。

### 它验证什么

该实验验证 edge-enhanced GAT 在低精度几何 CSI 下能保留多少物理层 sum-rate 性能。它比较的是不同 CSI 条件下各自匹配训练与测试的表现，不应写成一个 perfect-CSI 模型对未知 CSI 表示的 zero-shot transfer。

### 四阶段资产

| 阶段 | 检查位置 | 说明 |
|---|---|---|
| A | `datasets/paper/rician_K5_tx10/{perfect,large_scale,quantized_*}/N*/` | 每个 N 和 CSI 条件都需要 matching dataset |
| B | `results/models/paper/gat_numUAV=*_csi=*_channel=ricianK5dB.pth` | `resume.py` 只训练缺失 checkpoint |
| C | `fig07_csi_feedback/data/raw/chapter4_N*_v2_*.json` | 已存在 JSON 会被跳过 |
| D | `results/paper/tnse_current_plots/` | 从冻结的 Chapter IV JSON 生成 rate 图和 CSV，不覆盖历史图 |

如果 GAT、CSI 表示、endpoint decoder 或 true-channel settlement 改变，已有 Chapter IV JSON 即使文件完整也必须归档后重算。

---

## 8. 第五章的 imperfect-CSI 队列评测

### 脚本与设置

底层脚本：`experiments/evaluate_mobile_block_fading_csi.py`

论文入口：

```bash
uv run python -u paper_figures/fig07_csi_feedback/resume.py \
  --nodes 20,30,40,50 --chapters 5 --slots 10000 \
  --seeds 42 --arrival-rate 2.4 --device auto
uv run python -u paper_figures/fig07_csi_feedback/plot.py \
  --nodes 20,30,40,50 --chapters 5 \
  --chapter5-raw-root results/paper/ch5_local_coordination_v1/fig07_full \
  --output-root results/paper/tnse_current_plots
```

如果希望一次生成 Fig. 7 的两个面板，而不是分别运行 Chapter IV/V 的作图命令：

```bash
uv run python -u paper_figures/fig07_csi_feedback/plot.py \
  --nodes 20,30,40,50 --chapters 4,5 \
  --chapter4-raw-root paper_figures/fig07_csi_feedback/data/raw \
  --chapter5-raw-root results/paper/ch5_local_coordination_v1/fig07_full \
  --output-root results/paper/tnse_current_plots
```

这里使用 speed=0、每个 block 1 slot，把移动 trace 运行器用于统一生成 true/observed CSI。各 N 在 lambda=2.4 packet/slot、10000 slots、seed 42 下比较 perfect/latest、large-scale 和 quantized 2/4/6/8/10 bit。GAT-MP 对每种 observed CSI 使用匹配的 channel-GAT checkpoint，服务始终以当前 true CSI 结算。

### 它验证什么

该实验观察低精度 CSI 对 GAT-MP backlog 的影响，并与第四章的 rate 结果共同支持“较粗的调度信息仍可用于控制阶段”的论述。它没有单独改变或证明 data-transmission protocol，也不用于证明跨 CSI 模型的 zero-shot 泛化。

### 四阶段资产

| 阶段 | 检查位置 | 说明 |
|---|---|---|
| A | trace 在线生成；先确认 CSI 构造和 paired seed | 所有 CSI 方法共享同一 true trajectory 和 arrivals |
| B | 每个 N、每个 CSI 条件的匹配 channel-GAT checkpoint | 缺任一模型时 Chapter V runner 会失败或无法完成全条件比较 |
| C | `results/paper/ch5_local_coordination_v1/fig07_full/chapter5_N*_v2_matched_csi.json` | N=20/30/40/50 的七种 matched-CSI 正式结果已经完整 |
| D | `results/paper/tnse_current_plots/` | 从指定 raw root 生成 backlog 图和 plotting CSV，不覆盖历史图 |

### 修改实现后的断点文件

移动评测的断点文件现在记录并检查 `marginal_protocol_version`。同一协议内仍应保持 speed、block length、arrival rate、seed、scheduler variant 和 CSI method 一致；修改协议后应使用新的结果目录或文件名，保留旧结果作为历史基线。

---

## 9. Fig. 8 移动性诊断

### 脚本与设置

入口：

```bash
uv run python -u paper_figures/fig08_mobility/run.py \
  --nodes 20,30,40,50 --slots 5000 --seeds 42 --device auto
uv run python -u paper_figures/fig08_mobility/plot.py
```

该目录调用 `evaluate_mobile_block_fading_csi.py` 做两组 GAT-MP backlog sweep：

- speed sweep：速度 0/5/10/20 m/s，block length 固定为 5 slots；
- block-length sweep：block length 1/5/10/20 slots，速度固定为 10 m/s；
- 两者均使用 lambda=2.4、latest 与 stale CSI、4 轮协调和 GAT-assisted scheduler。

### 当前用途

现有结果没有形成稳定、清晰且适合写入论文的结论，因此 Fig. 8 已从最终论文中移除。该实验仍然有价值，但只用于检查 mobility、block aging、stale CSI 和 GAT-MP 修改之间是否出现异常趋势。除非后续得到可重复且可解释的结果，否则不要把它重新写成论文主证据。

它的 A/B/C/D 判断与上一节相同。长任务优先使用 `fig08_mobility/resume.py`，并在 GAT-MP 或移动信道实现变化后归档旧 progress/final JSON。

## 如何用这些实验验证代码修改

### 修改 edge-enhanced GAT

推荐顺序为：单元测试和小数据 smoke run，Fig. 2/3 检查训练行为，Fig. 1 检查物理层性能，Fig. 4 检查 sample efficiency，最后复查 cross-scale 和 Fig. 7 Chapter IV。若第五章继续使用该 GAT 生成 prior，Fig. 5 和 Fig. 7 Chapter V 也需要重新评测。

### 修改 GAT-MP

当前最重要的目标是让代码实现论文中的局部冲突图边际和邻域确认，而不是继续使用全局精确外部性。推荐验证顺序为：

1. 在 `tests/test_marginal_mp.py` 及新增测试中检查 inactive/active candidate 的加入与退出；
2. 构造两个 coordination neighborhoods 相交和不相交的小图，检查确认冲突与并行更新；
3. 对每轮更新检查 endpoint feasibility 和 conflict-graph utility 是否满足设计要求；
4. 用小 N、少量 slots 比较新旧 GAT-MP、Direct-MP、GAT-LocalMax，不把 smoke 数值写入论文；
5. 使用新的结果目录重跑 Fig. 5，分别检查 backlog、deadline-capped delay 和 goodput；
6. 重跑 Fig. 7 Chapter V，确认 imperfect-CSI 结论是否仍成立；
7. 仅在需要解释异常时运行 Fig. 8 mobility 诊断。

Fig. 5 是验证两阶段贡献的主实验，Fig. 7 Chapter V 是验证 CSI 条件的补充实验。Fig. 8 不承担“必须证明新算法更好”的任务。

### 修改 simulator 或指标

- 改信道生成、路径损耗、Rician fading 或 conflict graph 时，先判断静态 dataset 是否失效；若失效必须回到阶段 A。
- 改 GAT 输入处理、权重或 checkpoint schema 时，通常回到阶段 B。
- 改 queue update、deadline、drop、goodput 或 delay 定义时，不必重训 channel GAT，但 Fig. 5 和相关移动队列结果必须从阶段 C 重跑。
- 只改 `plot.py` 时保留 raw JSON，直接处于阶段 D。

## 结果完成前的最小检查表

一次结果要被视为可用于论文或代码回归，至少应满足：

- [ ] 运行命令、git revision 或源码版本、配置和时间已记录；
- [ ] dataset 和 checkpoint provenance 可追溯；
- [ ] 所有比较方法共享相同 topology、link set、channel/arrival trace 和 seed；
- [ ] 调度器只使用指定的 observed information；
- [ ] 最终服务在正确的 physical channel 上结算；
- [ ] 每个最终动作通过 shared-UAV/endpoint feasibility 审计；
- [ ] queue 中 packet count 和 packet work 守恒；
- [ ] result JSON 的方法、N、load、slot 数和 seed 完整；
- [ ] 没有因旧 task/progress cache 而跳过修改后的算法；
- [ ] 图中的指标与论文文字使用的是同一字段和同一 profile。

## 不应作为正式证据的内容

- `evaluate_channel_gat_cross_n_raw.py` 和 `investigate_cross_n_generalization.py` 的排错输出；
- `paper_figures/archive_*` 或历史 pipeline 中的旧图；
- `datasets/conference_rayleigh/` 上的会议版结果；
- 没有 provenance 的 checkpoint 或临时 JSON；
- 只跑少量 slots/单个小规模样本得到的 smoke 数值；
- 修改 GAT-MP 后被旧 `tasks/` 或 `.progress.json` 复用的结果；
- 当前 `fig08_mobility` 的趋势性结果。

逐图的完整绘图文件名和更多命令可参考 `paper_figures/README.md`。若那里保留了旧术语或旧图口径，应以当前论文、运行脚本生成的 metadata 和本文件列出的证据边界共同核对，不能只复制 README 中的一句话作为结论。
