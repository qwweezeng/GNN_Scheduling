# GNN Scheduling for Dense UAV Networks

本仓库是TNSE期刊投稿的可复现代码。当前研究已经收束：不再加入新的模型路线，只保留
会议版满缓存Channel-GAT、期刊版队列调度GAT-Assisted MarginalMP、去GAT的
Direct-Link必要消融，以及一个默认关闭的CG-CSI支线。

## 1. 从哪里开始阅读

1. [`docs/SYSTEM_MODEL.md`](docs/SYSTEM_MODEL.md)：系统模型、两类调度任务、算法公式、
   baseline边界和消息开销口径；
2. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)：三段仿真逻辑、七幅投稿主图、baseline
   改造表和正式运行前的接口闸门；
3. [`configs/paper.yaml`](configs/paper.yaml)：投稿默认物理参数和训练参数；
4. [`README-base.md`](README-base.md)：原会议代码的历史运行说明，仅用于追溯。

## 2. 论文与代码的对应关系

### 第三章：满缓存Channel-GAT

每条候选链路一直有数据。GAT读取直达SNR和干扰图，输出调度概率$\pi_i$，目标是最大化
网络总Shannon速率。

- `models/model.py`：Channel-GAT与统一Shannon速率；
- `train/train_channel_gat_rician.py`：正式Rician训练、测试和坍缩检查；
- `experiments/compare_gat_vs_spatial_dl.py`：满缓存baseline；
- `experiments/convergence_analysis.py`：收敛与训练样本量；
- `experiments/scaling_experiment.py`：跨UAV规模泛化；
- `experiments/csi_imperfection_study.py`：满缓存CSI输入敏感性。

### 第四章：GAT-assisted队列感知MarginalMP

每条链路有独立队列。主方案复用第3章GAT的连续调度倾向$\pi$，把它转换为考虑邻居
期望干扰的信道分数$C_i^{GAT}$，再用$Q_iC_i^{GAT}$产生局部胜者骨架；随后进行固定
$K=4$轮加入/退出边际效用消息传递。`Direct-Link MarginalMP`使用$Q_iC_i^{solo}$形成
骨架，其余后端完全相同，是必须报告的去GAT消融。

- `models/initial_schedule_construction.py`：LocalMax骨架和Join消融；
- `models/marginal_mp.py`：主算法及精确加入/退出边际；
- `models/local_physical_reuse.py`：共享物理计算和GAT-assisted消融；
- `simulator/queue_simulator.py`：固定CSI动态队列；
- `simulator/mobile_block_fading.py`：永久link identity、移动Rician block-fading、真实/
  观测CSI分离；
- `experiments/compare_marginal_mp_vs_lgs.py`：相同$(Q,H)$快照机制对比；
- `experiments/exp_fixed_csi_qos.py`：固定CSI goodput/backlog/delay；
- `experiments/evaluate_mobile_block_fading_csi.py`：latest/stale/large-scale/4-bit CSI。

### 可选CG-CSI支线

`models/complex_geometry_csi.py`根据aged复CSI和几何预测当前功率矩阵，然后仍调用同一个
MarginalMP。默认实验不启用它；只有命令行添加`--include-cg-csi`才运行。它不是论文主
贡献，也不包含GAT或GRU。

## 3. 目录结构

```text
GNN_Scheduling/
├── baseline/      第三方baseline原代码与本项目统一适配器
├── bin/           会议版兼容入口
├── configs/       投稿唯一默认配置paper.yaml和历史兼容配置
├── datasets/      原会议数据；正式Rician数据位于datasets/paper/
├── docs/          当前系统模型、实验说明、论文与历史归档
├── experiments/   投稿图对应的评测入口
├── models/        Channel-GAT、MarginalMP、必要消融、可选CG-CSI
├── scripts/       数据准备和artifact审计工具
├── simulator/     静态/移动信道与队列仿真
├── tests/         物理公式、接口和artifact一致性测试
├── train/         Channel-GAT及End-to-End GAT-Q消融训练
├── utils/         数据、指标、baseline和作图公共函数
└── archives/      已停止的研究探测源码及云端历史快照
```

## 4. 统一调度器接口

队列方法实现：

```python
scheduler.eval()
action = scheduler.schedule_with_state(state, queues)
```

其中`state`携带候选link identity、观测功率矩阵、冲突邻接和噪声；`queues`与link identity
一一对应；返回值是形状`[num_links, 1]`的二元动作。服务速率由仿真器使用真实信道统一
结算，调度器不能自行改写真实信道。

## 5. 最短复现流程

环境：

```bash
conda run --no-capture-output -n pytorch python -u <script.py> <args>
```

先生成N=20/30/40/50正式数据：

```bash
conda run --no-capture-output -n pytorch python -u \
  scripts/prepare_paper_datasets.py --num-uavs 50,40,30,20
```

训练一个N=20 Channel-GAT：

```bash
conda run --no-capture-output -n pytorch python -u \
  train/train_channel_gat_rician.py --num_uav 20 --epochs 30
```

固定CSI脚本已经能通过显式`--methods`同时选择两种MarginalMP。正式运行必须显式给出
经过审计、且与N/功率/Rician K匹配的Channel-GAT checkpoint：

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/exp_fixed_csi_qos.py --num_uav 20 --num_slots 10000 \
  --methods "GAT-Assisted MarginalMP,Direct-Link MarginalMP,Direct-Link LocalMax,Direct-Link Join,Hard-Conflict LGS,Random,Centralized Greedy" \
  --channel_gat_checkpoint <AUDITED_N20_RICIAN_K5_TX10_CHECKPOINT>
```

移动CSI脚本当前仍是上一版Direct-Link单调度器入口。下一项正式仿真任务必须先完成
`docs/EXPERIMENTS.md`第8节的二维配对接口，再运行移动主图；当前旧默认命令不能作为投稿
版GAT-assisted结果。

全部投稿图的定义、baseline和接口闸门见[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。
完成Gate 0后，下一任务再把最终可运行命令逐条回填到README。统一作图入口是
`scripts/plot_paper_figures.py`；固定CSI QoS脚本会直接生成自己的多指标图。

## 6. 数据与模型防误用规则

正式静态数据路径为：

```text
datasets/paper/rician_K5_tx10/perfect/N{N}/
```

每个目录同时保存pickle数据和`manifest.json`。代码校验信道类型、Rician K因子、发射
功率、UAV数量、split大小、随机种子和fingerprint；文件名相似但metadata不匹配时拒绝
加载。

快速检查数据manifest（训练完成后可加`--require-models`同时检查checkpoint）：

```bash
conda run --no-capture-output -n pytorch python -u \
  scripts/verify_paper_artifacts.py --num-uavs 20,30,40,50
```

正式GAT checkpoint位于`results/models/paper/`，并保存同样的物理metadata和非坍缩
审计结果。最终作图前重新训练；不得因目录中恰好已有`.pth`就静默加载。

原会议数据`datasets/dataset_numUAV=20/30/40/50.pkl`和原结果保留原位置，避免破坏历史
图的追溯，但它们不是Rician投稿数据。

## 7. 保留的baseline

满缓存：Random、Top-SNR、Local Greedy、Spatial DL、FPLinQ和Channel-GAT。

动态队列：Random、GAT/Direct-Link LocalMax、Direct-Link Join、Hard-Conflict LGS、
End-to-End GAT-Q、Weighted FPLinQ、Centralized Greedy、Direct-Link MarginalMP和
GAT-Assisted MarginalMP。

第三方依赖缺失时程序会明确失败，不会用另一方法替代。Spatial DL不能未经队列输入和
重新训练就称作QoS baseline；DistGCN只有完成checkpoint和TensorFlow适配验证后才能
报告数值。

## 8. 当前结论边界

- Channel-GAT保留为满缓存总速率方法；
- GAT-Assisted MarginalMP是为了跨章节连续性选定的队列调度主方案；
- Direct-Link MarginalMP是必要消融；现有证据不支持“GAT增强MarginalMP”的因果说法，
  正式配对实验必须如实报告二者差异；
- CG-CSI只保留为可选鲁棒性支线，默认关闭；
- Centralized Greedy是强参考而非数学上界；
- 所有消息轮数、payload、本地计算和CSI获取开销必须分开核算。
