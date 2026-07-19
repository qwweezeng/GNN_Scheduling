# 投稿实验入口

本目录只保留能直接对应论文图或必要审计的实验。

| 脚本 | 对应内容 |
|---|---|
| `compare_gat_vs_spatial_dl.py` | 满缓存Channel-GAT与baseline |
| `convergence_analysis.py` | GAT收敛和训练样本量 |
| `scaling_experiment.py` | GAT跨UAV规模泛化 |
| `csi_imperfection_study.py` | 满缓存CSI表示敏感性 |
| `compare_marginal_mp_vs_lgs.py` | 相同$(Q,H)$快照的机制对比 |
| `exp_fixed_csi_qos.py` | 固定CSI动态队列；正式任务需显式选择GAT-assisted主方案和Direct-Link消融 |
| `evaluate_mobile_block_fading_csi.py` | 当前仍是Direct-Link移动CSI入口；待扩展为两种初始化×四种CSI，CG可选 |
| `runner.py` | 会议实验兼容runner，不是单独论文方法 |

已停止的模型探测不在活动目录中。源码归档位于`archives/`，旧结果入口位于
`experiments/archive/`或`docs/archive/`。

完整设置、baseline依赖和正式接口对齐闸门见`docs/EXPERIMENTS.md`。在该文档第8节完成前，
不要把移动脚本的旧默认输出当作投稿版GAT-assisted结果。
