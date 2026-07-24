# 当前实验底层运行器

本目录不再保存模型探索、GAT 结构搜索或历史结果。投稿图的实际入口是 `paper_figures/figXX_*/run.py`；这些脚本只调用下列底层运行器：

| 脚本 | 作用 | 对应图 |
|---|---|---|
| `compare_gat_vs_spatial_dl.py` | 满缓存 Proposed-GAT 与外部/传统 baseline | Fig. 1 |
| `evaluate_channel_gat_cross_n.py` | Proposed-GAT train-N/test-N 互测 | Fig. 6 |
| `evaluate_channel_gat_csi.py` | 静态 CSI 反馈下的 Proposed-GAT | Fig. 7a |
| `exp_fixed_csi_qos.py` | 固定 Rician CSI 下的 GAT-MP、baseline 动态队列 | Fig. 5 |
| `evaluate_mobile_block_fading_csi.py` | 移动 block-fading 下的 GAT-MP | Fig. 7b、Fig. 8 |

命名规则：代码内部 `GAT-Assisted MarginalMP` 显示为 **GAT-MP**；`Direct-Link MarginalMP` 显示为 **Direct-MP**；`GATScheduler` 显示为 **Proposed-GAT**。任何新脚本应沿用这一映射。

不要直接在此目录添加新模型。若需要最小成本诊断，应先修改 `paper_figures` 的 Gate 0 参数，并记录配置、checkpoint、fingerprint 与 seed。
