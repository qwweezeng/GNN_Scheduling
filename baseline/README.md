# 必要 baseline、独立环境与比较边界

主 `pytorch` 环境保持不变；不得为旧 baseline 降级 PyTorch、NumPy、NetworkX 或 TensorFlow。当前只保留 FPLinQ、Spatial DL 与 DistGCN 三个外部 baseline。

| baseline | 独立环境 | 当前可用范围 | 论文边界 |
|---|---|---|---|
| Weighted FPLinQ | `gnn-fplinq` | 满缓存和静态队列 | 集中式、全接收功率矩阵、无训练 |
| Spatial DL | `gnn-spatial-dl-legacy` | 仅 Fig. 1 满缓存 | 原 50-link checkpoint 的 zero-shot pad/mask；完整 link set，不裁剪自身 |
| DistGCN-LGS | `gnn-distgcn-legacy` | 适配 smoke；可选 Fig. 5 | BA/hard-graph checkpoint 的 zero-shot adaptation，必须同时报告 Hard-Conflict LGS |

## 创建和 smoke

```bash
conda env create -f baseline/environments/fplinq.yml
conda run --no-capture-output -n gnn-fplinq python baseline/smoke_tests/smoke_fplinq.py

conda env create -f baseline/environments/spatial_dl_legacy.yml
conda run --no-capture-output -n gnn-spatial-dl-legacy python baseline/smoke_tests/smoke_spatial_dl.py
conda run --no-capture-output -n pytorch python baseline/smoke_tests/smoke_spatial_dl_paper_states.py

conda env create -f baseline/environments/distgcn_legacy.yml
conda run --no-capture-output -n gnn-distgcn-legacy python baseline/smoke_tests/smoke_distgcn.py
conda run --no-capture-output -n pytorch python baseline/smoke_tests/smoke_distgcn_end_to_end.py
```

适配器与共同接口：

- `fplinq_numpy.py`：无训练的 NumPy/SciPy FPLinQ；
- `spatial_dl_adapter.py` / `spatial_dl_worker.py`：隔离 TensorFlow 与 3D-to-2D、pad/mask/link-index 回环；
- `distgcn_adapter.py` / `distgcn_worker.py`：隔离 TensorFlow 1 checkpoint、link-index 回环和 LGS；
- `queue_schedulers.py`：Random、Hard-Conflict LGS、Centralized Greedy、Weighted FPLinQ。

Graph Embedding 因 MATLAB、Boost、旧 RDKit 和 C++ 编译依赖已从仓库移除；它不阻塞投稿，也不能被其他 baseline 冒名替代。
