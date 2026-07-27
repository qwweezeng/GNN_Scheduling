# 新 Chat 启动说明（避免重蹈覆辙）

## 项目核心任务
IEEE TNSE 专刊投稿，3 天内完成 Fig. 1-8 全部仿真与核验。

## 必读文件（优先顺序）
1. `paper_figures/README.md` — **四阶段入口协议，每个字都要读懂**
2. `docs/EXPERIMENTS.md`
3. `configs/paper.yaml` — 参数含义

## 最重要的规则：入口即一切

**所有仿真必须走 fig01–fig08 各自的 `run.py`**，不可自己写临时脚本。

| 阶段 | 命令 | 做什么 |
|---|---|---|
| A. 数据 | `--stage data` | 生成数据集（固定 seed 10086/10087/10088，300/128/128） |
| B. 模型 | `--stage train` | 训练 checkpoint |
| C. 评估 | `--stage evaluate --force` | 运行 baseline 对比（**所有方法都投影**） |
| D. 作图 | `plot.py` | 只读 JSON 出 EPS |

**每个阶段只做一件事，不混用。**

## fig01 N=20 最新评估结果（来自正确的 run.py）

| Method | bit/s/Hz |
|---|---|
| LocalMax | 22.98 |
| FPLinQ | 22.22 |
| Proposed-GAT | 20.27 |
| Local Greedy | 19.03 |
| Top-SNR | 17.37 |
| Random | 4.89 |

GAT 比 LocalMax 低约 12%，比 FPLinQ 低约 9%。

## 排查方向建议

LocalMax 天然满足 shared-UAV 约束（同一 UAV 的发射/接收链路互斥），但可能在 SNR 排序后存在"不同链路的邻居中 SNR 都不是局部最大"的情况导致激活率偏低。可以核验：
1. LocalMax 激活链路比例（确认是否激活链路太少）
2. `models/local_physical_reuse.py` 的 `select_local_maxima` 实现是否正确
3. fig01 中 LocalMax 的 `project_shared_uav_feasible` 投影后被修改了多少次

## 严禁事项
- 禁止自己写 `generate_dataset()` 不设 seed 直接调用
- 禁止跳过 `paper_figures/` 入口自己跑 baseline 评估
- 禁止用 5 epoch / lr=1e-3 / train_size=50 的非正式训练做对比
