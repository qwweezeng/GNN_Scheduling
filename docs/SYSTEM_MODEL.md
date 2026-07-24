# TNSE 当前系统模型与算法定义

本文档是论文、代码和实验命名的唯一依据。当前只保留第四章 **Proposed-GAT** 与第五章 **GAT-MP** 两条主线；历史 pipeline 与旧图仅在 `paper_figures/archive_pipeline/` 追溯，不能作为当前定义或结果证据。

## 1. 系统模型

无人机集合为 $\mathcal U=\{1,\ldots,N\}$。候选有向链路为 $i=(s_i,r_i)$，其中 $s_i\ne r_i$；一条候选链路同时是图学习的节点和队列系统的一条队列。正式区域为 $1000\,\mathrm m\times1000\,\mathrm m$，高度为 50--150 m，最小 UAV 间距为 20 m。

正式物理配置固定为：发射功率 10 dBm、载波 2.4 GHz、带宽 5 MHz、噪声 PSD -169 dBm/Hz、SNR 门限 5 dB、Rician $K=5$ dB。距离幂律大尺度功率为

$$P^{\rm LS}_{uv}=P_{\rm tx}\left(\frac{c}{4\pi f_c}\right)^2 d_{uv}^{-\xi},\qquad \xi=2.5,$$

小尺度复信道为

$$h_{uv}=\ell_{uv}\left[\sqrt{\frac{K}{K+1}}e^{-j2\pi d_{uv}/\lambda_c}+\sqrt{\frac1{K+1}}g_{uv}\right],\quad g_{uv}\sim\mathcal{CN}(0,1),$$

真实接收功率 $P_{uv}=P_{\rm tx}|h_{uv}|^2$。活动动作 $a_i\in\{0,1\}$ 下，真实服务速率一律以当前真实功率矩阵结算：

$$R_i(\mathbf a)=B\log_2\left(1+\frac{a_iP_{s_ir_i}}{N_0+\sum_{j\ne i}a_jP_{s_jr_i}}\right).$$

只有调度器看到的 CSI 可以变为 latest、stale、large-scale 或量化 CSI；服务端绝不使用过期/量化 CSI 结算速率。

## 2. 单射频/半双工硬约束

每架 UAV 在一个时隙至多参与一条活动有向链路：

$$\sum_{i:u\in\{s_i,r_i\}}a_i\le1,\quad\forall u\in\mathcal U.$$

共享端点冲突边的巨大输入权重只是一种图特征，不能使独立 sigmoid 在数学上保证该约束。最终动作必须经过端点互选或 `project_shared_uav_feasible`，并由 `shared_uav_action_audit` 检查。物理干扰边不是硬排斥：弱干扰链路可以并发，其影响由真实 SINR 结算。

## 3. 第四章：Proposed-GAT

**论文名称：** Proposed-GAT。
**代码：** `models.model.GATScheduler`；训练入口 `train/train_channel_gat_rician.py`；满缓存评测 `experiments/compare_gat_vs_spatial_dl.py`。

对每条链路 $i$，输入直达 SNR 特征及有向干扰图 $(\mathcal L,\mathcal E)$，GAT 输出连续优先级 $\pi_i\in[0,1]$。它训练于满缓存总速率目标，输出应解释为“候选提议/优先级”，不是最终无约束并发动作。阈值化候选经一轮端点互选后成为最终 matching；满缓存性能报告为

$$\bar R_{\rm PHY}=\frac{B}{10^6}\mathbb E\left[\sum_i R_i\right]\;\mathrm{Mbit/s}.$$

这不是队列效用 $\sum_iQ_iR_i$，也不是 goodput。

图、JSON 和 README 中统一写 **Proposed-GAT**。`paper_figures/archive_pipeline/` 内的旧内部名不参与正式图。

## 4. 第五章：GAT-MP

**论文名称：** GAT-MP。
**代码类：** `models.marginal_mp.GATAssistedMarginalMP`；运行器内部名 `GAT-Assisted MarginalMP`。
**队列入口：** `experiments/exp_fixed_csi_qos.py` 与 `experiments/evaluate_mobile_block_fading_csi.py`。

每个信道块，冻结 Proposed-GAT 仅计算一次并缓存 $\pi$ 与 GAT 诱导的局部信道分数 $C_i^{\rm GAT}$。每一个时隙才读取当前队列并以 $Q_iC_i^{\rm GAT}$ 形成 LocalMax 初始竞争骨架；随后进行固定 $K=4$ 轮加入/退出边际效用消息传递。每轮仍使用当前 $Q$，但不重新运行 GAT。

它的目标是长时队列表现：goodput、average backlog、完成率及完成包 delay。$\sum_iQ_iR_i$ 仅是固定 $(Q,H)$ 快照上的机制诊断指标，不能作为动态主图的最终物理性能。

### 必要消融与学习 baseline

| 论文名 | 代码类/内部名 | 区别 |
|---|---|---|
| Direct-MP | `DirectLinkMarginalMP` / `Direct-Link MarginalMP` | 用 $Q_iC_i^{\rm solo}$ 建骨架，后续 K=4 轮完全相同 |
| GAT-LocalMax | `GATAssistedLocalMaxScheduler` | 仅保留 GAT 骨架，不执行 MP |
| Direct-Link LocalMax | `DirectLinkLocalMaxScheduler` | 无 GAT、无 MP 的骨架参考 |
| End-to-End GAT-Q | `train/train_end_to_end_queue_gat.py` | 直接输入队列的学习 baseline，不是主方案 |

不能因结果好坏删除 Direct-MP，也不能把 GAT-MP 的全部收益归因于 GAT 先验。

## 5. 队列与移动信道

时隙为 1 ms，包大小为 12000 bit。到达为异质 Poisson：20% hotspot 链路承载 70% 总业务；默认无限缓存、无 deadline，另提供 2000 ms deadline 版本并报告 deadline drop。无 deadline 时，未完成包 delay 是 right-censored；只有完成率足够高时才解释 completed-packet delay。

移动实验在初始块建立按 $(s_i,r_i)$ 排序的 persistent link catalog。链路在个别块不可用时强制动作 0，但队列身份不改变；恢复可用后仍服务原队列。UAV 使用边界反射的恒速平面运动，Rician 散射按块相关 Gauss--Markov 演化。latest、stale、large-scale 和量化 CSI 必须共享同一条运动、信道和到达轨迹。

## 6. baseline 边界

- **Top-SNR (3%)**：集中式全局前 3% SNR 候选后再统一可行化；不是分布式算法。
- **LocalMax**：一跳冲突邻居交换直达 SNR 的分布式优先级最大选择。
- **Weighted FPLinQ**：集中式、全接收功率矩阵、无训练的优化参考。
- **Spatial DL**：固定 legacy checkpoint 的满缓存学习 baseline；仅 Fig. 1，使用完整 link set 的 pad/mask，不裁剪自身。
- **DistGCN-LGS**：零样本硬冲突图迁移；与 Hard-Conflict LGS 同时报告，不等价于连续 SINR 目标的重训学习器。
- **Centralized Greedy**：全局信息的强贪心参考，不是数学上界。

## 7. 代码入口

| 任务 | 当前入口 |
|---|---|
| 数据集 | `scripts/prepare_paper_datasets.py` |
| Proposed-GAT 训练 | `train/train_channel_gat_rician.py` |
| End-to-End GAT-Q 训练 | `train/train_end_to_end_queue_gat.py` |
| 满缓存基线 | `experiments/compare_gat_vs_spatial_dl.py` |
| 静态队列 | `experiments/exp_fixed_csi_qos.py` |
| 静态 CSI | `experiments/evaluate_channel_gat_csi.py` |
| 跨 N | `experiments/evaluate_channel_gat_cross_n.py` |
| 移动 CSI/队列 | `experiments/evaluate_mobile_block_fading_csi.py` |
| 投稿图命令 | `paper_figures/README.md` |
