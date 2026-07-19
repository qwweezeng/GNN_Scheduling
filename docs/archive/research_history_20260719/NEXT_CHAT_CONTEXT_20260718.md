# 下一 Chat 交接：TNSE 队列感知 UAV 调度当前冻结状态

> 更新时间：2026-07-18。固定CSI队列后端已正式命名为MarginalMP。本文中的Pure、
> LocalPhysicalReuse-v2等词只用于描述历史诊断；当前方法定义以
> `docs/MARGINAL_MP_SYSTEM_MODEL_AND_HANDOFF.md`为准。旧交接已移入
> `docs/archive/20260715_pre_correction/`，不得从归档版本继续迭代。

## 1. 首先阅读什么

按以下顺序阅读即可，不需要再翻散落实验文档：

1. `docs/MARGINAL_MP_SYSTEM_MODEL_AND_HANDOFF.md`：当前完整系统模型、
   Direct-Link/GAT-Assisted初始化、MarginalMP公式、算法、baseline和CSI接口；
2. `docs/GAT_MWS_TOPOLOGY_SCALING_TUTORIAL.md`：GAT-Pure 根因诊断、
   所有低成本探测、两次严格训练闸门和最终止损结论；
3. `docs/MOBILE_BLOCK_FADING_CSI_TUTORIAL.md`：功率/数据校准、Rician 移动信道、
   latest/stale/large-scale/quantized CSI、GRU 和旧 pi 接口的完整历史；
4. `datasets/README.md`：正式数据与会议遗留数据的可追溯规则。

## 2. 正式物理与数据口径

- UAV 数量：N=20；N=10 已退出当前实验线；
- 信道：LoS 主导 Rician，K=5 dB；
- 发射功率：训练 10 dBm，测试/部署 10 dBm；
- CSI：当前结构诊断使用 fixed/latest perfect CSI；
- 正式 channel GAT checkpoint：
  `results/models/gat_numUAV=20_tx=10dBm_csi=perfect_channel=ricianK5dB.pth`；
- 正式数据：`datasets/formal/channel_gat_power_transfer/N20/rician/`。

train-13/test-10 相对 matched-10 只有约 0.10%--0.14% 的不稳定小增益，因此已放弃。
重新生成的 Rayleigh 和 N=10 数据已删除；原始会议 N=20/30/40/50 缓存保留原位置，
只用于追溯，不能被正式 loader 静默复用。

## 3. 当前冻结架构：后续一律从这里开始

完整调度链只有两个顺序部分：

```text
Part I:  Quality score → LocalMax initial skeleton A^(0)
Part II: A^(0) → K rounds of marginal-utility message passing → A^(K)
```

- 方法一般写为$K$轮MarginalMP；当前统一实验预算为$K=4$；
- 每轮基于最新真实0/1活动集合，计算链路加入或退出对$\sum_iQ_iR_i$的边际变化；
- 正边际候选在冲突邻域内并行仲裁，获胜者加入或退出；
- 主流程**没有Join**。历史Join只允许加入，与MarginalMP功能重复，已由结构消融删除；
- `Direct-Link MarginalMP`使用$Q_iC_i^{\rm solo}$构造LocalMax，是固定CSI最终主方法；
- `GAT-Assisted MarginalMP`只替换Part I的质量来源，后端完全相同，保留给CSI/GAT归因；
- 不能声称$K=4$理论最优，也不能把Centralized Greedy称为数学上界。

正式代码只有以下入口：

```text
models/marginal_mp.py
    DirectLinkMarginalMP          # 最终主方法
    GATAssistedMarginalMP         # 学习辅助接口

models/initial_schedule_construction.py
    DirectLinkLocalMaxScheduler   # 主方法Part I
    GATAssistedLocalMaxScheduler  # GAT实例Part I
    DirectLinkJoinScheduler       # 唯一内部结构baseline

baseline/queue_schedulers.py
    HardConflictLocalGreedyScheduler  # 外部分布式LGS baseline
    CentralizedMWSGreedyScheduler     # 集中式强参考
```

正式实验入口：

```bash
conda run --no-capture-output -n pytorch python -u \
  experiments/compare_marginal_mp_vs_lgs.py

conda run --no-capture-output -n pytorch python -u \
  experiments/exp_fixed_csi_qos.py \
  --methods "Direct-Link MarginalMP,Direct-Link Join,Hard-Conflict LGS,Centralized Greedy"
```

除非明确进行GAT/CSI归因，下一任务不得重新启用Join-initialized MarginalMP、MWS-CGN、
PurePhysicalReuse、LocalPhysicalReuse-v2或其他归档版本。

## 4. 已确认的历史模型事实

### 4.1 会议 GAT 没有坏

正确 checkpoint 的 $\pi$ 是会议 channel-only sum-rate 调度概率，不是新定义的 prior。
在原任务中它保持稀疏、能提高无队列全网容量。此前“pi 接近全 1”来自错误/不匹配
checkpoint 与数据口径，不能作为会议模型结论。

### 4.2 会议 pi 不能直接证明队列调度有用

期刊逐时隙目标是

$$
\max_{\mathbf a}\sum_iQ_iR_i(\mathbf a,H).
$$

会议 $\pi$ 为 sum-rate 训练，偏好信道上容易取得高容量的链路；Pure 使用
$Q_iC_i^{\rm solo}$，更忠实保留队列紧急性。正确 N=20 对照发现：旧 GAT-assisted
接口在低/中负载不稳定或更差，极高负载的少量收益又能被同均值 Constant 复制，尚无
图特异性证据。

### 4.3 Pure 的真正缺口

Pure 知道“我自身有多紧急、多强”，不知道“我获胜会排挤哪些高价值邻居”。集中式
一次骨架替换 oracle 在 heterogeneous 和 saturated 队列分别有 +16.23% 和 +8.29%
平均空间，说明空间复用与多用户分集的权衡确实没有解决完。

## 5. MarginalMP定型前依次做了什么

### R0：骨架替换上界

从 Pure 骨架枚举一次 challenger 替换，重新执行相同条件复用，再用真实 SINR 结算。
36 个快照平均 +12.52%，30/36 改善。它只证明优化空间，不是可部署算法。

### R1：每链路 scalar-z GAT

冻结会议 encoder，新加一轮图读出，让 $W_i=z_iQ_iC_i^{\rm solo}$。未见验证排序
准确率只比 Pure 高 0.48 个百分点，完整效用闸门失败。结论：集合条件机会成本不能
稳定压成骨架形成前的单链路标量。

### R2：四种无训练静态拓扑公式

degree、邻居效用占比、物理净收益和收益/代价比都不稳定，没有一项同时通过异质、饱和
和稀疏队列闸门。结论：高 degree 不等于应该让步，统一扣分会破坏多用户分集。

### R3：局部 challenger 替换可实现性

一跳最终局部收益与真实全网收益相关系数 0.968，但仍有 8/36 次选择使全网效用下降。
两跳能逐样本复现 oracle，却在 98.8% 候选上覆盖全网；它是近集中式计算，不是低交互
分布式成功。

### R4：set-conditioned challenger GAT

冻结会议 encoder，新增两轮 GAT，输入只允许本地骨架、一跳物理量和队列；同时训练
同输入零消息 MLP。未见评估中 GAT 始终选择“不替换”，相对 Pure 增益 0；MLP 在
heterogeneous 平均下降 4.81%。最终闸门失败。

### R5：Q-aware GAT 直接输出完整骨架

用户进一步要求让 $Q$ 进入图消息，并由 GAT 同时输出所有链路的 keep/add/drop。已有
oracle 审计确认必要性：同一信道图只改变三种队列后，12/12 图的最佳修改变化，队列
状态两两比较时 94.44% 的 challenger 不同。但 8/2/8 张图、10 epochs 的小规模监督
闸门中，两轮 Q-aware GAT 和同输入 MLP 的 changed-node recall 都为 0，均完整保留
Pure；未见 heterogeneous/saturated oracle 仍有 +13.53%/+11.67% 空间。结论是
“Q-aware 必要但不充分”，整骨架逐节点输出仍未解决联合翻转的信用分配。

### R6：连续会议 pi 进入 expected-SINR 与空队列掩码

用户提出把未二元化的会议调度概率直接代入 SINR，再用该速率形成队列骨架。审计确认
这正是当时评估的 `LocalPhysicalReuse-v2`：它用连续 pi 加权交叉增益，得到 expected rate，
再以 `Q * expected_rate` 形成局部胜者。此前漏测的唯一自然修正，是只让非空队列链路
贡献期望干扰。12 张未见 N=20 Rician 图上，该 mask 在 sparse 队列相对原 v2 平均
+3.51%，但只赢 4/12；在 heterogeneous 反而 -1.58%，saturated 与原式完全相同。
同均值 Constant mask 也能复制 sparse 变化。因此这是物理语义修正，不是稳定的 GAT
增益，闸门结论为 `MASK_CORRECTION_NOT_MATERIAL`。

### R7：Pure/GAT 双候选上界

为避免要求 channel-only GAT 直接理解队列，进一步让 Pure 产生 queue-urgent 候选，
让连续-pi v2 和会议 hard seed 产生 channel-efficient 候选，再由集中式真实
`sum(QR)` oracle 选最好者。GAT 候选库相对 Pure 在 sparse/heterogeneous/saturated
平均为 +5.30%/+3.05%/+1.84%，但 GAT 候选分别只在 4/12、3/12、4/12 图中被选中；
同均值 Constant 候选库能复制或超过这些收益。闸门结论为
`STOP_GAT_CANDIDATE_PORTFOLIO`，不开发分布式二候选 selector。

### R8：块级骨架码本闸门

按用户要求，码本只产生 $K$ 个骨架；队列先选择一个，再只执行一次现有条件复用。
无队列 physics-multistart 的 $K=8$ 在 heterogeneous 即使把全部候选完成后由 oracle
选择也只有 Pure 的0.9305。更强的 calibration teacher 用独立校准队列收集 Pure 和
一次最佳替换骨架后压缩为固定码本；正式未见 heterogeneous 中，$K=8$ 的合法选择和
all-completions oracle 分别为0.8807/1.0147，oracle 只赢5/12图。saturated 的 oracle
虽达到1.0596、赢9/12，但合法的“骨架QR预选后只完成一次”只有0.9849，且只在2/12图
选中最终最佳候选。候选越多，覆盖上界提高而选择错误加重。闸门结论为
`STOP_CODEBOOK_CEILING_NOT_MATERIAL`，因此未运行现有GAT码本归因，也未训练多头GAT。

## 6. 当前冻结决定

停止所有 GAT-MWS 模型迭代：

- 不再增加层数、epoch、GRU、STGNN、辅助 loss 或阈值补丁；
- 不把 1% 以下或只在个别 seed 出现的收益当作进展；
- 不进入该失败模型的动态队列与 stale CSI 训练；
- 不把 R5 转入 direct-$\sum QR$ fine-tune；
- 不继续给连续 pi 加 scalar、mask、temperature 或二候选选择补丁；
- 不训练 R8 多头骨架码本，不用 $K$ 份协商掩盖选择接口失败；
- `set_conditioned_challenger_gate.../gat_model.pth` 是被拒绝的诊断模型，不能正式加载。

当前固定CSI主干是Direct-Link LocalMax以及固定预算的$K$轮MarginalMP，正式实验取
$K=4$。结构消融已删除主流程中功能重复的add-only Join；历史Direct-Link Join只保留
为关键baseline。
会议GAT仍保留原始channel-only贡献，并通过GAT-Assisted MarginalMP接口参与CSI扩展；
当前不能声称它已提高固定CSI下的队列稳定域。

## 7. 科学结论与可能的未来工作

当前限制不是“GAT 不够大”，而是信息—开销矛盾：一次 challenger 的真实价值取决于
替换后完整 active set 和远端累计干扰。局部信息偶尔判断错；收集到两跳又几乎等于
全网。未来若重新开启，必须先提出能同时满足以下三点的新信息接口，而不是再训练当前
头部：

1. 明确描述当前 active set，而非单链路静态概率；
2. 用固定、可核算的少量消息估计远端累计外部性；
3. 在未见拓扑稳定超过 Pure，且图模型超过同输入 MLP 至少 1%。

R8 已进一步验证信道块级骨架码本。异质队列覆盖不足，saturated 虽存在候选上界，
但骨架完成前无法可靠选择；解决它需要 $K$ 份协商或重新学习集合条件完成价值。因此
该方向已从“未验证未来工作”改为当前截稿期停止路线。

十天投稿期限下，当前不再承担这项高风险研究。优先写扎实：Rician UAV 信道修正、
queue-faithful 两阶段物理调度、Pure/GAT/Constant 消融、CSI 敏感性边界与真实消息开销。

## 8. 移动CSI最终闸门（2026-07-18新增）

在正式N=20、10 dBm、Rician K=5 dB条件下，观测CSI使Direct-Link MarginalMP损失约
6.3%--13.0%。因果分解显示主要损失来自四轮边际计算，而不是初始LocalMax骨架。真实
边际可追回6.4%--14.2%，但只删除“观测为正、真实为负”的提案时，六个CSI×队列单元
中只有4-bit/heterogeneous达到5% oracle门槛。

随后完成两项止损实验：

1. 现成N=10 Link-GRU接Direct-Link MarginalMP，只比最强非学习预测提高1.05%和
   1.34%，停止N=20重训与时间模型扩展；
2. N=20小规模提案可靠性分类中，本地MLP与一层GAT验证平衡准确率只有50.68%和
   50.51%；在线调度无任何单元达到5%，GAT也不超过MLP。独立核算时，GAT新增的一轮
   信息聚合使总节点广播数达到原MarginalMP的9.86--19.81倍，远大于省下的提案通知。

最终决定：`STOP_LEARNING_ASSISTED_MARGINAL_MP`。路线C（完整边际残差回归）由用户明确
取消，禁止重新开启；不再训练更深GAT、GRU或STGNN。详细教程和结果路径见
`docs/MARGINAL_MP_SYSTEM_MODEL_AND_HANDOFF.md`第16节。

用户随后只重新开放了一条更窄的物理假设：现有Link-GRU只输入功率而丢失Rician复相位。
第17节已完成全部最终闸门并得到新冻结结论：复CSI条件预测在$K=5$ dB区域通过；同输入
Complex MLP与Link-GRU测试MAE为2.281/2.275 dB，GRU没有独立价值；897参数Complex MLP
在动态队列中相对stale使$\Lambda=2.2/4$的delivered throughput提高2.59%/9.64%，backlog
降低12.94%/12.50%，通过4/5与5/5配对闸门。

正式接受`Complex-Geometry CSI Predictor (CG-CSI)`，完整鲁棒实例为
`CG-CSI-Assisted Direct-Link MarginalMP`。它集中训练、逐pair本地执行，无隐藏状态、
无GAT、无新增邻居消息轮。正式checkpoint为
`results/models/complex_geometry_csi_N20_ricianK5_20260718.pth`。复CSI若不能由接收端本地
保留而需集中反馈，则相对power-only多1个float/pair/block；N=20全部pair、5 ms块的
上限为2.432 Mbit/s。最终停止所有GRU/GAT/STGNN扩展，不得继续调参。

## 9. 运行环境与安全要求

```bash
conda run --no-capture-output -n pytorch python -u <script.py> ...
```

工作区含用户原有改动。不得 `git reset`、工作区级清理或从归档源码恢复旧模型。实验
记录继续采用“动机—定义—实现—与上一版区别—结果—分析—下一步”，新概念先解释。
