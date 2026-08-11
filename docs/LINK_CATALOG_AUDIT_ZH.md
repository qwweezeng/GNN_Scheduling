# Link catalog 一致性审计

## 审计问题

论文当前把候选链路目录定义为：在初始信道块中，根据 mean received
power 建立固定的链路身份。审计需要回答两个问题：

1. 静态数据生成代码是否遵循这个定义？
2. 如果不一致，差异是否小到可以忽略？

本审计只读取现有生成器，不修改数据、模型、checkpoint 或论文结果。

## 当前代码的两种口径

`utils/network_generate.py` 的静态 snapshot 生成器采用：

- perfect CSI：用瞬时 Rician SNR 筛选候选链路；
- large-scale / quantized CSI：用平均 SNR 筛选候选链路。

`simulator/mobile_block_fading.py` 的移动队列仿真采用：

- 第一信道块用平均功率建立 persistent flow catalog；
- 后续信道块保持链路身份，只更新 availability；
- observed state 用于调度，true instantaneous state 只用于服务结算。

因此，移动 Chapter V 仿真已经符合 persistent mean-power catalog；不一致
主要存在于静态 snapshot 数据及其训练、评测入口。

## 配对审计方法

运行指令：

```bash
.venv/bin/python -m experiments.audit_link_catalog --num-seeds 512
```

配置采用 `configs/paper.yaml`：Rician K = 5 dB、SNR threshold = 5 dB，
网络规模为 20、30、40、50。每个随机种子只生成一次拓扑和瞬时信道，随后
在完全相同的样本上分别应用：

- instantaneous-SNR catalog rule；
- mean-SNR catalog rule。

因此，差异只来自 catalog rule，不来自两次独立采样。

## 审计结果

| UAV 数量 | mean catalog 平均链路数 | instantaneous catalog 平均链路数 | 平均 Jaccard | 对称差 / mean catalog |
|---:|---:|---:|---:|---:|
| 20 | 170.73 | 157.54 | 0.7224 | 0.3113 |
| 30 | 388.73 | 361.25 | 0.7226 | 0.3114 |
| 40 | 696.73 | 646.65 | 0.7216 | 0.3121 |
| 50 | 1096.96 | 1017.55 | 0.7222 | 0.3111 |

以 N = 50 为例，每个样本平均有：

- 130.77 条链路只出现在 instantaneous catalog；
- 210.18 条链路只出现在 mean-power catalog；
- 两种 catalog 的对称差合计为 mean catalog 的约 31.1%。

差异链路的 mean SNR 中位数约为 5.5 dB，接近 5 dB 灵敏度阈值，但其
5%--95% 范围约为 2.25--10.60 dB。这说明差异并非一个可以忽略的数值边界
误差，而是瞬时 Rician 衰落导致链路在阈值两侧进出的正常结果。

原始逐种子结果保存在：

`results/validation/catalog_audit/mean_vs_instantaneous_ricianK5.json`

## 本阶段决策

本阶段不修改静态 catalog 生成器，也不重新训练 GAT，理由如下：

1. catalog 差异约为 31%，修改后会实质改变训练和测试图；
2. 已有 perfect-CSI checkpoint 是在 instantaneous-selected snapshots 上训练的；
3. 修改 catalog 会使第四章数据、checkpoint 和主要结果同时失效；
4. 当前任务明确冻结第四章，并优先完成 Chapter V local coordination。

本阶段采用的处理是：

- 保留静态 snapshot 实验的现有 condition-specific candidate-set 口径；
- 保留移动 Chapter V 的 persistent mean-power flow catalog；
- 在实验 metadata 中明确记录 catalog policy；
- 后续修改论文措辞，分别描述 static snapshot 与 mobile queue 两种评测协议。

这个决策不是说两种 catalog 等价，而是明确承认它们不等价，并避免在没有
全量重训的情况下把新 catalog 与旧 checkpoint 混用。

## 如果以后坚持统一 mean-power catalog

需要单独立项执行：

1. 静态生成器始终按 mean power 建立链路身份；
2. perfect CSI 只改变特征、边权和 availability；
3. 更新 dataset fingerprint；
4. 重新生成训练、验证和测试数据；
5. 重新训练全部 GAT checkpoint；
6. 重跑第四章和第五章全部依赖结果。

在完成这些步骤之前，不能只改生成器后继续加载旧 checkpoint，并把结果解释为
同一训练协议下的直接比较。
