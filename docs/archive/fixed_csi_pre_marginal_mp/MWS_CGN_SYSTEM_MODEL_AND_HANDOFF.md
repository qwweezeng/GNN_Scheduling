# 旧MWS-CGN名称迁移说明

> **历史归档（2026-07-18）**：旧文件曾错误保留`LocalMax → Join → MarginalMP`描述。
> 当前主流程已经通过消融删除Join，统一为`LocalMax → K轮MarginalMP`。

`MWS-CGN`和`QC-LMR`已于2026-07-18停止作为正式方法名称，因为它们没有直接体现
“边际效用消息传递”。归档中的旧结构不再代表当前算法。

当前正式文档为：

> `docs/MARGINAL_MP_SYSTEM_MODEL_AND_HANDOFF.md`

名称对应关系：

| 旧名称 | 当前正式名称 |
|---|---|
| MWS-CGN | Direct-Link MarginalMP |
| QC-LMR / PurePhysicalReuse | Direct-Link Join |
| LocalMax-SoloRate | Direct-Link LocalMax |
| LocalPhysicalReuse-v2 | GAT-Assisted Join |
| LocalMax-GAT | GAT-Assisted LocalMax |
| GAT-initialized MWS-CGN | GAT-Assisted MarginalMP |

旧代码导入和旧实验命令保留兼容映射，但新论文、图例和结果文件不得再使用旧名称。
