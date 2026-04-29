"""
Simulator Module - 无线网络调度仿真器

信道模型说明（三种场景）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 基础实验（单时隙评估）
   - 场景：评估单个网络状态下的调度性能
   - 信道：生成一次，使用一次
   - 对应：runner.py 的 evaluate() 函数
   
2. 队列仿真（多时隙块衰落）
   - 场景：队列感知调度，评估长期稳定性
   - 信道：仿真开始时生成一次，整个仿真周期内保持不变
   - 假设：一个传输块内信道是准静态的
   
3. 长记忆仿真（未来扩展）
   - 场景：秒级时间尺度，考虑信道缓慢变化
   - 信道：每 N 个时隙更新一次（块衰落模型）
   - 体现：无人机移动导致的拓扑慢变化

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

包含：
- BaseSimulator: 仿真器抽象基类
- SimpleSimulator: 简单仿真器（单时隙）
- QueueSimulator: 队列感知仿真器（多时隙块衰落）
"""

from .base import BaseSimulator, SimpleSimulator
from .queue_simulator import QueueSimulator, compare_scheduling_algorithms

__all__ = [
    'BaseSimulator',
    'SimpleSimulator',
    'QueueSimulator',
    'compare_scheduling_algorithms',
]
