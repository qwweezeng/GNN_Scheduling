"""
Base Simulator - 仿真器基础模块

提供仿真器的通用框架，子类只需实现核心逻辑。

设计说明：
- 使用组合而非继承：公共逻辑在基类，差异化逻辑在子类
- 显式区分"信道更新"和"状态更新"：便于理解不同场景的行为差异
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict


class BaseSimulator(ABC):
    """
    仿真器抽象基类
    
    定义仿真器的通用流程，子类实现以下方法：
    - _generate_channel(): 生成信道状态
    - _schedule(): 执行调度决策
    - _update_queue(): 更新队列状态（如果有）
    
    仿真流程：
        ┌─────────────────────────────────────────────┐
        │                  run()                       │
        │  ┌───────────────────────────────────────┐  │
        │  │ 初始化：_generate_channel() (只执行一次)  │  │
        │  └───────────────────────────────────────┘  │
        │                    ↓                        │
        │  ┌───────────────────────────────────────┐  │
        │  │         for slot in range(N):         │  │
        │  │                                       │  │
        │  │   ┌───────────────────────────────┐    │  │
        │  │   │  1. _schedule() → 调度决策    │    │  │
        │  │   └───────────────────────────────┘    │  │
        │  │                    ↓                   │  │
        │  │   ┌───────────────────────────────┐    │  │
        │  │   │  2. _update_queue() → 队列更新│    │  │
        │  │   └───────────────────────────────┘    │  │
        │  │                    ↓                   │  │
        │  │   ┌───────────────────────────────┐    │  │
        │  │   │  3. 记录历史                   │    │  │
        │  │   └───────────────────────────────┘    │  │
        │  └───────────────────────────────────────┘  │
        └─────────────────────────────────────────────┘
    """
    
    def __init__(self, network_generator, model, device='cpu', seed=None):
        """
        Args:
            network_generator: NetworkGenerator 实例
            model: 调度模型
            device: 计算设备
            seed: 随机种子
        """
        self.net_gen = network_generator
        self.model = model
        self.device = device
        self.seed = seed
        
        if seed is not None:
            self._set_seed(seed)
        
        # 历史记录
        self.history = defaultdict(list)
        
        # 当前状态
        self.current_state = None
        self.slot = 0
        
        # 调用子类初始化
        self._init()
    
    def _set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def _init(self):
        """子类实现：初始化特定的状态变量"""
        pass
    
    @abstractmethod
    def _generate_channel(self, csi_mode="perfect", q_bits=8):
        """
        子类实现：生成信道状态
        
        Args:
            csi_mode: 信道状态信息模式
            q_bits: 量化比特数
        
        Returns:
            state: NetworkState 对象
        """
        pass
    
    @abstractmethod
    def _schedule(self, state, use_ste=False):
        """
        子类实现：执行调度决策
        
        Args:
            state: 当前网络状态
            use_ste: 是否使用 STE
        
        Returns:
            dict: 包含调度决策信息的字典
        """
        pass
    
    @abstractmethod
    def _update_queue(self, schedule_result):
        """
        子类实现：更新队列状态
        
        Args:
            schedule_result: _schedule() 返回的结果
        
        注意：如果子类不需要队列，这个方法可以只做 pass
        """
        pass
    
    def step(self, use_ste=False):
        """
        执行一个时隙的仿真
        
        步骤：
        1. 调度决策
        2. 更新队列（如果有）
        3. 记录历史
        
        Args:
            use_ste: 是否使用 STE
        
        Returns:
            dict: 本时隙的仿真结果
        """
        # 1. 调度
        schedule_result = self._schedule(self.current_state, use_ste=use_ste)
        
        # 2. 更新队列
        self._update_queue(schedule_result)
        
        # 3. 记录
        self.slot += 1
        
        return schedule_result
    
    @abstractmethod
    def run(self, num_slots, **kwargs):
        """
        子类实现：运行多时隙仿真
        
        Args:
            num_slots: 仿真时隙数
            **kwargs: 其他参数
        
        Returns:
            dict: 仿真指标
        """
        pass
    
    @abstractmethod
    def compute_metrics(self):
        """
        子类实现：计算性能指标
        
        Returns:
            dict: 性能指标
        """
        pass
    
    def get_history(self, key=None):
        """获取历史记录"""
        if key is None:
            return dict(self.history)
        return self.history.get(key, [])
    
    def reset_history(self):
        """清空历史记录"""
        self.history.clear()


class SimpleSimulator(BaseSimulator):
    """
    简单仿真器：单时隙效用评估
    
    场景说明：
    - 用于评估单个网络状态下的调度性能
    - 信道：只生成一次，仿真结束后丢弃
    - 队列：不涉及
    
    使用场景：
    - 训练时的快速验证
    - 单时隙吞吐最大化实验
    """
    
    def _init(self):
        """初始化简单仿真器"""
        self.current_metrics = None
    
    def _generate_channel(self, csi_mode="perfect", q_bits=8):
        """生成信道状态（每次调用生成新的信道）"""
        self.current_state = self.net_gen.generate_network(
            csi_mode=csi_mode,
            q_bits=q_bits,
        )
        return self.current_state
    
    def _schedule(self, state, use_ste=False):
        """执行调度决策"""
        self.model.eval()
        with torch.no_grad():
            x = state.vertex_features.to(self.device)
            edge_index = state.edge_index.to(self.device)
            edge_weights = state.edge_weights.to(self.device)
            
            result = self.model(x, edge_index, edge_weights, use_ste=use_ste)
            
            if hasattr(self.model, 'use_power_control') and self.model.use_power_control:
                pi, alpha, _ = result
            else:
                pi, _ = result
                alpha = None
            
            from models.model import calculate_achievable_rate
            rates = calculate_achievable_rate(pi, state, binary=True, detach_hard=True, alpha=alpha)
            
            active_mask = (pi.squeeze(-1) > 0.5).float()
            active_indices = (active_mask > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
            
            return {
                'pi': pi,
                'alpha': alpha,
                'rates': rates,
                'active_mask': active_mask,
                'active_indices': active_indices,
                'utility': rates.sum().item(),
            }
    
    def _update_queue(self, schedule_result):
        """简单仿真器不涉及队列"""
        self.current_metrics = schedule_result
    
    def run(self, num_slots=1, csi_mode="perfect", q_bits=8, 
            verbose=False, **kwargs):
        """
        运行多时隙仿真
        
        注意：这个仿真器的设计初衷是单时隙评估。
        如果 num_slots > 1，会在每个时隙生成新的信道（用于蒙特卡洛评估）。
        如果你想要"信道保持不变"的多时隙仿真，请使用 QueueSimulator。
        
        Args:
            num_slots: 仿真时隙数
            csi_mode: 信道状态信息模式
            q_bits: 量化比特数
            verbose: 是否打印进度
        
        Returns:
            dict: 仿真指标
        """
        utilities = []
        
        for t in range(num_slots):
            # 生成新的信道（每次都不同，用于蒙特卡洛评估）
            self._generate_channel(csi_mode=csi_mode, q_bits=q_bits)
            
            # 执行调度
            result = self.step()
            utilities.append(result['utility'])
            
            if verbose and (t + 1) % 100 == 0:
                print(f"Slot {t+1}/{num_slots}: "
                      f"utility={result['utility']:.2f}, "
                      f"active={len(result['active_indices'])}")
        
        return {
            'utilities': utilities,
            'avg_utility': np.mean(utilities),
            'std_utility': np.std(utilities),
        }
    
    def compute_metrics(self):
        """计算性能指标"""
        return self.current_metrics or {}
