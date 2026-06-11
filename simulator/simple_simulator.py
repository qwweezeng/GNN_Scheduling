"""
Simple Simulator - 简单单时隙仿真器

用于基础场景的调度性能评估。不涉及队列动态，只是评估当前网络状态下的调度性能。

使用场景：
- 训练时的快速验证
- 单时隙吞吐最大化

信道模型：准静态衰落（块衰落）
- 信道在仿真期间保持恒定
- 每次运行仿真时重新生成信道
"""

import torch
import numpy as np


class SimpleSimulator:
    """
    简单仿真器：单时隙效用评估
    
    信道模型：准静态衰落
    - generate_network() 在 reset() 时调用一次
    - 整个仿真期间信道保持不变
    """
    
    def __init__(self, network_generator, model, device='cpu', seed=None,
                 include_queue_feature=False):
        """
        Args:
            network_generator: NetworkGenerator 实例
            model: 调度模型
            device: 计算设备
            seed: 随机种子
            include_queue_feature: 是否在节点特征中包含队列长度
                - True:  [SNR_dB, queue_normalized] (2维)
                - False: [SNR_dB]                   (1维)
                应与模型训练时的 NODE_RAW_DIM 保持一致。
        """
        self.net_gen = network_generator
        self.model = model
        self.device = device
        self.include_queue_feature = include_queue_feature
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # 状态
        self.current_state = None
        self.current_metrics = None
    
    def reset(self, csi_mode="perfect", q_bits=8):
        """
        重置仿真器，生成新的网络状态（准静态信道）
        
        Args:
            csi_mode: CSI 模式
            q_bits: 量化比特数
        
        Returns:
            初始网络状态
        """
        self.current_state = self.net_gen.generate_network(
            csi_mode=csi_mode,
            q_bits=q_bits,
            include_queue_feature=self.include_queue_feature,
        )
        return self.current_state
    
    def step(self, use_ste=False):
        """
        执行一次调度评估（单时隙）
        
        Args:
            use_ste: 是否使用 STE
        
        Returns:
            dict: 调度结果
        """
        model = self.model
        state = self.current_state
        device = self.device
        
        model.eval()
        with torch.no_grad():
            x = state.vertex_features.to(device)
            edge_index = state.edge_index.to(device)
            edge_weights = state.edge_weights.to(device)
            
            result = model(x, edge_index, edge_weights, use_ste=use_ste)
            
            # 根据是否有功率控制选择返回格式
            if hasattr(model, 'use_power_control') and model.use_power_control:
                pi, alpha, _ = result
            else:
                pi, _ = result
                alpha = None
            
            # 计算可达速率
            from models.model import calculate_achievable_rate
            rates = calculate_achievable_rate(pi, state, binary=True, detach_hard=True, alpha=alpha)
            
            # 激活链路
            active_mask = (pi.squeeze(-1) > 0.5).float()
            active_indices = (active_mask > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
            
            self.current_metrics = {
                'pi': pi,
                'alpha': alpha,
                'rates': rates,
                'active_mask': active_mask,
                'active_indices': active_indices,
                'utility': rates.sum().item(),
                'num_active': len(active_indices),
            }
            
            return self.current_metrics
    
    def run(self, num_slots=1, csi_mode="perfect", q_bits=8, verbose=False):
        """
        运行多次评估（使用同一个信道状态）
        
        Args:
            num_slots: 评估次数（同一信道状态）
            csi_mode: CSI 模式（只在 reset 时使用）
            q_bits: 量化比特数（只在 reset 时使用）
            verbose: 是否打印
        
        Returns:
            dict: 仿真结果
        """
        # 重置生成信道（准静态：只在开始时生成一次）
        self.reset(csi_mode=csi_mode, q_bits=q_bits)
        
        utilities = []
        for t in range(num_slots):
            result = self.step()
            utilities.append(result['utility'])
            
            if verbose and (t + 1) % 100 == 0:
                print(f"Slot {t+1}/{num_slots}: utility={result['utility']:.2f}, "
                      f"active={result['num_active']}")
        
        return {
            'utility': np.mean(utilities),
            'utility_std': np.std(utilities),
            'utilities': utilities,
            'num_active': self.current_metrics['num_active'] if self.current_metrics else 0,
        }
    
    def evaluate(self):
        """执行一次评估并返回结果"""
        return self.step()


if __name__ == '__main__':
    from utils.network_generate import NetworkGenerator
    
    net_gen = NetworkGenerator(
        num_uavs=10,
        area_width=500,
        height_min=50,
        height_max=100,
        tx_power=22,
    )
    
    print("SimpleSimulator initialized successfully!")
    print(f"Network: {net_gen.num_uavs} UAVs")
