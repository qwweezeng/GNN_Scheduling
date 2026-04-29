"""
Queue-aware Network Simulator - 队列感知仿真器

继承自 BaseSimulator，实现带队列感知的无线链路调度仿真。

信道模型说明：
- 块衰落（Block Fading）：信道在一个传输块内保持不变
- 默认行为：整个仿真周期使用同一个信道（slots_per_block = num_slots）
- 可配置：每 N 个时隙更新一次信道（用于长记忆仿真）

队列更新公式（标准形式）：
    1. 计算发送量：D_i(t) = π_i(t) × R_i(t) × τ / L
       其中：
       - π_i(t) ∈ {0,1}：链路激活决策
       - R_i(t)：可达速率，单位 bps/Hz
       - τ：时隙长度（秒）
       - L：包大小（bits/packet）
    
    2. 队列更新：Q_i(t+1) = max(0, Q_i(t) - D_i(t)) + A_i(t)
       其中：
       - Q_i(t)：当前队列长度（packets）
       - A_i(t)：新到达的包数（从 Poisson(λ) 采样）

单位说明：
- arrival_lambda：泊松到达过程的均值，单位是 **packets/slot**
- rate：从 calculate_achievable_rate() 返回，单位是 **bps/Hz**
- departure：发送量，单位是 **packets/slot**
- queue_states：队列长度，单位是 **packets**

注意：如果 arrival_lambda >> departure_rate，队列会持续增长（过载）。
这是合理的系统行为，需要选择合适的到达率来测试。
"""

import torch
import numpy as np
from .base import BaseSimulator


class QueueSimulator(BaseSimulator):
    """
    多时隙队列感知仿真器
    
    继承自 BaseSimulator，实现队列感知调度仿真。
    
    示例：
        # 默认：5000 时隙仿真，信道全程不变
        sim.run(num_slots=5000)
        
        # 每 100 时隙更新一次信道
        sim.run(num_slots=5000, slots_per_block=100)
    """
    
    def _init(self):
        """初始化队列仿真器"""
        self.queue_states = None
        
        # 到达过程参数
        self.arrival_lambda = 0.5  # 到达率（packets/slot）
        
        # 时隙和包参数（用于单位转换）
        self.slot_duration = 0.001  # 时隙长度 τ = 1 ms
        self.packet_size = 12000    # 包大小 L = 1500 bytes = 12000 bits
        
        # 信道块参数
        self.slots_per_block = None  # None 表示全程不变
    
    def configure(self, arrival_lambda=0.5, 
                  slot_duration=0.001, packet_size=12000,
                  slots_per_block=None):
        """
        配置仿真器参数
        
        Args:
            arrival_lambda: 泊松到达过程的均值（packets/slot）
            slot_duration: 时隙长度 τ（秒）。例如 τ=0.001 表示 1ms 时隙
            packet_size: 包大小（bits）。例如 12000 表示 1500 bytes
            slots_per_block: 每个信道块的时隙数。
                           - None 或 0：整个仿真周期信道不变（默认）
                           - 正整数：每 N 个时隙更新一次信道
        """
        self.arrival_lambda = arrival_lambda
        self.slot_duration = slot_duration
        self.packet_size = packet_size
        self.slots_per_block = slots_per_block if slots_per_block and slots_per_block > 0 else None
    
    def _generate_channel(self, csi_mode="perfect", q_bits=8, queue_states=None):
        """
        生成信道状态（包含队列信息）
        
        Args:
            csi_mode: 信道状态信息模式
            q_bits: 量化比特数
            queue_states: 当前队列状态，如果为 None 则不传入
        
        Returns:
            state: NetworkState 对象
        """
        self.current_state = self.net_gen.generate_network(
            csi_mode=csi_mode,
            q_bits=q_bits,
            queue_states=queue_states
        )
        return self.current_state
    
    def _schedule(self, state, use_ste=False):
        """
        执行调度决策
        
        Args:
            state: 当前 NetworkState
            use_ste: 是否使用 Straight-Through Estimator
        
        Returns:
            dict: 包含调度结果和队列更新的信息
        """
        device = self.device
        model = self.model
        model.eval()
        
        with torch.no_grad():
            x = state.vertex_features.to(device)
            edge_index = state.edge_index.to(device)
            edge_weights = state.edge_weights.to(device)
            
            result = model(x, edge_index, edge_weights, use_ste=use_ste)
            
            if hasattr(model, 'use_power_control') and model.use_power_control:
                pi, alpha, _ = result
                from models.model import calculate_achievable_rate
                rates = calculate_achievable_rate(pi, state, binary=True, alpha=alpha)
            else:
                pi, _ = result
                alpha = None
                from models.model import calculate_achievable_rate
                rates = calculate_achievable_rate(pi, state, binary=True)
            
            active_mask = (pi.squeeze(-1) > 0.5).float()
            active_indices = (active_mask > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
            
            return {
                'pi': pi,
                'alpha': alpha,
                'rates': rates,
                'active_mask': active_mask,
                'active_indices': active_indices,
            }
    
    def _generate_arrivals(self):
        """
        生成业务到达（泊松过程）
        
        Returns:
            arrivals: 每条链路的到达包数（packets），从 Poisson(λ) 采样
        """
        num_links = self.current_state.num_vertices  # 链路数 = UAV 数
        arrivals = torch.poisson(
            torch.ones(num_links) * self.arrival_lambda
        ).float()
        return arrivals
    
    def _update_queue(self, schedule_result):
        """
        更新队列状态
        
        队列更新公式：
            D_i(t) = π_i(t) × R_i(t) × B × τ / L  (发送量，packets)
            Q_i(t+1) = max(0, Q_i(t) - D_i(t)) + A_i(t)  (队列更新)
        
        其中：
            - R_i(t)：可达速率（bps/Hz）
            - B：带宽（Hz）
            - τ：时隙长度（秒）
            - L：包大小（bits/packet）
        """
        device = self.device
        state = self.current_state
        
        # 获取参数
        link_rates = schedule_result['rates'].squeeze(-1)  # [num_links], bps/Hz
        active_mask = schedule_result['active_mask'].squeeze(-1)  # [num_links], {0,1}
        
        # 获取带宽
        bandwidth = getattr(state, 'bandwidth', 5e6)  # 默认 5 MHz
        
        # 计算每条链路的发送量（packets/slot）
        # D_i = π_i × R_i × B × τ / L
        # - π_i ∈ {0,1}：激活决策
        # - R_i：bps/Hz
        # - B：Hz
        # - τ：秒
        # - L：bits/packet
        # 结果单位：packets/slot
        departures_per_link = active_mask * link_rates * bandwidth * self.slot_duration / self.packet_size
        
        # 每条链路对应一个发射 UAV，累加到 UAV 的总发送量
        links = state.links  # [num_links, 2], (tx_idx, rx_idx)
        num_uavs = self.net_gen.num_uavs
        departures = torch.zeros(num_uavs, device=device)  # [num_uavs]
        
        for link_idx in range(links.size(0)):
            tx_idx = links[link_idx, 0].item()
            departures[tx_idx] += departures_per_link[link_idx].item()
        
        # 生成到达（每条链路独立的泊松到达）
        arrivals = self._generate_arrivals()  # [num_links]
        
        # 队列更新：Q_i(t+1) = max(0, Q_i(t) - D_i(t)) + A_i(t)
        # 注意：先发送，后到达（标准顺序）
        self.queue_states = torch.clamp(
            self.queue_states - departures,
            min=0.0
        ) + arrivals.to(device)
        
        # 计算加权效用（基于当前队列和速率）
        # utility = Σ Q_i × R_i（加权holtz-werk）
        utility = torch.sum(self.queue_states.cpu() * link_rates.cpu()).item()
        
        # 记录历史
        self.history['queue_lengths'].append(self.queue_states.cpu().clone())
        self.history['arrivals'].append(arrivals.clone())
        self.history['departures'].append(departures.cpu().clone())
        self.history['departures_per_link'].append(departures_per_link.detach().cpu().clone())
        self.history['active_links'].append(schedule_result['active_indices'])
        self.history['rates'].append(link_rates.detach().cpu().clone())
        self.history['utilities'].append(utility)
        
        # 更新 NetworkState 中的队列（用于下一时隙的特征输入）
        self.current_state.queue_states = self.queue_states
    
    def run(self, num_slots, csi_mode="perfect", q_bits=8,
            verbose=True, log_interval=100, **kwargs):
        """
        运行多时隙队列仿真
        
        信道更新规则：
        - 如果 slots_per_block 为 None：整个仿真周期使用同一个信道
        - 如果 slots_per_block > 0：每 N 个时隙更新一次信道
        
        Args:
            num_slots: 仿真时隙数
            csi_mode: 信道状态信息模式
            q_bits: 量化比特数
            verbose: 是否打印进度
            log_interval: 打印间隔
        
        Returns:
            metrics: dict，包含仿真结果指标
        """
        # 重置队列（从空队列开始）
        self.queue_states = torch.zeros(self.net_gen.num_uavs)
        
        # 生成初始信道状态
        self._generate_channel(csi_mode=csi_mode, q_bits=q_bits, queue_states=self.queue_states)
        
        # 确定信道更新频率
        slots_per_block = self.slots_per_block if self.slots_per_block else num_slots
        
        if verbose:
            print(f"[QueueSimulator] Configuration:")
            print(f"  - Arrival rate λ: {self.arrival_lambda} packets/slot")
            print(f"  - Slot duration τ: {self.slot_duration * 1000:.1f} ms")
            print(f"  - Packet size: {self.packet_size} bits ({self.packet_size / 8} bytes)")
            print(f"  - Bandwidth: {getattr(self.current_state, 'bandwidth', 5e6) / 1e6:.1f} MHz")
            if slots_per_block >= num_slots:
                print(f"[QueueSimulator] Block fading: channel fixed ({num_slots} slots)")
            else:
                print(f"[QueueSimulator] Block fading: channel updates every {slots_per_block} slots")
        
        # 记录每个块的信道变化次数
        self.history['channel_updates'] = []
        self.history['slots_per_block'] = slots_per_block
        
        for t in range(num_slots):
            # 检查是否需要更新信道（块衰落边界）
            if t > 0 and t % slots_per_block == 0:
                self._generate_channel(csi_mode=csi_mode, q_bits=q_bits, queue_states=self.queue_states)
                self.history['channel_updates'].append(t)
                if verbose:
                    print(f"  [t={t}] Channel updated (block boundary)")
            
            # 执行一个时隙
            result = self.step()
            
            if verbose and (t + 1) % log_interval == 0:
                avg_queue = self.history['queue_lengths'][-1].mean().item()
                max_queue = self.history['queue_lengths'][-1].max().item()
                utility = self.history['utilities'][-1]
                active_count = len(self.history['active_links'][-1])
                print(f"  Slot {t+1}/{num_slots}: "
                      f"avg_queue={avg_queue:.2f}, max_queue={max_queue:.2f}, "
                      f"utility={utility:.4f}, active={active_count}")
        
        # 计算仿真指标
        metrics = self.compute_metrics()
        
        if verbose:
            print(f"\n=== Simulation Complete ({num_slots} slots) ===")
            print(f"Channel updates: {len(self.history['channel_updates'])} times")
            print(f"Final avg queue: {metrics['avg_queue']:.2f}")
            print(f"Final max queue: {metrics['max_queue']:.2f}")
            print(f"Total arrivals: {metrics['total_arrivals']:.1f}")
            print(f"Total departures: {metrics['total_departures']:.1f}")
            print(f"Throughput: {metrics['throughput']:.4f}")
            print(f"Queue overflow prob: {metrics['queue_overflow_prob']:.4f}")
        
        return metrics
    
    def compute_metrics(self):
        """
        计算仿真性能指标
        
        Returns:
            metrics: dict，包含各类指标
        """
        queue_lengths = torch.stack(self.history['queue_lengths'])  # [num_slots, num_uavs]
        arrivals = torch.stack(self.history['arrivals'])
        departures = torch.stack(self.history['departures'])
        
        num_slots = queue_lengths.size(0)
        
        # 基本统计
        avg_queue_per_slot = queue_lengths.mean(dim=1)
        max_queue_per_slot = queue_lengths.max(dim=1)[0]
        
        metrics = {
            # 队列长度统计
            'avg_queue': avg_queue_per_slot[-1].item(),
            'max_queue': max_queue_per_slot[-1].item(),
            'long_term_avg_queue': queue_lengths.mean().item(),
            'long_term_max_queue': queue_lengths.max().item(),
            
            # 吞吐量
            'total_arrivals': arrivals.sum().item(),
            'total_departures': departures.sum().item(),
            'throughput': (departures.sum() / arrivals.sum()).item() if arrivals.sum() > 0 else 0,
            
            # 队列轨迹（用于绘图）
            'queue_trajectory': queue_lengths.cpu().numpy(),
            'avg_queue_trajectory': avg_queue_per_slot.cpu().numpy(),
            'max_queue_trajectory': max_queue_per_slot.cpu().numpy(),
            
            # 效用轨迹
            'utility_trajectory': np.array(self.history['utilities']),
            
            # 队列溢出概率
            'queue_overflow_prob': (queue_lengths[-1] > 50).float().mean().item(),
            
            # 稳定性判断
            'is_stable': self._check_stability(queue_lengths),
        }
        
        return metrics
    
    def _check_stability(self, queue_lengths, window=100):
        """
        判断队列是否稳定：检查最近 window 个时隙的平均队列是否增长
        """
        if queue_lengths.size(0) < window:
            return None
        
        recent = queue_lengths[-window:]
        mean_per_slot = recent.mean(dim=1)
        
        slope = (mean_per_slot[-1].item() - mean_per_slot[0].item()) / window
        is_stable = abs(slope) < 0.5
        
        return is_stable


def compare_scheduling_algorithms(net_gen, model, num_slots=5000,
                                   arrival_rates=None, device='cpu',
                                   slots_per_block=None, **sim_kwargs):
    """
    比较不同到达率下各调度算法的性能
    
    Args:
        net_gen: NetworkGenerator 实例
        model: GAT 模型
        num_slots: 每个到达率下的仿真时隙数
        arrival_rates: 到达率列表（packets/slot）
        device: 计算设备
        slots_per_block: 每个信道块的时隙数
        **sim_kwargs: 传递给 simulator.configure() 的其他参数
    
    Returns:
        results: dict，key=到达率，value=metrics
    """
    if arrival_rates is None:
        # 覆盖 capacity region 的不同负载点
        arrival_rates = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    results = {}
    
    for lam in arrival_rates:
        print(f"\n=== Testing arrival rate λ = {lam} ===")
        
        simulator = QueueSimulator(
            network_generator=net_gen,
            model=model,
            device=device,
        )
        simulator.configure(arrival_lambda=lam, slots_per_block=slots_per_block, **sim_kwargs)
        
        metrics = simulator.run(
            num_slots=num_slots,
            csi_mode="perfect",
            verbose=True,
            log_interval=500
        )
        
        results[lam] = metrics
    
    return results


if __name__ == '__main__':
    # 简单测试
    from utils.network_generate import NetworkGenerator
    
    net_gen = NetworkGenerator(
        num_uavs=10,
        area_width=1000,
        height_min=50,
        height_max=150,
        uav_min_distance=20,
        tx_power=22,
    )
    
    print("QueueSimulator initialized successfully!")
    print(f"Network: {net_gen.num_uavs} UAVs")
    
    # 测试单位转换
    print("\n=== Unit Conversion Test ===")
    print(f"假设：rate = 1 bps/Hz, B = 5 MHz, τ = 1 ms, L = 12000 bits")
    rate = 1.0  # bps/Hz
    B = 5e6     # Hz
    tau = 0.001 # s
    L = 12000   # bits
    packets_per_slot = rate * B * tau / L
    print(f"每时隙发送量 = {rate} × {B/1e6}×10^6 × {tau*1000} × 10^-3 / {L}")
    print(f"         = {rate * B * tau / L:.4f} packets/slot")
