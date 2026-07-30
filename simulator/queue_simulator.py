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
from collections import deque
from .base import BaseSimulator
from utils.queue_features import relative_queue_feature


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

        # 外部预生成的到达过程；用于同一 Monte Carlo seed 内公平比较不同调度器
        self._arrival_realizations = None

        # Optional FIFO packet accounting for actual delivered-packet delay.
        # The fluid queue remains the source of truth for scheduling/service;
        # each integer Poisson arrival is additionally tracked as one packet
        # whose remaining service can be fractional across slots.
        self.track_packet_delay = False
        self.delay_deadline_slots = None
        self.buffer_capacity_packets = None
        self._packet_cohorts = None
        self._delivered_packet_delays = []
        self._delivered_packet_arrival_slots = []
        self._pending_deadline_drop_counts = None
        self._pending_deadline_drop_work = None

    def _sync_queue_features(self):
        """
        将仿真器中的真实动态队列同步到当前 NetworkState。

        Queue-aware GAT 的输入来自 state.vertex_features，因此只更新
        state.queue_states 还不够；第二维队列特征也必须在每次调度前
        反映当前时隙的 Q(t)。
        """
        if self.current_state is None or self.queue_states is None:
            return

        state = self.current_state
        q = self.queue_states.detach().float()
        if q.numel() != state.num_vertices:
            raise ValueError(
                "Queue length mismatch: "
                f"queue_states has {q.numel()} entries, "
                f"but current state has {state.num_vertices} links."
            )

        state.queue_states = q.to(state.vertex_features.device)

        if not self.include_queue_feature:
            return

        queue_feature = relative_queue_feature(q)
        queue_feature = queue_feature.to(state.vertex_features.device).unsqueeze(1)

        if state.vertex_features.size(1) == 1:
            state.vertex_features = torch.cat([state.vertex_features, queue_feature], dim=1)
        else:
            state.vertex_features[:, 1] = queue_feature.squeeze(1)
    
    def configure(self, arrival_lambda=0.5, 
                  slot_duration=0.001, packet_size=12000,
                  slots_per_block=None, track_packet_delay=False,
                  delay_deadline_ms=None, buffer_size_mbit=None):
        """
        配置仿真器参数
        
        Args:
            arrival_lambda: 泊松到达过程的均值（packets/slot）
            slot_duration: 时隙长度 τ（秒）。例如 τ=0.001 表示 1ms 时隙
            packet_size: 包大小（bits）。例如 12000 表示 1500 bytes
            slots_per_block: 每个信道块的时隙数。
                           - None 或 0：整个仿真周期信道不变（默认）
                           - 正整数：每 N 个时隙更新一次信道
            track_packet_delay: 是否额外进行 FIFO 包级完成时延记账。
            delay_deadline_ms: 包从到达到完成所允许的最大时延（ms）。
                超过该时限仍未完成的包会被丢弃；None 表示不设截止时间。
            buffer_size_mbit: 每条链路发射队列的有限缓存大小（Mbit）。
                新到达包超过容量时立即丢弃；None 表示无限缓存。
        """
        self.arrival_lambda = arrival_lambda
        self.slot_duration = slot_duration
        self.packet_size = packet_size
        self.slots_per_block = slots_per_block if slots_per_block and slots_per_block > 0 else None
        self.track_packet_delay = bool(
            track_packet_delay or delay_deadline_ms is not None
        )
        if delay_deadline_ms is None:
            self.delay_deadline_slots = None
        else:
            self.delay_deadline_slots = max(
                0,
                int(round(float(delay_deadline_ms) / (slot_duration * 1000.0))),
            )
        self.buffer_capacity_packets = (
            None if buffer_size_mbit is None else
            float(buffer_size_mbit) * 1e6 / float(packet_size)
        )

    def _expire_packet_deadlines(self):
        """Discard packets whose end-to-end queueing delay exceeds the deadline.

        Expiration is evaluated before scheduling the current slot, so an
        expired packet is neither included in Q(t) nor served in that slot.
        A packet with age exactly equal to the deadline is still eligible for
        service; it expires at the next slot if unfinished.
        """
        if self._packet_cohorts is None:
            return
        num_links = len(self._packet_cohorts)
        counts = torch.zeros(num_links, dtype=torch.float32, device=self.device)
        work = torch.zeros_like(counts)
        if self.delay_deadline_slots is not None:
            for link_idx, queue in enumerate(self._packet_cohorts):
                while queue:
                    arrival_slot, remaining = queue[0]
                    if int(self.slot) - int(arrival_slot) <= self.delay_deadline_slots:
                        break
                    queue.popleft()
                    counts[link_idx] += 1.0
                    work[link_idx] += float(remaining)
            if self.queue_states is not None:
                self.queue_states = torch.clamp(
                    self.queue_states - work, min=0.0
                )
                self._sync_queue_features()
        self._pending_deadline_drop_counts = counts
        self._pending_deadline_drop_work = work

    def _update_packet_delay_tracking(self, departures, arrivals):
        """Consume FIFO packet work and append new integer packet arrivals."""
        if not self.track_packet_delay:
            return

        eps = 1e-8
        departures = departures.detach().cpu().reshape(-1)
        arrivals = arrivals.detach().cpu().reshape(-1)
        for link_idx in (departures > eps).nonzero(as_tuple=True)[0].tolist():
            service = float(departures[link_idx])
            queue = self._packet_cohorts[link_idx]
            while service > eps and queue:
                packet = queue[0]
                consumed = min(service, packet[1])
                packet[1] -= consumed
                service -= consumed
                if packet[1] <= eps:
                    queue.popleft()
                    self._delivered_packet_delays.append(
                        max(0, int(self.slot) - int(packet[0]))
                    )
                    if not hasattr(self, "_delivered_packet_arrival_slots"):
                        self._delivered_packet_arrival_slots = []
                    self._delivered_packet_arrival_slots.append(int(packet[0]))
                else:
                    break

        for link_idx in (arrivals > 0).nonzero(as_tuple=True)[0].tolist():
            count = int(round(float(arrivals[link_idx])))
            for _ in range(count):
                self._packet_cohorts[link_idx].append(
                    [int(self.slot), 1.0]
                )
    
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
            queue_states=queue_states,  # [num_links] link-level, passed from outside
            include_queue_feature=self.include_queue_feature,
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
        self._sync_queue_features()
        try:
            model_device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            model_device = device

        with torch.no_grad():
            x = state.vertex_features.to(model_device)
            edge_index = state.edge_index.to(model_device)
            edge_weights = state.edge_weights.to(model_device)

            if hasattr(model, 'schedule_with_state'):
                pi = model.schedule_with_state(state, self.queue_states).to(device)
                from utils.feasibility import project_shared_uav_feasible
                pi, hard_feasibility = project_shared_uav_feasible(
                    pi.detach().cpu(), state.links,
                    priorities=state.vertex_features[:, 0].detach().cpu(),
                )
                pi = pi.to(device)
                alpha = None
                from models.model import calculate_achievable_rate
                rates = calculate_achievable_rate(pi, state, binary=True)
                active_mask = (pi.squeeze(-1) > 0.5).float()
                active_indices = (active_mask > 0.5).nonzero(
                    as_tuple=True
                )[0].cpu().tolist()
                return {
                    'pi': pi,
                    'alpha': alpha,
                    'rates': rates,
                    'active_mask': active_mask,
                    'active_indices': active_indices,
                    'hard_feasibility': hard_feasibility,
                }

            # 非队列感知模型（GAT-base）只吃 SNR 特征，slice 掉队列维度
            # 队列感知模型（GAT）直接用 [SNR, queue_norm] 完整特征
            if not self.include_queue_feature and x.size(1) > 1:
                x = x[:, :1]

            result = model(x, edge_index, edge_weights, use_ste=use_ste)
            
            if hasattr(model, 'use_power_control') and model.use_power_control:
                pi, alpha, _ = result
            else:
                pi, _ = result
                alpha = None

            if self.include_queue_feature:
                nonempty = (self.queue_states > 0).to(pi.device).unsqueeze(1)
                pi = pi * nonempty.to(pi.dtype)
            threshold = float(getattr(model, 'decision_threshold', 0.5))
            pi = (pi > threshold).to(pi.dtype)
            from utils.feasibility import project_shared_uav_feasible
            pi, hard_feasibility = project_shared_uav_feasible(
                pi.detach().cpu(), state.links,
                priorities=state.vertex_features[:, 0].detach().cpu(),
            )
            pi = pi.to(device)
            from models.model import calculate_achievable_rate
            rates = calculate_achievable_rate(
                pi, state, binary=False, alpha=alpha
            )

            active_mask = pi.squeeze(-1).float()
            active_indices = (active_mask > 0.5).nonzero(as_tuple=True)[0].cpu().tolist()
            
            return {
                'pi': pi,
                'alpha': alpha,
                'rates': rates,
                'active_mask': active_mask,
                'active_indices': active_indices,
                'hard_feasibility': hard_feasibility,
            }
    
    def _generate_arrivals(self):
        """
        生成业务到达（泊松过程）
        
        Returns:
            arrivals: 每条链路的到达包数（packets），从 Poisson(λ) 采样
        """
        num_links = self.current_state.num_vertices  # 链路数
        if self._arrival_realizations is not None:
            arrivals = self._arrival_realizations[self.slot]
            if arrivals.numel() != num_links:
                raise ValueError(
                    "Arrival length mismatch: "
                    f"arrival trace has {arrivals.numel()} entries, "
                    f"but current state has {num_links} links."
                )
            return arrivals.to(self.device).float()

        arrivals = torch.poisson(
            torch.ones(num_links, device=self.device) * self.arrival_lambda
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
        
        # 获取带宽
        bandwidth = getattr(state, 'bandwidth', 5e6)  # 默认 5 MHz
        
        # 计算 Shannon 容量（理想发送量，packets/slot）
        # 注意：统一在 self.device 上计算，避免 CPU/GPU 不匹配
        device = self.device
        link_rates = schedule_result['rates'].squeeze(-1).to(device)  # [num_links], on device
        active_mask = schedule_result['active_mask'].squeeze(-1).to(device)  # [num_links], on device
        departures_computed = active_mask * link_rates * bandwidth * self.slot_duration / self.packet_size

        # 生成业务请求：默认独立泊松采样；公平比较时来自共享 trace。
        # offered_arrivals 是外部提出的全部到达，admitted_arrivals 才进入有限缓存。
        offered_arrivals = self._generate_arrivals()  # [num_links]

        # MWS 权重使用调度决策时的 Q(t)，而不是更新后的 Q(t+1)。
        queue_before_service = self.queue_states.clone()

        # 实际发送量受限于当前队列长度（都在 self.device 上）
        departures_actual = torch.min(queue_before_service, departures_computed)

        queue_after_service = torch.clamp(
            self.queue_states - departures_actual, min=0.0
        )
        if self.buffer_capacity_packets is None:
            admitted_arrivals = offered_arrivals.to(device)
        else:
            available = torch.clamp(
                torch.floor(
                    self.buffer_capacity_packets - queue_after_service + 1e-8
                ),
                min=0.0,
            )
            admitted_arrivals = torch.minimum(
                offered_arrivals.to(device), available
            )
        overflow_drops = offered_arrivals.to(device) - admitted_arrivals

        # FIFO accounting sees exactly the same service and admitted arrivals
        # as the fluid queue. Arrivals occur after service and are first
        # eligible in the next slot.
        self._update_packet_delay_tracking(
            departures_actual, admitted_arrivals
        )

        # 队列更新
        self.queue_states = queue_after_service + admitted_arrivals

        # 加权效用（都在 CPU 上计算用于记录）
        R_eff = link_rates.cpu() * bandwidth * self.slot_duration / self.packet_size
        utility = torch.sum(queue_before_service.cpu() * R_eff).item()

        # 记录历史：用 departures_actual（实际发送量），而非 Shannon 容量
        self.history['queue_lengths'].append(self.queue_states.cpu().clone())
        self.history['arrivals'].append(offered_arrivals.clone())
        self.history['admitted_arrivals'].append(
            admitted_arrivals.detach().cpu().clone()
        )
        self.history['overflow_drop_counts'].append(
            overflow_drops.detach().cpu().clone()
        )
        deadline_counts = self._pending_deadline_drop_counts
        if deadline_counts is None:
            deadline_counts = torch.zeros_like(offered_arrivals, device=device)
        self.history['deadline_drop_counts'].append(
            deadline_counts.detach().cpu().clone()
        )
        self.history['departures'].append(departures_actual.detach().cpu().clone())
        self.history['active_links'].append(schedule_result['active_indices'])
        nonempty_mask = queue_before_service > 0
        self.history['nonempty_link_counts'].append(
            int(nonempty_mask.sum().item())
        )
        self.history['active_nonempty_counts'].append(
            int(((active_mask > 0) & nonempty_mask).sum().item())
        )
        self.history['active_empty_counts'].append(
            int(((active_mask > 0) & ~nonempty_mask).sum().item())
        )
        self.history['rates'].append(link_rates.detach().cpu().clone())
        self.history['utilities'].append(utility)
        
        # 更新 NetworkState 中的队列（用于下一时隙的特征输入）
        self._sync_queue_features()
    
    def run(self, num_slots, csi_mode="perfect", q_bits=8,
            verbose=True, log_interval=100,
            channel_realizations=None,
            arrival_realizations=None,
            **kwargs):
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
            channel_realizations: 预生成的信道状态列表（用于公平比较）
                                如果为 None，则每次独立生成（向后兼容）
            arrival_realizations: 预生成的到达过程，shape=[num_slots, num_links]。
                                  如果为 None，则每个调度器独立采样泊松到达（向后兼容）。
        
        Returns:
            metrics: dict，包含仿真结果指标
        """
        slots_per_block = self.slots_per_block if self.slots_per_block else num_slots
        num_blocks = (num_slots + slots_per_block - 1) // slots_per_block

        # 同一个 simulator 对象重复 run 时清空历史和时隙计数。
        self.history.clear()
        self.slot = 0
        self._arrival_realizations = None
        
        # 初始化队列为空（链路数未知）
        self.queue_states = None
        
        # 预生成信道：为所有调度器共享同一套信道实现（公平比较）
        if channel_realizations is not None and len(channel_realizations) > 0:
            import copy
            channel_realizations = [copy.deepcopy(cr) for cr in channel_realizations]
        
        if channel_realizations is not None and len(channel_realizations) > 0:
            self.current_state = channel_realizations[0]
        else:
            self._generate_channel(csi_mode=csi_mode, q_bits=q_bits, queue_states=None)
        
        # 初始化队列（全零）
        self.queue_states = torch.zeros(self.current_state.num_vertices, device=self.device)
        self._packet_cohorts = [
            deque() for _ in range(self.current_state.num_vertices)
        ]
        self._delivered_packet_delays = []
        self._delivered_packet_arrival_slots = []
        self._pending_deadline_drop_counts = None
        self._pending_deadline_drop_work = None
        self._sync_queue_features()

        if arrival_realizations is not None:
            if isinstance(arrival_realizations, torch.Tensor):
                arrivals_tensor = arrival_realizations.detach().clone().float()
            else:
                arrivals_tensor = torch.as_tensor(np.asarray(arrival_realizations), dtype=torch.float32)

            expected_shape = (num_slots, self.current_state.num_vertices)
            if tuple(arrivals_tensor.shape) != expected_shape:
                raise ValueError(
                    "arrival_realizations shape mismatch: "
                    f"expected {expected_shape}, got {tuple(arrivals_tensor.shape)}."
                )
            self._arrival_realizations = arrivals_tensor.to(self.device)
        
        if verbose:
            print(f"[QueueSimulator] Configuration:")
            print(f"  - Arrival rate λ: {self.arrival_lambda} packets/slot")
            print(f"  - Slot duration τ: {self.slot_duration * 1000:.1f} ms")
            print(f"  - Packet size: {self.packet_size} bits ({self.packet_size / 8} bytes)")
            print(f"  - Bandwidth: {getattr(self.current_state, 'bandwidth', 5e6) / 1e6:.1f} MHz")
            if channel_realizations is not None:
                print(f"[QueueSimulator] Fair comparison: {num_blocks} pre-generated channel blocks shared across all schedulers")
            if self._arrival_realizations is not None:
                print(f"[QueueSimulator] Fair comparison: pre-generated arrival trace shared across all schedulers")
            elif slots_per_block >= num_slots:
                print(f"[QueueSimulator] Block fading: channel fixed ({num_slots} slots)")
            else:
                print(f"[QueueSimulator] Block fading: channel updates every {slots_per_block} slots")
        
        # 记录每个块的信道变化次数
        self.history['channel_updates'] = []
        self.history['slots_per_block'] = slots_per_block
        
        for t in range(num_slots):
            # 检查是否需要更新信道（块衰落边界）
            if t > 0 and t % slots_per_block == 0:
                block_idx = t // slots_per_block
                if channel_realizations is not None and block_idx < len(channel_realizations):
                    # 使用预生成信道
                    self.current_state = channel_realizations[block_idx]
                else:
                    # 独立生成信道（向后兼容）
                    self._generate_channel(csi_mode=csi_mode, q_bits=q_bits,
                                          queue_states=self.queue_states)
                self.history['channel_updates'].append(t)
                if verbose:
                    print(f"  [t={t}] Channel updated (block boundary)")
            
            # 在本时隙调度前移除已经超过 QoS 截止时间的包。
            self._expire_packet_deadlines()

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
            print(f"Throughput (ratio): {metrics['throughput_ratio']:.4f}")
            print(f"Queue overflow prob: {metrics['queue_overflow_prob']:.4f}")
        
        return metrics
    
    def compute_metrics(self):
        """
        计算仿真性能指标
        
        Returns:
            metrics: dict，包含各类指标
        """
        queue_lengths = torch.stack(self.history['queue_lengths'])  # [num_slots, num_links]
        arrivals = torch.stack(self.history['arrivals'])
        admitted_arrivals = torch.stack(self.history['admitted_arrivals'])
        departures = torch.stack(self.history['departures'])
        deadline_drop_counts = torch.stack(
            self.history['deadline_drop_counts']
        )
        overflow_drop_counts = torch.stack(
            self.history['overflow_drop_counts']
        )
        
        num_slots = queue_lengths.size(0)
        
        # 基本统计
        avg_queue_per_slot = queue_lengths.mean(dim=1)
        max_queue_per_slot = queue_lengths.max(dim=1)[0]
        total_queue_per_slot = queue_lengths.sum(dim=1)
        avg_queue_backlog = queue_lengths.mean().item()
        avg_total_queue_backlog = total_queue_per_slot.mean().item()
        avg_departure_rate_pps = departures.mean().item()
        avg_arrival_rate_pps = arrivals.mean().item()
        avg_total_departure_rate_pps = departures.sum(dim=1).mean().item()
        avg_total_arrival_rate_pps = arrivals.sum(dim=1).mean().item()

        # Little's law delay estimate:
        #   W = L / lambda_eff
        # Here L is the time-average per-link backlog, and lambda_eff is the
        # per-link delivered packet rate. This approximates delay of delivered
        # traffic without adding packet-level FIFO bookkeeping.
        if avg_queue_backlog <= 0:
            average_delay_slots = 0.0
        elif avg_departure_rate_pps <= 0:
            average_delay_slots = None
        else:
            average_delay_slots = avg_queue_backlog / avg_departure_rate_pps
        average_delay_ms = (
            average_delay_slots * self.slot_duration * 1000.0
            if average_delay_slots is not None else None
        )
        if self._delivered_packet_delays:
            delivered_delays = np.asarray(
                self._delivered_packet_delays, dtype=float
            )
            packet_delay_count = int(delivered_delays.size)
            average_packet_delay_slots = float(delivered_delays.mean())
            p95_packet_delay_slots = float(np.percentile(
                delivered_delays, 95
            ))
            average_packet_delay_ms = (
                average_packet_delay_slots
                * self.slot_duration * 1000.0
            )
            p95_packet_delay_ms = (
                p95_packet_delay_slots
                * self.slot_duration * 1000.0
            )
        else:
            packet_delay_count = 0
            average_packet_delay_slots = None
            p95_packet_delay_slots = None
            average_packet_delay_ms = None
            p95_packet_delay_ms = None

        # 吞吐量轨迹（cumulative throughput ratio 随时隙变化）
        cumulative_departures = departures.cumsum(dim=0)  # [num_slots, num_links]
        cumulative_arrivals = arrivals.cumsum(dim=0)
        # 避免除零：用 clamp 限制分子分母
        denom = torch.clamp(cumulative_arrivals.sum(dim=1), min=1e-9).to(departures.device)
        throughput_trajectory = (cumulative_departures.sum(dim=1) / denom).cpu().numpy()
        # 每时隙总到达/发送量（packets）
        arrivals_per_slot = arrivals.sum(dim=1).cpu().numpy()
        departures_per_slot = departures.sum(dim=1).cpu().numpy()
        active_links_per_slot = np.asarray([
            len(indices) for indices in self.history['active_links']
        ], dtype=float)
        nonempty_links_per_slot = np.asarray(
            self.history['nonempty_link_counts'], dtype=float
        )
        active_nonempty_per_slot = np.asarray(
            self.history['active_nonempty_counts'], dtype=float
        )
        active_empty_per_slot = np.asarray(
            self.history['active_empty_counts'], dtype=float
        )
        nonempty_activation_trajectory = np.divide(
            active_nonempty_per_slot,
            nonempty_links_per_slot,
            out=np.zeros_like(active_nonempty_per_slot),
            where=nonempty_links_per_slot > 0,
        )
        total_arrivals = arrivals.sum().item()
        total_admitted_arrivals = admitted_arrivals.sum().item()
        total_departures = departures.sum().item()
        throughput_ratio = total_departures / total_arrivals if total_arrivals > 0 else 0.0
        queue_growth_slope = self._queue_growth_slope(queue_lengths)
        deadline_dropped_packets = int(round(deadline_drop_counts.sum().item()))
        overflow_dropped_packets = int(round(overflow_drop_counts.sum().item()))
        delivered_packet_count = len(self._delivered_packet_delays)
        successful_packet_total_rate_pps = (
            delivered_packet_count / num_slots if num_slots > 0 else 0.0
        )
        successful_delivery_ratio = (
            delivered_packet_count / total_arrivals
            if total_arrivals > 0 else 1.0
        )
        unfinished_packet_count = (
            sum(len(queue) for queue in self._packet_cohorts)
            if self._packet_cohorts is not None else 0
        )
        packet_loss_rate = (
            (deadline_dropped_packets + overflow_dropped_packets)
            / total_arrivals if total_arrivals > 0 else 0.0
        )
        unfinished_packet_fraction = (
            unfinished_packet_count / total_arrivals
            if total_arrivals > 0 else 0.0
        )
        resolved_packet_fraction = (
            (delivered_packet_count + deadline_dropped_packets
             + overflow_dropped_packets) / total_arrivals
            if total_arrivals > 0 else 1.0
        )

        # Censoring-free QoS loss: only include arrivals old enough that they
        # must have either completed or expired by the end of the run.
        qos_eligible_arrivals = None
        qos_eligible_delivered = None
        qos_packet_loss_rate = None
        if self.delay_deadline_slots is not None:
            eligible_slots = max(
                0, num_slots - self.delay_deadline_slots - 1
            )
            qos_eligible_arrivals = int(round(
                arrivals[:eligible_slots].sum().item()
            ))
            qos_eligible_delivered = sum(
                int(arrival_slot) < eligible_slots
                for arrival_slot in self._delivered_packet_arrival_slots
            )
            if qos_eligible_arrivals > 0:
                qos_packet_loss_rate = max(
                    0.0,
                    min(
                        1.0,
                        (qos_eligible_arrivals - qos_eligible_delivered)
                        / qos_eligible_arrivals,
                    ),
                )

        metrics = {
            # 队列长度统计
            'avg_queue': avg_queue_per_slot[-1].item(),
            'final_avg_queue': avg_queue_per_slot[-1].item(),
            'max_queue': max_queue_per_slot[-1].item(),
            'final_max_queue': max_queue_per_slot[-1].item(),
            'long_term_avg_queue': queue_lengths.mean().item(),
            'long_term_max_queue': queue_lengths.max().item(),
            'average_queue_backlog': avg_queue_backlog,
            'average_queue_backlog_mbit': avg_queue_backlog * self.packet_size / 1e6,
            'average_total_queue_backlog': avg_total_queue_backlog,
            'average_total_queue_backlog_mbit': avg_total_queue_backlog * self.packet_size / 1e6,
            'average_delay_slots': average_delay_slots,
            'average_delay_ms': average_delay_ms,
            'packet_delay_count': packet_delay_count,
            'average_packet_delay_slots': average_packet_delay_slots,
            'average_packet_delay_ms': average_packet_delay_ms,
            'p95_packet_delay_slots': p95_packet_delay_slots,
            'p95_packet_delay_ms': p95_packet_delay_ms,
            'delivered_packet_count': delivered_packet_count,
            'successful_packet_total_rate_pps': successful_packet_total_rate_pps,
            'successful_delivery_ratio': successful_delivery_ratio,
            'successful_goodput_mbps': (
                successful_packet_total_rate_pps
                * self.packet_size / self.slot_duration / 1e6
            ),
            'deadline_dropped_packets': deadline_dropped_packets,
            'overflow_dropped_packets': overflow_dropped_packets,
            'unfinished_packet_count': unfinished_packet_count,
            'packet_loss_rate': packet_loss_rate,
            'unfinished_packet_fraction': unfinished_packet_fraction,
            'resolved_packet_fraction': resolved_packet_fraction,
            'qos_eligible_arrivals': qos_eligible_arrivals,
            'qos_eligible_delivered': qos_eligible_delivered,
            'qos_packet_loss_rate': qos_packet_loss_rate,

            # 到达/发送统计
            'total_arrivals': total_arrivals,
            'total_admitted_arrivals': total_admitted_arrivals,
            'total_departures': total_departures,
            'arrival_rate_pps': avg_arrival_rate_pps,
            'arrival_total_rate_pps': avg_total_arrival_rate_pps,

            # 归一化吞吐量（稳定性比）：departures / arrivals，无量纲，表示到达包中被成功发送的比例
            # - ≈1.0：系统稳定（到达率 ≤ 服务能力）
            # - <1.0：系统过载（队列会持续增长）
            'throughput_ratio': throughput_ratio,

            # 实际数据吞吐量（Mbps）：真正的网络传输速率
            # 计算方式：total_packets × packet_size(bits) / total_time(s) / 1e6
            'throughput_mbps': (
                departures.sum().item() * self.packet_size / (num_slots * self.slot_duration) / 1e6
            ) if num_slots > 0 else 0.0,

            # 每链路的平均发送速率（packets/slot）
            'departure_rate_pps': avg_departure_rate_pps,
            'departure_total_rate_pps': avg_total_departure_rate_pps,
            'average_active_links': float(active_links_per_slot.mean()),
            'average_nonempty_links': float(nonempty_links_per_slot.mean()),
            'nonempty_activation_rate': float(
                nonempty_activation_trajectory.mean()
            ),
            'average_active_empty_links': float(active_empty_per_slot.mean()),
            'average_mws_utility': float(np.mean(self.history['utilities'])),

            # 队列轨迹（用于绘图）
            'queue_trajectory': queue_lengths.cpu().numpy(),
            'avg_queue_trajectory': avg_queue_per_slot.cpu().numpy(),
            'max_queue_trajectory': max_queue_per_slot.cpu().numpy(),
            'queue_backlog_mbit_trajectory': (
                avg_queue_per_slot * self.packet_size / 1e6
            ).cpu().numpy(),

            # 吞吐量轨迹（用于绘图）
            'throughput_trajectory': throughput_trajectory,
            'arrivals_per_slot': arrivals_per_slot,
            'departures_per_slot': departures_per_slot,
            'active_links_per_slot': active_links_per_slot,
            'nonempty_links_per_slot': nonempty_links_per_slot,
            'nonempty_activation_trajectory': nonempty_activation_trajectory,

            # 效用轨迹：Σ Q_i × R_i × B × τ / L，单位是 packets/slot（与 Loss 一致）
            # 这是在仿真中追踪的"加权holtz-werk"值
            'utility_trajectory': np.array(self.history['utilities']),

            # 队列溢出概率
            'queue_overflow_prob': (queue_lengths[-1] > 50).float().mean().item(),

            # 稳定性判断
            'queue_growth_slope': queue_growth_slope,
            'is_stable': (
                throughput_ratio >= 0.95 and queue_growth_slope <= 0.01
            ),
        }
        
        return metrics
    
    @staticmethod
    def _queue_growth_slope(queue_lengths, window=100):
        """Estimate recent per-link queue drift in packets/slot."""
        window = min(int(window), queue_lengths.size(0))
        if window < 2:
            return 0.0
        recent = queue_lengths[-window:].mean(dim=1).float()
        time = torch.arange(window, dtype=recent.dtype, device=recent.device)
        time = time - time.mean()
        return float(torch.sum(time * (recent - recent.mean())) / torch.sum(time ** 2))


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
