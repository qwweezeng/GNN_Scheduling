import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

def quantize_distance(d, q=8, d_min=2.0, d_max=1000.0):
    """
    Quantize a distance d into a q-bit binary vector.

    Args:
        d: distance value (scalar)
        q: number of bits (e.g., 8)
        d_min, d_max: range to quantize

    Returns:
        q-dim binary vector (torch tensor)
    """
    if d < d_min:
        d = d_min
    elif d > d_max:
        d = d_max

    # Normalize to [0, 1]
    normalized = (d - d_min) / (d_max - d_min)
    # Quantize to q bits
    bin_index = int(normalized * (2**q - 1))
    # Convert to binary vector
    binary = [int(b) for b in format(bin_index, f'0{q}b')]
    return torch.tensor(binary, dtype=torch.float32)


def dequantize_distance_from_binindex(bin_index, q=8, d_min=2.0, d_max=1000.0):
    """
    Convert a q-bit bin index back to a representative distance (center of the bin).
    bin_index: integer or tensor of integers in [0, 2^q-1]
    Returns: distance value(s)
    """
    max_index = 2**q - 1
    # center of bin: (i + 0.5) / (2^q) mapped to [d_min, d_max]
    frac = (bin_index + 0.5) / (2**q)
    return d_min + frac * (d_max - d_min)


class NetworkGenerator:
    def __init__(self, num_uavs, area_width, height_min, height_max, uav_min_distance,
                 tx_power, Gt, Gr, noise_psd=-169, bandwidth=5e6, carrier_f=2.4e9, 
                 snr_thres=5, snr_gap_db=0, flow_lambda=10e6):
        # network parameters
        self.num_uavs = num_uavs
        self.area_width = area_width
        self.height_min = height_min
        self.height_max = height_max
        self.min_distance = uav_min_distance
        # communication parameters
        self.tx_power = 10**(tx_power/10) * 1e-3
        self.Gt = 10**(Gt/10)
        self.Gr = 10**(Gr/10)
        # ✅ 修正：bandwidth = 20 MHz (系统带宽), carrier_f = 1.37 GHz (载波频率)
        self.bandwidth = bandwidth
        self.carrier_f = carrier_f
        # 噪声功率 = PSD * 带宽
        self.noise_power = 10**(noise_psd/10) * 1e-3 * bandwidth
        # 自由空间路径损耗参考: lambda = c / f (使用载波频率)
        self.lambda0 = self.Gt * self.Gr * (3e8/carrier_f/4/np.pi)**2
        self.xi = 2.5 # path loss exponent
        self.snr_thres = snr_thres
        # ✅ SNR gap（用于rate计算，0 dB表示无gap）
        self.snr_gap_db = snr_gap_db
        self.snr_gap = 10**(snr_gap_db/10)
        self.flow_lambda = flow_lambda # flow in queue

        # initiate network state
        # self._init_positions()
        # self._init_queue_states()

    def _init_positions(self):
        positions = torch.zeros(self.num_uavs, 3)
        for i in range(self.num_uavs):
            valid_position = False
            attempts = 0
            while not valid_position and attempts < 100:
                # generate a random positon in the whole area
                x = torch.rand(1) * self.area_width
                y = torch.rand(1) * self.area_width
                z = self.height_min + torch.rand(1) * (self.height_max - self.height_min)
                new_pos = torch.tensor([x, y, z])

                # check whether meet min distance
                if i == 0:
                    valid_position = True
                else:
                    dists = torch.norm(positions[:i] - new_pos, dim=1)
                    if torch.all(dists >= self.min_distance):
                        valid_position = True
                attempts += 1
            positions[i] = new_pos
        return positions

    def _generate_queue_states(self, mode="random"):
        if mode == "random":
            # 随机生成队列长度，模拟不同拥塞状态
            queue_states = torch.randint(low=0, high=20, size=(self.num_uavs,)).float()
        elif mode == "poisson":
            # 泊松分布生成队列长度
            queue_states = torch.poisson(torch.ones(self.num_uavs) * self.flow_lambda)
        else:
            raise ValueError(f"Unknown queue mode: {mode}")
        return queue_states.float()

    def _channel_gain(self, positions):
        dist = torch.cdist(positions, positions, p=2)
        large_scale = self.lambda0**(1/2) * dist**(-self.xi / 2)
        # 正确生成复信道 h_ss ~ CN(0,1)：实部虚部独立 N(0, 0.5)，使得 |h_ss|² ~ Exp(1)
        real_part = torch.randn_like(dist) / np.sqrt(2)
        imag_part = torch.randn_like(dist) / np.sqrt(2)
        h = large_scale * (real_part + 1j * imag_part)
        h.fill_diagonal_(0)  # hii=0 so snr=0 no link avoid self transmission
        rx_power = self.tx_power * torch.abs(h) ** 2
        snr = rx_power / self.noise_power
        return dist, h, snr, large_scale

    def update_network(self, time_step=1.0, mobility_factor=0.1):
        # update uav positions and relative network since uav mobility
        # update uav positions (random walk)
        movement = mobility_factor * self.area_width * time_step
        delta_pos = (torch.rand(self.num_uavs, 3) - 0.5) * movement
        delta_pos[:, 2] *= (self.height_max - self.height_min) / self.area_width
        new_pos = self.positions + delta_pos

        # boundary processing (reflect)
        for i in range(3):
            if i < 2:  # x,y轴
                over = new_pos[:, i] > self.area_width
                under = new_pos[:, i] < 0
                new_pos[over, i] = 2 * self.area_width - new_pos[over, i]
                new_pos[under, i] = -new_pos[under, i]
            else:  # z轴
                over = new_pos[:, i] > self.height_max
                under = new_pos[:, i] < self.height_min
                new_pos[over, i] = 2 * self.height_max - new_pos[over, i]
                new_pos[under, i] = 2 * self.height_min - new_pos[under, i]

        self.positions = new_pos
        self.queue_states += torch.poisson(torch.ones(self.num_uavs) * 0.5 * time_step)

    # 生成网络主函数，一定有很多问题，注意打断点查看
    # csi_mode: "perfect" | "large_scale" | "quantized"
    # queue_mode: "random" | "poisson"
    def generate_network(self, csi_mode="perfect", q_bits=8, queue_mode="random", queue_states=None):
        positions = self._init_positions()
        if queue_states is not None:
            # 使用外部传入的队列状态（如仿真过程中的动态队列）
            self.queue_states = queue_states
        else:
            self.queue_states = self._generate_queue_states(mode=queue_mode)
        dist, h, snr, large_scale = self._channel_gain(positions)
        snr_thres_beta = 10 ** (self.snr_thres / 10)

        # 完美 CSI 接收功率；以及仅路径损耗的接收功率（不含小尺度衰落）
        signal_power = self.tx_power * torch.abs(h) ** 2
        large_scale_power = self.tx_power * (large_scale ** 2)
        large_scale_power.fill_diagonal_(0.0)
        snr_large_scale = large_scale_power / self.noise_power

        # 链路存在性判定
        if csi_mode in ("large_scale", "quantized"):
            valid_links = snr_large_scale >= snr_thres_beta
        else:
            valid_links = snr >= snr_thres_beta

        src, dst = torch.where(valid_links)
        num_links = src.size(0)
        links = torch.stack([src, dst], dim=1)

        # build conflict graph
        vertex_features = []
        conflict_edges = []
        edge_weights = []

        # 量化距离时预计算估计 SNR 矩阵
        if csi_mode == "quantized":
            d_min = self.min_distance
            height_span = self.height_max - self.height_min
            d_max = (self.area_width**2 + self.area_width**2 + height_span**2) ** 0.5
            frac = (dist - d_min) / (d_max - d_min)
            frac = torch.clamp(frac, 0.0, 1.0)
            max_index = float(2**q_bits - 1)
            bin_index_mat = torch.floor(frac * max_index)
            quantized_dist_mat = dequantize_distance_from_binindex(bin_index_mat, q=q_bits, d_min=d_min, d_max=d_max)
            # estimate large-scale received power (no small-scale fading): P_rx ≈ tx_power * lambda0 * d^{-xi}
            power_est_matrix = self.tx_power * self.lambda0 * (quantized_dist_mat ** (-self.xi))
            snr_est_matrix = power_est_matrix / self.noise_power

        # 速率计算用哪个功率矩阵
        if csi_mode == "large_scale":
            rate_power = large_scale_power
        elif csi_mode == "quantized":
            rate_power = power_est_matrix
        else:
            rate_power = signal_power

        # calculate vertex feature and edge weight
        for u_idx in range(num_links):
            u_tx, u_rx = src[u_idx].item(), dst[u_idx].item()
            # vertex feature: [SNR(dB), dist_normalized, queue_length_normalized]
            if csi_mode == "quantized":
                snr_val = snr_est_matrix[u_tx, u_rx].item()
                snr_db = 10 * np.log10(snr_val + 1e-12)
            elif csi_mode == "large_scale":
                snr_val = snr_large_scale[u_tx, u_rx].item()
                snr_db = 10 * np.log10(snr_val + 1e-12)
            else:
                snr_val = snr[u_tx, u_rx].item()
                snr_db = 10 * np.log10(snr_val + 1e-12)
            
            # 队列长度特征：取对数并归一化（避免数值差异过大）
            queue_len = self.queue_states[u_tx].item()
            queue_normalized = np.log1p(queue_len) / 10.0  # log(1+x)/10，队列10以下基本归一化到1
            
            vertex_features.append([snr_db, queue_normalized])

            # edge weight (interference)
            for v_idx in range(num_links):
                if u_idx == v_idx:
                    continue
                v_tx, v_rx = src[v_idx].item(), dst[v_idx].item()

                if u_tx == v_tx or u_tx == v_rx or u_rx == v_tx or u_rx == v_rx:
                    conflict_edges.append([u_idx, v_idx])
                    edge_weights.append(1e5)
                else:
                    # large_scale 与 quantized 都用 rate_power（区别仅在于距离是否量化）
                    # perfect 用 signal_power（含小尺度衰落）
                    if csi_mode in ("large_scale", "quantized"):
                        interfere_power = float(rate_power[u_tx, v_rx].item())
                        denom = float(rate_power[v_tx, v_rx].item()) / snr_thres_beta - float(self.noise_power)
                        denom = max(denom, 1e-30)
                        if denom <= interfere_power:
                            weight_val = float(interfere_power / denom)
                            conflict_edges.append([u_idx, v_idx])
                            edge_weights.append(weight_val)
                    else:
                        interfere_power = float(signal_power[u_tx, v_rx].item())
                        denom = float(signal_power[v_tx, v_rx].item()) / snr_thres_beta - float(self.noise_power)
                        denom = max(denom, 1e-30)
                        if denom <= interfere_power:
                            weight_val = float(interfere_power / denom)
                            conflict_edges.append([u_idx, v_idx])
                            edge_weights.append(weight_val)

        # to tensor - stack the feature vectors
        # Normalize vertex_features into a single tensor regardless of how items were appended.
        if len(vertex_features) > 0 and isinstance(vertex_features[0], torch.Tensor):
            vertex_features = torch.stack(vertex_features)
        else:
            vertex_features = torch.tensor(vertex_features, dtype=torch.float32)
        if conflict_edges:
            edge_index = torch.tensor(conflict_edges, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
            
            # 🔥 关键修复：对边权重取log并归一化，防止数值爆炸
            # 原始权重范围: 1 ~ 100000，log后: 0 ~ 11.5
            # 这样可以保留相对大小关系，同时避免巨大数值进入神经网络
            edge_weights = torch.log10(edge_weights + 1.0)  # log10(1) ~ log10(100001)
            # 进一步归一化到 [0, 1]
            edge_weights = edge_weights / edge_weights.max()
        else:
            # no interference
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_weights = torch.empty(0, 1, dtype=torch.float32)

        # neighbor mapping
        neighbor_map = defaultdict(list)
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            neighbor_map[v].append(u)

        # 量化功率矩阵（仅 csi_mode=="quantized" 时有效；其余模式设为 None）
        power_quant = power_est_matrix if csi_mode == "quantized" else None

        return NetworkState(
            links=links,
            vertex_features=vertex_features,
            edge_index=edge_index,
            edge_weights=edge_weights,
            large_scale=large_scale[src, dst],
            signal_power=signal_power,
            signal_power_large_scale=large_scale_power,
            signal_power_quant=power_quant,
            noise_power=self.noise_power,
            bandwidth=self.bandwidth,
            snr_gap=self.snr_gap,
            neighbor_map=neighbor_map,
            positions=positions,
            tx_power=self.tx_power,
            carrier_f=self.carrier_f,
            csi_mode=csi_mode,
            queue_states=self.queue_states,
        )

    def visualize_network(self, state, save_path=None):
        fig = plt.figure(figsize=(20, 10))

        # subplot1: connectivity graph
        ax1 = fig.add_subplot(121, projection='3d')
        pos = state.positions.cpu().numpy()
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=100, c='blue', label='UAVs')

        # snr(dB) -> color（vertex_features 已经是 dB，无需再次 log）
        snr_values = state.vertex_features[:, 0].cpu().numpy()
        norm_snr = Normalize(vmin=snr_values.min(), vmax=snr_values.max())
        cmap_snr = cm.viridis

        # links
        links_np = state.links.cpu().numpy()
        # 按节点对分组链路
        link_groups = {}
        for i, (src_idx, dst_idx) in enumerate(links_np):
            # 使用排序后的元组作为键，确保(0,1)和(1,0)被视为同一组
            key = tuple(sorted((src_idx, dst_idx)))
            if key not in link_groups:
                link_groups[key] = []
            link_groups[key].append(i)

        # 绘制通信链路并标注序号
        for (src_idx, dst_idx), link_ids in link_groups.items():
            src_pos = pos[src_idx]
            dst_pos = pos[dst_idx]

            # 计算中点
            mid_point = (src_pos + dst_pos) / 2

            # 获取大尺度衰落功率（对称，取任一方向相同）
            ls_power = snr_values[link_ids[0]]
            color = cmap_snr(norm_snr(ls_power))

            # 绘制链路（一条线表示物理连接）
            ax1.plot(
                [src_pos[0], dst_pos[0]],
                [src_pos[1], dst_pos[1]],
                [src_pos[2], dst_pos[2]],
                color=color, alpha=0.8, lw=2
            )

            # 标注所有链路序号（逗号分隔）
            ids_str = ",".join(str(id) for id in link_ids)
            ax1.text(
                mid_point[0], mid_point[1], mid_point[2],
                ids_str, fontsize=10, color='red', weight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )


        # for i, (src_idx, dst_idx) in enumerate(links_np):
        #     src_pos = pos[src_idx]
        #     dst_pos = pos[dst_idx]
        #     mid_point = (src_pos + dst_pos) / 2
        #     snr_val = snr_values[i]
        #     color = cmap_snr(norm_snr(snr_val))
        #     ax1.plot([src_pos[0], dst_pos[0]],
        #              [src_pos[1], dst_pos[1]],
        #              [src_pos[2], dst_pos[2]],
        #              color=color, alpha=0.8, lw=2)
        #     ax1.text(mid_point[0], mid_point[1], mid_point[2],
        #              f"{i}", fontsize=10, color='red', weight='bold')
        ax1.set_xlim(0, self.area_width)
        ax1.set_ylim(0, self.area_width)
        ax1.set_zlim(self.height_min, self.height_max)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('Connectivity Graph')
        # color label
        sm = cm.ScalarMappable(cmap=cmap_snr, norm=norm_snr)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, pad=0.1)
        cbar.set_label('Link SNR (dB)')

        # subplot2: conflict graph
        ax2 = fig.add_subplot(122)
        dg = nx.DiGraph()   # networkx
        for i in range(len(links_np)):
            dg.add_node(i, snr=snr_values[i])

        # edge: interference
        edge_index_np = state.edge_index.cpu().numpy()
        # edge_weights 已经做过 log 和归一化，这里只做一个线性映射，避免再次 log
        if state.edge_weights.numel() > 0:
            edge_weights_np = state.edge_weights.cpu().numpy().flatten()
        else:
            edge_weights_np = np.array([0.0])
        for i in range(edge_index_np.shape[1]):
            src = edge_index_np[0, i]
            dst = edge_index_np[1, i]
            weight = edge_weights_np[i]
            dg.add_edge(src, dst, link=(src, dst), weight=weight)
        # interference level -> color
        if len(edge_weights_np) > 0:
            norm_interference = Normalize(vmin=edge_weights_np.min(), vmax=edge_weights_np.max())
        else:
            norm_interference = Normalize(vmin=0, vmax=1)
        cmap_interference = cm.plasma

        # 绘制冲突图（使用力导向布局）
        pos = nx.spring_layout(dg, seed=42)
        vertex_colors = [cmap_snr(norm_snr(dg.nodes[n]['snr'])) for n in dg.nodes]
        nx.draw_networkx_nodes(dg, pos, node_size=500, node_color=vertex_colors, ax=ax2)
        edge_colors = []
        for u, v in dg.edges():
            weight = dg[u][v]['weight']
            edge_colors.append(cmap_interference(norm_interference(weight)))
        nx.draw_networkx_edges(
            dg, pos, edge_color=edge_colors,
            width=2, alpha=0.7, arrowstyle='->',
            arrowsize=20, ax=ax2
        )
        # subplot2 label
        nx.draw_networkx_labels(dg, pos, font_color="white", font_size=10, font_weight='bold', ax=ax2)
        edge_labels = {(u, v): f"{dg[u][v]['link']}" for u, v in dg.edges()}
        # nx.draw_networkx_edge_labels(
        #     dg, pos, ax=ax2
        # )
        # color label
        sm_snr = cm.ScalarMappable(cmap=cmap_snr, norm=norm_snr)
        sm_snr.set_array([])
        cbar_snr = fig.colorbar(sm_snr, ax=ax2, pad=0.05)
        cbar_snr.set_label('Vertex SNR (dB)')

        sm_interference = cm.ScalarMappable(cmap=cmap_interference, norm=norm_interference)
        sm_interference.set_array([])
        cbar_interference = fig.colorbar(sm_interference, ax=ax2, pad=0.05)
        cbar_interference.set_label('Edge Interference Weight (dB)')
        ax2.set_title('Conflict Graph')
        ax2.axis('off')

        plt.tight_layout()
        return fig

    def visualize_network_with_pi(self, state, pi, threshold=0.5, save_path=None):
        """
        可视化单个样本在当前调度策略下的激活情况。
        
        Args:
            state: NetworkState 对象
            pi: [num_links, 1] 模型输出的激活概率（tensor）
            threshold: 判定“激活”的阈值（比如 0.5）
            save_path: 若不为 None，则保存图片到该路径
        """
        pi = pi.detach().cpu().view(-1).numpy()
        active_mask = pi >= threshold

        fig = plt.figure(figsize=(20, 10))

        # subplot1: 3D 拓扑图 + 激活链路高亮
        ax1 = fig.add_subplot(121, projection='3d')
        pos = state.positions.cpu().numpy()
        ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=100, c='blue', label='UAVs')

        snr_db = state.vertex_features[:, 0].cpu().numpy()
        norm_snr = Normalize(vmin=snr_db.min(), vmax=snr_db.max())
        cmap_snr = cm.viridis

        links_np = state.links.cpu().numpy()

        for i, (src_idx, dst_idx) in enumerate(links_np):
            src_pos = pos[src_idx]
            dst_pos = pos[dst_idx]
            mid_point = (src_pos + dst_pos) / 2

            color = 'red' if active_mask[i] else 'gray'
            linewidth = 3 if active_mask[i] else 1
            alpha = 0.9 if active_mask[i] else 0.3

            ax1.plot(
                [src_pos[0], dst_pos[0]],
                [src_pos[1], dst_pos[1]],
                [src_pos[2], dst_pos[2]],
                color=color, alpha=alpha, lw=linewidth
            )

            # 在中点标注：链路编号 + SNR(dB) + π
            ax1.text(
                mid_point[0], mid_point[1], mid_point[2],
                f"{i}\n{snr_db[i]:.1f} dB\nπ={pi[i]:.2f}",
                fontsize=8,
                color='black',
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

        ax1.set_xlim(0, self.area_width)
        ax1.set_ylim(0, self.area_width)
        ax1.set_zlim(self.height_min, self.height_max)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title(f'Connectivity & Scheduling (threshold={threshold})')

        # subplot2: 冲突图 + 激活节点高亮
        ax2 = fig.add_subplot(122)
        dg = nx.DiGraph()

        for i in range(len(links_np)):
            dg.add_node(i, snr=snr_db[i], active=active_mask[i], pi=pi[i])

        edge_index_np = state.edge_index.cpu().numpy()
        if state.edge_weights.numel() > 0:
            edge_weights_np = state.edge_weights.cpu().numpy().flatten()
        else:
            edge_weights_np = np.array([0.0])

        for e_idx in range(edge_index_np.shape[1]):
            src = edge_index_np[0, e_idx]
            dst = edge_index_np[1, e_idx]
            weight = edge_weights_np[min(e_idx, len(edge_weights_np) - 1)]
            dg.add_edge(src, dst, weight=weight)

        if len(dg) > 0:
            pos_2d = nx.spring_layout(dg, seed=42)
        else:
            pos_2d = {}

        node_colors = []
        for n in dg.nodes:
            base_color = cmap_snr(norm_snr(snr_db[n]))
            if dg.nodes[n]['active']:
                # 激活的节点加深颜色
                node_colors.append((1.0, 0.0, 0.0, 0.9))  # 红色
            else:
                node_colors.append(base_color)

        nx.draw_networkx_nodes(dg, pos_2d, node_size=500, node_color=node_colors, ax=ax2)

        edge_colors = []
        if edge_index_np.shape[1] > 0:
            norm_interference = Normalize(vmin=edge_weights_np.min(), vmax=edge_weights_np.max())
            cmap_interference = cm.plasma
            for u, v in dg.edges():
                w = dg[u][v]['weight']
                edge_colors.append(cmap_interference(norm_interference(w)))
        else:
            edge_colors = 'gray'

        nx.draw_networkx_edges(
            dg, pos_2d, edge_color=edge_colors,
            width=2, alpha=0.7, arrowstyle='->',
            arrowsize=20, ax=ax2
        )

        nx.draw_networkx_labels(
            dg, pos_2d,
            labels={n: f"{n}\n{snr_db[n]:.1f}dB" for n in dg.nodes},
            font_color="white", font_size=8, font_weight='bold', ax=ax2
        )

        ax2.set_title('Conflict Graph with Scheduling')
        ax2.axis('off')

        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.show()
        return fig, active_mask

class NetworkState:
    __slots__ = ['links', 'vertex_features', 'edge_index', 'edge_weights', 'large_scale',
                 'signal_power', 'signal_power_large_scale', 'signal_power_quant',
                 'noise_power', 'bandwidth', 'snr_gap', 'neighbor_map',
                 'positions', 'num_vertices', 'tx_power', 'carrier_f', 'csi_mode',
                 'queue_states']
    def __init__(self, links, vertex_features, edge_index, edge_weights,
                 large_scale, signal_power, signal_power_large_scale, signal_power_quant,
                 noise_power, bandwidth, snr_gap, 
                 neighbor_map, positions, tx_power=None, carrier_f=None, csi_mode="perfect",
                 queue_states=None):
        self.links = links
        self.vertex_features = vertex_features
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.large_scale = large_scale
        self.signal_power = signal_power
        self.signal_power_large_scale = signal_power_large_scale
        self.signal_power_quant = signal_power_quant
        self.noise_power = noise_power
        self.bandwidth = bandwidth
        self.snr_gap = snr_gap
        self.neighbor_map = neighbor_map
        self.positions = positions
        self.num_vertices = links.size(0)
        self.tx_power = tx_power
        self.carrier_f = carrier_f
        self.csi_mode = csi_mode
        self.queue_states = queue_states

if __name__ == '__main__':
    # 参数设置
    NUM_UAV = 5
    AREA_WIDTH = 1000  # 1km x 1km区域
    HEIGHT_RANGE = (50, 150)  # 高度范围50-150米
    MIN_DISTANCE = 20  # 最小安全距离20米
    TX_POWER = 22   # tx power (dBm)
    NOISE_PSD = -174    # noise power spectral density (dBm)
    G_T = 0 # TX antenna gain (dBi)
    G_R = 0 # RX antenna gain (dBi)
    generator = NetworkGenerator(
        num_uavs=NUM_UAV,
        area_width=AREA_WIDTH,
        height_min=HEIGHT_RANGE[0],
        height_max=HEIGHT_RANGE[1],
        uav_min_distance=MIN_DISTANCE,
        tx_power=TX_POWER, Gt=G_T, Gr=G_R
    )

    # 生成单个网络状态
    state = generator.generate_network()
    netwrok_state = state.generate_network()
    fig = generator.visualize_network()
    plt.show()