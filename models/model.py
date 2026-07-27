import torch.nn as nn
import torch
import torch.nn.functional as F

# ----- node embedding MLP -----
class NodeEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# ----- GAT Layer -----
class GATLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, num_heads=1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.total_out_dim = out_dim * num_heads
        self.leaky_relu = nn.LeakyReLU(0.2)  # the slope can modify!

        # Learnable weight matrices W_node and W_edge
        self.W_node = nn.ModuleList([
            nn.Linear(node_dim, out_dim, bias=False)
            for _ in range(num_heads)
        ])
        self.W_edge = nn.ModuleList([
            nn.Linear(edge_dim, out_dim, bias=False)
            for _ in range(num_heads)
        ])

        # Learnable attention parameter a
        self.attn = nn.ModuleList([
            nn.Linear(3 * out_dim, 1, bias=False)
            for _ in range(num_heads)
        ])

        # output layer (for multi-head result processing)
        self.W_out = nn.Linear(self.total_out_dim, self.total_out_dim)
        # GAT 论文 Eq.(5): 每个 head 在拼接前先做 σ（中间层用 ELU）
        self.head_activation = nn.ELU()

    def forward(self, x, edge_index, edge_attr):
        """
        x: node feature, [num_vertices(links), d_input]
        edge_index: [2, num_edges], edge_index[0/1] = source/target node. Here, source interferes with target.
        edge_attr: [num_edges, edge_dim]
        Returns: new node feature h_v(t+1) [num_vertices, total_out_dim]
        
        修复说明：
        1. 添加自环（self-loop）以保留节点自身信息，这是最关键的修复
        2. 修复softmax维度，确保在目标节点的所有邻居（包括自身）上归一化
        """
        num_vertices = x.size(0)
        num_edges = edge_attr.size(0)
         
        # 【关键修复1】添加自环：让每个节点在聚合信息时也考虑自己
        # 自环就是"自己指向自己"的边，比如节点0->节点0，节点1->节点1，等等
        # 这样在聚合时，节点不仅会接收邻居的信息，也会接收自己的信息
        node_ids = torch.arange(num_vertices, device=x.device)  # [0, 1, 2, ..., num_vertices-1]
        self_loops = torch.stack([node_ids, node_ids], dim=0)  # 自环：每个节点指向自己 [2, num_vertices]
        zero_edge_attr = torch.zeros(num_vertices, edge_attr.size(1), device=x.device)  # 自环边特征为0（无干扰）
        
        # 把原始边和自环拼接在一起
        edge_index_with_loop = torch.cat([edge_index, self_loops], dim=1)  # [2, num_edges + num_vertices]
        edge_attr_with_loop = torch.cat([edge_attr, zero_edge_attr], dim=0)  # [num_edges + num_vertices, edge_dim]
        
        src, dst = edge_index_with_loop[0], edge_index_with_loop[1]
        num_edges_with_loop = edge_attr_with_loop.size(0)
        head_outputs = []

        # for every head
        for head in range(self.num_heads):
            # vertex feature W_n*h_v
            W_h = self.W_node[head](x)  # [num_vertices, out_dim]
            # edge feature w_e*w_uv
            W_e = self.W_edge[head](edge_attr_with_loop) # [num_edges_with_loop, out_dim]

            # concat node and edge features for attention computation
            attention_input = torch.cat([W_h[src], W_h[dst], W_e], dim=-1) # [num_edges_with_loop, 3*out_dim]

            # compute attention scores for each edge
            e_uv = self.leaky_relu(self.attn[head](attention_input)).squeeze(-1)  # [num_edges_with_loop]

            # 对每个目标节点 dst，在其所有入边上做归一化 (group-wise softmax)
            if hasattr(torch.Tensor, "scatter_reduce_"):
                max_per_dst = torch.full((num_vertices,), -float("inf"), device=x.device)
                max_per_dst.scatter_reduce_(0, dst, e_uv, reduce="amax")
                max_per_dst = torch.where(
                    max_per_dst == -float("inf"),
                    torch.zeros_like(max_per_dst),
                    max_per_dst
                )
                exp_scores = torch.exp(e_uv - max_per_dst[dst])
                denom_per_dst = torch.zeros(num_vertices, device=x.device)
                denom_per_dst.scatter_reduce_(0, dst, exp_scores, reduce="sum")
            else:
                max_per_dst = torch.full((num_vertices,), -float("inf"), device=x.device)
                denom_per_dst = torch.zeros(num_vertices, device=x.device)
                exp_scores = torch.zeros_like(e_uv)
                for v in range(num_vertices):
                    mask = (dst == v)
                    if mask.any():
                        scores_v = e_uv[mask]
                        max_v = scores_v.max()
                        max_per_dst[v] = max_v
                        exp_v = torch.exp(scores_v - max_v)
                        exp_scores[mask] = exp_v
                        denom_per_dst[v] = exp_v.sum()
                max_per_dst[max_per_dst == -float("inf")] = 0.0
                denom_per_dst = denom_per_dst + 1e-12

            edge_alpha = exp_scores / (denom_per_dst[dst] + 1e-12)  # [num_edges_with_loop]

            # aggregate messages (including self-loops)
            weighted_message = edge_alpha.unsqueeze(-1) * (W_h[src] + W_e)  # [num_edges_with_loop, out_dim]
            aggregated = torch.zeros(num_vertices, self.out_dim, device=x.device)
            aggregated.index_add_(0, dst, weighted_message)
            # head_outputs.append(aggregated) #上版
            # GAT 论文 Eq.(5): 拼接前对每个 head 的输出施加 σ，即 h'_i = ||_{k=1}^K σ(Σ α^k_ij W^k h_j)
            head_outputs.append(self.head_activation(aggregated))

        # concat multi-head output
        multi_head_output = torch.cat(head_outputs, dim=1)  # [num_edges, total_out_dim]

        # new_node_feature = self.W_out(multi_head_output) #上版
        # 线性输出变换（GAT 原文在 concat 后还有一层线性/输出变换，此处用 W_out）
        new_node_feature = self.W_out(multi_head_output)
        return new_node_feature

# ----- GATScheduler -----
class GATScheduler(nn.Module):
    def __init__(self, raw_node_dim, mlp_hidden_dims, embed_node_dim,
                 edge_dim, gat_out_dim, num_heads, T_iterations,
                 schedule_hidden_dim=32, dropout=0.0,
                 use_power_control=False, power_hidden_dim=16,
                 input_feature_mean=None, input_feature_std=None,
                 use_input_skip=False, initial_schedule_bias=-0.5,
                 stochastic_ste=False):
        super().__init__()
        self.use_power_control = use_power_control
        self.power_hidden_dim = power_hidden_dim
        self.use_input_skip = use_input_skip
        self.stochastic_ste = stochastic_ste
        self.decision_threshold = 0.5

        # Input normalization is part of the model inference path so training
        # and multi-slot simulation always use the same statistics. These
        # buffers are saved as checkpoint metadata by the queue experiment.
        feature_mean = torch.zeros(raw_node_dim, dtype=torch.float32)
        feature_std = torch.ones(raw_node_dim, dtype=torch.float32)
        if input_feature_mean is not None:
            feature_mean = torch.as_tensor(input_feature_mean, dtype=torch.float32)
        if input_feature_std is not None:
            feature_std = torch.as_tensor(input_feature_std, dtype=torch.float32)
        if feature_mean.numel() != raw_node_dim or feature_std.numel() != raw_node_dim:
            raise ValueError(
                f"Input normalization expects {raw_node_dim} values, "
                f"got mean={feature_mean.numel()}, std={feature_std.numel()}."
            )
        self.register_buffer("input_feature_mean", feature_mean, persistent=False)
        self.register_buffer(
            "input_feature_std", torch.clamp(feature_std, min=1e-6), persistent=False
        )
        self.latest_scores = None
        self.latest_soft_probabilities = None

        # Initial MLP for node embedding
        self.node_embed = NodeEmbedding(
            input_dim=raw_node_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=embed_node_dim,
            dropout=dropout)

        # GAT layers
        self.T_iterations = T_iterations  # Number of GAT message passing iterations
        self.gat_layers = nn.ModuleList()
        for i in range(T_iterations):
            gat_input_dim = embed_node_dim if i == 0 else gat_out_dim
            gat_output_dim_per_head = gat_out_dim if i == T_iterations - 1 else gat_out_dim // num_heads
            heads = 1 if i == T_iterations - 1 else num_heads
            self.gat_layers.append(GATLayer(
                node_dim=gat_input_dim,
                edge_dim=edge_dim,
                out_dim=gat_output_dim_per_head,
                num_heads=heads))

        # Final MLP for scheduling probability decision
        # 🔥 关键修改：改用 sigmoid 直接输出概率，避免 softmax 饱和问题
        # 优点：
        # 1. 每个链路独立决策，不会因为 softmax 归一化导致竞争
        # 2. 梯度更稳定，不会突然饱和（sigmoid 的梯度在 [-3, 3] 范围内都有效）
        # 3. 避免了 softmax 的"一旦进入饱和区就无法回退"的问题
        # 缺点：
        # 1. 失去了"竞争"机制，需要更强的稀疏性惩罚和冲突惩罚
        # 2. 需要调整初始化，让初始概率接近 0.1-0.2（而不是 0.5）
        scheduler_input_dim = gat_out_dim + raw_node_dim if use_input_skip else gat_out_dim
        final_layers = [nn.Linear(scheduler_input_dim, schedule_hidden_dim), nn.ReLU()]
        if dropout and dropout > 0.0:
            final_layers.append(nn.Dropout(p=dropout))
        final_layers.append(nn.Linear(schedule_hidden_dim, 1))
        self.final_scheduler = nn.Sequential(*final_layers)
        # Queue v3 uses bias=0 so stochastic STE starts with both 0/1
        # decisions present; the non-queue baseline keeps its historical bias.
        last_linear_idx = len(final_layers) - 1  # 最后Linear层的索引
        nn.init.constant_(
            self.final_scheduler[last_linear_idx].bias,
            float(initial_schedule_bias),
        )
        # 权重初始化：使用Xavier初始化，让梯度更稳定
        nn.init.xavier_uniform_(self.final_scheduler[0].weight, gain=0.3)  # 第一个Linear层
        nn.init.xavier_uniform_(self.final_scheduler[last_linear_idx].weight, gain=0.3)  # 最后一个Linear层

        # ========== 功率控制双头 ==========
        # 第二个头：功率缩放系数 α_i ∈ (0,1]
        if self.use_power_control:
            power_layers = [nn.Linear(gat_out_dim, power_hidden_dim), nn.ReLU()]
            if dropout and dropout > 0.0:
                power_layers.append(nn.Dropout(p=dropout))
            power_layers.append(nn.Linear(power_hidden_dim, 1))
            self.power_head = nn.Sequential(*power_layers)
            # 初始化：让初始 α 接近 0.8（高功率基线，给模型更多降功率空间）
            # sigmoid(score) = 0.8 时，score ≈ 1.39
            power_last_linear_idx = len(power_layers) - 1
            nn.init.constant_(self.power_head[power_last_linear_idx].bias, 1.39)
            nn.init.xavier_uniform_(self.power_head[0].weight, gain=0.3)
            nn.init.xavier_uniform_(self.power_head[power_last_linear_idx].weight, gain=0.3)

    def encode(self, x, edge_index, edge_attr):
        """Return normalized local features and the final GAT embedding."""
        x_normalized = (x - self.input_feature_mean) / self.input_feature_std
        h_v = self.node_embed(x_normalized)
        for i in range(self.T_iterations):
            h_v = self.gat_layers[i](h_v, edge_index, edge_attr)
            if i != self.T_iterations - 1:
                h_v = F.elu(h_v)
        return x_normalized, h_v

    def schedule_logits(self, x, edge_index, edge_attr):
        """Return pre-sigmoid scheduling scores and the channel embedding."""
        x_normalized, h_v_final = self.encode(x, edge_index, edge_attr)
        scheduler_input = (
            torch.cat([h_v_final, x_normalized], dim=1)
            if self.use_input_skip else h_v_final
        )
        scores = self.final_scheduler(scheduler_input)
        return scores, h_v_final

    def forward(self, x, edge_index, edge_attr, use_ste=False):
        """
        Args:
            x: Initial node features [N, raw_node_dim] (SNR_v, Q_v, P_v, S_v)
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: w_uv [num_edges, 1]
            use_ste: 是否使用Straight-Through Estimator（训练时用硬决策，反向用软概率）
        Returns:
            pi: Scheduling probabilities [num_nodes, 1]，值域[0,1]
            alpha: Power scaling factors [num_nodes, 1]，值域(0,1]（仅 use_power_control=True 模式）
            h_v_final: Final node embedding [num_nodes, gat_out_dim * num_heads]
        """
        scores, h_v_final = self.schedule_logits(x, edge_index, edge_attr)
        self.latest_scores = scores.detach()
        
        # 🔥 温度缩放：训练时使用较大温度，让分布更平缓，梯度更稳定
        # 评估时使用较小温度，让决策更清晰
        temperature = 2.0 if self.training else 1.0
        pi_soft = torch.sigmoid(scores / temperature)  # [N, 1]，连续概率，值域[0,1]
        
        # 🔥 轻量化“夹紧”：保持梯度，不会像 clamp 那样把梯度置零
        # 线性缩放到 (ε, 1-ε) 区间，既避免完全饱和，又保留梯度
        eps = 1e-3
        pi_soft = pi_soft * (1 - 2 * eps) + eps
        self.latest_soft_probabilities = pi_soft
        
        if use_ste and self.training:
            # Straight-Through Estimator (STE):
            # 前向传播：使用硬决策（0/1）打破对称性
            # 反向传播：梯度通过软概率回传，保持可微
            pi_hard = (
                torch.bernoulli(pi_soft)
                if self.stochastic_ste
                else (pi_soft > 0.5).float()
            )
            pi_v = pi_hard - pi_soft.detach() + pi_soft  # STE技巧
        else:
            # 评估时或不用STE：直接返回连续概率
            pi_v = pi_soft

        # ========== 功率控制输出 ==========
        if self.use_power_control:
            alpha_scores = self.power_head(h_v_final)  # [N, 1]
            # 功率缩放：使用 sigmoid 确保 α ∈ (0,1)
            # 训练时用温度缩放，评估时用原始 sigmoid
            alpha_temp = 1.5 if self.training else 1.0
            alpha = torch.sigmoid(alpha_scores / alpha_temp)  # [N, 1]，值域 (0,1)
            # 夹紧到 (ε*0.5, 1-ε)，避免完全饱和
            eps = 1e-3
            alpha = alpha * (1 - eps) + eps * 0.5  # 最小约 0.0005
            return pi_v, alpha, h_v_final
        else:
            return pi_v, h_v_final

def calculate_achievable_rate(pi, state, binary=True, detach_hard=False, alpha=None):
    """
    计算可达速率（支持功率控制）。

    signal_power / signal_power_large_scale / signal_power_quant 均存储在 state 中，
    由 generate_network 根据 CSI_MODE 统一计算并传入。

    当 alpha 不为 None 时，启用功率控制模式：
        P_i = π_i * α_i * P_max
        干扰 = sum(g[j,i] * π_j * α_j) for j != i
    """
    num_links = state.num_vertices
    device = pi.device

    if binary:
        if detach_hard:
            pi_decision = torch.round(pi.detach())
        else:
            pi_decision = torch.round(pi)
    else:
        pi_decision = pi

    # 三套功率矩阵由 NetworkState 统一管理，此处直接取用
    sp = state.signal_power
    if getattr(state, "signal_power_large_scale", None) is not None:
        sp_ls = state.signal_power_large_scale
    else:
        sp_ls = sp
    if getattr(state, "signal_power_quant", None) is not None:
        sp_q = state.signal_power_quant
    else:
        sp_q = sp

    # 由 state 上的字段标记当前 CSI_MODE
    csi_mode = getattr(state, "csi_mode", "perfect")
    if csi_mode == "large_scale":
        signal_power = sp_ls.to(device)
    elif csi_mode == "quantized":
        signal_power = sp_q.to(device)
    else:
        signal_power = sp.to(device)

    # links[i] = (tx_uav_idx, rx_uav_idx)
    link_tx = state.links[:, 0]  # [num_links]
    link_rx = state.links[:, 1]  # [num_links]

    # 获取每条链路的直射功率增益 g_diag[i] = signal_power[tx_i, rx_i]
    g_diag = signal_power[link_tx, link_rx]  # [num_links]

    # 计算干扰: interference[i] = sum(signal_power[tx_j, rx_i] * pi[j]) for j != i
    # 构建 g_matrix [num_links, num_links]
    # 正确: g_matrix[i, j] = signal_power[tx_j, rx_i] (从链路 j 的发射端到链路 i 的接收端的增益)
    # 使用 broadcast: link_tx[None, :] = [1, num_links], link_rx[:, None] = [num_links, 1]
    g_matrix = signal_power[link_tx[None, :], link_rx[:, None]]  # [num_links, num_links]
    
    # 排除自干扰 (对角线)
    mask = 1 - torch.eye(num_links, device=device)
    g_nondiag = g_matrix * mask

    # ========== 功率控制模式 ==========
    pi_flat = pi_decision.squeeze(-1)  # [num_links]
    if alpha is not None:
        # alpha 是功率缩放系数，使用 detach 的值避免梯度异常
        alpha_decision = alpha.detach().squeeze(-1)  # [num_links]
        effective_pi = pi_flat * alpha_decision
        # 干扰 = sum(g[j,i] * π_j * α_j)
        interference_power = torch.matmul(g_nondiag, effective_pi)  # [num_links]
        # 信号功率需要乘以 α
        g_diag_eff = g_diag * alpha_decision
    else:
        # 非功率控制模式：与原逻辑一致
        interference_power = torch.matmul(g_nondiag, pi_flat)  # [num_links]
        g_diag_eff = g_diag

    # 计算 SINR（使用有效信号功率）
    # 注意：不需要额外 epsilon，因为 noise_power > 0 已保证分母不为 0
    sinr = g_diag_eff / (interference_power + state.noise_power)
    
    # 计算速率: rate = log2(1 + SINR / SNR_gap) [每Hz的速率，不乘以带宽]
    # 这样可以保持数值稳定，便于训练
    snr_gap = getattr(state, 'snr_gap', 1.0)
    # 使用原始 pi_flat，而非 effective_pi
    achievable_rates = pi_flat * torch.log2(1 + sinr / snr_gap)

    return achievable_rates
