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
                 schedule_hidden_dim=32, dropout=0.0):
        super().__init__()
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
        final_layers = [nn.Linear(gat_out_dim, schedule_hidden_dim), nn.ReLU()]
        if dropout and dropout > 0.0:
            final_layers.append(nn.Dropout(p=dropout))
        final_layers.append(nn.Linear(schedule_hidden_dim, 1))
        self.final_scheduler = nn.Sequential(*final_layers)
        # 🔥 改进初始化：让初始概率更大，避免过早陷入0
        # 初始化：让初始概率接近 0.3-0.4（中等激活，给模型更多探索空间）
        # sigmoid(score) = 0.3 时，score ≈ -0.85
        # sigmoid(score) = 0.4 时，score ≈ -0.41
        # 我们初始化 bias 为 -0.5，让初始概率在 0.38 左右（更高，避免过早陷入0）
        last_linear_idx = len(final_layers) - 1  # 最后Linear层的索引
        nn.init.constant_(self.final_scheduler[last_linear_idx].bias, -0.5)
        # 权重初始化：使用Xavier初始化，让梯度更稳定
        nn.init.xavier_uniform_(self.final_scheduler[0].weight, gain=0.3)  # 第一个Linear层
        nn.init.xavier_uniform_(self.final_scheduler[last_linear_idx].weight, gain=0.3)  # 最后一个Linear层

    def forward(self, x, edge_index, edge_attr, use_ste=False):
        """
        Args:
            x: Initial node features [N, raw_node_dim] (SNR_v, Q_v, P_v, S_v)
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: w_uv [num_edges, 1]
            use_ste: 是否使用Straight-Through Estimator（训练时用硬决策，反向用软概率）
        Returns:
            pi: Scheduling probabilities [num_nodes, 1]，值域[0,1]
            h_v_final: Final node embedding [num_nodes, gat_out_dim * num_heads]
        """
        h_v = self.node_embed(x)    # initial node embedding

        # GAT layers for T iterations (message passing)
        for i in range(self.T_iterations):
            h_v = self.gat_layers[i](h_v, edge_index, edge_attr)
            if i != self.T_iterations - 1:
                h_v = F.elu(h_v)    # initial multi-head GAT paper uses elu, may try other activation functions

        h_v_final = h_v

        # Final scheduling: 使用 sigmoid 直接输出概率
        # 🔥 关键改进：避免 softmax 饱和问题
        scores = self.final_scheduler(h_v_final)  # [N, 1]，原始分数
        
        # 🔥 温度缩放：训练时使用较大温度，让分布更平缓，梯度更稳定
        # 评估时使用较小温度，让决策更清晰
        temperature = 2.0 if self.training else 1.0
        pi_soft = torch.sigmoid(scores / temperature)  # [N, 1]，连续概率，值域[0,1]
        
        # 🔥 轻量化“夹紧”：保持梯度，不会像 clamp 那样把梯度置零
        # 线性缩放到 (ε, 1-ε) 区间，既避免完全饱和，又保留梯度
        eps = 1e-3
        pi_soft = pi_soft * (1 - 2 * eps) + eps
        
        if use_ste and self.training:
            # 🔥 Straight-Through Estimator (STE):
            # 前向传播：使用硬决策（0/1）打破对称性
            # 反向传播：梯度通过软概率回传，保持可微
            pi_hard = (pi_soft > 0.5).float()  # 硬决策：>0.5为1，否则为0
            pi_v = pi_hard - pi_soft.detach() + pi_soft  # STE技巧
        else:
            # 评估时或不用STE：直接返回连续概率
            pi_v = pi_soft
            
        return pi_v, h_v_final

def calculate_achievable_rate(pi, state, binary=True, detach_hard=False):
    """
    计算可达速率。
    signal_power / signal_power_large_scale / signal_power_quant 均存储在 state 中，
    由 generate_network 根据 CSI_MODE 统一计算并传入。
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

    # 计算干扰功率
    pi_flat = pi_decision.squeeze(-1)  # [num_links]
    interference_power = torch.matmul(g_nondiag, pi_flat)  # [num_links]

    # 计算 SINR
    # 注意：不需要额外 epsilon，因为 noise_power > 0 已保证分母不为 0
    sinr = g_diag / (interference_power + state.noise_power)
    
    # 计算速率: rate = log2(1 + SINR / SNR_gap) [每Hz的速率，不乘以带宽]
    # 这样可以保持数值稳定，便于训练
    snr_gap = getattr(state, 'snr_gap', 1.0)
    # ✅ 修复：非激活链路的速率应该为 0
    achievable_rates = pi_flat * torch.log2(1 + sinr / snr_gap)

    return achievable_rates

# ========== 方案1：冲突二次惩罚 ==========
def compute_conflict_penalty(pi, edge_index, edge_weights, penalty_scale=1.0, normalize=True):
    """
    计算冲突二次惩罚项
    
    Args:
        pi: [N, 1] 激活概率
        edge_index: [2, E] 冲突边索引，edge_index[0]是源节点，edge_index[1]是目标节点
        edge_weights: [E, 1] 冲突边权重（已经log归一化）
        penalty_scale: 惩罚缩放因子，用于调整惩罚强度
        normalize: 是否归一化（按链路数量），避免不同规模网络的惩罚项量级差异过大
    
    Returns:
        conflict_penalty: 标量，冲突惩罚值
    """
    if edge_index.size(1) == 0:  # 没有冲突边
        return torch.tensor(0.0, device=pi.device)
    
    src, dst = edge_index[0], edge_index[1]
    pi_src = pi[src].squeeze(-1)  # [E]
    pi_dst = pi[dst].squeeze(-1)  # [E]
    
    # 冲突惩罚：w_ij * π_i * π_j
    # 如果两个冲突链路都激活（π_i≈1, π_j≈1），惩罚项会很大
    # 如果其中一个关闭（π_i≈0或π_j≈0），惩罚项会很小
    edge_weights_flat = edge_weights.squeeze(-1)  # [E]
    
    # 计算原始惩罚项
    raw_penalty = torch.sum(edge_weights_flat * pi_src * pi_dst)
    
    # 🔥 关键改进：归一化到每个链路的平均惩罚，避免网络规模影响
    if normalize:
        num_links = pi.size(0)
        if num_links > 0:
            # 归一化：惩罚项 / 链路数量，让不同规模网络的惩罚项量级相似
            normalized_penalty = raw_penalty / num_links
        else:
            normalized_penalty = raw_penalty
    else:
        normalized_penalty = raw_penalty
    
    conflict_penalty = normalized_penalty * penalty_scale
    
    return conflict_penalty

# ========== 方案2：可微Top-K选择器（带冲突掩码） ==========
def build_conflict_mask(edge_index, num_nodes):
    """
    构建冲突掩码矩阵
    
    Args:
        edge_index: [2, E] 冲突边索引
        num_nodes: 节点数量
    
    Returns:
        conflict_mask: [N, N] 布尔矩阵，conflict_mask[i,j]=True表示链路i和j冲突
    """
    conflict_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=edge_index.device)
    if edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        conflict_mask[src, dst] = True
        # 冲突是对称的（如果i干扰j，那么j也可能干扰i，或者至少不能同时激活）
        conflict_mask[dst, src] = True
    return conflict_mask

def differentiable_topk_gumbel(scores, k, temperature=1.0):
    """
    使用Gumbel噪声实现可微的Top-K选择
    
    Args:
        scores: [N, 1] 分数
        k: 要选择的链路数量
        temperature: 温度参数，越小越接近硬选择
    
    Returns:
        mask: [N, 1] one-hot mask，选中的链路为1，否则为0
    """
    # Gumbel噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
    perturbed_scores = scores + temperature * gumbel_noise
    
    # Top-K选择
    k = min(k, scores.size(0))
    topk_values, topk_indices = torch.topk(perturbed_scores.squeeze(-1), k, dim=0)
    
    # 创建one-hot mask
    mask = torch.zeros_like(scores)
    mask[topk_indices] = 1.0
    return mask

def greedy_selection_with_conflict_mask(scores, edge_index, k, max_iterations=None):
    """
    逐步贪心选择，每次选择分数最高的未冲突链路
    
    Args:
        scores: [N, 1] 分数
        edge_index: [2, E] 冲突边索引
        k: 目标选择数量
        max_iterations: 最大迭代次数（如果为None，则使用k）
    
    Returns:
        selected: [N, 1] 选中的链路为1，否则为0
    """
    num_nodes = scores.size(0)
    selected = torch.zeros(num_nodes, device=scores.device)
    conflict_mask = build_conflict_mask(edge_index, num_nodes)
    
    if max_iterations is None:
        max_iterations = min(k, num_nodes)
    
    for _ in range(max_iterations):
        # 计算可用分数（已选链路和冲突链路分数置为-inf）
        available_scores = scores.clone().squeeze(-1)
        available_scores[selected > 0.5] = -float('inf')
        
        # 掩码冲突链路：如果链路i已选，则其所有冲突邻居的分数置为-inf
        for i in range(num_nodes):
            if selected[i] > 0.5:  # 如果链路i已选
                conflict_neighbors = conflict_mask[i]
                available_scores[conflict_neighbors] = -float('inf')
        
        # 选择分数最高的
        best_idx = available_scores.argmax()
        if available_scores[best_idx] > -float('inf'):
            selected[best_idx] = 1.0
        else:
            break  # 没有可用链路了
    
    return selected.unsqueeze(-1)

# ========== 方案3：自适应λ调整（目标并发度ρ） ==========
class AdaptiveSparsityController:
    """
    自适应稀疏控制器：通过拉格朗日对偶方法自适应调整λ
    
    核心思想：
    - 设置目标并发度ρ（期望激活的链路比例）
    - 如果当前激活率 > ρ，增大λ（惩罚激活）
    - 如果当前激活率 < ρ，减小λ（鼓励激活）
    """
    def __init__(self, target_rho=0.3, lambda_init=0.5, lr_lambda=0.01, lambda_min=0.0, lambda_max=10.0):
        """
        Args:
            target_rho: 目标并发度（期望激活的链路比例，如0.3表示30%的链路激活）
            lambda_init: 初始λ值
            lr_lambda: λ的学习率（对偶更新的步长）
            lambda_min: λ的最小值
            lambda_max: λ的最大值
        """
        self.target_rho = target_rho
        self.lambda_val = lambda_init
        self.lr_lambda = lr_lambda
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
    
    def update(self, current_rho):
        """
        根据当前激活率更新λ
        
        Args:
            current_rho: 当前epoch的平均激活率（mean(π)）
        
        Returns:
            lambda_val: 更新后的λ值
        """
        # 对偶更新：λ_{t+1} = [λ_t + η·(mean(π_t) - ρ)]_+
        # 如果激活率太高（current_rho > target_rho），增大λ
        # 如果激活率太低（current_rho < target_rho），减小λ
        self.lambda_val = self.lambda_val + self.lr_lambda * (current_rho - self.target_rho)
        # 裁剪到合理范围
        self.lambda_val = max(self.lambda_min, min(self.lambda_max, self.lambda_val))
        return self.lambda_val
    
    def get_lambda(self):
        """获取当前λ值"""
        return self.lambda_val

# ========== 方案4：特征增强 ==========
def enhance_node_features(state):
    """
    增强节点特征，添加度数、局部干扰统计等
    
    Args:
        state: NetworkState对象
    
    Returns:
        enhanced_features: [N, enhanced_dim] 增强后的特征
    """
    num_links = state.num_vertices
    edge_index = state.edge_index
    edge_weights = state.edge_weights
    device = state.vertex_features.device
    
    if num_links == 0:
        return state.vertex_features
    
    # 1. 计算度数
    in_degree = torch.zeros(num_links, device=device)
    out_degree = torch.zeros(num_links, device=device)
    if edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        in_degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    
    # 2. 计算局部干扰统计
    in_interference = torch.zeros(num_links, device=device)
    out_interference = torch.zeros(num_links, device=device)
    if edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        weights_flat = edge_weights.squeeze(-1)
        out_interference.scatter_add_(0, src, weights_flat)
        in_interference.scatter_add_(0, dst, weights_flat)
    
    # 3. SNR排名
    snr_values = state.vertex_features[:, 0]
    snr_rank = torch.argsort(torch.argsort(snr_values, descending=True)).float()
    if num_links > 1:
        snr_rank = snr_rank / (num_links - 1)  # 归一化到[0,1]
    else:
        snr_rank = torch.zeros(1, device=device)
    
    # 4. 归一化度数（避免数值过大）
    max_degree = max(in_degree.max().item(), out_degree.max().item(), 1.0)
    in_degree_norm = in_degree / max_degree
    out_degree_norm = out_degree / max_degree
    
    # 5. 归一化干扰（使用log，因为干扰值可能很大）
    max_interference = max(in_interference.max().item(), out_interference.max().item(), 1.0)
    if max_interference > 0:
        in_interference_norm = torch.log10(in_interference + 1) / torch.log10(torch.tensor(max_interference + 1, device=device))
        out_interference_norm = torch.log10(out_interference + 1) / torch.log10(torch.tensor(max_interference + 1, device=device))
    else:
        in_interference_norm = torch.zeros_like(in_interference)
        out_interference_norm = torch.zeros_like(out_interference)
    
    # 6. 拼接特征：[原始SNR, 入度, 出度, 入干扰, 出干扰, SNR排名]
    enhanced_features = torch.stack([
        state.vertex_features[:, 0],  # 原始SNR
        in_degree_norm,
        out_degree_norm,
        in_interference_norm,
        out_interference_norm,
        snr_rank
    ], dim=1)
    
    return enhanced_features