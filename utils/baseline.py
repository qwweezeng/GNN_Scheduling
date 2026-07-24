import pickle as pkl

import torch
import numpy as np
from collections import defaultdict
from models.model import calculate_achievable_rate
import os
import sys
from types import SimpleNamespace

from baseline.fplinq_numpy import binarize_fplinq, optimize_fplinq
from baseline.spatial_dl_adapter import (
    project_3d_links_to_spatial_dl,
    run_spatial_dl_checkpoint_isolated,
)

# ========== 完整的 TensorFlow/CUDA 日志抑制 ==========
# 1. 抑制 TF_CPP 日志 (GPU 设备信息等)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
# 2. 禁用 oneDNN 优化器的日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 3. 禁用 XLA 编译日志
os.environ['TF_XLA_FLAGS'] = '--tf_xla_min_cluster_size=1'
# 4. 禁用 CUDA 编译日志
os.environ['CUDA_CACHE_DISABLE'] = '0'
# 5. 设置 absl 日志级别
os.environ['ABSL_LOG_LEVEL'] = 'ERROR'

# Silence absl and tensorflow logging
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow.python').setLevel(logging.ERROR)

# ---- Spatial DL baseline paths ----
SPATIAL_DL_ROOT = os.path.join(os.path.dirname(__file__), "..", "baseline", "Spatial_Deep_Learning_for_Wireless_Scheduling-master")
sys.path.append(os.path.join(SPATIAL_DL_ROOT, "Neural_Network_Model"))
sys.path.append(os.path.join(SPATIAL_DL_ROOT, "Utilities"))

# Import our own network generator for consistency
from utils.network_generate import NetworkState

def prune_state(state, keep_indices):
    """
    Build a pruned NetworkState that only keeps links in keep_indices
    and remaps link indices in edge_index / neighbor_map accordingly.
    """
    import numpy as np
    from utils.network_generate import NetworkState
    keep_indices = np.array(keep_indices, dtype=np.int64)
    mapping = {int(old): int(new) for new, old in enumerate(keep_indices)}

    links = state.links[keep_indices]
    vertex_features = state.vertex_features[keep_indices]

    # Filter edges where both endpoints are kept
    mask_edges = []
    for e_idx in range(state.edge_index.size(1)):
        u = int(state.edge_index[0, e_idx])
        v = int(state.edge_index[1, e_idx])
        if u in mapping and v in mapping:
            mask_edges.append(e_idx)
    if mask_edges:
        edge_index_old = state.edge_index[:, mask_edges]
        edge_weights = state.edge_weights[mask_edges]
        edge_index = torch.zeros_like(edge_index_old)
        for k in range(edge_index_old.size(1)):
            edge_index[0, k] = mapping[int(edge_index_old[0, k])]
            edge_index[1, k] = mapping[int(edge_index_old[1, k])]
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
        edge_weights = torch.empty(0, 1, dtype=torch.float32)

    # Rebuild neighbor_map
    neighbor_map = {}
    for e in range(edge_index.size(1)):
        u = int(edge_index[0, e])
        v = int(edge_index[1, e])
        neighbor_map.setdefault(v, []).append(u)

    pruned_state = NetworkState(
        links=links,
        vertex_features=vertex_features,
        edge_index=edge_index,
        edge_weights=edge_weights,
        large_scale=state.large_scale[keep_indices] if hasattr(state, "large_scale") else None,
        signal_power=state.signal_power,
        signal_power_large_scale=getattr(state, "signal_power_large_scale", state.signal_power),
        signal_power_quant=getattr(state, "signal_power_quant", None),
        noise_power=state.noise_power,
        bandwidth=state.bandwidth,
        snr_gap=getattr(state, 'snr_gap', 1.0),
        neighbor_map=neighbor_map,
        positions=state.positions,
        tx_power=getattr(state, 'tx_power', None),
        carrier_f=getattr(state, 'carrier_f', None),
        csi_mode=getattr(state, 'csi_mode', 'perfect'),
    )
    return pruned_state

def build_power_gain_matrix(state, use_fading=True):
    """
    Build received-power matrix for numerical baselines.

    Entry ``[receiver_link, transmitter_link]`` is already ``P|h|^2``.  The
    returned matrix must never be multiplied by transmit power again.
    """
    num_links = state.links.size(0)
    links = state.links.cpu().numpy()
    
    # ✅ 从state获取发射功率（线性单位，W），如果不存在则回退到默认22 dBm
    if hasattr(state, 'tx_power') and state.tx_power is not None:
        tx_power = state.tx_power
    else:
        # 默认值：10 dBm = 0.01 W
        tx_power = 10**(10/10) * 1e-3
    
    power_gain = torch.zeros(num_links, num_links, device=state.signal_power.device)
    for i in range(num_links):
        rx_i = links[i, 1]
        for j in range(num_links):
            tx_j = links[j, 0]
            if use_fading:
                # 使用小尺度衰落（实际接收功率 = tx_power * |h|^2）
                power_gain[i, j] = state.signal_power[tx_j, rx_i]
            else:
                # 使用大尺度路径损耗估算
                pos_tx_j = state.positions[tx_j]
                pos_rx_i = state.positions[rx_i]
                dist = torch.norm(pos_tx_j - pos_rx_i)
                
                # 计算参考路径损耗因子 lambda0
                if hasattr(state, "large_scale") and state.large_scale is not None and len(state.large_scale) > 0:
                    ref_idx = min(i, len(state.large_scale)-1)
                    ref_tx, ref_rx = links[ref_idx, 0], links[ref_idx, 1]
                    ref_dist = torch.norm(state.positions[ref_tx] - state.positions[ref_rx])
                    if ref_dist > 0 and state.large_scale[ref_idx] > 0:
                        lambda0 = (state.large_scale[ref_idx] ** 2) * (ref_dist ** 2.5)
                    else:
                        lambda0 = 1e-10
                else:
                    lambda0 = 1e-10
                
                if dist > 0:
                    power_gain[i, j] = tx_power * lambda0 * (dist ** (-2.5))
                else:
                    power_gain[i, j] = 0.0
    return power_gain, tx_power

def compute_sum_rate_from_allocs(allocs, power_gain, noise_power, snr_gap=1.0):
    """
    Compute sum-rate from allocations and power gain matrix.
    allocs: [B,N] 0/1
    power_gain: [B,N,N] or [N,N]
    noise_power: 噪声功率
    snr_gap: SNR gap (default 1.0 = 0 dB, no gap)
    
    注意：返回的是每Hz的速率（不乘以带宽），便于数值稳定和模型训练对比
    """
    allocs = np.asarray(allocs, dtype=np.float32)
    if power_gain.ndim == 2:
        power_gain = np.expand_dims(power_gain, 0)
    B, N = allocs.shape
    g = power_gain if power_gain.shape[0] == B else np.repeat(power_gain, B, axis=0)
    g_diag = np.diagonal(g, axis1=1, axis2=2)
    g_nondiag = g * (1 - np.eye(N))[None, :, :]
    SINR_num = allocs * g_diag
    SINR_den = np.matmul(g_nondiag, allocs[..., None]).squeeze(-1) + noise_power
    # 移除 epsilon，因为 noise_power 已经足够大，可以防止除以零
    # 之前的 1e-12 相对于 noise_power (~6e-14) 太大了，会导致 SINR 被低估
    SINR = SINR_num / SINR_den
    # 计算速率: rate = log2(1 + SINR / SNR_gap) [每Hz的速率，不乘以带宽]
    # 这样保持数值稳定，便于与我们模型的结果对比
    rates = np.log2(1 + SINR / snr_gap)
    return rates.sum(axis=1)

def fplinq_schedule(state, use_fading=True, iterations=100):
    """Return FPLinQ's binary action and continuous allocation scores."""
    if not np.isclose(float(getattr(state, "snr_gap", 1.0)), 1.0):
        raise ValueError("FPLinQ is validated only for the paper's 0 dB SNR gap")
    power_gain, _ = build_power_gain_matrix(state, use_fading=use_fading)
    power_gain = power_gain.cpu().numpy()
    num_links = int(state.links.size(0))
    allocation = optimize_fplinq(
        power_gain,
        float(state.noise_power),
        np.ones(num_links, dtype=np.float64),
        iterations=int(iterations),
    )
    schedule = binarize_fplinq(allocation)
    return (
        torch.tensor(schedule[:, None], dtype=torch.float32),
        np.asarray(allocation, dtype=np.float64),
    )


def eval_fplinq(states, use_fading=True, *, iterations=100, return_actions=False):
    """
    Run FPLinQ on full link set (no pruning). Returns list of sum-rates.
    
    修复说明：
    - bandwidth 和 snr_gap 从 state 读取，而不是从 general_para 读取
    - 这确保与我们方法的参数一致
    """
    sum_rates = []
    actions = []
    for state in states:
        pi_tensor, _ = fplinq_schedule(
            state, use_fading=use_fading, iterations=iterations
        )
        sr = calculate_achievable_rate(pi_tensor, state, binary=True).sum().item()
        sum_rates.append(sr)
        actions.append(pi_tensor)
    return (sum_rates, actions) if return_actions else sum_rates


def eval_greedy_spatial(states):
    """
    Greedy scheduling similar to Spatial DL's greedy baseline.
    Returns list of sum-rates.
    
    修复说明：
    - bandwidth 和 snr_gap 从 state 读取
    - 直接速率计算也使用带宽
    """
    sum_rates = []
    for state in states:
        power_gain, tx_power = build_power_gain_matrix(state, use_fading=True)
        power_gain = power_gain.cpu().numpy()
        num_links = int(state.links.size(0))

        # ✅ 从state获取通信参数
        snr_gap = getattr(state, 'snr_gap', 1.0)      # 默认1.0 (0 dB)

        # Use full power_gain (no truncation/padding) and perform greedy over all links
        gains_diagonal = np.diagonal(power_gain, axis1=0, axis2=1)[None, :]
        prop_weights = np.ones((1, num_links))

        # Greedy scheduling: iterate in reverse order, keep best allocation
        # 注意：这里使用 compute_sum_rate_from_allocs 是因为循环内部需要频繁计算
        # 最终结果会使用 calculate_achievable_rate 验证
        SNRS = gains_diagonal / state.noise_power
        # 直接速率排序使用每Hz的速率（不乘以带宽）
        direct_rates = np.log2(1 + SNRS / snr_gap)
        sorted_idx = np.argsort(prop_weights * direct_rates, axis=1)
        allocs = np.zeros((1, num_links))
        prev_wsr = np.zeros((1,))
        for j in range(num_links - 1, -1, -1):
            allocs[np.arange(1), sorted_idx[:, j]] = 1
            sr = compute_sum_rate_from_allocs(
                allocs=allocs[:, :num_links],
                power_gain=power_gain[:num_links, :num_links][None, :, :],
                noise_power=state.noise_power,
                snr_gap=snr_gap,
            )[0]
            improved = sr > prev_wsr
            allocs[np.arange(1), sorted_idx[:, j]] = improved.astype(float)
            prev_wsr = np.maximum(prev_wsr, sr)

        final_alloc = allocs[0, :num_links]
        
        # ✅ 使用我们的 calculate_achievable_rate 函数，确保速率计算一致
        pi_tensor = torch.tensor(final_alloc[:, None], dtype=torch.float32)
        sr_final = calculate_achievable_rate(pi_tensor, state, binary=True).sum().item()
        sum_rates.append(sr_final)
    return sum_rates


def eval_local_greedy(states, interference_threshold=1.0):
    """
    Local Greedy Scheduling: Select links in SNR order, skipping conflicts.
    Returns list of sum-rates.
    """
    sum_rates = []
    for state in states:
        pi = local_greedy(state, interference_threshold=interference_threshold)
        sr = calculate_achievable_rate(pi, state, binary=True).sum().item()
        sum_rates.append(sr)
    return sum_rates


def build_layouts_from_states(
    states,
    general_para,
    common_subproblem_indices=None,
    *,
    return_mappings=False,
):
    """
    Build layouts tensor (batch, N_pad, 4) expected by Spatial DL.
    No link is silently dropped. ``general_para.n_links`` is the within-batch
    padding length, not a checkpoint restriction.
    """
    N_pad = int(general_para.n_links)
    if N_pad < 1:
        raise ValueError("Spatial-DL padding length must be positive")
    layouts = np.zeros((len(states), N_pad, 4), dtype=np.float32)
    masks = np.zeros((len(states), N_pad), dtype=np.float32)
    link_counts = []
    pruned_states = []
    mappings = []

    if common_subproblem_indices is None:
        common_sets = [None] * len(states)
    elif len(states) == 1:
        value = common_subproblem_indices
        if (
            isinstance(value, (list, tuple))
            and len(value) == 1
            and (value[0] is None or np.asarray(value[0]).ndim > 0)
        ):
            common_sets = list(value)
        else:
            common_sets = [value]
    else:
        common_sets = list(common_subproblem_indices)
        if len(common_sets) != len(states):
            raise ValueError("one common-subproblem index set is required per state")

    for i, state in enumerate(states):
        layout, mask, mapping = project_3d_links_to_spatial_dl(
            state.positions.detach().cpu().numpy(),
            state.links.detach().cpu().numpy(),
            common_subproblem_indices=common_sets[i],
            target_link_count=N_pad,
        )
        layouts[i] = layout
        masks[i] = mask
        link_counts.append(int(mask.sum()))
        full_indices = np.arange(int(state.links.size(0)), dtype=np.int64)
        pruned_states.append(
            state
            if np.array_equal(mapping.selected_project_indices, full_indices)
            else prune_state(state, mapping.selected_project_indices)
        )
        mappings.append(mapping)

    result = (layouts, masks, link_counts, pruned_states)
    return result + (mappings,) if return_mappings else result


def eval_spatial_dl(
    states,
    *,
    common_subproblem_indices=None,
    seed=0,
    batch_size=4,
    return_actions=False,
    return_metadata=False,
):
    """
    Run Spatial DL baseline on a batch of states.
    Returns list of sum-rates (Mbps) computed with our rate function.
    """
    sum_rates = []
    all_metadata = []
    all_actions = []
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if common_subproblem_indices is None:
        common_sets = [None] * len(states)
    elif len(states) == 1:
        value = common_subproblem_indices
        if (
            isinstance(value, (list, tuple))
            and len(value) == 1
            and (value[0] is None or np.asarray(value[0]).ndim > 0)
        ):
            common_sets = list(value)
        else:
            common_sets = [value]
    else:
        common_sets = list(common_subproblem_indices)
        if len(common_sets) != len(states):
            raise ValueError("one common-subproblem index set is required per state")

    for start in range(0, len(states), batch_size):
        state_batch = states[start:start + batch_size]
        common_batch = common_sets[start:start + batch_size]
        selected_counts = [
            int(state.links.size(0)) if indices is None else len(indices)
            for state, indices in zip(state_batch, common_batch)
        ]
        general_para = SimpleNamespace(n_links=max(1, max(selected_counts)))
        layouts, masks, link_counts, pruned_states, mappings = build_layouts_from_states(
            state_batch,
            general_para,
            common_batch,
            return_mappings=True,
        )
        schedules = run_spatial_dl_checkpoint_isolated(
            layouts,
            masks,
            source_root=SPATIAL_DL_ROOT,
            seed=int(seed) + start,
            conda_env=os.environ.get(
                "SPATIAL_DL_CONDA_ENV", "gnn-spatial-dl-legacy"
            ),
        )
        for i, state in enumerate(pruned_states):
            num_links = link_counts[i]
            pi_hard = torch.tensor(
                schedules[i, :num_links], dtype=torch.float32
            ).unsqueeze(-1)
            rates = calculate_achievable_rate(
                pi_hard, state, binary=True, detach_hard=True
            )
            sum_rates.append(rates.sum().item())
            project_action = mappings[i].legacy_to_project(
                schedules[i], omitted_value=0.0
            )
            all_actions.append(
                torch.tensor(project_action[:, None], dtype=torch.float32)
            )
            item = mappings[i].metadata()
            item.update({
                "active_links": int(schedules[i, :num_links].sum()),
                "active_fraction": float(
                    schedules[i, :num_links].mean()
                ) if num_links else 0.0,
                "inference_seed": int(seed) + start,
            })
            all_metadata.append(item)
    if return_actions and return_metadata:
        return sum_rates, all_actions, all_metadata
    if return_actions:
        return sum_rates, all_actions
    if return_metadata:
        return sum_rates, all_metadata
    return sum_rates
'''
三种baseline：
1 random_activate(state, activation_prob=0.3) 每条链路以固定概率被随机激活
2 simple_greedy(state, k_ratio=0.3) 简化版贪心，直接选择SNR最大的K条链路
3 local_greedy(state, interference_threshold=1.0) LGS
'''

def random_activate(state, activation_prob=0.3):
    """
    每条链路以固定概率独立激活
    """
    num_vertices = state.num_vertices
    pi = torch.bernoulli(torch.ones(num_vertices) * activation_prob)
    return pi


def simple_greedy(state, k_ratio=0.3):
    num_vertices = state.num_vertices
    k = max(1, int(num_vertices * k_ratio))

    # 使用SNR作为utility
    utility_values = state.vertex_features[:, 0]

    # 选择utility最大的k条链路
    _, indices = torch.topk(utility_values, k)

    pi = torch.zeros(num_vertices)
    pi[indices] = 1.0

    return pi


def local_greedy(state, interference_threshold=0.3):
    """
    Local Greedy Scheduling: Select links in SNR order, skipping conflicts.
    
    注意：edge_weights 已被归一化到 [0, 1]，所以 interference_threshold 也应该在 [0, 1] 范围内
    - 阈值越高，越多链路被认为是"冲突"，激活的链路越少
    - 阈值越低，激活的链路越多
    默认 0.3 表示：干扰增益 > 0.3*max_weight 的链路对认为有冲突
    """
    num_vertices = state.num_vertices
    snr_values = state.vertex_features[:, 0]

    # 构建简化的无向冲突图（只保留强干扰边）
    conflict_graph = build_binary_conflict_graph(state, interference_threshold)

    # 按SNR降序排列
    sorted_indices = torch.argsort(snr_values, descending=True)

    pi = torch.zeros(num_vertices)
    scheduled = set()

    for idx in sorted_indices:
        idx = idx.item()

        # 检查是否与已调度链路冲突
        conflict = False
        for scheduled_idx in scheduled:
            if conflict_graph[idx, scheduled_idx]:
                conflict = True
                break

        if not conflict:
            pi[idx] = 1.0
            scheduled.add(idx)

    return pi


def build_binary_conflict_graph(state, interference_threshold=0.3):
    """
    构建二值化冲突图
    
    注意：edge_weights 已被归一化到 [0, 1]，interference_threshold 应该在 [0, 1] 范围内
    干扰超过阈值的边设为1，否则为0
    
    阈值选择建议：
    - 0.1: 只忽略很强干扰，激活较多链路
    - 0.3: 平衡选择（推荐）
    - 0.5: 保守选择，激活较少链路
    """
    num_vertices = state.num_vertices
    conflict_graph = torch.zeros(num_vertices, num_vertices, dtype=torch.bool)

    edge_index = state.edge_index
    edge_weights = state.edge_weights

    # Exact vectorization of the historical edge loop. This preserves the
    # strict ``> threshold`` rule and undirected symmetrization while avoiding
    # one Python ``.item()`` synchronization per graph edge.
    strong = edge_weights.reshape(-1) > float(interference_threshold)
    if strong.any():
        u = edge_index[0, strong].long()
        v = edge_index[1, strong].long()
        conflict_graph[u, v] = True
        conflict_graph[v, u] = True

    return conflict_graph

def eval_baseline(dataset, baseline_names=None):
    baseline_configs = {
        'random': {'activation_prob': 0.05},
        'simple_greedy': {'k_ratio': 0.1},
        'local_greedy': {'interference_threshold': 0.3},  # 归一化权重 [0,1]，0.3 是平衡选择
        'greedy_spatial': {},
        'spatial_dl': {},
        'fplinq': {}
    }

    # determine which baselines to run
    if baseline_names is None:
        run_list = list(baseline_configs.keys())
    else:
        run_list = [b for b in baseline_names if b in baseline_configs]

    results = {}

    for base_name in run_list:
        config = baseline_configs.get(base_name, {})
        total_util = 0.0
        all_util = []
        for state in dataset:
            if base_name == 'random':
                pi = random_activate(state, **config)
            elif base_name == 'simple_greedy':
                pi = simple_greedy(state, **config)
            elif base_name == 'local_greedy':
                interference_threshold = config.get('interference_threshold', 1.0)
                sr = eval_local_greedy([state], interference_threshold=interference_threshold)[0]
                total_util += sr
                continue
            elif base_name == 'spatial_dl':
                sr = eval_spatial_dl([state])[0]
                total_util += sr
                continue
            elif base_name == 'greedy_spatial':
                sr = eval_greedy_spatial([state])[0]
                total_util += sr
                continue
            elif base_name == 'fplinq':
                sr = eval_fplinq([state])[0]
                total_util += sr
                continue
            else:
                continue

            rates = calculate_achievable_rate(pi, state, binary=True)
            if hasattr(state, 'queue_states') and state.queue_states is not None:
                Q = state.queue_states
            else:
                Q = torch.ones(state.num_vertices)
            util = torch.sum(torch.round(pi) * Q * rates).item()
            total_util += util
        avg_util = total_util / len(dataset)
        results[base_name] = avg_util
    return results

if __name__ == '__main__':
    with open("./datasets/dataset_numUAV=10.pkl", "rb") as f:
        datasets = pkl.load(f)
    val_set = datasets.get('val_set')
    if val_set is None:
        print("Error: val_set not found in dataset.pkl")
        exit(1)
    results = eval_baseline(val_set)
    print(f"Random: {results['random']:.2f}| Simple: {results['simple_greedy']:.2f}| LGS: {results['local_greedy']:.2f}")
