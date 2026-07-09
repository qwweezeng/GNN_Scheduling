#!/usr/bin/env python3
"""
Queue-aware Scheduling Experiment (队列感知调度实验)

验证 GAT 调度器在多时隙队列动态下是否能保持网络稳定。

Usage:
    uv run python experiments/run_queue_experiment.py
    uv run python experiments/run_queue_experiment.py --arrival_rates 0.5,1.0,1.5 --num_slots 5000 --seed 42
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
import pickle as pkl
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network_generate import NetworkGenerator
from utils.io import generate_dataset
from models.model import GATScheduler
from train.trainer import Trainer
from simulator.queue_simulator import QueueSimulator


# ============================================================================
# 全局路径常量
# ============================================================================
# 标准模型库：按 (num_uav, queue/seed) 命名，方便直接定位
MODEL_LIBRARY_DIR = "./results/models"

# Link-level hotspot traffic model:
#   20% hotspot links carry 70% of the total network arrivals.
# These are fixed constants on purpose, so running the experiment does not need
# another long command-line option.
HOTSPOT_LINK_FRACTION = 0.20
HOTSPOT_TRAFFIC_FRACTION = 0.70


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_generator(cfg):
    height_range = cfg.get("HEIGHT_RANGE", [50, 150])
    if isinstance(height_range, list):
        height_range = tuple(height_range)
    return NetworkGenerator(
        num_uavs=cfg.get("NUM_UAV", 30),
        area_width=cfg.get("AREA_WIDTH", 1000),
        height_min=height_range[0],
        height_max=height_range[1],
        uav_min_distance=cfg.get("MIN_DISTANCE", 20),
        tx_power=cfg.get("TX_POWER", 22),
        Gt=cfg.get("G_T", 0),
        Gr=cfg.get("G_R", 0),
        noise_psd=cfg.get("NOISE_PSD", -169),
        bandwidth=cfg.get("BANDWIDTH", 5e6),
        carrier_f=cfg.get("CARRIER_FREQ", 2.4e9),
        snr_thres=cfg.get("SNR_THRES", 5),
    )


def build_model(cfg, raw_node_dim):
    """构建 GAT 调度器。

    Args:
        cfg: 配置字典
        raw_node_dim: 节点原始特征维度（1=仅SNR，2=SNR+队列）
    """
    return GATScheduler(
        raw_node_dim=raw_node_dim,
        mlp_hidden_dims=cfg.get("MLP_HIDDEN_DIMS", [32, 32]),
        embed_node_dim=cfg.get("NODE_EMBED_DIM", 32),
        edge_dim=cfg.get("EDGE_DIM", 1),
        gat_out_dim=cfg.get("GAT_OUT_DIM", 48),
        num_heads=cfg.get("NUM_HEADS", 4),
        T_iterations=cfg.get("NUM_GAT_LAYERS", 4),
        schedule_hidden_dim=cfg.get("SCHEDULE_HIDDEN", 32),
        dropout=cfg.get("DROPOUT", 0.0),
        use_power_control=False,
    )


# ============================================================================
# 数据集缓存管理
# ============================================================================
#
# 【目录设计】
# datasets/                              ← perfect CSI 数据集
#     dataset_numUAV=10.pkl
#     dataset_numUAV=20.pkl
#     ...
# datasets/large_scale/                  ← 大尺度衰落 CSI 数据集
#     dataset_numUAV=20_large_scale.pkl
#     ...
# datasets/quantized/                     ← 量化 CSI 数据集
#     dataset_numUAV=20_quantized_8bits.pkl
#     ...
# datasets/queue/                         ← 队列感知实验数据集（per-queue_experiment）
#     dataset_numUAV=10_linklevel.pkl
#     ...
#
# 【队列实验命名】
# run_queue_experiment.py 使用 _get_dataset_cache_path() 生成路径：
#   - use_queue=True  → datasets/queue/dataset_numUAV=<N>_linklevel.pkl
#   - use_queue=False → datasets/dataset_numUAV=<N>.pkl
#
def _get_dataset_cache_path(num_uav, use_queue):
    """
    生成数据集缓存路径。

    Args:
        num_uav: 无人机数量
        use_queue: 是否为队列实验
            True  → datasets/queue/dataset_numUAV=<N>_linklevel.pkl
            False → datasets/dataset_numUAV=<N>.pkl
    """
    if use_queue:
        return f"./datasets/queue/dataset_numUAV={num_uav}_linklevel.pkl"
    return f"./datasets/dataset_numUAV={num_uav}.pkl"


def _load_or_generate_dataset(net_gen, cfg, num_uav, use_queue, split="train",
                              queue_range=(0, 25)):
    """
    加载或生成指定类型的数据集。

    【数据集 split 的实际用途】
    - train/val/test 都只用于单时隙快照评估（trainer.evaluate）
    - 多时隙仿真（QueueSimulator）完全独立，不依赖这些数据集
    因此所有 split 都统一用 uniform[queue_range[0], queue_range[1]) 覆盖宽范围。
    真正的泊松到达在 simulation_phase() 的 QueueSimulator 中体现。

    【缓存策略】
    - 按 (num_uav, use_queue, split) 三元组 + queue_range 标识
    - 非队列数据存在 ./datasets/
    - 队列数据存在  ./datasets/queue/

    Args:
        net_gen: NetworkGenerator 实例
        cfg: 配置字典
        num_uav: 无人机数量
        use_queue: 是否为队列实验
        split: 数据集类型 ("train", "val", "test")
        queue_range: 队列长度范围 (min, max)
    """
    cache_path = _get_dataset_cache_path(num_uav, use_queue)
    key_map    = {"train": "train_set", "val": "val_set", "test": "test_set"}
    size_map  = {"train": "TRAIN_SIZE", "val": "VAL_SIZE", "test": "TEST_SIZE"}

    dataset = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pkl.load(f)
            full = cached.get(key_map[split])
            if full is not None:
                req_size = cfg.get(size_map[split], 500)
                dataset = full[:req_size]
                if len(dataset) > 0:
                    cached_flag = getattr(dataset[0], 'include_queue_feature', None)
                    expected_flag = use_queue
                    queue_shape_ok = True
                    if use_queue:
                        q = getattr(dataset[0], 'queue_states', None)
                        queue_shape_ok = (
                            q is not None
                            and q.numel() == dataset[0].num_vertices
                            and dataset[0].vertex_features.size(1) >= 2
                        )
                    if cached_flag != expected_flag or not queue_shape_ok:
                        print(f"  [Warning] Cached {split} queue_flag={cached_flag}, "
                              f"expected {expected_flag}, queue_shape_ok={queue_shape_ok}. "
                              "Will regenerate.")
                        dataset = None
                    else:
                        print(f"  [Cache Hit] {split}: {len(dataset)} samples from {cache_path}")
        except Exception as e:
            print(f"  [Warning] Failed to load {split} ({e}), regenerating...")
            dataset = None

    if dataset is None:
        size = cfg.get(size_map[split], 500)
        print(f"  [Generating] {split}: {size} samples (uniform in {queue_range})...")
        dataset = generate_dataset(
            size, net_gen,
            batch_size=cfg.get("GEN_BATCH_SIZE", 50),
            csi_mode=cfg.get("CSI_MODE", "perfect"),
            include_queue_feature=use_queue,
            queue_mode="random",
            queue_range=queue_range,
        )
        # 写入缓存（写入 train split 时顺便写完整缓存）
        if split == "train":
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                val_data   = _load_or_generate_dataset(net_gen, cfg, num_uav, use_queue, "val",
                                                      queue_range=queue_range)
                test_data  = _load_or_generate_dataset(net_gen, cfg, num_uav, use_queue, "test",
                                                      queue_range=queue_range)
                with open(cache_path, "wb") as f:
                    pkl.dump({
                        "train_set": dataset,
                        "val_set":   val_data,
                        "test_set":  test_data,
                    }, f)
                print(f"  [Cache Saved] All splits → {cache_path}")
            except Exception as e:
                print(f"  [Warning] Failed to save cache: {e}")

    return dataset


# ============================================================================
# 模型库管理
# ============================================================================
#
# 【目录设计】
# results/models/                         ← 标准模型库
# ├── gat_numUAV=10.pth                 ← 非队列（Loss=-Σ R_i）
# ├── gat_numUAV=20.pth
# ├── gat_numUAV=30.pth
# ├── gat_queue_numUAV=10.pth           ← 队列（Loss=-Σ Q_i×R_i）
# ├── gat_queue_numUAV=20.pth
# └── gat_queue_numUAV=30.pth
#
# 【命名设计】直接以 num_uav 定位：gat[<_queue>]_numUAV=<N>.pth
#   - 无后缀：纯吞吐量最大化（GAT baseline）
#   - _queue 后缀：队列感知调度（GAT）
#
def _get_model_path(num_uav, use_queue):
    """
    生成标准模型库路径。

    Args:
        num_uav: 无人机数量
        use_queue: 是否为队列模型
            True  → gat_queue_numUAV=<N>.pth
            False → gat_numUAV=<N>.pth
    """
    os.makedirs(MODEL_LIBRARY_DIR, exist_ok=True)
    prefix = "gat_queue" if use_queue else "gat"
    return os.path.join(MODEL_LIBRARY_DIR, f"{prefix}_numUAV={num_uav}.pth")


def _train_model(net_gen, cfg, num_uav, raw_node_dim, use_queue, model_name, loss_label,
                 queue_range=(0, 25)):
    """
    训练单个 GAT 模型（支持从标准模型库加载已有模型）。

    【模型加载策略】
    优先从标准模型库 results/models/ 寻找已有模型：
      - 非队列：gat_numUAV=<N>.pth
      - 队列：gat_queue_numUAV=<N>.pth
    如果找到则直接加载；找不到则重新训练并保存到标准模型库。

    【与 bin/run_experiment.py 的关系】
    两个脚本共用同一套模型库：
      - bin/run_experiment.py 训练非队列模型（raw_node_dim=1）
      - run_queue_experiment.py 训练队列模型（raw_node_dim=2）
    因此在同一 num_uav 下，两者的模型可以互相复用（用于 GAT vs GAT-Queue 对比）

    Args:
        net_gen: NetworkGenerator
        cfg: 配置字典
        num_uav: 无人机数量（用于模型库路径）
        raw_node_dim: 节点特征维度（1=SNR，2=SNR+队列）
        use_queue: 是否启用队列感知 Loss
        model_name: 模型标识（用于打印）
        loss_label: Loss 描述字符串（用于打印）
        queue_range: 队列长度范围 (min, max)

    Returns:
        model: 训练好的或加载的 GATScheduler
        result: 训练结果字典（包含 loaded 标记）
    """
    model_path = _get_model_path(num_uav, use_queue)

    # 1. 尝试从标准模型库加载
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(model_path):
        print(f"  [Load] {model_name} from {model_path}")
        model = build_model(cfg, raw_node_dim)
        model.load_state_dict(torch.load(model_path, map_location=str(device)))
        model = model.to(device)
        return model, {"loaded": True, "path": model_path}

    # 2. 生成/加载数据集
    print(f"\n  ┌─── [{model_name}] Dataset (use_queue={use_queue}) ───────────────────────")
    train_set = _load_or_generate_dataset(net_gen, cfg, num_uav, use_queue, "train",
                                        queue_range=queue_range)
    val_set   = _load_or_generate_dataset(net_gen, cfg, num_uav, use_queue, "val",
                                        queue_range=queue_range)
    test_set  = _load_or_generate_dataset(net_gen, cfg, num_uav, use_queue, "test",
                                        queue_range=queue_range)
    print(f"  └── Dataset done")

    actual_dim = train_set[0].vertex_features.size(1)
    assert actual_dim == raw_node_dim, \
        f"Feature dim mismatch! Model expects {raw_node_dim}, data has {actual_dim}"
    print(f"  [{model_name}] Feature dim: {actual_dim} ✓")

    val_counts = [int(s.num_vertices) for s in val_set]
    print(f"  [{model_name}] Val link counts — avg: {np.mean(val_counts):.1f}, median: {np.median(val_counts):.1f}")

    # 3. 构建模型
    print(f"\n  ┌─── [{model_name}] Build Model ─────────────────────────────────────")
    model = build_model(cfg, raw_node_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  [{model_name}] Device: {device}")

    trainer = Trainer(
        model=model, optimizer=optimizer, device=device,
        log_dir=cfg.get("LOG_DIR", "./logs"), use_queue=use_queue,
        bandwidth=cfg.get("BANDWIDTH", 5e6),
        slot_duration=cfg.get("SLOT_DURATION", 0.001),
        packet_size=cfg.get("PACKET_SIZE", 12000),
    )
    print(f"  └── Model built")

    # 4. 训练
    print(f"\n  ┌─── [{model_name}] Training ────────────────────────────────────────")
    print(f"  [{model_name}] Loss = {loss_label}, raw_node_dim={raw_node_dim}")
    print(f"  [{model_name}] Epochs={cfg['EPOCHS']}, train={len(train_set)}, val={len(val_set)}")

    train_result = trainer.train(
        num_epochs=cfg["EPOCHS"],
        train_dataset=train_set,
        val_dataset=val_set,
        eval_interval=cfg.get("EVAL_INTERVAL", 5),
        lr_scheduler=scheduler, use_queue=use_queue,
    )
    print(f"  └── Training done")

    # 5. 测试
    print(f"\n  ┌─── [{model_name}] Test ────────────────────────────────────────────")
    print(f"  [{model_name}] Test ({len(test_set)} samples)...")
    test_result = trainer.evaluate(model, test_set, use_queue=use_queue)
    print(f"  [{model_name}] Test utility: {test_result:.4f}")
    print(f"  └── Test done")

    # 6. 保存到标准模型库
    torch.save(model.state_dict(), model_path)
    print(f"  [Save] {model_name} → {model_path}")

    return model, train_result


# ============================================================================
# Baseline 调度器
# ============================================================================
class RandomScheduler:
    def __init__(self, activation_prob=0.3):
        self.activation_prob = activation_prob

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, x, edge_index, edge_weights, use_ste=False):
        pi = (torch.rand(x.size(0), 1) < self.activation_prob).float()
        return (pi, None)


class GreedyScheduler:
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, x, edge_index, edge_weights, use_ste=False):
        snr = x[:, 0]
        pi = torch.zeros(snr.size(0), 1)
        if snr.numel() > 0:
            pi[snr.argmax()] = 1.0
        return (pi, None)


# ============================================================================
# 仿真阶段
# ============================================================================
def _infer_model_node_dim(model):
    """
    从模型权重推断节点特征维度。

    model.node_embed.mlp[0] 是第一层 Linear，权重 shape = [hidden_dim, raw_node_dim]
    因此 weight.shape[1] 就是 raw_node_dim。
    """
    try:
        return model.node_embed.mlp[0].weight.shape[1]
    except Exception:
        # fallback: 假设 1（SNR only）
        return 1


def _set_simulation_seed(seed):
    """设置仿真随机种子，不改变训练 seed 的语义。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pregenerate_arrival_realizations(num_slots, num_links, arrival_lambda,
                                      seed, hotspot_seed=None):
    """
    预生成业务到达过程，供同一 Monte Carlo seed 下所有调度器共享。

    arrival_lambda 现在表示全网总到达率 Lambda（packets/slot）。
    默认采用 link-level hotspot 模型：
      - 20% 热点链路承担 70% 总业务；
      - 80% 普通链路承担 30% 总业务；
      - sum_i lambda_i = Lambda。

    hotspot_seed 控制热点链路集合。仿真中同一个 MC seed 下，不同 Lambda
    复用同一组热点链路，只改变总业务强度，便于做负载扫描。
    """
    total_lambda = float(arrival_lambda)
    if total_lambda < 0:
        raise ValueError(f"arrival_lambda must be non-negative, got {arrival_lambda}.")

    if num_links <= 0:
        empty = torch.zeros((num_slots, 0), dtype=torch.float32)
        return empty, {
            'arrival_profile': 'link_hotspot_total',
            'arrival_total_lambda': total_lambda,
            'num_links': 0,
            'num_hotspot_links': 0,
            'num_normal_links': 0,
            'hotspot_lambda_per_link': 0.0,
            'normal_lambda_per_link': 0.0,
            'hotspot_indices': [],
        }

    hotspot_rng = np.random.default_rng(seed if hotspot_seed is None else hotspot_seed)
    num_hotspot = int(np.ceil(num_links * HOTSPOT_LINK_FRACTION))
    num_hotspot = min(max(1, num_hotspot), num_links)
    num_normal = num_links - num_hotspot
    hotspot_indices = np.sort(
        hotspot_rng.choice(num_links, size=num_hotspot, replace=False)
    )

    hotspot_traffic = HOTSPOT_TRAFFIC_FRACTION if num_normal > 0 else 1.0
    normal_traffic = 1.0 - hotspot_traffic

    link_lambdas = np.zeros(num_links, dtype=np.float64)
    hotspot_lambda = total_lambda * hotspot_traffic / num_hotspot
    normal_lambda = total_lambda * normal_traffic / num_normal if num_normal > 0 else 0.0
    link_lambdas[:] = normal_lambda
    link_lambdas[hotspot_indices] = hotspot_lambda

    rng = np.random.default_rng(seed)
    arrivals = rng.poisson(
        lam=link_lambdas[None, :],
        size=(num_slots, num_links),
    ).astype(np.float32)
    profile = {
        'arrival_profile': 'link_hotspot_total',
        'arrival_total_lambda': total_lambda,
        'hotspot_link_fraction': HOTSPOT_LINK_FRACTION,
        'hotspot_traffic_fraction': hotspot_traffic,
        'num_links': int(num_links),
        'num_hotspot_links': int(num_hotspot),
        'num_normal_links': int(num_normal),
        'hotspot_lambda_per_link': float(hotspot_lambda),
        'normal_lambda_per_link': float(normal_lambda),
        'hotspot_indices': hotspot_indices.astype(int).tolist(),
    }
    return torch.from_numpy(arrivals), profile


def _add_hotspot_group_metrics(metrics, hotspot_indices, packet_size):
    """补充热点链路/普通链路的平均队列积压，便于观察 QoS 差异。"""
    queue_trajectory = metrics.get('queue_trajectory')
    if queue_trajectory is None:
        return

    q = np.asarray(queue_trajectory, dtype=float)
    if q.ndim != 2 or q.shape[1] == 0:
        return

    mask = np.zeros(q.shape[1], dtype=bool)
    valid_hotspots = [idx for idx in hotspot_indices if 0 <= idx < q.shape[1]]
    if not valid_hotspots:
        return

    mask[valid_hotspots] = True
    hotspot_q = q[:, mask]
    normal_q = q[:, ~mask]

    metrics['hotspot_avg_queue'] = float(hotspot_q.mean())
    metrics['hotspot_avg_queue_mbit'] = float(hotspot_q.mean() * packet_size / 1e6)
    if normal_q.size > 0:
        metrics['normal_avg_queue'] = float(normal_q.mean())
        metrics['normal_avg_queue_mbit'] = float(normal_q.mean() * packet_size / 1e6)
    else:
        metrics['normal_avg_queue'] = None
        metrics['normal_avg_queue_mbit'] = None


_SCALAR_METRIC_KEYS = [
    'avg_queue',
    'final_avg_queue',
    'max_queue',
    'final_max_queue',
    'long_term_avg_queue',
    'long_term_max_queue',
    'average_queue_backlog',
    'average_queue_backlog_mbit',
    'average_total_queue_backlog',
    'average_total_queue_backlog_mbit',
    'average_delay_slots',
    'average_delay_ms',
    'total_arrivals',
    'total_departures',
    'arrival_rate_pps',
    'arrival_total_rate_pps',
    'throughput_ratio',
    'throughput_mbps',
    'departure_rate_pps',
    'departure_total_rate_pps',
    'queue_overflow_prob',
    'arrival_total_lambda',
    'num_links',
    'num_hotspot_links',
    'num_normal_links',
    'hotspot_lambda_per_link',
    'normal_lambda_per_link',
    'hotspot_avg_queue',
    'hotspot_avg_queue_mbit',
    'normal_avg_queue',
    'normal_avg_queue_mbit',
]

_TRAJECTORY_METRIC_KEYS = [
    'avg_queue_trajectory',
    'max_queue_trajectory',
    'queue_backlog_mbit_trajectory',
    'throughput_trajectory',
    'arrivals_per_slot',
    'departures_per_slot',
    'utility_trajectory',
]


def _ci95(values):
    """返回 Monte Carlo 均值的 95% 置信区间半宽。"""
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(values.size))


def _aggregate_mc_metrics(metrics_list, mc_seeds):
    """
    对同一算法、同一 λ 下的多组 Monte Carlo 结果做均值汇总。

    注意：不汇总完整的 queue_trajectory（二位矩阵太大），只汇总绘图需要的一维轨迹。
    """
    if not metrics_list:
        return {}

    summary = {
        'num_mc_runs': len(metrics_list),
        'mc_seeds': list(mc_seeds),
    }

    for key in _SCALAR_METRIC_KEYS:
        vals = []
        for m in metrics_list:
            if key not in m or m[key] is None:
                continue
            try:
                val = float(m[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                vals.append(val)
        if not vals:
            summary[key] = None
            summary[f'{key}_std'] = None
            summary[f'{key}_ci95'] = None
            continue
        vals = np.asarray(vals, dtype=float)
        summary[key] = float(vals.mean())
        summary[f'{key}_std'] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        summary[f'{key}_ci95'] = _ci95(vals)

    stable_vals = [
        float(m['is_stable']) for m in metrics_list
        if m.get('is_stable') is not None
    ]
    if stable_vals:
        summary['stable_fraction'] = float(np.mean(stable_vals))
        summary['is_stable'] = summary['stable_fraction'] >= 0.5
    else:
        summary['stable_fraction'] = None
        summary['is_stable'] = None

    for key in _TRAJECTORY_METRIC_KEYS:
        arrays = [np.asarray(m[key], dtype=float) for m in metrics_list if key in m]
        if not arrays:
            continue
        first_shape = arrays[0].shape
        if any(a.shape != first_shape for a in arrays):
            continue
        stack = np.stack(arrays, axis=0)
        summary[key] = stack.mean(axis=0)
        if stack.shape[0] > 1:
            summary[f'{key}_ci95'] = 1.96 * stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        else:
            summary[f'{key}_ci95'] = np.zeros_like(summary[key])

    return summary


def _compact_metrics_for_json(metrics):
    """保存每轮 Monte Carlo 明细时去掉全链路二维队列轨迹，避免 JSON 过大。"""
    compact = {}
    for key, value in metrics.items():
        if key == 'queue_trajectory':
            continue
        compact[key] = value
    return compact


def run_single_simulator(net_gen, model, arrival_lambda, cfg, num_slots, device='cpu',
                         channel_realizations=None, arrival_realizations=None,
                         slots_per_block=None):
    """运行单个调度器的多时隙仿真。"""
    model_node_dim = _infer_model_node_dim(model)
    include_queue_feature = (model_node_dim > 1)

    sim = QueueSimulator(
        network_generator=net_gen, model=model, device=device,
        include_queue_feature=include_queue_feature,
    )
    sim.configure(
        arrival_lambda=arrival_lambda,
        slot_duration=cfg.get("SLOT_DURATION", 0.001),
        packet_size=cfg.get("PACKET_SIZE", 12000),
        slots_per_block=slots_per_block,
    )
    return sim.run(
        num_slots=num_slots,
        csi_mode=cfg.get("CSI_MODE", "perfect"),
        verbose=False, log_interval=500,
        channel_realizations=channel_realizations,
        arrival_realizations=arrival_realizations,
    )


def _pregenerate_channel_realizations(net_gen, num_slots, cfg, slots_per_block=None):
    """
    预生成信道块列表，供所有调度器共享。
    确保每个到达率下所有方法使用完全相同的信道实现（公平比较）。
    """
    from utils.network_generate import NetworkState
    import copy
    
    # 用一个 dummy simulator 来生成信道
    dummy_sim = QueueSimulator(
        network_generator=net_gen, model=None, device='cpu',
        include_queue_feature=False,
    )
    dummy_sim.configure(
        arrival_lambda=0.0,
        slot_duration=cfg.get("SLOT_DURATION", 0.001),
        packet_size=cfg.get("PACKET_SIZE", 12000),
        slots_per_block=slots_per_block,
    )
    
    if slots_per_block is None or slots_per_block <= 0:
        slots_per_block = num_slots
    num_blocks = (num_slots + slots_per_block - 1) // slots_per_block
    
    realizations = []
    for b in range(num_blocks):
        # 生成第 b 个信道块（所有调度器共享）。
        # 队列特征由 QueueSimulator 在每个时隙用真实 link-level Q(t) 写入。
        state = net_gen.generate_network(
            csi_mode=cfg.get("CSI_MODE", "perfect"),
            q_bits=cfg.get("Q_BITS", 8),
            include_queue_feature=False,
        )
        realizations.append(state)
    
    return realizations


def simulation_phase(net_gen, models, cfg, args):
    """
    运行多时隙队列仿真，对比所有 baseline。

    Args:
        net_gen: NetworkGenerator
        models: dict，key=方法标识，value=(model, display_name)
        cfg: 配置字典
        args: 命令行参数
    """
    print("\n" + "=" * 65)
    print("Phase 2: Multi-slot Queue Simulation")
    print("=" * 65)
    print(f"  Simulation slots: {args.num_slots}")
    print(f"  Arrival rates  : {args.arrival_rates} total packets/slot")
    print(f"  Arrival model  : link-level hotspot "
          f"({HOTSPOT_LINK_FRACTION:.0%} links carry "
          f"{HOTSPOT_TRAFFIC_FRACTION:.0%} traffic)")
    print(f"  Monte Carlo    : {len(args.mc_seeds)} run(s), seeds={args.mc_seeds}")
    print(f"  Queue init     : Q=0")

    # 获取 device
    first_model = next(m for m, _ in models.values())
    if hasattr(first_model, 'parameters'):
        device = next(first_model.parameters()).device
        for m, _ in models.values():
            m.eval()
    else:
        device = 'cpu'

    results = {}
    mc_details = {}

    for lam_idx, lam in enumerate(args.arrival_rates):
        print(f"\n{'─' * 65}")
        print(f"  λ = {lam} total packets/slot")
        print(f"{'─' * 65}")

        per_method_runs = {key: [] for key in models}
        slots_per_block = getattr(args, 'slots_per_block', None)

        for run_idx, mc_seed in enumerate(args.mc_seeds, start=1):
            print(f"  [MC {run_idx}/{len(args.mc_seeds)}] seed={mc_seed}")
            _set_simulation_seed(mc_seed)

            # 同一个 MC seed 内，所有调度器共享信道块和业务到达 trace。
            channel_realizations = _pregenerate_channel_realizations(
                net_gen, args.num_slots, cfg, slots_per_block=slots_per_block
            )
            num_links = channel_realizations[0].num_vertices
            arrival_realizations, arrival_profile = _pregenerate_arrival_realizations(
                args.num_slots, num_links, lam,
                seed=mc_seed + 1000003 * lam_idx,
                hotspot_seed=mc_seed,
            )
            print(f"    [Fair] shared channel_blocks={len(channel_realizations)}, "
                  f"shared arrivals={tuple(arrival_realizations.shape)}")
            print(f"    [Arrival] hotspot_links={arrival_profile['num_hotspot_links']}/"
                  f"{arrival_profile['num_links']}, "
                  f"lambda_hot={arrival_profile['hotspot_lambda_per_link']:.4f}, "
                  f"lambda_normal={arrival_profile['normal_lambda_per_link']:.4f}")

            for key, (model, name) in models.items():
                metrics = run_single_simulator(
                    net_gen, model, lam, cfg, args.num_slots, device=str(device),
                    channel_realizations=channel_realizations,
                    arrival_realizations=arrival_realizations,
                    slots_per_block=slots_per_block,
                )
                metrics['mc_seed'] = mc_seed
                metrics['arrival_seed'] = mc_seed + 1000003 * lam_idx
                metrics['lambda'] = lam
                metrics.update(arrival_profile)
                _add_hotspot_group_metrics(
                    metrics,
                    arrival_profile['hotspot_indices'],
                    cfg.get("PACKET_SIZE", 12000),
                )
                per_method_runs[key].append(metrics)
                mc_details[f"{key}_{lam}_seed{mc_seed}"] = _compact_metrics_for_json(metrics)

                stability = "✓" if metrics.get('is_stable') is True else (
                    "✗" if metrics.get('is_stable') is False else "?")
                print(f"    [{name}] "
                      f"FinalQ={metrics['final_avg_queue']:>7.2f}  "
                      f"LTAvgQ={metrics['long_term_avg_queue']:>7.2f}  "
                      f"Throughput={metrics['throughput_ratio']:>6.4f}  "
                      f"Stable={stability}  "
                      f"{metrics['throughput_mbps']:>6.2f} Mbps")

        print("  [MC Summary]")
        for key, (model, name) in models.items():
            summary = _aggregate_mc_metrics(per_method_runs[key], args.mc_seeds)
            results[f"{key}_{lam}"] = summary

            stability = "✓" if summary.get('is_stable') is True else (
                "✗" if summary.get('is_stable') is False else "?")
            print(f"    [{name}] "
                  f"FinalQ={summary['final_avg_queue']:>7.2f}±{summary['final_avg_queue_ci95']:<6.2f}  "
                  f"LTAvgQ={summary['long_term_avg_queue']:>7.2f}±{summary['long_term_avg_queue_ci95']:<6.2f}  "
                  f"Throughput={summary['throughput_ratio']:>6.4f}±{summary['throughput_ratio_ci95']:<6.4f}  "
                  f"Stable={stability}")

    return results, mc_details


# ============================================================================
# 可视化：队列长度和吞吐量随时隙变化
# ============================================================================
def _plot_trajectories(sim_results, arrival_rates, results_dir):
    """
    绘制各调度算法在每个到达率下的队列长度和吞吐量轨迹。

    参考 Q-CSMA 论文的可视化方式：
    - 上图：平均队列长度 随时隙变化
    - 下图：累积吞吐量比 随时隙变化
    """
    all_keys = sorted(set(k.rsplit("_", 1)[0] for k in sim_results))
    # 配色方案
    color_map = {
        'GAT':      '#2196F3',  # 蓝色
        'GAT-base': '#FF9800',  # 橙色
        'Random':   '#9E9E9E',  # 灰色
        'Greedy':   '#E91E63',  # 粉红
    }

    for lam in arrival_rates:
        mc_runs = 1
        for key in all_keys:
            m = sim_results.get(f"{key}_{lam}")
            if m is not None:
                mc_runs = m.get('num_mc_runs', 1)
                break

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Queue & Throughput Trajectories  (λ = {lam} packets/slot, MC={mc_runs})",
                     fontsize=14, fontweight='bold')

        for key in all_keys:
            m = sim_results.get(f"{key}_{lam}")
            if m is None:
                continue
            color = color_map.get(key, '#607D8B')
            label = key

            # 上图：平均队列长度
            ax = axes[0]
            q_traj = m.get('avg_queue_trajectory')
            if q_traj is not None:
                q_traj = np.asarray(q_traj, dtype=float)
                ax.plot(q_traj, color=color, label=label, linewidth=1.5, alpha=0.85)
                q_ci = m.get('avg_queue_trajectory_ci95')
                if q_ci is not None and mc_runs > 1:
                    q_ci = np.asarray(q_ci, dtype=float)
                    ax.fill_between(
                        np.arange(len(q_traj)),
                        q_traj - q_ci,
                        q_traj + q_ci,
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                    )

            # 中图：每时隙总发送量 (departures per slot)
            ax = axes[1]
            dep_traj = m.get('departures_per_slot')
            if dep_traj is not None:
                dep_traj = np.asarray(dep_traj, dtype=float)
                # 画滑动平均，窗口=20，更容易观察趋势
                window = min(20, len(dep_traj) // 5)
                if window > 1:
                    kernel = np.ones(window) / window
                    smoothed = np.convolve(dep_traj, kernel, mode='valid')
                    x_smooth = np.arange(window - 1, len(dep_traj))
                    ax.plot(x_smooth, smoothed, color=color, label=label, linewidth=1.5, alpha=0.85)
                else:
                    ax.plot(dep_traj, color=color, label=label, linewidth=1.0, alpha=0.5)

            # 下图：累积吞吐量比
            ax = axes[2]
            tp_traj = m.get('throughput_trajectory')
            if tp_traj is not None:
                tp_traj = np.asarray(tp_traj, dtype=float)
                ax.plot(tp_traj, color=color, label=label, linewidth=1.5, alpha=0.85)
                tp_ci = m.get('throughput_trajectory_ci95')
                if tp_ci is not None and mc_runs > 1:
                    tp_ci = np.asarray(tp_ci, dtype=float)
                    ax.fill_between(
                        np.arange(len(tp_traj)),
                        tp_traj - tp_ci,
                        tp_traj + tp_ci,
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                    )

        # 上图装饰
        axes[0].set_ylabel("Avg Queue Length (packets)", fontsize=11)
        axes[0].legend(loc='upper left', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title("Average Queue Length per Slot", fontsize=10)

        # 中图装饰
        axes[1].set_ylabel("Departures (pkts/slot)", fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("Total Departures per Slot (20-slot Moving Average)", fontsize=10)
        axes[1].legend(loc='upper left', fontsize=9)

        # 下图装饰
        axes[2].set_ylabel("Cumulative Throughput Ratio", fontsize=11)
        axes[2].set_xlabel("Slot t", fontsize=11)
        axes[2].legend(loc='lower right', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_title("Cumulative Throughput Ratio (Departures / Arrivals)", fontsize=10)

        plt.tight_layout()
        fig_path = os.path.join(results_dir, f"trajectory_lambda{lam}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Plot] Trajectory saved: {fig_path}")

    # 综合对比图：所有到达率下的最终平均队列对比
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(arrival_rates))
    width = 0.2
    for i, key in enumerate(all_keys):
        vals = []
        yerr = []
        for lam in arrival_rates:
            m = sim_results.get(f"{key}_{lam}")
            vals.append(m.get('final_avg_queue', m.get('avg_queue')) if m else 0)
            yerr.append(m.get('final_avg_queue_ci95', m.get('avg_queue_ci95', 0)) if m else 0)
        ax2.bar(
            x + i * width, vals, width,
            yerr=yerr if any(v > 0 for v in yerr) else None,
            capsize=3,
            label=key,
            color=color_map.get(key, '#607D8B'),
            alpha=0.85,
        )
    ax2.set_xlabel("Total Arrival Rate Λ (packets/slot)", fontsize=11)
    ax2.set_ylabel("Final Avg Queue Length (packets)", fontsize=11)
    ax2.set_title("Average Queue Length vs Arrival Rate", fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * (len(all_keys) - 1) / 2)
    ax2.set_xticklabels([f"λ={lam}" for lam in arrival_rates])
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    fig2_path = os.path.join(results_dir, "queue_comparison_bar.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"[Plot] Comparison bar saved: {fig2_path}")

    # MAB-style summary: backlog and delay versus traffic arrival rate.
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4.5))
    for key in all_keys:
        backlog_vals = []
        backlog_errs = []
        delay_vals = []
        delay_errs = []
        valid_lams = []

        for lam in arrival_rates:
            m = sim_results.get(f"{key}_{lam}")
            if not m:
                continue
            valid_lams.append(lam)
            backlog_val = m.get('average_queue_backlog_mbit')
            backlog_err = m.get('average_queue_backlog_mbit_ci95')
            delay_val = m.get('average_delay_ms')
            delay_err = m.get('average_delay_ms_ci95')
            backlog_vals.append(np.nan if backlog_val is None else backlog_val)
            backlog_errs.append(0.0 if backlog_err is None else backlog_err)
            delay_vals.append(np.nan if delay_val is None else delay_val)
            delay_errs.append(0.0 if delay_err is None else delay_err)

        if not valid_lams:
            continue
        color = color_map.get(key, '#607D8B')
        axes3[0].errorbar(
            valid_lams, backlog_vals,
            yerr=backlog_errs if any(v > 0 for v in backlog_errs) else None,
            marker='o', linewidth=1.8, capsize=3, label=key, color=color,
        )
        axes3[1].errorbar(
            valid_lams, delay_vals,
            yerr=delay_errs if any(v > 0 for v in delay_errs) else None,
            marker='o', linewidth=1.8, capsize=3, label=key, color=color,
        )

    axes3[0].set_xlabel("Total Arrival Rate Λ (packets/slot)", fontsize=11)
    axes3[0].set_ylabel("Average Queue Backlog (Mbit/link)", fontsize=11)
    axes3[0].set_title("Average Queue Backlog vs Arrival Rate", fontsize=10)
    axes3[0].grid(True, alpha=0.3)
    axes3[0].legend(fontsize=9)

    axes3[1].set_xlabel("Total Arrival Rate Λ (packets/slot)", fontsize=11)
    axes3[1].set_ylabel("Average Delay (ms)", fontsize=11)
    axes3[1].set_title("Average Delay vs Arrival Rate", fontsize=10)
    axes3[1].grid(True, alpha=0.3)
    axes3[1].legend(fontsize=9)

    plt.tight_layout()
    fig3_path = os.path.join(results_dir, "queue_backlog_delay_vs_lambda.png")
    fig3.savefig(fig3_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"[Plot] Backlog/delay summary saved: {fig3_path}")


def _print_summary_table(sim_results, arrival_rates):
    """打印结果汇总表。"""
    print("\n" + "=" * 65)
    print("Experiment Complete!")
    print("=" * 65)

    all_keys = sorted(set(k.rsplit("_", 1)[0] for k in sim_results))

    def _fmt_with_ci(metrics, key, digits):
        value = metrics.get(key)
        if value is None:
            return "N/A"
        ci = metrics.get(f"{key}_ci95")
        if ci is None:
            ci = 0.0
        return f"{value:.{digits}f}±{ci:.{digits}f}"

    for lam in arrival_rates:
        print(f"\nλ = {lam} total packets/slot")
        header = (
            f"  {'Method':<10}  {'FinalQ':>18}  {'Backlog(Mb)':>18}  "
            f"{'Delay(ms)':>18}  {'Throughput':>16}  {'Mbps':>16}  "
            f"{'Arr/slot':>14}  {'Dep/slot':>14}"
        )
        sep = (
            f"  {'─'*10}  {'─'*18}  {'─'*18}  "
            f"{'─'*18}  {'─'*16}  {'─'*16}  "
            f"{'─'*14}  {'─'*14}"
        )
        print(header)
        print(sep)
        for k in all_keys:
            m = sim_results.get(f"{k}_{lam}")
            if not m:
                continue
            final_q = _fmt_with_ci(m, 'final_avg_queue', 2)
            backlog = _fmt_with_ci(m, 'average_queue_backlog_mbit', 3)
            delay = _fmt_with_ci(m, 'average_delay_ms', 2)
            thr = _fmt_with_ci(m, 'throughput_ratio', 4)
            mbps = _fmt_with_ci(m, 'throughput_mbps', 2)
            arr_slot = _fmt_with_ci(m, 'arrival_total_rate_pps', 3)
            dep_slot = _fmt_with_ci(m, 'departure_total_rate_pps', 3)
            print(
                f"  {k:<10}  {final_q:>18}  {backlog:>18}  "
                f"{delay:>18}  {thr:>16}  {mbps:>16}  "
                f"{arr_slot:>14}  {dep_slot:>14}"
            )


# ============================================================================
# 命令行参数
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Queue-aware GAT Scheduling Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 队列感知实验（num_uav=30，默认参数）
  uv run python experiments/run_queue_experiment.py --num_uav 30 --seed 42

  # 指定到达率和仿真时隙
  uv run python experiments/run_queue_experiment.py --num_uav 30 \\
      --arrival_rates 0.5,1.0,2.0 --num_slots 5000 --seed 42

  # Monte Carlo 仿真：每个 seed 内共享 channel trace + arrival trace，再对多 seed 取平均
  uv run python experiments/run_queue_experiment.py --num_uav 30 \\
      --arrival_rates 0.5 --num_slots 5000 --seed 42 --mc_runs 5

  # 大规模实验（num_uav=50，1000 训练样本）
  uv run python experiments/run_queue_experiment.py --num_uav 50 \\
      --train_size 1000 --epochs 50 --seed 42
        """
    )
    parser.add_argument('--config', '-c', type=str, default='configs/exp_example.yaml')
    parser.add_argument('--arrival_rates', type=str, default='0.5,1.0,1.5,2.0',
                        help='Comma-separated total network arrival rates in packets/slot. '
                             'The script splits arrivals by the fixed link-level hotspot model.')
    parser.add_argument('--num_slots', type=int, default=5000,
                        help='Number of simulation slots per arrival rate')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mc_runs', type=int, default=1,
                        help='Number of Monte Carlo simulation seeds for Phase 2. '
                             'Ignored when --mc_seeds is provided.')
    parser.add_argument('--mc_seeds', type=str, default='',
                        help='Comma-separated Monte Carlo seeds for Phase 2, e.g. "42,43,44". '
                             'Within each seed, all schedulers share channel and arrival traces.')
    parser.add_argument('--slots_per_block', type=int, default=0,
                        help='Channel block length in slots. 0 means fixed channel over the whole simulation.')
    parser.add_argument('--num_uav', type=int, default=30)
    parser.add_argument('--tx_power', type=float, default=22)
    parser.add_argument('--train_size', type=int, default=500)
    parser.add_argument('--val_size', type=int, default=64)
    parser.add_argument('--test_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--queue_range', type=str, default="0,200",
                        help='Training queue range as "min,max" (default: "0,200"). '
                             'Model uses log1p(normalized)/10, so set max large enough '
                             'to cover simulation queue lengths.')
    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    args.arrival_rates = [float(x.strip()) for x in args.arrival_rates.split(',')]
    if args.mc_seeds.strip():
        args.mc_seeds = [int(x.strip()) for x in args.mc_seeds.split(',') if x.strip()]
        if not args.mc_seeds:
            args.mc_seeds = [args.seed]
    else:
        args.mc_runs = max(1, int(args.mc_runs))
        args.mc_seeds = [args.seed + i for i in range(args.mc_runs)]

    cfg = load_config(args.config)
    cfg["NUM_UAV"]    = args.num_uav
    cfg["TX_POWER"]   = args.tx_power
    cfg["TRAIN_SIZE"] = args.train_size
    cfg["VAL_SIZE"]   = args.val_size
    cfg["TEST_SIZE"]  = args.test_size
    cfg["EPOCHS"]     = args.epochs
    cfg["LR"]         = args.lr
    cfg["SEED"]       = args.seed

    # Parse queue_range
    try:
        q_parts = args.queue_range.split(',')
        queue_range = (int(q_parts[0].strip()), int(q_parts[1].strip()))
    except Exception:
        queue_range = (0, 200)
    print(f"[main] Queue range for training: {queue_range}")

    ts = datetime.now().strftime('%Y%m%d_%H%M')
    exp_name = f"queue_exp{ts}_s{args.seed}_mc{len(args.mc_seeds)}"
    results_dir = os.path.join(cfg.get("RESULTS_DIR", "./results"), exp_name)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("Queue-aware GAT Scheduling Experiment")
    print("=" * 65)
    print(f"  Results dir   : {results_dir}")
    print(f"  Model library : {MODEL_LIBRARY_DIR}/")
    print(f"  Network       : {cfg['NUM_UAV']} UAVs, TX={cfg['TX_POWER']} dBm")
    print(f"  Train/Val/Test: {cfg['TRAIN_SIZE']}/{cfg['VAL_SIZE']}/{cfg['TEST_SIZE']}")
    print(f"  Sim slots     : {args.num_slots}")
    print(f"  Arrival rates : {args.arrival_rates}  total packets/slot")
    print(f"  Arrival model : link-level hotspot "
          f"({HOTSPOT_LINK_FRACTION:.0%} links carry "
          f"{HOTSPOT_TRAFFIC_FRACTION:.0%} traffic)")
    print(f"  Train seed    : {args.seed}")
    print(f"  MC seeds      : {args.mc_seeds}")
    print(f"  Slots/block   : {args.slots_per_block or args.num_slots}")

    net_gen = build_generator(cfg)

    # =========================================================================
    # Phase 1: 训练或加载两个模型
    #
    # 【命名语义】
    # - GAT（队列感知）  ：raw_node_dim=2，Loss=-Σ Q_i×R_i
    # - GAT-base（非队列）：raw_node_dim=1，Loss=-Σ departures_i
    # 两个模型共享同一套 (num_uav, seed) 的数据集（通过 use_queue 区分）
    # =========================================================================
    print("\n" + "=" * 65)
    print("Phase 1: Model Training / Loading")
    print("=" * 65)
    print(f"  Dataset  non-queue: ./datasets/dataset_numUAV={args.num_uav}.pkl")
    print(f"  Dataset  queue    : ./datasets/queue/dataset_numUAV={args.num_uav}_linklevel.pkl")
    print(f"  Model   non-queue: {MODEL_LIBRARY_DIR}/gat_numUAV={args.num_uav}.pth")
    print(f"  Model   queue    : {MODEL_LIBRARY_DIR}/gat_queue_numUAV={args.num_uav}.pth")
    print(f"  Model   strategy : Load existing → Train if missing → Save to model library")

    # --- GAT（非队列 baseline，bin/run_experiment.py 也用同一模型库）---
    print("\n" + "-" * 65)
    print("  [GAT (baseline)] raw_node_dim=1, Loss=-Σ R_i")
    gat_base_model, gat_base_result = _train_model(
        net_gen, cfg,
        num_uav=args.num_uav,
        raw_node_dim=1,
        use_queue=False,
        model_name="GAT (baseline)",
        loss_label="-Σ R_i",
        queue_range=queue_range,
    )

    # --- GAT（队列感知，真正的方法）---
    print("\n" + "-" * 65)
    print("  [GAT] raw_node_dim=2, Loss=-Σ(Q_i × R_i)")
    gat_model, gat_result = _train_model(
        net_gen, cfg,
        num_uav=args.num_uav,
        raw_node_dim=2,
        use_queue=True,
        model_name="GAT",
        loss_label="-Σ(Q_i × R_i)",
        queue_range=queue_range,
    )

    # =========================================================================
    # Phase 2: 队列仿真对比
    # =========================================================================
    models = {
        "GAT":      (gat_model,       "GAT"),
        "GAT-base": (gat_base_model,  "GAT (baseline)"),
        "Random":   (RandomScheduler(activation_prob=0.3), "Random"),
        "Greedy":   (GreedyScheduler(),                  "Greedy (SNR)"),
    }

    sim_results, mc_details = simulation_phase(net_gen, models, cfg, args)

    # 保存仿真结果
    def _jsonify(value):
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: _jsonify(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_jsonify(v) for v in value]
        return value

    serializable = {key: _jsonify(metrics) for key, metrics in sim_results.items()}
    sim_path = os.path.join(results_dir, 'simulation_results.json')
    with open(sim_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[Save] Simulation results: {sim_path}")

    runs_path = os.path.join(results_dir, 'simulation_runs.json')
    with open(runs_path, 'w') as f:
        json.dump(_jsonify(mc_details), f, indent=2)
    print(f"[Save] Monte Carlo run details: {runs_path}")

    # 绘制队列和吞吐量轨迹图（参考 Q-CSMA 论文的可视化方式）
    _plot_trajectories(sim_results, args.arrival_rates, results_dir)

    _print_summary_table(sim_results, args.arrival_rates)


if __name__ == '__main__':
    main()
