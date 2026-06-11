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
#     dataset_numUAV=10.pkl
#     ...
#
# 【队列实验命名】
# run_queue_experiment.py 使用 _get_dataset_cache_path() 生成路径：
#   - use_queue=True  → datasets/queue/dataset_numUAV=<N>.pkl
#   - use_queue=False → datasets/dataset_numUAV=<N>.pkl
#
def _get_dataset_cache_path(num_uav, use_queue):
    """
    生成数据集缓存路径。

    Args:
        num_uav: 无人机数量
        use_queue: 是否为队列实验
            True  → datasets/queue/dataset_numUAV=<N>.pkl
            False → datasets/dataset_numUAV=<N>.pkl
    """
    subdir = "queue" if use_queue else ""
    base = f"./datasets/{subdir}/dataset_numUAV={num_uav}.pkl" if subdir else f"./datasets/dataset_numUAV={num_uav}.pkl"
    return base


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
                    if cached_flag != expected_flag:
                        print(f"  [Warning] Cached {split} queue_flag={cached_flag}, "
                              f"expected {expected_flag}. Will regenerate.")
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


def run_single_simulator(net_gen, model, arrival_lambda, cfg, num_slots, device='cpu',
                         channel_realizations=None):
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
    )
    return sim.run(
        num_slots=num_slots,
        csi_mode=cfg.get("CSI_MODE", "perfect"),
        verbose=False, log_interval=500,
        channel_realizations=channel_realizations,
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
        # 生成第 b 个信道块（带队列特征，让所有调度器共享）
        # 注意：feature 永远是 [SNR, queue_norm]，非队列模型会自己 slice 掉 queue 维度
        queue_states_b = None if b == 0 else realizations[b-1].queue_states
        state = net_gen.generate_network(
            csi_mode=cfg.get("CSI_MODE", "perfect"),
            q_bits=cfg.get("Q_BITS", 8),
            queue_states=queue_states_b,
            include_queue_feature=True,
        )
        # 复制队列状态
        if queue_states_b is not None:
            state.queue_states = queue_states_b
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
    print(f"  Arrival rates  : {args.arrival_rates}")
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

    for lam in args.arrival_rates:
        print(f"\n{'─' * 65}")
        print(f"  λ = {lam} packets/slot")
        print(f"{'─' * 65}")

        # 为所有调度器预生成共享信道（公平比较）
        slots_per_block = getattr(args, 'slots_per_block', None)
        channel_realizations = _pregenerate_channel_realizations(
            net_gen, args.num_slots, cfg, slots_per_block=slots_per_block
        )
        print(f"  [Fair] Pre-generated {len(channel_realizations)} shared channel blocks "
              f"(slots_per_block={slots_per_block or args.num_slots})")

        for key, (model, name) in models.items():
            print(f"  [{name}]")
            metrics = run_single_simulator(
                net_gen, model, lam, cfg, args.num_slots, device=str(device),
                channel_realizations=channel_realizations,
            )
            results[f"{key}_{lam}"] = metrics

            stability = "✓" if metrics.get('is_stable') is True else (
                "✗" if metrics.get('is_stable') is False else "?")
            print(f"    AvgQ={metrics['avg_queue']:>7.2f}  "
                  f"Throughput={metrics['throughput_ratio']:>6.4f}  "
                  f"Stable={stability}  "
                  f"{metrics['throughput_mbps']:>6.2f} Mbps  "
                  f"DepRate={metrics['departure_rate_pps']:>5.3f} pps/slot")

    return results


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
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Queue & Throughput Trajectories  (λ = {lam} packets/slot, {len(all_keys)} algorithms)",
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
                ax.plot(q_traj, color=color, label=label, linewidth=1.5, alpha=0.85)

            # 中图：每时隙总发送量 (departures per slot)
            ax = axes[1]
            dep_traj = m.get('departures_per_slot')
            if dep_traj is not None:
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
                ax.plot(tp_traj, color=color, label=label, linewidth=1.5, alpha=0.85)

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
        for lam in arrival_rates:
            m = sim_results.get(f"{key}_{lam}")
            vals.append(m['avg_queue'] if m else 0)
        ax2.bar(x + i * width, vals, width, label=key, color=color_map.get(key, '#607D8B'), alpha=0.85)
    ax2.set_xlabel("Arrival Rate λ (packets/slot)", fontsize=11)
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


def _print_summary_table(sim_results, arrival_rates):
    """打印结果汇总表。"""
    print("\n" + "=" * 65)
    print("Experiment Complete!")
    print("=" * 65)

    all_keys = sorted(set(k.rsplit("_", 1)[0] for k in sim_results))

    header = f"  {'λ':>5}"
    sep    = f"  {'─'*5}"
    for k in all_keys:
        header += f"  {k[:12]:>12}"
        sep    += f"  {'─'*12}"
    header += f"  {'GAT Mbps':>10}"
    sep    += f"  {'─'*10}"
    print(header)
    print(sep)

    for lam in arrival_rates:
        row = f"  {lam:>5.1f}"
        for k in all_keys:
            m = sim_results.get(f"{k}_{lam}")
            row += f"  {(m['avg_queue'] if m else 'N/A'):>12.2f}"
        gat_q = sim_results.get(f"gat_{lam}")
        if gat_q:
            row += f"  {gat_q['throughput_mbps']:>10.2f}"
        print(row)


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

  # 大规模实验（num_uav=50，1000 训练样本）
  uv run python experiments/run_queue_experiment.py --num_uav 50 \\
      --train_size 1000 --epochs 50 --seed 42
        """
    )
    parser.add_argument('--config', '-c', type=str, default='configs/exp_example.yaml')
    parser.add_argument('--arrival_rates', type=str, default='0.5,1.0,1.5,2.0',
                        help='Comma-separated arrival rates in packets/slot')
    parser.add_argument('--num_slots', type=int, default=5000,
                        help='Number of simulation slots per arrival rate')
    parser.add_argument('--seed', type=int, default=42)
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
    exp_name = f"queue_exp{ts}_s{args.seed}"
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
    print(f"  Arrival rates : {args.arrival_rates}  packets/slot")
    print(f"  Seed          : {args.seed}")

    net_gen = build_generator(cfg)

    # =========================================================================
    # Phase 1: 训练或加载两个模型
    #
    # 【命名语义】
    # - GAT（队列感知）  ：raw_node_dim=2，Loss=-Σ log1p(Q_i)×departures_i
    # - GAT-base（非队列）：raw_node_dim=1，Loss=-Σ departures_i
    # 两个模型共享同一套 (num_uav, seed) 的数据集（通过 use_queue 区分）
    # =========================================================================
    print("\n" + "=" * 65)
    print("Phase 1: Model Training / Loading")
    print("=" * 65)
    print(f"  Dataset  non-queue: ./datasets/dataset_numUAV={args.num_uav}.pkl")
    print(f"  Dataset  queue    : ./datasets/queue/dataset_numUAV={args.num_uav}.pkl")
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

    sim_results = simulation_phase(net_gen, models, cfg, args)

    # 保存仿真结果
    serializable = {}
    for key, metrics in sim_results.items():
        serializable[key] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                v.tolist() if isinstance(v, np.ndarray) else
                v)
            for k, v in metrics.items()
        }
    sim_path = os.path.join(results_dir, 'simulation_results.json')
    with open(sim_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[Save] Simulation results: {sim_path}")

    # 绘制队列和吞吐量轨迹图（参考 Q-CSMA 论文的可视化方式）
    _plot_trajectories(sim_results, args.arrival_rates, results_dir)

    _print_summary_table(sim_results, args.arrival_rates)


if __name__ == '__main__':
    main()
