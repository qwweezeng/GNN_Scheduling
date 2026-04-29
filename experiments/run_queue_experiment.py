"""
Queue-aware Scheduling Experiment

基于队列感知的 GAT 链路调度实验。
参考 Q-CSMA 论文的评估方法。

实验内容：
1. 生成带队列长度的训练数据集
2. 训练 GAT 模型（带队列加权 loss）
3. 多时隙队列仿真评估
4. 与基线算法对比（Random, Greedy, GMS等）
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network_generate import NetworkGenerator
from utils.io import generate_dataset
from models.model import GATScheduler
from train.trainer import Trainer
from simulator.queue_simulator import QueueSimulator, compare_scheduling_algorithms


def parse_args():
    parser = argparse.ArgumentParser(description='Queue-aware GAT Scheduling Experiment')
    
    # 网络参数
    parser.add_argument('--num_uav', type=int, default=15, help='Number of UAVs')
    parser.add_argument('--area_width', type=float, default=1000, help='Area width (m)')
    parser.add_argument('--height_min', type=float, default=50, help='Min height (m)')
    parser.add_argument('--height_max', type=float, default=150, help='Max height (m)')
    parser.add_argument('--min_distance', type=float, default=20, help='Min UAV distance (m)')
    
    # 信道参数
    parser.add_argument('--tx_power', type=float, default=22, help='TX power (dBm)')
    parser.add_argument('--snr_thres', type=float, default=5, help='SNR threshold (dB)')
    parser.add_argument('--csi_mode', type=str, default='perfect', 
                       choices=['perfect', 'large_scale', 'quantized'],
                       help='CSI mode')
    parser.add_argument('--q_bits', type=int, default=8, help='Quantization bits')
    
    # 队列参数
    parser.add_argument('--arrival_lambda', type=float, default=5.0, 
                       help='Poisson arrival rate (packets/slot)')
    parser.add_argument('--use_queue', action='store_true', 
                       help='Enable queue-aware training (MWS loss)')
    
    # 训练参数
    parser.add_argument('--num_samples', type=int, default=500, help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--eval_interval', type=int, default=5, help='Eval interval')
    
    # GAT 参数
    parser.add_argument('--mlp_hidden', type=str, default='64,32', help='MLP hidden dims')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--gat_out_dim', type=int, default=32, help='GAT output dimension')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--t_iterations', type=int, default=3, help='GAT iterations')
    
    # 仿真参数
    parser.add_argument('--num_slots', type=int, default=5000, help='Simulation slots')
    parser.add_argument('--arrival_rates', type=str, default='1,2,3,4,5,6,7,8',
                       help='Comma-separated arrival rates for comparison')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # 输出
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')
    
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_network_generator(args):
    """创建网络生成器"""
    return NetworkGenerator(
        num_uavs=args.num_uav,
        area_width=args.area_width,
        height_min=args.height_min,
        height_max=args.height_max,
        uav_min_distance=args.min_distance,
        tx_power=args.tx_power,
        Gt=0,
        Gr=0,
        noise_psd=-174,
        bandwidth=5e6,
        carrier_f=2.4e9,
        snr_thres=args.snr_thres
    )


def create_model(args, raw_node_dim=2):
    """创建 GAT 调度器模型"""
    mlp_hidden = [int(x) for x in args.mlp_hidden.split(',')]
    
    model = GATScheduler(
        raw_node_dim=raw_node_dim,  # [SNR_dB, queue_norm]
        mlp_hidden_dims=mlp_hidden,
        embed_node_dim=args.embed_dim,
        edge_dim=1,
        gat_out_dim=args.gat_out_dim,
        num_heads=args.num_heads,
        T_iterations=args.t_iterations,
        schedule_hidden_dim=32,
        dropout=0.0,
        use_power_control=False
    )
    
    return model


def train_model(args, net_gen):
    """训练队列感知的 GAT 模型"""
    print("\n" + "="*60)
    print("Phase 1: Training Queue-aware GAT Model")
    print("="*60)
    
    # 生成数据集
    print(f"\n[Data] Generating {args.num_samples} training samples...")
    dataset = generate_dataset(
        num_samples=args.num_samples,
        simulator=net_gen,
        batch_size=args.batch_size,
        csi_mode=args.csi_mode,
        q_bits=args.q_bits
    )
    
    # 划分训练/验证/测试集
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)
    
    train_set = dataset[:train_size]
    val_set = dataset[train_size:train_size+val_size]
    test_set = dataset[train_size+val_size:]
    
    print(f"[Data] Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # 确认节点特征维度
    sample_x = train_set[0].vertex_features
    raw_node_dim = sample_x.size(1)
    print(f"[Model] Input feature dimension: {raw_node_dim}")
    
    # 创建模型
    model = create_model(args, raw_node_dim=raw_node_dim)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        log_dir=args.results_dir,
        use_queue=args.use_queue
    )
    
    # 训练
    print(f"\n[Train] Starting training for {args.epochs} epochs...")
    print(f"[Train] Queue-aware mode: {'ON' if args.use_queue else 'OFF'}")
    
    result = trainer.train(
        num_epochs=args.epochs,
        train_dataset=train_set,
        val_dataset=val_set,
        eval_interval=args.eval_interval,
        lr_scheduler=scheduler,
        use_queue=args.use_queue
    )
    
    # 测试
    print("\n[Test] Evaluating on test set...")
    test_util = trainer.evaluate(
        model=model,
        dataset=test_set,
        use_queue=args.use_queue
    )
    print(f"[Test] Test utility: {test_util:.4f}")
    
    return model, result


def run_simulation(model, net_gen, args):
    """运行多时隙队列仿真"""
    print("\n" + "="*60)
    print("Phase 2: Multi-slot Queue Simulation")
    print("="*60)
    
    # 创建仿真器
    simulator = QueueSimulator(
        network_generator=net_gen,
        model=model,
        device=args.device,
        seed=args.seed
    )
    simulator.configure(arrival_lambda=args.arrival_lambda)
    
    # 解析到达率列表
    arrival_rates = [float(x) for x in args.arrival_rates.split(',')]
    
    # 运行不同到达率下的仿真
    print(f"\n[Sim] Running simulation with arrival rates: {arrival_rates}")
    
    results = compare_scheduling_algorithms(
        net_gen=net_gen,
        model=model,
        num_slots=args.num_slots,
        arrival_rates=arrival_rates,
        device=args.device
    )
    
    return results


def main():
    args = parse_args()
    
    # 设置实验名
    if args.exp_name is None:
        args.exp_name = f"queue_exp_{args.num_uav}uav_{args.csi_mode}"
        if args.use_queue:
            args.exp_name += "_queue"
        args.exp_name += f"_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    # 设置输出目录
    args.results_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 确定设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("\n" + "="*60)
    print(f"Queue-aware GAT Scheduling Experiment")
    print(f"Experiment: {args.exp_name}")
    print("="*60)
    print(f"Network: {args.num_uav} UAVs, {args.area_width}m x {args.area_width}m")
    print(f"CSI Mode: {args.csi_mode}")
    print(f"Queue-aware Training: {args.use_queue}")
    print(f"Device: {args.device}")
    
    # 创建网络生成器
    net_gen = create_network_generator(args)
    
    # Phase 1: 训练
    model, train_result = train_model(args, net_gen)
    
    # 保存模型
    model_path = os.path.join(args.results_dir, 'best_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n[Save] Model saved to {model_path}")
    
    # Phase 2: 仿真（可选，如果指定了 --num_slots）
    if args.num_slots > 0:
        sim_results = run_simulation(model, net_gen, args)
        
        # 保存仿真结果
        import json
        sim_results_path = os.path.join(args.results_dir, 'simulation_results.json')
        
        # 转换 numpy 数组为列表（用于 JSON 序列化）
        serializable_results = {}
        for rate, metrics in sim_results.items():
            serializable_results[str(rate)] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in metrics.items() 
                           if not isinstance(v, np.ndarray) or v.size < 1000
            }
        
        with open(sim_results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n[Save] Simulation results saved to {sim_results_path}")
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
