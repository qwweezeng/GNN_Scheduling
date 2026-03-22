#!/usr/bin/env python3
"""
完整的性能对比测试
"""
import sys
sys.path.insert(0, '/workspace/GNN_Scheduling')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import torch
import numpy as np
from utils.network_generate import NetworkGenerator
from utils.baseline import build_power_gain_matrix, compute_sum_rate_from_allocs, eval_fplinq, eval_greedy_spatial
from models.model import GATScheduler, calculate_achievable_rate

print("="*90)
print("🔍 完整的性能对比测试")
print("="*90)

# 测试多个样本
num_samples = 10
gen = NetworkGenerator(
    num_uavs=20,
    area_width=1000,
    height_min=50,
    height_max=150,
    uav_min_distance=20,
    tx_power=10,
    Gt=0,
    Gr=0,
    noise_psd=-169,
    bandwidth=5e6,
    carrier_f=2.4e9,
)

# 加载模型
model = GATScheduler(
    raw_node_dim=1, mlp_hidden_dims=[32, 32], embed_node_dim=32,
    edge_dim=1, gat_out_dim=48, num_heads=4, T_iterations=4, schedule_hidden_dim=32,
)
model_path = "/workspace/GNN_Scheduling/logs/models/best_model_epoch19.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

results = {
    'our_model': [],
    'fplinq': [],
    'greedy_spatial': [],
}

print(f"\n📊 测试 {num_samples} 个样本...")

for i in range(num_samples):
    # 生成新网络
    state = gen.generate_network()
    num_links = state.links.size(0)
    
    # 模型预测
    with torch.no_grad():
        pi_soft, _ = model(state.vertex_features, state.edge_index, state.edge_weights, use_ste=False)
    
    # 使用阈值 0.38
    threshold = 0.38
    pi_hard = (pi_soft > threshold).float()
    
    # 我们的速率
    our_rate = calculate_achievable_rate(pi_hard, state, binary=True).sum().item()
    results['our_model'].append(our_rate)
    
    # 基线
    power_gain, _ = build_power_gain_matrix(state, use_fading=True)
    power_gain_np = power_gain.cpu().numpy()
    our_active = torch.where(pi_hard.squeeze() > 0)[0].cpu().numpy()
    
    allocs = np.zeros(num_links)
    allocs[our_active] = 1
    
    try:
        fplinq_rate = eval_fplinq([state])[0]
        results['fplinq'].append(fplinq_rate)
    except:
        results['fplinq'].append(0)
    
    try:
        gsp_rate = eval_greedy_spatial([state])[0]
        results['greedy_spatial'].append(gsp_rate)
    except:
        results['greedy_spatial'].append(0)

print(f"\n📈 性能对比（平均值）：")
our_mean = np.mean(results['our_model']) / 1e6
fplinq_mean = np.mean(results['fplinq']) / 1e6
gsp_mean = np.mean(results['greedy_spatial']) / 1e6

print(f"   我们的模型: {our_mean:.2f} Mbps")
print(f"   FPLinQ: {fplinq_mean:.2f} Mbps")
print(f"   Greedy Spatial: {gsp_mean:.2f} Mbps")

print(f"\n📊 相对性能：")
if fplinq_mean > 0:
    print(f"   我们 vs FPLinQ: {our_mean/fplinq_mean*100:.1f}%")
if gsp_mean > 0:
    print(f"   我们 vs GSp: {our_mean/gsp_mean*100:.1f}%")

print(f"\n📊 详细结果（每个样本）：")
print(f"   样本 | 我们 (Mbps) | FPLinQ (Mbps) | GSp (Mbps)")
print(f"   " + "-"*55)
for i in range(num_samples):
    our = results['our_model'][i] / 1e6
    fp = results['fplinq'][i] / 1e6
    gs = results['greedy_spatial'][i] / 1e6
    print(f"   {i:4d} | {our:11.2f} | {fp:13.2f} | {gs:11.2f}")
