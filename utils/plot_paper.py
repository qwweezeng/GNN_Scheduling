"""
综合画图模块
包含三个图的画图函数:
1. numuav与baseline对比的柱状图
2. train_loss/val_utility随epoch变化的曲线图
3. 性能随训练样本数量变化的曲线图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional, Tuple
from matplotlib.pyplot import cm

from utils.plot_config import (
    NUMUAV_COLORS, BASELINE_COLORS, PLOT_STYLE, SMOOTH_PARAMS,
    get_numuav_color, get_all_numuav_colors, update_colors
)


# 设置画图风格
plt.rcParams.update(PLOT_STYLE)


def smooth_curve(vals: List[float], window: int = None, method: str = None, 
                 ema_alpha: float = None) -> np.ndarray:
    """
    平滑曲线 - 改进版：避免过度平滑
    
    Args:
        vals: 原始数据
        window: 平滑窗口大小
        method: 平滑方法: 'ema', 'ma', 'savgol'
        ema_alpha: EMA系数 (越小越平滑，默认从配置读取)
    
    Returns:
        平滑后的数据
    """
    if window is None:
        window = SMOOTH_PARAMS['window']
    if method is None:
        method = SMOOTH_PARAMS['method']
    if ema_alpha is None:
        ema_alpha = SMOOTH_PARAMS.get('ema_alpha', 0.3)
    
    vals = np.array(vals, dtype=np.float64)
    
    if len(vals) < window:
        return vals
    
    if method == 'ma':
        # 简单移动平均
        kernel = np.ones(window) / window
        return np.convolve(vals, kernel, mode='same')
    
    elif method == 'ema':
        # 指数加权移动平均 - 使用更小的alpha避免过度平滑
        ema = np.zeros_like(vals)
        ema[0] = vals[0]
        for i in range(1, len(vals)):
            ema[i] = ema_alpha * vals[i] + (1 - ema_alpha) * ema[i-1]
        return ema
    
    elif method == 'savgol':
        # Savitzky-Golay 滤波器 - 保持峰值形状
        from scipy.signal import savgol_filter
        # 确保window是奇数且大于polyorder
        window = window if window % 2 == 1 else window + 1
        polyorder = min(3, window - 1)  # polyorder必须小于window
        if polyorder < 1:
            return vals  # 数据太少，无法平滑
        return savgol_filter(vals, window, polyorder)
    
    return vals


def compute_running_stats(vals: List[float], window: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算滑动窗口的均值和标准差，用于绘制置信区间阴影
    
    Args:
        vals: 原始数据
        window: 窗口大小
    
    Returns:
        (means, stds, smooth_vals): 均值、标准差、平滑值
    """
    if window is None:
        window = SMOOTH_PARAMS['window']
    
    vals = np.array(vals, dtype=np.float64)
    n = len(vals)
    means = np.zeros(n)
    stds = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        window_vals = vals[start:end]
        means[i] = np.mean(window_vals)
        stds[i] = np.std(window_vals)
    
    # 同时计算平滑曲线
    smooth_vals = smooth_curve(vals, window=window, method='savgol')
    
    return means, stds, smooth_vals


def sample_for_display(epochs: List[int], vals: List[float], step: int = 2) -> Tuple[List[int], List[float]]:
    """
    采样减少显示的原始数据点数量，避免曲线过于密集
    
    Args:
        epochs: epoch列表
        vals: 值列表
        step: 采样间隔
    
    Returns:
        采样后的epochs和vals
    """
    return epochs[::step], [v for i, v in enumerate(vals) if i % step == 0]


# ==================== 图1: numuav与baseline对比柱状图 ====================

def plot_baseline_comparison(
    results_dir: str = 'results',
    experiment_ids: List[str] = None,
    output_dir: str = None,
    figsize: Tuple[float, float] = (5.5, 4),  # 宽度统一减小
    show: bool = False
) -> Tuple[str, str]:
    """
    绘制numuav与baseline对比的柱状图
    
    改进版：
    - Ours放在每组最右边
    - 减小组间距适配论文半栏
    - 所有baseline都显示
    
    Args:
        results_dir: 结果目录
        experiment_ids: 实验ID列表
        output_dir: 输出目录
        figsize: 图形大小（默认适配论文半栏）
        show: 是否显示图形
    
    Returns:
        (eps_path, png_path): EPS和PNG文件的保存路径
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'img')
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认实验ID
    if experiment_ids is None:
        experiment_ids = [
            'exp_20_20260130_2147',
            'exp_30_20260130_2236',
            'exp_40_20260131_0118',
            'exp_50_20260131_1740'
        ]
    
    numuav_values = [20, 30, 40, 50]
    
    # 收集数据 - 包含所有baseline
    our_results = []
    baseline_results = {name: [] for name in BASELINE_COLORS.keys() if name != 'ours'}
    baseline_results = {'random':[], 'simple_greedy':[], 'local_greedy':[],'fplinq':[],'spatial_dl':[]}
    baseline_results = {'random':[], 'simple_greedy':[], 'local_greedy':[], 'spatial_dl':[]}

    for exp_id, numuav in zip(experiment_ids, numuav_values):
        exp_dir = os.path.join(results_dir, exp_id)
        
        # 读取我们的test_utility
        summary_path = os.path.join(results_dir, 'all_experiments_summary.xlsx')
        df = pd.read_excel(summary_path, sheet_name='Sheet1')
        our_utility = df[df['experiment_id'] == exp_id]['test_utility'].values
        our_results.append(our_utility[0] if len(our_utility) > 0 else 0)
        
        # 读取baseline结果
        baseline_path = os.path.join(exp_dir, f'{exp_id}_baselines.json')
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r') as f:
                baselines = json.load(f)
        else:
            baselines = {name: np.nan for name in baseline_results.keys()}
        
        for name in baseline_results.keys():
            baseline_results[name].append(baselines.get(name, np.nan))
    
    # 绘制分组柱状图
    fig, ax = plt.subplots(figsize=figsize)
    
    n_groups = len(numuav_values)
    n_methods = len(baseline_results) + 1  # baseline数量 + Ours
    
    # 调整柱子宽度和间距
    bar_width = 0.03  # 柱子宽度
    group_gap = 0.05  # 组间距
    
    # 从左到右: baselines (从右到左排列) → Ours (最右)
    method_order = list(baseline_results.keys()) + ['proposed GAT']
    
    # 计算每组左边界
    group_left = np.arange(n_groups) * (n_methods * bar_width + group_gap)
    
    # 绘制柱子
    colors = {**BASELINE_COLORS, 'proposed GAT': BASELINE_COLORS['proposed GAT']}
    
    for i, name in enumerate(method_order):
        if name == 'proposed GAT':
            values = our_results
            values = [v * 5 for v in values]
            positions = group_left + i * bar_width
        else:
            values = baseline_results[name]
            values = [v * 5 for v in values]
            positions = group_left + i * bar_width
        
        color = colors.get(name, '#888888')
        # 处理特殊大小写：Proposed GAT, Spatial DL
        if name == 'proposed GAT':
            label = 'Proposed GAT'
        elif name == 'spatial_dl':
            label = 'Spatial DL'
        elif name == 'fplinq':
            label = 'FPLinQ'
        else:
            label = name.replace('_', ' ').title()
        
        ax.bar(positions, values, bar_width, 
              label=label, color=color, 
              edgecolor='black', linewidth=0.8,
              alpha=0.85)
    
    # 设置图形属性
    ax.set_xlabel('Number of UAVs', fontsize=11)
    ax.set_ylabel('Test Utility (Mbps)', fontsize=11)
    ax.set_title('Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(group_left + (n_methods - 1) * bar_width / 2)
    ax.set_xticklabels(numuav_values, fontsize=10)
    
    # 图例放在图内左上方
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 调整y轴范围，留出顶部空间 - 直接从柱子高度计算
    heights = [p.get_height() for p in ax.patches]
    y_max = max(heights) * 1.15  # 增加15%空间
    y_max = max(y_max, 10)  # 确保最小y轴范围为10
    ax.set_ylim(0, y_max)
    
    plt.tight_layout()
    
    # 保存图形（同时保存EPS和PNG）
    eps_path = os.path.join(output_dir, 'baseline_comparison.eps')
    png_path = os.path.join(output_dir, 'baseline_comparison.png')
    
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    plt.close()
    
    return eps_path, png_path


# ==================== 图2: train_loss/val_utility随epoch变化 ====================

def plot_training_curves(
    results_dir: str = 'results',
    experiment_ids: List[str] = None,
    numuav_values: List[int] = None,
    metric: str = 'val_utility',  # 'train_loss' 或 'val_utility'
    output_dir: str = None,
    figsize: Tuple[float, float] = (5.5, 4),  # 宽度统一减小
    show: bool = False,
    with_confidence_interval: bool = False,
    with_inset: bool = False,  # 是否添加早期放大的小图
    smooth_window: int = None,
    smooth_method: str = 'savgol'  # 改用savgol保留峰值
) -> Tuple[str, str]:
    """
    绘制train_loss或val_utility随epoch变化的曲线图 - 改进版
    
    使用带阴影的置信区间形式展示数据波动:
    - 主曲线: Savitzky-Golay平滑（保留峰值）
    - 阴影区域: 滑动窗口的标准差范围
    - 可选: 早期epoch放大插图
    
    Args:
        results_dir: 结果目录
        experiment_ids: 实验ID列表
        numuav_values: numuav值列表
        metric: 绘制指标: 'train_loss' 或 'val_utility'
        output_dir: 输出目录
        figsize: 图形大小
        show: 是否显示图形
        with_confidence_interval: 是否显示置信区间阴影
        with_inset: 是否添加早期epoch放大插图
        smooth_window: 平滑窗口大小
        smooth_method: 平滑方法: 'ema', 'ma', 'savgol'
    
    Returns:
        (eps_path, png_path): EPS和PNG文件的保存路径
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'img')
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认实验ID
    if experiment_ids is None:
        experiment_ids = [
            'exp_20_20260130_2147',
            'exp_30_20260130_2236',
            'exp_40_20260131_0118',
            'exp_50_20260131_1740'
        ]
    if numuav_values is None:
        numuav_values = [20, 30, 40, 50]
    if smooth_window is None:
        smooth_window = SMOOTH_PARAMS['window']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = get_all_numuav_colors()
    shadow_alpha = SMOOTH_PARAMS['shadow_alpha']
    raw_alpha = SMOOTH_PARAMS['raw_line_alpha']
    
    # 用于插图的数据
    inset_data = {numuav: {'epochs': [], 'values': []} for numuav in numuav_values}
    
    for exp_id, numuav in zip(experiment_ids, numuav_values):
        # 读取per_epoch数据
        exp_dir = os.path.join(results_dir, exp_id)
        json_path = os.path.join(exp_dir, f'{exp_id}_per_epoch.json')
        
        with open(json_path, 'r') as f:
            logs = json.load(f)
        
        epochs = [e.get('epoch', idx) for idx, e in enumerate(logs)]
        values = [e.get(metric, np.nan) for e in logs]
        
        # 只显示前30个epoch
        epochs = epochs[:31]
        values = values[:31]
        
        # 乘上系数5转换为Mbps（仿真时为了方便没有乘5e6带宽）
        if metric in ['val_utility', 'train_loss', 'utility']:
            values = [v * 5 for v in values]
        
        # 保存前15个epoch的数据用于插图
        inset_epochs = [e for e in epochs if e < 15]
        inset_values = [v for i, v in enumerate(values) if epochs[i] < 15]
        inset_data[numuav]['epochs'] = inset_epochs
        inset_data[numuav]['values'] = inset_values
        
        # 平滑处理 - 使用savgol滤波器
        smoothed = smooth_curve(values, window=smooth_window, method=smooth_method)
        color = colors.get(numuav, '#333333')
        
        # 绘制采样后的原始数据（避免过于密集）
        sample_epochs, sample_values = sample_for_display(epochs, values, step=1)
        ax.plot(sample_epochs, sample_values, color=color, linestyle=':', 
               linewidth=1, alpha=raw_alpha, marker='None')
        
        # 绘制置信区间阴影
        if with_confidence_interval:
            _, _, smooth_vals = compute_running_stats(values, window=smooth_window)
            # 使用平滑后的值计算标准差
            stds = np.abs(smoothed - smooth_vals) + SMOOTH_PARAMS.get('window', 5) * 0.5
            ax.fill_between(epochs, smoothed - stds, smoothed + stds, 
                           color=color, alpha=shadow_alpha)
        
        # 绘制平滑后的主曲线
        ax.plot(epochs, smoothed, color=color, linewidth=2.5, 
               label=f'numUAV={numuav}', marker='o', markersize=6, markevery=5)
    
    # 设置图形属性
    if metric == 'val_utility':
        ylabel = 'Validation Utility (Mbps)'
        title = 'Validation Utility over Training Epochs'
    elif metric == 'train_loss':
        ylabel = 'Training Loss (Mbps)'
        title = 'Training Loss over Training Epochs'
    else:
        ylabel = metric.replace('_', ' ').title()
        title = f'{ylabel} over Training Epochs'
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 添加早期epoch放大插图
    if with_inset:
        ax_inset = fig.add_axes([0.55, 0.25, 0.35, 0.35])  # [left, bottom, width, height]
        
        for numuav in numuav_values:
            data = inset_data[numuav]
            if data['epochs']:
                color = colors.get(numuav, '#333333')
                # 平滑 - 使用更小的polyorder避免窗口太小时的问题
                window_inset = min(5, len(data['values']))
                polyorder = max(2, window_inset - 2)
                smoothed_inset = smooth_curve(data['values'], window=window_inset, method='savgol')
                # 绘制原始和 smoothed
                ax_inset.plot(data['epochs'], data['values'], color=color, 
                             linestyle=':', linewidth=1, alpha=raw_alpha)
                ax_inset.plot(data['epochs'], smoothed_inset, color=color, 
                             linewidth=2, marker='o', markersize=6)
        
        ax_inset.set_xlim(-0.5, 15)
        ax_inset.set_xlabel('Epoch', fontsize=9)
        ax_inset.set_ylabel(ylabel, fontsize=9)
        ax_inset.set_title('Early Phase (0-15 epochs)', fontsize=10)
        ax_inset.grid(True, linestyle='--', alpha=0.3)
        ax_inset.tick_params(labelsize=8)
    
    plt.tight_layout()
    
    # 保存图形（同时保存EPS和PNG）
    eps_path = os.path.join(output_dir, f'training_curves_{metric}.eps')
    png_path = os.path.join(output_dir, f'training_curves_{metric}.png')
    
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    plt.close()
    
    return eps_path, png_path


# ==================== 图3: 性能随训练样本数量变化 ====================

def plot_performance_vs_samples(
    results_dir: str = 'results',
    excel_file: str = 'all_experiments_summary.xlsx',
    sheet_name: str = 'Sheet2',
    numuav_values: List[int] = None,
    output_dir: str = None,
    figsize: Tuple[float, float] = (5.5, 4),  # 宽度减小，让点更紧凑
    show: bool = False
) -> Tuple[str, str]:
    """
    绘制性能随训练样本数量变化的曲线图
    
    Args:
        results_dir: 结果目录
        excel_file: Excel文件名
        sheet_name: sheet名称
        numuav_values: numuav值列表，目前只生成了20的，其他还没跑完
        output_dir: 输出目录
        figsize: 图形大小
        show: 是否显示图形
    
    Returns:
        (eps_path, png_path): EPS和PNG文件的保存路径
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'img')
    os.makedirs(output_dir, exist_ok=True)
    
    if numuav_values is None:
        numuav_values = [20, 30, 40, 50]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = get_all_numuav_colors()
    
    # 读取Excel数据
    excel_path = os.path.join(results_dir, excel_file)
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    for numuav in numuav_values:
        # 筛选当前numuav的数据并按train_size排序
        df_numuav = df[df['num_uav'] == numuav].sort_values('train_size').reset_index(drop=True)
        
        if len(df_numuav) == 0:
            # 如果没有数据，仍然画出空点到x轴上作为占位
            ax.plot([], [], color=colors.get(numuav, '#333333'), 
                   linewidth=2, marker='o', markersize=6,
                   label=f'numuav={numuav} (pending)')
            continue
        
        train_sizes = df_numuav['train_size'].values
        test_utilities = df_numuav['best_val_utility'].values
        
        
        color = colors.get(numuav, '#333333')
        
        # 绘制曲线
        want = [0,1,2,3,5,7,9,10,11,12,13]
        ax.plot(train_sizes[want], test_utilities[want] * 5, color=color, 
               linewidth=2, marker='o', markersize=6,
               label=f'numUAV={numuav}')
    
    # 设置图形属性
    ax.set_xlabel('Training Sample Size', fontsize=12)
    ax.set_ylabel('Test Utility (Mbps)', fontsize=12)
    ax.set_title('Test Utility vs Training Sample Size', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图形（同时保存EPS和PNG）
    eps_path = os.path.join(output_dir, 'performance_vs_samples.eps')
    png_path = os.path.join(output_dir, 'performance_vs_samples.png')
    
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
    
    if show:
        plt.show()
    plt.close()
    
    return eps_path, png_path


# ==================== 主函数 ====================

def main():
    """
    主函数：按需调用各个画图函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='生成实验结果图表')
    parser.add_argument('--plot', type=str, default='all',
                       choices=['all', 'baseline', 'training', 'samples'],
                       help='指定要绑制的图表类型')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='结果目录')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--show', action='store_true',
                       help='是否显示图形（需要图形界面环境）')
    parser.add_argument('--format', type=str, default='both',
                       choices=['eps', 'png', 'both'],
                       help='输出格式: eps(论文用), png(预览用), both(都生成)')
    parser.add_argument('--colors', type=str, default=None,
                       help='可选：指定配色方案，如 "vibrant" 或 "pastel"')
    
    args = parser.parse_args()
    
    # 可选：更新配色方案
    if args.colors == 'vibrant':
        update_colors({
            20: '#FF6B6B',  # 珊瑚红
            30: '#4ECDC4',  # 青绿色
            40: '#45B7D1',  # 天蓝色
            50: '#96CEB4',  # 浅绿色
        })
    elif args.colors == 'v1':
        update_colors({
            20: '#0000FF',  # 深紫
            30: '#8080FF',  # 浅紫
            40: '#FF9090',  # 浅红
            50: '#FF5959',  # 深红
        })
    elif args.colors == 'v2':
        update_colors({
            20: '#478CCA',  # 浅粉色
            30: '#757BBB',  # 浅绿色
            40: '#F37B79',  # 浅蓝色
            50: '#EE3223',  # 浅黄色
        })
    
    output_dir = args.output_dir
    
    if args.plot in ['all', 'baseline']:
        print("正在生成: numuav与baseline对比图...")
        eps_path, png_path = plot_baseline_comparison(
            args.results_dir, output_dir=output_dir, show=args.show
        )
        if args.format in ['eps', 'both']:
            print(f"EPS已保存到: {eps_path}")
        if args.format in ['png', 'both']:
            print(f"PNG已保存到: {png_path}")
        if args.show:
            print("图形窗口已打开...")
    
    if args.plot in ['all', 'training']:
        print("正在生成: 训练曲线图 (val_utility)...")
        eps_path, png_path = plot_training_curves(
            args.results_dir, metric='val_utility', 
            output_dir=output_dir, show=args.show
        )
        if args.format in ['eps', 'both']:
            print(f"EPS已保存到: {eps_path}")
        if args.format in ['png', 'both']:
            print(f"PNG已保存到: {png_path}")
        
        print("正在生成: 训练曲线图 (train_loss)...")
        eps_path, png_path = plot_training_curves(
            args.results_dir, metric='train_loss', 
            output_dir=output_dir, show=args.show
        )
        if args.format in ['eps', 'both']:
            print(f"EPS已保存到: {eps_path}")
        if args.format in ['png', 'both']:
            print(f"PNG已保存到: {png_path}")
    
    if args.plot in ['all', 'samples']:
        print("正在生成: 性能随样本数量变化图...")
        eps_path, png_path = plot_performance_vs_samples(
            args.results_dir, output_dir=output_dir, show=args.show
        )
        if args.format in ['eps', 'both']:
            print(f"EPS已保存到: {eps_path}")
        if args.format in ['png', 'both']:
            print(f"PNG已保存到: {png_path}")


if __name__ == '__main__':
    main()
