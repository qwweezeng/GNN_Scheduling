#!/usr/bin/env python
"""
功率控制实验：对比有/无功率控制的 GAT 调度性能。

实验设计：
1. 基准组（无功率控制）：USE_POWER_CONTROL=False
2. 实验组（有功率控制）：USE_POWER_CONTROL=True

关键对比指标：
- 网络总吞吐量（Utility）
- 链路激活率（pi_mean）
- 功率缩放系数分布（alpha_mean）
- 收敛速度
- 泛化能力

使用方法：
    # 运行对比实验（numUAV=50，训练样本300）
    uv run python experiments/power_control_experiment.py --num-uav 50 --train-size 300 --gpu-id 0

    # 快速测试（少量样本）
    uv run python experiments/power_control_experiment.py --num-uav 20 --train-size 50 --epochs 10

    # 网格搜索 λ_power_penalty
    uv run python experiments/power_control_experiment.py --num-uav 30 --train-size 100 \
        --power-penalty-search 0.001 0.01 0.05 0.1
"""

import os
import sys
import argparse
import yaml
import json
import random
import numpy as np
import torch
import pickle as pkl
from time import strftime

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.network_generate import NetworkGenerator
from models.model import GATScheduler, calculate_achievable_rate
from train.trainer import Trainer
from utils.io import save_experiment_results, generate_dataset
from utils.baseline import eval_baseline


def load_config(path):
    """Load configuration from YAML or JSON file."""
    with open(path, "r") as f:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_generator(params):
    """Create NetworkGenerator from params dict."""
    height_range = params.get("HEIGHT_RANGE", (50, 150))
    return NetworkGenerator(
        num_uavs=params.get("NUM_UAV", 50),
        area_width=params.get("AREA_WIDTH", 1000),
        height_min=height_range[0],
        height_max=height_range[1],
        uav_min_distance=params.get("MIN_DISTANCE", 20),
        tx_power=params.get("TX_POWER", 22),
        Gt=params.get("G_T", 0),
        Gr=params.get("G_R", 0),
        bandwidth=params.get("BANDWIDTH", 5e6),
        carrier_f=params.get("CARRIER_FREQ", 2.4e9),
        noise_psd=params.get("NOISE_PSD", -169),
        snr_gap_db=params.get("SNR_GAP_DB", 0),
    )


def run_power_control_experiment(params, use_power_control,
                                 results_dir="./results", logs_dir="./logs",
                                 experiment_group="power_control"):
    """
    运行单个功率控制实验。

    Args:
        params: 配置参数字典
        use_power_control: 是否启用功率控制
        results_dir: 结果保存目录
        logs_dir: 日志保存目录
        experiment_group: 实验组名称（用于 Excel 分组）

    Returns:
        dict: 包含训练日志、测试指标等
    """
    # 设置随机种子
    seed = params.get("SEED", 42)
    set_seed(seed)

    # 更新参数
    params = params.copy()
    params["USE_POWER_CONTROL"] = use_power_control

    print(f"\n{'='*60}")
    mode_str = "POWER_CONTROL" if use_power_control else "BASELINE"
    print(f"Running experiment: {mode_str}")
    print(f"{'='*60}")

    # 创建网络生成器
    gen = create_generator(params)

    # CSI 模式
    csi_mode = params.get("CSI_MODE", "perfect")
    q_bits = params.get("Q_BITS", 8)
    print(f"[runner] CSI_MODE={csi_mode}, Q_BITS={q_bits}")

    # 数据集缓存
    cache_path = params.get("DATASET_CACHE_PATH")

    # 尝试从缓存加载数据集
    train_set = None
    val_set = None
    test_set = None

    if cache_path and os.path.exists(cache_path):
        print(f"[runner] Loading cached datasets from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                cached_datasets = pkl.load(f)
            full_train_set = cached_datasets['train_set']
            full_val_set = cached_datasets['val_set']
            full_test_set = cached_datasets['test_set']

            train_size = params.get("TRAIN_SIZE", 300)
            val_size = params.get("VAL_SIZE", 50)
            test_size = params.get("TEST_SIZE", 50)

            train_set = full_train_set[:train_size]
            val_set = full_val_set[:val_size]
            test_set = full_test_set[:test_size]

            print(f"[runner] Sampled from cache: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        except Exception as e:
            print(f"[runner] Failed to load cache: {e}, generating new datasets")
            train_set = val_set = test_set = None

    if train_set is None or val_set is None or test_set is None:
        # 功率控制实验默认不使用队列特征（纯吞吐量最大化）
        include_queue_feature = params.get("USE_QUEUE", False)
        print("[runner] Generating new datasets (include_queue_feature={})...".format(include_queue_feature))
        train_set = generate_dataset(
            params.get("TRAIN_SIZE", 300), gen,
            batch_size=params.get("GEN_BATCH_SIZE", 50),
            csi_mode=csi_mode, q_bits=q_bits,
            include_queue_feature=include_queue_feature,
        )
        val_set = generate_dataset(
            params.get("VAL_SIZE", 50), gen,
            batch_size=params.get("GEN_BATCH_SIZE", 50),
            csi_mode=csi_mode, q_bits=q_bits,
            include_queue_feature=include_queue_feature,
        )
        test_set = generate_dataset(
            params.get("TEST_SIZE", 50), gen,
            batch_size=params.get("GEN_BATCH_SIZE", 50),
            csi_mode=csi_mode, q_bits=q_bits,
            include_queue_feature=include_queue_feature,
        )
        print(f"[runner] Generated: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # 打印链路数量统计
    val_counts = [int(s.links.size(0)) for s in val_set]
    print(f"[runner] Val set link counts — avg: {np.mean(val_counts):.1f}, median: {np.median(val_counts):.1f}")

    # 从实际数据推断节点特征维度（最可靠）
    # 基础配置 NODE_RAW_DIM=1 用于纯吞吐量实验
    # 功率控制实验默认不使用队列感知
    use_queue = params.get("USE_QUEUE", False)
    expected_dim = 2 if use_queue else 1
    sample_x = val_set[0].vertex_features
    actual_dim = sample_x.size(1)
    if actual_dim != expected_dim:
        print(f"[Warning] Dataset feature dim={actual_dim} but expected={expected_dim} for USE_QUEUE={use_queue}. "
              f"Using actual_dim={actual_dim}.")
    print(f"[Model] Input feature dimension: {actual_dim} (use_queue={use_queue})")

    # 创建模型
    node_raw_dim = actual_dim
    mlp_hidden = params.get("MLP_HIDDEN_DIMS", [32, 32])
    model = GATScheduler(
        raw_node_dim=node_raw_dim,
        mlp_hidden_dims=mlp_hidden,
        embed_node_dim=params.get("NODE_EMBED_DIM", 32),
        edge_dim=params.get("EDGE_DIM", 1),
        gat_out_dim=params.get("GAT_OUT_DIM", 48),
        num_heads=params.get("NUM_HEADS", 4),
        T_iterations=params.get("NUM_GAT_LAYERS", 4),
        schedule_hidden_dim=params.get("SCHEDULE_HIDDEN", 32),
        use_power_control=use_power_control,
        power_hidden_dim=params.get("POWER_HIDDEN_DIM", 16),
    )

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get("LR", 1e-4),
    )

    # 学习率调度器
    lr_scheduler = None
    lr_type = params.get("LR_SCHEDULER_TYPE", None)
    if lr_type == "ReduceLROnPlateau":
        lr_cfg = params.get("LR_SCHEDULER_CONFIG", {}) or {}
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_cfg)

    # 自动 GPU 检测
    gpu_id = params.get("GPU_ID", 0)
    device = None
    if torch.cuda.is_available():
        if gpu_id >= torch.cuda.device_count():
            gpu_id = 0
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[Trainer] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("[Trainer] Using CPU")

    # 创建 Trainer
    trainer = Trainer(
        model, optimizer, device=device, gpu_id=gpu_id,
        log_dir=logs_dir,
        grad_clip_config=params.get("GRAD_CLIP_CONFIG", None),
        use_power_control=use_power_control,
        use_queue=use_queue,
    )

    # 训练
    trained = trainer.train(
        num_epochs=params.get("EPOCHS", 30),
        train_dataset=train_set,
        val_dataset=val_set,
        eval_interval=params.get("EVAL_INTERVAL", 1),
        lr_scheduler=lr_scheduler,
        use_power_control=use_power_control,
        use_queue=use_queue,
    )

    # 评估
    test_result = trainer.evaluate(model, test_set, use_power_control=use_power_control, use_queue=use_queue)
    if use_power_control:
        test_util, test_alpha = test_result
    else:
        test_util = test_result
        test_alpha = None

    # 收集日志中的 alpha 统计
    per_epoch_logs = trained.get("per_epoch_logs", [])
    alpha_means = [log.get("alpha_mean") for log in per_epoch_logs if log.get("alpha_mean") is not None]
    avg_alpha_final = alpha_means[-1] if alpha_means else None

    # 更新 Excel 表头
    params["EXPERIMENT_GROUP"] = experiment_group

    # 保存结果
    info = save_experiment_results(
        trained, train_set, test_util, params,
        results_dir=results_dir, log_dir=logs_dir,
        excel_sheet="Sheet4"  # 功率控制实验
    )

    return {
        "use_power_control": use_power_control,
        "test_util": test_util,
        "test_alpha": test_alpha,
        "avg_alpha_final": avg_alpha_final,
        "best_val_util": trained.get("best_val_util", None),
        "final_train_loss": trained.get("final_train_loss", None),
        "per_epoch_logs": per_epoch_logs,
    }


def compare_baseline_vs_power_control(base_params, results_dir="./results", logs_dir="./logs"):
    """
    对比基准（无功率控制）vs 功率控制方案。

    Args:
        base_params: 基础配置参数字典
        results_dir: 结果保存目录
        logs_dir: 日志保存目录

    Returns:
        dict: 包含两个实验的结果
    """
    # 基准组：无功率控制
    print("\n" + "="*70)
    print("基准组：无功率控制 (USE_POWER_CONTROL=False)")
    print("="*70)
    baseline_result = run_power_control_experiment(
        base_params,
        use_power_control=False,
        results_dir=os.path.join(results_dir, "baseline"),
        logs_dir=os.path.join(logs_dir, "baseline"),
        experiment_group="power_control_baseline",
    )

    # 实验组：有功率控制
    print("\n" + "="*70)
    print("实验组：功率控制 (USE_POWER_CONTROL=True)")
    print("="*70)
    power_result = run_power_control_experiment(
        base_params,
        use_power_control=True,
        results_dir=os.path.join(results_dir, "power_control"),
        logs_dir=os.path.join(logs_dir, "power_control"),
        experiment_group="power_control_enabled",
    )

    # 打印对比摘要
    print("\n" + "="*70)
    print("对比摘要")
    print("="*70)
    print(f"{'指标':<25} {'无功率控制':>15} {'有功率控制':>15} {'变化':>15}")
    print("-"*70)
    print(f"{'测试效用 (Mbps)':<25} {baseline_result['test_util']:>15.2f} "
          f"{power_result['test_util']:>15.2f} "
          f"{((power_result['test_util'] - baseline_result['test_util']) / max(baseline_result['test_util'], 1e-6) * 100):>14.1f}%")
    print(f"{'最优验证效用':<25} {baseline_result['best_val_util']:>15.2f} "
          f"{power_result['best_val_util']:>15.2f}")
    if baseline_result['test_alpha'] is not None:
        print(f"{'平均功率系数 (测试)':<25} {'N/A':>15} "
              f"{power_result['test_alpha']:>15.4f}")
    if baseline_result['avg_alpha_final'] is not None:
        print(f"{'最终平均功率系数':<25} {'N/A':>15} "
              f"{power_result['avg_alpha_final']:>15.4f}")
    print("-"*70)

    return {
        "baseline": baseline_result,
        "power_control": power_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="功率控制实验：对比有/无功率控制的 GAT 调度性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本对比实验
  uv run python experiments/power_control_experiment.py --num-uav 50 --train-size 300

  # 仅运行功率控制模式
  uv run python experiments/power_control_experiment.py --num-uav 50 --train-size 300 \\
      --mode power-control-only

  # 指定 GPU
  uv run python experiments/power_control_experiment.py --num-uav 50 --gpu-id 0
        """
    )

    # 基本参数
    parser.add_argument("--config", "-c", default="configs/exp_example.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU 编号")

    # 实验模式
    parser.add_argument("--mode", choices=["compare", "power-control-only"],
                        default="compare", help="实验模式")

    # 网络参数
    parser.add_argument("--num-uav", type=int, help="无人机/链路数量")
    parser.add_argument("--train-size", type=int, help="训练样本数")
    parser.add_argument("--val-size", type=int, help="验证样本数")
    parser.add_argument("--test-size", type=int, help="测试样本数")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--csi-mode", choices=["perfect", "large_scale", "quantized"],
                        help="CSI 模式")
    parser.add_argument("--q-bits", type=int, help="量化比特数")

    # 输出目录
    parser.add_argument("--results-dir", default="./results", help="结果保存目录")
    parser.add_argument("--logs-dir", default="./logs", help="日志保存目录")

    args = parser.parse_args()

    # 加载基础配置
    base_params = load_config(args.config)
    base_params["SEED"] = args.seed
    base_params["GPU_ID"] = args.gpu_id

    # 命令行参数覆盖配置文件
    if args.num_uav is not None:
        base_params["NUM_UAV"] = args.num_uav
    if args.train_size is not None:
        base_params["TRAIN_SIZE"] = args.train_size
    if args.val_size is not None:
        base_params["VAL_SIZE"] = args.val_size
    if args.test_size is not None:
        base_params["TEST_SIZE"] = args.test_size
    if args.epochs is not None:
        base_params["EPOCHS"] = args.epochs
    if args.csi_mode is not None:
        base_params["CSI_MODE"] = args.csi_mode
    if args.q_bits is not None:
        base_params["Q_BITS"] = args.q_bits

    # 确保输出目录存在
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    # 打印实验配置
    print("="*70)
    print("功率控制实验配置")
    print("="*70)
    print(f"num_uav: {base_params.get('NUM_UAV', 'N/A')}")
    print(f"train_size: {base_params.get('TRAIN_SIZE', 'N/A')}")
    print(f"csi_mode: {base_params.get('CSI_MODE', 'perfect')}")
    print(f"epochs: {base_params.get('EPOCHS', 'N/A')}")
    print(f"seed: {args.seed}")
    print(f"gpu_id: {args.gpu_id}")
    print("="*70)

    # 根据模式运行实验
    if args.mode == "compare":
        results = compare_baseline_vs_power_control(
            base_params,
            results_dir=args.results_dir,
            logs_dir=args.logs_dir,
        )
    elif args.mode == "power-control-only":
        result = run_power_control_experiment(
            base_params,
            use_power_control=True,
            results_dir=args.results_dir,
            logs_dir=args.logs_dir,
            experiment_group="power_control_enabled",
        )
        print(f"\n功率控制实验结果:")
        print(f"  测试效用: {result['test_util']:.2f} Mbps")
        print(f"  平均功率系数: {result['test_alpha']:.4f}")

    print("\n实验完成！")


if __name__ == "__main__":
    main()
