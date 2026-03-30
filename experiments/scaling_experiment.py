#!/usr/bin/env python3
"""
Scaling Experiment: Cross-scale generalization analysis.

研究问题：
1. 在 numUAV=X 上训练的模型能否泛化到 numUAV=Y？
2. 混合多尺度数据训练能否提升跨尺度泛化能力？

Usage:
    python -m experiments.scaling_experiment --mode paired    # 配对实验
    python -m experiments.scaling_experiment --mode mixed     # 混合训练实验
    python -m experiments.scaling_experiment --mode all       # 全部实验
    python -m experiments.scaling_experiment --mode all --quick  # 快速调试
    
    # 使用已有模型进行测试（不重新训练）
    python -m experiments.scaling_experiment --mode paired --use-existing-models
    
    # 指定模型库路径
    python -m experiments.scaling_experiment --mode paired --model-library ./results/models
"""

import os
import sys
import pickle
import argparse
import torch
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.runner import ExperimentRunner
from models.model import GATScheduler, calculate_achievable_rate
from train.trainer import Trainer


def load_config(path):
    """Load configuration from YAML file."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_model_library_path(base_dir="./results/models"):
    """获取模型库路径，如果不存在则创建"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def get_model_path(model_library, num_uav, prefix="scaling"):
    """获取指定 numUAV 的模型保存路径"""
    return os.path.join(model_library, f"{prefix}_numUAV{num_uav}.pth")


def save_model(model, model_library, num_uav, prefix="scaling"):
    """保存模型到模型库"""
    model_path = get_model_path(model_library, num_uav, prefix)
    torch.save(model.state_dict(), model_path)
    print(f"  [Save] Model saved to {model_path}")
    return model_path


def load_model(model_library, num_uav, model_params, device, prefix="scaling"):
    """从模型库加载模型"""
    model_path = get_model_path(model_library, num_uav, prefix)
    if not os.path.exists(model_path):
        return None, None
    
    model = GATScheduler(
        raw_node_dim=model_params["raw_node_dim"],
        mlp_hidden_dims=model_params["mlp_hidden_dims"],
        embed_node_dim=model_params["embed_node_dim"],
        edge_dim=model_params["edge_dim"],
        gat_out_dim=model_params["gat_out_dim"],
        num_heads=model_params["num_heads"],
        T_iterations=model_params["T_iterations"],
        schedule_hidden_dim=model_params["schedule_hidden_dim"],
        dropout=model_params["dropout"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"  [Load] Model loaded from {model_path}")
    return model, model_path


def evaluate_model(model, test_set, device):
    """评估模型在测试集上的性能"""
    model.eval()
    total_util = 0.0
    with torch.no_grad():
        for state in test_set:
            x = state.vertex_features.to(device)
            edge_index = state.edge_index.to(device)
            edge_weights = state.edge_weights.to(device)
            pi, _ = model(x, edge_index, edge_weights, use_ste=False)
            rates = calculate_achievable_rate(pi, state, binary=True, detach_hard=True)
            util = torch.sum(rates).item()
            total_util += util
    test_util = total_util / len(test_set) if len(test_set) > 0 else 0.0
    return test_util


def run_paired_experiment(config_path, train_sizes, test_sizes, quick, gpu_id, 
                          use_existing_models=False, model_library=None):
    """
    配对实验：在 numUAV=X 上训练，在所有尺度上测试。

    流程：
    1. 对每个 train_size，训练一个模型（或加载已有模型）
    2. 对每个 test_size，从缓存加载测试集，使用模型评估
    """
    base_cfg = load_config(config_path)

    # 快速模式参数
    epochs = 3 if quick else base_cfg.get("EPOCHS", 30)
    train_per_scale = 20 if quick else base_cfg.get("TRAIN_SIZE", 200)
    val_per_scale = 10 if quick else base_cfg.get("VAL_SIZE", 20)

    # 通用配置
    base_cfg["EPOCHS"] = epochs
    base_cfg["EVAL_BASELINES"] = False
    base_cfg["EXPERIMENT_GROUP"] = "scaling_paired"

    # 模型架构参数（从配置文件读取，必要时提供默认值）
    # 注意：NODE_RAW_DIM 在 YAML 中为 null，会被解析为 None，需要特殊处理
    raw_node_dim = base_cfg.get("NODE_RAW_DIM")
    if raw_node_dim is None:
        raw_node_dim = 1  # 默认使用 1 维特征（SNR）
    
    model_params = {
        "raw_node_dim": raw_node_dim,
        "mlp_hidden_dims": base_cfg.get("MLP_HIDDEN_DIMS", [32, 32]),
        "embed_node_dim": base_cfg.get("NODE_EMBED_DIM", 32),
        "edge_dim": base_cfg.get("EDGE_DIM", 1),
        "gat_out_dim": base_cfg.get("GAT_OUT_DIM", 48),
        "num_heads": base_cfg.get("NUM_HEADS", 4),
        "T_iterations": base_cfg.get("NUM_GAT_LAYERS", 4),
        "schedule_hidden_dim": base_cfg.get("SCHEDULE_HIDDEN", 32),
        "dropout": base_cfg.get("DROPOUT", 0.0)
    }

    # 确定 device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # 模型库路径
    if model_library is None:
        model_library = get_model_library_path()
    else:
        model_library = get_model_library_path(model_library)

    all_results = []

    print("=" * 60)
    print("SCALING EXPERIMENT (Paired): Train X → Test Y")
    print("=" * 60)
    print(f"Train sizes: {train_sizes}, Test sizes: {test_sizes}")
    print(f"Use existing models: {use_existing_models}")
    print(f"Model library: {model_library}")

    for train_size in train_sizes:
        print(f"\n[Train] numUAV={train_size}...")

        # 配置训练 - 关键：修改 DATASET_CACHE_PATH 为对应 numUAV 的数据
        train_cfg = base_cfg.copy()
        train_cfg["NUM_UAV"] = train_size
        train_cfg["TRAIN_SIZE"] = train_per_scale
        train_cfg["VAL_SIZE"] = val_per_scale
        train_cfg["NOTES"] = f"scaling_paired_{train_size}"
        
        # 【关键修复】动态设置 DATASET_CACHE_PATH，使其与 NUM_UAV 匹配
        dataset_cache_path = f"./datasets/dataset_numUAV={train_size}.pkl"
        train_cfg["DATASET_CACHE_PATH"] = dataset_cache_path
        print(f"  [Config] DATASET_CACHE_PATH = {dataset_cache_path}")

        # 检查是否使用已有模型
        trained_model = None
        best_val_util = None
        
        if use_existing_models:
            trained_model, model_path = load_model(model_library, train_size, model_params, device)
            if trained_model is not None:
                print(f"  [Skip] Using existing model from {model_path}")
            else:
                print(f"  [Warning] No existing model found, will train new one")

        # 如果没有已有模型，则训练
        if trained_model is None:
            # 训练模型
            runner = ExperimentRunner(train_cfg, gpu_id=gpu_id)
            train_out = runner.run()
            trained = train_out["trained"]
            trained_model = trained["model"]
            best_val_util = trained["best_val_util"]
            
            # 保存模型到模型库
            save_model(trained_model, model_library, train_size)
            print(f"  Best val utility: {best_val_util:.2f}")

        # Phase 2: 在所有测试尺度上评估
        print(f"[Test] Testing on all scales...")

        trained_model.eval()
        trained_model.to(device)

        for test_size in test_sizes:
            # 从缓存加载测试集
            cache_path = f"./datasets/dataset_numUAV={test_size}.pkl"
            if not os.path.exists(cache_path):
                print(f"  [Error] {cache_path} not found, generating new dataset...")
                # 如果测试数据集不存在，需要临时生成
                test_cfg = base_cfg.copy()
                test_cfg["NUM_UAV"] = test_size
                test_cfg["DATASET_CACHE_PATH"] = cache_path
                test_runner = ExperimentRunner(test_cfg, gpu_id=gpu_id)
                # 只生成数据，不训练
                from utils.io import generate_dataset
                from utils.network_generate import NetworkGenerator
                height_range = test_cfg.get("HEIGHT_RANGE", (50, 150))
                gen = NetworkGenerator(
                    num_uavs=test_cfg.get("NUM_UAV", test_size),
                    area_width=test_cfg.get("AREA_WIDTH", 1000),
                    height_min=height_range[0],
                    height_max=height_range[1],
                    uav_min_distance=test_cfg.get("MIN_DISTANCE", 20),
                    tx_power=test_cfg.get("TX_POWER", 10),
                    Gt=test_cfg.get("G_T", 0),
                    Gr=test_cfg.get("G_R", 0),
                    bandwidth=test_cfg.get("BANDWIDTH", 5e6),
                    carrier_f=test_cfg.get("CARRIER_FREQ", 2.4e9),
                    noise_psd=test_cfg.get("NOISE_PSD", -169),
                    snr_gap_db=test_cfg.get("SNR_GAP_DB", 0)
                )
                csi_mode = test_cfg.get("CSI_MODE", "perfect")
                q_bits = test_cfg.get("Q_BITS", 8)
                test_set = generate_dataset(
                    test_cfg.get("TEST_SIZE", 20), gen,
                    batch_size=test_cfg.get("GEN_BATCH_SIZE", 50),
                    csi_mode=test_cfg.get("CSI_MODE", "perfect"),
                    q_bits=q_bits,
                )
            else:
                with open(cache_path, "rb") as f:
                    test_set = pickle.load(f)["test_set"]

            # 使用训练好的模型直接评估
            test_util = evaluate_model(trained_model, test_set, device)

            all_results.append({
                "train_size": train_size,
                "test_size": test_size,
                "test_util": test_util,
                "best_val_util": best_val_util
            })
            print(f"    numUAV={test_size}: {test_util:.2f}")

    # 创建结果 DataFrame 和矩阵
    df = pd.DataFrame(all_results)
    matrix = df.pivot(index="train_size", columns="test_size", values="test_util")

    print("\n" + "=" * 60)
    print("PAIRED RESULTS")
    print("=" * 60)
    print(matrix.to_string())

    return df, matrix


def run_mixed_experiment(config_path, train_sizes, test_sizes, quick, gpu_id, 
                        use_existing_models=False, model_library=None):
    """
    混合训练实验：在多个尺度的组合数据上训练，测试所有尺度。

    流程：
    1. 从缓存加载各尺度的训练数据
    2. 合并为一个混合训练集
    3. 训练一个模型（或加载已有模型）
    4. 在所有测试尺度上评估
    """
    base_cfg = load_config(config_path)

    epochs = 3 if quick else base_cfg.get("EPOCHS", 30)
    samples_per_scale = 50 if quick else min(125, base_cfg.get("TRAIN_SIZE", 200))

    # 模型架构参数
    # 注意：NODE_RAW_DIM 在 YAML 中为 null，会被解析为 None，需要特殊处理
    raw_node_dim = base_cfg.get("NODE_RAW_DIM")
    if raw_node_dim is None:
        raw_node_dim = 1  # 默认使用 1 维特征（SNR）
    
    model_params = {
        "raw_node_dim": raw_node_dim,
        "mlp_hidden_dims": base_cfg.get("MLP_HIDDEN_DIMS", [32, 32]),
        "embed_node_dim": base_cfg.get("NODE_EMBED_DIM", 32),
        "edge_dim": base_cfg.get("EDGE_DIM", 1),
        "gat_out_dim": base_cfg.get("GAT_OUT_DIM", 48),
        "num_heads": base_cfg.get("NUM_HEADS", 4),
        "T_iterations": base_cfg.get("NUM_GAT_LAYERS", 4),
        "schedule_hidden_dim": base_cfg.get("SCHEDULE_HIDDEN", 32),
        "dropout": base_cfg.get("DROPOUT", 0.0)
    }

    # 确定 device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # 模型库路径
    if model_library is None:
        model_library = get_model_library_path()
    else:
        model_library = get_model_library_path(model_library)

    all_results = []

    print("\n" + "=" * 60)
    print("SCALING EXPERIMENT (Mixed): Combined Training")
    print("=" * 60)
    print(f"Train sizes: {train_sizes}, Test sizes: {test_sizes}")
    print(f"Use existing models: {use_existing_models}")

    # Phase 1: 加载并合并训练数据
    print("\n[Phase 1] Loading and combining datasets...")
    combined_train = []
    combined_val = []

    for size in train_sizes:
        cache_path = f"./datasets/dataset_numUAV={size}.pkl"
        if not os.path.exists(cache_path):
            print(f"  [Error] {cache_path} not found")
            continue

        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        n_train = min(samples_per_scale, len(data["train_set"]))
        n_val = min(32, len(data["val_set"]))

        combined_train.extend(data["train_set"][:n_train])
        combined_val.extend(data["val_set"][:n_val])
        print(f"  numUAV={size}: {n_train} train, {n_val} val")

    print(f"  Total: {len(combined_train)} train, {len(combined_val)} val")

    # 生成混合训练的模型标识
    mixed_name = "_".join(map(str, train_sizes))

    # Phase 2: 检查是否使用已有模型
    trained_model = None
    best_val_util = None
    
    if use_existing_models:
        # 尝试加载混合模型
        model_path = os.path.join(model_library, f"mixed_numUAV{mixed_name}.pth")
        if os.path.exists(model_path):
            trained_model = GATScheduler(
                raw_node_dim=model_params["raw_node_dim"],
                mlp_hidden_dims=model_params["mlp_hidden_dims"],
                embed_node_dim=model_params["embed_node_dim"],
                edge_dim=model_params["edge_dim"],
                gat_out_dim=model_params["gat_out_dim"],
                num_heads=model_params["num_heads"],
                T_iterations=model_params["T_iterations"],
                schedule_hidden_dim=model_params["schedule_hidden_dim"],
                dropout=model_params["dropout"]
            )
            trained_model.load_state_dict(torch.load(model_path, map_location=device))
            trained_model.to(device)
            trained_model.eval()
            print(f"  [Load] Mixed model loaded from {model_path}")
        else:
            print(f"  [Warning] No existing mixed model found at {model_path}, will train new one")

    # Phase 3: 如果没有已有模型，则训练
    if trained_model is None:
        print(f"\n[Phase 2] Training on combined dataset ({epochs} epochs)...")

        model = GATScheduler(
            raw_node_dim=model_params["raw_node_dim"],
            mlp_hidden_dims=model_params["mlp_hidden_dims"],
            embed_node_dim=model_params["embed_node_dim"],
            edge_dim=model_params["edge_dim"],
            gat_out_dim=model_params["gat_out_dim"],
            num_heads=model_params["num_heads"],
            T_iterations=model_params["T_iterations"],
            schedule_hidden_dim=model_params["schedule_hidden_dim"],
            dropout=model_params["dropout"]
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=base_cfg.get("LR", 1e-4))
        trainer = Trainer(model, optimizer, gpu_id=gpu_id)

        trained = trainer.train(
            num_epochs=epochs,
            train_dataset=combined_train,
            val_dataset=combined_val,
            lambda_sparsity=base_cfg.get("LAMBDA_SPARSITY", 0.01),
            lambda_conflict=base_cfg.get("LAMBDA_CONFLICT", 0.1)
        )
        trained_model = trained["model"]
        best_val_util = trained["best_val_util"]

        # 保存混合模型
        mixed_model_path = os.path.join(model_library, f"mixed_numUAV{mixed_name}.pth")
        torch.save(trained_model.state_dict(), mixed_model_path)
        print(f"  [Save] Mixed model saved to {mixed_model_path}")
        print(f"  Best val utility: {best_val_util:.2f}")

    # Phase 4: 测试所有尺度
    print(f"\n[Phase 3] Testing on all scales...")

    trained_model.eval()
    trained_model.to(device)

    for test_size in test_sizes:
        cache_path = f"./datasets/dataset_numUAV={test_size}.pkl"
        if not os.path.exists(cache_path):
            print(f"  [Error] {cache_path} not found")
            continue

        with open(cache_path, "rb") as f:
            test_set = pickle.load(f)["test_set"]

        # 直接评估模型
        test_util = evaluate_model(trained_model, test_set, device)

        all_results.append({
            "train_scales": str(train_sizes),
            "test_size": test_size,
            "test_util": test_util,
            "best_val_util": best_val_util,
            "total_samples": len(combined_train)
        })
        print(f"  numUAV={test_size}: {test_util:.2f}")

    df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("MIXED RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))

    return df


def save_results(paired_df, paired_matrix, mixed_df):
    """保存结果到 Excel 文件。"""
    output_path = "experiments/scaling_results.xlsx"

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        if paired_df is not None:
            paired_df.to_excel(writer, sheet_name="Paired_Results", index=False)
        if paired_matrix is not None:
            paired_matrix.to_excel(writer, sheet_name="Paired_Matrix")
        if mixed_df is not None:
            mixed_df.to_excel(writer, sheet_name="Mixed_Results", index=False)

    print(f"\n[Save] Results saved to {output_path}")
    return output_path


def list_models(model_library):
    """列出模型库中已有的模型"""
    if not os.path.exists(model_library):
        print(f"Model library {model_library} does not exist.")
        return
    
    print(f"\nExisting models in {model_library}:")
    for f in os.listdir(model_library):
        if f.endswith(".pth"):
            print(f"  - {f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Scaling experiment")
    parser.add_argument("--mode", choices=["paired", "mixed", "all"], default="all")
    parser.add_argument("--config", default="configs/exp_example.yaml")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20, 30, 40, 50])
    parser.add_argument("--test-sizes", type=int, nargs="+", default=[20, 30, 40, 50])
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer epochs and samples")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--use-existing-models", action="store_true", 
                        help="Use existing models from model library instead of training new ones")
    parser.add_argument("--model-library", type=str, default="./results/models",
                        help="Path to model library directory")
    parser.add_argument("--list-models", action="store_true",
                        help="List existing models in the model library")
    args = parser.parse_args()

    # GPU 配置
    cfg = load_config(args.config)
    gpu_id = args.gpu_id if args.gpu_id is not None else cfg.get("GPU_ID", 0)

    # 列出已有模型
    if args.list_models:
        list_models(args.model_library)
        return

    paired_df = paired_matrix = mixed_df = None

    if args.mode in ["paired", "all"]:
        paired_df, paired_matrix = run_paired_experiment(
            args.config, args.train_sizes, args.test_sizes, args.quick, gpu_id,
            use_existing_models=args.use_existing_models,
            model_library=args.model_library
        )

    if args.mode in ["mixed", "all"]:
        mixed_df = run_mixed_experiment(
            args.config, args.train_sizes, args.test_sizes, args.quick, gpu_id,
            use_existing_models=args.use_existing_models,
            model_library=args.model_library
        )

    save_results(paired_df, paired_matrix, mixed_df)


if __name__ == "__main__":
    main()
