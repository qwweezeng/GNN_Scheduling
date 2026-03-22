#!/usr/bin/env python3
"""
Generalization experiment: Test how well models trained on small-scale networks
(e.g., numUAV=20) perform on larger-scale networks (e.g., numUAV=50).

This experiment evaluates the scalability and generalization capability of the
GNN-based scheduling model.

Key research questions:
1. Does training on small networks generalize to larger networks?
2. What is the performance gap between same-scale and cross-scale evaluation?
3. How does the performance scale with network size?

Usage:
    python experiments/generalization_experiment.py --config configs/exp_example.yaml
"""
import os
import copy
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_root)

from experiments.runner import ExperimentRunner
from train.trainer import Trainer
from models.model import GATScheduler, calculate_achievable_rate
from utils.baseline import eval_baseline
from utils.io import save_experiment_results


def load_config(path):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_generalization_experiment(
    config_path,
    train_sizes=[20],           # 训练用的网络规模列表
    test_sizes=[20, 30, 40, 50], # 测试用的网络规模列表
    results_dir="./results",
    logs_dir="./logs",
    quick=False,
    gpu_id=0
):
    """
    Run generalization experiment.
    
    Args:
        config_path: Path to base YAML config
        train_sizes: List of UAV counts for training (small-scale networks)
        test_sizes: List of UAV counts for testing (various scales)
        results_dir: Directory to save results
        logs_dir: Directory to save logs
        quick: If True, use small dataset sizes for debugging
        gpu_id: GPU ID to use
    """
    base_cfg = load_config(config_path)
    
    # Quick mode: reduce dataset size and epochs
    if quick:
        base_cfg["TRAIN_SIZE"] = 50
        base_cfg["VAL_SIZE"] = 10
        base_cfg["TEST_SIZE"] = 20
        base_cfg["EPOCHS"] = 3
    
    base_cfg["EXPERIMENT_GROUP"] = "generalization"
    base_cfg["EXCEL_SHEET"] = "Sheet5"
    base_cfg["EVAL_BASELINES"] = True
    
    # Generate unique experiment ID
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    results = []
    
    print("=" * 70)
    print("Generalization Experiment: Small-scale Training → Large-scale Testing")
    print("=" * 70)
    
    for train_size in train_sizes:
        print(f"\n{'='*60}")
        print(f"Training on small-scale network: numUAV={train_size}")
        print(f"{'='*60}")
        
        # Create config for training
        train_cfg = copy.deepcopy(base_cfg)
        train_cfg["NUM_UAV"] = train_size
        train_cfg["DATASET_CACHE_PATH"] = f"./datasets/dataset_numUAV={train_size}_generalization.pkl"
        train_cfg["NOTES"] = f"generalization_train_{train_size}"
        
        # Adjust dataset sizes based on network scale (larger networks need more samples)
        if not quick:
            train_cfg["TRAIN_SIZE"] = max(200, train_size * 10)
            train_cfg["VAL_SIZE"] = max(20, train_size // 2)
            train_cfg["TEST_SIZE"] = max(30, train_size)
        
        # Train the model
        train_runner = ExperimentRunner(
            train_cfg,
            results_dir=results_dir,
            log_dir=logs_dir,
            gpu_id=gpu_id
        )
        train_out = train_runner.run()
        trained_model = train_out.get("model")
        
        if trained_model is None:
            print(f"[generalization] WARNING: Training failed for numUAV={train_size}")
            continue
        
        trainer = train_runner.trainer
        device = trainer.device
        
        # Get the best model path if available
        best_model_path = None
        trained = train_out.get("trained", {})
        per_epoch_logs = trained.get("per_epoch_logs", [])
        if per_epoch_logs:
            best_epoch = trained.get("best_epoch", 0)
            best_model_path = os.path.join(logs_dir, "models", f"best_model_epoch{best_epoch}.pth")
        
        # Test on different scales
        print(f"\n[generalization] Testing generalization to different scales:")
        
        for test_size in test_sizes:
            print(f"\n  Testing on numUAV={test_size}...")
            
            # Generate or load test dataset for this scale
            test_cfg = copy.deepcopy(base_cfg)
            test_cfg["NUM_UAV"] = test_size
            test_cfg["DATASET_CACHE_PATH"] = f"./datasets/dataset_numUAV={test_size}_generalization_test.pkl"
            test_cfg["TRAIN_SIZE"] = 0  # Don't need training data
            test_cfg["VAL_SIZE"] = max(20, test_size)  # Use val set as test set
            test_cfg["TEST_SIZE"] = 0
            
            test_runner = ExperimentRunner(
                test_cfg,
                results_dir=results_dir,
                log_dir=logs_dir,
                gpu_id=gpu_id
            )
            test_out = test_runner.run()
            test_set = test_runner.test_set
            
            # Evaluate on test set
            test_util = trainer.evaluate(trained_model, test_set, device=device)
            
            # Run baselines for comparison
            baseline_results = eval_baseline(test_set)
            
            # Calculate performance ratios
            random_util = baseline_results.get('random', 0)
            greedy_util = baseline_results.get('simple_greedy', 0)
            
            # Performance improvement over baselines
            if random_util > 0:
                vs_random = (test_util / random_util - 1) * 100  # % improvement
            else:
                vs_random = float('nan')
            
            if greedy_util > 0:
                vs_greedy = (test_util / greedy_util - 1) * 100
            else:
                vs_greedy = float('nan')
            
            result = {
                "train_size": train_size,
                "test_size": test_size,
                "scale_ratio": test_size / train_size,
                "test_util": test_util,
                "random_baseline": random_util,
                "greedy_baseline": greedy_util,
                "vs_random_%": vs_random,
                "vs_greedy_%": vs_greedy,
                "best_val_util": trained.get("best_val_util", None),
                "final_train_loss": trained.get("final_train_loss", None),
                "scale_gap": test_util - random_util  # Absolute gap from random
            }
            results.append(result)
            
            print(f"    Test Utility: {test_util:.2f}")
            print(f"    Random Baseline: {random_util:.2f}")
            print(f"    Greedy Baseline: {greedy_util:.2f}")
            print(f"    vs Random: {vs_random:+.1f}%")
            print(f"    vs Greedy: {vs_greedy:+.1f}%")
        
        # Clean up model
        del trained_model
    
    # reference Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    os.makedirs("experiments", exist_ok=True)
    out_path = os.path.join("experiments", "generalization_results.xlsx")
    df.to_excel(out_path, index=False, sheet_name="generalization")
    
    # Print summary
    print("\n" + "=" * 70)
    print("GENERALIZATION EXPERIMENT SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Analysis: Performance vs scale
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)
    
    for train_size in train_sizes:
        train_results = df[df["train_size"] == train_size]
        print(f"\n[Train on numUAV={train_size}]")
        
        # Performance on same scale
        same_scale = train_results[train_results["test_size"] == train_size]
        if not same_scale.empty:
            same_util = same_scale["test_util"].values[0]
            print(f"  Same-scale performance: {same_util:.2f}")
        
        # Performance degradation as scale increases
        for _, row in train_results.iterrows():
            if row["test_size"] > train_size:
                degradation = (row["test_util"] / same_util - 1) * 100 if same_util > 0 else float('nan')
                print(f"  → numUAV={int(row['test_size'])}: {row['test_util']:.2f} ({degradation:+.1f}%)")
    
    print(f"\n[Results saved to: {out_path}]")
    return df


def create_generalization_summary_plots(df):
    """
    Create visualization plots for generalization results.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Test utility vs test scale for different training scales
        ax1 = axes[0]
        for train_size in df["train_size"].unique():
            subset = df[df["train_size"] == train_size].sort_values("test_size")
            ax1.plot(subset["test_size"], subset["test_util"], 
                    marker='o', label=f'Train on {train_size}')
        ax1.set_xlabel('Test Network Size (numUAV)')
        ax1.set_ylabel('Test Utility')
        ax1.set_title('Generalization: Test Utility vs Network Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Performance ratio vs scale ratio
        ax2 = axes[1]
        for train_size in df["train_size"].unique():
            subset = df[df["train_size"] == train_size].sort_values("test_size")
            # Normalize by same-scale performance
            same_scale_util = subset[subset["test_size"] == train_size]["test_util"].values
            if len(same_scale_util) > 0:
                normalized = subset["test_util"] / same_scale_util[0]
                ax2.plot(subset["scale_ratio"], normalized, 
                        marker='s', label=f'Train on {train_size}')
        ax2.set_xlabel('Scale Ratio (Test/Train)')
        ax2.set_ylabel('Normalized Performance')
        ax2.set_title('Performance Retention vs Scale Ratio')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison with baselines
        ax3 = axes[2]
        test_sizes = sorted(df["test_size"].unique())
        x = np.arange(len(test_sizes))
        width = 0.25
        
        for i, train_size in enumerate(df["train_size"].unique()):
            subset = df[df["train_size"] == train_size]
            subset = subset.set_index("test_size").reindex(test_sizes)
            ax3.bar(x + i * width, subset["test_util"], width, 
                   label=f'Train on {train_size}')
        
        # Add baseline comparison
        random_utils = df[df["train_size"] == df["train_size"].unique()[0]].groupby("test_size")["random_baseline"].first()
        ax3.plot(x, random_utils.values, 'k--', marker='x', label='Random', linewidth=2)
        
        ax3.set_xlabel('Test Network Size (numUAV)')
        ax3.set_ylabel('Utility')
        ax3.set_title('Comparison with Baselines')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(test_sizes)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("experiments/generalization_plots.png", dpi=150)
        print("[generalization] Plots saved to experiments/generalization_plots.png")
        
    except ImportError:
        print("[generalization] matplotlib not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(
        description="Generalization experiment: Test small-scale trained models on larger networks"
    )
    parser.add_argument("--config", "-c", required=True, help="Base YAML config file")
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[20, 30],
                       help="UAV counts for training (default: 20 30)")
    parser.add_argument("--test-sizes", type=int, nargs="+", default=[20, 30, 40, 50],
                       help="UAV counts for testing (default: 20 30 40 50)")
    parser.add_argument("--quick", action="store_true", 
                       help="Debug mode with small dataset sizes")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate summary plots (requires matplotlib)")
    
    args = parser.parse_args()
    
    df = run_generalization_experiment(
        config_path=args.config,
        train_sizes=args.train_sizes,
        test_sizes=args.test_sizes,
        quick=args.quick,
        gpu_id=args.gpu_id
    )
    
    if args.plot:
        create_generalization_summary_plots(df)


if __name__ == "__main__":
    main()
