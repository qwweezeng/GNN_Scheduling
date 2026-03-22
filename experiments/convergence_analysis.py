#!/usr/bin/env python3
"""
收敛性分析实验：研究不同训练样本数量(train_size)对模型性能的影响
用于确定最少需要多少训练样本才能达到稳定的性能表现

Usage:
    python experiments/convergence_analysis.py
"""

import os
import yaml
import json
import torch
import numpy as np
from time import strftime
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_root)

from experiments.runner import ExperimentRunner

def load_config(path):
    """Load configuration from YAML or JSON file."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def run_convergence_analysis(config_path, train_sizes, results_dir="./results", logs_dir="./logs", gpu_id=0):
    """
    运行收敛性分析实验，测试不同训练样本数量下的性能

    Args:
        config_path: 配置文件路径
        train_sizes: 要测试的训练样本数量列表
        results_dir: 结果保存目录
        logs_dir: 日志保存目录
    """
    print(f"{'='*60}")
    print("收敛性分析实验：研究不同训练样本数量对模型性能的影响")
    print(f"{'='*60}")
    print(f"测试的训练样本数量: {train_sizes}")

    # 存储所有实验结果
    all_results = []

    # Load base configuration
    base_params = load_config(config_path).copy()
    base_params["EXPERIMENT_GROUP"] = "convergence_analysis"
    base_params["NOTES"] = "收敛性分析：不同训练样本数量的影响"

    for i, train_size in enumerate(train_sizes):
        print(f"\n{'-'*50}")
        print(f"运行实验 {i+1}/{len(train_sizes)}: train_size = {train_size}")
        print(f"{'-'*50}")

        # 复制基础配置并修改训练样本数量
        params = base_params.copy()
        params["TRAIN_SIZE"] = train_size
        params["NOTES"] = f"收敛性分析_train_size={train_size}"
        params["EXCEL_SHEET"] = "Sheet2"  # 收敛性分析实验放在Sheet2中

        # 设置种子以保证可重复性
        params["SEED"] = 10086  # 为每个实验使用不同的种子

        # 运行实验（不捕获异常，遇到错误直接中止以便快速定位）
        runner = ExperimentRunner(params, results_dir=results_dir, log_dir=logs_dir, gpu_id=gpu_id)
        res = runner.run()

        # 提取结果
        trained = res.get('trained', {})
        test_util = res.get('test_util', 0)

        # 收集实验结果
        result = {
            "train_size": train_size,
            "test_utility": test_util,
            "best_val_utility": trained.get("best_val_util", 0),
            "final_train_loss": trained.get("final_train_loss", 0),
            "experiment_id": res.get("experiment_id", ""),
            "seed": params["SEED"],
            "convergence_info": {
                "epochs": len(trained.get("per_epoch_logs", [])),
                "best_epoch": None
            }
        }

            # 分析收敛情况
        per_epoch_logs = trained.get("per_epoch_logs", [])
        if per_epoch_logs:
            best_val_util = trained.get("best_val_util", 0)
            # 找到最佳验证性能的epoch
            for epoch_log in per_epoch_logs:
                if abs(epoch_log.get("val_utility", 0) - best_val_util) < 1e-3:
                    result["convergence_info"]["best_epoch"] = epoch_log.get("epoch")
                    break

            # 分析收敛稳定性（最后10个epoch的性能方差）
            recent_epochs = per_epoch_logs[-min(10, len(per_epoch_logs)):]
            recent_val_utils = [log.get("val_utility", 0) for log in recent_epochs if log.get("val_utility") is not None]
            if recent_val_utils:
                result["convergence_info"]["stability_std"] = np.std(recent_val_utils)
                result["convergence_info"]["stability_mean"] = np.mean(recent_val_utils)

        all_results.append(result)
        print(f"train_size={train_size} -> test_util={test_util:.4f}, best_val={result['best_val_utility']:.4f}")

    # 结果已通过Excel保存，无需额外保存JSON
    print(f"\n{'='*60}")
    print("收敛性分析实验完成！")
    print(f"结果已保存到: {results_dir}/all_experiments_summary.xlsx (Sheet2)")
    print(f"{'='*60}")

    # 打印简要结果摘要
    successful_results = [r for r in all_results if r.get("test_utility") is not None]
    if successful_results:
        print("\n结果摘要:")
        print("Train Size | Test Utility | Best Val Utility | Final Loss")
        print("-" * 55)
        for result in sorted(successful_results, key=lambda x: x["train_size"]):
            print("6d")

    return all_results

def main():
    # 添加简单的CLI支持，主要用于gpu选择
    parser = argparse.ArgumentParser(description="收敛性分析实验：研究不同训练样本数量对模型性能的影响")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()

    # 直接在代码中配置参数，无需CLI
    config_path = "configs/exp_example.yaml"
    train_sizes = [20,80]  # 测试的训练样本数量
    results_dir = "./results"
    logs_dir = "./logs"

    # 运行收敛性分析
    run_convergence_analysis(
        config_path=config_path,
        train_sizes=train_sizes,
        results_dir=results_dir,
        logs_dir=logs_dir,
        gpu_id=args.gpu_id
    )

if __name__ == "__main__":
    main()
