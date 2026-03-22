#!/usr/bin/env python3
"""
量化分析实验：研究不同量化比特位数对距离特征的影响
测试量化比特数：2, 4, 6, 8, 10, 12, 14, 16
每次实验重新生成对应的数据集，存储在datasets下
实验结果存储在all_experiments_summary.xlsx的Sheet3

Usage:
    python experiments/quantization_analysis.py --gpu-id 0
"""

import os
import yaml
import json
import torch
import numpy as np
import argparse
from time import strftime

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

def run_quantization_analysis(config_path, quantization_bits, results_dir="./results",
                            logs_dir="./logs", gpu_id=0):
    """
    运行量化分析实验，测试不同量化比特位数对模型性能的影响

    Args:
        config_path: 配置文件路径
        quantization_bits: 要测试的量化比特位数列表
        results_dir: 结果保存目录
        logs_dir: 日志保存目录
        gpu_id: GPU ID
    """
    print(f"{'='*60}")
    print("量化分析实验：研究不同量化比特位数对距离特征的影响")
    print(f"{'='*60}")
    print(f"测试的量化比特位数: {quantization_bits}")

    # 存储所有实验结果
    all_results = []

    # Load base configuration
    base_params = load_config(config_path).copy()
    base_params["EXPERIMENT_GROUP"] = "quantization_analysis"
    base_params["NOTES"] = "量化分析：不同量化比特位数的影响"
    base_params["TRAIN_SIZE"] = 400
    # base_params["VAL_SIZE"] = 2
    # base_params["TEST_SIZE"] = 2

    for i, q_bits in enumerate(quantization_bits):
        print(f"\n{'-'*50}")
        print(f"运行实验 {i+1}/{len(quantization_bits)}: 量化比特位数 = {q_bits}")
        print(f"{'-'*50}")

        # 复制基础配置并修改量化比特位数
        params = base_params.copy()
        params["Q_BITS"] = q_bits
        params["USE_QUANTIZED_DISTANCE"] = True  # 启用量化距离
        params["EXCEL_SHEET"] = "Sheet3"  # 量化分析实验放在Sheet3中

        # 设置种子以保证可重复性
        params["SEED"] = 10086  # 为每个实验使用不同的种子

        # 为每次实验生成新的数据集缓存路径
        dataset_cache_path = f"./datasets/dataset_numUAV=50_qbits={q_bits}.pkl"
        params["DATASET_CACHE_PATH"] = dataset_cache_path

        # 运行实验（不捕获异常，遇到错误直接中止以便快速定位）
        runner = ExperimentRunner(params, results_dir=results_dir, log_dir=logs_dir, gpu_id=gpu_id)
        res = runner.run()

        # 提取结果
        trained = res.get('trained', {})
        test_util = res.get('test_util', 0)

        # 收集实验结果
        result = {
            "quantization_bits": q_bits,
            "test_utility": test_util,
            "best_val_utility": trained.get("best_val_util", 0),
            "final_train_loss": trained.get("final_train_loss", 0),
            "experiment_id": res.get("experiment_id", ""),
            "seed": params["SEED"],
            "dataset_cache": dataset_cache_path,
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
        print(f"quantization_bits={q_bits} -> test_util={test_util:.4f}, best_val={result['best_val_utility']:.4f}")

    print(f"\n{'='*60}")
    print("量化分析实验完成！")
    print(f"结果已保存到: {results_dir}/all_experiments_summary.xlsx (Sheet3)")
    print(f"{'='*60}")

    # 打印简要结果摘要
    successful_results = [r for r in all_results if r.get("test_utility") is not None]
    if successful_results:
        print("\n结果摘要:")
        print("Q-Bits | Test Utility | Best Val Utility | Final Loss")
        print("-" * 55)
        for result in sorted(successful_results, key=lambda x: x["quantization_bits"]):
            print("6d")

    return all_results

def main():
    # 添加简单的CLI支持，主要用于gpu选择
    parser = argparse.ArgumentParser(description="量化分析实验：研究不同量化比特位数对距离特征的影响")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()

    # 直接在代码中配置参数，无需CLI
    config_path = "configs/exp_example.yaml"
    quantization_bits = [12, 16]  # 测试的量化比特位数
    results_dir = "./results"
    logs_dir = "./logs"

    # 运行量化分析
    run_quantization_analysis(
        config_path=config_path,
        quantization_bits=quantization_bits,
        results_dir=results_dir,
        logs_dir=logs_dir,
        gpu_id=args.gpu_id
    )

if __name__ == "__main__":
    main()