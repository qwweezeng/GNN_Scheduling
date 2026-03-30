#!/usr/bin/env python3
"""
不完美 CSI 实验组：量化 / 大尺度衰落 对调度性能的影响。

将 large_scale_only 和 quantization_analysis 合并为一个脚本：
  - 完美 CSI 基线（CSI_MODE="perfect"）
  - 仅大尺度衰落（CSI_MODE="large_scale"）
  - 量化距离近似（CSI_MODE="quantized"，不同 q_bits）

每个实验独立走 ExperimentRunner，自动写 Excel；结束时追加汇总 sheet。

Usage:
    python experiments/csi_imperfection_study.py --gpu-id 0
"""

import os
import sys
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import yaml
from experiments.runner import ExperimentRunner


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_cache_path(num_uav, csi_mode, q_bits=None):
    if csi_mode == "perfect":
        return f"./datasets/dataset_numUAV={num_uav}.pkl"
    if csi_mode == "large_scale":
        return f"./datasets/dataset_numUAV={num_uav}_large_scale.pkl"
    return f"./datasets/dataset_numUAV={num_uav}_quantized_{q_bits}bits.pkl"


def run_single(csi_mode, num_uav, q_bits, base_params, results_dir, logs_dir, gpu_id, excel_sheet):
    params = base_params.copy()
    params["CSI_MODE"] = csi_mode
    params["NUM_UAV"] = num_uav
    params["Q_BITS"] = q_bits if q_bits else 8
    params["EXPERIMENT_GROUP"] = "csi_imperfection"
    params["DATASET_CACHE_PATH"] = make_cache_path(num_uav, csi_mode, params["Q_BITS"])

    label = {
        "perfect":      f"perfect(N={num_uav})",
        "large_scale":  f"large_scale(N={num_uav})",
        "quantized":    f"quantized_{q_bits}bits(N={num_uav})",
    }[csi_mode]

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    runner = ExperimentRunner(params, results_dir=results_dir, log_dir=logs_dir, gpu_id=gpu_id)
    res = runner.run()
    test_util = res.get("test_util", 0)
    trained = res.get("trained", {})
    best_val = trained.get("best_val_util", 0)
    print(f"  -> test_util={test_util:.4f}, best_val={best_val:.4f}")
    return {
        "csi_mode": csi_mode,
        "q_bits": params["Q_BITS"],
        "test_util": test_util,
        "best_val_util": best_val,
        "cache": params["DATASET_CACHE_PATH"],
    }


def main():
    parser = argparse.ArgumentParser(description="CSI 不完美性实验组")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--config", default="configs/exp_example.yaml")
    parser.add_argument("--num-uav", type=int, default=20)
    parser.add_argument("--quantized-bits", nargs="+", type=int,
                        default=[2, 4, 6, 8, 10],
                        help="量化比特数列表")
    args = parser.parse_args()

    base_params = load_config(args.config)
    # 统一的基础参数（覆盖 yaml）
    base_params["TRAIN_SIZE"] = 500
    base_params["VAL_SIZE"] = 128
    base_params["TEST_SIZE"] = 128
    base_params["EPOCHS"] = 30
    base_params["EXCEL_SHEET"] = "Sheet3"
    base_params["NOTES"] = "CSI不完美性实验组"

    results_dir = "./results"
    logs_dir = "./logs"
    num_uav = args.num_uav
    results = []

    # 完美 CSI 基线
    results.append(run_single(
        "perfect", num_uav, None,
        base_params, results_dir, logs_dir, args.gpu_id, "Sheet3",
    ))

    # 大尺度衰落（连续精度上限）
    results.append(run_single(
        "large_scale", num_uav, None,
        base_params, results_dir, logs_dir, args.gpu_id, "Sheet3",
    ))

    # 量化距离近似（不同精度）
    # for qb in args.quantized_bits:
    #     results.append(run_single(
    #         "quantized", num_uav, qb,
    #         base_params, results_dir, logs_dir, args.gpu_id, "Sheet3",
    #     ))

    # 汇总
    print("\n" + "="*60)
    print("  实验完成！")
    print("="*60)
    print(f"\n{'CSI_MODE':<20} {'Q_bits':>6}  {'Test Utility':>14}  {'Best Val':>14}")
    print("-"*60)
    for r in results:
        print(f"{r['csi_mode']:<20} {str(r.get('q_bits') or '-'):>6}  "
              f"{r['test_util']:>14.4f}  {r['best_val_util']:>14.4f}")


if __name__ == "__main__":
    main()
