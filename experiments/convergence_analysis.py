#!/usr/bin/env python3
"""
收敛性分析实验：研究不同训练样本数量(train_size)对模型性能的影响

Usage:
    uv run python experiments/convergence_analysis.py
    uv run python experiments/convergence_analysis.py --train_sizes 20 50 100 200
"""

import os
import sys
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.runner import ExperimentRunner


def run_convergence_analysis(cfg, train_sizes):
    all_results = []
    print(f"{'=' * 60}")
    print("收敛性分析实验")
    print(f"{'=' * 60}")
    print(f"测试的训练样本数量: {train_sizes}")

    for i, train_size in enumerate(train_sizes):
        print(f"\n{'-' * 50}")
        print(f"运行实验 {i+1}/{len(train_sizes)}: train_size = {train_size}")

        params = cfg.copy()
        params["TRAIN_SIZE"] = train_size
        params["NOTES"] = f"收敛性分析_train_size={train_size}"
        params["EXPERIMENT_GROUP"] = "convergence_analysis"
        params["EXCEL_SHEET"] = "Sheet2"
        params["SEED"] = cfg["SEED"] + i

        # 自动推导 NODE_RAW_DIM（与 runner.py 保持一致）
        use_queue = params.get("USE_QUEUE", False)
        if params.get("NODE_RAW_DIM") is not None:
            node_raw_dim = params.get("NODE_RAW_DIM")
        else:
            node_raw_dim = 2 if use_queue else 1
        params["NODE_RAW_DIM"] = node_raw_dim

        runner = ExperimentRunner(
            params,
            results_dir=params.get("RESULTS_DIR", "./results"),
            log_dir=params.get("LOG_DIR", "./logs"),
            gpu_id=cfg.get("GPU_ID", 0),
        )
        res = runner.run()
        trained = res.get('trained', {})

        result = {
            "train_size": train_size,
            "test_util": res.get('test_util', 0),
            "best_val_utility": trained.get("best_val_utility", 0),
            "final_train_loss": trained.get("final_train_loss", 0),
        }
        all_results.append(result)
        print(f"train_size={train_size} -> "
              f"test_util={result['test_util']:.4f}, "
              f"best_val={result['best_val_utility']:.4f}")

    print(f"\n{'=' * 60}")
    print("收敛性分析实验完成")
    print(f"{'=' * 60}")
    print(f"\n  {'Train Size':>10}  {'Test Util':>12}  {'Best Val':>12}  {'Final Loss':>12}")
    print(f"  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 12}")
    for r in sorted(all_results, key=lambda x: x["train_size"]):
        print(f"  {r['train_size']:>10}  {r['test_util']:>12.4f}  "
              f"{r['best_val_utility']:>12.4f}  {r['final_train_loss']:>12.4f}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="收敛性分析实验")
    parser.add_argument('--config', '-c', type=str, default='configs/exp_example.yaml')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--train_sizes', type=int, nargs='+', default=[20, 50, 100, 200])
    args = parser.parse_args()

    import yaml
    def load_config(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    cfg = load_config(args.config)

    print("=" * 60)
    print("Convergence Analysis Experiment")
    print("=" * 60)
    print(f"  Config  : {args.config}")
    print(f"  Network : {cfg.get('NUM_UAV', 40)} UAVs")
    print(f"  GPU ID  : {args.gpu_id}")

    run_convergence_analysis(cfg, args.train_sizes)


if __name__ == "__main__":
    main()
