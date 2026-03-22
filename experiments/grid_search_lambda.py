#!/usr/bin/env python3
"""
Coarse grid search over LAMBDA_SPARSITY and LAMBDA_CONFLICT for numUAV=50.
Creates one experiment per (lambda_sparsity, lambda_conflict) combination using
the existing ExperimentRunner. Saves a summary Excel file at experiments/grid_lambda_results.xlsx (Sheet1).

Usage:
  python experiments/grid_search_lambda.py --config configs/exp_example.yaml

Notes:
 - Default train_size=1024, epochs=20 (overrides config).
 - Does not parallelize; run sequentially. Each run creates its own results/<experiment_id>/ folder.
"""
import os
import csv
import argparse
import copy
import yaml
import pandas as pd

from experiments.runner import ExperimentRunner

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True, help="base YAML config")
    parser.add_argument("--out", default="experiments/grid_lambda_results.xlsx", help="summary Excel file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    # fixed settings for grid
    base_cfg["NUM_UAV"] = 50
    base_cfg["TRAIN_SIZE"] = 1024
    base_cfg["EPOCHS"] = 20
    base_cfg["SEED"] = args.seed

    # coarse grid (adjustable)
    sparsities = [0.001, 0.005]
    conflicts = [0.001, 0.003]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 收集所有结果
    all_results = []

    for ls in sparsities:
        for lc in conflicts:
            cfg = copy.deepcopy(base_cfg)
            cfg["LAMBDA_SPARSITY"] = ls
            cfg["LAMBDA_CONFLICT"] = lc
            cfg["EXPERIMENT_GROUP"] = "grid_lambda"
            cfg["EXCEL_SHEET"] = "Sheet1"  # 网格搜索实验放在Sheet1中
            cfg["NOTES"] = f"grid_search_ls={ls}_lc={lc}"

            print(f"[grid] Running ls={ls}, lc={lc}")
            runner = ExperimentRunner(cfg, results_dir=cfg.get("RESULTS_DIR","./results"), log_dir=cfg.get("LOG_DIR","./logs"), gpu_id=args.gpu_id)
            out = runner.run()
            test_util = out.get("test_util", None)
            trained = out.get("trained", out.get("trained", {}))
            best_val = None
            exp_id = None
            exp_dir = None
            if isinstance(out, dict):
                exp_id = out.get("experiment_id", None)
            # try to read returned trained dict for best_val
            if isinstance(trained, dict):
                best_val = trained.get("best_val_util", None)
                exp_dir = trained.get("experiment_dir", None)

            # fallback: runner saved info in results; try to extract experiment_id from returned info
            if exp_id is None and isinstance(out, dict):
                exp_id = out.get("experiment_id", None)
            row = {
                "lambda_sparsity": ls,
                "lambda_conflict": lc,
                "test_util": test_util,
                "best_val_util": best_val,
                "experiment_id": exp_id,
                "experiment_dir": exp_dir
            }
            all_results.append(row)
            print(f"[grid] Done ls={ls}, lc={lc} -> test_util={test_util}, best_val={best_val}")

    # 保存到Excel文件
    df = pd.DataFrame(all_results)
    df.to_excel(args.out, sheet_name="Sheet1", index=False)

    print(f"[grid] Grid search complete, summary saved to {args.out} (Sheet1)")

if __name__ == "__main__":
    main()


