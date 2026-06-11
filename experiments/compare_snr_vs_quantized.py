#!/usr/bin/env python
"""
Compare SNR vs Quantized Distance features for GAT scheduling.

Usage:
    uv run python experiments/compare_snr_vs_quantized.py --config configs/exp_example.yaml
"""

import os
import argparse
import yaml
import numpy as np
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.runner import ExperimentRunner


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_experiment(cfg, csi_mode):
    params = cfg.copy()
    params["CSI_MODE"] = csi_mode

    # 自动推导 NODE_RAW_DIM（与 runner.py 保持一致）
    # 基础 YAML 中 NODE_RAW_DIM: 1（纯吞吐量实验）
    # 当 USE_QUEUE=True 时自动切换为 2
    use_queue = params.get("USE_QUEUE", False)
    if params.get("NODE_RAW_DIM") is not None:
        node_raw_dim = params.get("NODE_RAW_DIM")
    else:
        node_raw_dim = 2 if use_queue else 1
    params["NODE_RAW_DIM"] = node_raw_dim

    print(f"\n{'=' * 50}")
    print(f"Running: CSI_MODE={csi_mode}")
    print(f"{'=' * 50}")

    runner = ExperimentRunner(
        params,
        results_dir=params.get("RESULTS_DIR", "./results"),
        log_dir=params.get("LOG_DIR", "./logs"),
    )
    return runner.run()


def main():
    parser = argparse.ArgumentParser(
        description="Compare SNR vs Quantized Distance features"
    )
    parser.add_argument('--config', '-c', type=str, default='configs/exp_example.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 60)
    print("Compare: SNR vs Quantized Distance Features")
    print("=" * 60)
    print(f"  Config  : {args.config}")
    print(f"  Network : {cfg.get('NUM_UAV', 40)} UAVs")

    results_snr = run_experiment(cfg, csi_mode="perfect")
    results_quantized = run_experiment(cfg, csi_mode="quantized")

    snr_test = results_snr.get('test_util', 0)
    snr_best = results_snr.get('trained', {}).get('best_val_util', 0)
    q_test = results_quantized.get('test_util', 0)
    q_best = results_quantized.get('trained', {}).get('best_val_util', 0)

    snr_losses = [log.get('train_loss') for log in
                  results_snr.get('trained', {}).get('per_epoch_logs', [])
                  if log.get('train_loss') is not None]
    q_losses = [log.get('train_loss') for log in
                results_quantized.get('trained', {}).get('per_epoch_logs', [])
                if log.get('train_loss') is not None]

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"SNR Features:")
    print(f"  Test Utility     : {snr_test:.4f}")
    print(f"  Best Val Utility : {snr_best:.4f}")
    print(f"  Final Train Loss : {snr_losses[-1] if snr_losses else 'N/A':.4f}")
    print(f"Quantized Distance Features:")
    print(f"  Test Utility     : {q_test:.4f}")
    print(f"  Best Val Utility : {q_best:.4f}")
    print(f"  Final Train Loss : {q_losses[-1] if q_losses else 'N/A':.4f}")

    snr_best_epoch = np.argmin(snr_losses) + 1 if snr_losses else 'N/A'
    q_best_epoch = np.argmin(q_losses) + 1 if q_losses else 'N/A'

    print(f"\nConvergence:")
    print(f"  SNR       : best loss at epoch {snr_best_epoch}")
    print(f"  Quantized : best loss at epoch {q_best_epoch}")

    perf_loss = snr_test - q_test
    print(f"\n  Performance gap: {perf_loss:.4f}")


if __name__ == "__main__":
    main()
