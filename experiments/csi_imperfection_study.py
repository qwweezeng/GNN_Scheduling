#!/usr/bin/env python3
"""
不完美 CSI 实验组：量化 / 大尺度衰落 对调度性能的影响。

Usage:
    uv run python experiments/csi_imperfection_study.py
    uv run python experiments/csi_imperfection_study.py --num_uav 30
"""

import os
import sys
import argparse
import yaml

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.runner import ExperimentRunner


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_cache_path(num_uav, csi_mode, q_bits=None):
    """
    根据 CSI 模式返回数据集缓存路径。

    目录结构：
        datasets/                              ← perfect CSI（默认）
        datasets/large_scale/                  ← 大尺度衰落CSI
        datasets/quantized/                    ← 量化CSI
    """
    if csi_mode == "perfect":
        return f"./datasets/dataset_numUAV={num_uav}.pkl"
    if csi_mode == "large_scale":
        return f"./datasets/large_scale/dataset_numUAV={num_uav}_large_scale.pkl"
    return f"./datasets/quantized/dataset_numUAV={num_uav}_quantized_{q_bits}bits.pkl"


def run_single(csi_mode, cfg, q_bits, gpu_id):
    params = cfg.copy()
    params["CSI_MODE"] = csi_mode
    params["Q_BITS"] = q_bits if q_bits else 8
    params["EXPERIMENT_GROUP"] = "csi_imperfection"
    params["DATASET_CACHE_PATH"] = make_cache_path(cfg["NUM_UAV"], csi_mode, params["Q_BITS"])

    # 自动推导 NODE_RAW_DIM（与 runner.py 保持一致）
    use_queue = params.get("USE_QUEUE", False)
    if params.get("NODE_RAW_DIM") is not None:
        node_raw_dim = params.get("NODE_RAW_DIM")
    else:
        node_raw_dim = 2 if use_queue else 1
    params["NODE_RAW_DIM"] = node_raw_dim

    label = {
        "perfect":     f"perfect(N={cfg['NUM_UAV']})",
        "large_scale": f"large_scale(N={cfg['NUM_UAV']})",
        "quantized":   f"quantized_{q_bits}bits(N={cfg['NUM_UAV']})",
    }[csi_mode]

    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    runner = ExperimentRunner(
        params,
        results_dir=params.get("RESULTS_DIR", "./results"),
        log_dir=params.get("LOG_DIR", "./logs"),
        gpu_id=gpu_id,
    )
    res = runner.run()
    trained = res.get('trained', {})
    print(f"  -> test_util={res.get('test_util', 0):.4f}, "
          f"best_val={trained.get('best_val_util', 0):.4f}")
    return {
        "csi_mode":       csi_mode,
        "q_bits":         params["Q_BITS"],
        "test_util":      res.get('test_util', 0),
        "best_val_util":  trained.get('best_val_util', 0),
    }


def main():
    parser = argparse.ArgumentParser(description="CSI 不完美性实验组")
    parser.add_argument('--config', '-c', type=str, default='configs/exp_example.yaml')
    parser.add_argument('--num_uav', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_uav is not None:
        cfg["NUM_UAV"] = args.num_uav
    cfg["EXCEL_SHEET"] = "Sheet3"
    cfg["NOTES"] = "CSI不完美性实验组"

    print("=" * 60)
    print("CSI Imperfection Study")
    print("=" * 60)
    print(f"  Config : {args.config}")
    print(f"  Network: {cfg['NUM_UAV']} UAVs")
    print(f"  GPU ID : {args.gpu_id}")

    results = []
    results.append(run_single("perfect",     cfg, None, args.gpu_id))
    results.append(run_single("large_scale", cfg, None, args.gpu_id))

    print("\n" + "=" * 60)
    print(f"\n{'CSI_MODE':<20} {'Q_bits':>6}  {'Test Utility':>14}  {'Best Val':>14}")
    print("-" * 60)
    for r in results:
        print(f"{r['csi_mode']:<20} {str(r.get('q_bits') or '-'):>6}  "
              f"{r['test_util']:>14.4f}  {r['best_val_util']:>14.4f}")


if __name__ == "__main__":
    main()
