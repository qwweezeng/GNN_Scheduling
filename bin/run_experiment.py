#!/usr/bin/env python3
import argparse
import json
import os
import sys

# ========== 完整的 TensorFlow/CUDA 日志抑制 ==========
# 1. 抑制 TF_CPP 日志 (GPU 设备信息等)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL

# 2. 禁用 oneDNN 优化器的日志
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. 禁用 XLA 编译日志
os.environ['TF_XLA_FLAGS'] = '--tf_xla_min_cluster_size=1'

# 4. 禁用 CUDA 编译日志
os.environ['CUDA_CACHE_DISABLE'] = '0'

# 5. 设置 absl 日志级别
os.environ['ABSL_LOG_LEVEL'] = 'ERROR'

# 设置 Python 日志级别
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow.python').setLevel(logging.ERROR)

import yaml

# Add project root to Python path (so we can import modules from project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from experiments.runner import ExperimentRunner

def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f)
        else:
            return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Run an experiment (wrapper for ExperimentRunner).")
    parser.add_argument("--config", "-c", required=True, help="Path to JSON/YAML experiment config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--results", default="./results", help="Results directory")
    parser.add_argument("--logs", default="./logs", help="Logs / models directory")
    parser.add_argument("--dry-run", action="store_true", help="Load config and do a dry-run (no long training)")
    parser.add_argument("--tx-power", type=float, help="Override TX_POWER (dBm)")
    parser.add_argument("--csi-mode", choices=["perfect", "large_scale", "quantized"], help="Override CSI_MODE")
    parser.add_argument("--q-bits", type=int, help="Override Q_BITS")
    parser.add_argument("--train-size", type=int, help="Override TRAIN_SIZE")
    parser.add_argument("--lambda-sparsity", type=float, help="Override LAMBDA_SPARSITY")
    parser.add_argument("--lambda-conflict", type=float, help="Override LAMBDA_CONFLICT")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID to use (overrides GPU_ID in config, default: from config)")
    parser.add_argument("--group", help="Experiment group (overrides EXPERIMENT_GROUP)")
    parser.add_argument("--notes", help="Experiment notes (overrides NOTES)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["SEED"] = args.seed
    # apply CLI overrides if provided
    if args.tx_power is not None:
        cfg["TX_POWER"] = args.tx_power
    if args.csi_mode is not None:
        cfg["CSI_MODE"] = args.csi_mode
    if args.q_bits is not None:
        cfg["Q_BITS"] = args.q_bits
    if args.train_size is not None:
        cfg["TRAIN_SIZE"] = args.train_size
    if args.lambda_sparsity is not None:
        cfg["LAMBDA_SPARSITY"] = args.lambda_sparsity
    if args.lambda_conflict is not None:
        cfg["LAMBDA_CONFLICT"] = args.lambda_conflict
    if args.group is not None:
        cfg["EXPERIMENT_GROUP"] = args.group
    if args.notes is not None:
        cfg["NOTES"] = args.notes

    print(f"[run_experiment] Loaded config: {args.config}")
    runner = ExperimentRunner(cfg, results_dir=args.results, log_dir=args.logs, gpu_id=args.gpu_id)

    if args.dry_run:
        print("[run_experiment] Dry run: generating small dataset and exiting")
        cfg_preview = cfg.copy()
        cfg_preview["TRAIN_SIZE"] = cfg_preview.get("TRAIN_SIZE", 10)
        cfg_preview["VAL_SIZE"] = cfg_preview.get("VAL_SIZE", 2)
        cfg_preview["TEST_SIZE"] = cfg_preview.get("TEST_SIZE", 2)
        # mark dry run to avoid saving outputs
        cfg_preview["DRY_RUN"] = True
        runner.params = cfg_preview
        _ = runner.run()
        print("[run_experiment] Dry run complete")
        sys.exit(0)

    res = runner.run()
    print("[run_experiment] Finished. Test utility:", res.get("test_util"))

if __name__ == "__main__":
    main()


