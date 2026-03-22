#!/usr/bin/env python
"""
Compare SNR vs Quantized Distance features for GAT scheduling.
Run two experiments: one with SNR, one with quantized distance.
"""

import os
import argparse
import yaml
import json
import torch
import numpy as np
import sys

# Add project root to Python path (so we can import modules from project root)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

def run_experiment(config_path, use_quantized_distance, results_dir="./results", logs_dir="./logs"):
    """Run a single experiment with given configuration."""
    print(f"\n{'='*50}")
    print(f"Running experiment: {'Quantized Distance' if use_quantized_distance else 'SNR'}")
    print(f"{'='*50}")

    # Load base configuration
    params = load_config(config_path).copy()

    # Override quantization setting
    params["USE_QUANTIZED_DISTANCE"] = use_quantized_distance

    print(f"[compare] Loaded config from {config_path}")
    print(f"[compare] USE_QUANTIZED_DISTANCE: {use_quantized_distance}")

    # Run experiment using ExperimentRunner
    runner = ExperimentRunner(params, results_dir=results_dir, log_dir=logs_dir)
    res = runner.run()

    # Extract results
    trained = res.get('trained', {})
    test_util = res.get('test_util', None)

    return {
        'train_losses': [log.get('train_loss') for log in trained.get('per_epoch_logs', []) if log.get('train_loss') is not None],
        'val_utils': [log.get('val_utility') for log in trained.get('per_epoch_logs', []) if log.get('val_utility') is not None],
        'test_util': test_util,
        'best_val_util': trained.get('best_val_util', None),
        'baselines': None  # baselines are saved separately by runner
    }

def main():
    parser = argparse.ArgumentParser(description="Compare SNR vs Quantized Distance features")
    parser.add_argument("--config", "-c", required=True, help="Path to base YAML/JSON experiment config")
    parser.add_argument("--results", default="./results", help="Results directory")
    parser.add_argument("--logs", default="./logs", help="Logs directory")
    args = parser.parse_args()

    # Experiment 1: SNR features
    print("\nExperiment 1: SNR Features")
    results_snr = run_experiment(
        config_path=args.config,
        use_quantized_distance=False,
        results_dir=args.results,
        logs_dir=args.logs
    )

    # Experiment 2: Quantized Distance features
    print("\nExperiment 2: Quantized Distance Features")
    results_quantized = run_experiment(
        config_path=args.config,
        use_quantized_distance=True,
        results_dir=args.results,
        logs_dir=args.logs
    )
    
    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"SNR Features:")
    print(f"  Test Utility: {results_snr['test_util']:.4f}")
    print(f"  Best Val Utility: {results_snr['best_val_util']:.4f}")
    print(f"  Final Train Loss: {results_snr['train_losses'][-1]:.4f}")
    print(f"Quantized Distance Features:")
    print(f"  Test Utility: {results_quantized['test_util']:.4f}")
    print(f"  Best Val Utility: {results_quantized['best_val_util']:.4f}")
    print(f"  Final Train Loss: {results_quantized['train_losses'][-1]:.4f}")

    # Convergence comparison
    snr_convergence_epoch = np.argmin(results_snr['train_losses']) + 1
    quantized_convergence_epoch = np.argmin(results_quantized['train_losses']) + 1

    print(f"\nConvergence Analysis:")
    print(f"SNR: Best train loss at epoch {snr_convergence_epoch}")
    print(f"Quantized: Best train loss at epoch {quantized_convergence_epoch}")

    if quantized_convergence_epoch < snr_convergence_epoch:
        print("✓ Quantized distance converges faster!")
    elif quantized_convergence_epoch > snr_convergence_epoch:
        print("✗ Quantized distance converges slower")
    else:
        print("~ Similar convergence speed")

    # Performance loss
    performance_loss = results_snr['test_util'] - results_quantized['test_util']
    if performance_loss > 0:
        print(f"✗ Performance loss: {performance_loss:.2f} (quantized worse)")
    else:
        print(f"✓ Performance gain: {-performance_loss:.2f} (quantized better)")

    print("\nNote: Baselines are automatically saved by ExperimentRunner to results directory.")
    print("Check the *_baselines.json files for detailed baseline comparisons.")

if __name__ == "__main__":
    main()
