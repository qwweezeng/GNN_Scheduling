import os
import json
import csv
from time import strftime
import torch
import pandas as pd
from openpyxl import load_workbook

def save_experiment_results(trained_model, train_set, test_util, params,
                            results_dir="./results", log_dir="./logs", excel_sheet="Sheet1"):
    """
    Concise saver: write per-epoch JSON, append Excel summary (all_experiments_summary.xlsx), save model state.
    No longer saves CSV files - now uses Excel as the default format.
    Raises RuntimeError on any I/O failure with aggregated messages.
    params: dict containing keys used below (NUM_UAV, CSI_MODE, Q_BITS, LR, BATCH_SIZE, EPOCHS, NOTES, seed)
    excel_sheet: Excel sheet name for saving results (default: "Sheet1")
    """
    os.makedirs(results_dir, exist_ok=True)
    errors = []

    # extract
    if isinstance(trained_model, dict):
        per_epoch_logs = trained_model.get("per_epoch_logs", [])
        best_val_util = trained_model.get("best_val_util", None)
        final_train_loss = trained_model.get("final_train_loss", None)
        model_obj = trained_model.get("model", None)
    else:
        per_epoch_logs = []
        best_val_util = None
        final_train_loss = None
        model_obj = trained_model if hasattr(trained_model, "state_dict") else None

    exp_time = strftime("%Y%m%d_%H%M")
    feature_name = f"Quantized{params.get('Q_BITS')}" if params.get("CSI_MODE") == "quantized" else (
        "LargeScale" if params.get("CSI_MODE") == "large_scale" else "SNR")
    experiment_id = f"exp_{params.get('NUM_UAV')}_{exp_time}"
    # create per-experiment directory for organized storage
    experiment_dir = os.path.join(results_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)

    # If dry run, skip saving files
    if params.get("DRY_RUN", False):
        print("[io] Dry run - skipping saving experiment results.")
        return {"excel_path": None, "json_path": None, "experiment_id": experiment_id}

    # json - write per-epoch logs into the per-experiment folder
    json_path = os.path.join(experiment_dir, f"{experiment_id}_per_epoch.json")
    try:
        with open(json_path, "w") as jf:
            json.dump(per_epoch_logs, jf, indent=2)
    except Exception as e:
        errors.append(f"JSON write failed: {e}")

    # consolidated Excel at results root (one line per experiment)
    excel_path = os.path.join(results_dir, "all_experiments_summary.xlsx")
    # Excel header ordered: description/group/tags -> parameters -> results
    header = [
        "experiment_id", "group", "notes", "feature",
        "num_uav", "train_size", "seed",
        "lr",
        "use_power_control", "lambda_power_penalty", "lambda_alpha_smooth",
        "batch_size", "epochs",
        "final_train_loss", "best_val_utility", "test_utility",
        "convergence_epoch"
    ]

    # infer convergence epoch
    convergence_epoch = None
    if per_epoch_logs and best_val_util is not None:
        # use a small tolerance for floating point comparison
        tolerance = 1e-3  # allow difference of 0.001
        for e in per_epoch_logs:
            val_util = e.get("val_utility")
            if val_util is not None and abs(val_util - best_val_util) < tolerance:
                convergence_epoch = e.get("epoch")
                break
    if convergence_epoch is None and per_epoch_logs:
        try:
            convergence_epoch = min(per_epoch_logs, key=lambda x: x.get("train_loss", float("inf"))).get("epoch")
        except Exception:
            convergence_epoch = None


    row = {
        "experiment_id": experiment_id,
        "group": params.get("EXPERIMENT_GROUP", ""),
        "notes": params.get("NOTES", ""),
        "feature": feature_name,
        "num_uav": params.get("NUM_UAV"),
        "train_size": len(train_set),
        "seed": params.get("SEED", ""),
        "lr": params.get("LR"),
        "use_power_control": params.get("USE_POWER_CONTROL", False),
        "lambda_power_penalty": params.get("LAMBDA_POWER_PENALTY", None),
        "lambda_alpha_smooth": params.get("LAMBDA_ALPHA_SMOOTH", None),
        "batch_size": params.get("BATCH_SIZE"),
        "epochs": params.get("EPOCHS"),
        "final_train_loss": final_train_loss,
        "best_val_utility": best_val_util,
        "test_utility": test_util,
        "convergence_epoch": convergence_epoch
    }

    # save model to experiment directory (results/exp_XXX/)
    save_model_flag = params.get("SAVE_MODEL", True)
    if save_model_flag:
        try:
            if model_obj is not None:
                try:
                    model_path = os.path.join(experiment_dir, f"{experiment_id}_model.pth")
                    torch.save(model_obj.state_dict(), model_path)
                    print(f"[io] Model saved to {model_path}")
                except Exception as e:
                    errors.append(f"Model save failed: {e}")
            else:
                errors.append("No model object found to save.")
        except Exception as e:
            errors.append(f"Model save failed: {e}")

    # Append summary row to consolidated Excel at results root
    excel_path = os.path.join(results_dir, "all_experiments_summary.xlsx")
    try:
        # Convert row to DataFrame for easier Excel handling
        df_row = pd.DataFrame([row])

        # Check if Excel file exists
        if os.path.exists(excel_path):
            # Load existing workbook and append to specified sheet
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                # Check if sheet exists
                try:
                    existing_df = pd.read_excel(excel_path, sheet_name=excel_sheet)
                    # Append new row
                    combined_df = pd.concat([existing_df, df_row], ignore_index=True)
                    combined_df.to_excel(writer, sheet_name=excel_sheet, index=False)
                except ValueError:
                    # Sheet doesn't exist, create it
                    df_row.to_excel(writer, sheet_name=excel_sheet, index=False)
        else:
            # Create new Excel file with the specified sheet
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df_row.to_excel(writer, sheet_name=excel_sheet, index=False)

    except Exception as e:
        errors.append(f"Metrics Excel append failed: {e}")

    if errors:
        raise RuntimeError(" | ".join(errors))

    return {"excel_path": excel_path, "json_path": json_path, "experiment_id": experiment_id, "experiment_dir": experiment_dir}


def generate_dataset(num_samples, simulator, batch_size=100, csi_mode="perfect", q_bits=8,
                    include_queue_feature=True,
                    queue_mode="random", queue_range=(0, 25), poisson_lambda=None):
    """
    Generate dataset in batches to avoid memory overflow.
    Used by ExperimentRunner.

    Args:
        num_samples: total number of samples
        simulator: network simulator instance
        batch_size: batch size for generation
        csi_mode: "perfect" | "large_scale" | "quantized"
            Controls model input and rate computation uniformly.
        q_bits: number of quantization bits (only used when csi_mode == "quantized")
        include_queue_feature: 是否在节点特征中包含队列长度
            - True:  [SNR_dB, queue_normalized] (2维)，用于队列感知训练/仿真
            - False: [SNR_dB]                   (1维)，用于纯吞吐量最大化实验
            应与 USE_QUEUE 参数保持一致。
        queue_mode: 队列状态生成模式
            - "random":  uniform[queue_range[0], queue_range[1])  ← 训练推荐
            - "poisson": Poisson(poisson_lambda)
        queue_range: 仅 queue_mode="random" 时生效，(min, max)
        poisson_lambda: 仅 queue_mode="poisson" 时生效
    """
    dataset = []
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch = []
        for _ in range(i, batch_end):
            sample = simulator.generate_network(
                csi_mode=csi_mode, q_bits=q_bits,
                include_queue_feature=include_queue_feature,
                queue_mode=queue_mode,
                queue_range=queue_range,
                poisson_lambda=poisson_lambda,
            )
            batch.append(sample)
        dataset.extend(batch)
        if (i + batch_size) % 50 == 0 or batch_end == num_samples:
            print(f"  Generated {batch_end}/{num_samples} samples...")
    return dataset