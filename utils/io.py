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
    params: dict containing keys used below (NUM_UAV, USE_QUANTIZED_DISTANCE, Q_BITS, LR, LAMBDA_SPARSITY, LAMBDA_CONFLICT, BATCH_SIZE, EPOCHS, GRAD_CLIP_NORMAL, NOTES, seed)
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
    feature_name = f"Quantized{params.get('Q_BITS')}" if params.get("USE_QUANTIZED_DISTANCE") else "SNR"
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
        "lr", "lambda_sparsity", "lambda_conflict", "batch_size", "epochs",
        "final_train_loss", "best_val_utility", "test_utility",
        "final_sparsity_penalty", "final_conflict_penalty",
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
        "lambda_sparsity": params.get("LAMBDA_SPARSITY", None),
        "lambda_conflict": params.get("LAMBDA_CONFLICT", None),
        "batch_size": params.get("BATCH_SIZE"),
        "epochs": params.get("EPOCHS"),
        "final_train_loss": final_train_loss,
        "best_val_utility": best_val_util,
        "test_utility": test_util,
        "final_sparsity_penalty": None,
        "final_conflict_penalty": None,
        "convergence_epoch": convergence_epoch
    }

    # determine final per-epoch penalties (last epoch values) if available
    try:
        if per_epoch_logs:
            last = per_epoch_logs[-1]
            row["final_sparsity_penalty"] = last.get("sparsity_penalty", None)
            row["final_conflict_penalty"] = last.get("conflict_penalty", None)
    except Exception:
        pass

    # save model to results/models/
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


