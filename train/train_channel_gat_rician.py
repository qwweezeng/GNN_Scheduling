#!/usr/bin/env python3
"""Retrain the conference channel-only GAT on the official Rician baseline."""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import calculate_achievable_rate
from train.trainer import Trainer
from utils.io import generate_dataset
from utils.artifacts import (
    atomic_pickle_dump,
    expected_dataset_metadata,
    expected_dataset_metadata_quantized,
    load_validated_dataset,
    paper_dataset_paths,
    paper_dataset_paths_quantized,
    write_dataset_manifest,
)
from utils.project_setup import (
    build_gat_scheduler,
    build_network_generator,
    channel_checkpoint_path,
    load_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--train_size", type=int, default=300)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rician_k_db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--val_seed", type=int, default=10087)
    parser.add_argument("--test_seed", type=int, default=10088)
    parser.add_argument(
        "--model-seed", type=int, default=None,
        help=(
            "Optional optimizer/model-initialization seed. When omitted, "
            "--seed is also used for training randomness."
        ),
    )
    parser.add_argument(
        "--prefix_pool_size", type=int, default=0,
        help=(
            "If positive, load this canonical training-pool size and train on "
            "its first --train_size samples. This is used by the paper data-"
            "requirement audit so every subset is a true prefix of one pool."
        ),
    )
    parser.add_argument(
        "--checkpoint_out", default="",
        help="Optional best-validation checkpoint path; defaults to the production model path.",
    )
    parser.add_argument(
        "--final-checkpoint_out", default="",
        help="Optional checkpoint path for the final training epoch before best-model restoration.",
    )
    parser.add_argument(
        "--dataset-root", default="datasets/paper",
        help="Dataset root; the production cache is datasets/paper.",
    )
    parser.add_argument(
        "--report_out", default="",
        help="Optional JSON report path; defaults to the production report path.",
    )
    parser.add_argument(
        "--log-dir", default="logs/channel_gat_rician",
        help="Training log directory; catalog pilots should use an isolated path.",
    )
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
        help="Training device; auto selects CUDA when available.",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=0,
        help="CUDA device index used when --device is auto or cuda.",
    )
    parser.add_argument("--use-ste", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--stochastic-ste", action=argparse.BooleanOptionalAction, default=False,
        help="Sample Bernoulli hard actions during STE training.",
    )
    parser.add_argument(
        "--endpoint-penalty-weight", type=float, default=0.0,
        help="Weight for pairwise shared-endpoint activation penalty.",
    )
    parser.add_argument(
        "--csi-mode", default="perfect",
        choices=("perfect", "large_scale", "quantized"),
        help="CSI quality used for both dataset generation and training.",
    )
    parser.add_argument(
        "--q-bits", type=int, default=None,
        help=(
            "Number of quantization bits; required when --csi-mode is 'quantized'."
        ),
    )
    parser.add_argument(
        "--catalog-policy",
        default="condition_specific",
        choices=("condition_specific", "instantaneous", "mean_power"),
        help=(
            "Candidate-link rule. The default preserves existing paper data; "
            "explicit alternatives are intended for isolated catalog pilots."
        ),
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Generate or validate the requested dataset, then stop before training.",
    )
    return parser.parse_args()


def _load_or_generate(config, args, csi_mode, q_bits):
    pool_size = int(args.prefix_pool_size or args.train_size)
    if int(args.train_size) > pool_size:
        raise ValueError("--train_size cannot exceed --prefix_pool_size")
    seeds = {"train": args.seed, "val": args.val_seed, "test": args.test_seed}
    sizes = {
        "train": pool_size,
        "val": args.val_size,
        "test": args.test_size,
    }
    explicit_catalog = (
        None if args.catalog_policy == "condition_specific"
        else args.catalog_policy
    )
    metadata = expected_dataset_metadata_quantized(
        config, args.num_uav, seeds, sizes, csi_mode, q_bits,
        catalog_policy=explicit_catalog,
    )
    train_val_path, test_path = paper_dataset_paths_quantized(
        Path(args.dataset_root), args.num_uav, seeds, sizes, csi_mode, q_bits,
        catalog_policy=explicit_catalog,
    )
    if train_val_path.exists() and test_path.exists() and not args.force_regenerate:
        train_val = load_validated_dataset(
            train_val_path, metadata, ("train", "val")
        )
        test = load_validated_dataset(test_path, metadata, ("test",))
        return (
            train_val["train_set"][: int(args.train_size)],
            train_val["val_set"], test["test_set"],
            train_val_path,
        )
    generator = build_network_generator(config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set = generate_dataset(
        pool_size, generator, include_queue_feature=False,
        csi_mode=csi_mode, q_bits=q_bits,
        catalog_policy=args.catalog_policy,
    )
    random.seed(args.val_seed)
    np.random.seed(args.val_seed)
    torch.manual_seed(args.val_seed)
    val_set = generate_dataset(
        args.val_size, generator, include_queue_feature=False,
        csi_mode=csi_mode, q_bits=q_bits,
        catalog_policy=args.catalog_policy,
    )
    random.seed(args.test_seed)
    np.random.seed(args.test_seed)
    torch.manual_seed(args.test_seed)
    test_set = generate_dataset(
        args.test_size, generator, include_queue_feature=False,
        csi_mode=csi_mode, q_bits=q_bits,
        catalog_policy=args.catalog_policy,
    )
    atomic_pickle_dump(
        {"metadata": metadata, "train_set": train_set, "val_set": val_set},
        train_val_path,
    )
    atomic_pickle_dump({"metadata": metadata, "test_set": test_set}, test_path)
    write_dataset_manifest(
        train_val_path.parent, metadata, train_val_path, test_path
    )
    return train_set[: int(args.train_size)], val_set, test_set, train_val_path


def _output_audit(model, dataset):
    probabilities, binary_counts, link_counts = [], [], []
    hard_utilities, all_active_utilities = [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for state in dataset:
            pi, _ = model(
                state.vertex_features[:, :1].to(device),
                state.edge_index.to(device), state.edge_weights.to(device)
            )
            value = pi.reshape(-1).cpu()
            hard = (value >= 0.5).float().unsqueeze(1)
            probabilities.append(value)
            binary_counts.append(float(hard.sum()))
            link_counts.append(float(value.numel()))
            hard_utilities.append(float(
                calculate_achievable_rate(
                    hard, state, binary=False
                ).sum().item()
            ))
            all_active_utilities.append(float(
                calculate_achievable_rate(
                    torch.ones_like(hard), state, binary=False
                ).sum().item()
            ))
    values = torch.cat(probabilities)
    all_active_fraction = float(np.mean([
        active == total for active, total in zip(binary_counts, link_counts)
    ]))
    all_inactive_fraction = float(np.mean([
        active == 0 for active in binary_counts
    ]))
    hard_utility = float(np.mean(hard_utilities))
    all_active_utility = float(np.mean(all_active_utilities))
    integrity_passed = bool(
        all_active_fraction < 0.95
        and all_inactive_fraction < 0.95
        and hard_utility > 1.05 * all_active_utility
    )
    return {
        "probability_mean": float(values.mean()),
        "probability_std": float(values.std(unbiased=False)),
        "probability_min": float(values.min()),
        "probability_max": float(values.max()),
        "mean_binary_active_fraction": float(
            np.mean(np.asarray(binary_counts) / np.asarray(link_counts))
        ),
        "mean_binary_active_links": float(np.mean(binary_counts)),
        "all_links_active_state_fraction": all_active_fraction,
        "all_links_inactive_state_fraction": all_inactive_fraction,
        "hard_test_utility": hard_utility,
        "all_active_test_utility": all_active_utility,
        "hard_gain_over_all_active": hard_utility - all_active_utility,
        "integrity_rule": (
            "all-active states < 95%, all-inactive states < 95%, and hard "
            "utility > 1.05 * all-active utility"
        ),
        "integrity_passed": integrity_passed,
    }


def main():
    args = parse_args()
    csi_mode = str(args.csi_mode)
    q_bits = int(args.q_bits) if args.q_bits is not None else None
    if csi_mode == "quantized" and q_bits is None:
        raise ValueError("--q-bits is required when --csi-mode is 'quantized'")
    if csi_mode != "quantized" and q_bits is not None:
        raise ValueError("--q-bits may only be set when --csi-mode is 'quantized'")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": args.rician_k_db,
        "CSI_MODE": csi_mode,
        "Q_BITS": q_bits,
    })
    train_set, val_set, test_set, dataset_path = _load_or_generate(config, args, csi_mode, q_bits)
    pool_size = int(args.prefix_pool_size or args.train_size)
    dataset_metadata = expected_dataset_metadata_quantized(
        config,
        args.num_uav,
        {"train": args.seed, "val": args.val_seed, "test": args.test_seed},
        {
            "train": pool_size,
            "val": args.val_size,
            "test": args.test_size,
        },
        csi_mode,
        q_bits,
        catalog_policy=(
            None if args.catalog_policy == "condition_specific"
            else args.catalog_policy
        ),
    )
    if args.prepare_only:
        summary = {
            "catalog_policy": args.catalog_policy,
            "dataset_path": str(dataset_path),
            "dataset_fingerprint": dataset_metadata["fingerprint"],
            "split_sizes": {
                "train": len(train_set),
                "val": len(val_set),
                "test": len(test_set),
            },
            "mean_links": {
                "train": float(np.mean([state.links.size(0) for state in train_set])),
                "val": float(np.mean([state.links.size(0) for state in val_set])),
                "test": float(np.mean([state.links.size(0) for state in test_set])),
            },
        }
        print(json.dumps(summary, indent=2), flush=True)
        return
    model_seed = args.seed if args.model_seed is None else args.model_seed
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    model = build_gat_scheduler(config, raw_node_dim=1)
    model.stochastic_ste = bool(args.stochastic_ste)
    if args.use_ste and args.stochastic_ste:
        final_layer = model.final_scheduler[-1]
        torch.nn.init.constant_(final_layer.bias, 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.device in {"auto", "cuda"} and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[train] device={device}", flush=True)
    trainer = Trainer(
        model, optimizer, device=device, gpu_id=args.gpu_id, use_queue=False,
        log_dir=args.log_dir,
    )
    trained = trainer.train(
        num_epochs=args.epochs,
        train_dataset=train_set,
        val_dataset=val_set,
        eval_interval=1,
        use_queue=False,
        use_ste=args.use_ste,
        endpoint_penalty_weight=args.endpoint_penalty_weight,
    )
    test_utility = float(trainer.evaluate(model, test_set, use_queue=False))
    audit = _output_audit(model, test_set)
    checkpoint_path = Path(args.checkpoint_out) if args.checkpoint_out else (
        channel_checkpoint_path(config, args.num_uav, args.train_size)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = (
        Path(args.final_checkpoint_out) if args.final_checkpoint_out else None
    )
    if final_checkpoint_path is not None:
        final_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_version": "channel_gat_v1",
        "model_state_dict": model.state_dict(),
        "tx_power_dbm": float(config.get("TX_POWER", 10.0)),
        "csi_mode": csi_mode,
        "quantization_bits": q_bits,
        "catalog_policy": args.catalog_policy,
        "channel_model": "rician",
        "rician_k_db": float(args.rician_k_db),
        "num_uav": int(args.num_uav),
        "dataset_path": str(dataset_path),
        "dataset_fingerprint": dataset_metadata["fingerprint"],
        "config": {
            "NUM_UAV": int(args.num_uav),
            "TX_POWER": float(config["TX_POWER"]),
            "CHANNEL_MODEL": "rician",
            "RICIAN_K_DB": float(args.rician_k_db),
            "CSI_MODE": csi_mode,
            "Q_BITS": q_bits,
            "CATALOG_POLICY": args.catalog_policy,
            "TRAIN_SIZE": int(args.train_size),
            "TRAIN_POOL_SIZE": pool_size,
            "VAL_SIZE": int(args.val_size),
            "TEST_SIZE": int(args.test_size),
            "SEED": int(args.seed),
            "MODEL_SEED": int(model_seed),
        },
        "checkpoint_selection": "best_validation",
        "best_checkpoint": str(checkpoint_path),
        "final_checkpoint": (
            str(final_checkpoint_path) if final_checkpoint_path is not None else None
        ),
        "best_epoch": trained["best_epoch"],
        "best_val_utility": trained["best_val_util"],
        "test_utility": test_utility,
        "output_audit": audit,
        "per_epoch_logs": trained["per_epoch_logs"],
        "training_subset": {
            "rule": "first_n_prefix_of_canonical_training_pool",
            "selected_size": int(args.train_size),
            "pool_size": pool_size,
            "pool_dataset_fingerprint": dataset_metadata["fingerprint"],
        },
        "integrity_status": "passed" if audit["integrity_passed"] else "rejected",
        "timestamp": datetime.now().isoformat(),
    }
    report_path = Path(args.report_out) if args.report_out else (
        Path("results/paper/channel_gat_training") /
        f"rician_channel_gat_N{args.num_uav}_K{args.rician_k_db:g}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        key: value for key, value in checkpoint.items()
        if key not in {"model_state_dict"}
    }, indent=2), encoding="utf-8")
    if not audit["integrity_passed"]:
        raise RuntimeError(
            "Rejected collapsed channel-only GAT; the production checkpoint "
            f"was not overwritten. See {report_path}: {audit}"
        )
    torch.save(checkpoint, checkpoint_path)
    if final_checkpoint_path is not None:
        final_checkpoint = {
            **checkpoint,
            "checkpoint_selection": "final_epoch",
            "selected_epoch": int(args.epochs) - 1,
            "model_state_dict": trained["final_model_state_dict"],
            "decision_threshold": trained["final_decision_threshold"],
        }
        torch.save(final_checkpoint, final_checkpoint_path)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path) if final_checkpoint_path else None,
        "report": str(report_path),
        "test_utility": test_utility,
        "output_audit": audit,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
