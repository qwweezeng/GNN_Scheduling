#!/usr/bin/env python3
"""Cross-N Proposed-GAT evaluation with the formal endpoint-matching decoder.

Each train-N model is first trained (or reused if checkpoint already exists) with
the specified --train_size, then evaluated against all test-N test sets.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model import calculate_achievable_rate
from utils.artifacts import (
    dataset_split_fingerprint,
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.feasibility import (
    decode_endpoint_mutual_matching,
    shared_uav_action_audit,
)
from utils.project_setup import (
    build_gat_scheduler,
    channel_checkpoint_path,
    load_channel_gat_checkpoint,
    load_config,
    proposed_gat_checkpoint_dir,
)


def direct_rate_priority(pi, state):
    tx, rx = state.links[:, 0], state.links[:, 1]
    snr = state.signal_power[tx, rx].cpu() / state.noise_power
    return pi.reshape(-1).float().cpu() * torch.log2(1.0 + snr / state.snr_gap).float()


def cfg(n: int):
    c = load_config(ROOT / "configs/paper.yaml")
    c.update({
        "NUM_UAV": n,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    return c


def test_states(n: int, train_size: int):
    """Return the fixed, provenance-checked 128-sample test set for network size n.

    The test-cache filename is intentionally independent of ``train_size``.
    Its seed and physical scenario must match, but its metadata may record a
    different training-pool size from a prior experiment.
    """
    c = cfg(n)
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": train_size, "val": 128, "test": 128}
    _, p = paper_dataset_paths(ROOT / "datasets/paper", n, seeds, sizes)
    expected = expected_dataset_metadata(c, n, seeds, sizes)
    payload = load_validated_dataset(p, expected, ("test",))
    return (
        payload["test_set"],
        dataset_split_fingerprint(expected, "test"),
        expected["fingerprint"],
        str(p),
    )


def ensure_checkpoint(num_uav: int, train_size: int, device: str) -> Path:
    """Train (or reuse) a channel GAT checkpoint for the given num_uav and train_size."""
    if train_size == 300:
        # Reuse the canonical 50-epoch seed-0 full_300 model shared by the
        # conference experiments and the journal's physical-layer section.
        ckpt_path = (
            proposed_gat_checkpoint_dir(num_uav, 0, "full_300")
            / "best_model.pth"
        )
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Frozen 300-sample checkpoint missing: {ckpt_path}"
            )
        print(f"[ensure_checkpoint] reuse 50-epoch seed-0 {ckpt_path}", flush=True)
        return ckpt_path
    ckpt_path = channel_checkpoint_path(cfg(num_uav), num_uav, train_size)
    if ckpt_path.exists():
        print(f"[ensure_checkpoint] reuse existing {ckpt_path}", flush=True)
        return ckpt_path
    print(f"[ensure_checkpoint] training N={num_uav}, train_size={train_size}", flush=True)
    cmd = [
        sys.executable, "-u",
        "train/train_channel_gat_rician.py",
        "--num_uav", str(num_uav),
        "--train_size", str(train_size),
        "--prefix_pool_size", str(train_size),
        "--val_size", "128",
        "--test_size", "128",
        "--epochs", "50",
        "--seed", "10086",
        "--val_seed", "10087",
        "--test_seed", "10088",
        "--model-seed", "0",
        "--device", device,
        "--gpu-id", "0",
        "--dataset-root", f"datasets/paper/fig06_train_size_{train_size}",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return ckpt_path


def load_mixed_channel_gat(path: str | Path, config, device: str):
    """Load a balanced multi-N checkpoint without imposing a single train N."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Mixed-N GAT checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "model_version": "channel_gat_v1",
        "training_scope": "mixed_network_size",
        "tx_power_dbm": 10.0,
        "csi_mode": "perfect",
        "channel_model": "rician",
        "rician_k_db": 5.0,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in required.items()
        if checkpoint.get(key) != value
    }
    if mismatches or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Mixed-N checkpoint metadata mismatch in {checkpoint_path}: {mismatches}"
        )
    model = build_gat_scheduler(config, raw_node_dim=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), str(checkpoint_path), checkpoint


def evaluate(model, test):
    rates = []
    active = []
    audit = []
    with torch.no_grad():
        for i, s in enumerate(test):
            device = next(model.parameters()).device
            pi, _ = model(
                s.vertex_features[:, :1].to(device),
                s.edge_index.to(device),
                s.edge_weights.to(device),
            )
            candidate = (pi >= 0.5).float().cpu()
            act, dec = decode_endpoint_mutual_matching(
                candidate,
                s.links,
                priorities=direct_rate_priority(pi, s),
                max_rounds=1,
            )
            final = shared_uav_action_audit(act, s.links)
            if not final["feasible"]:
                raise RuntimeError("infeasible decoded action")
            r = float(
                calculate_achievable_rate(
                    act.reshape(-1, 1), s, binary=True, detach_hard=True
                ).sum()
            )
            rates.append(r)
            active.append(float(act.float().mean()))
            audit.append({
                "sample_index": i,
                "active_project_link_indices": torch.nonzero(
                    act.reshape(-1) > 0.5, as_tuple=True
                )[0].tolist(),
                "hard_feasibility": dec,
            })
    a = np.asarray(rates)
    return {
        "mean_sum_spectral_efficiency_bit_s_hz": float(a.mean()),
        "sem_sum_spectral_efficiency_bit_s_hz": float(np.std(a, ddof=1) / np.sqrt(len(a))),
        "mean_aggregate_phy_rate_mbps": float(a.mean() * 5),
        "mean_active_fraction": float(np.mean(active)),
        "per_sample_actions": audit,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        default="20,30,40,50",
        help="Comma-separated network sizes (default: 20,30,40,50)",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=300,
        help="Training set size (default: 300)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for cross-N results",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument(
        "--mixed-checkpoint",
        default="",
        help="Optional mixed-N checkpoint path",
    )
    args = parser.parse_args()
    ns = [int(x) for x in args.nodes.split(",") if x.strip()]
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[cross_n] device={device}  train_size={args.train_size}", flush=True)

    # Load test sets (all use the same train_size seed pool)
    tests = {n: test_states(n, args.train_size) for n in ns}

    matrix: dict = {}
    checkpoints: dict = {}
    checkpoint_metadata: dict = {}

    for tr in ns:
        ckpt_path = ensure_checkpoint(tr, args.train_size, device)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if args.train_size == 300:
            checkpoint_config = checkpoint.get("config", {})
            subset = checkpoint.get("training_subset", {})
            if (
                int(checkpoint_config.get("MODEL_SEED", -1)) != 0
                or len(checkpoint.get("per_epoch_logs", [])) != 50
                or int(subset.get("selected_size", -1)) != 300
                or int(subset.get("pool_size", -1)) != 300
            ):
                raise ValueError(
                    f"Unexpected canonical 50-epoch seed-0 checkpoint: {ckpt_path}"
                )
            m, ck = load_channel_gat_checkpoint(
                cfg(tr),
                ckpt_path,
                train_num_uav=tr,
                device=device,
                expected_dataset_fingerprint=tests[tr][2],
            )
        else:
            m, ck = load_channel_gat_checkpoint(
                cfg(tr), ckpt_path, train_num_uav=tr, device=device
            )
        checkpoints[str(tr)] = ck
        checkpoint_metadata[str(tr)] = {
            "model_seed": int(checkpoint.get("config", {}).get("MODEL_SEED", -1)),
            "epochs": len(checkpoint.get("per_epoch_logs", [])),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "dataset_fingerprint": checkpoint.get("dataset_fingerprint"),
        }
        matrix[str(tr)] = {}
        for te in ns:
            r = evaluate(m, tests[te][0])
            matrix[str(tr)][str(te)] = r
            print(
                f"  train N={tr} -> test N={te}: {r['mean_aggregate_phy_rate_mbps']:.3f} Mbit/s",
                flush=True,
            )

    if args.mixed_checkpoint:
        mixed, ck, meta = load_mixed_channel_gat(
            args.mixed_checkpoint, cfg(ns[0]), device
        )
        checkpoints["mixed"] = ck
        matrix["mixed"] = {}
        for te in ns:
            r = evaluate(mixed, tests[te][0])
            matrix["mixed"][str(te)] = r
            print(
                f"  train mixed -> test N={te}: {r['mean_aggregate_phy_rate_mbps']:.3f} Mbit/s",
                flush=True,
            )

    dedicated = {
        str(n): matrix[str(n)][str(n)]["mean_aggregate_phy_rate_mbps"] for n in ns
    }
    relative = {
        tr: {te: v["mean_aggregate_phy_rate_mbps"] / dedicated[te] for te, v in row.items()}
        for tr, row in matrix.items()
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "schema": "tnse_fig06_cross_n_v4",
                "nodes": ns,
                "train_size": args.train_size,
                "physics": {
                    "tx_power_dbm": 10,
                    "channel_model": "rician",
                    "rician_k_db": 5,
                    "service_channel": "current true instantaneous Rician channel",
                    "decoder": "one_round_endpoint_mutual_matching",
                },
                "checkpoints": checkpoints,
                "checkpoint_metadata": checkpoint_metadata,
                "test_sets": {
                    str(n): {
                        "path": tests[n][3],
                        "fingerprint": tests[n][1],
                        "fingerprint_scope": "test_split_only",
                        "dataset_family_fingerprint": tests[n][2],
                        "samples": len(tests[n][0]),
                    }
                    for n in ns
                },
                "results": matrix,
                "relative_to_dedicated_test_model": relative,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cross_n] results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
