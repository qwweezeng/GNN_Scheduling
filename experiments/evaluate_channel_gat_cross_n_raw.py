#!/usr/bin/env python3
"""Raw-π cross-N evaluation: measure GAT's intrinsic generalization without
any feasibility decoder.

Each model checkpoint produces scheduling probabilities π_i for every link in
every test sample.  The binary decision a_i = 1(π_i ≥ 0.5) is settled
directly on the true channel -- no endpoint matching, no SNR-priority greedy,
no projection.  This isolates the GAT's learned link-quality estimator from
any downstream heuristic, revealing whether the model truly generalizes across
network scales or merely benefits from a decoder that masks its errors.

Usage:
    uv run python experiments/evaluate_channel_gat_cross_n_raw.py \
        --nodes 20,30,40,50 \
        --device cuda \
        --output paper_figures/fig06_cross_n/data/raw/cross_n_raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model import calculate_achievable_rate
from utils.artifacts import expected_dataset_metadata, load_validated_dataset
from utils.artifacts import paper_dataset_paths as _paper_dataset_paths
from utils.project_setup import load_channel_gat, load_config


def paper_dataset_paths(root, n_uav, seeds, sizes):
    _, test_path = _paper_dataset_paths(root, n_uav, seeds, sizes)
    return test_path


def cfg(n):
    c = load_config(ROOT / "configs/paper.yaml")
    c.update({
        "NUM_UAV": n,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    return c


def load_test_set(n):
    c = cfg(n)
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 300, "val": 128, "test": 128}
    meta = expected_dataset_metadata(c, n, seeds, sizes)
    test_path = paper_dataset_paths(ROOT / "datasets/paper", n, seeds, sizes)
    payload = load_validated_dataset(test_path, meta, ("test",))
    return payload["test_set"], meta["fingerprint"], str(test_path)


def evaluate_raw_pi(model, test_states):
    """Return per-sample results using raw GAT π without any feasibility decoder.

    Each returned sample record contains:
    - sample_index, num_links
    - raw_pi: list of π values (one per link)
    - binary_action: list of 0/1 decisions (π ≥ 0.5)
    - sum_spectral_efficiency_bit_s_hz: rate from binary decision
    - active_link_indices: which links are active
    """
    results = []
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for i, state in enumerate(test_states):
            pi, _ = model(
                state.vertex_features[:, :1].to(device),
                state.edge_index.to(device),
                state.edge_weights.to(device),
            )
            pi_cpu = pi.reshape(-1).float().cpu()
            binary = (pi_cpu >= 0.5).float()

            rate = float(
                calculate_achievable_rate(
                    binary.reshape(-1, 1),
                    state,
                    binary=True,
                    detach_hard=True,
                ).sum()
            )

            active = torch.nonzero(binary > 0.5, as_tuple=True)[0].tolist()
            results.append({
                "sample_index": i,
                "num_links": int(state.num_vertices),
                "raw_pi": pi_cpu.tolist(),
                "binary_action": binary.tolist(),
                "sum_spectral_efficiency_bit_s_hz": rate,
                "aggregate_phy_rate_mbps": rate * float(state.bandwidth) / 1e6,
                "active_link_indices": active,
                "active_fraction": float(binary.mean()),
                "pi_top10_values": sorted(pi_cpu.tolist(), reverse=True)[:10],
            })

    return results


def summarize(results):
    rates = [r["sum_spectral_efficiency_bit_s_hz"] for r in results]
    active = [r["active_fraction"] for r in results]
    n = len(rates)
    return {
        "num_samples": n,
        "mean_sum_spectral_efficiency_bit_s_hz": float(np.mean(rates)),
        "sem_sum_spectral_efficiency_bit_s_hz": float(
            np.std(rates, ddof=1) / np.sqrt(n)
        ),
        "mean_aggregate_phy_rate_mbps": float(np.mean(rates) * 5),
        "sem_aggregate_phy_rate_mbps": float(
            np.std(rates, ddof=1) / np.sqrt(n) * 5
        ),
        "mean_active_fraction": float(np.mean(active)),
        "std_active_fraction": float(np.std(active, ddof=1)),
        "per_sample": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes", default="20,30,40,50",
        help="Comma-separated list of N values",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
    )
    args = parser.parse_args()

    ns = [int(x) for x in args.nodes.split(",") if x]
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[raw-pi-crossn] device={device}, nodes={ns}", flush=True)

    # Load test sets
    test_sets = {}
    for n in ns:
        states, fp, path = load_test_set(n)
        test_sets[n] = (states, fp, path)
        print(f"  test N={n}: {len(states)} samples, fp={fp[:16]}...", flush=True)

    # Evaluate every (train_N, test_N) pair
    matrix = {}
    checkpoints = {}
    for train_n in ns:
        model, ckpt_path = load_channel_gat(cfg(train_n), train_n, device=device)
        checkpoints[str(train_n)] = ckpt_path
        matrix[str(train_n)] = {}
        print(f"\n[checkpoint N={train_n}] {ckpt_path}", flush=True)

        for test_n in ns:
            results = evaluate_raw_pi(model, test_sets[test_n][0])
            summary = summarize(results)
            matrix[str(train_n)][str(test_n)] = summary
            print(
                f"  train N={train_n} -> test N={test_n}: "
                f"{summary['mean_sum_spectral_efficiency_bit_s_hz']:.4f} bit/s/Hz",
                flush=True,
            )

    # Relative to dedicated (diagonal) model
    dedicated = {
        str(n): matrix[str(n)][str(n)]["mean_sum_spectral_efficiency_bit_s_hz"]
        for n in ns
    }
    relative = {
        tr: {
            te: matrix[tr][te]["mean_sum_spectral_efficiency_bit_s_hz"]
            / dedicated[te]
            for te in matrix[tr]
        }
        for tr in matrix
    }

    # Per-sample model agreement analysis
    # For each test_N, check how often different train_N models agree on binary action
    agreement = {}
    for test_n in ns:
        states = test_sets[test_n][0]
        pi_by_model = {}
        for train_n in ns:
            model, _ = load_channel_gat(cfg(train_n), train_n, device=device)
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                pis = []
                for s in states:
                    pi, _ = model(
                        s.vertex_features[:, :1].to(device),
                        s.edge_index.to(device),
                        s.edge_weights.to(device),
                    )
                    pis.append(pi.reshape(-1).float().cpu())
            pi_by_model[str(train_n)] = pis

        # Agreement: for each sample, count how many model pairs agree on each link
        n_models = len(pi_by_model)
        n_samples = len(states)
        n_links = len(pi_by_model[str(ns[0])][0])

        per_sample_agree = []
        for si in range(n_samples):
            all_pi = torch.stack([pi_by_model[str(tr)][si] for tr in ns], dim=0)
            binary_stack = (all_pi >= 0.5).float()
            mean_binary = binary_stack.mean(dim=0)
            # Agreement fraction: fraction of links where all models agree
            all_agree = (mean_binary == 0.0) | (mean_binary == 1.0)
            agree_frac = float(all_agree.float().mean())
            per_sample_agree.append(agree_frac)

        agreement[str(test_n)] = {
            "mean_agreement": float(np.mean(per_sample_agree)),
            "sem_agreement": float(np.std(per_sample_agree, ddof=1) / np.sqrt(len(per_sample_agree))),
        }

    payload = {
        "schema": "tnse_fig06_cross_n_raw_v1",
        "description": (
            "Raw-π cross-N evaluation: GAT π → binary(π ≥ 0.5) → Shannon rate "
            "settlement, NO feasibility decoder. Isolates GAT's learned link-quality "
            "estimator from any downstream heuristic."
        ),
        "nodes": ns,
        "physics": {
            "tx_power_dbm": 10.0,
            "channel_model": "rician",
            "rician_k_db": 5.0,
            "service_channel": "current true instantaneous Rician channel",
            "decision_rule": "binary(pi >= 0.5)",
            "feasibility_decoder": "none",
        },
        "checkpoints": checkpoints,
        "test_sets": {
            str(n): {
                "path": test_sets[n][2],
                "fingerprint": test_sets[n][1],
                "samples": len(test_sets[n][0]),
            }
            for n in ns
        },
        "results": matrix,
        "relative_to_dedicated_model": relative,
        "model_agreement": agreement,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2))
    print(f"\n[saved] {args.output}", flush=True)


if __name__ == "__main__":
    main()
