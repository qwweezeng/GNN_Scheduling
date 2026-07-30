#!/usr/bin/env python3
"""Evaluate one CSI-conditioned channel GAT checkpoint.

For each (CSI mode, quantization bits) the model was trained on its own
observed-CSI dataset.  At evaluation time the same observed CSI is the model
input, and service is settled on that same observed power matrix (matching
the training data-generating distribution).

Usage:
    # N=20, large_scale CSI
    uv run python experiments/evaluate_channel_gat_csi.py \
        --num-uav 20 --csi-mode large_scale --device cuda \
        --output paper_figures/fig07_csi_feedback/data/raw/chapter4_N20_v2_large_scale.json

    # N=20, 4-bit quantized CSI
    uv run python experiments/evaluate_channel_gat_csi.py \
        --num-uav 20 --csi-mode quantized --q-bits 4 --device cuda \
        --output paper_figures/fig07_csi_feedback/data/raw/chapter4_N20_v2_quantized_4bit.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model import calculate_achievable_rate
from simulator.mobile_block_fading import build_conflict_graph_from_power
from utils.artifacts import (
    expected_dataset_metadata_quantized,
    load_validated_dataset,
    paper_dataset_paths_quantized,
)
from utils.baseline import local_greedy
from utils.feasibility import (
    decode_endpoint_mutual_matching,
    project_shared_uav_feasible,
    shared_uav_action_audit,
)
from utils.network_generate import NetworkState, dequantize_distance_from_binindex
from utils.project_setup import (
    build_network_generator,
    channel_checkpoint_path,
    build_gat_scheduler,
    load_config,
)


def make_config(num_uav, csi_mode, q_bits):
    cfg = load_config(ROOT / "configs/paper.yaml")
    cfg.update({
        "NUM_UAV": num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": csi_mode,
        "Q_BITS": q_bits,
    })
    return cfg


def load_trained_model(config, num_uav, device):
    """Load a checkpoint that was trained under the same CSI condition."""
    ckpt_path = channel_checkpoint_path(config, num_uav)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train a model with: python train/train_channel_gat_rician.py "
            f"--num_uav {num_uav} --csi-mode {config['CSI_MODE']} "
            + (f"--q-bits {config['Q_BITS']} " if config.get("Q_BITS") else "")
            + "--device cuda --epochs 50"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Build model and load weights
    raw_dim = 1  # channel-only training uses SNR only
    model = build_gat_scheduler(config, raw_node_dim=raw_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), str(ckpt_path), ckpt


def load_test_dataset(config, num_uav, csi_mode, q_bits):
    """Load the test split for the matching CSI condition."""
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 300, "val": 128, "test": 128}
    metadata = expected_dataset_metadata_quantized(
        config, num_uav, seeds, sizes, csi_mode, q_bits,
    )
    _, test_path = paper_dataset_paths_quantized(
        ROOT / "datasets/paper", num_uav, seeds, sizes, csi_mode, q_bits,
    )
    dataset = load_validated_dataset(test_path, metadata, ("test",))
    return dataset["test_set"], metadata["fingerprint"], str(test_path)


def build_observed_network_state(state, power_matrix, label, snr_threshold):
    """Build a NetworkState from the dataset sample and an observed power matrix."""
    links = state.links
    tx, rx = links[:, 0], links[:, 1]
    snr_linear = power_matrix[tx, rx] / state.noise_power
    snr_db = 10.0 * torch.log10(snr_linear + 1e-12)
    ei, ew = build_conflict_graph_from_power(
        links, power_matrix, state.noise_power, 10 ** (snr_threshold / 10.0)
    )
    neighbor_map = {i: [] for i in range(len(links))}
    for src, dst in ei.t().tolist():
        neighbor_map[dst].append(src)
    return NetworkState(
        links=links,
        vertex_features=snr_db.unsqueeze(1).float(),
        edge_index=ei,
        edge_weights=ew,
        large_scale=state.large_scale,
        signal_power=power_matrix,
        signal_power_large_scale=power_matrix,
        signal_power_quant=power_matrix,
        noise_power=state.noise_power,
        bandwidth=state.bandwidth,
        snr_gap=state.snr_gap,
        neighbor_map=neighbor_map,
        positions=state.positions,
        tx_power=state.tx_power,
        carrier_f=state.carrier_f,
        csi_mode=label,
        queue_states=None,
        include_queue_feature=False,
    )


def direct_rate_priority(pi, observed_state):
    """Priority = π_i · log2(1 + SNR_i / γ)."""
    tx = observed_state.links[:, 0]
    rx = observed_state.links[:, 1]
    snr = observed_state.signal_power[tx, rx].cpu() / observed_state.noise_power
    return pi.reshape(-1).float().cpu() * torch.log2(1.0 + snr / observed_state.snr_gap).float()


def settle_with_matching(a, priorities, observed_state, service_state, use_matching):
    """Project an observed-CSI decision, then settle it on the true channel.

    ``observed_state`` is used only for feasible-action decoding.  The service
    rate must use ``service_state.signal_power``, which preserves the true
    Rician realization carried by the matched dataset sample.
    """
    if use_matching:
        a, proj = decode_endpoint_mutual_matching(
            a, observed_state.links, priorities=priorities, max_rounds=1
        )
    else:
        a, proj = project_shared_uav_feasible(
            a, observed_state.links, priorities=priorities
        )
    audit = shared_uav_action_audit(a, observed_state.links)
    if not audit["feasible"]:
        raise RuntimeError(f"infeasible action: {audit}")

    a_flat = torch.as_tensor(a).reshape(-1)
    rate = float(
        calculate_achievable_rate(
            a_flat.reshape(-1, 1), service_state,
            binary=True, detach_hard=True,
        ).sum()
    )
    return rate, float(a_flat.float().mean()), proj


def evaluate_one(model, test_states, power_extractor, label, snr_threshold, device):
    """Run GAT + local greedy on all test samples, return per-sample results."""
    gat_results = []
    greedy_results = []

    with torch.no_grad():
        for i, state in enumerate(test_states):
            power = power_extractor(state)

            # Build observed graph for this CSI mode
            obs_state = build_observed_network_state(state, power, label, snr_threshold)

            # --- GAT ---
            pi, _ = model(
                obs_state.vertex_features.to(device),
                obs_state.edge_index.to(device),
                obs_state.edge_weights.to(device),
            )
            gat_action = (pi >= 0.5).float().cpu()
            gat_priority = direct_rate_priority(pi, obs_state)
            gat_rate, gat_active_frac, gat_proj = settle_with_matching(
                gat_action, gat_priority, obs_state, state, use_matching=True
            )
            gat_results.append((gat_rate, gat_active_frac, {
                "sample_index": i,
                "hard_feasibility": gat_proj,
            }))

            # --- Local Greedy baseline ---
            greedy_action = local_greedy(obs_state, interference_threshold=0.3)
            greedy_priority = obs_state.vertex_features[:, 0].float().cpu()
            greedy_rate, greedy_active_frac, greedy_proj = settle_with_matching(
                greedy_action, greedy_priority, obs_state, state, use_matching=False
            )
            greedy_results.append((greedy_rate, greedy_active_frac, {
                "sample_index": i,
                "hard_feasibility": greedy_proj,
            }))

    def summarize(name, results):
        rates = [r[0] for r in results]
        active = [r[1] for r in results]
        return {
            f"mean_aggregate_phy_rate_mbps": float(np.mean(rates) * 5),
            f"sem_aggregate_phy_rate_mbps": float(
                np.std(rates, ddof=1) / np.sqrt(len(rates)) * 5
            ),
            "mean_active_fraction": float(np.mean(active)),
            "per_sample_actions": [r[2] for r in results],
        }

    return {
        "Proposed-GAT": summarize("Proposed-GAT", gat_results),
        "Local Greedy": summarize("Local Greedy", greedy_results),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-uav", type=int, default=20)
    parser.add_argument(
        "--csi-mode", default="perfect",
        choices=("perfect", "large_scale", "quantized"),
    )
    parser.add_argument("--q-bits", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.csi_mode == "quantized" and args.q_bits is None:
        raise ValueError("--q-bits is required when --csi-mode is 'quantized'")
    if args.csi_mode != "quantized" and args.q_bits is not None:
        raise ValueError("--q-bits may only be set for quantized mode")

    config = make_config(args.num_uav, args.csi_mode, args.q_bits)
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[evaluate] csi_mode={args.csi_mode} q_bits={args.q_bits} device={device}", flush=True)

    # Load model
    model, ckpt_path, _ = load_trained_model(config, args.num_uav, device)
    print(f"[evaluate] checkpoint: {ckpt_path}", flush=True)

    # Load test dataset
    test_set, fp, dataset_path = load_test_dataset(
        config, args.num_uav, args.csi_mode, args.q_bits
    )
    print(f"[evaluate] test_set={len(test_set)} samples, fingerprint={fp[:16]}...", flush=True)

    # Build power extractor
    if args.csi_mode == "large_scale":
        power_extractor = lambda s: s.signal_power_large_scale
    elif args.csi_mode == "quantized":
        # signal_power_quant carries the quantized power for quantized-trained samples
        power_extractor = lambda s: s.signal_power_quant
    else:
        power_extractor = lambda s: s.signal_power

    # Run evaluation
    snr_threshold = float(config["SNR_THRES"])
    results = evaluate_one(
        model, test_set, power_extractor,
        args.csi_mode, snr_threshold, device,
    )

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema": "tnse_fig07_csi_v2",
        "num_uav": args.num_uav,
        "csi_mode": args.csi_mode,
        "q_bits": args.q_bits,
        "dataset_fingerprint": fp,
        "dataset_path": dataset_path,
        "checkpoint": ckpt_path,
        "definition": (
            "model trained under matching observed CSI; observed CSI forms the "
            "GAT input and decoding graph, while service is settled on the "
            "underlying true Rician channel realization"
        ),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"[evaluate] saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
