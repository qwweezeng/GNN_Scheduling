#!/usr/bin/env python3
"""Paper Fig. 4: full-buffer Channel-GAT sensitivity to observed CSI.

One audited perfect-CSI GAT is frozen.  Only its observed direct/cross-power
matrix changes; every schedule is settled on the original true Rician channel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.model import calculate_achievable_rate  # noqa: E402
from simulator.mobile_block_fading import build_conflict_graph_from_power  # noqa: E402
from utils.artifacts import (  # noqa: E402
    expected_dataset_metadata,
    load_validated_dataset,
    paper_dataset_paths,
)
from utils.network_generate import (  # noqa: E402
    NetworkState,
    dequantize_distance_from_binindex,
)
from utils.project_setup import (  # noqa: E402
    build_network_generator,
    load_channel_gat,
    load_config,
)


def _observed_state(true_state, power, label, snr_threshold_db):
    links = true_state.links
    direct = power[links[:, 0], links[:, 1]]
    snr_db = 10.0 * torch.log10(direct / true_state.noise_power + 1e-12)
    edge_index, edge_weights = build_conflict_graph_from_power(
        links, power, true_state.noise_power, 10.0 ** (snr_threshold_db / 10.0)
    )
    neighbors = {idx: [] for idx in range(int(links.size(0)))}
    for source, target in edge_index.t().tolist():
        neighbors[target].append(source)
    return NetworkState(
        links=links,
        vertex_features=snr_db.unsqueeze(1).float(),
        edge_index=edge_index,
        edge_weights=edge_weights,
        large_scale=true_state.large_scale,
        signal_power=power,
        signal_power_large_scale=power,
        signal_power_quant=power,
        noise_power=true_state.noise_power,
        bandwidth=true_state.bandwidth,
        snr_gap=true_state.snr_gap,
        neighbor_map=neighbors,
        positions=true_state.positions,
        tx_power=true_state.tx_power,
        carrier_f=true_state.carrier_f,
        csi_mode=label,
        queue_states=None,
        include_queue_feature=False,
    )


def _quantized_large_scale(state, generator, bits):
    distance = torch.cdist(state.positions, state.positions)
    d_min = float(generator.min_distance)
    height_span = float(generator.height_max - generator.height_min)
    d_max = float((generator.area_width ** 2 * 2 + height_span ** 2) ** 0.5)
    fraction = ((distance - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
    index = torch.floor(fraction * (2 ** bits - 1))
    reconstructed = dequantize_distance_from_binindex(
        index, q=bits, d_min=d_min, d_max=d_max
    )
    power = generator.tx_power * generator.lambda0 * reconstructed.pow(-generator.xi)
    power.fill_diagonal_(0.0)
    return power.float()


def _evaluate(model, true_states, observed_powers, label, snr_threshold_db):
    utilities, active_fractions, errors = [], [], []
    with torch.no_grad():
        for true_state, power in zip(true_states, observed_powers):
            observed = _observed_state(
                true_state, power, label, snr_threshold_db
            )
            probability, _ = model(
                observed.vertex_features, observed.edge_index, observed.edge_weights
            )
            action = (probability >= 0.5).float()
            utilities.append(float(calculate_achievable_rate(
                action, true_state, binary=True, detach_hard=True
            ).sum()))
            active_fractions.append(float(action.mean()))
            links = true_state.links
            actual = true_state.signal_power[links[:, 0], links[:, 1]]
            estimate = power[links[:, 0], links[:, 1]]
            errors.append(float(torch.mean(torch.abs(
                10.0 * torch.log10(estimate.clamp_min(1e-30))
                - 10.0 * torch.log10(actual.clamp_min(1e-30))
            ))))
    values = np.asarray(utilities)
    return {
        "mean_true_channel_utility": float(values.mean()),
        "standard_error": float(values.std(ddof=1) / np.sqrt(len(values)))
        if len(values) > 1 else 0.0,
        "mean_active_fraction": float(np.mean(active_fractions)),
        "mean_direct_power_absolute_error_db": float(np.mean(errors)),
        "per_sample_utility": values.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--q_bits", default="2,3,4,6,8")
    parser.add_argument(
        "--output", default="results/paper/channel_gat_csi/sensitivity_N20.json"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": 500, "val": 128, "test": 128}
    metadata = expected_dataset_metadata(config, args.num_uav, seeds, sizes)
    _, test_path = paper_dataset_paths(Path("datasets/paper"), args.num_uav, seeds, sizes)
    true_states = load_validated_dataset(
        test_path, metadata, ("test",)
    )["test_set"]
    model, checkpoint = load_channel_gat(config, args.num_uav, device="cpu")
    generator = build_network_generator(config)

    powers = {
        "perfect": [state.signal_power for state in true_states],
        "large_scale": [state.signal_power_large_scale for state in true_states],
    }
    for bits in [int(value) for value in args.q_bits.split(",") if value.strip()]:
        powers[f"quantized_{bits}bit"] = [
            _quantized_large_scale(state, generator, bits) for state in true_states
        ]
    results = {
        label: _evaluate(model, true_states, matrices, label, float(config["SNR_THRES"]))
        for label, matrices in powers.items()
    }
    for label, result in results.items():
        print(f"{label}: {result['mean_true_channel_utility']:.4f}")

    output = Path(args.output)
    if output.name.endswith("N20.json") and args.num_uav != 20:
        output = output.with_name(f"sensitivity_N{args.num_uav}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "experiment": "paper_channel_gat_csi_observation_sensitivity_v1",
        "definition": (
            "same frozen perfect-CSI GAT; only observed CSI changes; all "
            "service is settled on each sample's true Rician channel"
        ),
        "num_uav": args.num_uav,
        "dataset_path": str(test_path),
        "dataset_fingerprint": metadata["fingerprint"],
        "checkpoint": checkpoint,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
