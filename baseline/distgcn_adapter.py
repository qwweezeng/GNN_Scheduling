"""Physics-to-MWIS adapter for the Zhao et al. DistGCN baseline.

This file deliberately stops at constructing the input expected by a hard
conflict-graph solver.  It does not claim that the adapted MWIS objective is
equivalent to the original directed physical-interference objective.
"""

import torch


def build_hardened_mwis_instance(
    state, queues, inr_threshold_db=0.0, enforce_shared_uav=True
):
    """Return symmetric hard adjacency and queue-weighted solo-link values.

    A directed physical relation ``j -> i`` is hardened when the interference
    to link i's receiver is at least ``inr_threshold_db`` relative to noise.
    The final MWIS adjacency is the OR of both directions, as required by an
    undirected independent-set solver.  Links sharing any UAV are always made
    mutually exclusive when ``enforce_shared_uav`` is true.
    """
    queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1).cpu()
    links = state.links.detach().cpu()
    mode = getattr(state, "csi_mode", "perfect")
    if mode == "large_scale":
        signal_power = getattr(state, "signal_power_large_scale", None)
    elif mode == "quantized":
        signal_power = getattr(state, "signal_power_quant", None)
    else:
        signal_power = state.signal_power
    if signal_power is None:
        signal_power = state.signal_power
    signal_power = signal_power.detach().cpu().float()

    gain = signal_power[links[:, 0, None], links[None, :, 1]]
    direct = signal_power[links[:, 0], links[:, 1]]
    threshold_linear = 10.0 ** (float(inr_threshold_db) / 10.0)
    directed = gain >= float(state.noise_power) * threshold_linear
    directed.fill_diagonal_(False)
    adjacency = directed | directed.T

    if enforce_shared_uav:
        endpoints = (
            links[:, :, None, None] == links[None, None, :, :]
        )
        adjacency |= endpoints.any(dim=1).any(dim=-1)
        adjacency.fill_diagonal_(False)

    solo_rate = torch.log2(
        1.0
        + direct / float(state.noise_power)
        / float(getattr(state, "snr_gap", 1.0))
    )
    node_weights = queues.clamp_min(0.0) * solo_rate
    metadata = {
        "adapter": "directed_physical_to_symmetric_hard_mwis_v1",
        "inr_threshold_db": float(inr_threshold_db),
        "enforce_shared_uav": bool(enforce_shared_uav),
        "num_nodes": int(links.size(0)),
        "num_undirected_edges": int(torch.triu(adjacency, diagonal=1).sum()),
        "warning": (
            "MWIS forbids every hardened pair and therefore cannot represent "
            "weak-interference concurrency or direction-dependent loss."
        ),
    }
    return adjacency, node_weights, metadata
