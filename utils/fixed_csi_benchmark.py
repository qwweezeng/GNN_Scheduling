"""Reusable mechanics for paired fixed-CSI queue benchmarks."""

import copy
import random
from pathlib import Path

import numpy as np
import torch

from simulator.queue_simulator import QueueSimulator
from utils.project_setup import build_network_generator


HOTSPOT_LINK_FRACTION = 0.20
HOTSPOT_TRAFFIC_FRACTION = 0.70

SCALAR_METRICS = (
    "final_avg_queue",
    "average_queue_backlog",
    "average_queue_backlog_mbit",
    "average_delay_ms",
    "average_packet_delay_ms",
    "p95_packet_delay_ms",
    "packet_delay_count",
    "delivered_packet_count",
    "successful_packet_total_rate_pps",
    "successful_delivery_ratio",
    "successful_goodput_mbps",
    "deadline_dropped_packets",
    "overflow_dropped_packets",
    "unfinished_packet_count",
    "packet_loss_rate",
    "unfinished_packet_fraction",
    "resolved_packet_fraction",
    "qos_eligible_arrivals",
    "qos_eligible_delivered",
    "qos_packet_loss_rate",
    "total_arrivals",
    "total_departures",
    "throughput_ratio",
    "throughput_mbps",
    "departure_total_rate_pps",
    "average_active_links",
    "average_nonempty_links",
    "average_mws_utility",
    "queue_growth_slope",
    "queue_overflow_prob",
    "hotspot_average_queue_backlog_mbit",
    "normal_average_queue_backlog_mbit",
)

TRAJECTORY_METRICS = (
    "avg_queue_trajectory",
    "queue_backlog_mbit_trajectory",
    "throughput_trajectory",
    "departures_per_slot",
    "active_links_per_slot",
)


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def lambda_key(value):
    return format(float(value), ".12g")


def arrival_trace_seed(mc_seed, arrival_lambda):
    code = int(round(float(arrival_lambda) * 1_000_000))
    return int(mc_seed) + 1_000_003 * code


def generate_hotspot_arrivals(
    num_slots, num_links, total_lambda, arrival_seed, hotspot_seed
):
    """Generate one paired link-hotspot Poisson arrival trace."""
    total_lambda = float(total_lambda)
    hotspot_rng = np.random.default_rng(int(hotspot_seed))
    num_hotspot = min(
        max(1, int(np.ceil(num_links * HOTSPOT_LINK_FRACTION))),
        num_links,
    )
    hotspot_indices = np.sort(hotspot_rng.choice(
        num_links, size=num_hotspot, replace=False
    ))
    num_normal = num_links - num_hotspot
    hotspot_share = HOTSPOT_TRAFFIC_FRACTION if num_normal > 0 else 1.0
    rates = np.zeros(num_links, dtype=np.float64)
    rates[:] = (
        total_lambda * (1.0 - hotspot_share) / num_normal
        if num_normal > 0 else 0.0
    )
    rates[hotspot_indices] = total_lambda * hotspot_share / num_hotspot
    rng = np.random.default_rng(int(arrival_seed))
    arrivals = rng.poisson(
        lam=rates[None, :], size=(int(num_slots), int(num_links))
    ).astype(np.float32)
    profile = {
        "arrival_profile": "link_hotspot_total",
        "arrival_total_lambda": total_lambda,
        "hotspot_link_fraction": HOTSPOT_LINK_FRACTION,
        "hotspot_traffic_fraction": hotspot_share,
        "num_links": int(num_links),
        "num_hotspot_links": int(num_hotspot),
        "hotspot_indices": hotspot_indices.astype(int).tolist(),
    }
    return torch.from_numpy(arrivals), profile


def build_fixed_csi_trace_cache(
    config, num_slots, arrival_rates, mc_seeds
):
    """Store one fixed channel graph per seed plus lightweight arrivals."""
    generator = build_network_generator(config)
    traces = {}
    topology_by_seed = {}
    for seed in mc_seeds:
        set_seed(seed)
        state = generator.generate_network(
            csi_mode=config.get("CSI_MODE", "perfect"),
            include_queue_feature=False,
        )
        links = int(state.num_vertices)
        edges = int(state.edge_index.size(1))
        topology_by_seed[str(int(seed))] = {
            "num_links": links,
            "num_directed_conflict_edges": edges,
            "directed_conflict_density": (
                edges / max(links * (links - 1), 1)
            ),
        }
        arrivals, profiles = {}, {}
        for arrival_lambda in arrival_rates:
            key = lambda_key(arrival_lambda)
            arrivals[key], profiles[key] = generate_hotspot_arrivals(
                num_slots,
                state.num_vertices,
                arrival_lambda,
                arrival_trace_seed(seed, arrival_lambda),
                seed,
            )
        traces[str(int(seed))] = {
            "channel_state": state,
            "arrival_realizations": arrivals,
            "arrival_profiles": profiles,
        }
    return {
        "format": "fixed_csi_queue_trace_v2",
        "metadata": {
            "num_uav": int(config["NUM_UAV"]),
            "num_slots": int(num_slots),
            "tx_power_dbm": float(config.get("TX_POWER", 10.0)),
            "csi_mode": str(config.get("CSI_MODE", "perfect")),
            "channel_model": str(config.get("CHANNEL_MODEL", "rayleigh")),
            "rician_k_db": (
                float(config.get("RICIAN_K_DB", 5.0))
                if str(config.get("CHANNEL_MODEL", "rayleigh")) == "rician"
                else None
            ),
            "arrival_rates": [float(value) for value in arrival_rates],
            "mc_seeds": [int(value) for value in mc_seeds],
            "topology_by_seed": topology_by_seed,
        },
        "traces": traces,
    }


def save_trace_cache(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


def load_trace_cache(path, config, num_slots, arrival_rates, mc_seeds):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") not in {
        "fixed_csi_queue_trace_v2", "queue_benchmark_trace_v1"
    }:
        raise ValueError(f"Unsupported trace cache format: {path}")
    metadata = payload.get("metadata", {})
    expected = {
        "num_uav": int(config["NUM_UAV"]),
        "num_slots": int(num_slots),
        "tx_power_dbm": float(config.get("TX_POWER", 10.0)),
        "csi_mode": str(config.get("CSI_MODE", "perfect")),
        "channel_model": str(config.get("CHANNEL_MODEL", "rayleigh")),
        "rician_k_db": (
            float(config.get("RICIAN_K_DB", 5.0))
            if str(config.get("CHANNEL_MODEL", "rayleigh")) == "rician"
            else None
        ),
    }
    mismatches = [
        f"{key}: cache={metadata.get(key)!r}, expected={value!r}"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    for seed in mc_seeds:
        if str(int(seed)) not in payload.get("traces", {}):
            mismatches.append(f"missing seed={seed}")
            continue
        item = payload["traces"][str(int(seed))]
        arrivals = item.get("arrival_realizations", {})
        for value in arrival_rates:
            if lambda_key(value) not in arrivals:
                mismatches.append(f"missing Lambda={value} at seed={seed}")
    if mismatches:
        raise ValueError("Trace cache mismatch: " + "; ".join(mismatches))
    return payload


def trace_item(payload, seed, arrival_lambda):
    item = payload["traces"][str(int(seed))]
    state = item.get("channel_state")
    if state is None:
        states = item.get("channel_realizations", [])
        state = states[0]
    key = lambda_key(arrival_lambda)
    return (
        state,
        item["arrival_realizations"][key],
        item["arrival_profiles"][key],
    )


def run_scheduler(
    config,
    state,
    arrivals,
    scheduler,
    include_queue_feature=False,
    track_packet_delay=True,
    delay_deadline_ms=None,
    buffer_size_mbit=None,
):
    simulator = QueueSimulator(
        network_generator=build_network_generator(config),
        model=scheduler,
        device="cpu",
        include_queue_feature=bool(include_queue_feature),
    )
    simulator.configure(
        arrival_lambda=0.0,
        slot_duration=config.get("SLOT_DURATION", 0.001),
        packet_size=config.get("PACKET_SIZE", 12000),
        slots_per_block=None,
        track_packet_delay=track_packet_delay,
        delay_deadline_ms=delay_deadline_ms,
        buffer_size_mbit=buffer_size_mbit,
    )
    return simulator.run(
        num_slots=int(arrivals.shape[0]),
        csi_mode=config.get("CSI_MODE", "perfect"),
        verbose=False,
        channel_realizations=[copy.deepcopy(state)],
        arrival_realizations=arrivals,
    )


def add_hotspot_metrics(metrics, profile, packet_size):
    """Add hotspot/normal backlog from the full queue trajectory."""
    queues = np.asarray(metrics.get("queue_trajectory"), dtype=float)
    if queues.ndim != 2 or queues.shape[1] == 0:
        return metrics
    mask = np.zeros(queues.shape[1], dtype=bool)
    indices = [
        int(index) for index in profile.get("hotspot_indices", [])
        if 0 <= int(index) < queues.shape[1]
    ]
    if not indices:
        return metrics
    mask[indices] = True
    scale = float(packet_size) / 1e6
    metrics["hotspot_average_queue_backlog_mbit"] = float(
        queues[:, mask].mean() * scale
    )
    metrics["normal_average_queue_backlog_mbit"] = (
        float(queues[:, ~mask].mean() * scale) if (~mask).any() else None
    )
    return metrics


def ci95(values):
    values = np.asarray(values, dtype=float)
    if values.size <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(values.size))


def aggregate_runs(runs):
    """Aggregate scalar metrics and selected trajectories across MC seeds."""
    summary = {"num_mc_runs": len(runs)}
    for key in SCALAR_METRICS:
        values = [
            float(run[key]) for run in runs
            if run.get(key) is not None and np.isfinite(float(run[key]))
        ]
        summary[key] = float(np.mean(values)) if values else None
        summary[f"{key}_ci95"] = ci95(values) if values else None
    stable = [float(run["is_stable"]) for run in runs]
    summary["stable_fraction"] = float(np.mean(stable)) if stable else None

    trajectories = {}
    for key in TRAJECTORY_METRICS:
        arrays = [np.asarray(run[key], dtype=float) for run in runs if key in run]
        if not arrays or any(array.shape != arrays[0].shape for array in arrays):
            continue
        stack = np.stack(arrays)
        trajectories[key] = stack.mean(axis=0)
        trajectories[f"{key}_ci95"] = (
            1.96 * stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
            if stack.shape[0] > 1 else np.zeros_like(stack[0])
        )
    return summary, trajectories


def compact_run(metrics):
    """Keep scalar evidence without embedding large trajectories in JSON."""
    compact = {}
    for key, value in metrics.items():
        if key == "queue_trajectory" or key in TRAJECTORY_METRICS:
            continue
        if isinstance(value, np.ndarray):
            continue
        compact[key] = value
    return compact
