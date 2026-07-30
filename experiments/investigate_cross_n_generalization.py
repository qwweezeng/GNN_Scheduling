#!/usr/bin/env python3
"""
Diagnostic Investigation: Why does N=40-trained GAT fail on N=20 topology?

This script performs a thorough investigation of the generalization failure,
testing multiple hypotheses:
1. π distribution difference between N=40 and N=20 models
2. Priority computation: pi vs pi*C_solo
3. Projection algorithm: greedy vs endpoint mutual matching
4. Topology differences affecting decision boundaries

Usage:
    uv run python experiments/investigate_cross_n_generalization.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model import calculate_achievable_rate
from utils.artifacts import expected_dataset_metadata, paper_dataset_paths
from utils.feasibility import (
    decode_endpoint_mutual_matching,
    project_shared_uav_feasible,
    shared_uav_action_audit,
)
from utils.project_setup import (
    load_channel_gat,
    load_config,
)


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


def load_test_set(n: int, train_size: int = 500):
    """Load test set for network size n."""
    seeds = {"train": 10086, "val": 10087, "test": 10088}
    sizes = {"train": train_size, "val": 128, "test": 128}
    _, test_path = paper_dataset_paths(ROOT / "datasets/paper", n, seeds, sizes)
    with test_path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["test_set"], payload.get("metadata", {})


def compute_solo_rate(state):
    """Compute solo (interference-free) rate for each link."""
    tx, rx = state.links[:, 0], state.links[:, 1]
    signal_power = state.signal_power
    g_diag = signal_power[tx, rx]
    noise = state.noise_power
    snr_gap = getattr(state, 'snr_gap', 1.0)
    solo_rate = torch.log2(1.0 + g_diag / noise / snr_gap)
    return solo_rate


def compute_direct_rate_priority(pi, state):
    """Priority = pi * C_solo (new code style)."""
    solo_rate = compute_solo_rate(state)
    return pi.reshape(-1).float().cpu() * solo_rate.float().cpu()


def compute_snr_based_priority(state):
    """Priority = SNR_dB (for reference)."""
    return state.vertex_features[:, 0].cpu().float()


def decode_greedy(pi, state, priorities):
    """Greedy projection (descending priority order)."""
    binary = (pi >= 0.5).float().cpu()
    action, metadata = project_shared_uav_feasible(
        binary, state.links, priorities=priorities
    )
    return action, metadata


def decode_endpoint_mutual(pi, state, priorities, max_rounds=1):
    """Endpoint mutual matching decoder."""
    binary = (pi >= 0.5).float().cpu()
    action, metadata = decode_endpoint_mutual_matching(
        binary, state.links, priorities=priorities, max_rounds=max_rounds
    )
    return action, metadata


def analyze_topology(state):
    """Analyze the topology of a single state."""
    links = state.links
    num_links = links.size(0)
    num_uavs = links.max().item() + 1

    # Count unique UAVs involved in links
    unique_uavs = set(links[:, 0].tolist() + links[:, 1].tolist())

    # Conflict graph analysis
    edge_index = state.edge_index
    if edge_index.numel() > 0:
        num_conflict_edges = edge_index.size(1)
        # Convert to undirected edges for density calculation
        src, dst = edge_index[0].numpy(), edge_index[1].numpy()
        undirected_edges = set()
        for s, d in zip(src, dst):
            undirected_edges.add((min(s, d), max(s, d)))
        num_undirected_conflicts = len(undirected_edges)
        max_possible = num_links * (num_links - 1) // 2
        conflict_density = num_undirected_conflicts / max_possible if max_possible > 0 else 0
    else:
        num_conflict_edges = 0
        num_undirected_conflicts = 0
        conflict_density = 0.0

    # Endpoint sharing analysis
    endpoint_counts = defaultdict(int)
    for tx, rx in links:
        endpoint_counts[int(tx)] += 1
        endpoint_counts[int(rx)] += 1
    max_links_per_uav = max(endpoint_counts.values()) if endpoint_counts else 0
    avg_links_per_uav = np.mean(list(endpoint_counts.values())) if endpoint_counts else 0

    return {
        "num_uavs": num_uavs,
        "num_links": num_links,
        "unique_uavs_in_links": len(unique_uavs),
        "num_conflict_edges": int(num_conflict_edges),
        "num_undirected_conflicts": int(num_undirected_conflicts),
        "conflict_density": float(conflict_density),
        "max_links_per_uav": int(max_links_per_uav),
        "avg_links_per_uav": float(avg_links_per_uav),
    }


def evaluate_sample(
    model, state, priority_mode="pi_c_solo", decoder="greedy"
):
    """
    Evaluate a single sample with specified priority and decoder.

    priority_mode: "pi_only" | "pi_c_solo" | "snr"
    decoder: "greedy" | "endpoint_mutual"
    """
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        # Get GAT output
        pi, _ = model(
            state.vertex_features[:, :1].to(device),
            state.edge_index.to(device),
            state.edge_weights.to(device),
        )
        pi_cpu = pi.reshape(-1).float().cpu()

        # Compute priorities based on mode
        if priority_mode == "pi_c_solo":
            priorities = compute_direct_rate_priority(pi, state)
        elif priority_mode == "pi_only":
            priorities = pi_cpu
        elif priority_mode == "snr":
            priorities = compute_snr_based_priority(state)
        else:
            raise ValueError(f"Unknown priority_mode: {priority_mode}")

        # Decode based on decoder
        if decoder == "greedy":
            action, proj_meta = decode_greedy(pi, state, priorities)
        elif decoder == "endpoint_mutual":
            action, proj_meta = decode_endpoint_mutual(pi, state, priorities)
        else:
            raise ValueError(f"Unknown decoder: {decoder}")

        # Compute metrics
        solo_rate = compute_solo_rate(state)
        audit = shared_uav_action_audit(action, state.links)

        # Get decoder-specific metadata
        if decoder == "endpoint_mutual":
            rejected = len(proj_meta.get("unresolved_candidate_link_indices", []))
        else:
            rejected = len(proj_meta.get("rejected_project_link_indices", []))

        rate = calculate_achievable_rate(
            action.reshape(-1, 1), state, binary=True, detach_hard=True
        ).sum().item()

        return {
            "pi_raw": pi_cpu.tolist(),
            "pi_mean": float(pi_cpu.mean()),
            "pi_std": float(pi_cpu.std()),
            "pi_above_05": int((pi_cpu > 0.5).sum()),
            "pi_above_03": int((pi_cpu > 0.3).sum()),
            "pi_above_07": int((pi_cpu > 0.7).sum()),
            "solo_rate_mean": float(solo_rate.mean()),
            "solo_rate_std": float(solo_rate.std()),
            "priorities_mean": float(priorities.mean()) if priorities.numel() > 0 else 0,
            "priorities_std": float(priorities.std()) if priorities.numel() > 0 else 0,
            "priorities_max": float(priorities.max()) if priorities.numel() > 0 else 0,
            "candidate_links": proj_meta["candidate"]["active_links"],
            "projected_links": proj_meta["projected"]["active_links"],
            "rejected_count": rejected,
            "active_links": int(audit["active_links"]),
            "feasible": audit["feasible"],
            "rate_bit_s_hz": rate,
            "rate_mbps": rate * 5.0,  # 5 MHz bandwidth
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--output-dir", default="paper_figures/fig06_cross_n/investigation")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit samples per test set (for quick testing)")
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[investigation] device={device}, train_size={args.train_size}", flush=True)

    # Load test sets
    test_sets = {}
    for n in [20, 40]:
        print(f"Loading test set N={n}...", flush=True)
        states, meta = load_test_set(n, args.train_size)
        if args.num_samples:
            states = states[:args.num_samples]
        test_sets[n] = states
        print(f"  N={n}: {len(states)} samples")

    # Load models
    models = {}
    for n in [20, 40]:
        print(f"Loading model trained on N={n}...", flush=True)
        try:
            model, ckpt_path = load_channel_gat(cfg(n), n, device=device, train_size=args.train_size)
            models[n] = model
            print(f"  N={n} model loaded from {ckpt_path}")
        except FileNotFoundError as e:
            print(f"  WARNING: Could not load N={n} model: {e}")
            models[n] = None

    # Define experiment grid
    experiments = [
        # (train_n, test_n, priority_mode, decoder)
        # N=40 model on N=20
        ("40→20 baseline", 40, 20, "pi_c_solo", "greedy"),
        ("40→20 pi_only", 40, 20, "pi_only", "greedy"),
        ("40→20 endpoint", 40, 20, "pi_c_solo", "endpoint_mutual"),
        ("40→20 both", 40, 20, "pi_only", "endpoint_mutual"),
        # N=20 model on N=20 (baseline)
        ("20→20 baseline", 20, 20, "pi_c_solo", "greedy"),
        ("20→20 pi_only", 20, 20, "pi_only", "greedy"),
        ("20→20 endpoint", 20, 20, "pi_c_solo", "endpoint_mutual"),
    ]

    results = {}
    for exp_name, train_n, test_n, priority_mode, decoder in experiments:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp_name}")
        print(f"  Priority mode: {priority_mode}")
        print(f"  Decoder: {decoder}")
        print(f"  Test samples: {len(test_sets[test_n])}")
        print(f"{'='*60}")

        if models.get(train_n) is None:
            print(f"  SKIPPED: Model N={train_n} not found")
            continue

        model = models[train_n]

        # Collect per-sample results
        sample_results = []
        topology_stats = []

        for i, state in enumerate(test_sets[test_n]):
            if i % 20 == 0:
                print(f"  Processing sample {i}/{len(test_sets[test_n])}...", flush=True)

            # Analyze topology once
            if i == 0:
                topo = analyze_topology(state)
                topology_stats.append(topo)

            result = evaluate_sample(model, state, priority_mode, decoder)
            sample_results.append(result)

        # Aggregate statistics
        rates = [r["rate_mbps"] for r in sample_results]
        active_fracs = [r["active_links"] / r["candidate_links"] if r["candidate_links"] > 0 else 0
                        for r in sample_results]
        rejected_rates = [r["rejected_count"] / r["candidate_links"] if r["candidate_links"] > 0 else 0
                          for r in sample_results]

        # π distribution stats
        pi_means = [r["pi_mean"] for r in sample_results]
        pi_stds = [r["pi_std"] for r in sample_results]
        pi_above_05 = [r["pi_above_05"] for r in sample_results]

        exp_result = {
            "train_n": train_n,
            "test_n": test_n,
            "priority_mode": priority_mode,
            "decoder": decoder,
            "num_samples": len(sample_results),
            "topology": topology_stats[0] if topology_stats else {},
            "throughput": {
                "mean_mbps": float(np.mean(rates)),
                "std_mbps": float(np.std(rates, ddof=1)),
                "sem_mbps": float(np.std(rates, ddof=1) / np.sqrt(len(rates))),
                "min_mbps": float(np.min(rates)),
                "max_mbps": float(np.max(rates)),
            },
            "pi_distribution": {
                "mean": float(np.mean(pi_means)),
                "std": float(np.mean(pi_stds)),
                "mean_above_05": float(np.mean(pi_above_05)),
                "samples": sample_results[:5],  # First 5 samples for detailed view
            },
            "active_fraction": {
                "mean": float(np.mean(active_fracs)),
                "std": float(np.std(active_fracs, ddof=1)),
            },
            "rejection_rate": {
                "mean": float(np.mean(rejected_rates)),
                "std": float(np.std(rejected_rates, ddof=1)),
            },
        }

        results[exp_name] = exp_result

        # Print summary
        print(f"\n  Results:")
        print(f"    Throughput: {exp_result['throughput']['mean_mbps']:.3f} ± "
              f"{exp_result['throughput']['sem_mbps']:.3f} Mbps")
        print(f"    π mean: {exp_result['pi_distribution']['mean']:.4f}")
        print(f"    Active fraction: {exp_result['active_fraction']['mean']:.3f}")
        print(f"    Rejection rate: {exp_result['rejection_rate']['mean']:.3f}")
        if topology_stats:
            print(f"    Topology: {topology_stats[0]}")

    # Detailed comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)

    # Create comparison table
    print("\n{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Experiment", "Mbps", "π mean", "Active%", "Reject%", "Links"
    ))
    print("-"*75)

    for name, res in results.items():
        topo = res.get("topology", {})
        print("{:<25} {:>10.3f} {:>10.4f} {:>10.1f} {:>10.1f} {:>10}".format(
            name,
            res["throughput"]["mean_mbps"],
            res["pi_distribution"]["mean"],
            res["active_fraction"]["mean"] * 100,
            res["rejection_rate"]["mean"] * 100,
            topo.get("num_links", "?"),
        ))

    # Specific analysis
    print("\n" + "="*80)
    print("HYPOTHESIS ANALYSIS")
    print("="*80)

    # Hypothesis 1: Priority mode effect
    if "40→20 baseline" in results and "40→20 pi_only" in results:
        baseline = results["40→20 baseline"]["throughput"]["mean_mbps"]
        pi_only = results["40→20 pi_only"]["throughput"]["mean_mbps"]
        diff = pi_only - baseline
        print(f"\n1. Priority Mode Effect (N=40 model on N=20):")
        print(f"   pi*C_solo (baseline): {baseline:.3f} Mbps")
        print(f"   pi only:               {pi_only:.3f} Mbps")
        print(f"   Difference: {diff:+.3f} Mbps ({diff/baseline*100:+.1f}%)")
        print(f"   → {'pi_only helps' if diff > 0 else 'pi*C_solo is better'}")

    # Hypothesis 2: Decoder effect
    if "40→20 baseline" in results and "40→20 endpoint" in results:
        baseline = results["40→20 baseline"]["throughput"]["mean_mbps"]
        endpoint = results["40→20 endpoint"]["throughput"]["mean_mbps"]
        diff = endpoint - baseline
        print(f"\n2. Decoder Effect (N=40 model on N=20):")
        print(f"   Greedy:               {baseline:.3f} Mbps")
        print(f"   Endpoint mutual:       {endpoint:.3f} Mbps")
        print(f"   Difference: {diff:+.3f} Mbps ({diff/baseline*100:+.1f}%)")
        print(f"   → {'Endpoint mutual helps' if diff > 0 else 'Greedy is better'}")

    # Hypothesis 3: Model generalization
    if "40→20 baseline" in results and "20→20 baseline" in results:
        n40_on_n20 = results["40→20 baseline"]["throughput"]["mean_mbps"]
        n20_on_n20 = results["20→20 baseline"]["throughput"]["mean_mbps"]
        diff = n40_on_n20 - n20_on_n20
        ratio = n40_on_n20 / n20_on_n20 if n20_on_n20 > 0 else 0
        print(f"\n3. Generalization Analysis:")
        print(f"   N=40 model on N=20: {n40_on_n20:.3f} Mbps")
        print(f"   N=20 model on N=20: {n20_on_n20:.3f} Mbps")
        print(f"   Gap: {diff:+.3f} Mbps ({ratio*100:.1f}% relative to N=20 model)")

        # π distribution comparison
        pi_n40 = results["40→20 baseline"]["pi_distribution"]["mean"]
        pi_n20 = results["20→20 baseline"]["pi_distribution"]["mean"]
        print(f"\n   π distribution:")
        print(f"   N=40 model π mean: {pi_n40:.4f}")
        print(f"   N=20 model π mean: {pi_n20:.4f}")
        print(f"   → {'Models produce similar π' if abs(pi_n40 - pi_n20) < 0.1 else 'π distributions differ significantly'}")

    # Topology comparison
    if "20→20 baseline" in results and "40→20 baseline" in results:
        topo_20 = results["20→20 baseline"].get("topology", {})
        topo_40 = results["40→20 baseline"].get("topology", {})
        print(f"\n4. Topology Comparison:")
        print(f"   N=20 topology:")
        print(f"     - Links: {topo_20.get('num_links', '?')}")
        print(f"     - Conflict density: {topo_20.get('conflict_density', 0):.3f}")
        print(f"     - Max links/UAV: {topo_20.get('max_links_per_uav', '?')}")
        print(f"   N=40 topology:")
        print(f"     - Links: {topo_40.get('num_links', '?')}")
        print(f"     - Conflict density: {topo_40.get('conflict_density', 0):.3f}")
        print(f"     - Max links/UAV: {topo_40.get('max_links_per_uav', '?')}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"investigation_trainSize={args.train_size}.json"

    # Convert non-serializable types
    serializable_results = {}
    for name, res in results.items():
        serializable_results[name] = {
            k: v for k, v in res.items() if k != "pi_distribution" or True
        }

    with output_file.open("w") as f:
        json.dump({
            "schema": "investigation_cross_n_v1",
            "train_size": args.train_size,
            "device": device,
            "experiments": results,
        }, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80)

    # Print first few sample details for debugging
    if "40→20 baseline" in results:
        print("\n" + "="*80)
        print("DETAILED SAMPLE ANALYSIS (N=40 model on N=20)")
        print("="*80)
        samples = results["40→20 baseline"]["pi_distribution"]["samples"]
        for i, s in enumerate(samples[:3]):
            print(f"\nSample {i}:")
            print(f"  π values: {[f'{p:.3f}' for p in s['pi_raw'][:10]]}...")
            print(f"  π mean: {s['pi_mean']:.4f}, std: {s['pi_std']:.4f}")
            print(f"  Active (candidate→projected): {s['candidate_links']}→{s['projected_links']}")
            print(f"  Rate: {s['rate_mbps']:.3f} Mbps")


if __name__ == "__main__":
    main()
