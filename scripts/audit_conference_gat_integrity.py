#!/usr/bin/env python3
"""Audit whether a checkpoint still implements the conference GAT scheduler.

The conference output ``tilde_pi`` is a soft scheduling probability.  Its
deployment action is ``pi = 1(tilde_pi >= 0.5)``.  This script deliberately
keeps that terminology and reports both quantities separately.

This is a read-only diagnostic: it does not train a model or modify a
checkpoint.  It compares the four paper-producing experiments with later
N=10 checkpoints that have been used by the queue/CSI experiments.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.model import GATScheduler, calculate_achievable_rate


@dataclass(frozen=True)
class AuditCase:
    name: str
    checkpoint: Path
    dataset: Path
    expected_test_utility: float | None = None


CASES = (
    AuditCase(
        "conference_N20",
        ROOT / "results/exp_20_20260130_2147/exp_20_20260130_2147_model.pth",
        ROOT / "datasets/dataset_numUAV=20.pkl",
        24.455303,
    ),
    AuditCase(
        "conference_N30",
        ROOT / "results/exp_30_20260130_2236/exp_30_20260130_2236_model.pth",
        ROOT / "datasets/dataset_numUAV=30.pkl",
        32.991520,
    ),
    AuditCase(
        "conference_N40",
        ROOT / "results/exp_40_20260131_0118/exp_40_20260131_0118_model.pth",
        ROOT / "datasets/dataset_numUAV=40.pkl",
        42.906655,
    ),
    AuditCase(
        "conference_N50",
        ROOT / "results/exp_50_20260131_1740/exp_50_20260131_1740_model.pth",
        ROOT / "datasets/dataset_numUAV=50.pkl",
        47.783648,
    ),
    AuditCase(
        "legacy_N10",
        ROOT / "results/models/gat_numUAV=10.pth",
        ROOT / "datasets/dataset_numUAV=10.pkl",
    ),
    AuditCase(
        "later_N10_rayleigh",
        ROOT / "results/models/gat_numUAV=10_tx=10dBm_csi=perfect.pth",
        ROOT / "datasets/dataset_numUAV=10_tx=10dBm_csi=perfect.pkl",
    ),
    AuditCase(
        "later_N10_rician_K5dB",
        ROOT / "results/models/gat_numUAV=10_tx=10dBm_csi=perfect_channel=ricianK5dB.pth",
        ROOT / "datasets/dataset_numUAV=10_tx=10dBm_csi=perfect_channel=ricianK5dB.pkl",
    ),
    AuditCase(
        "conference_N20_on_later_N10_rayleigh",
        ROOT / "results/exp_20_20260130_2147/exp_20_20260130_2147_model.pth",
        ROOT / "datasets/dataset_numUAV=10_tx=10dBm_csi=perfect.pkl",
    ),
    AuditCase(
        "conference_N20_on_later_N10_rician_K5dB",
        ROOT / "results/exp_20_20260130_2147/exp_20_20260130_2147_model.pth",
        ROOT / "datasets/dataset_numUAV=10_tx=10dBm_csi=perfect_channel=ricianK5dB.pkl",
    ),
    AuditCase(
        "current_rician_N20_native",
        ROOT / "results/models/gat_numUAV=20_tx=10dBm_csi=perfect_channel=ricianK5dB.pth",
        ROOT / "datasets/dataset_numUAV=20_tx=10dBm_csi=perfect_channel=ricianK5dB.pkl",
    ),
    AuditCase(
        "current_rician_N20_on_current_rician_N10",
        ROOT / "results/models/gat_numUAV=20_tx=10dBm_csi=perfect_channel=ricianK5dB.pth",
        ROOT / "datasets/dataset_numUAV=10_tx=10dBm_csi=perfect_channel=ricianK5dB.pkl",
    ),
)


def build_model() -> GATScheduler:
    return GATScheduler(
        raw_node_dim=1,
        mlp_hidden_dims=[32, 32],
        embed_node_dim=32,
        edge_dim=1,
        gat_out_dim=48,
        num_heads=4,
        T_iterations=4,
        schedule_hidden_dim=32,
    )


def load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        metadata = {
            key: value
            for key, value in payload.items()
            if key != "model_state_dict"
        }
        return payload["model_state_dict"], metadata
    return payload, {}


def load_test_set(path: Path) -> list[Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload["test_set"]


def paper_relaxed_rates(tilde_pi: torch.Tensor, state: Any) -> torch.Tensor:
    """Conference Eq. (12)-(13), before summing across links.

    The soft scheduling probability occurs twice in the relaxed objective:
    once as the probability of scheduling the link and once in the desired
    signal term of the relaxed SINR.  This distinction disappears after
    thresholding to a binary schedule but matters during training.
    """
    p = tilde_pi.reshape(-1)
    signal_power = state.signal_power.to(p.device)
    link_tx = state.links[:, 0]
    link_rx = state.links[:, 1]
    direct = signal_power[link_tx, link_rx]
    gain_matrix = signal_power[link_tx[None, :], link_rx[:, None]]
    gain_matrix = gain_matrix * (
        1.0 - torch.eye(state.num_vertices, device=p.device)
    )
    interference = gain_matrix @ p
    snr_gap = float(getattr(state, "snr_gap", 1.0))
    relaxed_sinr = p * direct / (interference + state.noise_power)
    return p * torch.log2(1.0 + relaxed_sinr / snr_gap)


def finite_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, dict):
            result[key] = finite_metadata(value)
    return result


def audit_case(case: AuditCase, max_states: int | None) -> dict[str, Any]:
    model = build_model()
    state_dict, metadata = load_state_dict(case.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    test_set = load_test_set(case.dataset)
    if max_states is not None:
        test_set = test_set[:max_states]

    probabilities: list[torch.Tensor] = []
    active_counts: list[int] = []
    link_counts: list[int] = []
    edge_counts: list[int] = []
    all_active_states = 0
    all_inactive_states = 0
    hard_utilities: list[float] = []
    all_active_utilities: list[float] = []
    repo_soft_utilities: list[float] = []
    paper_soft_utilities: list[float] = []

    with torch.no_grad():
        for state in test_set:
            logits, _ = model.schedule_logits(
                state.vertex_features[:, :1],
                state.edge_index,
                state.edge_weights,
            )
            tilde_pi = torch.sigmoid(logits)
            pi = (tilde_pi >= 0.5).to(tilde_pi.dtype)

            probabilities.append(tilde_pi.reshape(-1))
            num_links = int(state.num_vertices)
            num_active = int(pi.sum().item())
            link_counts.append(num_links)
            edge_counts.append(int(state.edge_index.shape[1]))
            active_counts.append(num_active)
            all_active_states += int(num_active == num_links)
            all_inactive_states += int(num_active == 0)

            hard_utilities.append(
                float(
                    calculate_achievable_rate(
                        pi, state, binary=False
                    ).sum().item()
                )
            )
            all_active_utilities.append(
                float(
                    calculate_achievable_rate(
                        torch.ones_like(pi), state, binary=False
                    ).sum().item()
                )
            )
            repo_soft_utilities.append(
                float(
                    calculate_achievable_rate(
                        tilde_pi, state, binary=False
                    ).sum().item()
                )
            )
            paper_soft_utilities.append(
                float(paper_relaxed_rates(tilde_pi, state).sum().item())
            )

    all_probabilities = torch.cat(probabilities)
    total_links = max(sum(link_counts), 1)
    num_states = max(len(test_set), 1)
    hard_utility = float(np.mean(hard_utilities))
    expected = case.expected_test_utility
    return {
        "name": case.name,
        "checkpoint": str(case.checkpoint.relative_to(ROOT)),
        "dataset": str(case.dataset.relative_to(ROOT)),
        "num_test_states": len(test_set),
        "checkpoint_metadata": finite_metadata(metadata),
        "state_dict_missing_keys": list(missing),
        "state_dict_unexpected_keys": list(unexpected),
        "mean_num_links": float(np.mean(link_counts)),
        "mean_num_conflict_edges": float(np.mean(edge_counts)),
        "mean_in_degree": float(np.mean(np.asarray(edge_counts) / np.asarray(link_counts))),
        "soft_pi": {
            "mean": float(all_probabilities.mean().item()),
            "std": float(all_probabilities.std().item()),
            "min": float(all_probabilities.min().item()),
            "median": float(all_probabilities.median().item()),
            "max": float(all_probabilities.max().item()),
        },
        "hard_pi_threshold": 0.5,
        "hard_active_fraction": float(sum(active_counts) / total_links),
        "mean_active_links_per_state": float(np.mean(active_counts)),
        "all_links_active_state_fraction": float(all_active_states / num_states),
        "all_links_inactive_state_fraction": float(all_inactive_states / num_states),
        "hard_test_utility": hard_utility,
        "all_active_test_utility": float(np.mean(all_active_utilities)),
        "repo_soft_test_utility": float(np.mean(repo_soft_utilities)),
        "paper_relaxed_test_utility": float(np.mean(paper_soft_utilities)),
        "expected_paper_test_utility": expected,
        "hard_utility_absolute_error_vs_paper_record": (
            abs(hard_utility - expected) if expected is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional case names; default audits every case.",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Optional quick-audit limit; default audits the full test set.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/diagnostics/conference_gat_integrity_audit_20260716.json",
    )
    args = parser.parse_args()

    selected = CASES
    if args.cases:
        requested = set(args.cases)
        selected = tuple(case for case in CASES if case.name in requested)
        missing = requested - {case.name for case in selected}
        if missing:
            raise ValueError(f"Unknown audit cases: {sorted(missing)}")
    results = [audit_case(case, args.max_states) for case in selected]
    payload = {
        "purpose": (
            "Verify that the loaded GAT produces the conference soft scheduling "
            "probability tilde_pi and a sparse thresholded binary schedule."
        ),
        "conference_action_definition": "pi_i = 1(tilde_pi_i >= 0.5)",
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved audit to {args.output}")


if __name__ == "__main__":
    main()
