#!/usr/bin/env python3
"""Render paper figures from metadata-checked experiment JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISPLAY = {
    "direct_link_marginal_mp": "MarginalMP",
    "direct_link_join": "Join",
    "direct_link_localmax": "LocalMax",
    "hard_conflict_lgs": "Hard-Conflict LGS",
    "centralized_greedy": "Centralized Greedy",
    "quantized_4bit": "4-bit large-scale",
    "large_scale": "Large-scale",
    "latest": "Latest",
    "stale": "One-block stale",
    "cg_csi": "CG-CSI (optional)",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def full_buffer(inputs, output_dir):
    payloads = sorted((_read(path) for path in inputs), key=lambda p: p["num_uav"])
    methods = sorted(set.intersection(*[
        {key for key, value in p["results"].items() if isinstance(value, dict)}
        for p in payloads
    ]))
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = [p["num_uav"] for p in payloads]
    for method in methods:
        y = [p["results"][method]["mean_sum_spectral_efficiency"] for p in payloads]
        error = [p["results"][method]["standard_error"] for p in payloads]
        ax.errorbar(x, y, yerr=error, marker="o", capsize=3, label=method)
    ax.set(xlabel="Number of UAVs", ylabel="Sum spectral efficiency (bit/s/Hz)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    _save(fig, output_dir, "fig1_full_buffer_scaling")


def training(inputs, output_dir):
    payload = _read(inputs[0])
    runs = sorted(payload["runs"], key=lambda row: row["train_size"])
    largest = runs[-1]
    epochs = [row["epoch"] + 1 for row in largest["per_epoch"]]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    axes = axes.reshape(-1)
    axes[0].plot(epochs, [row["train_loss"] for row in largest["per_epoch"]])
    axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].plot(
        epochs,
        [row["val_utility"] for row in largest["per_epoch"]],
        marker=".",
    )
    axes[1].set(xlabel="Epoch", ylabel="Validation utility")
    axes[2].plot(
        epochs, [row["pi_mean"] for row in largest["per_epoch"]],
        label="mean"
    )
    axes[2].fill_between(
        epochs,
        [max(0.0, row["pi_mean"] - row["pi_std"]) for row in largest["per_epoch"]],
        [min(1.0, row["pi_mean"] + row["pi_std"]) for row in largest["per_epoch"]],
        alpha=0.2,
        label="mean ± std",
    )
    axes[2].set(xlabel="Epoch", ylabel="Continuous scheduling probability")
    axes[2].legend(fontsize=8)
    axes[3].plot(
        [row["train_size"] for row in runs],
        [row["test_utility"] for row in runs], marker="o"
    )
    axes[3].set(xlabel="Training samples", ylabel="Test utility")
    for ax in axes:
        ax.grid(alpha=0.25)
    _save(fig, output_dir, "fig2_channel_gat_training")


def scaling(inputs, output_dir):
    payload = _read(inputs[0])
    rows = sorted(payload["relative_to_test_scale_dedicated_model"], key=int)
    columns = sorted({
        key for row in payload["relative_to_test_scale_dedicated_model"].values()
        for key in row
    }, key=int)
    matrix = np.asarray([
        [payload["relative_to_test_scale_dedicated_model"][row].get(col, np.nan)
         for col in columns]
        for row in rows
    ])
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    image = ax.imshow(matrix * 100.0, vmin=70, vmax=110, cmap="viridis")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j] * 100:.1f}", ha="center", va="center", color="white")
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.set(xlabel="Test UAV count", ylabel="Training UAV count")
    fig.colorbar(image, ax=ax, label="Relative utility (%)")
    _save(fig, output_dir, "fig3_channel_gat_generalization")


def csi(inputs, output_dir):
    payload = _read(inputs[0])
    preferred = [
        "perfect", "large_scale", "quantized_2bit", "quantized_3bit",
        "quantized_4bit", "quantized_6bit", "quantized_8bit",
    ]
    labels = [name for name in preferred if name in payload["results"]]
    values = [payload["results"][name]["mean_true_channel_utility"] for name in labels]
    errors = [payload["results"][name]["standard_error"] for name in labels]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.bar(range(len(labels)), values, yerr=errors, capsize=3)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("True-channel sum spectral efficiency")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir, "fig4_channel_gat_csi")


def snapshot(inputs, output_dir):
    payloads = [_read(path) for path in inputs]
    payload = next(
        (item for item in payloads if int(item["settings"]["marginal_mp_rounds"]) == 4),
        payloads[0],
    )
    names, ratios = [], []
    for method, summary in payload["overall"].items():
        names.append(payload.get("method_display_names", {}).get(method, DISPLAY.get(method, method)))
        ratios.append(summary["mean_ratio_vs_direct_link_join"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.bar(range(len(names)), np.asarray(ratios) * 100.0)
    ax.axhline(100.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel("Utility relative to Direct-Link Join (%)")
    ax.grid(axis="y", alpha=0.25)

    by_round = {
        int(item["settings"]["marginal_mp_rounds"]): item
        for item in payloads
    }
    rounds = [0] + sorted(by_round)
    utility = [
        payload["overall"]["direct_link_localmax"][
            "mean_ratio_vs_direct_link_join"
        ]
    ] + [
        by_round[value]["overall"]["direct_link_marginal_mp"][
            "mean_ratio_vs_direct_link_join"
        ] for value in sorted(by_round)
    ]
    localmax_messages = float(np.mean([
        row["num_directed_conflict_edges"] for row in payload["records"]
    ]))
    message_counts = [localmax_messages]
    for value in sorted(by_round):
        rows = by_round[value]["records"]
        message_counts.append(float(np.mean([
            row["num_directed_conflict_edges"]
            + sum(row["marginal_mp_physical_externality_pairs_per_round"])
            + sum(row["marginal_mp_arbitration_edge_messages_per_round"])
            + row["num_links"] * len(
                row["marginal_mp_physical_externality_pairs_per_round"]
            )
            for row in rows
        ])))
    axes[1].plot(rounds, np.asarray(utility) * 100.0, marker="o")
    axes[1].set(
        xlabel="MarginalMP refinement rounds K",
        ylabel="Utility relative to Direct-Link Join (%)",
    )
    axes[1].grid(alpha=0.25)
    message_axis = axes[1].twinx()
    message_axis.plot(
        rounds, message_counts, marker="s", linestyle="--", color="tab:red"
    )
    message_axis.set_ylabel("Mean scalar-equivalent messages", color="tab:red")
    _save(fig, output_dir, "fig5_fixed_csi_snapshot")


def mobile(inputs, output_dir, arrival_rate, block, age):
    payload = _read(inputs[0])
    records = [row for row in payload["records"] if
               abs(float(row["arrival_rate"]) - arrival_rate) < 1e-9
               and int(row["slots_per_block"]) == block
               and int(row["stale_blocks"]) == age]
    methods = sorted({row["method"] for row in records})
    speeds = sorted({float(row["speed_mps"]) for row in records})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for method in methods:
        goodput, backlog = [], []
        for speed in speeds:
            cell = [row for row in records if row["method"] == method and float(row["speed_mps"]) == speed]
            goodput.append(np.mean([row["delivered_throughput_mbps"] for row in cell]))
            backlog.append(np.mean([row["average_total_queue_backlog_mbit"] for row in cell]))
        label = DISPLAY.get(method, method)
        axes[0].plot(speeds, goodput, marker="o", label=label)
        axes[1].plot(speeds, backlog, marker="o", label=label)
    axes[0].set(xlabel="UAV speed (m/s)", ylabel="Delivered goodput (Mbps)")
    axes[1].set(xlabel="UAV speed (m/s)", ylabel="Average total backlog (Mbit)")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    _save(fig, output_dir, "fig7_mobile_imperfect_csi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure",
        required=True,
        choices=("full_buffer", "training", "scaling", "csi", "snapshot", "mobile"),
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/paper/figures")
    parser.add_argument("--arrival-rate", type=float, default=2.2)
    parser.add_argument("--slots-per-block", type=int, default=5)
    parser.add_argument("--stale-blocks", type=int, default=1)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.figure == "full_buffer":
        full_buffer(args.inputs, output_dir)
    elif args.figure == "training":
        training(args.inputs, output_dir)
    elif args.figure == "scaling":
        scaling(args.inputs, output_dir)
    elif args.figure == "csi":
        csi(args.inputs, output_dir)
    elif args.figure == "snapshot":
        snapshot(args.inputs, output_dir)
    else:
        mobile(
            args.inputs, output_dir, args.arrival_rate,
            args.slots_per_block, args.stale_blocks,
        )


if __name__ == "__main__":
    main()
