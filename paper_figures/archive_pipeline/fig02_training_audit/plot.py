#!/usr/bin/env python3
"""Build TNSE Fig. 2 exclusively from saved JSON training/evaluation records."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "gnn-scheduling-mpl")
)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
RAW_DIR = FIGURE_DIR / "data/raw"
PLOTTING_DIR = FIGURE_DIR / "data/plotting"
FIGURES_DIR = FIGURE_DIR / "figures"
sys.path.insert(0, str(REPO_ROOT))

from paper_figures.common.plot_style import NODE_COLORS, paper_style  # noqa: E402

PUBLICATION_SAFE_FONT_FAMILY = "Arial"
VECTOR_TEXT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}
FIGURE_FONT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}
IEEE_SINGLE_WIDTH_MM = 88.9
IEEE_DOUBLE_WIDTH_MM = 181.9
fig_width_mm = IEEE_DOUBLE_WIDTH_MM
PHY_RATE_FACTOR = 5.0  # 5 MHz * bit/s/Hz -> Mbit/s


def _production_report(num_uav: int) -> Path:
    return REPO_ROOT / "results/paper/channel_gat_training" / (
        f"rician_channel_gat_N{num_uav}_K5.json"
    )


def _training_report(num_uav: int, train_size: int) -> Path:
    return (
        _production_report(num_uav)
        if train_size == 500
        else RAW_DIR / f"training_N{num_uav}_n{train_size}.json"
    )


def _evaluation_report(num_uav: int, train_size: int) -> Path:
    return RAW_DIR / f"evaluation_N{num_uav}_n{train_size}.json"


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing Fig. 2 source {path}; run pipeline.py train/evaluate")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_tables(nodes: list[int], train_sizes: list[int]) -> dict[str, list[dict]]:
    epochs, sample_efficiency, collapse = [], [], []
    for num_uav in nodes:
        full = _load_json(_training_report(num_uav, 500))
        for row in full["per_epoch_logs"]:
            epochs.append({
                "num_uav": num_uav,
                "epoch": int(row["epoch"]),
                "train_surrogate_sum_spectral_efficiency_bit_s_hz": float(row["utility"]),
                "train_surrogate_phy_rate_mbps": float(row["utility"]) * PHY_RATE_FACTOR,
                "validation_binary_sum_spectral_efficiency_bit_s_hz": float(row["val_utility"]),
                "validation_binary_phy_rate_mbps": float(row["val_utility"]) * PHY_RATE_FACTOR,
                "training_pi_mean": float(row["pi_mean"]),
                "training_pi_std": float(row["pi_std"]),
                "learning_rate": float(row["learning_rate"]),
                "best_epoch": int(full["best_epoch"]),
                "dataset_fingerprint": full["dataset_fingerprint"],
            })
        audit = full["output_audit"]
        collapse.append({
            "num_uav": num_uav,
            "probability_mean": float(audit["probability_mean"]),
            "probability_std": float(audit["probability_std"]),
            "raw_binary_active_fraction": float(audit["mean_binary_active_fraction"]),
            "all_active_snapshot_fraction": float(audit["all_links_active_state_fraction"]),
            "all_inactive_snapshot_fraction": float(audit["all_links_inactive_state_fraction"]),
            "integrity_passed": bool(audit["integrity_passed"]),
            "dataset_fingerprint": full["dataset_fingerprint"],
        })
        for train_size in train_sizes:
            training = _load_json(_training_report(num_uav, train_size))
            evaluation = _load_json(_evaluation_report(num_uav, train_size))
            sample_efficiency.append({
                "num_uav": num_uav,
                "train_size": train_size,
                "train_pool_size": 500,
                "subset_rule": "first_n_prefix_of_canonical_training_pool",
                "best_epoch": int(training["best_epoch"]),
                "predecoder_test_sum_spectral_efficiency_bit_s_hz": float(training["test_utility"]),
                "predecoder_test_phy_rate_mbps": float(training["test_utility"]) * PHY_RATE_FACTOR,
                "final_endpoint_decoded_test_phy_rate_mbps": float(evaluation["mean_aggregate_phy_rate_mbps"]),
                "final_endpoint_decoded_test_sem_mbps": float(evaluation["sem_aggregate_phy_rate_mbps"]),
                "final_active_fraction": float(evaluation["mean_final_active_fraction"]),
                "num_test_samples": int(evaluation["num_test_samples"]),
                "dataset_fingerprint": evaluation["dataset_fingerprint"],
                "checkpoint": evaluation["checkpoint"],
            })
    tables = {
        "epoch_metrics.csv": epochs,
        "sample_efficiency.csv": sample_efficiency,
        "collapse_audit.csv": collapse,
    }
    for filename, rows in tables.items():
        _write_csv(PLOTTING_DIR / filename, rows)
    return tables


def _panel_label(axis, label: str) -> None:
    axis.text(
        0.015, 0.985, f"({label})", transform=axis.transAxes,
        ha="left", va="top", fontweight="bold", fontsize=8,
    )


def _draw(axes, tables: dict[str, list[dict]], nodes: list[int], train_sizes: list[int]) -> None:
    epoch_rows = tables["epoch_metrics.csv"]
    sample_rows = tables["sample_efficiency.csv"]
    collapse_rows = tables["collapse_audit.csv"]

    for num_uav in nodes:
        rows = sorted(
            (row for row in epoch_rows if row["num_uav"] == num_uav),
            key=lambda row: row["epoch"],
        )
        x = [row["epoch"] + 1 for row in rows]
        color = NODE_COLORS[num_uav]
        axes[0].plot(
            x, [row["train_surrogate_phy_rate_mbps"] for row in rows],
            color=color, label=f"N={num_uav}",
        )
        axes[1].plot(
            x, [row["validation_binary_phy_rate_mbps"] for row in rows],
            color=color, label=f"N={num_uav}",
        )
        best = int(rows[0]["best_epoch"]) + 1
        best_row = rows[best - 1]
        axes[1].scatter(
            [best], [best_row["validation_binary_phy_rate_mbps"]],
            s=13, color=color, edgecolor="white", linewidth=0.4, zorder=5,
        )
        efficiency = sorted(
            (row for row in sample_rows if row["num_uav"] == num_uav),
            key=lambda row: row["train_size"],
        )
        axes[2].plot(
            [row["train_size"] for row in efficiency],
            [row["final_endpoint_decoded_test_phy_rate_mbps"] for row in efficiency],
            marker="o", markersize=3.4, color=color, label=f"N={num_uav}",
        )

    ordered = sorted(collapse_rows, key=lambda row: row["num_uav"])
    x_nodes = [row["num_uav"] for row in ordered]
    audit_styles = (
        ("Mean $\\pi$", "probability_mean", "#4F76B8", "o"),
        ("Std. $\\pi$", "probability_std", "#C98B4A", "s"),
        ("Binary active fraction", "raw_binary_active_fraction", "#C65A78", "D"),
    )
    for label, key, color, marker in audit_styles:
        axes[3].plot(
            x_nodes, [row[key] for row in ordered], color=color,
            marker=marker, markersize=3.4, label=label,
        )
    all_collapse_zero = all(
        row["all_active_snapshot_fraction"] == 0.0
        and row["all_inactive_snapshot_fraction"] == 0.0
        for row in ordered
    )
    if all_collapse_zero:
        axes[3].text(
            0.98, 0.42, "All-on/off snapshots: 0%",
            transform=axes[3].transAxes, ha="right", va="top", fontsize=7,
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training surrogate rate (Mbit/s)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation binary rate (Mbit/s)")
    axes[2].set_xlabel("Number of training samples")
    axes[2].set_ylabel("Endpoint-decoded test rate (Mbit/s)")
    axes[2].set_xticks(train_sizes)
    axes[3].set_xlabel("Number of UAVs")
    axes[3].set_ylabel("Output statistic")
    axes[3].set_xticks(nodes)
    axes[3].set_ylim(bottom=0.0)

    for label, axis in zip("abcd", axes):
        _panel_label(axis, label)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, ncol=2, loc="lower right")
    axes[3].legend(frameon=False, loc="upper right", fontsize=6.5)


def render(
    tables: dict[str, list[dict]], nodes: list[int], train_sizes: list[int], layout: str
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with paper_style():
        mpl.rcParams.update(VECTOR_TEXT_RC)
        # IEEE figure lettering is kept in a compact sans-serif face while the
        # manuscript body remains Times; this also survives SVG/PDF as text.
        mpl.rcParams.update(FIGURE_FONT_RC)
        if layout == "double":
            fig, grid = plt.subplots(2, 2, figsize=(7.16, 5.05), squeeze=False)
            axes = grid.ravel()
            fig.subplots_adjust(
                left=0.085, right=0.985, bottom=0.09, top=0.985,
                wspace=0.27, hspace=0.34,
            )
            stem = "fig02_training_audit_double_column"
        else:
            fig, grid = plt.subplots(4, 1, figsize=(3.50, 7.45), squeeze=False)
            axes = grid.ravel()
            fig.subplots_adjust(
                left=0.19, right=0.98, bottom=0.065, top=0.992, hspace=0.48,
            )
            stem = "fig02_training_audit_single_column"
        _draw(axes, tables, nodes, train_sizes)
        fig.savefig(FIGURES_DIR / f"{stem}.eps", format="eps")
        fig.savefig(FIGURES_DIR / f"{stem}.pdf", format="pdf")
        fig.savefig(FIGURES_DIR / f"{stem}.svg", format="svg")
        fig.savefig(FIGURES_DIR / f"{stem}.tiff", format="tiff", dpi=600)
        fig.savefig(FIGURES_DIR / f"{stem}_review.png", format="png", dpi=300)
        plt.close(fig)


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--train-sizes", default="50,100,200,300,500")
    parser.add_argument("--layout", choices=("single", "double", "both"), default="both")
    args = parser.parse_args()
    nodes = _csv_ints(args.nodes)
    train_sizes = _csv_ints(args.train_sizes)
    tables = build_tables(nodes, train_sizes)
    layouts = ("single", "double") if args.layout == "both" else (args.layout,)
    for layout in layouts:
        render(tables, nodes, train_sizes, layout)
    print(f"saved Fig. 2 tables and figures under {FIGURE_DIR}", flush=True)


if __name__ == "__main__":
    main()
