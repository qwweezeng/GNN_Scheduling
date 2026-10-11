#!/usr/bin/env python3
"""Render Fig. 6 train-scale curves from an existing cross-N result file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_figures.common.plot_style import NODE_COLORS, paper_style


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "data/raw/cross_n_trainSize=300_seed0_epoch50.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=HERE,
        help="Root containing figures/ and data/plotting/ for this rendering.",
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    nodes = [int(value) for value in payload["nodes"]]
    train_scopes = [str(node) for node in nodes]
    if "mixed" in payload["results"]:
        train_scopes.append("mixed")

    rows = []
    for train_scope in train_scopes:
        for test_n in nodes:
            result = payload["results"][train_scope][str(test_n)]
            rows.append({
                "training_num_uav": train_scope,
                "test_num_uav": test_n,
                "average_sum_rate_mbps": result["mean_aggregate_phy_rate_mbps"],
            })

    plotting_dir = args.output_root / "data/plotting"
    plotting_dir.mkdir(parents=True, exist_ok=True)
    csv_path = plotting_dir / f"fig06_cross_n_{args.input.stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with paper_style():
        mpl.rcParams.update({
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        for train_scope in train_scopes:
            selected = [
                row for row in rows
                if row["training_num_uav"] == train_scope
            ]
            x = [row["test_num_uav"] for row in selected]
            y = [row["average_sum_rate_mbps"] for row in selected]
            if train_scope == "mixed":
                ax.plot(
                    x, y, marker="D", ms=3, linestyle="--",
                    color="#333333", label="Train mixed N",
                )
            else:
                ax.plot(
                    x, y, marker="o", ms=3,
                    color=NODE_COLORS[int(train_scope)],
                    label=f"Train N={train_scope}",
                )

        ax.set(
            xlabel="Test number of UAVs",
            ylabel="Average sum rate (Mbit/s)",
        )
        ax.set_xticks(nodes)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", linestyle="--", linewidth=0.5)
        ax.legend(ncol=2, fontsize=6)

        figure_dir = args.output_root / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        output = figure_dir / "fig06_cross_network_generalization.eps"
        fig.savefig(output, format="eps")
        fig.savefig(
            output.with_suffix(".jpg"), format="jpg", dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
