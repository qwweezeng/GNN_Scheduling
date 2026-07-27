#!/usr/bin/env python3
"""Replicate the conference sample-efficiency plotting convention.

The conference plot uses each run's maximum raw validation utility, omits the
same indexed points as the original plotting function, and converts b/s/Hz to
Mbit/s by multiplying by the 5 MHz bandwidth divided by 1e6. It does not use a
held-out test set or endpoint matching projection.
"""
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

CONFERENCE_POINT_INDICES = (0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 13)
MBPS_PER_BPS_HZ = 5.0


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--sizes", default="10,20,30,40,50,60,80,100,200,300")
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()

    nodes = parse_ints(args.nodes)
    sizes = parse_ints(args.sizes)
    selected_indices = [index for index in CONFERENCE_POINT_INDICES if index < len(sizes)]
    selected_sizes = [sizes[index] for index in selected_indices]
    rows: list[dict[str, int | float]] = []

    for num_uav in nodes:
        for index, train_size in zip(selected_indices, selected_sizes):
            artifact = (
                HERE / "data/raw" / f"N{num_uav}" / f"n{train_size}"
                / f"seed{args.model_seed}" / "training.json"
            )
            report = json.loads(artifact.read_text())
            best_validation = float(report["best_val_utility"])
            rows.append({
                "num_uav": num_uav,
                "training_samples": train_size,
                "original_sorted_point_index": index,
                "model_seed": args.model_seed,
                "conference_best_val_utility_bps_hz": best_validation,
                "conference_plotted_utility_mbps": best_validation * MBPS_PER_BPS_HZ,
            })

    diagnostics = HERE / "data/diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / "fig04_conference_exact_seed0.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with paper_style():
        mpl.rcParams.update({
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        })
        figure, axis = plt.subplots(figsize=(3.5, 2.35))
        for num_uav in nodes:
            curve = [row for row in rows if row["num_uav"] == num_uav]
            axis.plot(
                [row["training_samples"] for row in curve],
                [row["conference_plotted_utility_mbps"] for row in curve],
                "o-",
                ms=3,
                lw=1.0,
                color=NODE_COLORS[num_uav],
                markerfacecolor=NODE_COLORS[num_uav],
                label=f"N={num_uav}",
            )
        axis.set_xlabel("Training sample size")
        axis.set_ylabel("Best validation utility (Mbit/s)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", linestyle="--", linewidth=0.5)
        axis.get_xaxis().get_major_formatter().set_scientific(False)
        axis.get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(
                lambda value, _: f"{int(value)}" if value == int(value) else f"{value}"
            )
        )
        axis.legend(ncol=2, frameon=True, fontsize=8)
        figures = HERE / "figures"
        figures.mkdir(exist_ok=True)
        stem = "fig04_conference_exact_seed0"
        figure.savefig(figures / f"{stem}.eps", format="eps")
        figure.savefig(figures / f"{stem}.jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
