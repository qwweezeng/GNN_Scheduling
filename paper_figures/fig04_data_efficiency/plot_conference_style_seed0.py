#!/usr/bin/env python3
"""Render a one-seed Fig. 4 diagnostic using the conference raw test utility.

Each point evaluates the restored best-hard-validation checkpoint on the same
128 held-out topologies. Actions are ``round(pi)`` with no endpoint projection.
The resulting actions may violate the half-duplex shared-UAV constraint, so this
is a diagnostic rather than a deployable scheduling result.
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
    rows: list[dict[str, int | float]] = []
    for num_uav in nodes:
        for train_size in sizes:
            artifact = (
                HERE / "data/raw" / f"N{num_uav}" / f"n{train_size}"
                / f"seed{args.model_seed}" / "no_projection_evaluation.json"
            )
            report = json.loads(artifact.read_text())
            if report.get("num_test_samples") != 128:
                raise ValueError(f"expected 128 fixed test topologies: {artifact}")
            if report.get("decision_rule") != "direct_threshold_pi_ge_0.5_without_endpoint_projection":
                raise ValueError(f"unexpected decision rule: {artifact}")
            rows.append({
                "num_uav": num_uav,
                "training_samples": train_size,
                "model_seed": args.model_seed,
                "fixed_test_topologies": 128,
                "conference_raw_test_utility_mbps": float(
                    report["test_unprojected_threshold_utility_mbps"]
                ),
            })

    diagnostics = HERE / "data/diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / "fig04_conference_style_seed0_test.csv"
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
                [row["conference_raw_test_utility_mbps"] for row in curve],
                "o-",
                ms=3,
                lw=1.0,
                color=NODE_COLORS[num_uav],
                markerfacecolor=NODE_COLORS[num_uav],
                label=f"N={num_uav}",
            )
        axis.set_xlabel("Number of training samples")
        axis.set_ylabel("Raw test utility (Mbit/s)")
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
        stem = "fig04_conference_style_seed0_test"
        figure.savefig(figures / f"{stem}.eps", format="eps")
        figure.savefig(figures / f"{stem}.jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.close(figure)


if __name__ == "__main__":
    main()
