#!/usr/bin/env python3
"""Render Fig. 4 projection and checkpoint-selection diagnostic panels."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_figures.common.plot_style import NODE_COLORS, paper_style


METRICS = {
    "projected_test": {
        "artifact": "evaluation.json",
        "field": "test_binary_rate_mbps",
        "title": "Projected feasible test rate",
        "ylabel": "Test binary rate (Mbit/s)",
        "figure": "fig04_projected_test_diagnostic",
    },
    "unprojected_test": {
        "artifact": "no_projection_evaluation.json",
        "field": "test_unprojected_threshold_utility_mbps",
        "title": "Unprojected threshold test utility (infeasible)",
        "ylabel": "Unprojected threshold utility (Mbit/s)",
        "figure": "fig04_unprojected_test_diagnostic",
    },
    "best_validation": {
        "artifact": "training.json",
        "field": "best_val_utility",
        "title": "Best validation threshold utility (unprojected)",
        "ylabel": "Best validation threshold utility (Mbit/s)",
        "figure": "fig04_best_validation_diagnostic",
    },
}


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing diagnostic input: {path}")
    return json.loads(path.read_text())


def aggregate_metric(
    *,
    name: str,
    nodes: list[int],
    sizes: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    spec = METRICS[name]
    rows: list[dict[str, Any]] = []
    for num_uav in nodes:
        for train_size in sizes:
            per_seed = []
            for model_seed in seeds:
                path = HERE / "data/raw" / f"N{num_uav}" / f"n{train_size}" / f"seed{model_seed}" / spec["artifact"]
                data = load_artifact(path)
                if name != "best_validation":
                    if data.get("num_test_samples") != 128:
                        raise ValueError(f"unexpected test sample count in {path}")
                    if not data.get("canonical_test_fingerprint"):
                        raise ValueError(f"missing canonical test fingerprint in {path}")
                per_seed.append(float(data[spec["field"]]))
            rows.append({
                "num_uav": num_uav,
                "training_samples": train_size,
                "num_model_seeds": len(per_seed),
                "mean_mbps": float(np.mean(per_seed)),
                "sem_mbps": sem(per_seed),
                "seed0_mbps": per_seed[0],
                "seed1_mbps": per_seed[1],
                "seed2_mbps": per_seed[2],
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(name: str, rows: list[dict[str, Any]], nodes: list[int]) -> None:
    spec = METRICS[name]
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
            axis.errorbar(
                [row["training_samples"] for row in curve],
                [row["mean_mbps"] for row in curve],
                yerr=[row["sem_mbps"] for row in curve],
                fmt="o-",
                ms=3,
                lw=1.0,
                capsize=2,
                color=NODE_COLORS[num_uav],
                markerfacecolor=NODE_COLORS[num_uav],
                label=f"N={num_uav}",
            )
        axis.set_title(spec["title"])
        axis.set_xlabel("Number of training samples")
        axis.set_ylabel(spec["ylabel"])
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
        figure.savefig(figures / f"{spec['figure']}.eps", format="eps")
        figure.savefig(figures / f"{spec['figure']}.jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--sizes", default="10,20,30,40,50,60,80,100,200,300")
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    nodes = parse_ints(args.nodes)
    sizes = parse_ints(args.sizes)
    seeds = parse_ints(args.seeds)
    if len(seeds) != 3:
        raise ValueError("the Fig. 4 diagnostic requires exactly three model seeds")

    output_dir = HERE / "data/diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in METRICS:
        rows = aggregate_metric(name=name, nodes=nodes, sizes=sizes, seeds=seeds)
        write_csv(output_dir / f"{METRICS[name]['figure']}.csv", rows)
        plot_metric(name, rows, nodes)


if __name__ == "__main__":
    main()
