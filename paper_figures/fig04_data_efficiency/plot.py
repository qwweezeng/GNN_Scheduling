#!/usr/bin/env python3
"""Render all Fig. 4 checkpoint-selection and decision-rule curves.

The script preserves every requested metric rather than collapsing evaluation
modes into a single number:
  - best-validation checkpoint / final-epoch checkpoint
  - soft pi utility / hard threshold without projection / hard feasible rate
  - best validation utility recorded during training

A long-form seed-level CSV and a mean/SEM summary CSV are written before plots.
The paper main curve is ``fig04_best_projected_rate``; the remaining EPS files
are explicitly named diagnostics and retain the complete audit trail.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_figures.common.plot_style import NODE_COLORS, paper_style

SIZES = (10, 20, 30, 40, 60, 80, 100, 200, 300)
EVALUATIONS = {
    "soft": ("test_soft_utility_mbps", "Soft test utility (Mbit/s)", "soft_utility"),
    "hard": (
        "test_hard_unprojected_utility_mbps",
        "Hard unprojected test utility (Mbit/s)",
        "hard_unprojected_utility",
    ),
    "projected": (
        "test_projected_binary_rate_mbps",
        "Hard-feasible test rate (Mbit/s)",
        "projected_rate",
    ),
}
SELECTIONS = ("best", "final")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Fig.4 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(
    rows: list[dict[str, Any]],
    nodes: list[int],
    selection: str,
    metric: str,
    ylabel: str,
    filename_stem: str,
) -> None:
    figure_dir = HERE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with paper_style():
        mpl.rcParams.update({
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        for num_uav in nodes:
            selected = [
                row for row in rows
                if row["num_uav"] == num_uav
                and row["checkpoint_selection"] == selection
                and row["metric"] == metric
            ]
            selected.sort(key=lambda row: row["training_samples"])
            ax.errorbar(
                [row["training_samples"] for row in selected],
                [row["mean_mbps"] for row in selected],
                yerr=[row["sem_mbps"] for row in selected],
                fmt="o-",
                ms=3,
                lw=1.0,
                capsize=2,
                color=NODE_COLORS[num_uav],
                markerfacecolor=NODE_COLORS[num_uav],
                label=f"N={num_uav}",
            )
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(size) for size in SIZES])
        ax.legend(ncol=2, frameon=True, fontsize=8)
        fig.savefig(figure_dir / f"{filename_stem}.eps", format="eps", bbox_inches="tight")
        fig.savefig(figure_dir / f"{filename_stem}.jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_best_validation(rows: list[dict[str, Any]], nodes: list[int]) -> None:
    figure_dir = HERE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with paper_style():
        mpl.rcParams.update({
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        for num_uav in nodes:
            selected = [row for row in rows if row["num_uav"] == num_uav]
            selected.sort(key=lambda row: row["training_samples"])
            ax.errorbar(
                [row["training_samples"] for row in selected],
                [row["best_val_utility_mean"] for row in selected],
                yerr=[row["best_val_utility_sem"] for row in selected],
                fmt="o-",
                ms=3,
                lw=1.0,
                capsize=2,
                color=NODE_COLORS[num_uav],
                markerfacecolor=NODE_COLORS[num_uav],
                label=f"N={num_uav}",
            )
        ax.set_xlabel("Number of training samples")
        ax.set_ylabel("Best validation utility")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(size) for size in SIZES])
        ax.legend(ncol=2, frameon=True, fontsize=8)
        fig.savefig(figure_dir / "fig04_best_validation_utility.eps", format="eps", bbox_inches="tight")
        fig.savefig(figure_dir / "fig04_best_validation_utility.jpg", format="jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--sizes", default=",".join(str(size) for size in SIZES))
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    nodes = [int(value) for value in args.nodes.split(",") if value]
    sizes = [int(value) for value in args.sizes.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]

    seed_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for num_uav in nodes:
        for train_size in sizes:
            for model_seed in seeds:
                artifact_dir = (
                    HERE / "data/raw" / f"N{num_uav}" / f"n{train_size}" / f"seed{model_seed}"
                )
                training = _load_json(artifact_dir / "training.json")
                validation_rows.append({
                    "num_uav": num_uav,
                    "training_samples": train_size,
                    "model_seed": model_seed,
                    "best_epoch": int(training["best_epoch"]),
                    "best_val_utility": float(training["best_val_utility"]),
                    "best_checkpoint": training.get("best_checkpoint", ""),
                    "final_checkpoint": training.get("final_checkpoint", ""),
                })
                for selection in SELECTIONS:
                    for mode, (value_key, _ylabel, metric) in EVALUATIONS.items():
                        evaluation = _load_json(
                            artifact_dir / f"eval_{selection}_{mode}.json"
                        )
                        seed_rows.append({
                            "num_uav": num_uav,
                            "training_samples": train_size,
                            "model_seed": model_seed,
                            "checkpoint_selection": selection,
                            "evaluation_mode": mode,
                            "metric": metric,
                            "value_mbps": float(evaluation[value_key]),
                            "checkpoint": evaluation["checkpoint"],
                            "canonical_test_fingerprint": evaluation["canonical_test_fingerprint"],
                            "decision_rule": evaluation["decision_rule"],
                            # Soft evaluation predates the schedulable audit field.
                            "schedulable": bool(
                                evaluation.get("schedulable", metric == "projected_rate")
                            ),
                            "best_epoch": int(training["best_epoch"]),
                            "best_val_utility": float(training["best_val_utility"]),
                        })

    grouped: dict[tuple[int, int, str, str], list[float]] = defaultdict(list)
    for row in seed_rows:
        grouped[
            (
                row["num_uav"],
                row["training_samples"],
                row["checkpoint_selection"],
                row["metric"],
            )
        ].append(row["value_mbps"])

    summary_rows = []
    for (num_uav, train_size, selection, metric), values in sorted(grouped.items()):
        summary_rows.append({
            "num_uav": num_uav,
            "training_samples": train_size,
            "checkpoint_selection": selection,
            "metric": metric,
            "mean_mbps": float(np.mean(values)),
            "sem_mbps": _sem(values),
            "num_model_seeds": len(values),
        })

    validation_grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in validation_rows:
        validation_grouped[(row["num_uav"], row["training_samples"])].append(
            row["best_val_utility"]
        )
    validation_summary_rows = [
        {
            "num_uav": num_uav,
            "training_samples": train_size,
            "best_val_utility_mean": float(np.mean(values)),
            "best_val_utility_sem": _sem(values),
            "num_model_seeds": len(values),
        }
        for (num_uav, train_size), values in sorted(validation_grouped.items())
    ]

    data_dir = HERE / "data/plotting"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(data_dir / "fig04_checkpoint_metric_seed_values.csv", seed_rows)
    _write_csv(data_dir / "fig04_checkpoint_metric_summary.csv", summary_rows)
    _write_csv(data_dir / "fig04_best_validation_seed_values.csv", validation_rows)
    _write_csv(data_dir / "fig04_best_validation_summary.csv", validation_summary_rows)

    for selection in SELECTIONS:
        for mode, (_key, ylabel, metric) in EVALUATIONS.items():
            _plot_metric(
                summary_rows,
                nodes,
                selection,
                metric,
                ylabel,
                f"fig04_{selection}_{metric}",
            )
    _plot_best_validation(validation_summary_rows, nodes)


if __name__ == "__main__":
    main()
