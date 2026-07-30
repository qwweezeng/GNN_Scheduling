#!/usr/bin/env python3
"""Render the two Fig. 8 GAT-MP mobile-CSI backlog EPS panels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_figures.common.plot_style import NODE_COLORS, paper_style

DEFAULT_NODES = (20, 30, 40, 50)


def load_records(nodes: tuple[int, ...]) -> list[dict]:
    records = []
    for num_uav in nodes:
        for sweep_name in ("speed", "block"):
            path = HERE / "data/raw" / f"gatmp_{sweep_name}_N{num_uav}.json"
            records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    return records


def save_csv(rows: list[dict]) -> None:
    output = HERE / "data/plotting/fig08_gatmp_backlog.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def panel(
    rows: list[dict],
    *,
    xkey: str,
    xlabel: str,
    stem: str,
    nodes: tuple[int, ...],
) -> None:
    with paper_style():
        mpl.rcParams.update({
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
        })
        figure, axis = plt.subplots(figsize=(3.5, 2.35))
        for num_uav in nodes:
            for csi_method, linestyle in (("latest", "-"), ("stale", "--")):
                grouped = defaultdict(list)
                for row in rows:
                    if row["num_uav"] == num_uav and row["method"] == csi_method:
                        grouped[row[xkey]].append(row["average_total_queue_backlog_mbit"])
                x_values = sorted(grouped)
                axis.plot(
                    x_values,
                    [np.mean(grouped[value]) for value in x_values],
                    linestyle=linestyle,
                    color=NODE_COLORS[num_uav],
                    marker="o",
                    markersize=2.7,
                    label=f"N={num_uav}, {'perfect' if csi_method == 'latest' else '1-block stale'}",
                )
        axis.set(xlabel=xlabel, ylabel="Average queue backlog (Mbit)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="both", linestyle="--", linewidth=0.5)
        axis.legend(ncol=2, fontsize=5.5)
        output = HERE / "figures" / stem
        output.parent.mkdir(exist_ok=True)
        figure.savefig(output, format="eps")
        figure.savefig(output.with_suffix(".jpg"), format="jpg", dpi=300, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        default=",".join(map(str, DEFAULT_NODES)),
        help="Comma-separated completed network sizes to render.",
    )
    args = parser.parse_args()
    nodes = tuple(int(value) for value in args.nodes.split(",") if value)
    if not nodes:
        raise ValueError("--nodes must contain at least one network size")

    records = load_records(nodes)
    save_csv(records)
    panel(
        [row for row in records if row["slots_per_block"] == 5],
        xkey="speed_mps",
        xlabel="UAV speed (m/s)",
        stem="fig08_gatmp_speed.eps",
        nodes=nodes,
    )
    panel(
        [row for row in records if row["speed_mps"] == 10],
        xkey="slots_per_block",
        xlabel="Block length (slots)",
        stem="fig08_gatmp_block_length.eps",
        nodes=nodes,
    )


if __name__ == "__main__":
    main()
