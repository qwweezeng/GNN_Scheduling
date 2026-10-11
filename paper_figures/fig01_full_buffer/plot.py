#!/usr/bin/env python3
"""Render the single Fig. 1 rate panel from saved full-buffer JSON results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-scheduling-mpl")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_figures.common.plot_style import METHOD_COLORS, METHOD_ORDER, paper_style


def _load(nodes: list[int], raw_dir: Path) -> list[dict]:
    rows = []
    for num_uav in nodes:
        payload = json.loads((raw_dir / f"benchmark_N{num_uav}.json").read_text())
        if int(payload["num_uav"]) != num_uav:
            raise ValueError(f"N mismatch for {num_uav}")
        for method in METHOD_ORDER:
            result = payload["results"].get(method)
            if result is None:
                raise ValueError(f"{method} missing for N={num_uav}")
            rows.append({
                "num_uav": num_uav,
                "method": method,
                "mean_rate_mbps": float(result["mean_aggregate_phy_rate_mbps"]),
                "sem_rate_mbps": float(result["standard_error_aggregate_phy_rate_mbps"]),
                "num_samples": int(payload["num_samples"]),
            })
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict], output: Path) -> None:
    nodes = sorted({row["num_uav"] for row in rows})
    lookup = {(row["num_uav"], row["method"]): row for row in rows}
    x = np.arange(len(nodes), dtype=float)
    width = 0.12
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2) * width
    with paper_style():
        mpl.rcParams.update({
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        for index, method in enumerate(METHOD_ORDER):
            selected = [lookup[(node, method)] for node in nodes]
            ax.bar(
                x + offsets[index],
                [row["mean_rate_mbps"] for row in selected],
                width=width,
                color=METHOD_COLORS[method],
                edgecolor="#303030",
                linewidth=0.35,
                label=method,
            )
        ax.set_xticks(x, [str(node) for node in nodes])
        ax.set_xlabel("Number of UAVs")
        ax.set_ylabel("Total physical-layer rate (Mbit/s)")
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22),
                  columnspacing=0.8, handlelength=1.1)
        fig.subplots_adjust(left=0.13, right=0.99, bottom=0.25, top=0.78)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="eps")
        fig.savefig(output.with_suffix(".jpg"), format="jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--raw-dir", default=str(HERE / "data/raw"))
    parser.add_argument(
        "--output-root",
        default=str(HERE),
        help="Root containing figures/ and data/plotting/ for this rendering.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional EPS path. Defaults to <output-root>/figures/fig01_total_rate.eps.",
    )
    args = parser.parse_args()
    nodes = [int(item) for item in args.nodes.split(",") if item.strip()]
    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)
    output = Path(args.output) if args.output else output_root / "figures/fig01_total_rate.eps"
    rows = _load(nodes, raw_dir)
    _write_csv(rows, output_root / "data/plotting/fig01_total_rate.csv")
    render(rows, output)


if __name__ == "__main__":
    main()
