#!/usr/bin/env python3
"""Render the Chapter IV CSI-feedback rate panel from CSI-conditioned models."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_figures.common.plot_style import NODE_COLORS, paper_style


def load_one(num_uav: int) -> dict[str, float]:
    raw = HERE / "data" / "raw"
    values = {}
    for path in sorted(raw.glob(f"chapter4_N{num_uav}_v2_*.json")):
        payload = json.loads(path.read_text())
        mode = payload["csi_mode"]
        if mode == "quantized":
            mode = f"quantized_{int(payload['q_bits'])}bit"
        values[mode] = payload["results"]["Proposed-GAT"][
            "mean_aggregate_phy_rate_mbps"
        ]

    required = {
        "perfect", "large_scale", "quantized_2bit", "quantized_4bit",
        "quantized_6bit", "quantized_8bit", "quantized_10bit",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise FileNotFoundError(
            f"N={num_uav} is missing v2 Chapter IV outputs for: {', '.join(missing)}"
        )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20")
    args = parser.parse_args()
    nodes = tuple(int(item) for item in args.nodes.split(",") if item)
    records = {num_uav: load_one(num_uav) for num_uav in nodes}
    bits = (2, 4, 6, 8, 10)

    csv_dir = HERE / "data" / "plotting"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "fig07_chapter4_csi_rate_v2.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("num_uav", "mode", "value_mbps"))
        writer.writeheader()
        for num_uav, values in records.items():
            for mode, value in values.items():
                writer.writerow({"num_uav": num_uav, "mode": mode, "value_mbps": value})

    with paper_style():
        mpl.rcParams.update({
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        x = np.arange(len(bits))
        for num_uav in nodes:
            values = records[num_uav]
            color = NODE_COLORS[num_uav]
            ax.plot(
                x,
                [values[f"quantized_{bits_i}bit"] for bits_i in bits],
                marker="o",
                markersize=3.2,
                color=color,
                label=f"N={num_uav}",
            )
            ax.hlines(
                values["perfect"], -0.15, len(bits) - 0.85,
                color=color, linestyle=":", linewidth=0.85,
            )
            ax.hlines(
                values["large_scale"], -0.15, len(bits) - 0.85,
                color=color, linestyle="--", linewidth=0.85,
            )

        ax.set(
            xticks=x,
            xticklabels=[f"{bits_i}-bit" for bits_i in bits],
            xlabel="CSI feedback precision",
            ylabel="True-channel test sum rate (Mbit/s)",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", linestyle="--", linewidth=0.5)
        ax.legend(
            handles=[
                Line2D([0], [0], color="#808080", linestyle=":", lw=0.85),
                Line2D([0], [0], color="#808080", linestyle="--", lw=0.85),
                *[Line2D([0], [0], color=NODE_COLORS[n], linestyle="-", lw=1.4) for n in nodes],
            ],
            labels=["Perfect CSI", "Large-scale CSI", *[f"N={n}" for n in nodes]],
            ncol=2,
        )
        fig.tight_layout(pad=0.2)

        fig_dir = HERE / "figures"
        fig_dir.mkdir(exist_ok=True)
        stem = fig_dir / "fig07_chapter4_csi_rate_v2"
        fig.savefig(stem.with_suffix(".eps"), format="eps")
        fig.savefig(stem.with_suffix(".jpg"), format="jpg", dpi=300)
        plt.close(fig)

    print(f"saved {stem.with_suffix('.eps')}")
    print(f"saved {stem.with_suffix('.jpg')}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
