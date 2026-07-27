#!/usr/bin/env python3
"""Render independent Fig. 5 EPS panels from saved aggregate JSON only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_figures.common.plot_style import paper_style

HERE = Path(__file__).resolve().parent
DISPLAY = {
    "Random": "random", "GAT-Assisted MarginalMP": "GAT-MP",
    "Direct-Link MarginalMP": "Direct-MP", "GAT-Assisted LocalMax": "GAT-LocalMax",
    "Direct-Link LocalMax": "Direct-Link LocalMax", "End-to-End GAT-Q": "End-to-End GAT-Q",
    "Hard-Conflict LGS": "Hard-Conflict LGS", "Centralized Greedy": "Centralized Greedy",
    "Weighted FPLinQ": "Weighted FPLinQ",
}
COLORS = {
    "random": "#BEBEBE", "GAT-MP": "#C65A78", "Direct-MP": "#7884B4",
    "GAT-LocalMax": "#65A9A6", "Direct-Link LocalMax": "#7A9E6A",
    "End-to-End GAT-Q": "#C98B4A", "Hard-Conflict LGS": "#8B8B8B",
    "Centralized Greedy": "#343434", "Weighted FPLinQ": "#484878",
}
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
METRICS = {
    "backlog": ("average_total_queue_backlog_mbit", "Average queue backlog (Mbit)", None),
    "delay": ("average_packet_delay_ms", "Average completed-packet delay (ms)", (0, 2000)),
    "goodput": ("successful_goodput_mbps", "Goodput (Mbit/s)", None),
}


def collect(raw: Path, n: int) -> list[dict]:
    config = json.loads((raw / f"N{n}" / "config.json").read_text())
    metrics = json.loads((raw / f"N{n}" / "metrics.json").read_text())
    rows = []
    for method in config["methods"]:
        for lam in config["traffic"]["arrival_rates"]:
            x = metrics[f"{method}|lambda={float(lam):g}"]
            rows.append({"num_uav": n, "method": DISPLAY[method], "arrival_rate_packets_per_slot": float(lam), **x})
    return rows


def save_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def draw(rows: list[dict], n: int, metric: str, out: Path) -> None:
    key, ylabel, ylim = METRICS[metric]
    with paper_style():
        mpl.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8})
        fig, ax = plt.subplots(figsize=(3.5, 2.45))
        methods = list(dict.fromkeys(row["method"] for row in rows))
        for i, method in enumerate(methods):
            series = [row for row in rows if row["method"] == method and row.get(key) is not None]
            if not series:
                continue
            ax.plot([r["arrival_rate_packets_per_slot"] for r in series], [r[key] for r in series],
                    label=method, color=COLORS[method], marker=MARKERS[i], markersize=3.0)
        ax.set_xlabel("Traffic arrival rate $\\lambda$ (packet/slot)")
        ax.set_ylabel(ylabel)
        if ylim: ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=2, frameon=False, loc="best", fontsize=5.7)
        out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, format="eps"); fig.savefig(out.with_suffix(".jpg"), format="jpg", dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("no_deadline", "deadline_2000ms"), default="no_deadline")
    p.add_argument("--nodes", default="20,50")
    args = p.parse_args(); raw = HERE / "data/raw" / args.profile
    all_rows = []
    for n in [int(x) for x in args.nodes.split(",") if x.strip()]:
        rows = collect(raw, n); all_rows += rows
        for metric in METRICS:
            draw(rows, n, metric, HERE / "figures" / f"fig05_{metric}_N{n}_{args.profile}.eps")
    save_csv(all_rows, HERE / "data/plotting" / f"fig05_{args.profile}_summary.csv")


if __name__ == "__main__":
    main()
