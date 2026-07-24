#!/usr/bin/env python3
"""Render Fig. 4 from its saved paired dynamic-queue aggregate JSON."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-scheduling-mpl")
import matplotlib as mpl
import matplotlib.pyplot as plt

FIGURE_DIR = Path(__file__).resolve().parent
RAW_DIR = FIGURE_DIR / "data/raw/formal_N20"
PLOTTING = FIGURE_DIR / "data/plotting"
FIGURES = FIGURE_DIR / "figures"
COLORS = {
    "GAT-Assisted MarginalMP": "#C65A78", "Direct-Link MarginalMP": "#7884B4",
    "Direct-Link LocalMax": "#65A9A6", "Direct-Link Join": "#C98B4A",
    "Hard-Conflict LGS": "#8F8F8F", "Random": "#D4D4D4",
    "Weighted FPLinQ": "#484878", "Centralized Greedy": "#303030",
}
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "*")
VECTOR_TEXT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}
FONT_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _draw_panel(axis, panel_index, panel, rows, methods, *, legend=False) -> None:
    metric, ylabel, title = panel
    for method_index, method in enumerate(methods):
        items = [r for r in rows if r["method"] == method]
        if metric == "average_packet_delay_ms":
            items = [r for r in items if float(r["successful_delivery_ratio"] or 0.0) >= 0.99]
        x = [r["arrival_rate_packets_per_slot"] for r in items]
        y = [r[metric] for r in items]
        if x and not any(v is None for v in y):
            axis.plot(x, y, color=COLORS.get(method, "#555555"), marker=MARKERS[method_index % len(MARKERS)], ms=3.3, lw=1.25, label=method)
    axis.set_xlabel("Total offered load $\\Lambda$ (packet/slot)"); axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
    axis.text(0.015, 0.97, f"({chr(97 + panel_index)})", transform=axis.transAxes, ha="left", va="top", fontweight="bold", fontsize=8)
    axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E1E1E1", lw=0.45)
    if legend: axis.legend(ncol=2, frameon=False, fontsize=6.2, loc="best")


def main() -> None:
    config_path, metrics_path = RAW_DIR / "config.json", RAW_DIR / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("missing formal Fig. 4 raw metrics; run pipeline.py run first")
    config = json.loads(config_path.read_text(encoding="utf-8")); metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    methods, lambdas, rows = config["methods"], config["traffic"]["arrival_rates"], []
    for method in methods:
        for lam in lambdas:
            item = metrics[f"{method}|lambda={float(lam):g}"]
            rows.append({
                "method": method, "arrival_rate_packets_per_slot": float(lam),
                "num_mc_runs": item["num_mc_runs"],
                "successful_goodput_mbps": item.get("successful_goodput_mbps"),
                "successful_goodput_mbps_ci95": item.get("successful_goodput_mbps_ci95"),
                "average_total_queue_backlog_mbit": item.get("average_total_queue_backlog_mbit"),
                "average_total_queue_backlog_mbit_ci95": item.get("average_total_queue_backlog_mbit_ci95"),
                "successful_delivery_ratio": item.get("successful_delivery_ratio"),
                "successful_delivery_ratio_ci95": item.get("successful_delivery_ratio_ci95"),
                "average_packet_delay_ms": item.get("average_packet_delay_ms"),
                "average_packet_delay_ms_ci95": item.get("average_packet_delay_ms_ci95"),
            })
    _write_csv(PLOTTING / "summary.csv", rows)
    panels = (
        ("successful_goodput_mbps", "Delivered goodput (Mbit/s)", "Goodput"),
        ("average_total_queue_backlog_mbit", "Average total backlog (Mbit)", "Backlog"),
        ("successful_delivery_ratio", "Successful delivery ratio", "Completion"),
        ("average_packet_delay_ms", "Completed-packet delay (ms)", "Delay"),
    )
    with mpl.rc_context({**VECTOR_TEXT_RC, **FONT_RC, "font.size": 7, "axes.labelsize": 7.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.4,
                         "axes.linewidth": 0.75}):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.65), squeeze=False); axes = axes.ravel()
        for panel_index, (metric, ylabel, title) in enumerate(panels):
            axis = axes[panel_index]
            for method_index, method in enumerate(methods):
                items = [r for r in rows if r["method"] == method]
                if metric == "average_packet_delay_ms":
                    items = [r for r in items if float(r["successful_delivery_ratio"] or 0.0) >= 0.99]
                x = [r["arrival_rate_packets_per_slot"] for r in items]
                y = [r[metric] for r in items]
                if not x or any(v is None for v in y):
                    continue
                axis.plot(x, y, color=COLORS.get(method, "#555555"), marker=MARKERS[method_index % len(MARKERS)],
                          ms=3.3, lw=1.25, label=method)
            axis.set_xlabel("Total offered load $\\Lambda$ (packet/slot)"); axis.set_ylabel(ylabel)
            axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
            axis.text(0.015, 0.97, f"({chr(97 + panel_index)})", transform=axis.transAxes,
                      ha="left", va="top", fontweight="bold", fontsize=8)
            axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E1E1E1", lw=0.45)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.13, top=0.84, wspace=0.33, hspace=0.40)
        FIGURES.mkdir(parents=True, exist_ok=True); stem = FIGURES / "fig04_dynamic_queue_double_column"
        fig.savefig(stem.with_suffix(".eps")); fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".tiff"), dpi=600); fig.savefig(FIGURES / "fig04_dynamic_queue_double_column_review.png", dpi=300)
        plt.close(fig)
        # Author-review alternatives: one IEEE single-column stack and four
        # independent single-column panels.  They reuse exactly the same rows.
        fig, axes = plt.subplots(4, 1, figsize=(3.50, 7.45), squeeze=False)
        for i, panel in enumerate(panels): _draw_panel(axes.ravel()[i], i, panel, rows, methods, legend=(i == 0))
        fig.tight_layout(); stem = FIGURES / "fig04_dynamic_queue_single_column_stack"
        fig.savefig(stem.with_suffix(".eps")); fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".tiff"), dpi=600); fig.savefig(FIGURES / "fig04_dynamic_queue_single_column_stack_review.png", dpi=300); plt.close(fig)
        for i, panel in enumerate(panels):
            fig, axis = plt.subplots(figsize=(3.50, 2.55)); _draw_panel(axis, i, panel, rows, methods, legend=True)
            fig.tight_layout(); stem = FIGURES / f"fig04_{panel[2].lower()}_single_panel"
            fig.savefig(stem.with_suffix(".eps")); fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".tiff"), dpi=600); fig.savefig(FIGURES / f"fig04_{panel[2].lower()}_single_panel_review.png", dpi=300); plt.close(fig)
    print(f"saved Fig. 4 sources and exports under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
