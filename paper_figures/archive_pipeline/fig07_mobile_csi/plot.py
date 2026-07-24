#!/usr/bin/env python3
"""Render Fig. 7 from archived mobile-CSI records only."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "data/raw"
PLOT = HERE / "data/plotting"
FIG = HERE / "figures"
COLORS = {"latest": "#4F76B8", "stale": "#C65A78", "large_scale": "#65A9A6", "quantized_4bit": "#C98B4A"}
LABELS = {"latest": "Latest", "stale": "Stale", "large_scale": "Large-scale", "quantized_4bit": "4-bit"}
STYLES = {"gat_assisted": "-", "direct_link": "--"}


def rows(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))["records"]


def aggregate(data, xkey, methods):
    groups = defaultdict(list)
    for row in data:
        if row["method"] in methods:
            groups[(float(row[xkey]), row["scheduler_variant"], row["method"])].append(row)
    output = []
    for (x, scheduler, method), values in groups.items():
        goodput = np.asarray([v["delivered_throughput_mbps"] for v in values], float)
        backlog = np.asarray([v["average_total_queue_backlog_mbit"] for v in values], float)
        output.append({"x": x, "scheduler": scheduler, "method": method,
                       "goodput": goodput.mean(), "goodput_ci95": 1.96 * goodput.std(ddof=1) / np.sqrt(len(goodput)) if len(goodput)>1 else 0,
                       "backlog": backlog.mean(), "n": len(values)})
    return sorted(output, key=lambda r: (r["scheduler"], r["method"], r["x"]))


def panel(ax, data, xkey, methods, xlabel, metric="goodput"):
    summary = aggregate(data, xkey, methods)
    for scheduler in ("gat_assisted", "direct_link"):
        for method in methods:
            values = [r for r in summary if r["scheduler"] == scheduler and r["method"] == method]
            if not values:
                continue
            ax.plot([r["x"] for r in values], [r[metric] for r in values],
                    color=COLORS[method], ls=STYLES[scheduler], marker="o", ms=3, lw=1.25,
                    label=f"{LABELS[method]} / {'GAT' if scheduler=='gat_assisted' else 'Direct'}")
    # Optional centralized reference: show only its standard current/full-CSI
    # row to avoid turning the paired CSI-sensitivity panel into 12 curves.
    fplinq = [
        r for r in summary
        if r["scheduler"] == "weighted_fplinq" and r["method"] == "latest"
    ]
    if fplinq:
        ax.plot(
            [r["x"] for r in fplinq], [r[metric] for r in fplinq],
            color="#2F2F2F", ls=":", marker="^", ms=3, lw=1.25,
            label="Weighted FPLinQ / Latest",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Delivered goodput (Mbit/s)" if metric == "goodput" else "Average total backlog (Mbit)")
    ax.grid(axis="y", color="#E2E2E2", lw=.45)
    ax.spines[["top", "right"]].set_visible(False)
    return summary


def main():
    PLOT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    all_summary = []
    with mpl.rc_context({"font.family":"sans-serif", "font.sans-serif":["Arial","Helvetica","DejaVu Sans"], "font.size":7, "svg.fonttype":"none", "pdf.fonttype":42}):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.65))
        specs = [
            (axes[0,0], rows("speed.json"), "speed_mps", ("latest","stale","large_scale","quantized_4bit"), "Speed (m/s)", "goodput", "(a) Speed"),
            (axes[0,1], rows("block_length.json"), "slots_per_block", ("latest","stale"), "Block length (slots)", "goodput", "(b) Block length"),
            (axes[1,0], rows("stale_age.json"), "stale_blocks", ("latest","stale"), "CSI age (blocks)", "goodput", "(c) Stale age"),
            (axes[1,1], rows("load.json"), "arrival_rate", ("latest","stale"), "Offered load, $\\Lambda$ (packets/slot)", "backlog", "(d) Load stress"),
        ]
        for ax, data, xkey, methods, xlabel, metric, title in specs:
            summary = panel(ax, data, xkey, methods, xlabel, metric)
            for row in summary: row.update({"panel": title, "xkey": xkey})
            all_summary.extend(summary)
            ax.set_title(title, loc="left", fontweight="bold")
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=6.5)
        fig.tight_layout(rect=(0,0,1,.91))
        stem = FIG / "fig07_mobile_csi_double_column"
        fig.savefig(stem.with_suffix(".eps")); fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg")); fig.savefig(stem.with_suffix(".tiff"), dpi=600)
        fig.savefig(FIG / "fig07_mobile_csi_double_column_review.png", dpi=300)
    with (PLOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary[0])); writer.writeheader(); writer.writerows(all_summary)


if __name__ == "__main__": main()
