#!/usr/bin/env python3
"""Render independent Fig. 5 EPS panels from saved aggregate JSON only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper_figures.common.plot_style import paper_style

HERE = Path(__file__).resolve().parent
DISPLAY = {
    "Random": "Random",
    "Centralized Greedy": "Centralized Greedy",
    "Weighted FPLinQ": "Weighted FPLinQ",
    "Direct-Link LocalMax": "Direct-LocalMax",
    "GAT-Assisted LocalMax": "GAT-LocalMax",
    "Direct-Link MarginalMP": "DirectMP",
    "GAT-Assisted MarginalMP": "GAT-MP",
}
COLORS = {
    "Random": "#BEBEBE", "GAT-MP": "#C65A78", "DirectMP": "#7884B4",
    "GAT-LocalMax": "#65A9A6", "Direct-LocalMax": "#7A9E6A",
    "Centralized Greedy": "#343434", "Weighted FPLinQ": "#484878",
}
METHOD_ORDER = tuple(DISPLAY)
ARRIVAL_RATES = (0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0)
MARKERS = ("o", "s", "^", "D", "P", "X", "v")
METRICS = {
    "backlog": ("average_total_queue_backlog_mbit", "Average queue backlog (Mbit)", None),
    "delay": (
        "resolved_deadline_capped_delay_ms",
        "Resolved-packet latency (ms)",
        (0, 2000),
    ),
    "completed_delay": (
        "average_packet_delay_ms",
        "Average completed-packet delay (ms)",
        (0, 2000),
    ),
    "goodput": ("successful_goodput_mbps", "Goodput (Mbit/s)", None),
}


def add_resolved_capped_delay(metrics: dict, deadline_ms: float | None) -> dict:
    """Add resolved-packet capped latency from saved aggregate scalar metrics."""
    if deadline_ms is None:
        return metrics
    delivered = float(metrics.get("delivered_packet_count") or 0.0)
    dropped = float(metrics.get("deadline_dropped_packets") or 0.0)
    overflow = float(metrics.get("overflow_dropped_packets") or 0.0)
    completed_delay = metrics.get("average_packet_delay_ms")
    resolved = delivered + dropped + overflow
    if completed_delay is None or resolved <= 0.0:
        metrics["resolved_deadline_capped_delay_ms"] = None
    else:
        metrics["resolved_deadline_capped_delay_ms"] = (
            delivered * float(completed_delay) + (dropped + overflow) * deadline_ms
        ) / resolved
    return metrics


def collect(raw: Path, n: int) -> list[dict]:
    config = json.loads((raw / f"N{n}" / "config.json").read_text())
    metrics = json.loads((raw / f"N{n}" / "metrics.json").read_text())
    deadline_ms = config.get("physical_setting", {}).get("delay_deadline_ms")
    available_methods = set(config["methods"])
    missing_methods = [method for method in METHOD_ORDER if method not in available_methods]
    available_rates = {float(rate) for rate in config["traffic"]["arrival_rates"]}
    missing_rates = [rate for rate in ARRIVAL_RATES if rate not in available_rates]
    if missing_methods or missing_rates:
        details = []
        if missing_methods:
            details.append(f"methods={missing_methods}")
        if missing_rates:
            details.append(f"arrival_rates={missing_rates}")
        raise ValueError(
            f"{raw / f'N{n}'} does not contain the required Fig. 5 pressure-test data: "
            + ", ".join(details)
        )
    rows = []
    for method in METHOD_ORDER:
        for lam in ARRIVAL_RATES:
            x = add_resolved_capped_delay(
                dict(metrics[f"{method}|lambda={float(lam):g}"]), deadline_ms
            )
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
        mpl.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0, "legend.fontsize": 6.5})
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
        ax.grid(axis="both", linestyle="--", linewidth=0.5)
        ax.legend(ncol=2, loc="best", fontsize=5.7)
        out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, format="eps"); fig.savefig(out.with_suffix(".jpg"), format="jpg", dpi=300, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("no_deadline", "deadline_2000ms"), default="deadline_2000ms")
    p.add_argument("--nodes", default="50")
    p.add_argument(
        "--raw-root",
        default="",
        help="Optional result root created by run.py --output-root.",
    )
    p.add_argument(
        "--metrics", default=",".join(METRICS),
        help="Comma-separated panel names; defaults to all standard panels.",
    )
    args = p.parse_args()
    raw_root = Path(args.raw_root) if args.raw_root else HERE / "data/raw"
    if not raw_root.is_absolute():
        raw_root = ROOT / raw_root
    raw = raw_root / args.profile
    selected_metrics = tuple(x.strip() for x in args.metrics.split(",") if x.strip())
    unknown_metrics = set(selected_metrics) - set(METRICS)
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {sorted(unknown_metrics)}")
    all_rows = []
    for n in [int(x) for x in args.nodes.split(",") if x.strip()]:
        rows = collect(raw, n); all_rows += rows
        for metric in selected_metrics:
            output_metric = "delay" if metric == "completed_delay" else metric
            suffix = "_completed_only" if metric == "completed_delay" else ""
            draw(
                rows, n, metric,
                HERE / "figures" / f"fig05_{output_metric}_N{n}_{args.profile}{suffix}.eps",
            )
    save_csv(all_rows, HERE / "data/plotting" / f"fig05_{args.profile}_summary.csv")


if __name__ == "__main__":
    main()
