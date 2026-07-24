#!/usr/bin/env python3
"""Render TNSE Fig. 3 only from the saved paired snapshot JSON."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-scheduling-mpl")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIGURE_DIR = Path(__file__).resolve().parent
RAW = FIGURE_DIR / "data/raw/snapshot_ablation_N20.json"
PLOTTING = FIGURE_DIR / "data/plotting"
FIGURES = FIGURE_DIR / "figures"
METHODS = (
    "GAT-Assisted MarginalMP", "Direct-Link MarginalMP",
    "GAT-Assisted LocalMax", "Direct-Link LocalMax", "Direct-Link Join",
    "Hard-Conflict LGS", "Centralized Greedy",
)
PROFILES = ("sparse", "heterogeneous", "saturated")
COLORS = {
    "GAT-Assisted MarginalMP": "#C65A78", "Direct-Link MarginalMP": "#7884B4",
    "GAT-Assisted LocalMax": "#CDB8D6", "Direct-Link LocalMax": "#65A9A6",
    "Direct-Link Join": "#C98B4A", "Hard-Conflict LGS": "#8F8F8F",
    "Centralized Greedy": "#303030",
}
SHORT_LABELS = {
    "GAT-Assisted MarginalMP": "GAT-MP", "Direct-Link MarginalMP": "DL-MP",
    "GAT-Assisted LocalMax": "GAT-LM", "Direct-Link LocalMax": "DL-LM",
    "Direct-Link Join": "DL-Join", "Hard-Conflict LGS": "LGS",
    "Centralized Greedy": "CG",
}
VECTOR_TEXT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}
FONT_RC = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _tables(payload: dict) -> tuple[list[dict], list[dict]]:
    records = payload["records"]
    central = {(r["graph_index"], r["queue_profile"]): r["mws_utility_queue_weighted_bit_s_hz"]
               for r in records if r["method"] == "Centralized Greedy"}
    per_snapshot, groups = [], defaultdict(list)
    for r in records:
        ratio = r["mws_utility_queue_weighted_bit_s_hz"] / max(central[(r["graph_index"], r["queue_profile"])], 1e-12)
        message = r["message_accounting"]
        item = {
            "graph_index": r["graph_index"], "graph_seed": r["graph_seed"],
            "queue_profile": r["queue_profile"], "method": r["method"],
            "mws_utility_queue_weighted_bit_s_hz": r["mws_utility_queue_weighted_bit_s_hz"],
            "normalized_mws_vs_centralized_greedy": ratio,
            "final_active_links": r["final_hard_feasibility"]["active_links"],
            "candidate_endpoint_excess": r["candidate_hard_feasibility"]["endpoint_excess_count"],
            "logical_neighbor_exchange_rounds": message["logical_neighbor_exchange_rounds"],
            "physical_externality_pairs": message["physical_externality_pairs"],
            "arbitration_directed_edge_messages": message["arbitration_directed_edge_messages"],
        }
        per_snapshot.append(item); groups[(item["queue_profile"], item["method"])].append(item)
    summary = []
    for (profile, method), rows in groups.items():
        utility = np.asarray([row["normalized_mws_vs_centralized_greedy"] for row in rows])
        summary.append({
            "queue_profile": profile, "method": method, "num_paired_snapshots": len(rows),
            "mean_normalized_mws_vs_centralized_greedy": float(utility.mean()),
            "sem_normalized_mws_vs_centralized_greedy": float(utility.std(ddof=1) / np.sqrt(len(utility))) if len(utility) > 1 else 0.0,
            "mean_final_active_links": float(np.mean([row["final_active_links"] for row in rows])),
            "mean_logical_neighbor_exchange_rounds": float(np.mean([row["logical_neighbor_exchange_rounds"] for row in rows])),
            "mean_physical_externality_pairs": float(np.mean([row["physical_externality_pairs"] for row in rows])),
        })
    return per_snapshot, summary


def main() -> None:
    if not RAW.is_file():
        raise FileNotFoundError(f"missing {RAW}; run pipeline.py evaluate first")
    payload = json.loads(RAW.read_text(encoding="utf-8"))
    per_snapshot, summary = _tables(payload)
    _write_csv(PLOTTING / "per_snapshot_metrics.csv", per_snapshot)
    _write_csv(PLOTTING / "summary.csv", summary)
    lookup = {(r["queue_profile"], r["method"]): r for r in summary}
    with mpl.rc_context({
        **VECTOR_TEXT_RC, **FONT_RC, "font.size": 7, "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.75,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.70), squeeze=False)
        axes = axes.ravel(); x = np.arange(len(METHODS))
        for panel, profile in enumerate(PROFILES):
            axis = axes[panel]
            rows = [lookup[(profile, method)] for method in METHODS]
            axis.bar(x, [r["mean_normalized_mws_vs_centralized_greedy"] for r in rows],
                     color=[COLORS[m] for m in METHODS], edgecolor="#303030", linewidth=0.3)
            axis.axhline(1.0, color="#303030", lw=0.75, ls="--")
            axis.set_ylim(0, 1.08); axis.set_xticks(x, [SHORT_LABELS[m] for m in METHODS], rotation=0)
            axis.set_ylabel("Normalized $\\sum_i Q_iR_i$")
            axis.set_title(profile.capitalize(), loc="center", fontsize=8, fontweight="bold")
            axis.text(0.015, 0.97, f"({chr(97 + panel)})", transform=axis.transAxes,
                      ha="left", va="top", fontweight="bold", fontsize=8)
            axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E1E1E1", lw=0.45)
        axis = axes[3]
        for method in METHODS:
            rows = [lookup[(p, method)] for p in PROFILES]
            axis.plot(np.arange(3), [r["mean_logical_neighbor_exchange_rounds"] for r in rows],
                      marker="o", ms=3.4, lw=1.25, color=COLORS[method], label=method)
        axis.set_xticks(np.arange(3), [p.capitalize() for p in PROFILES])
        axis.set_ylabel("Logical neighbor-exchange rounds")
        axis.text(0.015, 0.97, "(d)", transform=axis.transAxes, ha="left", va="top", fontweight="bold", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", color="#E1E1E1", lw=0.45)
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.84, wspace=0.30, hspace=0.42)
        FIGURES.mkdir(parents=True, exist_ok=True)
        stem = FIGURES / "fig03_snapshot_ablation_double_column"
        fig.savefig(stem.with_suffix(".eps")); fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".svg"))
        fig.savefig(stem.with_suffix(".tiff"), dpi=600); fig.savefig(FIGURES / "fig03_snapshot_ablation_double_column_review.png", dpi=300)
        plt.close(fig)
    print(f"saved Fig. 3 sources and exports under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
