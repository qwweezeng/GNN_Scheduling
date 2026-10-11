#!/usr/bin/env python3
"""Render independent Chapter IV/V CSI-feedback EPS panels (v2 multi-file layout)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_figures.common.plot_style import NODE_COLORS, paper_style

# ------------------------------------------------------------------
# Chapter 4: per-mode files, each with a single Proposed-GAT result.
# ------------------------------------------------------------------
CSI_MODES = ("perfect", "large_scale", "quantized_2bit",
             "quantized_4bit", "quantized_6bit",
             "quantized_8bit", "quantized_10bit")


def _load_ch4_rate(raw_root: Path, num_uav: int, mode: str) -> float | None:
    path = raw_root / f"chapter4_N{num_uav}_v2_{mode}.json"
    if not path.exists():
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
        return data["results"]["Proposed-GAT"]["mean_aggregate_phy_rate_mbps"]
    except Exception:
        return None


# ------------------------------------------------------------------
# Chapter 5: single queue-backlog file per N; method encoded in key.
# ------------------------------------------------------------------

def _load_ch5_backlog(raw_root: Path, num_uav: int) -> dict[str, float]:
    """Load only matched-model Chapter V data; legacy single-model output is rejected."""
    path = raw_root / f"chapter5_N{num_uav}_v2_matched_csi.json"
    if not path.exists():
        return {}
    with path.open() as fh:
        data = json.load(fh)
    if data.get("experiment") != "paper_mobile_imperfect_csi_v2_matched_models":
        raise ValueError(f"Chapter V output has invalid provenance: {path}")
    required = {"latest", "large_scale", "quantized_2bit", "quantized_4bit",
                "quantized_6bit", "quantized_8bit", "quantized_10bit"}
    provenance = data.get("gat_checkpoint_provenance_by_method", {})
    if not required.issubset(provenance):
        raise ValueError(f"Chapter V output lacks CSI-matched provenance: {path}")

    summary = data.get("summary", {})
    result: dict[str, float] = {}
    for key, stats in summary.items():
        # key: "v=0|block=1|age=1|Lambda=2.4|scheduler=gat_assisted|method=latest"
        method_part = key.split("method=")[-1] if "method=" in key else ""
        backlog = (stats.get("average_total_queue_backlog_mbit", {})
                   .get("mean") if isinstance(stats, dict) else None)
        if backlog is not None:
            result[method_part] = backlog
    return result


def _quantization_bits_intersection(
    raw_root: Path, nodes: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the bit values that appear across ALL requested nodes."""
    # Sample the first node to discover available modes; then intersect.
    first_n = nodes[0]
    available = {}
    for mode in CSI_MODES:
        val = _load_ch4_rate(raw_root, first_n, mode)
        if val is not None:
            available[mode] = val
    if not available:
        raise ValueError(f"No Chapter 4 v2 files found for N={first_n}")

    # Keep only modes that exist for ALL nodes
    common = set(available)
    for n in nodes[1:]:
        present = set()
        for mode in CSI_MODES:
            if _load_ch4_rate(raw_root, n, mode) is not None:
                present.add(mode)
        common &= present

    quant_modes = sorted(
        (m for m in common if m.startswith("quantized_")),
        key=lambda m: int(m.split("_")[1][:-3]),
    )
    return tuple(int(m.split("_")[1][:-3]) for m in quant_modes)


def _build_ch4_rows(raw_root: Path, nodes: tuple[int, ...]) -> list[dict]:
    rows = []
    for n in nodes:
        for mode, val in {
            m: _load_ch4_rate(raw_root, n, m)
            for m in CSI_MODES
        }.items():
            if val is not None:
                rows.append({"num_uav": n, "mode": mode, "value": val})
    return rows


def _build_ch5_rows(raw_root: Path, nodes: tuple[int, ...]) -> list[dict]:
    rows = []
    for n in nodes:
        backlogs = _load_ch5_backlog(raw_root, n)
        for method, val in backlogs.items():
            rows.append({"num_uav": n, "method": method, "value": val})
    return rows


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def _draw_ch4(rate_rows: list[dict], nodes: tuple[int, ...],
              bits: tuple[int, ...], *, output_root: Path, stem: str) -> None:
    """Chapter 4: sum rate vs. quantization bits, with perfect/large-scale baselines."""
    from matplotlib.lines import Line2D
    with paper_style():
        mpl.rcParams.update({
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        for n in nodes:
            by_mode = {r["mode"]: r["value"] for r in rate_rows if r["num_uav"] == n}
            perfect_val = by_mode.get("perfect")
            ls_val = by_mode.get("large_scale")
            x = range(len(bits))
            ax.plot(
                x,
                [by_mode[f"quantized_{b}bit"] for b in bits],
                marker="o", ms=3,
                color=NODE_COLORS[n], label=f"N={n}",
            )
            if perfect_val is not None:
                ax.axhline(perfect_val, color=NODE_COLORS[n], linestyle=":", lw=0.8,
                           xmin=0, xmax=1)
            if ls_val is not None:
                ax.axhline(ls_val, color=NODE_COLORS[n], linestyle="--", lw=0.8,
                           xmin=0, xmax=1)

        # Build explicit reference-line legend entries
        ref_handles = [
            Line2D([0], [0], color="#808080", linestyle=":", lw=0.8),
            Line2D([0], [0], color="#808080", linestyle="--", lw=0.8),
        ]
        ref_labels = ["Perfect CSI", "Large-scale CSI"]

        ax.set(
            xticks=list(range(len(bits))),
            xticklabels=[f"{b}-bit" for b in bits],
            xlabel="CSI feedback precision",
            ylabel="Test sum rate (Mbit/s)",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="both", linestyle="--", linewidth=0.5)
        # Combine line+reference handles; suppress the redundant "N legend" entry
        ax.legend(
            handles=ref_handles + [Line2D([0], [0], color=NODE_COLORS[n], linestyle="-", lw=1.4)
                                     for n in nodes],
            labels=ref_labels + [f"N={n}" for n in nodes],
            ncol=2, fontsize=6.5
        )
        out = output_root / "figures" / stem
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="eps")
        fig.savefig(out.with_suffix(".jpg"), format="jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def _draw_ch5(backlog_rows: list[dict], nodes: tuple[int, ...],
              *, output_root: Path, stem: str) -> None:
    """Chapter 5: backlog versus quantization precision with CSI references."""
    from matplotlib.lines import Line2D
    if not backlog_rows:
        print(f"[plot] no Chapter 5 backlog data — skipping {stem}")
        return

    quantized_methods = (
        "quantized_2bit", "quantized_4bit", "quantized_6bit",
        "quantized_8bit", "quantized_10bit",
    )
    bits = tuple(int(method.split("_")[1][:-3]) for method in quantized_methods)
    values = {
        (n, method): next(
            (row["value"] for row in backlog_rows
             if row["num_uav"] == n and row["method"] == method),
            None,
        )
        for n in nodes
        for method in ("latest", "large_scale", *quantized_methods)
    }
    missing = [
        f"N={n}/{method}"
        for n in nodes
        for method in ("latest", "large_scale", *quantized_methods)
        if values[(n, method)] is None
    ]
    if missing:
        raise ValueError(
            "Chapter 5 plot requires complete matched-CSI data; missing "
            + ", ".join(missing)
        )

    with paper_style():
        mpl.rcParams.update({
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.5,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        x = list(range(len(bits)))
        for n in nodes:
            color = NODE_COLORS[n]
            ax.plot(
                x,
                [values[(n, method)] for method in quantized_methods],
                marker="o", ms=3, color=color, label=f"N={n}",
            )
            ax.axhline(values[(n, "latest")], color=color, linestyle=":", lw=0.8)
            ax.axhline(values[(n, "large_scale")], color=color, linestyle="--", lw=0.8)

        ref_handles = [
            Line2D([0], [0], color="#808080", linestyle=":", lw=0.8),
            Line2D([0], [0], color="#808080", linestyle="--", lw=0.8),
        ]
        ref_labels = ["Perfect CSI", "Large-scale CSI"]

        ax.set(
            xlabel="CSI feedback precision",
            ylabel="Average queue backlog (Mbit)",
            xticks=x,
            xticklabels=[f"{bit}-bit" for bit in bits],
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
        ax.grid(axis="both", linestyle="--", linewidth=0.5)
        ax.legend(
            handles=ref_handles + [
                Line2D([0], [0], color=NODE_COLORS[n], linestyle="-", lw=1.4)
                for n in nodes
            ],
            labels=ref_labels + [f"N={n}" for n in nodes],
            ncol=2, fontsize=6.5
        )
        out = output_root / "figures" / stem
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="eps")
        fig.savefig(out.with_suffix(".jpg"), format="jpg", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes", default="20,30,40,50",
        help="Comma-separated completed network sizes (default: 20,30,40,50)",
    )
    parser.add_argument(
        "--chapters", default="4,5",
        help="Comma-separated panels to render: 4, 5, or 4,5 (default).",
    )
    parser.add_argument(
        "--chapter4-raw-root",
        default="",
        help="Optional directory containing Chapter IV per-condition JSON files.",
    )
    parser.add_argument(
        "--chapter5-raw-root",
        default="",
        help="Optional directory containing Chapter V matched-CSI JSON files.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Optional root for generated figures and plotting CSV. "
            "Outputs go to <root>/figures and <root>/data/plotting; "
            "the historical paper_figures directory is used only when omitted."
        ),
    )
    args = parser.parse_args()
    nodes = tuple(int(x) for x in args.nodes.split(",") if x)
    chapters = {int(value) for value in args.chapters.split(",") if value}
    if not nodes:
        raise ValueError("--nodes must contain at least one network size")
    if not chapters or not chapters.issubset({4, 5}):
        raise ValueError("--chapters must contain only 4 and/or 5")

    default_raw = HERE / "data/raw"
    chapter4_raw = Path(args.chapter4_raw_root) if args.chapter4_raw_root else default_raw
    chapter5_raw = Path(args.chapter5_raw_root) if args.chapter5_raw_root else default_raw
    output_root = Path(args.output_root) if args.output_root else HERE
    if not chapter4_raw.is_absolute():
        chapter4_raw = ROOT / chapter4_raw
    if not chapter5_raw.is_absolute():
        chapter5_raw = ROOT / chapter5_raw
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    plotting = output_root / "data/plotting"
    plotting.mkdir(parents=True, exist_ok=True)

    # ---------- Chapter 4 ----------
    if 4 in chapters:
        rate_rows = _build_ch4_rows(chapter4_raw, nodes)
        if not rate_rows:
            raise ValueError("No Chapter 4 v2 data found for the requested nodes")
        bits = _quantization_bits_intersection(chapter4_raw, nodes)

        ch4_csv = plotting / "fig07_chapter4_csi_rate.csv"
        with ch4_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rate_rows[0])
            writer.writeheader()
            writer.writerows(rate_rows)
        print(
            f"[plot] chapter4 CSV: {ch4_csv}  "
            f"({len(rate_rows)} rows, {len(bits)} quantization levels)"
        )
        _draw_ch4(
            rate_rows,
            nodes,
            bits,
            output_root=output_root,
            stem="fig07_chapter4_csi_rate.eps",
        )

    # ---------- Chapter 5 ----------
    if 5 in chapters:
        backlog_rows = _build_ch5_rows(chapter5_raw, nodes)
        if backlog_rows:
            ch5_csv = plotting / "fig07_chapter5_gat_mp_backlog.csv"
            with ch5_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=backlog_rows[0])
                writer.writeheader()
                writer.writerows(backlog_rows)
            print(f"[plot] chapter5 CSV: {ch5_csv}  ({len(backlog_rows)} rows)")
            _draw_ch5(
                backlog_rows,
                nodes,
                output_root=output_root,
                stem="fig07_chapter5_gat_mp_backlog.eps",
            )
        else:
            print("[plot] no Chapter 5 matched-model data found — skipping panel")


if __name__ == "__main__":
    main()
