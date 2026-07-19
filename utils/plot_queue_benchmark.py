#!/usr/bin/env python3
"""Plot one formal fixed-CSI queue benchmark result directory."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


METHOD_STYLE = {
    "LocalPhysicalReuse-v2": ("#d62728", "o", "-"),
    "PurePhysicalReuse": ("#ff7f0e", "s", "--"),
    "EndToEnd-GAT-q": ("#9467bd", "^", "-."),
    "Conference-GAT-nonempty": ("#1f77b4", "D", "--"),
    "Centralized-MWS-Greedy": ("#2ca02c", "*", "-"),
    "Random-nonempty": ("#7f7f7f", "x", ":"),
    "SNR-SingleLink": ("#8c564b", "v", ":"),
    "LocalMax-GAT": ("#17becf", "P", "--"),
    "LocalMax-SoloRate": ("#bcbd22", "X", "--"),
}


DEADLINE_PLOTS = (
    ("successful_packet_total_rate_pps", "Successful goodput (packet/slot)", "goodput"),
    ("departure_total_rate_pps", "Physical service (packet-equivalent/slot)", "physical_service"),
    ("average_queue_backlog_mbit", "Average queue backlog (Mbit/link)", "backlog"),
    ("average_packet_delay_ms", "Mean delivered-packet delay (ms)", "mean_packet_delay"),
    ("p95_packet_delay_ms", "95th-percentile delivered-packet delay (ms)", "p95_packet_delay"),
    ("qos_packet_loss_rate", "QoS packet loss rate", "packet_loss"),
    ("unfinished_packet_fraction", "Unfinished packet fraction at run end", "unfinished_packets"),
    ("queue_growth_slope", "Queue growth slope (packet/link/slot)", "queue_growth"),
)

NO_DEADLINE_CORE_PLOTS = (
    ("successful_packet_total_rate_pps", "Delivered throughput (packet/slot)", "delivered_throughput"),
    ("average_queue_backlog_mbit", "Average queue backlog (Mbit/link)", "backlog"),
    ("average_packet_delay_ms", "Mean delivered-packet delay (ms)", "mean_packet_delay"),
)


def _load(result_dir):
    with open(result_dir / "config.json", encoding="utf-8") as file:
        config = json.load(file)
    with open(result_dir / "metrics.json", encoding="utf-8") as file:
        metrics = json.load(file)
    return config, metrics


def _method_series(config, metrics, method, metric):
    xs, ys, cis = [], [], []
    for arrival_lambda in config["traffic"]["arrival_rates"]:
        key = f"{method}|lambda={float(arrival_lambda):g}"
        item = metrics.get(key, {})
        value = item.get(metric)
        ci = item.get(f"{metric}_ci95")
        if (
            metric == "successful_packet_total_rate_pps"
            and value is None
            and item.get("delivered_packet_count") is not None
        ):
            # Backward-compatible derivation for completed runs that already
            # recorded exact delivered packet counts before this named metric
            # was added to the simulator.
            num_slots = float(config["traffic"]["num_slots"])
            value = float(item["delivered_packet_count"]) / num_slots
            ci = float(item.get("delivered_packet_count_ci95") or 0.0) / num_slots
        if (
            config.get("figure_profile") == "no_deadline_core"
            and metric in {"average_packet_delay_ms", "p95_packet_delay_ms"}
            and float(item.get("successful_delivery_ratio") or 0.0) < 0.99
        ):
            # Without a deadline, unfinished packets are right-censored. Mean
            # delay is shown only when at least 99% of offered packets completed.
            continue
        if value is None:
            continue
        xs.append(float(arrival_lambda))
        ys.append(float(value))
        cis.append(float(ci or 0.0))
    return xs, ys, cis


def plot_result(result_dir):
    result_dir = Path(result_dir)
    config, metrics = _load(result_dir)
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    methods = config["methods"]
    plots = (
        NO_DEADLINE_CORE_PLOTS
        if config.get("figure_profile") == "no_deadline_core"
        else DEADLINE_PLOTS
    )
    for metric, ylabel, stem in plots:
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for method in methods:
            xs, ys, cis = _method_series(config, metrics, method, metric)
            if not xs:
                continue
            color, marker, linestyle = METHOD_STYLE.get(
                method, (None, "o", "-")
            )
            lower = [max(0.0, y - ci) for y, ci in zip(ys, cis)]
            upper = [y + ci for y, ci in zip(ys, cis)]
            axis.fill_between(
                xs, lower, upper, color=color, alpha=0.10, linewidth=0,
            )
            axis.plot(
                xs, ys, label=method, color=color,
                marker=marker, linestyle=linestyle, linewidth=1.6,
                markersize=5,
            )
        axis.set_xlabel(r"Total offered load $\Lambda$ (packet/slot)")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle=":", alpha=0.5)
        axis.legend(
            fontsize=7.2, frameon=False, ncol=2,
            loc="lower center", bbox_to_anchor=(0.5, 1.01),
            columnspacing=1.0, handlelength=2.4,
        )
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(
                figure_dir / f"{stem}.{suffix}", dpi=300,
                bbox_inches="tight",
            )
        plt.close(figure)
    return figure_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir")
    args = parser.parse_args()
    print(plot_result(args.result_dir))


if __name__ == "__main__":
    main()
