#!/usr/bin/env python3
"""Print a compact comparison table from fixed-CSI QoS metrics.json files."""

import argparse
import json
from pathlib import Path


METRICS = (
    "successful_packet_total_rate_pps",
    "average_total_queue_backlog_mbit",
    "average_packet_delay_ms",
    "average_active_links",
    "average_hard_projection_rejections",
    "average_mp_accepted_toggles",
    "minimum_mp_local_utility_delta",
)


def load_rows(result_directory):
    path = Path(result_directory) / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for label, metrics in payload.items():
        method, arrival = label.split("|lambda=")
        key = (method, float(arrival))
        rows[key] = {
            "method": method,
            "arrival_rate": float(arrival),
            **{name: metrics.get(name) for name in METRICS},
        }
    return rows


def add_reference_differences(rows, reference_rows):
    for key, row in rows.items():
        reference = reference_rows.get(key)
        if reference is None:
            continue
        for metric in METRICS:
            current = row.get(metric)
            old = reference.get(metric)
            if current is not None and old is not None:
                row[f"delta_{metric}"] = current - old


def print_rows(rows):
    header = (
        "lambda  method                         goodput   backlog   delay_ms  "
        "active  toggles  min_local_dU"
    )
    print(header)
    print("-" * len(header))
    for key in sorted(rows, key=lambda item: (item[1], item[0])):
        row = rows[key]
        value = lambda name: row.get(name)
        print(
            f"{row['arrival_rate']:>6g}  {row['method']:<29} "
            f"{value('successful_packet_total_rate_pps') or 0:>7.3f}  "
            f"{value('average_total_queue_backlog_mbit') or 0:>8.3f}  "
            f"{value('average_packet_delay_ms') or 0:>8.2f}  "
            f"{value('average_active_links') or 0:>6.2f}  "
            f"{value('average_mp_accepted_toggles') or 0:>7.3f}  "
            f"{value('minimum_mp_local_utility_delta') or 0:>12.5f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory")
    parser.add_argument("--reference-directory", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = load_rows(args.result_directory)
    if args.reference_directory:
        add_reference_differences(
            rows, load_rows(args.reference_directory)
        )
    print_rows(rows)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(list(rows.values()), indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
