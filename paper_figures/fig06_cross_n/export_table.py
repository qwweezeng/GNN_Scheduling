#!/usr/bin/env python3
"""Export the auditable small-to-large Fig. 6 table from raw cross-N results.

The output intentionally contains only train_N <= test_N cells and excludes the
mixed-N checkpoint.  Every relative-performance percentage is normalized by the
matched-scale checkpoint evaluated on the same test set.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RATE_KEY = "mean_aggregate_phy_rate_mbps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=HERE / "data/raw/cross_n_trainSize=300_seed0_epoch50.json",
        help="Raw cross-N JSON produced by the Fig. 6 evaluator.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "data/plotting/fig06_small_to_large_table.csv",
        help="Derived CSV for the manuscript table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text())
    try:
        source_path = args.input.resolve().relative_to(ROOT.resolve())
    except ValueError:
        source_path = args.input
    nodes = [int(node) for node in source["nodes"]]
    results = source["results"]

    missing = [str(node) for node in nodes if str(node) not in results]
    if missing:
        raise ValueError(f"Missing matched checkpoints for N={', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for train_n in nodes:
        for test_n in nodes:
            if train_n > test_n:
                continue
            trained_rate = results[str(train_n)][str(test_n)][RATE_KEY]
            matched_rate = results[str(test_n)][str(test_n)][RATE_KEY]
            if matched_rate <= 0:
                raise ValueError(f"Non-positive matched rate for test N={test_n}")
            rows.append(
                {
                    "train_num_uav": train_n,
                    "test_num_uav": test_n,
                    "sum_rate_mbps": f"{trained_rate:.3f}",
                    "matched_scale_sum_rate_mbps": f"{matched_rate:.3f}",
                    "relative_performance_percent": f"{100.0 * trained_rate / matched_rate:.2f}",
                    "raw_metric_key": RATE_KEY,
                    "source_json": source_path.as_posix(),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} small-to-large cells to {args.output}")


if __name__ == "__main__":
    main()
