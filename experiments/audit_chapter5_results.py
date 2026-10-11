#!/usr/bin/env python3
"""Audit the completed Chapter V Fig. 5 and Fig. 7 result files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROTOCOL = "conflict_graph_local_coordination_v1"
FIG5_METHODS = (
    "Random",
    "Centralized Greedy",
    "Weighted FPLinQ",
    "Direct-Link LocalMax",
    "GAT-Assisted LocalMax",
    "Direct-Link MarginalMP",
    "GAT-Assisted MarginalMP",
)
FIG5_RATES = (0.5, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5, 5.0)
FIG7_METHODS = (
    "latest",
    "large_scale",
    "quantized_2bit",
    "quantized_4bit",
    "quantized_6bit",
    "quantized_8bit",
    "quantized_10bit",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/paper/ch5_local_coordination_v1"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/validation/chapter5_final_audit"),
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def add_check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def audit_fig5(root: Path, manifest: dict, checks: list[dict]) -> dict:
    directory = root / "fig05_merged_N50_seed42/deadline_2000ms/N50"
    config_path = directory / "config.json"
    metrics_path = directory / "metrics.json"
    per_seed_path = directory / "per_seed_metrics.json"
    trajectory_path = directory / "trajectories.npz"
    config = load_json(config_path)
    metrics = load_json(metrics_path)
    per_seed = load_json(per_seed_path)

    expected_keys = {
        f"{method}|lambda={rate:g}"
        for method in FIG5_METHODS for rate in FIG5_RATES
    }
    add_check(checks, "fig05_aggregate_cells", set(metrics) == expected_keys,
              f"found={len(metrics)}, expected={len(expected_keys)}")
    add_check(checks, "fig05_per_seed_cells", len(per_seed) == len(expected_keys),
              f"found={len(per_seed)}, expected={len(expected_keys)}")
    with np.load(trajectory_path) as trajectories:
        trajectory_count = len(trajectories.files)
    add_check(checks, "fig05_trajectory_arrays", trajectory_count == 980,
              f"found={trajectory_count}, expected=980")
    add_check(checks, "fig05_protocol_version",
              config.get("marginal_protocol_version") == PROTOCOL,
              str(config.get("marginal_protocol_version")))
    add_check(checks, "fig05_all_values_finite", all_finite(metrics),
              "all aggregate scalar values are finite")

    mp_rows = {
        key: value for key, value in metrics.items() if "MarginalMP" in key
    }
    min_delta = min(row["minimum_mp_local_utility_delta"] for row in mp_rows.values())
    max_projection = max(row["average_hard_projection_rejections"] for row in mp_rows.values())
    add_check(checks, "fig05_mp_local_utility_nondecreasing", min_delta >= -1e-6,
              f"minimum recorded delta={min_delta:.6g}")
    add_check(checks, "fig05_mp_disjoint_neighborhoods",
              all(row["mp_all_neighborhoods_disjoint"] == 1.0 for row in mp_rows.values()),
              "all MP cells report disjoint accepted neighborhoods")
    add_check(checks, "fig05_mp_shared_uav_feasible",
              all(row["mp_all_slots_feasible"] == 1.0 for row in mp_rows.values()),
              "all MP cells report feasible slots")
    add_check(checks, "fig05_mp_no_post_projection", max_projection == 0.0,
              f"maximum mean projection rejections={max_projection:g}")

    hash_results = {}
    for field, path in (
        ("config_sha256", config_path),
        ("metrics_sha256", metrics_path),
        ("per_seed_metrics_sha256", per_seed_path),
        ("trajectories_sha256", trajectory_path),
    ):
        actual = sha256(path)
        expected = manifest["fig05"][field]
        hash_results[field] = actual
        add_check(checks, f"fig05_{field}", actual == expected,
                  f"actual={actual}, manifest={expected}")

    trends = {}
    for method in FIG5_METHODS:
        rows = [metrics[f"{method}|lambda={rate:g}"] for rate in FIG5_RATES]
        backlog = [row["average_total_queue_backlog_mbit"] for row in rows]
        goodput = [row["successful_goodput_mbps"] for row in rows]
        trends[method] = {
            "backlog_decrease_count": int(sum(b < a for a, b in zip(backlog, backlog[1:]))),
            "goodput_decrease_count": int(sum(b < a for a, b in zip(goodput, goodput[1:]))),
        }

    selected = {}
    for rate in (2.4, 3.5, 5.0):
        selected[f"lambda={rate:g}"] = {}
        for method in (
            "Direct-Link LocalMax", "GAT-Assisted LocalMax",
            "Direct-Link MarginalMP", "GAT-Assisted MarginalMP",
        ):
            row = metrics[f"{method}|lambda={rate:g}"]
            selected[f"lambda={rate:g}"][method] = {
                "backlog_mbit": row["average_total_queue_backlog_mbit"],
                "completed_delay_ms": row["average_packet_delay_ms"],
                "goodput_mbps": row["successful_goodput_mbps"],
                "delivery_ratio": row["successful_delivery_ratio"],
            }
    return {
        "directory": str(directory),
        "aggregate_cells": len(metrics),
        "trajectory_arrays": trajectory_count,
        "hashes": hash_results,
        "trend_diagnostics": trends,
        "selected_operating_points": selected,
    }


def audit_fig7(root: Path, manifest: dict, checks: list[dict]) -> dict:
    results = {}
    expected_methods = set(FIG7_METHODS)
    for num_uav in (20, 30, 40, 50):
        path = root / f"fig07_full/chapter5_N{num_uav}_v2_matched_csi.json"
        payload = load_json(path)
        records = payload.get("records", [])
        by_method = {row["method"]: row for row in records}
        add_check(checks, f"fig07_N{num_uav}_conditions",
                  set(by_method) == expected_methods,
                  f"found={sorted(by_method)}, expected={sorted(expected_methods)}")
        add_check(checks, f"fig07_N{num_uav}_protocol",
                  payload.get("marginal_protocol_version") == PROTOCOL,
                  str(payload.get("marginal_protocol_version")))
        add_check(checks, f"fig07_N{num_uav}_finite", all_finite(records),
                  "all record scalar values are finite")
        add_check(checks, f"fig07_N{num_uav}_feasible",
                  all(row["shared_uav_feasible_all_slots"] for row in records),
                  "all CSI conditions report feasible slots")
        add_check(checks, f"fig07_N{num_uav}_no_projection",
                  all(row["average_hard_projection_rejections"] == 0.0 for row in records),
                  "all CSI conditions report zero projection rejections")
        max_count_error = max(abs(row["packet_count_conservation_error"]) for row in records)
        max_work_error = max(abs(row["packet_work_conservation_error"]) for row in records)
        add_check(checks, f"fig07_N{num_uav}_packet_count_conservation",
                  max_count_error == 0.0, f"maximum error={max_count_error:g}")
        add_check(checks, f"fig07_N{num_uav}_packet_work_conservation",
                  max_work_error <= 0.01, f"maximum error={max_work_error:g}")
        latest_goodput = by_method["latest"]["delivered_throughput_mbps"]
        add_check(checks, f"fig07_N{num_uav}_latest_best_goodput",
                  latest_goodput == max(row["delivered_throughput_mbps"] for row in records),
                  f"latest={latest_goodput:g} Mbps")

        quantized = [by_method[f"quantized_{bits}bit"] for bits in (2, 4, 6, 8, 10)]
        errors = [row["mean_csi_absolute_error_db"] for row in quantized]
        error_monotone = all(right <= left + 1e-9 for left, right in zip(errors, errors[1:]))
        add_check(checks, f"fig07_N{num_uav}_quantization_error_monotone",
                  error_monotone, f"errors_db={[round(value, 4) for value in errors]}")
        actual_hash = sha256(path)
        expected_hash = manifest["fig07"][f"N{num_uav}_sha256"]
        add_check(checks, f"fig07_N{num_uav}_sha256", actual_hash == expected_hash,
                  f"actual={actual_hash}, manifest={expected_hash}")
        results[f"N{num_uav}"] = {
            "path": str(path),
            "sha256": actual_hash,
            "records": records,
            "quantized_goodput_mbps": {
                str(bits): by_method[f"quantized_{bits}bit"]["delivered_throughput_mbps"]
                for bits in (2, 4, 6, 8, 10)
            },
        }
    return results


def render_markdown(report: dict) -> str:
    failed = [item for item in report["checks"] if not item["passed"]]
    lines = [
        "# Chapter V Final Result Audit",
        "",
        f"Overall status: **{report['status']}**",
        f"Checks: {report['passed_checks']}/{report['total_checks']} passed",
        "",
        "## Failed checks",
        "",
    ]
    if failed:
        lines.extend(f"- {item['name']}: {item['detail']}" for item in failed)
    else:
        lines.append("- None")
    lines += ["", "## Fig. 5 selected operating points", ""]
    for load, methods in report["fig05"]["selected_operating_points"].items():
        lines.append(f"### {load}")
        lines.append("")
        lines.append("| Method | Backlog (Mbit) | Delay (ms) | Goodput (Mbps) | Delivery ratio |")
        lines.append("|---|---:|---:|---:|---:|")
        for method, values in methods.items():
            lines.append(
                f"| {method} | {values['backlog_mbit']:.4f} | "
                f"{values['completed_delay_ms']:.2f} | {values['goodput_mbps']:.4f} | "
                f"{values['delivery_ratio']:.4f} |"
            )
        lines.append("")
    lines += ["## Fig. 7 CSI summary", ""]
    for size, data in report["fig07"].items():
        latest = next(row for row in data["records"] if row["method"] == "latest")
        lines.append(
            f"- {size}: latest goodput={latest['delivered_throughput_mbps']:.4f} Mbps, "
            f"backlog={latest['average_total_queue_backlog_mbit']:.4f} Mbit; "
            f"quantized goodput={data['quantized_goodput_mbps']}"
        )
    lines += ["", "## All checks", ""]
    lines.extend(
        f"- [{'x' if item['passed'] else ' '}] {item['name']}: {item['detail']}"
        for item in report["checks"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(root / "RUN_MANIFEST.json")
    checks: list[dict] = []
    fig5 = audit_fig5(root, manifest, checks)
    fig7 = audit_fig7(root, manifest, checks)
    failed_checks = sum(not item["passed"] for item in checks)
    report = {
        "audit_version": "chapter5_final_result_audit_v1",
        "result_root": str(root),
        "status": "PASS" if failed_checks == 0 else "REVIEW_REQUIRED",
        "total_checks": len(checks),
        "passed_checks": len(checks) - failed_checks,
        "failed_checks": failed_checks,
        "checks": checks,
        "fig05": fig5,
        "fig07": fig7,
    }
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "passed_checks": report["passed_checks"],
        "total_checks": report["total_checks"],
        "json": str(json_path),
        "markdown": str(markdown_path),
    }, indent=2))
    if failed_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
