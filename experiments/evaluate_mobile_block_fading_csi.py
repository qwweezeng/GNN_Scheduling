#!/usr/bin/env python3
"""Paper experiment: paired schedulers under mobile imperfect CSI.

The default experiment contains no learned CSI predictor.  ``--include-cg-csi``
adds the frozen optional branch without changing the scheduler or true-channel
service settlement.  Every method shares the same trajectory and arrivals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.queue_schedulers import WeightedFPLinQScheduler  # noqa: E402
from models.marginal_mp import (  # noqa: E402
    DirectLinkMarginalMP,
    GATAssistedMarginalMP,
)
from simulator.mobile_block_fading import (  # noqa: E402
    generate_mobile_block_fading_trace,
    run_mobile_queue_simulation,
)
from utils.fixed_csi_benchmark import (  # noqa: E402
    arrival_trace_seed,
    generate_hotspot_arrivals,
)
from utils.artifacts import expected_dataset_metadata  # noqa: E402
from utils.project_setup import (  # noqa: E402
    build_network_generator,
    load_channel_gat,
    load_config,
)


DEFAULT_METHODS = ("latest", "stale", "large_scale", "quantized_4bit")
DEFAULT_SCHEDULERS = ("gat_assisted", "direct_link")


def csv_values(text, cast):
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def mean_ci(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "ci95": float(1.96 * array.std(ddof=1) / np.sqrt(array.size))
        if array.size > 1 else 0.0,
    }


def compact_metrics(metrics):
    keep = (
        "delivered_throughput_mbps",
        "physical_service_mbps",
        "successful_delivery_ratio",
        "unfinished_packet_fraction",
        "average_total_queue_backlog_mbit",
        "average_packet_delay_ms",
        "p95_packet_delay_ms",
        "average_active_links",
        "average_hard_projection_rejections",
        "shared_uav_feasible_all_slots",
        "mean_csi_absolute_error_db",
        "packet_count_conservation_error",
        "packet_work_conservation_error",
    )
    return {key: metrics.get(key) for key in keep}


def aggregate(records):
    legacy_keys = all("scheduler_variant" not in row for row in records)
    grouped = defaultdict(list)
    for row in records:
        key = (
            row["speed_mps"], row["slots_per_block"], row["stale_blocks"],
            row["arrival_rate"], row.get("scheduler_variant", "direct_link"), row["method"],
        )
        grouped[key].append(row)
    output = {}
    metrics = (
        "delivered_throughput_mbps",
        "physical_service_mbps",
        "successful_delivery_ratio",
        "unfinished_packet_fraction",
        "average_total_queue_backlog_mbit",
        "average_packet_delay_ms",
        "p95_packet_delay_ms",
        "average_active_links",
        "mean_csi_absolute_error_db",
    )
    for key, rows in grouped.items():
        speed, block, age, load, scheduler_variant, method = key
        label = f"v={speed:g}|block={block}|age={age}|Lambda={load:g}|"
        if not legacy_keys:
            label += f"scheduler={scheduler_variant}|"
        label += f"method={method}"
        output[label] = {
            metric: mean_ci([
                row[metric] for row in rows if row.get(metric) is not None
            ])
            for metric in metrics
            if any(row.get(metric) is not None for row in rows)
        }
    return output


def relative_to_latest(records):
    legacy_keys = all("scheduler_variant" not in row for row in records)
    by_cell = defaultdict(dict)
    for row in records:
        cell = (
            row["speed_mps"], row["slots_per_block"], row["stale_blocks"],
            row["arrival_rate"], row["seed"], row.get("scheduler_variant", "direct_link"),
        )
        by_cell[cell][row["method"]] = row
    differences = defaultdict(lambda: defaultdict(list))
    for cell, methods in by_cell.items():
        if "latest" not in methods:
            continue
        latest = methods["latest"]
        for method, row in methods.items():
            if method == "latest":
                continue
            key = (*cell[:4], cell[5], method)
            differences[key]["delivered_loss_fraction"].append(
                (latest["delivered_throughput_mbps"] - row["delivered_throughput_mbps"])
                / max(latest["delivered_throughput_mbps"], 1e-12)
            )
            differences[key]["backlog_increase_mbit"].append(
                row["average_total_queue_backlog_mbit"]
                - latest["average_total_queue_backlog_mbit"]
            )
    output = {}
    for key, metrics in differences.items():
        label = f"v={key[0]:g}|block={key[1]}|age={key[2]}|Lambda={key[3]:g}|"
        if not legacy_keys:
            label += f"scheduler={key[4]}|"
        label += f"method={key[5]}"
        output[label] = {metric: mean_ci(values) for metric, values in metrics.items()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num-uav", type=int, default=20)
    parser.add_argument("--num-slots", type=int, default=5000)
    parser.add_argument("--arrival-rates", default="1,2,2.2,2.4,3,4")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--speeds-mps", default="0,5,10,20")
    parser.add_argument("--slots-per-block", default="5")
    parser.add_argument("--stale-blocks", default="1")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument(
        "--csi-methods", default=",".join(DEFAULT_METHODS),
        help="latest,stale,large_scale,quantized_<B>bit",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument(
        "--scheduler-variants", default=",".join(DEFAULT_SCHEDULERS),
        help="gat_assisted,direct_link; optionally add weighted_fplinq",
    )
    parser.add_argument("--fplinq-iterations", type=int, default=100)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps for Proposed-GAT block inference")
    parser.add_argument(
        "--output", default="results/paper/mobile_csi/marginal_mp_mobile_csi.json"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CHANNEL_MODEL": "rician",
        "RICIAN_K_DB": 5.0,
        "CSI_MODE": "perfect",
    })
    generator = build_network_generator(config)
    methods = csv_values(args.csi_methods, str)
    def _valid_csi_method(name):
        return name in {"latest", "stale", "large_scale"} or (
            name.startswith("quantized_") and name.endswith("bit")
            and name[len("quantized_"):-3].isdigit()
        )
    unknown_csi = {name for name in methods if not _valid_csi_method(name)}
    if unknown_csi:
        raise ValueError(f"unknown CSI methods: {sorted(unknown_csi)}")
    scheduler_variants = csv_values(args.scheduler_variants, str)
    allowed_schedulers = {"gat_assisted", "direct_link", "weighted_fplinq"}
    unknown_schedulers = set(scheduler_variants) - allowed_schedulers
    if unknown_schedulers:
        raise ValueError(f"unknown scheduler variants: {sorted(unknown_schedulers)}")
    quantization_bits = tuple(sorted({
        int(name[len("quantized_"):-3])
        for name in methods if name.startswith("quantized_")
    })) or (int(args.q_bits),)

    channel_model = channel_checkpoint = checkpoint_fingerprint = None
    if "gat_assisted" in scheduler_variants:
        canonical = expected_dataset_metadata(
            config, args.num_uav,
            {"train": 10086, "val": 10087, "test": 10088},
            {"train": 500, "val": 128, "test": 128},
        )
        channel_model, channel_checkpoint = load_channel_gat(
            config, args.num_uav, device=args.device,
            expected_dataset_fingerprint=canonical["fingerprint"],
        )
        checkpoint_payload = torch.load(
            channel_checkpoint, map_location="cpu", weights_only=False
        )
        checkpoint_fingerprint = checkpoint_payload.get("dataset_fingerprint")

    def make_scheduler(variant):
        if variant == "gat_assisted":
            return GATAssistedMarginalMP(channel_model, rounds=args.rounds).eval()
        if variant == "direct_link":
            return DirectLinkMarginalMP(rounds=args.rounds).eval()
        if variant == "weighted_fplinq":
            return WeightedFPLinQScheduler(iterations=args.fplinq_iterations).eval()
        raise AssertionError(variant)
    output = Path(args.output)
    progress = output.with_suffix(".progress.json")
    records = []
    if progress.exists():
        records = json.loads(progress.read_text(encoding="utf-8"))["records"]
    completed = {
        (
            row["speed_mps"], row["slots_per_block"], row["stale_blocks"],
            row["arrival_rate"], row["seed"], row.get("scheduler_variant", "direct_link"), row["method"],
        )
        for row in records
    }

    for speed in csv_values(args.speeds_mps, float):
        for block_slots in csv_values(args.slots_per_block, int):
            for age in csv_values(args.stale_blocks, int):
                for seed in csv_values(args.seeds, int):
                    trace = generate_mobile_block_fading_trace(
                        generator,
                        num_slots=args.num_slots,
                        slots_per_block=block_slots,
                        slot_duration=float(config["SLOT_DURATION"]),
                        speed_mps=speed,
                        seed=seed,
                        quantization_bits=quantization_bits,
                        rician_k_db=5.0,
                    )
                    for load in csv_values(args.arrival_rates, float):
                        arrivals, profile = generate_hotspot_arrivals(
                            args.num_slots,
                            trace.num_links,
                            load,
                            arrival_seed=arrival_trace_seed(seed, load),
                            hotspot_seed=seed,
                        )
                        for scheduler_variant in scheduler_variants:
                            for method in methods:
                                key = (speed, block_slots, age, load, seed, scheduler_variant, method)
                                if key in completed:
                                    continue
                                csi_mode = method
                                kwargs = {}
                                if method == "stale":
                                    kwargs["stale_blocks"] = age
                                elif method.startswith("quantized_"):
                                    csi_mode = "quantized"
                                    kwargs["q_bits"] = int(method[len("quantized_"):-3])
                                metrics = run_mobile_queue_simulation(
                                    generator, make_scheduler(scheduler_variant), trace, arrivals,
                                    csi_mode=csi_mode, q_bits=kwargs.pop("q_bits", args.q_bits),
                                    packet_size=int(config["PACKET_SIZE"]), **kwargs,
                                )
                                row = {
                                    "speed_mps": speed,
                                    "slots_per_block": block_slots,
                                    "stale_blocks": age,
                                    "csi_age_ms": (
                                        age * block_slots
                                        * float(config["SLOT_DURATION"]) * 1000.0
                                    ),
                                    "arrival_rate": load,
                                    "seed": seed,
                                    "scheduler_variant": scheduler_variant,
                                    "method": method,
                                    "arrival_profile": profile["arrival_profile"],
                                    **compact_metrics(metrics),
                                }
                                records.append(row)
                                progress.parent.mkdir(parents=True, exist_ok=True)
                                progress.write_text(
                                    json.dumps({"settings": vars(args), "records": records}, indent=2),
                                    encoding="utf-8",
                                )
                                print(
                                    f"v={speed:g} block={block_slots} age={age} "
                                    f"seed={seed} Lambda={load:g} "
                                    f"{scheduler_variant}/{method}: "
                                    f"goodput={row['delivered_throughput_mbps']:.3f} "
                                    f"backlog={row['average_total_queue_backlog_mbit']:.3f}",
                                    flush=True,
                                )

    payload = {
        "experiment": "paper_mobile_imperfect_csi_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "frozen_git_revision": "ad2ee67d76e05effa4953db18ab677502ddd3ba1",
        "settings": vars(args),
        "config_snapshot": config,
        "scheduler_variants": scheduler_variants,
        "channel_gat_checkpoint": channel_checkpoint,
        "channel_gat_checkpoint_sha256": (
            hashlib.sha256(Path(channel_checkpoint).read_bytes()).hexdigest()
            if channel_checkpoint is not None else None
        ),
        "channel_gat_dataset_fingerprint": checkpoint_fingerprint,
        "fairness": (
            "All CSI modes share the same true Rician trajectory and arrivals; "
            "only scheduler-observed CSI changes; service uses current true CSI."
        ),
        "summary": aggregate(records),
        "relative_to_latest": relative_to_latest(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
