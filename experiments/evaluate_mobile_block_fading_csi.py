#!/usr/bin/env python3
"""Paper experiment: Direct-Link MarginalMP under mobile imperfect CSI.

The default experiment contains no learned CSI predictor.  ``--include-cg-csi``
adds the frozen optional branch without changing the scheduler or true-channel
service settlement.  Every method shares the same trajectory and arrivals.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.complex_geometry_csi import (  # noqa: E402
    load_cg_csi_predictor,
    predict_power_matrices,
)
from models.marginal_mp import DirectLinkMarginalMP  # noqa: E402
from simulator.mobile_block_fading import (  # noqa: E402
    generate_mobile_block_fading_trace,
    run_mobile_queue_simulation,
)
from utils.fixed_csi_benchmark import (  # noqa: E402
    arrival_trace_seed,
    generate_hotspot_arrivals,
)
from utils.project_setup import build_network_generator, load_config  # noqa: E402


DEFAULT_METHODS = ("latest", "stale", "large_scale", "quantized_4bit")


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
        "mean_csi_absolute_error_db",
        "packet_count_conservation_error",
        "packet_work_conservation_error",
    )
    return {key: metrics.get(key) for key in keep}


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        key = (
            row["speed_mps"], row["slots_per_block"], row["stale_blocks"],
            row["arrival_rate"], row["method"],
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
        speed, block, age, load, method = key
        label = (
            f"v={speed:g}|block={block}|age={age}|"
            f"Lambda={load:g}|method={method}"
        )
        output[label] = {
            metric: mean_ci([
                row[metric] for row in rows if row.get(metric) is not None
            ])
            for metric in metrics
            if any(row.get(metric) is not None for row in rows)
        }
    return output


def relative_to_latest(records):
    by_cell = defaultdict(dict)
    for row in records:
        cell = (
            row["speed_mps"], row["slots_per_block"], row["stale_blocks"],
            row["arrival_rate"], row["seed"],
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
            key = (*cell[:4], method)
            differences[key]["delivered_loss_fraction"].append(
                (latest["delivered_throughput_mbps"] - row["delivered_throughput_mbps"])
                / max(latest["delivered_throughput_mbps"], 1e-12)
            )
            differences[key]["backlog_increase_mbit"].append(
                row["average_total_queue_backlog_mbit"]
                - latest["average_total_queue_backlog_mbit"]
            )
    return {
        (
            f"v={key[0]:g}|block={key[1]}|age={key[2]}|"
            f"Lambda={key[3]:g}|method={key[4]}"
        ): {metric: mean_ci(values) for metric, values in metrics.items()}
        for key, metrics in differences.items()
    }


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
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--include-cg-csi", action="store_true")
    parser.add_argument(
        "--cg-csi-checkpoint",
        default="results/models/complex_geometry_csi_N20_ricianK5_20260718.pth",
    )
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
    methods = list(DEFAULT_METHODS)
    cg_model = cg_checkpoint = None
    if args.include_cg_csi:
        if args.num_uav != 20:
            raise ValueError("The retained CG-CSI checkpoint is frozen to N=20")
        cg_model, cg_checkpoint = load_cg_csi_predictor(
            args.cg_csi_checkpoint,
            expected_num_uav=args.num_uav,
            expected_rician_k_db=5.0,
        )
        methods.append("cg_csi")

    output = Path(args.output)
    progress = output.with_suffix(".progress.json")
    records = []
    if progress.exists():
        records = json.loads(progress.read_text(encoding="utf-8"))["records"]
    completed = {
        (
            row["speed_mps"], row["slots_per_block"], row["stale_blocks"],
            row["arrival_rate"], row["seed"], row["method"],
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
                        quantization_bits=(args.q_bits,),
                        rician_k_db=5.0,
                    )
                    predicted = None
                    if args.include_cg_csi:
                        predicted = predict_power_matrices(
                            cg_model,
                            cg_checkpoint,
                            trace,
                            stale_blocks=age,
                            speed_mps=speed,
                            minimum_distance=generator.min_distance,
                            carrier_frequency_hz=generator.carrier_f,
                            tx_power_w=generator.tx_power,
                        )
                    for load in csv_values(args.arrival_rates, float):
                        arrivals, profile = generate_hotspot_arrivals(
                            args.num_slots,
                            trace.num_links,
                            load,
                            arrival_seed=arrival_trace_seed(seed, load),
                            hotspot_seed=seed,
                        )
                        for method in methods:
                            key = (speed, block_slots, age, load, seed, method)
                            if key in completed:
                                continue
                            csi_mode = method
                            kwargs = {}
                            if method == "stale":
                                kwargs["stale_blocks"] = age
                            elif method == "quantized_4bit":
                                csi_mode = "quantized"
                            elif method == "cg_csi":
                                csi_mode = "predicted"
                                kwargs.update({
                                    "observed_power_by_block": predicted,
                                    "observed_csi_label": "cg_csi",
                                })
                            metrics = run_mobile_queue_simulation(
                                generator,
                                DirectLinkMarginalMP(rounds=args.rounds),
                                trace,
                                arrivals,
                                csi_mode=csi_mode,
                                q_bits=args.q_bits,
                                packet_size=int(config["PACKET_SIZE"]),
                                **kwargs,
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
                                f"seed={seed} Lambda={load:g} {method}: "
                                f"goodput={row['delivered_throughput_mbps']:.3f} "
                                f"backlog={row['average_total_queue_backlog_mbit']:.3f}",
                                flush=True,
                            )

    payload = {
        "experiment": "paper_mobile_imperfect_csi_v1",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "settings": vars(args),
        "primary_method": "Direct-Link MarginalMP without learned CSI prediction",
        "optional_branch": "CG-CSI" if args.include_cg_csi else None,
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
