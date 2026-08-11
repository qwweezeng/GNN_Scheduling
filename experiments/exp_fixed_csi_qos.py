#!/usr/bin/env python3
"""Paper experiment: fixed/perfect-CSI queue QoS versus offered load.

Outputs paired Monte Carlo throughput, backlog, exact delivered-packet delay,
and queue-growth metrics for MarginalMP and its baselines.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.queue_schedulers import (
    CentralizedMWSGreedyScheduler,
    HardConflictLocalGreedyScheduler,
    RandomNonemptyScheduler,
    WeightedFPLinQScheduler,
)
from baseline.distgcn_adapter import DistGCNQueueScheduler
from models.marginal_mp import (
    DirectLinkMarginalMP,
    GATAssistedMarginalMP,
    MARGINAL_PROTOCOL_VERSION,
)
from models.initial_schedule_construction import (
    DirectLinkLocalMaxScheduler,
    DirectLinkJoinScheduler,
    GATAssistedLocalMaxScheduler,
)
from utils.fixed_csi_benchmark import (
    add_hotspot_metrics,
    aggregate_runs,
    build_fixed_csi_trace_cache,
    compact_run,
    load_trace_cache,
    run_scheduler,
    trace_item,
)
from utils.project_setup import (
    load_channel_gat,
    load_channel_gat_checkpoint,
    load_config,
    load_end_to_end_queue_gat,
    physics_tag,
    queue_checkpoint_path,
)


def comma_list(text, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def checkpoint_signature(path):
    if not path:
        return None
    stat = Path(path).stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def task_fingerprint(args, config, method, arrival_lambda, seed, channel_path, queue_path):
    payload = {
        "schema": "fixed_csi_queue_task_v1",
        "num_uav": int(args.num_uav),
        "num_slots": int(args.num_slots),
        "arrival_lambda": float(arrival_lambda),
        "seed": int(seed),
        "method": str(method),
        "rounds": int(args.rounds),
        "channel_model": str(config.get("CHANNEL_MODEL")),
        "rician_k_db": float(config.get("RICIAN_K_DB", 5.0)),
        "tx_power_dbm": float(config.get("TX_POWER", 10.0)),
        "csi_mode": str(config.get("CSI_MODE", "perfect")),
        "catalog_policy": "instantaneous_snr_snapshot",
        "delay_deadline_ms": args.delay_deadline_ms,
        "buffer_size_mbit": args.buffer_size_mbit,
        "channel_checkpoint": checkpoint_signature(channel_path),
        "end_to_end_queue_gat_checkpoint": checkpoint_signature(queue_path),
    }
    if method in {"Direct-Link MarginalMP", "GAT-Assisted MarginalMP"}:
        payload["marginal_protocol_version"] = MARGINAL_PROTOCOL_VERSION
        payload["scheduler_model_version"] = (
            "direct_mp_local_coordination_v1"
            if method == "Direct-Link MarginalMP"
            else "gat_mp_local_coordination_v1"
        )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def load_completed_task(path, fingerprint):
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("fingerprint") != fingerprint:
        return None
    return payload.get("metrics")


def save_completed_task(path, fingerprint, metadata, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "fingerprint": fingerprint,
        "metadata": metadata,
        "metrics": metrics,
    }, temp)
    os.replace(temp, path)


def method_factories(channel_model, queue_gat_model, include_mws, rounds):
    methods = {
        "Direct-Link MarginalMP": (
            lambda seed: DirectLinkMarginalMP(
                rounds=rounds
            ),
            False,
        ),
        "GAT-Assisted MarginalMP": (
            lambda seed: GATAssistedMarginalMP(
                channel_model, rounds=rounds
            ),
            False,
        ),
        "Hard-Conflict LGS": (
            lambda seed: HardConflictLocalGreedyScheduler(),
            False,
        ),
        "Direct-Link LocalMax": (
            lambda seed: DirectLinkLocalMaxScheduler(),
            False,
        ),
        "GAT-Assisted LocalMax": (
            lambda seed: GATAssistedLocalMaxScheduler(channel_model),
            False,
        ),
        "Direct-Link Join": (
            lambda seed: DirectLinkJoinScheduler(),
            False,
        ),
        "Random": (
            lambda seed: RandomNonemptyScheduler(0.05, seed),
            False,
        ),
        "Weighted FPLinQ": (
            lambda seed: WeightedFPLinQScheduler(),
            False,
        ),
        "DistGCN-LGS": (
            lambda seed: DistGCNQueueScheduler(),
            False,
        ),
    }
    if queue_gat_model is not None:
        methods["End-to-End GAT-Q"] = (lambda seed: queue_gat_model, True)
    if include_mws:
        methods["Centralized Greedy"] = (
            lambda seed: CentralizedMWSGreedyScheduler(),
            False,
        )
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--num_uav", type=int, default=20)
    parser.add_argument("--num_slots", type=int, default=10000)
    parser.add_argument(
        "--arrival_rates", default="0.5,1,2,2.2,2.4,2.6,2.8,3,3.2"
    )
    parser.add_argument("--mc_seeds", default="42,43,44,45,46")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--channel_model_num_uav", type=int, default=20)
    parser.add_argument(
        "--allow_cross_scale_gat", action="store_true",
        help="Allow an explicitly labeled train/evaluation UAV-count mismatch.",
    )
    parser.add_argument(
        "--channel_model", choices=("rayleigh", "rician"), default="rician",
        help="Fading family for both the fixed channel and the channel GAT.",
    )
    parser.add_argument("--rician_k_db", type=float, default=5.0)
    parser.add_argument(
        "--channel_gat_checkpoint", default="",
        help=(
            "Explicit audited channel-only GAT checkpoint. When omitted, "
            "the standard results/models path is used."
        ),
    )
    parser.add_argument(
        "--allow_train_test_power_mismatch", action="store_true",
        help=(
            "Label an intentional channel-GAT training/deployment power "
            "transfer. A mismatch otherwise fails closed."
        ),
    )
    parser.add_argument("--trace_cache", default="")
    parser.add_argument(
        "--end_to_end_gat_checkpoint", default="",
        help="Checkpoint used only when End-to-End GAT-Q is selected.",
    )
    parser.add_argument(
        "--delay_deadline_ms", type=float, default=None,
        help="Optional packet deadline. Default None keeps infinite waiting.",
    )
    parser.add_argument(
        "--buffer_size_mbit", type=float, default=None,
        help="Optional finite per-link buffer. Default None is infinite.",
    )
    parser.add_argument("--skip_mws", action="store_true")
    parser.add_argument("--methods", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
        help="GAT inference device; auto selects CUDA when available.",
    )
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[evaluate] GAT device={device}", flush=True)
    requested = (
        set(comma_list(args.methods, str)) if args.methods else set()
    )
    needs_gat = bool(
        {"GAT-Assisted MarginalMP", "GAT-Assisted LocalMax"}.intersection(requested)
    )
    needs_queue_gat = "End-to-End GAT-Q" in requested
    if (
        needs_gat
        and
        args.channel_model_num_uav != args.num_uav
        and not args.allow_cross_scale_gat
    ):
        raise ValueError(
            "GAT training scale must match the fixed-CSI evaluation scale: "
            f"got train N={args.channel_model_num_uav}, "
            f"eval N={args.num_uav}."
        )

    arrival_rates = comma_list(args.arrival_rates, float)
    mc_seeds = comma_list(args.mc_seeds, int)
    config = load_config(args.config)
    config.update({
        "NUM_UAV": args.num_uav,
        "TX_POWER": 10.0,
        "CSI_MODE": "perfect",
        "CHANNEL_MODEL": args.channel_model,
        "RICIAN_K_DB": args.rician_k_db,
    })
    if needs_gat:
        if args.channel_gat_checkpoint:
            channel_model, channel_path = load_channel_gat_checkpoint(
                config,
                args.channel_gat_checkpoint,
                args.channel_model_num_uav,
                device=device,
                allow_power_transfer=args.allow_train_test_power_mismatch,
            )
        else:
            channel_model, channel_path = load_channel_gat(
                config, args.channel_model_num_uav, device=device
            )
    else:
        channel_model, channel_path = None, None
    queue_checkpoint_for_fingerprint = (
        Path(args.end_to_end_gat_checkpoint)
        if args.end_to_end_gat_checkpoint
        else queue_checkpoint_path(config, args.num_uav)
    )
    if needs_queue_gat:
        queue_gat_model, queue_model_path = load_end_to_end_queue_gat(
            config, queue_checkpoint_for_fingerprint, device=device
        )
    else:
        queue_gat_model, queue_model_path = None, None
    # Centralized Greedy is retained at N=50 for the paper's like-for-like
    # comparison. It is a greedy scheduler, not an exhaustive oracle.
    include_mws = not args.skip_mws
    methods = method_factories(
        channel_model, queue_gat_model, include_mws, args.rounds
    )
    if args.methods:
        unknown = requested.difference(methods)
        if unknown:
            raise ValueError(
                f"Unknown methods={sorted(unknown)}; available={list(methods)}"
            )
        methods = {
            name: value for name, value in methods.items()
            if name in requested
        }
    else:
        # Keep the formal default readable: proposed method, one internal
        # structural baseline, one external distributed baseline, and the
        # centralized strong reference when its scale gate permits it.
        formal_default = (
            "Direct-Link MarginalMP",
            "Direct-Link LocalMax",
            "Direct-Link Join",
            "Hard-Conflict LGS",
            "Random",
            "Centralized Greedy",
        )
        methods = {
            name: methods[name] for name in formal_default if name in methods
        }

    traces = (
        load_trace_cache(
            args.trace_cache, config, args.num_slots,
            arrival_rates, mc_seeds,
        )
        if args.trace_cache else build_fixed_csi_trace_cache(
            config, args.num_slots, arrival_rates, mc_seeds
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or (
        f"results/paper/fixed_csi_qos/{timestamp}_N{args.num_uav}"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dir = output_dir / "tasks"
    aggregate = {}
    per_seed = {}
    trajectory_payload = {}
    for arrival_lambda in arrival_rates:
        print(f"\nLambda={arrival_lambda:g} packets/slot")
        runs_by_method = {name: [] for name in methods}
        for seed in mc_seeds:
            state, arrivals, profile = trace_item(
                traces, seed, arrival_lambda
            )
            for name, (factory, include_queue_feature) in methods.items():
                fingerprint, task_metadata = task_fingerprint(
                    args, config, name, arrival_lambda, seed,
                    channel_path, queue_checkpoint_for_fingerprint,
                )
                task_path = task_dir / f"{fingerprint}.pt"
                metrics = load_completed_task(task_path, fingerprint)
                reused = metrics is not None
                if not reused:
                    scheduler = factory(seed)
                    metrics = run_scheduler(
                        config,
                        state,
                        arrivals,
                        scheduler,
                        include_queue_feature=include_queue_feature,
                        track_packet_delay=True,
                        delay_deadline_ms=args.delay_deadline_ms,
                        buffer_size_mbit=args.buffer_size_mbit,
                        device=device,
                    )
                    add_hotspot_metrics(
                        metrics, profile, config.get("PACKET_SIZE", 12000)
                    )
                    save_completed_task(
                        task_path, fingerprint, task_metadata, metrics
                    )
                runs_by_method[name].append(metrics)
                per_seed[f"{name}|lambda={arrival_lambda:g}|seed={seed}"] = (
                    compact_run(metrics)
                )
                status = "reused" if reused else "completed"
                print(
                    f"  seed={seed:>3} {name:<28} [{status}] "
                    f"dep={metrics['departure_total_rate_pps']:.3f} "
                    f"goodput={metrics['successful_packet_total_rate_pps']:.3f} "
                    f"backlog={metrics['average_queue_backlog_mbit']:.4f}Mb "
                    f"delay={metrics['average_packet_delay_ms']}"
                )

        for name, runs in runs_by_method.items():
            summary, trajectories = aggregate_runs(runs)
            key = f"{name}|lambda={arrival_lambda:g}"
            aggregate[key] = summary
            safe = name.replace("-", "_").replace(" ", "_")
            lam = str(arrival_lambda).replace(".", "p")
            for metric, values in trajectories.items():
                trajectory_payload[f"{safe}__lambda_{lam}__{metric}"] = values

    metadata = {
        "experiment": "fixed_csi_queue_qos",
        "timestamp": timestamp,
        "physical_setting": {
            "num_uav": args.num_uav,
            "tx_power_dbm": 10.0,
            "csi_mode": "perfect",
            "channel": "fixed within each Monte Carlo run",
            "channel_model": config.get("CHANNEL_MODEL", "rayleigh"),
            "rician_k_db": config.get("RICIAN_K_DB"),
            "slot_duration_s": config.get("SLOT_DURATION", 0.001),
            "packet_size_bits": config.get("PACKET_SIZE", 12000),
            "delay_deadline_ms": args.delay_deadline_ms,
            "buffer_size_mbit_per_link": args.buffer_size_mbit,
        },
        "traffic": {
            "arrival_rates": arrival_rates,
            "mc_seeds": mc_seeds,
            "num_slots": args.num_slots,
            "model": "20% hotspot links carry 70% of total traffic",
        },
        "methods": list(methods),
        "channel_checkpoint": channel_path,
        "end_to_end_queue_gat_checkpoint": queue_model_path,
        "marginal_mp_rounds": args.rounds,
        "marginal_mp_initializer": "LocalMax (no Join)",
        "marginal_protocol_version": MARGINAL_PROTOCOL_VERSION,
        "catalog_policy": "instantaneous_snr_snapshot",
        "mws_note": (
            "Centralized full-information greedy reference; not an optimum "
            "or mathematical upper bound."
        ),
        "delay_note": (
            "average_packet_delay_ms uses exact FIFO completion tracking. "
            + (
                "Deadline/buffer failures are counted as loss and unfinished "
                "packets report end-of-run censoring."
                if args.delay_deadline_ms is not None
                else
                "No deadline or finite buffer is used; unfinished packets are "
                "right-censored, so delay plots require >=99% completion."
            )
        ),
        "figure_profile": (
            "deadline_qos" if args.delay_deadline_ms is not None
            else "no_deadline_core"
        ),
        "trace_cache": args.trace_cache or "generated_in_memory",
        "resume_schema": "fixed_csi_queue_task_v1",
        "resume_task_directory": str(task_dir),
        "topology_by_seed": traces.get("metadata", {}).get(
            "topology_by_seed", {}
        ),
        "physics_tag": physics_tag(config),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(json_ready(metadata), file, indent=2, ensure_ascii=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(json_ready(aggregate), file, indent=2, ensure_ascii=False)
    with open(
        output_dir / "per_seed_metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(json_ready(per_seed), file, indent=2, ensure_ascii=False)
    np.savez_compressed(output_dir / "trajectories.npz", **trajectory_payload)
    print(f"\nSaved formal benchmark inputs/results to {output_dir}")


if __name__ == "__main__":
    main()
