"""Compare mean-power and instantaneous-power link catalogs.

This audit does not change how datasets are generated.  It samples the same
topology and channel once, then applies the two catalog rules to that paired
sample.  The result tells us how often the two rules assign different link
identities under the paper configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from utils.project_setup import build_network_generator, load_config


def catalog_masks(generator, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return instantaneous, mean-power, and mean-SNR matrices for one sample."""
    torch.manual_seed(seed)
    positions = generator._init_positions()
    _, _, instantaneous_snr, large_scale = generator._channel_gain(positions)

    mean_power = generator.tx_power * large_scale.square()
    mean_snr = mean_power / generator.noise_power
    threshold = 10.0 ** (generator.snr_thres / 10.0)

    instantaneous = instantaneous_snr >= threshold
    mean_power_catalog = mean_snr >= threshold
    instantaneous.fill_diagonal_(False)
    mean_power_catalog.fill_diagonal_(False)
    return instantaneous, mean_power_catalog, mean_snr


def compare_catalog_masks(
    instantaneous: torch.Tensor,
    mean_power: torch.Tensor,
) -> dict[str, float | int]:
    """Summarize the difference between two Boolean catalog masks."""
    intersection = instantaneous & mean_power
    union = instantaneous | mean_power
    instantaneous_only = instantaneous & ~mean_power
    mean_only = mean_power & ~instantaneous

    union_count = int(union.sum().item())
    mean_count = int(mean_power.sum().item())
    symmetric_difference = int((instantaneous ^ mean_power).sum().item())
    return {
        "instantaneous_count": int(instantaneous.sum().item()),
        "mean_count": mean_count,
        "intersection_count": int(intersection.sum().item()),
        "union_count": union_count,
        "instantaneous_only_count": int(instantaneous_only.sum().item()),
        "mean_only_count": int(mean_only.sum().item()),
        "symmetric_difference_count": symmetric_difference,
        "jaccard": 1.0 if union_count == 0 else int(intersection.sum()) / union_count,
        "symmetric_difference_over_mean": (
            0.0 if mean_count == 0 else symmetric_difference / mean_count
        ),
    }


def summarize_samples(samples: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    """Report mean, standard deviation, minimum, and maximum for each metric."""
    summary = {}
    for name in samples[0]:
        values = np.asarray([sample[name] for sample in samples], dtype=float)
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary


def audit_network_size(config: dict, num_uav: int, seeds: list[int]) -> dict:
    """Run the paired catalog comparison for one UAV-network size."""
    size_config = dict(config)
    size_config["NUM_UAV"] = num_uav
    generator = build_network_generator(size_config)

    samples = []
    differing_mean_snr_db = []
    for seed in seeds:
        instantaneous, mean_power, mean_snr = catalog_masks(generator, seed)
        sample = compare_catalog_masks(instantaneous, mean_power)
        sample["seed"] = seed
        samples.append(sample)

        different = instantaneous ^ mean_power
        if different.any():
            differing_mean_snr_db.extend(
                (10.0 * torch.log10(mean_snr[different])).tolist()
            )

    result = {
        "num_uav": num_uav,
        "num_seeds": len(seeds),
        "summary": summarize_samples(samples),
        "samples": samples,
    }
    if differing_mean_snr_db:
        values = np.asarray(differing_mean_snr_db)
        result["differing_link_mean_snr_db"] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        }
    return result


def print_summary(results: list[dict]) -> None:
    print("N   mean links  inst. links  Jaccard  symmetric diff / mean")
    for result in results:
        summary = result["summary"]
        print(
            f"{result['num_uav']:<3d} "
            f"{summary['mean_count']['mean']:>10.2f} "
            f"{summary['instantaneous_count']['mean']:>11.2f} "
            f"{summary['jaccard']['mean']:>8.4f} "
            f"{summary['symmetric_difference_over_mean']['mean']:>21.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.yaml")
    parser.add_argument("--network-sizes", nargs="+", type=int, default=[20, 30, 40, 50])
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/validation/catalog_audit/mean_vs_instantaneous_ricianK5.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    results = [audit_network_size(config, num_uav, seeds) for num_uav in args.network_sizes]

    report = {
        "audit_version": "mean_vs_instantaneous_catalog_v1",
        "config": args.config,
        "channel_model": config.get("CHANNEL_MODEL"),
        "rician_k_db": config.get("RICIAN_K_DB"),
        "snr_threshold_db": config.get("SNR_THRES"),
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
