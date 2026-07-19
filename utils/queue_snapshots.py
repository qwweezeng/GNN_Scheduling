"""Reproducible queue profiles used by paper snapshot experiments."""

import numpy as np
import torch


def heterogeneous_queue_snapshot(num_links, seed):
    """Return the historical reproducible bursty-hotspot queue profile."""
    rng = np.random.default_rng(int(seed))
    queues = rng.poisson(2.0, size=num_links).astype(np.float32)
    count = max(1, int(np.ceil(0.2 * num_links)))
    hotspots = rng.choice(num_links, size=count, replace=False)
    queues[hotspots] += rng.poisson(8.0, size=count).astype(np.float32)
    if not np.any(queues > 0):
        queues[rng.integers(0, num_links)] = 1.0
    return torch.from_numpy(queues)


def queue_snapshot(num_links, seed, profile):
    """Return one of the three queue profiles used in the paper.

    ``sparse`` tests mostly empty queues, ``heterogeneous`` adds a 20-percent
    hotspot population, and ``saturated`` keeps every link backlogged.
    """
    rng = np.random.default_rng(int(seed))
    profile = str(profile).lower()
    if profile == "sparse":
        queues = rng.poisson(0.35, size=num_links).astype(np.float32)
    elif profile == "heterogeneous":
        return heterogeneous_queue_snapshot(num_links, seed)
    elif profile == "saturated":
        queues = 1.0 + rng.poisson(8.0, size=num_links).astype(np.float32)
    else:
        raise ValueError(
            f"Unknown queue profile {profile!r}; expected sparse, "
            "heterogeneous, or saturated"
        )
    if not np.any(queues > 0):
        queues[rng.integers(0, num_links)] = 1.0
    return torch.from_numpy(queues)


__all__ = ["heterogeneous_queue_snapshot", "queue_snapshot"]
