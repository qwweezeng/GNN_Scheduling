"""Small shared objective used only by queue-scheduler unit tests.

The dynamic-paper figures do not report this quantity as a physical metric.
It remains here because MarginalMP evaluates local schedule changes with the
instantaneous max-weight diagnostic on a fixed ``(Q, H)`` snapshot.
"""

from __future__ import annotations

import torch

from models.model import calculate_achievable_rate


def mws_objective(schedule, queues, state) -> float:
    """Return the fixed-snapshot diagnostic :math:`\\sum_i Q_i R_i`."""
    rates = calculate_achievable_rate(
        torch.as_tensor(schedule).reshape(-1, 1), state, binary=True
    )
    queue_tensor = torch.as_tensor(
        queues, dtype=rates.dtype, device=rates.device
    ).reshape(-1)
    return torch.sum(queue_tensor * rates.reshape(-1)).item()
