"""Shared mathematical objectives for queue-aware link scheduling."""

import torch

from models.model import calculate_achievable_rate


def mws_objective(schedule, queues, state):
    """Return the instantaneous max-weight utility ``sum_i Q_i R_i``.

    This is a metric/objective, not a scheduling algorithm.  Baseline solvers
    that optimize this quantity live under ``baseline/``.
    """
    rates = calculate_achievable_rate(
        torch.as_tensor(schedule).reshape(-1, 1), state, binary=True
    )
    queues = torch.as_tensor(
        queues, dtype=rates.dtype, device=rates.device
    ).reshape(-1)
    return torch.sum(queues * rates.reshape(-1)).item()
