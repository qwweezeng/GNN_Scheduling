"""Hard single-radio/half-duplex feasibility for link schedules.

Every active directed link consumes both endpoint UAVs.  A feasible action is
therefore a matching on the undirected endpoint graph: no UAV may appear in
more than one active link, regardless of whether it is transmitter or receiver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _flat_numpy(values: Any, *, dtype=None) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype).reshape(-1)


def shared_uav_action_audit(action: Any, links: Any) -> dict[str, Any]:
    """Audit a binary link action under the hard shared-UAV constraint."""
    values = _flat_numpy(action)
    endpoints = np.asarray(
        links.detach().cpu().numpy() if isinstance(links, torch.Tensor) else links,
        dtype=np.int64,
    )
    if endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("links must have shape [num_links, 2]")
    if values.size != endpoints.shape[0]:
        raise ValueError(
            f"action has {values.size} links, expected {endpoints.shape[0]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("action must contain only finite values")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-6):
        raise ValueError("hard-feasibility audit requires a binary action")
    if not np.all((rounded == 0) | (rounded == 1)):
        raise ValueError("hard-feasibility audit requires a binary action")

    active = np.flatnonzero(rounded.astype(bool))
    if endpoints.size and np.any(endpoints < 0):
        raise ValueError("links contains a negative UAV endpoint")
    num_uav = int(endpoints.max()) + 1 if endpoints.size else 0
    counts = np.zeros(num_uav, dtype=np.int64)
    if active.size:
        np.add.at(counts, endpoints[active, 0], 1)
        np.add.at(counts, endpoints[active, 1], 1)
    violating = np.flatnonzero(counts > 1)
    return {
        "binary": True,
        "active_links": int(active.size),
        "active_fraction": float(active.size / max(1, len(values))),
        "feasible": bool(violating.size == 0),
        "violating_uavs": violating.tolist(),
        "num_violating_uavs": int(violating.size),
        "endpoint_excess_count": int(np.maximum(counts - 1, 0).sum()),
        "max_active_links_per_uav": int(counts.max()) if counts.size else 0,
    }


def project_shared_uav_feasible(
    action: Any,
    links: Any,
    *,
    priorities: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Greedily project a candidate action to a deterministic matching.

    Candidate links are visited by descending ``priorities``; equal priorities
    use the stable project link index.  The function returns a float32 tensor
    with the same one-dimensional/column shape convention as the input.
    """
    original_shape = tuple(action.shape) if hasattr(action, "shape") else None
    values = _flat_numpy(action)
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-6) or not np.all(
        (rounded == 0) | (rounded == 1)
    ):
        raise ValueError("hard-feasibility projection requires a binary candidate")
    endpoints = np.asarray(
        links.detach().cpu().numpy() if isinstance(links, torch.Tensor) else links,
        dtype=np.int64,
    )
    before = shared_uav_action_audit(rounded, endpoints)
    if priorities is None:
        scores = rounded.astype(np.float64)
    else:
        scores = _flat_numpy(priorities, dtype=np.float64)
        if scores.size != values.size or not np.isfinite(scores).all():
            raise ValueError("priorities must be finite and match the link count")

    candidates = np.flatnonzero(rounded.astype(bool))
    order = sorted(candidates.tolist(), key=lambda index: (-scores[index], index))
    accepted = np.zeros(values.size, dtype=np.float32)
    occupied: set[int] = set()
    rejected = []
    for index in order:
        tx, rx = (int(v) for v in endpoints[index])
        if tx in occupied or rx in occupied:
            rejected.append(index)
            continue
        accepted[index] = 1.0
        occupied.add(tx)
        occupied.add(rx)

    after = shared_uav_action_audit(accepted, endpoints)
    if not after["feasible"]:
        raise RuntimeError("internal error: shared-UAV projection is infeasible")
    result = torch.from_numpy(accepted)
    if original_shape and len(original_shape) == 2 and original_shape[1] == 1:
        result = result.unsqueeze(1)
    metadata = {
        "constraint": "single_radio_half_duplex_matching",
        "tie_break": "descending_priority_then_project_link_index",
        "candidate": before,
        "projected": after,
        "rejected_project_link_indices": rejected,
    }
    return result, metadata


def decode_endpoint_mutual_matching(
    action: Any,
    links: Any,
    *,
    priorities: Any,
    max_rounds: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode link scores by local endpoint proposals into a matching.

    In each round every currently free UAV proposes its highest-priority
    incident candidate link. A link is accepted only when both endpoints
    propose that same link. Accepted endpoints then leave later rounds. This
    is a local handshake decoder: it guarantees single-radio feasibility
    without adding a hard-edge indicator to the GAT input.
    """
    if max_rounds is not None and int(max_rounds) < 1:
        raise ValueError("max_rounds must be positive or None")
    original_shape = tuple(action.shape) if hasattr(action, "shape") else None
    values = _flat_numpy(action)
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-6) or not np.all(
        (rounded == 0) | (rounded == 1)
    ):
        raise ValueError("endpoint-mutual decoding requires a binary candidate")
    endpoints = np.asarray(
        links.detach().cpu().numpy() if isinstance(links, torch.Tensor) else links,
        dtype=np.int64,
    )
    before = shared_uav_action_audit(rounded, endpoints)
    scores = _flat_numpy(priorities, dtype=np.float64)
    if scores.size != values.size or not np.isfinite(scores).all():
        raise ValueError("priorities must be finite and match the link count")

    accepted = np.zeros(values.size, dtype=np.float32)
    occupied: set[int] = set()
    remaining = set(np.flatnonzero(rounded.astype(bool)).tolist())
    round_records = []
    round_index = 0
    while remaining and (max_rounds is None or round_index < int(max_rounds)):
        eligible = [
            index for index in remaining
            if int(endpoints[index, 0]) not in occupied
            and int(endpoints[index, 1]) not in occupied
        ]
        if not eligible:
            break
        proposal: dict[int, int] = {}
        for index in eligible:
            for endpoint in (int(endpoints[index, 0]), int(endpoints[index, 1])):
                previous = proposal.get(endpoint)
                if previous is None or (-scores[index], index) < (
                    -scores[previous], previous
                ):
                    proposal[endpoint] = index
        chosen = [
            index for index in eligible
            if proposal[int(endpoints[index, 0])] == index
            and proposal[int(endpoints[index, 1])] == index
        ]
        if not chosen:
            raise RuntimeError("endpoint-mutual decoder made no progress")
        for index in chosen:
            accepted[index] = 1.0
            occupied.update(int(value) for value in endpoints[index])
        removed = {
            index for index in remaining
            if int(endpoints[index, 0]) in occupied
            or int(endpoints[index, 1]) in occupied
        }
        remaining.difference_update(removed)
        round_records.append({
            "round": round_index + 1,
            "accepted_project_link_indices": sorted(chosen),
            "num_accepted": len(chosen),
        })
        round_index += 1

    after = shared_uav_action_audit(accepted, endpoints)
    if not after["feasible"]:
        raise RuntimeError("internal error: endpoint-mutual action is infeasible")
    result = torch.from_numpy(accepted)
    if original_shape and len(original_shape) == 2 and original_shape[1] == 1:
        result = result.unsqueeze(1)
    return result, {
        "constraint": "single_radio_half_duplex_matching",
        "decoder": "endpoint_mutual_highest_priority",
        "tie_break": "higher_priority_then_lower_project_link_index",
        "max_rounds": max_rounds,
        "rounds_executed": len(round_records),
        "round_records": round_records,
        "candidate": before,
        "projected": after,
        "unresolved_candidate_link_indices": sorted(remaining),
    }


__all__ = [
    "decode_endpoint_mutual_matching",
    "project_shared_uav_feasible",
    "shared_uav_action_audit",
]
