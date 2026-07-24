"""Physics-to-MWIS adapter for the Zhao et al. DistGCN baseline.

The adapter also restores the retained TensorFlow checkpoint in its isolated
legacy environment and maps the resulting GCN scores back to project links.
It does not claim that the adapted MWIS objective is equivalent to the
original directed physical-interference objective.
"""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


DEFAULT_DISTGCN_CHECKPOINT = Path(__file__).resolve().parent / (
    "distgcn-main/model/"
    "result_DQNBA_deep_ld1_c32_l3_cheb1_diver1_mwis_dqn/model.ckpt"
)
_DISTGCN_EMBEDDING_CACHE = {}


@dataclass(frozen=True)
class LinkIndexRoundTrip:
    """Reversible project-link to third-party-node index mapping."""

    project_link_count: int
    third_party_to_project: np.ndarray

    @classmethod
    def identity(cls, project_link_count):
        count = int(project_link_count)
        if count < 0:
            raise ValueError("project_link_count must be nonnegative")
        return cls(count, np.arange(count, dtype=np.int64))

    @classmethod
    def from_project_indices(cls, project_link_count, project_indices):
        count = int(project_link_count)
        indices = np.asarray(project_indices, dtype=np.int64).reshape(-1)
        if np.unique(indices).size != indices.size:
            raise ValueError("project_indices must be unique")
        if np.any(indices < 0) or np.any(indices >= count):
            raise IndexError("project_indices contains an invalid link index")
        return cls(count, indices)

    def project_to_third_party(self, values):
        array = np.asarray(values)
        if array.shape[0] != self.project_link_count:
            raise ValueError(
                f"expected {self.project_link_count} project values, got {array.shape}"
            )
        return array[self.third_party_to_project]

    def third_party_to_project_action(self, action):
        action = np.asarray(action).reshape(-1)
        if action.size != self.third_party_to_project.size:
            raise ValueError(
                f"expected {self.third_party_to_project.size} third-party actions, "
                f"got {action.size}"
            )
        if not np.all((action == 0) | (action == 1)):
            raise ValueError("DistGCN action must be binary")
        project = np.zeros(self.project_link_count, dtype=np.float32)
        project[self.third_party_to_project] = action.astype(np.float32)
        return project

    def metadata(self):
        return {
            "mapping": "explicit_third_party_node_to_project_link_v1",
            "project_link_count": int(self.project_link_count),
            "third_party_node_count": int(self.third_party_to_project.size),
            "third_party_to_project": self.third_party_to_project.tolist(),
        }


def build_hardened_mwis_instance(
    state,
    queues,
    inr_threshold_db=0.0,
    enforce_shared_uav=True,
    graph_source="inr_threshold",
):
    """Return symmetric hard adjacency and queue-weighted solo-link values.

    A directed physical relation ``j -> i`` is hardened when the interference
    to link i's receiver is at least ``inr_threshold_db`` relative to noise.
    The final MWIS adjacency is the OR of both directions, as required by an
    undirected independent-set solver.  Links sharing any UAV are always made
    mutually exclusive when ``enforce_shared_uav`` is true.
    """
    # PyTorch belongs to the main simulator, not the isolated DistGCN legacy
    # environment.  Import it only for physics-to-MWIS construction so the
    # NumPy-only link-index bridge remains usable in that legacy environment.
    import torch

    queues = torch.as_tensor(queues, dtype=torch.float32).reshape(-1).cpu()
    links = state.links.detach().cpu()
    mode = getattr(state, "csi_mode", "perfect")
    if mode == "large_scale":
        signal_power = getattr(state, "signal_power_large_scale", None)
    elif mode == "quantized":
        signal_power = getattr(state, "signal_power_quant", None)
    else:
        signal_power = state.signal_power
    if signal_power is None:
        signal_power = state.signal_power
    signal_power = signal_power.detach().cpu().float()

    gain = signal_power[links[:, 0, None], links[None, :, 1]]
    direct = signal_power[links[:, 0], links[:, 1]]
    if graph_source == "project_conflict":
        adjacency = torch.zeros(
            (links.size(0), links.size(0)), dtype=torch.bool
        )
        source, target = state.edge_index.detach().cpu()
        adjacency[source, target] = True
        adjacency = adjacency | adjacency.T
    elif graph_source == "inr_threshold":
        threshold_linear = 10.0 ** (float(inr_threshold_db) / 10.0)
        directed = gain >= float(state.noise_power) * threshold_linear
        directed.fill_diagonal_(False)
        adjacency = directed | directed.T
    else:
        raise ValueError(
            "graph_source must be 'project_conflict' or 'inr_threshold'"
        )

    if enforce_shared_uav:
        endpoints = (
            links[:, :, None, None] == links[None, None, :, :]
        )
        adjacency |= endpoints.any(dim=1).any(dim=-1)
        adjacency.fill_diagonal_(False)

    solo_rate = torch.log2(
        1.0
        + direct / float(state.noise_power)
        / float(getattr(state, "snr_gap", 1.0))
    )
    node_weights = queues.clamp_min(0.0) * solo_rate
    metadata = {
        "adapter": "directed_physical_to_symmetric_hard_mwis_v1",
        "graph_source": str(graph_source),
        "inr_threshold_db": float(inr_threshold_db),
        "enforce_shared_uav": bool(enforce_shared_uav),
        "num_nodes": int(links.size(0)),
        "num_undirected_edges": int(torch.triu(adjacency, diagonal=1).sum()),
        "warning": (
            "MWIS forbids every hardened pair and therefore cannot represent "
            "weak-interference concurrency or direction-dependent loss."
        ),
    }
    return adjacency, node_weights, metadata


def to_distgcn_numpy(adjacency, node_weights, *, project_indices=None):
    """Convert one hardened instance while preserving link identity."""
    import torch

    adjacency = torch.as_tensor(adjacency, dtype=torch.bool).cpu()
    weights = torch.as_tensor(node_weights, dtype=torch.float32).reshape(-1).cpu()
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if adjacency.shape[0] != weights.numel():
        raise ValueError("adjacency and node_weights size mismatch")
    if not torch.equal(adjacency, adjacency.T) or torch.diagonal(adjacency).any():
        raise ValueError("DistGCN requires a symmetric loop-free hard graph")
    count = weights.numel()
    mapping = (
        LinkIndexRoundTrip.identity(count)
        if project_indices is None
        else LinkIndexRoundTrip.from_project_indices(count, project_indices)
    )
    selected = torch.as_tensor(mapping.third_party_to_project, dtype=torch.long)
    selected_adjacency = adjacency[selected][:, selected]
    selected_weights = weights[selected]
    return (
        selected_adjacency.numpy().astype(np.float32),
        selected_weights.numpy().astype(np.float32).reshape(-1, 1),
        mapping,
    )


def run_distgcn_checkpoint_isolated(
    adjacency,
    weights,
    *,
    checkpoint=DEFAULT_DISTGCN_CHECKPOINT,
    conda_env="gnn-distgcn-legacy",
):
    """Restore the retained GCN and return its scalar node embeddings."""
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda is required for the isolated DistGCN worker")
    worker = Path(__file__).with_name("distgcn_worker.py").resolve()
    checkpoint = Path(checkpoint).resolve()
    with tempfile.TemporaryDirectory(prefix="distgcn-") as directory:
        directory = Path(directory)
        input_path = directory / "input.npz"
        output_path = directory / "output.npz"
        np.savez_compressed(
            input_path,
            adjacency=np.asarray(adjacency, dtype=np.float32),
            weights=np.asarray(weights, dtype=np.float32).reshape(1, -1),
        )
        environment = os.environ.copy()
        environment.update({
            "MPLCONFIGDIR": str(directory / "matplotlib"),
            "TF_CPP_MIN_LOG_LEVEL": "3",
        })
        completed = subprocess.run(
            [
                conda, "run", "--no-capture-output", "-n", conda_env,
                "python", str(worker), "--checkpoint", str(checkpoint),
                "--input", str(input_path), "--output", str(output_path),
            ],
            cwd=str(worker.parents[1]),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"isolated DistGCN inference failed: {detail}")
        payload = np.load(output_path, allow_pickle=False)
        scores = payload["scores"][0].astype(np.float32, copy=False)
    if scores.shape != np.asarray(weights).reshape(-1).shape:
        raise RuntimeError("DistGCN returned a mis-shaped embedding")
    if not np.isfinite(scores).all():
        raise RuntimeError("DistGCN returned non-finite embeddings")
    return scores


def distgcn_local_greedy(adjacency, priorities, valid=None):
    """Exact deterministic LGS post-processing used by the source baseline."""
    adjacency = np.asarray(adjacency, dtype=bool)
    priorities = np.asarray(priorities, dtype=float).reshape(-1)
    if valid is None:
        valid = np.ones(len(priorities), dtype=bool)
    remaining = set(np.flatnonzero(np.asarray(valid, dtype=bool)).tolist())
    scheduled = set()
    rounds = messages = 0
    while remaining:
        winners = set()
        for vertex in sorted(remaining):
            neighbors = set(np.flatnonzero(adjacency[vertex])).intersection(remaining)
            messages += len(neighbors)
            if not neighbors:
                winners.add(vertex)
                continue
            best = max(priorities[index] for index in neighbors)
            if priorities[vertex] > best:
                winners.add(vertex)
            elif priorities[vertex] == best:
                tied = min(index for index in neighbors if priorities[index] == best)
                if vertex < tied:
                    winners.add(vertex)
        if not winners:
            raise RuntimeError("DistGCN-LGS made no progress")
        blocked = set()
        for winner in winners:
            blocked.update(np.flatnonzero(adjacency[winner]).tolist())
        scheduled.update(winners)
        remaining.difference_update(scheduled)
        remaining.difference_update(blocked)
        rounds += 1
    action = np.zeros(len(priorities), dtype=np.float32)
    action[list(scheduled)] = 1.0
    return action, {"contention_rounds": rounds, "directed_message_proxy": messages}


class DistGCNQueueScheduler:
    """Retained topology GCN plus per-slot queue-weighted distributed LGS.

    The checkpoint's row-normalized positive node input makes its learned
    embedding topology-only for every positive node weight.  We therefore run
    the legacy GCN once per fixed channel graph, then update ``Q*C_solo`` and
    exclude empty queues before the source LGS rule in each slot.
    """

    model_version = "distgcn_lgs_retained_checkpoint_uav_hard_graph_v1"

    def __init__(
        self,
        inr_threshold_db=0.0,
        graph_source="project_conflict",
        checkpoint=DEFAULT_DISTGCN_CHECKPOINT,
        conda_env="gnn-distgcn-legacy",
    ):
        self.inr_threshold_db = float(inr_threshold_db)
        self.graph_source = str(graph_source)
        self.checkpoint = str(Path(checkpoint).resolve())
        self.conda_env = str(conda_env)
        self._cache = {}
        self.last_diagnostics = {}

    def eval(self):
        return self

    def _topology(self, state):
        cached = self._cache.get(id(state))
        if cached is not None and cached[0] is state:
            return cached[1], cached[2], cached[3]
        import torch

        adjacency, _, metadata = build_hardened_mwis_instance(
            state,
            torch.ones(state.num_vertices),
            inr_threshold_db=self.inr_threshold_db,
            enforce_shared_uav=True,
            graph_source=self.graph_source,
        )
        adjacency_np, _, _ = to_distgcn_numpy(
            adjacency, torch.ones(state.num_vertices)
        )
        topology_hash = hashlib.sha256(adjacency_np.tobytes()).hexdigest()
        cache_key = (self.checkpoint, self.conda_env, topology_hash)
        embeddings = _DISTGCN_EMBEDDING_CACHE.get(cache_key)
        if embeddings is None:
            embeddings = run_distgcn_checkpoint_isolated(
                adjacency_np,
                np.ones(state.num_vertices, dtype=np.float32),
                checkpoint=self.checkpoint,
                conda_env=self.conda_env,
            )
            _DISTGCN_EMBEDDING_CACHE[cache_key] = embeddings
        metadata = {**metadata, "topology_sha256": topology_hash}
        self._cache[id(state)] = (state, adjacency_np, embeddings, metadata)
        return adjacency_np, embeddings, metadata

    def schedule_with_state(self, state, queues):
        import torch

        adjacency, embeddings, metadata = self._topology(state)
        _, weights, _ = build_hardened_mwis_instance(
            state,
            queues,
            inr_threshold_db=self.inr_threshold_db,
            enforce_shared_uav=True,
            graph_source=self.graph_source,
        )
        weights = weights.detach().cpu().numpy().reshape(-1)
        action, lgs = distgcn_local_greedy(
            adjacency, embeddings * weights, valid=weights > 0
        )
        if np.any((action[:, None] * action[None, :]) * adjacency):
            raise RuntimeError("DistGCN-LGS returned a non-independent action")
        self.last_diagnostics = {
            "checkpoint": self.checkpoint,
            "embedding_scope": "once_per_fixed_channel_hard_graph",
            "per_slot_weight": "Q_i_times_C_i_solo",
            "empty_queue_policy": "excluded_before_LGS",
            "gcn_embedding_min": float(embeddings.min()),
            "gcn_embedding_max": float(embeddings.max()),
            **metadata,
            **lgs,
        }
        return torch.from_numpy(action).unsqueeze(1)


__all__ = [
    "LinkIndexRoundTrip",
    "build_hardened_mwis_instance",
    "DEFAULT_DISTGCN_CHECKPOINT",
    "DistGCNQueueScheduler",
    "distgcn_local_greedy",
    "run_distgcn_checkpoint_isolated",
    "to_distgcn_numpy",
]
