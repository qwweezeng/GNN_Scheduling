"""Link-identity preserving UAV mobility and block-fading simulation.

The existing :mod:`simulator.queue_simulator` assumes that every channel block
has exactly the same link tensor.  That is appropriate for the fixed-channel
benchmark, but a freshly generated UAV graph can add, remove, or reorder links.
This module instead gives every directed UAV pair a persistent ID and keeps its
queue outside the per-block conflict graph.

The true channel used for service is deliberately separated from the observed
CSI used by the scheduler.  This makes latest, one-block-stale, large-scale,
and distance-quantized CSI comparable on exactly the same mobility, fading,
and arrival trace.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from models.model import calculate_achievable_rate
from utils.feasibility import project_shared_uav_feasible, shared_uav_action_audit
from utils.network_generate import NetworkState


CSI_MODES = ("latest", "stale", "large_scale", "quantized")


@dataclass(frozen=True)
class MobileChannelBlock:
    positions: torch.Tensor
    true_channel: torch.Tensor
    true_power: torch.Tensor
    large_scale_power: torch.Tensor
    quantized_power: Dict[int, torch.Tensor]
    available: torch.Tensor
    fading_rho: float


@dataclass(frozen=True)
class MobileBlockFadingTrace:
    """A compact true-channel trajectory with a persistent flow catalog."""

    links: torch.Tensor
    blocks: Tuple[MobileChannelBlock, ...]
    slots_per_block: int
    num_slots: int
    slot_duration: float
    speed_mps: float
    seed: int
    channel_model: str = "rayleigh"
    rician_k_db: float = None

    @property
    def num_links(self) -> int:
        return int(self.links.size(0))


def _randn_complex(shape, generator):
    real = torch.randn(shape, generator=generator) / np.sqrt(2.0)
    imag = torch.randn(shape, generator=generator) / np.sqrt(2.0)
    return torch.complex(real, imag)


def _initial_positions(net_gen, generator):
    positions = torch.zeros(net_gen.num_uavs, 3)
    for idx in range(net_gen.num_uavs):
        candidate = None
        for _ in range(1000):
            candidate = torch.tensor([
                torch.rand((), generator=generator) * net_gen.area_width,
                torch.rand((), generator=generator) * net_gen.area_width,
                net_gen.height_min
                + torch.rand((), generator=generator)
                * (net_gen.height_max - net_gen.height_min),
            ])
            if idx == 0 or torch.all(
                torch.linalg.vector_norm(positions[:idx] - candidate, dim=1)
                >= net_gen.min_distance
            ):
                break
        positions[idx] = candidate
    return positions


def _initial_velocities(num_uavs, speed_mps, generator):
    heading = 2.0 * np.pi * torch.rand(num_uavs, generator=generator)
    velocity = torch.zeros(num_uavs, 3)
    velocity[:, 0] = float(speed_mps) * torch.cos(heading)
    velocity[:, 1] = float(speed_mps) * torch.sin(heading)
    return velocity


def _reflect_positions(positions, velocity, net_gen, elapsed_s):
    new_pos = positions + velocity * float(elapsed_s)
    bounds = (
        (0.0, float(net_gen.area_width)),
        (0.0, float(net_gen.area_width)),
        (float(net_gen.height_min), float(net_gen.height_max)),
    )
    velocity = velocity.clone()
    for axis, (low, high) in enumerate(bounds):
        # A block displacement is far smaller than the region in the intended
        # experiments; the loop also makes unusually large test steps safe.
        while bool(((new_pos[:, axis] < low) | (new_pos[:, axis] > high)).any()):
            below = new_pos[:, axis] < low
            above = new_pos[:, axis] > high
            new_pos[below, axis] = 2.0 * low - new_pos[below, axis]
            new_pos[above, axis] = 2.0 * high - new_pos[above, axis]
            velocity[below | above, axis] *= -1.0
    return new_pos, velocity


def _large_scale_power(net_gen, positions):
    distance = torch.cdist(positions, positions).clamp_min(net_gen.min_distance)
    amplitude = (net_gen.lambda0 ** 0.5) * distance.pow(-net_gen.xi / 2.0)
    amplitude.fill_diagonal_(0.0)
    power = net_gen.tx_power * amplitude.square()
    return distance, amplitude, power


def _quantized_large_scale_power(net_gen, distance, q_bits):
    height_span = net_gen.height_max - net_gen.height_min
    d_min = float(net_gen.min_distance)
    d_max = float(
        np.sqrt(2.0 * net_gen.area_width ** 2 + height_span ** 2)
    )
    levels = 2 ** int(q_bits)
    normalized = ((distance - d_min) / (d_max - d_min)).clamp(0.0, 1.0)
    index = torch.floor(normalized * (levels - 1))
    representative = d_min + (index + 0.5) / levels * (d_max - d_min)
    power = net_gen.tx_power * net_gen.lambda0 * representative.pow(-net_gen.xi)
    power.fill_diagonal_(0.0)
    return power


def generate_mobile_block_fading_trace(
    net_gen,
    num_slots: int,
    slots_per_block: int,
    slot_duration: float,
    speed_mps: float,
    seed: int,
    quantization_bits: Sequence[int] = (2, 4, 8),
    rician_k_db=None,
):
    """Generate a continuous-mobility, correlated block-fading trajectory.

    The persistent flow catalog is the set of directed UAV pairs whose initial
    *large-scale* SNR exceeds the configured threshold.  Its ordering is the
    lexicographic ``(source, destination)`` ordering returned by ``torch.where``.
    A catalog link may later become unavailable; its queue remains in the
    catalog and only its scheduling action is forced to zero.

    The diffuse component follows a blockwise complex Gauss--Markov process
    with Jakes correlation ``J0(2*pi*f_D*T_block)``.  ``rician_k_db=None``
    retains the historical pure-Rayleigh model.  A finite nonnegative value
    adds a normalized deterministic LoS phasor with that Rician K-factor;
    ``float('inf')`` gives the pure-LoS sensitivity boundary.  The mean
    small-scale power is one for every finite K, so large-scale link budgets
    remain paired across channel variants.
    """
    if num_slots <= 0 or slots_per_block <= 0:
        raise ValueError("num_slots and slots_per_block must be positive")
    if rician_k_db is not None and (
        np.isnan(float(rician_k_db))
        or np.isneginf(float(rician_k_db))
        or (
            not np.isposinf(float(rician_k_db))
            and float(rician_k_db) < 0.0
        )
    ):
        raise ValueError("rician_k_db must be nonnegative, infinite, or None")
    generator = torch.Generator().manual_seed(int(seed))
    positions = _initial_positions(net_gen, generator)
    velocity = _initial_velocities(net_gen.num_uavs, speed_mps, generator)
    fading = _randn_complex((net_gen.num_uavs, net_gen.num_uavs), generator)
    fading.fill_diagonal_(0.0)

    block_seconds = float(slots_per_block) * float(slot_duration)
    doppler_hz = abs(float(speed_mps)) * float(net_gen.carrier_f) / 3.0e8
    argument = torch.tensor(2.0 * np.pi * doppler_hz * block_seconds)
    rho = float(torch.special.bessel_j0(argument).item())
    innovation_scale = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    num_blocks = (int(num_slots) + int(slots_per_block) - 1) // int(slots_per_block)

    blocks: List[MobileChannelBlock] = []
    catalog_links = None
    snr_threshold = 10.0 ** (float(net_gen.snr_thres) / 10.0)
    for block_idx in range(num_blocks):
        distance, amplitude, large_power = _large_scale_power(net_gen, positions)
        amplitude_complex = amplitude.to(fading.dtype)
        if rician_k_db is None:
            small_scale = fading
        else:
            wavelength = 3.0e8 / float(net_gen.carrier_f)
            los_phasor = torch.exp(
                -1j * 2.0 * np.pi * distance.to(fading.dtype) / wavelength
            )
            if np.isposinf(float(rician_k_db)):
                small_scale = los_phasor
            else:
                k_linear = 10.0 ** (float(rician_k_db) / 10.0)
                los_scale = np.sqrt(k_linear / (k_linear + 1.0))
                diffuse_scale = np.sqrt(1.0 / (k_linear + 1.0))
                small_scale = los_scale * los_phasor + diffuse_scale * fading
        true_h = amplitude_complex * small_scale
        true_power = net_gen.tx_power * true_h.abs().square()
        true_power.fill_diagonal_(0.0)
        if catalog_links is None:
            src, dst = torch.where(large_power / net_gen.noise_power >= snr_threshold)
            catalog_links = torch.stack([src, dst], dim=1)
            if catalog_links.numel() == 0:
                raise RuntimeError("Initial large-scale threshold produced no persistent links")
        direct_large = large_power[catalog_links[:, 0], catalog_links[:, 1]]
        available = direct_large / net_gen.noise_power >= snr_threshold
        quantized = {
            int(bits): _quantized_large_scale_power(net_gen, distance, int(bits))
            for bits in quantization_bits
        }
        blocks.append(MobileChannelBlock(
            positions=positions.clone(),
            true_channel=true_h.clone(),
            true_power=true_power,
            large_scale_power=large_power,
            quantized_power=quantized,
            available=available.clone(),
            fading_rho=rho,
        ))
        if block_idx + 1 < num_blocks:
            positions, velocity = _reflect_positions(
                positions, velocity, net_gen, block_seconds
            )
            innovation = _randn_complex(fading.shape, generator)
            innovation.fill_diagonal_(0.0)
            fading = rho * fading + innovation_scale * innovation
            fading.fill_diagonal_(0.0)

    return MobileBlockFadingTrace(
        links=catalog_links,
        blocks=tuple(blocks),
        slots_per_block=int(slots_per_block),
        num_slots=int(num_slots),
        slot_duration=float(slot_duration),
        speed_mps=float(speed_mps),
        seed=int(seed),
        channel_model=(
            "rayleigh"
            if rician_k_db is None else
            ("pure_los" if np.isposinf(float(rician_k_db)) else "rician")
        ),
        rician_k_db=(
            None if rician_k_db is None else float(rician_k_db)
        ),
    )


def _observed_power(trace, block_index, csi_mode, q_bits, stale_blocks=1):
    block = trace.blocks[block_index]
    if csi_mode == "latest":
        return block.true_power
    if csi_mode == "stale":
        if int(stale_blocks) < 1:
            raise ValueError("stale_blocks must be at least one")
        return trace.blocks[max(block_index - int(stale_blocks), 0)].true_power
    if csi_mode == "large_scale":
        return block.large_scale_power
    if csi_mode == "quantized":
        if int(q_bits) not in block.quantized_power:
            raise ValueError(f"Trace has no {q_bits}-bit quantized CSI")
        return block.quantized_power[int(q_bits)]
    raise ValueError(f"Unknown CSI mode {csi_mode}; expected {CSI_MODES}")


def build_conflict_graph_from_power(links, power, noise_power, snr_threshold):
    """Build the directed scheduler graph from one coherent power matrix."""
    num_links = int(links.size(0))
    if num_links == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 1), dtype=torch.float32),
        )
    tx, rx = links[:, 0], links[:, 1]
    direct = power[tx, rx]
    # gains[target, source] is the power from source transmitter at target RX.
    gains = power[tx[None, :], rx[:, None]]
    denominator = (direct / float(snr_threshold) - float(noise_power)).clamp_min(1e-30)
    physical = gains >= denominator[:, None]
    shared = (
        (tx[:, None] == tx[None, :])
        | (tx[:, None] == rx[None, :])
        | (rx[:, None] == tx[None, :])
        | (rx[:, None] == rx[None, :])
    )
    # ``shared`` is indexed [target, source] as well; it is symmetric.
    mask = physical | shared
    mask.fill_diagonal_(False)
    target, source = torch.where(mask)
    edge_index = torch.stack([source, target], dim=0)
    ratio = gains[target, source] / denominator[target]
    endpoint = shared[target, source]
    raw_weight = torch.where(endpoint, torch.full_like(ratio, 1e5), ratio)
    edge_weight = torch.log10(raw_weight + 1.0)
    if edge_weight.numel() > 0:
        edge_weight = edge_weight / edge_weight.max().clamp_min(1e-12)
    return edge_index.long(), edge_weight.float().unsqueeze(1)


def build_observed_network_state(
    net_gen, trace, block_index, csi_mode, q_bits=4, stale_blocks=1
):
    """Build the current available-link graph and its persistent-ID mapping."""
    block = trace.blocks[int(block_index)]
    power = _observed_power(
        trace, int(block_index), csi_mode, int(q_bits), int(stale_blocks)
    ).float()
    return build_network_state_from_power(
        net_gen, trace, block_index, power, csi_label=csi_mode
    )


def build_network_state_from_power(
    net_gen, trace, block_index, power, csi_label="predicted"
):
    """Build a coherent local scheduler state from one UAV-pair power matrix.

    This is the interface used by physical CSI predictors.  The supplied
    matrix defines both direct and cross gains, so LocalMax and conditional
    reuse never see mutually inconsistent channel estimates.
    """
    block = trace.blocks[int(block_index)]
    persistent_indices = torch.where(block.available)[0]
    links = trace.links[persistent_indices]
    power = torch.as_tensor(power, dtype=torch.float32).clone()
    expected_shape = (int(block.positions.size(0)), int(block.positions.size(0)))
    if tuple(power.shape) != expected_shape:
        raise ValueError(
            f"power must have shape {expected_shape}, got {tuple(power.shape)}"
        )
    if not torch.isfinite(power).all() or bool((power < 0).any()):
        raise ValueError("power must contain finite nonnegative values")
    power.fill_diagonal_(0.0)
    direct = power[links[:, 0], links[:, 1]]
    snr_db = 10.0 * torch.log10(direct / net_gen.noise_power + 1e-12)
    edge_index, edge_weights = build_conflict_graph_from_power(
        links,
        power,
        net_gen.noise_power,
        10.0 ** (float(net_gen.snr_thres) / 10.0),
    )
    neighbor_map = {idx: [] for idx in range(int(links.size(0)))}
    for source, target in edge_index.t().tolist():
        neighbor_map[target].append(source)
    state = NetworkState(
        links=links,
        vertex_features=snr_db.unsqueeze(1).float(),
        edge_index=edge_index,
        edge_weights=edge_weights,
        large_scale=torch.sqrt(
            block.large_scale_power[links[:, 0], links[:, 1]] / net_gen.tx_power
        ),
        signal_power=power,
        signal_power_large_scale=power,
        signal_power_quant=power,
        noise_power=net_gen.noise_power,
        bandwidth=net_gen.bandwidth,
        snr_gap=net_gen.snr_gap,
        neighbor_map=neighbor_map,
        positions=block.positions,
        tx_power=net_gen.tx_power,
        carrier_f=net_gen.carrier_f,
        csi_mode=str(csi_label),
        queue_states=None,
        include_queue_feature=False,
    )
    return state, persistent_indices


def build_true_network_state(net_gen, trace, block_index):
    """Build the service-evaluation state on the same current available links."""
    return build_observed_network_state(
        net_gen, trace, block_index, "latest", q_bits=4
    )


def csi_error_db(
    net_gen, trace, block_index, csi_mode, q_bits=4, stale_blocks=1
):
    """Mean absolute dB error over current persistent direct and cross gains."""
    available = torch.where(trace.blocks[block_index].available)[0]
    links = trace.links[available]
    if links.numel() == 0:
        return 0.0
    true = trace.blocks[block_index].true_power
    observed = _observed_power(
        trace, block_index, csi_mode, q_bits, stale_blocks
    )
    tx, rx = links[:, 0], links[:, 1]
    # Include direct and cross gains used by scheduling; omit zero self-UAV paths.
    true_gain = true[tx[None, :], rx[:, None]].reshape(-1)
    observed_gain = observed[tx[None, :], rx[:, None]].reshape(-1)
    valid = (true_gain > 0) & (observed_gain > 0)
    if not bool(valid.any()):
        return 0.0
    error = torch.abs(
        10.0 * torch.log10(observed_gain[valid])
        - 10.0 * torch.log10(true_gain[valid])
    )
    return float(error.mean().item())


def _update_fifo_completion_tracking(
    packet_cohorts, departures, arrivals, slot, delivered_delays
):
    """Apply fluid service to integer FIFO packets, then append new arrivals.

    The mobile simulator uses the same timing convention as ``QueueSimulator``:
    service happens first, arrivals enter after service, and a newly arrived
    packet is first eligible in the next slot.  The fluid queue remains the
    scheduling source of truth; this bookkeeping only determines when a whole
    packet has completed service.
    """
    eps = 1e-8
    departures = torch.as_tensor(departures).detach().cpu().reshape(-1)
    arrivals = torch.as_tensor(arrivals).detach().cpu().reshape(-1)
    for link_idx in (departures > eps).nonzero(as_tuple=True)[0].tolist():
        service = float(departures[link_idx])
        queue = packet_cohorts[link_idx]
        while service > eps and queue:
            packet = queue[0]
            consumed = min(service, packet[1])
            packet[1] -= consumed
            service -= consumed
            if packet[1] <= eps:
                queue.popleft()
                delivered_delays.append(
                    max(0, int(slot) - int(packet[0]))
                )
            else:
                break

    for link_idx in (arrivals > 0).nonzero(as_tuple=True)[0].tolist():
        count = int(round(float(arrivals[link_idx])))
        for _ in range(count):
            packet_cohorts[link_idx].append([int(slot), 1.0])


def run_mobile_queue_simulation(
    net_gen,
    scheduler,
    trace: MobileBlockFadingTrace,
    arrivals: torch.Tensor,
    csi_mode: str,
    q_bits: int = 4,
    packet_size: int = 12000,
    stale_blocks: int = 1,
    observed_power_by_block=None,
    observed_csi_label: str = None,
):
    """Run one scheduler with persistent queues and true-channel service."""
    arrivals = torch.as_tensor(arrivals, dtype=torch.float32)
    expected = (trace.num_slots, trace.num_links)
    if tuple(arrivals.shape) != expected:
        raise ValueError(f"arrivals must have shape {expected}, got {tuple(arrivals.shape)}")
    queues = torch.zeros(trace.num_links, dtype=torch.float32)
    packet_cohorts = [deque() for _ in range(trace.num_links)]
    delivered_packet_delays = []
    queue_history, departure_history = [], []
    active_counts, available_counts, available_tx_counts, edge_counts = [], [], [], []
    utilities, csi_errors, hard_projection_rejections = [], [], []
    current_block_index = -1
    observed_state = true_state = persistent_indices = None
    scale = net_gen.bandwidth * trace.slot_duration / float(packet_size)

    for slot in range(trace.num_slots):
        block_index = slot // trace.slots_per_block
        if block_index != current_block_index:
            current_block_index = block_index
            if observed_power_by_block is None:
                observed_state, persistent_indices = build_observed_network_state(
                    net_gen, trace, block_index, csi_mode, q_bits,
                    stale_blocks=stale_blocks,
                )
            else:
                observed_power = observed_power_by_block[block_index]
                observed_state, persistent_indices = build_network_state_from_power(
                    net_gen, trace, block_index, observed_power,
                    csi_label=observed_csi_label or csi_mode,
                )
            true_state, true_indices = build_true_network_state(
                net_gen, trace, block_index
            )
            if not torch.equal(persistent_indices, true_indices):
                raise RuntimeError("Observed and true state lost link-identity alignment")
            if hasattr(scheduler, "clear_cache"):
                scheduler.clear_cache()
            if observed_power_by_block is None:
                csi_errors.append(csi_error_db(
                    net_gen, trace, block_index, csi_mode, q_bits,
                    stale_blocks=stale_blocks,
                ))
            else:
                true_power = trace.blocks[block_index].true_power
                valid = (observed_power > 0) & (true_power > 0)
                error = torch.abs(
                    10.0 * torch.log10(observed_power[valid])
                    - 10.0 * torch.log10(true_power[valid])
                )
                csi_errors.append(float(error.mean()) if bool(valid.any()) else 0.0)

        local_queues = queues[persistent_indices]
        action = scheduler.schedule_with_state(
            observed_state, local_queues
        ).reshape(-1).float().cpu()
        action *= (local_queues > 0).float()
        # Apply the paper's single-radio/half-duplex constraint uniformly,
        # including third-party schedulers such as FPLinQ.  Proposed
        # MarginalMP variants are already feasible, so this is an auditable
        # no-op for them and a required final arbitration for other methods.
        direct_power = observed_state.signal_power[
            observed_state.links[:, 0], observed_state.links[:, 1]
        ].detach().cpu()
        solo_rate = torch.log2(
            1.0
            + direct_power
            / (float(observed_state.noise_power) * float(observed_state.snr_gap))
        )
        priority = local_queues.detach().cpu() * solo_rate
        action, hard_feasibility = project_shared_uav_feasible(
            action, observed_state.links, priorities=priority
        )
        hard_projection_rejections.append(
            float(len(hard_feasibility["rejected_project_link_indices"]))
        )
        action = action.reshape(-1).float().cpu()
        if not shared_uav_action_audit(action, observed_state.links)["feasible"]:
            raise RuntimeError("mobile scheduler emitted a shared-UAV conflict")
        rates = calculate_achievable_rate(
            action.unsqueeze(1), true_state, binary=True
        ).reshape(-1).cpu()
        capacity_packets = rates * scale
        departures_local = torch.minimum(local_queues, capacity_packets)
        departures = torch.zeros_like(queues)
        departures[persistent_indices] = departures_local
        utility = float((local_queues * rates * scale).sum().item())
        _update_fifo_completion_tracking(
            packet_cohorts,
            departures,
            arrivals[slot],
            slot,
            delivered_packet_delays,
        )
        queues = (queues - departures).clamp_min(0.0) + arrivals[slot]

        queue_history.append(queues.clone())
        departure_history.append(departures)
        active_counts.append(float(action.sum().item()))
        available_counts.append(float(persistent_indices.numel()))
        available_tx_counts.append(float(torch.unique(observed_state.links[:, 0]).numel()))
        edge_counts.append(float(observed_state.edge_index.size(1)))
        utilities.append(utility)

    queue_tensor = torch.stack(queue_history)
    departure_tensor = torch.stack(departure_history)
    total_arrivals = float(arrivals.sum().item())
    total_departures = float(departure_tensor.sum().item())
    avg_departure = float(departure_tensor.sum(dim=1).mean().item())
    avg_total_backlog = float(queue_tensor.sum(dim=1).mean().item())
    delay_slots = avg_total_backlog / avg_departure if avg_departure > 0 else None
    delivered_packet_count = len(delivered_packet_delays)
    unfinished_packet_count = sum(len(queue) for queue in packet_cohorts)
    successful_delivery_ratio = (
        delivered_packet_count / total_arrivals if total_arrivals > 0 else 1.0
    )
    if delivered_packet_delays:
        delivered_delays = np.asarray(delivered_packet_delays, dtype=float)
        average_packet_delay_slots = float(delivered_delays.mean())
        p95_packet_delay_slots = float(np.percentile(delivered_delays, 95))
    else:
        average_packet_delay_slots = None
        p95_packet_delay_slots = None
    duration_s = trace.num_slots * trace.slot_duration
    physical_service_mbps = (
        total_departures * packet_size / duration_s / 1e6
        if duration_s > 0 else 0.0
    )
    delivered_throughput_mbps = (
        delivered_packet_count * packet_size / duration_s / 1e6
        if duration_s > 0 else 0.0
    )
    final_total_backlog = float(queue_tensor[-1].sum().item())
    return {
        # Backward-compatible aliases retain the old fluid-service meaning.
        "throughput_ratio": total_departures / total_arrivals if total_arrivals > 0 else 0.0,
        "throughput_mbps": physical_service_mbps,
        "physical_service_ratio": (
            total_departures / total_arrivals if total_arrivals > 0 else 0.0
        ),
        "physical_service_mbps": physical_service_mbps,
        "delivered_packet_count": delivered_packet_count,
        "delivered_throughput_mbps": delivered_throughput_mbps,
        "successful_goodput_mbps": delivered_throughput_mbps,
        "successful_delivery_ratio": successful_delivery_ratio,
        "unfinished_packet_count": unfinished_packet_count,
        "unfinished_packet_fraction": (
            unfinished_packet_count / total_arrivals if total_arrivals > 0 else 0.0
        ),
        "total_arrivals": total_arrivals,
        "total_departures": total_departures,
        "final_total_backlog": final_total_backlog,
        "long_term_total_backlog": avg_total_backlog,
        "average_queue_backlog_mbit": (
            float(queue_tensor.mean().item()) * packet_size / 1e6
        ),
        "average_total_queue_backlog_mbit": (
            avg_total_backlog * packet_size / 1e6
        ),
        "average_delay_slots": delay_slots,
        "average_delay_ms": delay_slots * trace.slot_duration * 1000.0
        if delay_slots is not None else None,
        "packet_delay_count": delivered_packet_count,
        "average_packet_delay_slots": average_packet_delay_slots,
        "average_packet_delay_ms": (
            average_packet_delay_slots * trace.slot_duration * 1000.0
            if average_packet_delay_slots is not None else None
        ),
        "p95_packet_delay_slots": p95_packet_delay_slots,
        "p95_packet_delay_ms": (
            p95_packet_delay_slots * trace.slot_duration * 1000.0
            if p95_packet_delay_slots is not None else None
        ),
        "packet_count_conservation_error": float(
            abs(total_arrivals - delivered_packet_count - unfinished_packet_count)
        ),
        "packet_work_conservation_error": float(
            abs(total_arrivals - total_departures - final_total_backlog)
        ),
        "average_active_links": float(np.mean(active_counts)),
        "average_available_links": float(np.mean(available_counts)),
        "average_available_transmitter_uavs": float(np.mean(available_tx_counts)),
        "average_directed_conflict_edges": float(np.mean(edge_counts)),
        "average_mws_utility_packets": float(np.mean(utilities)),
        "average_hard_projection_rejections": float(
            np.mean(hard_projection_rejections)
        ),
        "shared_uav_feasible_all_slots": True,
        "mean_csi_absolute_error_db": float(np.mean(csi_errors)),
        "fading_rho": float(trace.blocks[0].fading_rho),
        "num_persistent_links": trace.num_links,
    }


def message_overhead_audit(
    average_links: float,
    average_transmitter_uavs: float,
    average_directed_edges: float,
    slots_per_block: int,
    slot_duration: float = 0.001,
    gat_rounds: int = 4,
    hidden_dim: int = 48,
    uses_channel_gat: bool = True,
):
    """Account for algorithm messages; CSI acquisition is reported separately.

    Link-edge unicast is a conservative upper-bound accounting.  The aggregated
    broadcast proxy packs all link payloads owned by a transmitter UAV into one
    packet per round.  It excludes headers, fragmentation, contention, and
    retransmission, so it is a lower bound rather than a deployable packet format.
    """
    block_rounds = (
        int(gat_rounds) + 1 if uses_channel_gat else 0
    )  # GAT layers plus final activity prior; Pure has neither.
    queue_rounds = 3  # W, (a0,Q), and (candidate,Delta).
    queue_bits_per_edge_slot = 32 + (1 + 32) + (1 + 32)
    block_bits_per_edge = (
        int(gat_rounds) * int(hidden_dim) * 32 + 32
        if uses_channel_gat else 0
    )
    edge_bits_slot = (
        average_directed_edges * queue_bits_per_edge_slot
        + average_directed_edges * block_bits_per_edge / int(slots_per_block)
    )
    aggregate_payload_bits_slot = (
        average_links * queue_bits_per_edge_slot
        + average_links * block_bits_per_edge / int(slots_per_block)
    )
    broadcast_packets_slot = average_transmitter_uavs * (
        queue_rounds + block_rounds / float(slots_per_block)
    )
    return {
        "queue_rounds_per_slot": queue_rounds,
        "channel_rounds_per_block": block_rounds,
        "amortized_algorithm_rounds_per_slot": queue_rounds
        + block_rounds / float(slots_per_block),
        "link_edge_unicast_bits_per_slot": float(edge_bits_slot),
        "link_edge_unicast_mbit_per_second": float(
            edge_bits_slot / float(slot_duration) / 1e6
        ),
        "transmitter_aggregated_payload_bits_per_slot_lower_bound": float(
            aggregate_payload_bits_slot
        ),
        "transmitter_aggregated_payload_mbit_per_second_lower_bound": float(
            aggregate_payload_bits_slot / float(slot_duration) / 1e6
        ),
        "transmitter_broadcast_packets_per_slot_lower_bound": float(
            broadcast_packets_slot
        ),
        "csi_acquisition_included": False,
        "uses_channel_gat": bool(uses_channel_gat),
        "assumptions": {
            "float_bits": 32,
            "flag_bits": 1,
            "gat_hidden_dim": int(hidden_dim),
            "gat_rounds": int(gat_rounds),
            "queue_message_fields": ["W", "a0+Q", "candidate+Delta"],
        },
    }
