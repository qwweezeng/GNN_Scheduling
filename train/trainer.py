import copy
import os
import random
from time import strftime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.model import calculate_achievable_rate
from utils.feasibility import decode_endpoint_mutual_matching
from utils.queue_features import relative_queue_feature


def sample_link_queues(num_links, queue_range=(0, 200), generator=None):
    """Generate a link-level queue snapshot covering sparse and dense loads."""
    if num_links <= 0:
        return torch.empty(0, dtype=torch.float32)
    low, high = int(queue_range[0]), int(queue_range[1])
    low, high = max(1, low), max(max(1, low) + 1, high)
    num_nonempty = torch.randint(
        1, num_links + 1, (1,), generator=generator
    ).item()
    active = torch.randperm(num_links, generator=generator)[:num_nonempty]
    queues = torch.zeros(num_links, dtype=torch.float32)
    queues[active] = torch.randint(
        low, high, (num_nonempty,), generator=generator
    ).float()
    return queues


def compute_queue_feature_stats(dataset, queue_range=(0, 200), seed=0):
    """Compute mean/std for [SNR_dB, Q/max(Q)]."""
    generator = torch.Generator().manual_seed(int(seed))
    snr, queue_features = [], []
    for state in dataset:
        snr.append(state.vertex_features[:, 0].detach().cpu().float())
        queues = sample_link_queues(state.num_vertices, queue_range, generator)
        queue_features.append(relative_queue_feature(queues))
    if not snr:
        return torch.zeros(2), torch.ones(2)
    snr, queue_features = torch.cat(snr), torch.cat(queue_features)
    mean = torch.stack([snr.mean(), queue_features.mean()])
    std = torch.stack([
        snr.std(unbiased=False), queue_features.std(unbiased=False)
    ]).clamp_min(1e-6)
    return mean, std


class Trainer:
    def __init__(self, model, optimizer, device=None, gpu_id=None, log_dir="./logs",
                 grad_clip_config=None, use_power_control=False, use_queue=False,
                 bandwidth=5e6, slot_duration=0.001, packet_size=12000,
                 queue_range=(0, 200), queue_seed=0):
        self.model = model
        self.optimizer = optimizer
        self.use_power_control = use_power_control
        self.use_queue = use_queue
        self.bandwidth = bandwidth
        self.slot_duration = slot_duration
        self.packet_size = packet_size
        self.queue_range = queue_range
        self.queue_seed = int(queue_seed)
        self.log_dir = log_dir
        self.last_eval_metrics = {}
        self.grad_clip_config = grad_clip_config or {
            'explosion_threshold': 100,
            'explosion_clip': 10.0,
            'normal_clip': 5.0,
        }

        if device is None or device == "auto":
            if torch.cuda.is_available():
                gpu_id = gpu_id if gpu_id is not None else 0
                gpu_id = gpu_id if gpu_id < torch.cuda.device_count() else 0
                device = torch.device(f"cuda:{gpu_id}")
                print(f"[Trainer] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                device = torch.device("cpu")
                print("[Trainer] Using CPU (no GPU available)")
        self.device = torch.device(device) if isinstance(device, str) else device

    @property
    def packets_per_spectral_efficiency(self):
        return self.bandwidth * self.slot_duration / self.packet_size

    def _queue_input(self, state, x, generator):
        queues = sample_link_queues(
            state.num_vertices, self.queue_range, generator
        ).to(x.device)
        queue_feature = relative_queue_feature(queues).to(x.device).unsqueeze(1)
        x = x.clone()
        if x.size(1) == 1:
            x = torch.cat([x, queue_feature], dim=1)
        else:
            x[:, 1] = queue_feature.squeeze(1)
        return x, queues

    @staticmethod
    def _nonempty_mask(pi, queues):
        return pi * (queues > 0).to(pi.dtype).unsqueeze(1)

    @staticmethod
    def _endpoint_conflict_penalty(pi, links):
        """Penalize simultaneous soft activation of shared-endpoint links."""
        values = pi.reshape(-1)
        endpoints = links.to(values.device)
        if values.numel() < 2 or endpoints.numel() == 0:
            return values.new_zeros(())
        shared = (
            (endpoints[:, None, 0] == endpoints[None, :, 0])
            | (endpoints[:, None, 0] == endpoints[None, :, 1])
            | (endpoints[:, None, 1] == endpoints[None, :, 0])
            | (endpoints[:, None, 1] == endpoints[None, :, 1])
        )
        upper = torch.triu(shared, diagonal=1)
        pair_indices = upper.nonzero(as_tuple=False)
        if pair_indices.numel() == 0:
            return values.new_zeros(())
        pair_products = values[pair_indices[:, 0]] * values[pair_indices[:, 1]]
        return pair_products.sum() / max(1, values.numel())

    @staticmethod
    def _model_output(model, x, edge_index, edge_weights, use_power_control,
                      use_ste=False):
        result = model(x, edge_index, edge_weights, use_ste=use_ste)
        if use_power_control:
            return result[0], result[1]
        return result[0], None

    def calibrate_decision_threshold(self, model, dataset,
                                     use_power_control=False):
        """Choose one global threshold by hard validation MWS utility."""
        model.eval().to(self.device)
        generator = torch.Generator().manual_seed(
            self.queue_seed + 1_000_003
        )
        records, nonempty_probabilities = [], []
        with torch.no_grad():
            for state in dataset:
                x = state.vertex_features.to(self.device)
                x, queues = self._queue_input(state, x, generator)
                pi_raw, alpha = self._model_output(
                    model,
                    x,
                    state.edge_index.to(self.device),
                    state.edge_weights.to(self.device),
                    use_power_control,
                )
                pi = self._nonempty_mask(pi_raw, queues)
                nonempty = queues > 0
                if nonempty.any():
                    nonempty_probabilities.append(pi.squeeze(1)[nonempty])
                records.append((state, queues, pi, alpha))

        if not nonempty_probabilities:
            model.decision_threshold = 0.5
            return 0.5

        probabilities = torch.cat(nonempty_probabilities)
        active_fractions = (0.01, 0.02, 0.03, 0.05, 0.08,
                            0.10, 0.15, 0.20, 0.30, 0.50)
        thresholds = {0.5}
        for fraction in active_fractions:
            thresholds.add(float(torch.quantile(
                probabilities, 1.0 - fraction
            ).item()))

        best_threshold, best_utility = 0.5, -float('inf')
        scale = self.packets_per_spectral_efficiency
        for threshold in sorted(thresholds):
            total = 0.0
            for state, queues, pi, alpha in records:
                hard_pi = (pi > threshold).to(pi.dtype)
                rates = calculate_achievable_rate(
                    hard_pi, state, binary=False, alpha=alpha
                )
                total += torch.sum(queues * rates * scale).item()
            if total > best_utility + 1e-9:
                best_threshold, best_utility = threshold, total

        model.decision_threshold = best_threshold
        return best_threshold

    @staticmethod
    def _direct_rate_priority(pi, state):
        tx, rx = state.links[:, 0], state.links[:, 1]
        snr = state.signal_power[tx, rx].cpu() / state.noise_power
        return (
            pi.reshape(-1).float().cpu()
            * torch.log2(1.0 + snr / state.snr_gap).float()
        )

    def _evaluate_channel_validation(self, model, dataset, device):
        """Return soft, raw-hard, and projected-hard validation utilities."""
        model.eval().to(device)
        totals = {"soft": 0.0, "hard": 0.0, "projected": 0.0}
        with torch.no_grad():
            for state in dataset:
                pi, _ = self._model_output(
                    model,
                    state.vertex_features[:, :1].to(device),
                    state.edge_index.to(device),
                    state.edge_weights.to(device),
                    False,
                )
                soft_rates = calculate_achievable_rate(
                    pi, state, binary=False
                )
                hard = (pi >= 0.5).float()
                hard_rates = calculate_achievable_rate(
                    hard, state, binary=True, detach_hard=True
                )
                projected, _ = decode_endpoint_mutual_matching(
                    hard.cpu(),
                    state.links,
                    priorities=self._direct_rate_priority(pi, state),
                    max_rounds=1,
                )
                projected_rates = calculate_achievable_rate(
                    projected.to(device), state, binary=True, detach_hard=True
                )
                totals["soft"] += float(soft_rates.sum().item())
                totals["hard"] += float(hard_rates.sum().item())
                totals["projected"] += float(projected_rates.sum().item())
        count = max(len(dataset), 1)
        return {key: value / count for key, value in totals.items()}

    def evaluate(self, model, dataset, device=None, use_power_control=False,
                 use_queue=False):
        device = device or self.device
        model.eval().to(device)
        queue_generator = torch.Generator().manual_seed(
            self.queue_seed + 1_000_003
        )
        hard_total = soft_total = alpha_total = 0.0
        empty_rates, probability_changes, decision_changes, scores = [], [], [], []
        snr_probability_changes, snr_decision_changes = [], []
        active_fractions, nonempty_active_fractions = [], []

        with torch.no_grad():
            for state_idx, state in enumerate(dataset):
                x = state.vertex_features.to(device)
                queues = None
                if use_queue:
                    x, queues = self._queue_input(state, x, queue_generator)
                edge_index = state.edge_index.to(device)
                edge_weights = state.edge_weights.to(device)
                pi_raw, alpha = self._model_output(
                    model, x, edge_index, edge_weights, use_power_control
                )
                if alpha is not None:
                    alpha_total += alpha.mean().item()
                if model.latest_scores is not None:
                    scores.append(model.latest_scores.flatten().cpu())

                if not use_queue:
                    rates = calculate_achievable_rate(
                        pi_raw, state, binary=True,
                        detach_hard=True, alpha=alpha
                    )
                    hard_total += rates.sum().item()
                    continue

                empty = queues <= 0
                threshold = float(getattr(model, "decision_threshold", 0.5))
                if empty.any():
                    empty_rates.append(
                        (pi_raw.squeeze(1)[empty] > threshold).float().mean().item()
                    )
                pi = self._nonempty_mask(pi_raw, queues)
                hard = pi > threshold
                active_fractions.append(hard.float().mean().item())
                nonempty = queues > 0
                nonempty_active_fractions.append(
                    hard.squeeze(1)[nonempty].float().mean().item()
                )
                soft_rates = calculate_achievable_rate(
                    pi, state, binary=False, alpha=alpha
                )
                hard_rates = calculate_achievable_rate(
                    hard.to(pi.dtype), state, binary=False, alpha=alpha
                )
                scale = self.packets_per_spectral_efficiency
                soft_total += torch.sum(queues * soft_rates * scale).item()
                hard_total += torch.sum(queues * hard_rates * scale).item()

                # Permute only positive queue magnitudes so this diagnostic is
                # not driven by the deterministic empty-queue mask.
                nonempty_indices = nonempty.nonzero(as_tuple=True)[0]
                if state_idx < 10 and nonempty_indices.numel() >= 2:
                    order = torch.randperm(
                        nonempty_indices.numel(), generator=queue_generator
                    ).to(nonempty_indices.device)
                    queues_perm = queues.clone()
                    queues_perm[nonempty_indices] = queues[nonempty_indices[order]]
                    x_perm = x.clone()
                    x_perm[:, 1] = relative_queue_feature(queues_perm).to(device)
                    pi_perm, _ = self._model_output(
                        model, x_perm, edge_index, edge_weights,
                        use_power_control
                    )
                    pi_perm = self._nonempty_mask(pi_perm, queues_perm)
                    probability_changes.append((pi - pi_perm).abs().mean().item())
                    decision_changes.append(
                        ((pi > threshold) != (pi_perm > threshold))
                        .float().mean().item()
                    )

                    snr_order = torch.randperm(
                        x.size(0), generator=queue_generator
                    ).to(x.device)
                    x_snr_perm = x.clone()
                    x_snr_perm[:, 0] = x[snr_order, 0]
                    pi_snr, _ = self._model_output(
                        model, x_snr_perm, edge_index, edge_weights,
                        use_power_control
                    )
                    pi_snr = self._nonempty_mask(pi_snr, queues)
                    snr_probability_changes.append(
                        (pi - pi_snr).abs().mean().item()
                    )
                    snr_decision_changes.append(
                        ((pi > threshold) != (pi_snr > threshold))
                        .float().mean().item()
                    )

        count = max(len(dataset), 1)
        if use_queue:
            gap_scale = max(abs(soft_total), abs(hard_total), 1e-12)
            self.last_eval_metrics = {
                "soft_utility": soft_total / count,
                "soft_hard_gap": abs(soft_total - hard_total) / gap_scale,
                "raw_empty_activation_rate": float(np.mean(empty_rates))
                if empty_rates else 0.0,
                "queue_probability_change": float(np.mean(probability_changes))
                if probability_changes else 0.0,
                "queue_hard_decision_change": float(np.mean(decision_changes))
                if decision_changes else 0.0,
                "snr_probability_change": float(np.mean(snr_probability_changes))
                if snr_probability_changes else 0.0,
                "snr_hard_decision_change": float(np.mean(snr_decision_changes))
                if snr_decision_changes else 0.0,
                "hard_active_fraction": float(np.mean(active_fractions))
                if active_fractions else 0.0,
                "nonempty_activation_rate": float(
                    np.mean(nonempty_active_fractions)
                ) if nonempty_active_fractions else 0.0,
                "decision_threshold": float(
                    getattr(model, "decision_threshold", 0.5)
                ),
            }
            if scores:
                scores = torch.cat(scores)
                for name, q in (("p01", 0.01), ("p50", 0.5), ("p99", 0.99)):
                    self.last_eval_metrics[f"score_{name}"] = torch.quantile(
                        scores, q
                    ).item()
        else:
            self.last_eval_metrics = {}

        avg_utility = hard_total / count
        if use_power_control:
            return avg_utility, alpha_total / count
        return avg_utility

    def train(self, num_epochs, train_dataset, val_dataset, eval_interval=10,
              lr_scheduler=None, use_power_control=False, use_queue=None,
              use_ste=False, endpoint_penalty_weight=0.0):
        use_queue = self.use_queue if use_queue is None else use_queue
        model, optimizer, device = self.model, self.optimizer, self.device
        model.to(device)
        writer = SummaryWriter(
            log_dir=os.path.join(
                self.log_dir, "train_" + strftime("%Y-%m-%d_%H-%M")
            ),
            flush_secs=60,
        )
        queue_generator = torch.Generator().manual_seed(self.queue_seed)
        best_utility = -float('inf')
        best_state = best_epoch = None
        best_threshold = 0.5
        logs = []

        for epoch in range(num_epochs):
            model.train()
            random.shuffle(train_dataset)
            (
                losses, utilities, pi_means, pi_stds, binary_active_fractions,
                grad_norms, alpha_means,
            ) = (
                [] for _ in range(7)
            )

            for state in train_dataset:
                x = state.vertex_features.to(device)
                queues = None
                if use_queue:
                    x, queues = self._queue_input(state, x, queue_generator)
                edge_index = state.edge_index.to(device)
                edge_weights = state.edge_weights.to(device)
                pi, alpha = self._model_output(
                    model, x, edge_index, edge_weights,                     use_power_control,
                    use_ste=(use_queue or use_ste),
                )
                if alpha is not None:
                    alpha_means.append(alpha.mean().item())
                if use_queue:
                    pi = self._nonempty_mask(pi, queues)

                rates = calculate_achievable_rate(
                    pi, state, binary=False, alpha=alpha
                )
                if use_queue:
                    utility = torch.sum(
                        queues * rates * self.packets_per_spectral_efficiency
                    )
                    # sum(Q) is independent of the scheduling decision, so this
                    # preserves the per-snapshot MWS argmax while keeping
                    # gradients comparable across queue snapshots.
                    loss = -utility / (queues.sum().detach() + 1e-8)
                else:
                    utility = rates.sum()
                    loss = -utility

                soft_probabilities = getattr(
                    model, "latest_soft_probabilities", pi
                )
                endpoint_penalty = self._endpoint_conflict_penalty(
                    soft_probabilities, state.links
                )
                loss = loss + float(endpoint_penalty_weight) * endpoint_penalty

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float('inf')
                )
                clip = (
                    self.grad_clip_config['explosion_clip']
                    if grad_norm.item() > self.grad_clip_config['explosion_threshold']
                    else self.grad_clip_config['normal_clip']
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

                losses.append(loss.item())
                utilities.append(utility.item())
                pi_means.append(pi.mean().item())
                pi_stds.append(pi.std().item())
                binary_active_fractions.append((pi >= 0.5).float().mean().item())
                grad_norms.append(grad_norm.item())

            means = {
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "utility": float(np.mean(utilities)) if utilities else 0.0,
                "pi_mean": float(np.mean(pi_means)) if pi_means else 0.0,
                "pi_std": float(np.mean(pi_stds)) if pi_stds else 0.0,
                "binary_active_fraction": (
                    float(np.mean(binary_active_fractions))
                    if binary_active_fractions else 0.0
                ),
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
                "grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
                "alpha_mean": float(np.mean(alpha_means)) if alpha_means else None,
            }
            val_utility = val_alpha = None
            channel_val_metrics = {}
            should_evaluate = (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1
            if should_evaluate:
                if use_queue:
                    self.calibrate_decision_threshold(
                        model, val_dataset, use_power_control
                    )
                val_result = self.evaluate(
                    model, val_dataset, device, use_power_control, use_queue
                )
                if use_power_control:
                    val_utility, val_alpha = val_result
                else:
                    val_utility, val_alpha = val_result, 1.0
                if not use_queue:
                    channel_val_metrics = self._evaluate_channel_validation(
                        model, val_dataset, device
                    )
                if val_utility > best_utility + 1e-9:
                    best_utility = val_utility
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    best_threshold = float(
                        getattr(model, "decision_threshold", 0.5)
                    )
                if lr_scheduler is not None:
                    if isinstance(
                        lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        lr_scheduler.step(val_utility)
                    else:
                        lr_scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            epoch_log = {
                "epoch": epoch,
                **means,
                "learning_rate": float(current_lr),
                "val_utility": val_utility,
                "val_alpha": val_alpha,
            }
            if should_evaluate:
                epoch_log.update({
                    f"val_{k}": float(v)
                    for k, v in self.last_eval_metrics.items()
                })
                epoch_log.update({
                    f"val_{key}_utility": float(value)
                    for key, value in channel_val_metrics.items()
                })
            logs.append(epoch_log)
            for key, value in epoch_log.items():
                if key != "epoch" and value is not None:
                    writer.add_scalar(key, value, epoch)
            print(
                f"[Epoch {epoch}] loss={means['train_loss']:.4f}, "
                f"utility={means['utility']:.4f}, "
                f"pi={means['pi_mean']:.4f}±{means['pi_std']:.4f}, "
                f"grad={means['grad_norm_mean']:.4f}, lr={current_lr:.6f}"
            )
            if should_evaluate:
                print(
                    f"  val_utility={val_utility:.4f}, "
                    f"diagnostics={self.last_eval_metrics}"
                )

        final_state = copy.deepcopy(model.state_dict())
        final_threshold = float(getattr(model, "decision_threshold", 0.5))
        if best_state is not None:
            model.load_state_dict(best_state)
            model.decision_threshold = best_threshold
            print(
                f"[Trainer] Restored best hard-validation checkpoint: "
                f"epoch={best_epoch}, utility={best_utility:.4f}"
            )
        writer.close()
        return {
            "model": model,
            "final_model_state_dict": final_state,
            "final_decision_threshold": final_threshold,
            "per_epoch_logs": logs,
            "best_val_util": best_utility,
            "best_epoch": best_epoch,
            "final_train_loss": logs[-1]["train_loss"] if logs else None,
        }
