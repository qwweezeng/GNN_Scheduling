"""Optional link-local complex/geometry CSI prediction branch.

The predictor is trained centrally but evaluated independently for every
directed UAV pair.  It has no recurrent state and performs no graph message
passing.  Its output is the normalized current dB power residual that is
combined with current large-scale power before Direct-Link MarginalMP.
"""

import math
from pathlib import Path

import torch
from torch import nn


PAPER_MODULE_NAME = "Complex-Geometry CSI Predictor"
PAPER_FULL_METHOD_NAME = "CG-CSI-Assisted Direct-Link MarginalMP"


class ComplexGeometryCSIPredictor(nn.Module):
    """A 897-parameter shared MLP for one directed UAV-pair observation."""

    model_version = "complex_geometry_csi_mlp_v1"

    def __init__(self, input_dim=10, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        return self.net(features).squeeze(1)

    def initial_hidden(self, num_pairs, device=None):
        return None

    def forward_step(self, features, hidden=None):
        return self(features), None


def directed_pair_catalog(num_uav):
    """Return every non-self directed UAV pair in a stable order."""
    return torch.tensor(
        [
            (source, target)
            for source in range(int(num_uav))
            for target in range(int(num_uav))
            if source != target
        ],
        dtype=torch.long,
    )


def complex_geometry_rows(
    trace,
    pairs,
    stale_blocks,
    speed_mps,
    training_maximum_age,
    minimum_distance,
    carrier_frequency_hz,
    tx_power_w,
):
    """Build legal aged-CSI features and current residual targets.

    Targets are returned for training/auditing.  In deployment only the
    feature tensor is consumed; current true channel values are never inputs.
    """
    pairs = torch.as_tensor(pairs, dtype=torch.long)
    tx, rx = pairs[:, 0], pairs[:, 1]
    wavelength = 3.0e8 / float(carrier_frequency_hz)
    rows = []
    age = int(stale_blocks)
    for block_index in range(int(training_maximum_age), len(trace.blocks)):
        previous = trace.blocks[block_index - age]
        current = trace.blocks[block_index]
        previous_amplitude = torch.sqrt(
            previous.large_scale_power[tx, rx] / float(tx_power_w)
        ).clamp_min(1e-30)
        normalized_previous = previous.true_channel[tx, rx] / previous_amplitude
        previous_distance = torch.linalg.vector_norm(
            previous.positions[tx] - previous.positions[rx], dim=1
        ).clamp_min(float(minimum_distance))
        current_distance = torch.linalg.vector_norm(
            current.positions[tx] - current.positions[rx], dim=1
        ).clamp_min(float(minimum_distance))
        previous_angle = 2.0 * math.pi * previous_distance / wavelength
        current_angle = 2.0 * math.pi * current_distance / wavelength
        rho_age = float(current.fading_rho) ** age
        features = torch.stack([
            normalized_previous.real.float(),
            normalized_previous.imag.float(),
            torch.cos(previous_angle),
            torch.sin(previous_angle),
            torch.cos(current_angle),
            torch.sin(current_angle),
            torch.full_like(current_distance, rho_age),
            ((current_distance - previous_distance) / wavelength).float(),
            torch.full_like(
                current_distance, float(age) / float(training_maximum_age)
            ),
            torch.full_like(current_distance, float(speed_mps) / 5.0),
        ], dim=1)
        target = 10.0 * torch.log10(
            (
                current.true_power[tx, rx]
                / current.large_scale_power[tx, rx].clamp_min(1e-30)
            ).clamp_min(1e-30)
        ).float()
        rows.append({
            "block_index": block_index,
            "features": features,
            "target_residual_db": target,
        })
    return rows


def power_matrix_from_residual(block, pairs, residual_db):
    matrix = torch.zeros_like(block.large_scale_power)
    pairs = torch.as_tensor(pairs, dtype=torch.long)
    tx, rx = pairs[:, 0], pairs[:, 1]
    matrix[tx, rx] = block.large_scale_power[tx, rx] * torch.pow(
        10.0, torch.as_tensor(residual_db) / 10.0
    )
    return matrix


def load_cg_csi_predictor(path, expected_num_uav=20, expected_rician_k_db=5.0):
    """Load the accepted standalone predictor with strict metadata checks."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "model_version": ComplexGeometryCSIPredictor.model_version,
        "decision": "ACCEPT_COMPLEX_GEOMETRY_MLP_CSI_MODULE",
        "num_uav": int(expected_num_uav),
        "rician_k_db": float(expected_rician_k_db),
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"CG-CSI checkpoint metadata mismatch: {mismatches}")
    model = ComplexGeometryCSIPredictor(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def predict_power_matrices(
    model,
    checkpoint,
    trace,
    stale_blocks,
    speed_mps,
    minimum_distance,
    carrier_frequency_hz,
    tx_power_w,
    training_maximum_age=4,
):
    """Predict one coherent current power matrix for every usable block."""
    pairs = checkpoint["pair_catalog"].long()
    rows = complex_geometry_rows(
        trace,
        pairs,
        stale_blocks,
        speed_mps,
        training_maximum_age,
        minimum_distance,
        carrier_frequency_hz,
        tx_power_w,
    )
    powers = [block.large_scale_power.clone() for block in trace.blocks]
    for row in rows:
        normalized = (
            row["features"] - checkpoint["feature_mean"]
        ) / checkpoint["feature_std"]
        residual_db = (
            model(normalized).cpu() * checkpoint["target_std"]
            + checkpoint["target_mean"]
        )
        powers[row["block_index"]] = power_matrix_from_residual(
            trace.blocks[row["block_index"]], pairs, residual_db
        )
    return powers


__all__ = [
    "PAPER_MODULE_NAME",
    "PAPER_FULL_METHOD_NAME",
    "ComplexGeometryCSIPredictor",
    "complex_geometry_rows",
    "directed_pair_catalog",
    "load_cg_csi_predictor",
    "power_matrix_from_residual",
    "predict_power_matrices",
]
