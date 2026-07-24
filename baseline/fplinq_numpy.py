"""Pure NumPy implementation of the retained FPLinQ numerical core.

The project channel stores *received powers* ``P |h|^2`` rather than bare
channel gains.  Consequently this adapter must not multiply the matrix by the
transmit power a second time.  Matrix entry ``g[b, i, j]`` is the received
power at receiver-link ``i`` from transmitter-link ``j``.

This module deliberately has no TensorFlow, PyTorch, or Spatial-DL imports so
that FPLinQ remains an independent centralized numerical baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FPLinQDiagnostics:
    iterations: int
    max_last_update: float
    converged: bool


def _validate_inputs(received_power, noise_power, weights):
    gains = np.asarray(received_power, dtype=np.float64)
    squeeze_batch = gains.ndim == 2
    if squeeze_batch:
        gains = gains[None, :, :]
    if gains.ndim != 3 or gains.shape[1] != gains.shape[2]:
        raise ValueError(
            "received_power must have shape [L,L] or [B,L,L]; "
            f"got {gains.shape}"
        )
    if not np.isfinite(gains).all() or np.any(gains < 0.0):
        raise ValueError("received_power must be finite and nonnegative")

    batch, num_links, _ = gains.shape
    if weights is None:
        utility_weights = np.ones((batch, num_links), dtype=np.float64)
    else:
        utility_weights = np.asarray(weights, dtype=np.float64)
        if utility_weights.ndim == 1:
            utility_weights = utility_weights[None, :]
        if utility_weights.shape == (1, num_links) and batch > 1:
            utility_weights = np.repeat(utility_weights, batch, axis=0)
        if utility_weights.shape != (batch, num_links):
            raise ValueError(
                "weights must have shape [L] or [B,L] matching gains; "
                f"got {utility_weights.shape} for gains {gains.shape}"
            )
        if not np.isfinite(utility_weights).all() or np.any(utility_weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")

    noise = np.asarray(noise_power, dtype=np.float64)
    if noise.ndim == 0:
        noise = np.full((batch, 1, 1), float(noise), dtype=np.float64)
    elif noise.ndim == 1 and noise.shape == (batch,):
        noise = noise[:, None, None]
    elif noise.shape != (batch, 1, 1):
        raise ValueError(
            "noise_power must be scalar, [B], or [B,1,1]; "
            f"got {noise.shape}"
        )
    if not np.isfinite(noise).all() or np.any(noise <= 0.0):
        raise ValueError("noise_power must be finite and strictly positive")
    return gains, noise, utility_weights, squeeze_batch


def optimize_fplinq(
    received_power,
    noise_power,
    weights=None,
    *,
    iterations=100,
    tolerance=None,
    return_diagnostics=False,
):
    """Return continuous FPLinQ transmit fractions in ``[0,1]``.

    The update is the retained Shen--Yu fractional-programming iteration used
    by the Spatial-DL reference repository, rewritten for a received-power
    matrix.  ``weights`` are full-buffer ones or the current queue values.
    """
    gains, noise, utility_weights, squeeze_batch = _validate_inputs(
        received_power, noise_power, weights
    )
    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if tolerance is not None and float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive when provided")

    batch, num_links, _ = gains.shape
    diagonal = np.diagonal(gains, axis1=1, axis2=2)[:, :, None]
    off_diagonal = gains.copy()
    diagonal_index = np.arange(num_links)
    off_diagonal[:, diagonal_index, diagonal_index] = 0.0
    weights_column = utility_weights[:, :, None]
    allocation = np.ones((batch, num_links, 1), dtype=np.float64)
    last_update = float("inf")
    converged = False

    for completed in range(1, iterations + 1):
        signal = diagonal * allocation
        interference = np.matmul(off_diagonal, allocation) + noise
        z = np.divide(signal, interference, out=np.zeros_like(signal), where=interference > 0.0)

        y_denominator = np.matmul(gains, allocation) + noise
        y_numerator = np.sqrt(np.maximum(signal * weights_column * (z + 1.0), 0.0))
        y = np.divide(
            y_numerator,
            y_denominator,
            out=np.zeros_like(y_numerator),
            where=y_denominator > 0.0,
        )

        x_denominator = np.matmul(np.transpose(gains, (0, 2, 1)), y**2)
        x_numerator = y * np.sqrt(
            np.maximum(weights_column * (z + 1.0) * diagonal, 0.0)
        )
        ratio = np.divide(
            x_numerator,
            x_denominator,
            out=np.zeros_like(x_numerator),
            where=x_denominator > 0.0,
        )
        updated = np.clip(ratio**2, 0.0, 1.0)
        updated[weights_column <= 0.0] = 0.0
        last_update = float(np.max(np.abs(updated - allocation)))
        allocation = updated
        if tolerance is not None and last_update <= float(tolerance):
            converged = True
            break

    result = allocation[:, :, 0]
    if squeeze_batch:
        result = result[0]
    if not return_diagnostics:
        return result
    diagnostics = FPLinQDiagnostics(
        iterations=completed,
        max_last_update=last_update,
        converged=converged,
    )
    return result, diagnostics


def binarize_fplinq(allocation, *, amplitude_threshold=0.5):
    """Apply the retained Spatial-DL convention ``sqrt(x) > threshold``."""
    values = np.asarray(allocation, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("allocation must be finite and nonnegative")
    threshold = float(amplitude_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("amplitude_threshold must lie in [0,1]")
    return (np.sqrt(values) > threshold).astype(np.float32)


__all__ = ["FPLinQDiagnostics", "binarize_fplinq", "optimize_fplinq"]
