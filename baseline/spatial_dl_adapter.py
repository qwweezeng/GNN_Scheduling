"""Strict full-link adapter for the retained Spatial-DL checkpoint.

The public parameter file trained the checkpoint with 50 links, but every
trainable tensor is independent of link cardinality: the convolution kernel
is spatial and the MLP is shared link-wise.  We therefore rebuild only the
TensorFlow placeholders for the requested link count while restoring the
original weights unchanged.  Variable-length states are padded within a
batch, masked, and mapped back to the original project link indices.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


CHECKPOINT_TRAINING_LINK_COUNT = 50
# Backward-compatible name for code that records the original training size.
LEGACY_LINK_COUNT = CHECKPOINT_TRAINING_LINK_COUNT
LEGACY_FILTER_SIZE = 63
LEGACY_CELL_LENGTH_M = 5.0


@dataclass(frozen=True)
class SpatialDLMapping:
    project_link_count: int
    selected_project_indices: np.ndarray
    valid_mask: np.ndarray
    horizontal_scale: float
    scope: str

    def project_to_legacy(self, values):
        array = np.asarray(values)
        if array.shape[0] != self.project_link_count:
            raise ValueError(
                f"expected {self.project_link_count} project-link values, "
                f"got {array.shape}"
            )
        result = np.zeros(
            (len(self.valid_mask),) + array.shape[1:], dtype=array.dtype
        )
        count = len(self.selected_project_indices)
        result[:count] = array[self.selected_project_indices]
        return result

    def legacy_to_selected(self, values):
        array = np.asarray(values)
        if array.shape[0] != len(self.valid_mask):
            raise ValueError(
                f"expected {len(self.valid_mask)} adapted-link values, got {array.shape}"
            )
        return array[: len(self.selected_project_indices)]

    def legacy_to_project(self, values, *, omitted_value=0):
        selected = self.legacy_to_selected(values)
        result = np.full(
            (self.project_link_count,) + selected.shape[1:],
            omitted_value,
            dtype=selected.dtype,
        )
        result[self.selected_project_indices] = selected
        return result

    def metadata(self):
        return {
            "adapter": "spatial_dl_dynamic_cardinality_xy_pad_mask_v2",
            "project_link_count": int(self.project_link_count),
            "checkpoint_training_link_count": CHECKPOINT_TRAINING_LINK_COUNT,
            "inference_link_count": int(len(self.valid_mask)),
            "cardinality_transfer": bool(
                len(self.valid_mask) != CHECKPOINT_TRAINING_LINK_COUNT
            ),
            "selected_project_indices": (
                "identity"
                if self.scope == "full_link_set"
                else self.selected_project_indices.tolist()
            ),
            "real_link_count": int(self.valid_mask.sum()),
            "horizontal_scale": float(self.horizontal_scale),
            "scope": self.scope,
            "height_mapping": "drop_z_then_uniformly_scale_xy_if_needed",
        }


def _selected_indices(num_links, common_subproblem_indices):
    if common_subproblem_indices is None:
        return np.arange(num_links, dtype=np.int64), "full_link_set"

    indices = np.asarray(common_subproblem_indices, dtype=np.int64).reshape(-1)
    if indices.size == 0:
        raise ValueError("common_subproblem_indices must not be empty")
    if np.unique(indices).size != indices.size:
        raise ValueError("common_subproblem_indices must be unique")
    if np.any(indices < 0) or np.any(indices >= num_links):
        raise IndexError("common_subproblem_indices contains an invalid project link index")
    scope = (
        "full_link_set"
        if indices.size == num_links and np.array_equal(indices, np.arange(num_links))
        else "explicit_common_subproblem"
    )
    return indices, scope


def project_3d_links_to_spatial_dl(
    positions,
    links,
    *,
    common_subproblem_indices=None,
    target_link_count=None,
):
    """Return one padded 2D layout, mask, and reversible link mapping."""
    positions = np.asarray(positions, dtype=np.float64)
    links = np.asarray(links, dtype=np.int64)
    if positions.ndim != 2 or positions.shape[1] < 3:
        raise ValueError("positions must have shape [num_uavs,3]")
    if links.ndim != 2 or links.shape[1] != 2:
        raise ValueError("links must have shape [num_links,2]")
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    if links.size and (np.any(links < 0) or np.any(links >= positions.shape[0])):
        raise IndexError("links contains an invalid UAV endpoint")

    indices, scope = _selected_indices(len(links), common_subproblem_indices)
    if target_link_count is None:
        target_link_count = max(1, len(indices))
    target_link_count = int(target_link_count)
    if target_link_count < max(1, len(indices)):
        raise ValueError(
            "target_link_count must fit every selected real link; links are never cropped"
        )
    selected_links = links[indices]
    xy = positions[:, :2].copy()
    if len(selected_links):
        offsets = xy[selected_links[:, 0]] - xy[selected_links[:, 1]]
        max_axis_offset = float(np.max(np.abs(offsets)))
    else:
        max_axis_offset = 0.0
    half_span_m = ((LEGACY_FILTER_SIZE - 1) // 2) * LEGACY_CELL_LENGTH_M
    scale = 1.0 if max_axis_offset <= half_span_m else half_span_m / max_axis_offset
    xy *= scale

    layout = np.zeros((target_link_count, 4), dtype=np.float32)
    for legacy_index, (tx, rx) in enumerate(selected_links):
        layout[legacy_index, :2] = xy[tx]
        layout[legacy_index, 2:] = xy[rx]
    valid_mask = np.zeros(target_link_count, dtype=np.float32)
    valid_mask[: len(indices)] = 1.0
    mapping = SpatialDLMapping(
        project_link_count=len(links),
        selected_project_indices=indices,
        valid_mask=valid_mask,
        horizontal_scale=scale,
        scope=scope,
    )
    return layout, valid_mask, mapping


def _checkpoint_prefix(source_root):
    prefix = Path(source_root) / "Neural_Network_Model" / "Trained_Model" / "conv_net_model.ckpt"
    required = [
        Path(str(prefix) + ".index"),
        Path(str(prefix) + ".meta"),
        Path(str(prefix) + ".data-00000-of-00001"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Spatial-DL checkpoint is incomplete: " + ", ".join(missing))
    return prefix


def run_spatial_dl_checkpoint(layouts, valid_masks, *, source_root, seed=0):
    """Restore the TF1 checkpoint and return masked binary schedules."""
    layouts = np.asarray(layouts, dtype=np.float32)
    masks = np.asarray(valid_masks, dtype=np.float32)
    if layouts.ndim == 2:
        layouts = layouts[None, :, :]
    if masks.ndim == 1:
        masks = masks[None, :]
    if layouts.ndim != 3 or layouts.shape[2] != 4 or layouts.shape[1] < 1:
        raise ValueError("layouts must have shape [B,L,4] with L >= 1")
    if masks.shape != layouts.shape[:2]:
        raise ValueError("valid_masks must have shape [B,L]")
    if not np.all((masks == 0.0) | (masks == 1.0)):
        raise ValueError("valid_masks must be binary")

    source_root = Path(source_root).resolve()
    model_dir = source_root / "Neural_Network_Model"
    utilities_dir = source_root / "Utilities"
    for path in (model_dir, utilities_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    prefix = _checkpoint_prefix(source_root)

    try:
        import tensorflow.compat.v1 as tf
    except ImportError as exc:
        raise RuntimeError(
            "Spatial DL requires its independent TensorFlow legacy environment"
        ) from exc
    tf.disable_v2_behavior()
    import Convolutional_Neural_Network_Model
    import general_parameters
    import helper_functions

    parameters = general_parameters.parameters()
    parameters.n_links = int(layouts.shape[1])
    model = Convolutional_Neural_Network_Model.Conv_Network(parameters, len(layouts))
    with model.TFgraph.as_default():
        tf.set_random_seed(int(seed))
        model.build_network()
        saver = tf.train.Saver()
        processed = helper_functions.process_layouts_inputs(parameters, layouts)
        feed = {
            model.placeholders["tx_indices_hash"]: processed["tx_indices_hash"],
            model.placeholders["rx_indices_hash"]: processed["rx_indices_hash"],
            model.placeholders["tx_indices_extract"]: processed["tx_indices_ext"],
            model.placeholders["rx_indices_extract"]: processed["rx_indices_ext"],
            model.placeholders["pair_tx_convfilter_indices"]: processed["pair_tx_convfilter_indices"],
            model.placeholders["pair_rx_convfilter_indices"]: processed["pair_rx_convfilter_indices"],
            model.placeholders["subset_links"]: masks,
        }
        with tf.Session(graph=model.TFgraph) as session:
            saver.restore(session, str(prefix))
            schedules = session.run(model.outputs_final, feed_dict=feed)
    schedules = np.asarray(schedules, dtype=np.float32) * masks
    if schedules.shape != masks.shape or not np.all((schedules == 0.0) | (schedules == 1.0)):
        raise RuntimeError("Spatial-DL checkpoint returned a non-binary or mis-shaped action")
    if np.any(schedules[masks == 0.0] != 0.0):
        raise RuntimeError("Spatial-DL padding mask failed")
    return schedules


def run_spatial_dl_checkpoint_isolated(
    layouts,
    valid_masks,
    *,
    source_root,
    seed=0,
    conda_env="gnn-spatial-dl-legacy",
):
    """Run the checkpoint in its conda env without modifying main PyTorch."""
    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError("conda is required to launch the isolated Spatial-DL environment")
    worker = Path(__file__).with_name("spatial_dl_worker.py").resolve()
    if not worker.is_file():
        raise FileNotFoundError(f"missing Spatial-DL subprocess worker: {worker}")

    with tempfile.TemporaryDirectory(prefix="spatial-dl-") as directory:
        directory = Path(directory)
        input_path = directory / "input.npz"
        output_path = directory / "output.npz"
        np.savez_compressed(
            input_path,
            layouts=np.asarray(layouts, dtype=np.float32),
            valid_masks=np.asarray(valid_masks, dtype=np.float32),
        )
        environment = os.environ.copy()
        environment.update({
            "MPLCONFIGDIR": str(directory / "matplotlib"),
            "TF_CPP_MIN_LOG_LEVEL": "3",
        })
        command = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            str(conda_env),
            "python",
            str(worker),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--source-root",
            str(Path(source_root).resolve()),
            "--seed",
            str(int(seed)),
        ]
        completed = subprocess.run(
            command,
            cwd=str(worker.parents[1]),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"isolated Spatial-DL checkpoint process failed ({conda_env}): {detail}"
            )
        if not output_path.is_file():
            raise RuntimeError("isolated Spatial-DL process returned no schedule artifact")
        schedules = np.load(output_path, allow_pickle=False)["schedules"]

    masks = np.asarray(valid_masks, dtype=np.float32)
    if schedules.shape != masks.shape or not np.all((schedules == 0) | (schedules == 1)):
        raise RuntimeError("isolated Spatial-DL process returned an invalid action")
    if np.any(schedules[masks == 0] != 0):
        raise RuntimeError("isolated Spatial-DL process activated padded links")
    return schedules.astype(np.float32, copy=False)


__all__ = [
    "LEGACY_LINK_COUNT",
    "SpatialDLMapping",
    "project_3d_links_to_spatial_dl",
    "run_spatial_dl_checkpoint",
    "run_spatial_dl_checkpoint_isolated",
]
