"""TensorFlow checkpoint checks for the retained DistGCN source tree."""

from __future__ import annotations

from pathlib import Path


def resolve_checkpoint_prefix(path):
    """Resolve a checkpoint directory, prefix, or ``.index`` path."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        checkpoint_file = path / "checkpoint"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"missing TensorFlow checkpoint manifest: {checkpoint_file}")
        text = checkpoint_file.read_text(encoding="utf-8", errors="replace")
        marker = 'model_checkpoint_path: "'
        line = next((item for item in text.splitlines() if item.startswith(marker)), None)
        if line is None:
            raise ValueError(f"invalid TensorFlow checkpoint manifest: {checkpoint_file}")
        name = line[len(marker):].split('"', 1)[0]
        prefix = path / name
    else:
        value = str(path)
        for suffix in (".index", ".meta", ".data-00000-of-00001"):
            if value.endswith(suffix):
                value = value[: -len(suffix)]
                break
        prefix = Path(value)
    required = [Path(str(prefix) + ".index"), Path(str(prefix) + ".meta")]
    data_files = list(prefix.parent.glob(prefix.name + ".data-*-of-*"))
    missing = [str(item) for item in required if not item.is_file()]
    if not data_files:
        missing.append(str(prefix) + ".data-*-of-*")
    if missing:
        raise FileNotFoundError("incomplete DistGCN checkpoint: " + ", ".join(missing))
    return prefix


def restore_and_inspect_checkpoint(path):
    """Import the original TF1 meta graph and restore every variable."""
    prefix = resolve_checkpoint_prefix(path)
    try:
        import tensorflow.compat.v1 as tf
    except ImportError as exc:
        raise RuntimeError("DistGCN requires its independent TensorFlow environment") from exc
    tf.disable_v2_behavior()
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.train.import_meta_graph(str(prefix) + ".meta", clear_devices=True)
        variables = tf.global_variables()
        with tf.Session(graph=graph) as session:
            saver.restore(session, str(prefix))
            shapes = {variable.name: variable.shape.as_list() for variable in variables}
    if not shapes:
        raise RuntimeError("DistGCN checkpoint restored but contains no variables")
    return {
        "checkpoint_prefix": str(prefix),
        "variable_count": len(shapes),
        "variables": shapes,
    }


__all__ = ["resolve_checkpoint_prefix", "restore_and_inspect_checkpoint"]
