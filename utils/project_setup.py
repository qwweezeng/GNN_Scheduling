"""Small shared constructors for reproducible paper experiments."""

import json
from pathlib import Path

import torch
import yaml

from models.model import GATScheduler
from utils.network_generate import NetworkGenerator


MODEL_DIR = Path("results/models/paper")
PROPOSED_GAT_CHECKPOINT_ROOT = Path("checkpoints/paper/proposed_gat/rician_K5_tx10")


def proposed_gat_checkpoint_dir(
    num_uav: int,
    model_seed: int,
    training_scope: str,
) -> Path:
    """Return the canonical directory for a reproducible Proposed-GAT run."""
    if not training_scope:
        raise ValueError("training_scope must be a non-empty provenance label")
    return (
        PROPOSED_GAT_CHECKPOINT_ROOT
        / f"N{int(num_uav)}"
        / f"seed{int(model_seed)}"
        / training_scope
    )


def load_config(path="configs/paper.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def physics_tag(config):
    tx_power = float(config.get("TX_POWER", 10.0))
    tx_text = str(int(tx_power)) if tx_power.is_integer() else str(tx_power)
    csi_mode = str(config.get("CSI_MODE", "perfect"))
    if csi_mode == "quantized":
        q_bits = config.get("Q_BITS")
        csi_tag = f"quantized{int(q_bits)}bit" if q_bits is not None else "quantized"
    else:
        csi_tag = csi_mode
    base = f"tx={tx_text}dBm_csi={csi_tag}"
    channel_model = str(config.get("CHANNEL_MODEL", "rayleigh")).lower()
    if channel_model == "rayleigh":
        return base
    if channel_model == "pure_los":
        return f"{base}_channel=pureLoS"
    k_db = float(config.get("RICIAN_K_DB", 5.0))
    k_text = str(int(k_db)) if k_db.is_integer() else str(k_db).replace(".", "p")
    return f"{base}_channel=ricianK{k_text}dB"


def channel_checkpoint_path(config, train_num_uav, train_size=None):
    base = f"gat_numUAV={int(train_num_uav)}_{physics_tag(config)}"
    if train_size is not None:
        base += f"_trainSize={int(train_size)}"
    return MODEL_DIR / f"{base}.pth"


def queue_checkpoint_path(config, train_num_uav):
    return MODEL_DIR / (
        f"gat_queue_mws_v3_numUAV={int(train_num_uav)}_"
        f"{physics_tag(config)}.pth"
    )


def queue_training_cache_path(config, train_num_uav):
    return Path("datasets/paper/queue_gat_baseline") / (
        f"dataset_numUAV={int(train_num_uav)}_"
        f"{physics_tag(config)}_formal.pkl"
    )


def build_network_generator(config):
    height = config.get("HEIGHT_RANGE", [50, 150])
    return NetworkGenerator(
        num_uavs=config.get("NUM_UAV", 30),
        area_width=config.get("AREA_WIDTH", 1000),
        height_min=height[0],
        height_max=height[1],
        uav_min_distance=config.get("MIN_DISTANCE", 20),
        tx_power=config.get("TX_POWER", 10),
        Gt=config.get("G_T", 0),
        Gr=config.get("G_R", 0),
        noise_psd=config.get("NOISE_PSD", -169),
        bandwidth=config.get("BANDWIDTH", 5e6),
        carrier_f=config.get("CARRIER_FREQ", 2.4e9),
        snr_thres=config.get("SNR_THRES", 5),
        snr_gap_db=config.get("SNR_GAP_DB", 0),
        channel_model=config.get("CHANNEL_MODEL", "rayleigh"),
        rician_k_db=config.get("RICIAN_K_DB", 5.0),
    )


def build_gat_scheduler(
    config,
    raw_node_dim,
    input_feature_mean=None,
    input_feature_std=None,
    use_input_skip=False,
):
    return GATScheduler(
        raw_node_dim=raw_node_dim,
        mlp_hidden_dims=config.get("MLP_HIDDEN_DIMS", [32, 32]),
        embed_node_dim=config.get("NODE_EMBED_DIM", 32),
        edge_dim=config.get("EDGE_DIM", 1),
        gat_out_dim=config.get("GAT_OUT_DIM", 48),
        num_heads=config.get("NUM_HEADS", 4),
        T_iterations=config.get("NUM_GAT_LAYERS", 4),
        schedule_hidden_dim=config.get("SCHEDULE_HIDDEN", 32),
        dropout=config.get("DROPOUT", 0.0),
        use_power_control=False,
        input_feature_mean=input_feature_mean,
        input_feature_std=input_feature_std,
        use_input_skip=use_input_skip,
        initial_schedule_bias=0.0 if use_input_skip else -0.5,
        stochastic_ste=use_input_skip,
    )


def load_channel_gat(
    config,
    train_num_uav=10,
    device="cpu",
    *,
    expected_dataset_fingerprint=None,
    train_size=None,
):
    path = channel_checkpoint_path(config, train_num_uav, train_size)
    if not path.exists():
        raise FileNotFoundError(f"Channel-only GAT checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "model_version": "channel_gat_v1",
        "tx_power_dbm": float(config.get("TX_POWER", 10.0)),
        "csi_mode": config.get("CSI_MODE", "perfect"),
        "num_uav": int(train_num_uav),
    }
    channel_model = str(config.get("CHANNEL_MODEL", "rayleigh")).lower()
    if channel_model != "rayleigh":
        required["channel_model"] = channel_model
        if channel_model == "rician":
            required["rician_k_db"] = float(config.get("RICIAN_K_DB", 5.0))
    mismatches = {
        key: (checkpoint.get(key), expected)
        for key, expected in required.items()
        if checkpoint.get(key) != expected
    }
    if mismatches or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Channel checkpoint metadata mismatch in {path}: {mismatches}"
        )
    if (
        expected_dataset_fingerprint is not None
        and checkpoint.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise ValueError(
            f"Channel checkpoint dataset fingerprint mismatch in {path}: "
            f"{checkpoint.get('dataset_fingerprint')!r} != "
            f"{expected_dataset_fingerprint!r}"
        )
    output_audit = checkpoint.get("output_audit", {})
    collapsed_all_active = (
        float(output_audit.get("all_links_active_state_fraction", 0.0)) >= 0.95
    )
    collapsed_all_inactive = (
        float(output_audit.get("all_links_inactive_state_fraction", 0.0)) >= 0.95
    )
    if (
        checkpoint.get("integrity_status") == "rejected"
        or collapsed_all_active
        or collapsed_all_inactive
    ):
        raise ValueError(
            f"Rejected collapsed channel-only GAT in {path}: "
            f"integrity_status={checkpoint.get('integrity_status')}, "
            f"output_audit={output_audit}. Retrain a sparse conference-style "
            "scheduler before using it in LocalPhysicalReuse."
        )
    model = build_gat_scheduler(config, raw_node_dim=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), str(path)


def load_channel_gat_checkpoint(
    config,
    path,
    train_num_uav,
    device="cpu",
    allow_power_transfer=False,
    expected_dataset_fingerprint=None,
):
    """Load an explicitly selected audited channel-only GAT checkpoint.

    This path is used by the power-transfer audit before a model is promoted to
    ``results/models``.  Channel family, Rician K factor, UAV count, and CSI
    mode remain strict.  A train/deployment power mismatch is accepted only
    when the caller labels it explicitly via ``allow_power_transfer``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Channel-only GAT checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    version = checkpoint.get("model_version")
    supported = {
        "channel_gat_v1",
        "conference_gat_power_transfer_v1",
        "channel_gat_regeneration_audit_v1",
    }
    if version not in supported or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Unsupported channel checkpoint schema in {path}: {version!r}"
        )

    train_config = checkpoint.get("config", {}) or {}
    checkpoint_num_uav = int(train_config.get("NUM_UAV", train_num_uav))
    expected_channel = str(config.get("CHANNEL_MODEL", "rayleigh")).lower()
    checkpoint_channel = str(
        checkpoint.get(
            "channel_model", train_config.get("CHANNEL_MODEL", "rayleigh")
        )
    ).lower()
    checkpoint_csi = checkpoint.get(
        "csi_mode", train_config.get("CSI_MODE", "perfect")
    )
    expected_csi = config.get("CSI_MODE", "perfect")
    train_power = float(checkpoint.get(
        "train_power_dbm",
        checkpoint.get("tx_power_dbm", train_config.get("TX_POWER", 10.0)),
    ))
    deployment_power = float(config.get("TX_POWER", 10.0))
    mismatches = {}
    if checkpoint_num_uav != int(train_num_uav):
        mismatches["num_uav"] = (checkpoint_num_uav, int(train_num_uav))
    if checkpoint_channel != expected_channel:
        mismatches["channel_model"] = (checkpoint_channel, expected_channel)
    if checkpoint_csi != expected_csi:
        mismatches["csi_mode"] = (checkpoint_csi, expected_csi)
    if expected_channel == "rician":
        checkpoint_k = float(
            checkpoint.get(
                "rician_k_db", train_config.get("RICIAN_K_DB", 5.0)
            )
        )
        expected_k = float(config.get("RICIAN_K_DB", 5.0))
        if checkpoint_k != expected_k:
            mismatches["rician_k_db"] = (checkpoint_k, expected_k)
    if train_power != deployment_power and not allow_power_transfer:
        mismatches["train_vs_deployment_power_dbm"] = (
            train_power, deployment_power
        )
    if mismatches:
        raise ValueError(
            f"Channel checkpoint metadata mismatch in {path}: {mismatches}"
        )
    if (
        expected_dataset_fingerprint is not None
        and checkpoint.get("dataset_fingerprint") != expected_dataset_fingerprint
    ):
        raise ValueError(
            f"Channel checkpoint dataset fingerprint mismatch in {path}: "
            f"{checkpoint.get('dataset_fingerprint')!r} != "
            f"{expected_dataset_fingerprint!r}"
        )

    output_audit = checkpoint.get("output_audit")
    if output_audit is None and version == "conference_gat_power_transfer_v1":
        report_path = path.with_name("report.json")
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            output_audit = report.get("transfer", {}).get(
                format(deployment_power, "g"), {}
            ).get("output_audit")
    if not output_audit:
        raise ValueError(
            f"Checkpoint {path} has no output audit for deployment power "
            f"{deployment_power:g} dBm"
        )
    if not bool(output_audit.get("integrity_passed", False)):
        raise ValueError(
            f"Rejected collapsed channel-only GAT in {path}: "
            f"output_audit={output_audit}"
        )

    architecture_config = dict(config)
    architecture_config.update(train_config)
    model = build_gat_scheduler(architecture_config, raw_node_dim=1)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), str(path)


def load_end_to_end_queue_gat(config, path, device="cpu"):
    """Load the direct GAT-q baseline with strict physical metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"End-to-end GAT-q checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {
        "model_version": "mws_v3",
        "queue_feature": "relative_max",
        "training_decision": "stochastic_ste",
        "tx_power_dbm": float(config.get("TX_POWER", 10.0)),
        "csi_mode": config.get("CSI_MODE", "perfect"),
        "channel_model": str(config.get("CHANNEL_MODEL", "rician")).lower(),
        "rician_k_db": float(config.get("RICIAN_K_DB", 5.0)),
        "num_uav": int(config.get("NUM_UAV", 20)),
    }
    mismatches = {
        key: (checkpoint.get(key), expected)
        for key, expected in required.items()
        if checkpoint.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"End-to-end GAT-q checkpoint metadata mismatch in {path}: "
            f"{mismatches}. Retrain it under the formal 10 dBm setting."
        )
    model = build_gat_scheduler(
        config,
        raw_node_dim=2,
        input_feature_mean=checkpoint["input_feature_mean"],
        input_feature_std=checkpoint["input_feature_std"],
        use_input_skip=checkpoint.get("use_input_skip", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.decision_threshold = float(
        checkpoint.get("decision_threshold", 0.5)
    )
    return model.to(device).eval(), str(path)
