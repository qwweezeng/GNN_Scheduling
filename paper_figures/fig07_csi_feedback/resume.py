#!/usr/bin/env python3
"""Resume Fig. 7 CSI experiments without replacing completed outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.marginal_mp import MARGINAL_PROTOCOL_VERSION

CSI_CASES = (
    ("perfect", None, "perfect"),
    ("large_scale", None, "large_scale"),
    ("quantized", 2, "quantized_2bit"),
    ("quantized", 4, "quantized_4bit"),
    ("quantized", 6, "quantized_6bit"),
    ("quantized", 8, "quantized_8bit"),
    ("quantized", 10, "quantized_10bit"),
)
MOBILE_METHODS = (
    "latest,large_scale,quantized_2bit,quantized_4bit,"
    "quantized_6bit,quantized_8bit,quantized_10bit"
)


def call(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def mobile_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("records", [])


def mobile_complete(path: Path, expected_methods: set[str]) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != "paper_mobile_imperfect_csi_v2_matched_models":
        return False
    if payload.get("marginal_protocol_version") != MARGINAL_PROTOCOL_VERSION:
        return False
    records = payload.get("records", [])
    return expected_methods.issubset({record["method"] for record in records})


def chapter4_checkpoint(num_uav: int, csi_mode: str, q_bits: int | None) -> Path:
    csi_tag = csi_mode if q_bits is None else f"quantized{q_bits}bit"
    return ROOT / "results/models/paper" / (
        f"gat_numUAV={num_uav}_tx=10dBm_csi={csi_tag}_channel=ricianK5dB.pth"
    )


def resume_chapter4(num_uav: int, device: str, epochs: int) -> None:
    for csi_mode, q_bits, suffix in CSI_CASES:
        output = HERE / "data/raw" / f"chapter4_N{num_uav}_v2_{suffix}.json"
        if output.exists():
            print(f"[skip] {output} already exists", flush=True)
            continue

        checkpoint = chapter4_checkpoint(num_uav, csi_mode, q_bits)
        if not checkpoint.exists():
            train = [
                sys.executable, "-u", "train/train_channel_gat_rician.py",
                "--config", "configs/paper.yaml", "--num_uav", str(num_uav),
                "--train_size", "300", "--val_size", "128", "--test_size", "128",
                "--epochs", str(epochs), "--learning_rate", "1e-4",
                "--seed", "10086", "--val_seed", "10087", "--test_seed", "10088",
                "--rician_k_db", "5", "--csi-mode", csi_mode,
                "--checkpoint_out", str(checkpoint), "--device", device,
            ]
            if q_bits is not None:
                train.extend(["--q-bits", str(q_bits)])
            call(train)

        evaluate = [
            sys.executable, "-u", "experiments/evaluate_channel_gat_csi.py",
            "--num-uav", str(num_uav), "--csi-mode", csi_mode,
            "--device", device, "--output", str(output),
        ]
        if q_bits is not None:
            evaluate.extend(["--q-bits", str(q_bits)])
        call(evaluate)


def resume_chapter5(
    num_uav: int,
    slots: int,
    seeds: str,
    arrival_rate: str,
    device: str,
    output_root: Path,
) -> None:
    output = output_root / f"chapter5_N{num_uav}_v2_matched_csi.json"
    progress = output.with_suffix(".progress.json")
    expected = set(MOBILE_METHODS.split(","))
    if mobile_complete(output, expected) or mobile_complete(progress, expected):
        print(f"[skip] N={num_uav} Chapter V CSI records are complete", flush=True)
        return
    call([
        sys.executable, "-u", "experiments/evaluate_mobile_block_fading_csi.py",
        "--num-uav", str(num_uav), "--num-slots", str(slots),
        "--seeds", seeds, "--speeds-mps", "0", "--slots-per-block", "1",
        "--stale-blocks", "1", "--arrival-rates", arrival_rate,
        "--csi-methods", MOBILE_METHODS, "--scheduler-variants", "gat_assisted",
        "--rounds", "4", "--device", device, "--output", str(output),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,30,40,50")
    parser.add_argument("--slots", type=int, default=10000)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--arrival-rate", default="2.4")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--chapters", default="4,5",
        help="Comma-separated chapters to run: 4, 5, or 4,5 (default).",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--chapter5-output-root",
        default="",
        help="Optional new Chapter V result root; existing paper data is untouched.",
    )
    args = parser.parse_args()
    chapter5_output_root = (
        Path(args.chapter5_output_root)
        if args.chapter5_output_root else HERE / "data/raw"
    )
    if not chapter5_output_root.is_absolute():
        chapter5_output_root = ROOT / chapter5_output_root

    chapters = {int(value) for value in args.chapters.split(",") if value.strip()}
    if not chapters or not chapters.issubset({4, 5}):
        raise ValueError("--chapters must contain only 4 and/or 5")

    for num_uav in (int(value) for value in args.nodes.split(",") if value.strip()):
        if 4 in chapters:
            resume_chapter4(num_uav, args.device, args.epochs)
        if 5 in chapters:
            resume_chapter5(
                num_uav,
                args.slots,
                args.seeds,
                args.arrival_rate,
                args.device,
                chapter5_output_root,
            )


if __name__ == "__main__":
    main()
