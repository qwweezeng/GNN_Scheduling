#!/usr/bin/env python3
"""Generate paired Fig. 5 queue traces and metrics, without plotting.

``--deadline-ms none`` is the original infinite-buffer/no-deadline setting.
``--deadline-ms 2000`` keeps the same arrival/topology trace but drops packets
whose FIFO waiting time exceeds 2000 ms; loss metrics remain in the JSON.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METHODS = (
    "Random,GAT-Assisted MarginalMP,Direct-Link MarginalMP,"
    "GAT-Assisted LocalMax,Direct-Link LocalMax,End-to-End GAT-Q,"
    "Hard-Conflict LGS,Centralized Greedy,Weighted FPLinQ"
)


def call(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", default="20,50")
    parser.add_argument("--slots", type=int, default=10000)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--lambdas", default="0.5,1,2,2.2,2.4,2.6,2.8,3,3.2")
    parser.add_argument("--deadline-ms", default="none", choices=("none", "2000"))
    parser.add_argument("--end-to-end-checkpoint", default="")
    args = parser.parse_args()
    for n in [int(x) for x in args.nodes.split(",") if x.strip()]:
        profile = "no_deadline" if args.deadline_ms == "none" else "deadline_2000ms"
        output = HERE / "data/raw" / profile / f"N{n}"
        command = [
            sys.executable, "-u", "experiments/exp_fixed_csi_qos.py",
            "--num_uav", str(n), "--num_slots", str(args.slots),
            "--arrival_rates", args.lambdas, "--mc_seeds", args.seeds,
            "--rounds", "4", "--channel_model", "rician", "--rician_k_db", "5",
            "--methods", METHODS, "--output_dir", str(output),
        ]
        if args.deadline_ms != "none":
            command += ["--delay_deadline_ms", args.deadline_ms]
        if args.end_to_end_checkpoint:
            command += ["--end_to_end_gat_checkpoint", args.end_to_end_checkpoint]
        call(command)


if __name__ == "__main__":
    main()
