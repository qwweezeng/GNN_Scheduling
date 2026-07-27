#!/usr/bin/env python3
"""Run the 50-epoch Proposed-GAT paper-training workload across two GPUs.

The scheduler keeps each (N, train_size, seed) experiment atomic: one worker
per GPU performs its training and all checkpoint evaluations before taking the
next task.  Existing complete artifacts are reused by the invoked runners.

Execution phases:
  fig02  Train full_300 replicas for Fig. 2/3 and create their checkpoint pair.
  fig04  Train Fig. 4 prefix subsets n=10,...,200 and reference Fig. 2 for n=300.
  all    Run fig02 then fig04.  Fig. 4 cannot start until Fig. 2 succeeds.

Fig. 1 intentionally remains a manual two-step workflow (run.py --stage train,
then run.py --stage evaluate) and is not scheduled here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NODES = (20, 30, 40, 50)
SEEDS = (0, 1, 2)
FIG04_PREFIX_SIZES = (10, 20, 30, 40, 60, 80, 100, 200)
FIG04_FULL_SIZE = 300


@dataclass(frozen=True)
class Task:
    figure: str
    num_uav: int
    model_seed: int
    train_size: int | None = None

    @property
    def label(self) -> str:
        size = "full_300" if self.train_size is None else f"n{self.train_size}"
        return f"{self.figure}:N{self.num_uav}:{size}:seed{self.model_seed}"


def run_task(task: Task, gpu_id: int, force: bool) -> None:
    if task.figure == "fig02":
        command = [
            sys.executable,
            "-u",
            "paper_figures/fig02_training_curves/run.py",
            "--nodes",
            str(task.num_uav),
            "--seeds",
            str(task.model_seed),
            "--epochs",
            "50",
            "--device",
            "cuda",
            "--gpu-id",
            str(gpu_id),
        ]
    elif task.figure == "fig04":
        if task.train_size is None:
            raise ValueError("fig04 task requires train_size")
        command = [
            sys.executable,
            "-u",
            "paper_figures/fig04_data_efficiency/run.py",
            "--nodes",
            str(task.num_uav),
            "--sizes",
            str(task.train_size),
            "--seeds",
            str(task.model_seed),
            "--epochs",
            "50",
            "--device",
            "cuda",
            "--gpu-id",
            str(gpu_id),
        ]
    else:
        raise ValueError(f"unsupported figure phase: {task.figure}")

    if force:
        command.append("--force")
    print(f"[GPU {gpu_id}] START {task.label}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[GPU {gpu_id}] DONE  {task.label}", flush=True)


def run_phase(tasks: list[Task], gpu_ids: list[int], force: bool) -> None:
    assignments = {gpu_id: [] for gpu_id in gpu_ids}
    for index, task in enumerate(tasks):
        assignments[gpu_ids[index % len(gpu_ids)]].append(task)

    def worker(gpu_id: int) -> None:
        for task in assignments[gpu_id]:
            run_task(task, gpu_id, force)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, gpu_id) for gpu_id in gpu_ids]
        for future in futures:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fig02", "fig04", "all"), default="all")
    parser.add_argument("--gpus", default="0,1", help="Physical CUDA GPU IDs, e.g. 0,1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    gpu_ids = [int(value) for value in args.gpus.split(",") if value]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("--gpus must contain exactly two distinct GPU IDs")

    fig02_tasks = [Task("fig02", n, seed) for n in NODES for seed in SEEDS]
    fig04_prefix_tasks = [
        Task("fig04", n, seed, train_size)
        for n in NODES
        for train_size in FIG04_PREFIX_SIZES
        for seed in SEEDS
    ]
    fig04_full_reference_tasks = [
        Task("fig04", n, seed, FIG04_FULL_SIZE)
        for n in NODES
        for seed in SEEDS
    ]

    if args.phase in {"fig02", "all"}:
        run_phase(fig02_tasks, gpu_ids, args.force)
    if args.phase in {"fig04", "all"}:
        if args.phase == "fig04":
            print(
                "[scheduler] Fig.4 assumes the centralized full_300 checkpoints "
                "from Fig.2 already exist.",
                flush=True,
            )
        run_phase(fig04_prefix_tasks, gpu_ids, args.force)
        # n=300 is checkpoint reuse plus its Fig.4-local evaluation records.
        run_phase(fig04_full_reference_tasks, gpu_ids, args.force)


if __name__ == "__main__":
    main()
