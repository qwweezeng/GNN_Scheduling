#!/usr/bin/env python3
"""Create Fig. 2 validation and training curves from seed-resolved logs.

The script writes four EPS/JPG figure pairs:
  - training loss
  - hard validation rate without endpoint projection
  - soft validation rate
  - hard validation rate after endpoint-mutual projection

The last two metrics must be present in logs produced by the updated trainer;
old training reports are rejected instead of silently reusing an incompatible
validation definition.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS, paper_style

MARKER_EVERY = 3
BANDWIDTH_MHZ = 5.0


def load(nodes, seeds, raw_root):
    rows = []
    for n in nodes:
        for s in seeds:
            report = raw_root / f'N{n}' / f'seed{s}' / 'training.json'
            for log in json.loads(report.read_text())['per_epoch_logs']:
                missing = [
                    key for key in ('val_soft_utility', 'val_projected_utility')
                    if key not in log or log[key] is None
                ]
                if missing:
                    raise ValueError(
                        f'{report} lacks {missing}; rerun Fig. 2 training with '
                        'the updated train/trainer.py before plotting.'
                    )
                rows.append({
                    'num_uav': n,
                    'seed': s,
                    'epoch': int(log['epoch']) + 1,
                    'training_loss': float(log['train_loss']),
                    'validation_binary_rate_mbps': float(log['val_utility']) * BANDWIDTH_MHZ,
                    'validation_soft_rate_mbps': float(log['val_soft_utility']) * BANDWIDTH_MHZ,
                    'validation_projected_rate_mbps': float(log['val_projected_utility']) * BANDWIDTH_MHZ,
                })
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['num_uav'], row['epoch'])].append(row)
    summary = []
    for (n, epoch), replicas in sorted(grouped.items()):
        for metric in (
            'training_loss',
            'validation_binary_rate_mbps',
            'validation_soft_rate_mbps',
            'validation_projected_rate_mbps',
        ):
            values = np.array([r[metric] for r in replicas])
            summary.append({
                'num_uav': n,
                'epoch': epoch,
                'metric': metric,
                'mean': values.mean(),
                'sem': values.std(ddof=1) / np.sqrt(len(values)),
            })
    return summary


def draw(rows, nodes, metric, ylabel, output):
    with paper_style():
        mpl.rcParams.update({
            'font.size': 7.5,
            'axes.labelsize': 7.5,
            'xtick.labelsize': 7.0,
            'ytick.labelsize': 7.0,
            'legend.fontsize': 6.5,
        })
        fig, ax = plt.subplots(figsize=(3.5, 2.35))
        for n in nodes:
            curve = [r for r in rows if r['num_uav'] == n and r['metric'] == metric]
            ax.plot(
                [r['epoch'] for r in curve],
                [r['mean'] for r in curve],
                'o-',
                color=NODE_COLORS[n],
                lw=1.0,
                ms=3,
                markevery=MARKER_EVERY,
                label=f'N={n}',
            )
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='both', linestyle='--', linewidth=0.5)
        ax.legend(ncol=2, fontsize=6.5)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format='eps')
        fig.savefig(output.with_suffix('.jpg'), format='jpg', dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default='20,30,40,50')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--raw-root', type=Path, default=HERE / 'data/raw')
    parser.add_argument('--output-root', type=Path, default=HERE)
    args = parser.parse_args()
    nodes = [int(x) for x in args.nodes.split(',') if x]
    seeds = [int(x) for x in args.seeds.split(',') if x]
    rows = summarize(load(nodes, seeds, args.raw_root))

    plotting_dir = args.output_root / 'data/plotting'
    plotting_dir.mkdir(parents=True, exist_ok=True)
    with (plotting_dir / 'fig02_epoch_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    figure_dir = args.output_root / 'figures'
    draw(rows, nodes, 'training_loss', 'Training loss (bit/s/Hz)', figure_dir / 'fig02_training_loss.eps')
    draw(rows, nodes, 'validation_binary_rate_mbps', 'Validation hard rate (Mbps)', figure_dir / 'fig02_validation_binary_rate.eps')
    draw(rows, nodes, 'validation_soft_rate_mbps', 'Soft validation rate (Mbps)', figure_dir / 'fig02_validation_soft_rate.eps')
    draw(rows, nodes, 'validation_projected_rate_mbps', 'Projected hard validation rate (Mbps)', figure_dir / 'fig02_validation_projected_rate.eps')


if __name__ == '__main__':
    main()
