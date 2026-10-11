#!/usr/bin/env python3
"""Render the two independent Fig. 3 activation-audit EPS panels."""
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
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS, paper_style

MARKER_EVERY = 5
PANEL_SIZE = (2.0, 2.35)
LINE_WIDTH = 1.0
MARKER_SIZE = 2.5


def compact_legend(ax, nodes, include_line_style):
    handles = [Line2D([], [], color=NODE_COLORS[n], lw=LINE_WIDTH, label=f'N={n}') for n in nodes]
    if include_line_style:
        handles.extend([
            Line2D([], [], color='black', lw=LINE_WIDTH, label='mean'),
            Line2D([], [], color='black', lw=LINE_WIDTH, ls='--', label='std'),
        ])
    ax.legend(
        handles=handles,
        ncol=2,
        loc='upper right',
        frameon=True,
        fontsize=5.8,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.65,
        borderpad=0.35,
        labelspacing=0.35,
    )


def save(fig, figure_dir, stem):
    fig.savefig(figure_dir / f'{stem}.eps', format='eps')
    fig.savefig(figure_dir / f'{stem}.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source', type=Path,
        default=HERE.parent / 'fig02_training_curves/data/raw',
    )
    parser.add_argument('--nodes', default='20,30,40,50')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--output-root', type=Path, default=HERE)
    args = parser.parse_args()
    source = args.source.resolve()
    nodes = [int(x) for x in args.nodes.split(',') if x]
    seeds = [int(x) for x in args.seeds.split(',') if x]

    grouped = defaultdict(list)
    for n in nodes:
        for seed in seeds:
            report = source / f'N{n}' / f'seed{seed}' / 'training.json'
            for log in json.loads(report.read_text())['per_epoch_logs']:
                grouped[(n, int(log['epoch']) + 1)].append(log)

    rows = []
    for (n, epoch), replicas in sorted(grouped.items()):
        rows.append({
            'num_uav': n,
            'epoch': epoch,
            'pi_mean': np.mean([r['pi_mean'] for r in replicas]),
            'pi_std': np.mean([r['pi_std'] for r in replicas]),
            'binary_active_fraction': np.mean([r['binary_active_fraction'] for r in replicas]),
        })

    plotting_dir = args.output_root / 'data/plotting'
    plotting_dir.mkdir(parents=True, exist_ok=True)
    with (plotting_dir / 'fig03_epoch_audit.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    figure_dir = args.output_root / 'figures'
    figure_dir.mkdir(parents=True, exist_ok=True)

    with paper_style():
        mpl.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5})
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        for n in nodes:
            curve = [r for r in rows if r['num_uav'] == n]
            epochs = [r['epoch'] for r in curve]
            ax.plot(epochs, [r['pi_mean'] for r in curve], color=NODE_COLORS[n], lw=LINE_WIDTH, marker='o', ms=MARKER_SIZE, markevery=MARKER_EVERY)
            ax.plot(epochs, [r['pi_std'] for r in curve], color=NODE_COLORS[n], lw=LINE_WIDTH, ls='--', marker='o', ms=MARKER_SIZE, markevery=MARKER_EVERY)
        ymax = max([r['pi_mean'] for r in rows] + [r['pi_std'] for r in rows])
        ax.set_ylim(0, max(0.20, ymax * 1.15))
        ax.set(xlabel='Epoch', ylabel=r'Continuous $\pi$')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='both', linestyle='--', linewidth=0.5)
        compact_legend(ax, nodes, include_line_style=True)
        save(fig, figure_dir, 'fig03_continuous_pi_statistics')

    with paper_style():
        mpl.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5})
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        for n in nodes:
            curve = [r for r in rows if r['num_uav'] == n]
            ax.plot(
                [r['epoch'] for r in curve],
                [r['binary_active_fraction'] for r in curve],
                'o-', color=NODE_COLORS[n], lw=LINE_WIDTH, ms=MARKER_SIZE,
                markevery=MARKER_EVERY,
            )
        ymax = max(r['binary_active_fraction'] for r in rows)
        ax.set_ylim(0, max(0.10, ymax * 1.2))
        ax.set(xlabel='Epoch', ylabel='Active-link fraction')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='both', linestyle='--', linewidth=0.5)
        compact_legend(ax, nodes, include_line_style=False)
        save(fig, figure_dir, 'fig03_binary_active_fraction')


if __name__ == '__main__':
    main()
