#!/usr/bin/env python3
"""Run the frozen-GAT train-N/test-N grid used only by Fig. 6."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
parser.add_argument('--train_size',type=int,default=300,
    help='Training set size (default: 300; reuses the 50-epoch seed-0 checkpoints)')
args=parser.parse_args()
subprocess.run([
    sys.executable,'-u','experiments/evaluate_channel_gat_cross_n.py',
    '--nodes','20,30,40,50',
    '--train_size',str(args.train_size),
    '--device',args.device,
    '--output',str(HERE/(
        'data/raw/cross_n_trainSize=300_seed0_epoch50.json'
        if args.train_size == 300
        else 'data/raw/cross_n_trainSize='+str(args.train_size)+'.json'
    )),
], cwd=ROOT, check=True)
