#!/usr/bin/env python3
"""Run the frozen-GAT train-N/test-N grid used only by Fig. 6."""
from __future__ import annotations
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
import argparse

parser=argparse.ArgumentParser(); parser.add_argument('--device',default='auto',choices=('auto','cpu','cuda')); args=parser.parse_args()
subprocess.run([sys.executable,'-u','experiments/evaluate_channel_gat_cross_n.py','--nodes','20,30,40,50','--device',args.device,'--output',str(HERE/'data/raw/cross_n.json')],cwd=ROOT,check=True)
