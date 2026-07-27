#!/usr/bin/env python3
"""Run Fig. 7 CSI feedback tests for Chapter IV and GAT-MP (Chapter V)."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def call(x):print('+ '+' '.join(x),flush=True);subprocess.run(x,cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--slots',type=int,default=10000);p.add_argument('--seeds',default='42,43,44,45,46');p.add_argument('--arrival-rate',default='2.4');p.add_argument('--device',default='auto',choices=('auto','cpu','cuda'));a=p.parse_args();modes='latest,large_scale,quantized_2bit,quantized_4bit,quantized_5bit,quantized_8bit'
 for n in [int(x)for x in a.nodes.split(',')if x]:
  call([sys.executable,'-u','experiments/evaluate_channel_gat_csi.py','--num-uav',str(n),'--device',a.device,'--output',str(HERE/'data/raw'/f'chapter4_N{n}.json')])
  call([sys.executable,'-u','experiments/evaluate_mobile_block_fading_csi.py','--num-uav',str(n),'--num-slots',str(a.slots),'--seeds',a.seeds,'--speeds-mps','0','--slots-per-block','1','--stale-blocks','1','--arrival-rates',a.arrival_rate,'--csi-methods',modes,'--scheduler-variants','gat_assisted','--rounds','4','--device',a.device,'--output',str(HERE/'data/raw'/f'chapter5_N{n}.json')])
if __name__=='__main__':main()
