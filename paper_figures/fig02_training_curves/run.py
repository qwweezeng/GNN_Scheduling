#!/usr/bin/env python3
"""Train the 4 x 10 frozen-architecture replicas needed by Fig. 2/3.

Outputs are seed-resolved reports in this figure folder.  The fixed model,
physics, 500/128/128 split sizes, and epoch count are passed explicitly.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
def call(x): print('+ ' + ' '.join(x), flush=True); subprocess.run(x, cwd=ROOT, check=True)
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');p.add_argument('--epochs',type=int,default=30);p.add_argument('--force',action='store_true');a=p.parse_args()
 for n in [int(x) for x in a.nodes.split(',') if x]:
  for seed in [int(x) for x in a.seeds.split(',') if x]:
   report=HERE/'data/raw'/f'N{n}'/f'seed{seed}'/'training.json'; ckpt=report.with_name('model.pth')
   if report.exists() and not a.force: print(f'reuse {report}'); continue
   report.parent.mkdir(parents=True,exist_ok=True)
   call([sys.executable,'-u','train/train_channel_gat_rician.py','--num_uav',str(n),'--train_size','500','--val_size','128','--test_size','128','--epochs',str(a.epochs),'--seed','10086','--val_seed','10087','--test_seed','10088','--model-seed',str(seed),'--checkpoint_out',str(ckpt),'--report_out',str(report)])
if __name__=='__main__': main()
