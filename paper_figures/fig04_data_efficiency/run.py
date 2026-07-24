#!/usr/bin/env python3
"""Train/evaluate the exact 4 x 11 x 10 Fig. 4 data-efficiency grid."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def call(x):print('+ '+' '.join(x),flush=True);subprocess.run(x,cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--sizes',default='10,20,30,40,50,60,80,100,200,300,500');p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');p.add_argument('--epochs',type=int,default=30);p.add_argument('--force',action='store_true');a=p.parse_args()
 for n in [int(x)for x in a.nodes.split(',')if x]:
  for k in [int(x)for x in a.sizes.split(',')if x]:
   for s in [int(x)for x in a.seeds.split(',')if x]:
    d=HERE/'data/raw'/f'N{n}'/f'n{k}'/f'seed{s}';r=d/'training.json';c=d/'model.pth';e=d/'evaluation.json'
    if not r.exists() or a.force:
     d.mkdir(parents=True,exist_ok=True);call([sys.executable,'-u','train/train_channel_gat_rician.py','--num_uav',str(n),'--train_size',str(k),'--prefix_pool_size','500','--val_size','128','--test_size','128','--epochs',str(a.epochs),'--seed','10086','--val_seed','10087','--test_seed','10088','--model-seed',str(s),'--checkpoint_out',str(c),'--report_out',str(r)])
    if not e.exists() or a.force:call([sys.executable,'-u','paper_figures/common/evaluate_channel_checkpoint.py','--num-uav',str(n),'--checkpoint',str(c),'--output',str(e)])
if __name__=='__main__':main()
