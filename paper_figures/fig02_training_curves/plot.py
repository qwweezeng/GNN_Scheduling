#!/usr/bin/env python3
"""Create Fig. 2a/b independent EPS files from seed-resolved training JSON."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS,paper_style
def load(nodes,seeds):
 rows=[]
 for n in nodes:
  for s in seeds:
   p=HERE/'data/raw'/f'N{n}'/f'seed{s}'/'training.json'
   for x in json.loads(p.read_text())['per_epoch_logs']: rows.append({'num_uav':n,'seed':s,'epoch':int(x['epoch'])+1,'training_loss':float(x['train_loss']),'validation_binary_rate_mbps':float(x['val_utility'])*5})
 return rows
def table(rows):
 out=[]
 for (n,e),v in sorted(__import__('itertools').groupby(sorted(rows,key=lambda r:(r['num_uav'],r['epoch'])),lambda r:(r['num_uav'],r['epoch']))):
  z=list(v)
  for key in ('training_loss','validation_binary_rate_mbps'):
   a=np.array([r[key] for r in z]);out.append({'num_uav':n,'epoch':e,'metric':key,'mean':a.mean(),'sem':a.std(ddof=1)/np.sqrt(len(a))})
 return out
def draw(rows,nodes,key,ylabel,out):
 with paper_style():
  fig,ax=plt.subplots(figsize=(3.5,2.35))
  for n in nodes:
   z=[r for r in rows if r['num_uav']==n and r['metric']==key]; ax.plot([r['epoch'] for r in z],[r['mean'] for r in z],color=NODE_COLORS[n],label=f'N={n}')
  ax.set_xlabel('Epoch');ax.set_ylabel(ylabel);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',visible=False);ax.legend(ncol=2,frameon=False);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,format='eps');plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');a=p.parse_args();ns=[int(x)for x in a.nodes.split(',')if x];ss=[int(x)for x in a.seeds.split(',')if x];rows=table(load(ns,ss));q=HERE/'data/plotting';q.mkdir(parents=True,exist_ok=True)
 with (q/'fig02_epoch_summary.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 draw(rows,ns,'training_loss','Training loss',HERE/'figures/fig02_training_loss.eps');draw(rows,ns,'validation_binary_rate_mbps','Validation binary rate (Mbit/s)',HERE/'figures/fig02_validation_binary_rate.eps')
if __name__=='__main__':main()
