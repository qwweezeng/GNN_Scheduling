#!/usr/bin/env python3
"""Render the two independent Fig. 3 activation-audit EPS panels."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS,paper_style
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',default='../fig02_training_curves/data/raw');p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');a=p.parse_args();src=(HERE/a.source).resolve();ns=[int(x)for x in a.nodes.split(',')if x];ss=[int(x)for x in a.seeds.split(',')if x];g=defaultdict(list)
 for n in ns:
  for s in ss:
   for r in json.loads((src/f'N{n}'/f'seed{s}'/'training.json').read_text())['per_epoch_logs']:g[(n,int(r['epoch'])+1)].append(r)
 rows=[]
 for (n,e),v in sorted(g.items()):
  rows.append({'num_uav':n,'epoch':e,'pi_mean':np.mean([x['pi_mean']for x in v]),'pi_std':np.mean([x['pi_std']for x in v]),'binary_active_fraction':np.mean([x['binary_active_fraction']for x in v])})
 q=HERE/'data/plotting';q.mkdir(parents=True,exist_ok=True)
 with(q/'fig03_epoch_audit.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with paper_style():
  fig,ax=plt.subplots(figsize=(3.5,2.35))
  for n in ns:
   z=[r for r in rows if r['num_uav']==n];ax.plot([r['epoch']for r in z],[r['pi_mean']for r in z],color=NODE_COLORS[n],label=f'N={n}, mean');ax.plot([r['epoch']for r in z],[r['pi_std']for r in z],color=NODE_COLORS[n],ls='--',label=f'N={n}, std.')
  ax.set(xlabel='Epoch',ylabel='Continuous activation statistic');ax.set_ylim(0,1);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',visible=False);ax.legend(ncol=2,frameon=False,fontsize=5.7);(HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/fig03_continuous_pi_statistics.eps',format='eps');plt.close(fig)
  fig,ax=plt.subplots(figsize=(3.5,2.35))
  for n in ns:
   z=[r for r in rows if r['num_uav']==n];ax.plot([r['epoch']for r in z],[r['binary_active_fraction']for r in z],color=NODE_COLORS[n],label=f'N={n}')
  ax.set(xlabel='Epoch',ylabel='Fraction of active links');ax.set_ylim(0,1);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',visible=False);ax.legend(ncol=2,frameon=False);fig.savefig(HERE/'figures/fig03_binary_active_fraction.eps',format='eps');plt.close(fig)
if __name__=='__main__':main()
