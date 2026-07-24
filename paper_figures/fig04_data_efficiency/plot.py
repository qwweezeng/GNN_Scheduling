#!/usr/bin/env python3
"""Render the independent Fig. 4 sample-efficiency EPS panel."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS,paper_style
def main():
 p=argparse.ArgumentParser();p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--sizes',default='10,20,30,40,50,60,80,100,200,300,500');p.add_argument('--seeds',default='0,1,2,3,4,5,6,7,8,9');a=p.parse_args();ns=[int(x)for x in a.nodes.split(',')if x];ks=[int(x)for x in a.sizes.split(',')if x];ss=[int(x)for x in a.seeds.split(',')if x];rows=[]
 for n in ns:
  for k in ks:
   v=[json.loads((HERE/'data/raw'/f'N{n}'/f'n{k}'/f'seed{s}'/'evaluation.json').read_text())['test_binary_rate_mbps']for s in ss];rows.append({'num_uav':n,'training_samples':k,'test_binary_rate_mbps':np.mean(v),'seed_sem_mbps':np.std(v,ddof=1)/np.sqrt(len(v))})
 q=HERE/'data/plotting';q.mkdir(parents=True,exist_ok=True)
 with(q/'fig04_sample_efficiency.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with paper_style():
  fig,ax=plt.subplots(figsize=(3.5,2.35))
  for n in ns:
   z=[r for r in rows if r['num_uav']==n];ax.plot([r['training_samples']for r in z],[r['test_binary_rate_mbps']for r in z],marker='o',ms=3,color=NODE_COLORS[n],label=f'N={n}')
  ax.set(xlabel='Number of training samples',ylabel='Test binary rate (Mbit/s)');ax.set_xscale('symlog',linthresh=10);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',visible=False);ax.legend(ncol=2,frameon=False);(HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/fig04_data_efficiency.eps',format='eps');plt.close(fig)
if __name__=='__main__':main()
