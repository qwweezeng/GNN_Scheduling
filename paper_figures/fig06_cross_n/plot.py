#!/usr/bin/env python3
"""Render Fig. 6 as train-scale curves, not a heat map."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS,paper_style
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',default='data/raw/cross_n_trainSize=500.json');p.add_argument('--output',default='fig06_cross_network_generalization');a=p.parse_args();d=json.loads((HERE/a.input).read_text());ns=d['nodes'];rows=[]
 train_scopes=[str(n) for n in ns]+(['mixed'] if 'mixed' in d['results'] else [])
 for tr in train_scopes:
  for te in ns: rows.append({'training_num_uav':tr,'test_num_uav':te,'average_sum_rate_mbps':d['results'][str(tr)][str(te)]['mean_aggregate_phy_rate_mbps']})
 q=HERE/'data/plotting';q.mkdir(parents=True,exist_ok=True)
 with(q/('fig06_cross_n_'+Path(a.input).stem+'.csv')).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 with paper_style():
  mpl.rcParams.update({"font.size":7.5,"axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":6.5})
  fig,ax=plt.subplots(figsize=(3.5,2.35))
  for tr in train_scopes:
   z=[r for r in rows if r['training_num_uav']==tr]
   if tr == 'mixed':
    ax.plot([r['test_num_uav']for r in z],[r['average_sum_rate_mbps']for r in z],marker='D',ms=3,label='Train mixed N',color='#333333',linestyle='--')
   else:
    ax.plot([r['test_num_uav']for r in z],[r['average_sum_rate_mbps']for r in z],marker='o',ms=3,label=f'Train N={tr}',color=NODE_COLORS[int(tr)])
  ax.set(xlabel='Test number of UAVs',ylabel='Average sum rate (Mbit/s)');ax.set_xticks(ns);ax.spines[['top','right']].set_visible(False);ax.grid(axis='both',linestyle='--',linewidth=0.5);ax.legend(ncol=2,fontsize=6);(HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/fig06_cross_network_generalization.eps',format='eps');fig.savefig(HERE/'figures/fig06_cross_network_generalization.jpg',format='jpg',dpi=300,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
