#!/usr/bin/env python3
"""Render independent Chapter IV/V CSI-feedback EPS panels."""
from __future__ import annotations
import csv,json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS,paper_style
BITS=(2,4,5,8)
def main():
 ns=(20,30,40,50);r4=[];r5=[]
 for n in ns:
  d=json.loads((HERE/'data/raw'/f'chapter4_N{n}.json').read_text());
  for m,x in d['results'].items(): r4.append({'num_uav':n,'mode':m,'value':x['Proposed-GAT']['mean_aggregate_phy_rate_mbps']})
  d=json.loads((HERE/'data/raw'/f'chapter5_N{n}.json').read_text());g=defaultdict(list)
  for x in d['records']:g[x['method']].append(x['average_total_queue_backlog_mbit'])
  for m,v in g.items():r5.append({'num_uav':n,'mode':m,'value':float(np.mean(v))})
 q=HERE/'data/plotting';q.mkdir(parents=True,exist_ok=True)
 for name,rows in [('chapter4_rate',r4),('chapter5_backlog',r5)]:
  with(q/f'fig07_{name}.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 def draw(rows,ylabel,stem):
  with paper_style():
   fig,ax=plt.subplots(figsize=(3.5,2.35))
   for n in ns:
    z={r['mode']:r['value']for r in rows if r['num_uav']==n};x=np.arange(len(BITS));ax.plot(x,[z[f'quantized_{b}bit']for b in BITS],marker='o',ms=3,color=NODE_COLORS[n],label=f'N={n}');ax.hlines(z['perfect'] if 'perfect'in z else z['latest'],-.15,3.15,color=NODE_COLORS[n],ls=':',lw=.8);ax.hlines(z['large_scale'],-.15,3.15,color=NODE_COLORS[n],ls='--',lw=.8)
   ax.set(xticks=range(4),xticklabels=[f'{b}-bit'for b in BITS],xlabel='CSI feedback precision',ylabel=ylabel);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',visible=False);ax.legend(ncol=2,frameon=False);(HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures'/stem,format='eps');plt.close(fig)
 draw(r4,'Test sum rate (Mbit/s)','fig07_chapter4_csi_rate.eps');draw(r5,'Average queue backlog (Mbit)','fig07_chapter5_gat_mp_backlog.eps')
if __name__=='__main__':main()
