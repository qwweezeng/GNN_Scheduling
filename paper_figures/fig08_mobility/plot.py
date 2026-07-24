#!/usr/bin/env python3
"""Render the four independent Fig. 8 EPS panels."""
from __future__ import annotations
import csv,json,sys
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from paper_figures.common.plot_style import NODE_COLORS,paper_style
def main():
 r4=json.loads((HERE/'data/raw/chapter4_rate.json').read_text());r5=[]
 for n in (20,30,40,50):
  for name in ('speed','block'):
   for x in json.loads((HERE/'data/raw'/f'gatmp_{name}_N{n}.json').read_text())['records']:r5.append(x)
 q=HERE/'data/plotting';q.mkdir(parents=True,exist_ok=True)
 for name,rows in [('chapter4_rate',r4),('chapter5_backlog',r5)]:
  with(q/f'fig08_{name}.csv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 def panel(rows,xkey,ykey,xlabel,ylabel,stem):
  with paper_style():
   fig,ax=plt.subplots(figsize=(3.5,2.35))
   for n in (20,30,40,50):
    for mode,ls in [('latest','-'),('stale','--')]:
     v=[r for r in rows if r['num_uav']==n and r['method']==mode];g=defaultdict(list)
     for r in v:g[r[xkey]].append(r[ykey])
     x=sorted(g);ax.plot(x,[np.mean(g[z])for z in x],ls=ls,color=NODE_COLORS[n],marker='o',ms=2.7,label=f'N={n}, {"perfect" if mode=="latest" else "1-block stale"}')
   ax.set(xlabel=xlabel,ylabel=ylabel);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',visible=False);ax.legend(ncol=2,fontsize=5.5,frameon=False);(HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures'/stem,format='eps');plt.close(fig)
 panel([r for r in r4 if r['slots_per_block']==5],'speed_mps','average_sum_rate_mbps','UAV speed (m/s)','Average sum rate (Mbit/s)','fig08a_chapter4_speed.eps');panel([r for r in r4 if r['speed_mps']==10],'slots_per_block','average_sum_rate_mbps','Block length (slots)','Average sum rate (Mbit/s)','fig08b_chapter4_block_length.eps');panel([r for r in r5 if r['slots_per_block']==5],'speed_mps','average_total_queue_backlog_mbit','UAV speed (m/s)','Average queue backlog (Mbit)','fig08c_chapter5_speed.eps');panel([r for r in r5 if r['speed_mps']==10],'slots_per_block','average_total_queue_backlog_mbit','Block length (slots)','Average queue backlog (Mbit)','fig08d_chapter5_block_length.eps')
if __name__=='__main__':main()
