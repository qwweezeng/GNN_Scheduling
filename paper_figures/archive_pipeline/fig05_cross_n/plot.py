#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent; raw=json.loads((HERE/'data/raw/cross_n.json').read_text()); ns=raw['nodes']; rel=raw['relative_to_dedicated_test_model']; out=HERE/'data/plotting';out.mkdir(parents=True,exist_ok=True)
rows=[{'train_num_uav':tr,'test_num_uav':te,'relative_rate':rel[str(tr)][str(te)],'active_fraction':raw['results'][str(tr)][str(te)]['mean_active_fraction']} for tr in ns for te in ns]
with (out/'heatmap.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
with mpl.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42}):
 fig,axs=plt.subplots(1,2,figsize=(7.16,2.85)); data=[np.array([[rel[str(i)][str(j)] for j in ns] for i in ns]),np.array([[raw['results'][str(i)][str(j)]['mean_active_fraction'] for j in ns] for i in ns])]; labs=['Relative aggregate PHY rate','Active-link fraction']
 for k,(ax,z,lab) in enumerate(zip(axs,data,labs)):
  im=ax.imshow(z,cmap='Blues' if k==0 else 'Purples',aspect='auto');ax.set_xticks(range(4),ns);ax.set_yticks(range(4),ns);ax.set_xlabel('Test number of UAVs');ax.set_ylabel('Training number of UAVs');ax.set_title(f'({chr(97+k)}) {lab}',loc='left',fontweight='bold')
  for i in range(4):
   for j in range(4):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',fontsize=7)
  fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
 fig.tight_layout(); d=HERE/'figures';d.mkdir(parents=True,exist_ok=True);stem=d/'fig05_cross_n_double_column';fig.savefig(stem.with_suffix('.eps'));fig.savefig(stem.with_suffix('.pdf'));fig.savefig(stem.with_suffix('.svg'));fig.savefig(stem.with_suffix('.tiff'),dpi=600);fig.savefig(d/'fig05_cross_n_double_column_review.png',dpi=300)
