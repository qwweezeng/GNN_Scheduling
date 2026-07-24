#!/usr/bin/env python3
import csv,json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; ns=(20,30,40,50); modes=('perfect','large_scale','quantized_2bit','quantized_3bit','quantized_4bit','quantized_6bit','quantized_8bit'); labels=('Perfect','Large-scale','2-bit','3-bit','4-bit','6-bit','8-bit'); rows=[]
for n in ns:
 d=json.loads((HERE/f'data/raw/csi_N{n}.json').read_text())
 for mo in modes:
  for me,x in d['results'][mo].items(): rows.append({'num_uav':n,'csi_mode':mo,'method':me,'aggregate_phy_rate_mbps':x['mean_aggregate_phy_rate_mbps'],'sem_mbps':x['sem_aggregate_phy_rate_mbps'],'active_fraction':x['mean_active_fraction']})
out=HERE/'data/plotting';out.mkdir(parents=True,exist_ok=True)
with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
with mpl.rc_context({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'svg.fonttype':'none','pdf.fonttype':42}):
 fig,axs=plt.subplots(2,2,figsize=(7.16,4.55)); co={'Channel-GAT':'#C65A78','Local Greedy':'#7884B4'}
 for ax,n,le in zip(axs.ravel(),ns,'abcd'):
  for me in co:
   z=[r for r in rows if r['num_uav']==n and r['method']==me];ax.plot(range(7),[r['aggregate_phy_rate_mbps'] for r in z],marker='o',ms=3,lw=1.3,color=co[me],label=me)
  ax.set_xticks(range(7),labels,rotation=32,ha='right');ax.set_ylabel('Aggregate PHY rate (Mbit/s)');ax.set_title(f'({le}) N={n}',loc='left',fontweight='bold');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#E1E1E1',lw=.45)
 h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False);fig.tight_layout(rect=(0,0,1,.92));d=HERE/'figures';d.mkdir(parents=True,exist_ok=True);s=d/'fig06_channel_csi_double_column';fig.savefig(s.with_suffix('.eps'));fig.savefig(s.with_suffix('.pdf'));fig.savefig(s.with_suffix('.svg'));fig.savefig(s.with_suffix('.tiff'),dpi=600);fig.savefig(d/'fig06_channel_csi_double_column_review.png',dpi=300)
