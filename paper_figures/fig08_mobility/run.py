#!/usr/bin/env python3
"""Generate Fig. 8 mobile full-buffer GAT rate and GAT-MP queue records."""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from models.model import calculate_achievable_rate
from simulator.mobile_block_fading import build_observed_network_state,build_true_network_state,generate_mobile_block_fading_trace
from utils.feasibility import decode_endpoint_mutual_matching
from utils.project_setup import build_network_generator,load_channel_gat,load_config
def cfg(n):
 c=load_config(ROOT/'configs/paper.yaml');c.update({'NUM_UAV':n,'TX_POWER':10.,'CHANNEL_MODEL':'rician','RICIAN_K_DB':5.,'CSI_MODE':'perfect'});return c
def static(n,speed,block,seeds,slots):
 c=cfg(n);g=build_network_generator(c);m,_=load_channel_gat(c,n,device='cpu');rows=[]
 with torch.no_grad():
  for seed in seeds:
   tr=generate_mobile_block_fading_trace(g,slots,block,float(c['SLOT_DURATION']),speed,seed,quantization_bits=(4,),rician_k_db=5.)
   for mode in ('latest','stale'):
    v=[]
    for t in range(slots):
     b=min(t//block,len(tr.blocks)-1);o,_=build_observed_network_state(g,tr,b,mode,stale_blocks=1);true,_=build_true_network_state(g,tr,b);pi,_=m(o.vertex_features,o.edge_index,o.edge_weights);a,_=decode_endpoint_mutual_matching((pi>=.5).float().cpu(),true.links,priorities=pi.flatten().cpu(),max_rounds=1);v.append(float(calculate_achievable_rate(a.reshape(-1,1),true,binary=True,detach_hard=True).sum())*5)
    rows.append({'num_uav':n,'speed_mps':speed,'slots_per_block':block,'seed':seed,'method':mode,'average_sum_rate_mbps':float(np.mean(v))})
 return rows
def call(x):print('+ '+' '.join(x),flush=True);subprocess.run(x,cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--slots',type=int,default=5000);p.add_argument('--seeds',default='42,43,44,45,46');a=p.parse_args();ns=[int(x)for x in a.nodes.split(',')if x];seeds=[int(x)for x in a.seeds.split(',')if x];raw=HERE/'data/raw';raw.mkdir(parents=True,exist_ok=True);rate=[]
 for n in ns:
  for speed in (0,5,10,20): rate+=static(n,speed,5,seeds,a.slots)
  for block in (1,5,10,20): rate+=static(n,10,block,seeds,a.slots)
  for name,speeds,blocks in [('speed','0,5,10,20','5'),('block','10','1,5,10,20')]:call([sys.executable,'-u','experiments/evaluate_mobile_block_fading_csi.py','--num-uav',str(n),'--num-slots',str(a.slots),'--seeds',a.seeds,'--speeds-mps',speeds,'--slots-per-block',blocks,'--stale-blocks','1','--arrival-rates','2.4','--csi-methods','latest,stale','--scheduler-variants','gat_assisted','--rounds','4','--output',str(raw/f'gatmp_{name}_N{n}.json')])
 (raw/'chapter4_rate.json').write_text(json.dumps(rate,indent=2))
if __name__=='__main__':main()
