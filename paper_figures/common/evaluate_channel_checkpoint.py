#!/usr/bin/env python3
"""Evaluate one channel-GAT checkpoint on the canonical binary test split."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from models.model import calculate_achievable_rate
from utils.artifacts import expected_dataset_metadata,load_validated_dataset,paper_dataset_paths
from utils.feasibility import decode_endpoint_mutual_matching,shared_uav_action_audit
from utils.project_setup import build_gat_scheduler,load_config
def main():
 p=argparse.ArgumentParser();p.add_argument('--num-uav',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();c=load_config(ROOT/'configs/paper.yaml');c.update({'NUM_UAV':a.num_uav,'TX_POWER':10.,'CHANNEL_MODEL':'rician','RICIAN_K_DB':5.,'CSI_MODE':'perfect'});seeds={'train':10086,'val':10087,'test':10088};sizes={'train':500,'val':128,'test':128};meta=expected_dataset_metadata(c,a.num_uav,seeds,sizes);_,path=paper_dataset_paths(ROOT/'datasets/paper',a.num_uav,seeds,sizes);states=load_validated_dataset(path,meta,('test',))['test_set'];ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=build_gat_scheduler(c,raw_node_dim=1);m.load_state_dict(ckpt['model_state_dict']);m.eval();rates=[]
 with torch.no_grad():
  for s in states:
   pi,_=m(s.vertex_features[:,:1],s.edge_index,s.edge_weights);act,dec=decode_endpoint_mutual_matching((pi>=.5).float().cpu(),s.links,priorities=pi.flatten().cpu(),max_rounds=1)
   if not shared_uav_action_audit(act,s.links)['feasible']:raise RuntimeError('infeasible decoded action')
   rates.append(float(calculate_achievable_rate(act.reshape(-1,1),s,binary=True,detach_hard=True).sum())*float(s.bandwidth)/1e6)
 out={'num_uav':a.num_uav,'checkpoint':a.checkpoint,'canonical_test_fingerprint':meta['fingerprint'],'num_test_samples':len(rates),'test_binary_rate_mbps':float(np.mean(rates)),'test_binary_rate_sem_mbps':float(np.std(rates,ddof=1)/np.sqrt(len(rates)))};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
