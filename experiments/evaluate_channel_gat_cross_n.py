#!/usr/bin/env python3
"""Cross-N Proposed-GAT evaluation with the formal endpoint-matching decoder."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from models.model import calculate_achievable_rate
from utils.artifacts import expected_dataset_metadata, load_validated_dataset, paper_dataset_paths
from utils.feasibility import decode_endpoint_mutual_matching, shared_uav_action_audit
from utils.project_setup import load_channel_gat, load_config

def cfg(n):
    c=load_config(ROOT/'configs/paper.yaml'); c.update({'NUM_UAV':n,'TX_POWER':10.0,'CHANNEL_MODEL':'rician','RICIAN_K_DB':5.0,'CSI_MODE':'perfect'}); return c
def states(n):
    c=cfg(n); seeds={'train':10086,'val':10087,'test':10088}; sizes={'train':500,'val':128,'test':128}
    meta=expected_dataset_metadata(c,n,seeds,sizes); _,p=paper_dataset_paths(ROOT/'datasets/paper',n,seeds,sizes)
    return load_validated_dataset(p,meta,('test',))['test_set'], meta['fingerprint'], str(p)
def evaluate(model, test):
    rates=[]; active=[]; audit=[]
    with torch.no_grad():
        for i,s in enumerate(test):
            pi,_=model(s.vertex_features[:,:1],s.edge_index,s.edge_weights); candidate=(pi>=.5).float().cpu()
            act,dec=decode_endpoint_mutual_matching(candidate,s.links,priorities=pi.reshape(-1).cpu(),max_rounds=1)
            final=shared_uav_action_audit(act,s.links)
            if not final['feasible']: raise RuntimeError('infeasible decoded action')
            r=float(calculate_achievable_rate(act.reshape(-1,1),s,binary=True,detach_hard=True).sum())
            rates.append(r); active.append(float(act.float().mean())); audit.append({'sample_index':i,'active_project_link_indices':torch.nonzero(act.reshape(-1)>0.5,as_tuple=True)[0].tolist(),'hard_feasibility':dec})
    a=np.asarray(rates); return {'mean_sum_spectral_efficiency_bit_s_hz':float(a.mean()),'sem_sum_spectral_efficiency_bit_s_hz':float(a.std(ddof=1)/np.sqrt(len(a))),'mean_aggregate_phy_rate_mbps':float(a.mean()*5),'mean_active_fraction':float(np.mean(active)),'per_sample_actions':audit}
def main():
 p=argparse.ArgumentParser();p.add_argument('--nodes',default='20,30,40,50');p.add_argument('--output',required=True);a=p.parse_args(); ns=[int(x) for x in a.nodes.split(',')]
 tests={n:states(n) for n in ns}; matrix={}; checkpoints={}
 for tr in ns:
  m,ck=load_channel_gat(cfg(tr),tr,device='cpu'); checkpoints[str(tr)]=ck; matrix[str(tr)]={}
  for te in ns:
   r=evaluate(m,tests[te][0]); matrix[str(tr)][str(te)]=r; print(f'train N={tr} -> test N={te}: {r["mean_aggregate_phy_rate_mbps"]:.3f} Mbit/s',flush=True)
 dedicated={str(n):matrix[str(n)][str(n)]['mean_aggregate_phy_rate_mbps'] for n in ns}
 relative={tr:{te:v['mean_aggregate_phy_rate_mbps']/dedicated[te] for te,v in row.items()} for tr,row in matrix.items()}
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps({'schema':'tnse_fig06_cross_n_v1','nodes':ns,'physics':{'tx_power_dbm':10,'channel_model':'rician','rician_k_db':5,'service_channel':'current true instantaneous Rician channel','decoder':'one_round_endpoint_mutual_matching'},'checkpoints':checkpoints,'test_sets':{str(n):{'path':tests[n][2],'fingerprint':tests[n][1],'samples':len(tests[n][0])} for n in ns},'results':matrix,'relative_to_dedicated_test_model':relative},indent=2))
if __name__=='__main__': main()
