#!/usr/bin/env python3
"""Evaluate observed-CSI variants; settle every action on true Rician CSI."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.model import calculate_achievable_rate
from simulator.mobile_block_fading import build_conflict_graph_from_power
from utils.artifacts import expected_dataset_metadata, load_validated_dataset, paper_dataset_paths
from utils.baseline import local_greedy
from utils.feasibility import decode_endpoint_mutual_matching, project_shared_uav_feasible, shared_uav_action_audit
from utils.network_generate import NetworkState, dequantize_distance_from_binindex
from utils.project_setup import build_network_generator, load_channel_gat, load_config

def cfg(n):
    c=load_config(ROOT/'configs/paper.yaml'); c.update({'NUM_UAV':n,'TX_POWER':10.,'CHANNEL_MODEL':'rician','RICIAN_K_DB':5.,'CSI_MODE':'perfect'}); return c
def load_test(c,n):
    sd={'train':10086,'val':10087,'test':10088}; sz={'train':300,'val':128,'test':128}; m=expected_dataset_metadata(c,n,sd,sz); _,p=paper_dataset_paths(ROOT/'datasets/paper',n,sd,sz)
    return load_validated_dataset(p,m,('test',))['test_set'],m['fingerprint'],str(p)
def obs(s,p,label,thr):
    l=s.links; snr=10*torch.log10(p[l[:,0],l[:,1]]/s.noise_power+1e-12); ei,ew=build_conflict_graph_from_power(l,p,s.noise_power,10**(thr/10)); ne={i:[] for i in range(len(l))}
    for a,b in ei.t().tolist(): ne[b].append(a)
    return NetworkState(l,snr[:,None].float(),ei,ew,s.large_scale,p,p,p,s.noise_power,s.bandwidth,s.snr_gap,ne,s.positions,s.tx_power,s.carrier_f,label,None,False)
def qpower(s,g,b):
    d=torch.cdist(s.positions,s.positions); lo=float(g.min_distance); hi=float((g.area_width**2*2+(g.height_max-g.height_min)**2)**.5); ix=torch.floor(((d-lo)/(hi-lo)).clamp(0,1)*(2**b-1)); r=dequantize_distance_from_binindex(ix,q=b,d_min=lo,d_max=hi); p=g.tx_power*g.lambda0*r.pow(-g.xi); p.fill_diagonal_(0); return p.float()
def direct_rate_priority(pi, state):
    tx, rx = state.links[:, 0], state.links[:, 1]
    snr = state.signal_power[tx, rx].cpu() / state.noise_power
    return pi.reshape(-1).float().cpu() * torch.log2(1.0 + snr / state.snr_gap).float()
def settle(a,p,s,mutual):
    a,d=(decode_endpoint_mutual_matching(a,s.links,priorities=p,max_rounds=1) if mutual else project_shared_uav_feasible(a,s.links,priorities=p)); au=shared_uav_action_audit(a,s.links)
    if not au['feasible']: raise RuntimeError('infeasible action')
    # Both decoders return a column vector in the current implementation.
    # Canonicalize to [num_links, 1]; adding another singleton axis would
    # broadcast the rate computation to a spurious link-by-link matrix.
    a=torch.as_tensor(a).reshape(-1)
    return float(calculate_achievable_rate(a.reshape(-1,1),s,binary=True,detach_hard=True).sum()),float(a.float().mean()),d
def evaluate(model,states,powers,label,thr):
    data={m:[] for m in ('Proposed-GAT','Local Greedy')}
    with torch.no_grad():
      for i,(s,p) in enumerate(zip(states,powers)):
        o=obs(s,p,label,thr); device=next(model.parameters()).device; pi,_=model(o.vertex_features.to(device),o.edge_index.to(device),o.edge_weights.to(device)); r,a,d=settle((pi>=.5).float().cpu(),direct_rate_priority(pi, o),s,True); data['Proposed-GAT'].append((r,a,{'sample_index':i,'hard_feasibility':d}))
        ca=local_greedy(o,interference_threshold=.3); r,a,d=settle(ca,o.vertex_features[:,0],s,False); data['Local Greedy'].append((r,a,{'sample_index':i,'hard_feasibility':d}))
    return {m:{'mean_aggregate_phy_rate_mbps':float(np.mean([x[0] for x in v])*5),'sem_aggregate_phy_rate_mbps':float(np.std([x[0] for x in v],ddof=1)/np.sqrt(len(v))*5),'mean_active_fraction':float(np.mean([x[1] for x in v])),'per_sample_actions':[x[2] for x in v]} for m,v in data.items()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--num-uav',type=int,default=20);p.add_argument('--output',required=True);p.add_argument('--device',default='auto',choices=('auto','cpu','cuda'));a=p.parse_args();c=cfg(a.num_uav); device='cuda' if a.device=='auto' and torch.cuda.is_available() else a.device if a.device!='auto' else 'cpu'; print(f'[evaluate] GAT device={device}',flush=True); st,fp,path=load_test(c,a.num_uav); m,ck=load_channel_gat(c,a.num_uav,device=device);g=build_network_generator(c); modes={'perfect':[x.signal_power for x in st],'large_scale':[x.signal_power_large_scale for x in st]}; modes.update({f'quantized_{b}bit':[qpower(x,g,b) for x in st] for b in (2,4,5,8)});res={k:evaluate(m,st,v,k,float(c['SNR_THRES'])) for k,v in modes.items()}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps({'schema':'tnse_fig07_csi_v1','num_uav':a.num_uav,'dataset_fingerprint':fp,'dataset_path':path,'checkpoint':ck,'definition':'only observed CSI changes; true Rician channel settles service','results':res},indent=2))
if __name__=='__main__': main()
