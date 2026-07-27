#!/usr/bin/env python3
"""Ultra-quick comparison: conference pipeline + Rayleigh vs Rician data-efficiency."""
import sys, random, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def gen_states(num_uav, n, seed, channel):
    torch.manual_seed(seed)
    from conference_GNN_Scheduling.utils.network_generate import NetworkGenerator
    gen = NetworkGenerator(
        num_uavs=num_uav, area_width=1000, height_min=50, height_max=150,
        uav_min_distance=20, tx_power=10, Gt=0, Gr=0,
        noise_psd=-169, bandwidth=5e6, carrier_f=2.4e9,
        snr_thres=5, snr_gap_db=0,
        channel_model=channel, rician_k_db=5.0,
    )
    return [gen.generate_network(csi_mode="perfect", include_queue_feature=False)
            for _ in range(n)]


def quick_train_test(train_states, val_states, test_states, n_epochs=15, seed=10086):
    """Train using conference pipeline, return test utilities as list."""
    from conference_GNN_Scheduling.train.trainer import Trainer
    from conference_GNN_Scheduling.models.model import GATScheduler as ConfGAT, calculate_achievable_rate
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = ConfGAT(raw_node_dim=1, mlp_hidden_dims=[32, 32], embed_node_dim=32,
                edge_dim=1, gat_out_dim=48, num_heads=4,
                T_iterations=4, schedule_hidden_dim=32).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                      patience=5, min_lr=1e-5)
    tr = Trainer(m, opt, device=DEVICE, grad_clip_config={"explosion_threshold": 100,
                                                          "explosion_clip": 10.0, "normal_clip": 5.0})
    out = tr.train(n_epochs, train_states, val_states,
                   eval_interval=1, lr_scheduler=sch, use_queue=False)
    m.eval()
    test_utils = []
    with torch.no_grad():
        for state in test_states:
            x = state.vertex_features[:, :1].to(DEVICE)
            pi, _ = m(x, state.edge_index.to(DEVICE), state.edge_weights.to(DEVICE))
            rates = calculate_achievable_rate(pi, state, binary=True, detach_hard=True)
            test_utils.append(rates.sum().item())
    return test_utils


def main():
    num_uav, seeds = 40, [10086, 10087, 10088]
    sizes = [10, 30, 60, 100]

    print(f"Device: {DEVICE}", flush=True)

    for channel in ["rayleigh", "rician"]:
        print(f"\n{'='*60}", flush=True)
        print(f"Channel: {channel}", flush=True)
        print(f"{'='*60}", flush=True)

        print("Generating pool=300, val=128, test=128...", flush=True)
        pool = gen_states(num_uav, 300, seeds[0], channel)
        val  = gen_states(num_uav, 128, seeds[1], channel)
        test = gen_states(num_uav, 128, seeds[2], channel)
        print(f"  links/sample: {pool[0].num_vertices}", flush=True)

        print(f"\n{'n':>5} | {'test utilities (3 seeds)':>40} | {'mean':>8} {'std':>6}", flush=True)
        print("-" * 70, flush=True)

        for n in sizes:
            test_utils_all = []
            for s_seed in [10086, 10087, 10088]:
                torch.manual_seed(s_seed)
                random.seed(s_seed)
                np.random.seed(s_seed)
                utils = quick_train_test(pool[:n], val, test, n_epochs=15, seed=s_seed)
                test_utils_all.append(np.mean(utils))
                print(f"  seed {s_seed%10000}: mean={np.mean(utils):.2f}, std={np.std(utils):.2f}", flush=True)

            print(f"{n:>5} | {[f'{u:.1f}' for u in test_utils_all]} | "
                  f"{np.mean(test_utils_all):>8.2f} {np.std(test_utils_all):>6.2f}", flush=True)

    print(f"\n\nDone. Compare the noise level between Rayleigh and Rician.", flush=True)


if __name__ == "__main__":
    main()
