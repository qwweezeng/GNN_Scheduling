#!/usr/bin/env python3
"""
逐条隔离测试（GPU + 内存生成版）：会议版 vs 期刊版差异排查

测试 #4: 在 GPU 上用会议版生成器生成数据，验证 conference pipeline 是否平滑
测试 #1: 两版 model forward 是否等价（forward 等价则 model.py 无差异）
"""
import sys, os, pickle, json, random, argparse
from pathlib import Path

# Prepend workspace root so "import conference_GNN_Scheduling..." resolves
_WS = Path(__file__).resolve().parents[2]
if str(_WS) not in sys.path:
    sys.path.insert(0, str(_WS))

import numpy as np
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# 数据生成
# ------------------------------------------------------------------
def gen_conf_dataset(num_uav, n, seed):
    """会议版 Rayleigh 生成器"""
    torch.manual_seed(seed)
    from conference_GNN_Scheduling.utils.network_generate import NetworkGenerator
    gen = NetworkGenerator(
        num_uavs=num_uav, area_width=1000, height_min=50, height_max=150,
        uav_min_distance=20, tx_power=10, Gt=0, Gr=0, noise_psd=-169,
        bandwidth=5e6, carrier_f=2.4e9, snr_thres=5, snr_gap_db=0)
    return [gen.generate_network(csi_mode="perfect", include_queue_feature=False)
            for _ in range(n)]


# ------------------------------------------------------------------
# 训练辅助
# ------------------------------------------------------------------
def train(model_cls, model_kwargs, train_states, val_states,
          n_epochs=30, lr=1e-4, seed=10086):
    from conference_GNN_Scheduling.train.trainer import Trainer
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = model_cls(**model_kwargs).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=10, min_lr=1e-5)
    tr = Trainer(m, opt, device=DEVICE,
                 grad_clip_config={"explosion_threshold": 100,
                                   "explosion_clip": 10.0, "normal_clip": 5.0})
    out = tr.train(n_epochs, train_states, val_states,
                  eval_interval=1, lr_scheduler=sch, use_queue=False)
    vals = [x["val_utility"] or 0.0 for x in out["per_epoch_logs"]]
    return {
        "n": len(train_states), "best": out["best_val_util"],
        "std": float(np.std(vals)), "neg": int(np.sum(np.diff(vals) < 0)),
        "min": float(np.min(vals)), "max": float(np.max(vals)),
        "per_epoch": vals,
    }


# ------------------------------------------------------------------
# TEST #4: 完全 conference pipeline
# ------------------------------------------------------------------
def test04_conf_pipeline():
    print("\n" + "=" * 70)
    print("TEST #4: conference model + conference trainer + conference data")
    print("=" * 70)
    from conference_GNN_Scheduling.models.model import GATScheduler as ConfGAT

    print("  生成 conference 数据...", flush=True)
    pool = gen_conf_dataset(40, 300, 10086)
    val  = gen_conf_dataset(40, 128, 10087)
    print(f"  完成: pool={len(pool)}, val={len(val)}, links={pool[0].num_vertices}")

    kw = {"raw_node_dim": 1, "mlp_hidden_dims": [32, 32], "embed_node_dim": 32,
          "edge_dim": 1, "gat_out_dim": 48, "num_heads": 4,
          "T_iterations": 4, "schedule_hidden_dim": 32}
    res = {}
    for n in [20, 40, 60]:
        print(f"\n  Training n={n}...", flush=True)
        r = train(ConfGAT, kw, pool[:n], val, n_epochs=30)
        res[n] = r
        print(f"  n={n:>3}: best={r['best']:.2f}  std={r['std']:.2f}  "
              f"neg={r['neg']}  [min={r['min']:.1f}, max={r['max']:.1f}]")

    print("\n  per-epoch val_utility (n=40):")
    for i, v in enumerate(res[40]["per_epoch"]):
        print(f"    epoch {i:>2}: {v:.2f}"
              + (" <-- best" if abs(v - res[40]["best"]) < 0.01 else ""))

    return res


# ------------------------------------------------------------------
# TEST #1: 两版 forward 等价
# ------------------------------------------------------------------
def test01_forward_eq():
    print("\n" + "=" * 70)
    print("TEST #1: 两版 model forward 是否等价")
    print("=" * 70)
    from conference_GNN_Scheduling.models.model import GATScheduler as ConfGAT
    from models.model import GATScheduler as JrnlGAT

    c = ConfGAT(1, [32, 32], 32, 1, 48, 4, 4, 32).to(DEVICE)
    j = JrnlGAT(1, [32, 32], 32, 1, 48, 4, 4, 32).to(DEVICE)

    with torch.no_grad():
        for (cn, cv), (jn, jv) in zip(c.named_parameters(), j.named_parameters()):
            jv.copy_(cv)

    s = gen_conf_dataset(40, 1, 99999)[0]
    x, e, w = s.vertex_features[:, :1].to(DEVICE), s.edge_index.to(DEVICE), s.edge_weights.to(DEVICE)
    c.eval(); j.eval()
    with torch.no_grad():
        pi_c, _ = c(x, e, w, use_ste=False)
        pi_j, _ = j(x, e, w, use_ste=False)

    diff = (pi_c - pi_j).abs().max().item()
    print(f"  Conf pi: mean={pi_c.mean():.4f}  std={pi_c.std():.4f}")
    print(f"  Jrnl pi: mean={pi_j.mean():.4f}  std={pi_j.std():.4f}")
    print(f"  Max abs diff: {diff:.8f}")
    print("  => " + ("forward 完全相同" if diff < 1e-6 else f"forward 存在差异: {diff:.6f}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--test", default="all", choices=["all", "01", "04"])
    a = p.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(a.gpu_id)
    print("=" * 70)
    print(f"Isolation test  DEVICE={DEVICE}  test={a.test}")
    print("=" * 70)
    if a.test in ("all", "04"):
        test04_conf_pipeline()
    if a.test in ("all", "01"):
        test01_forward_eq()
    print("\nDone.")


if __name__ == "__main__":
    main()
