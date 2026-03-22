import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_metrics_from_json(json_path: str, out_dir: str = None, show: bool = False) -> List[str]:
    """
    Read per-epoch json (list of dicts) and save common plots:
      - train_loss
      - val_utility
      - pi_mean
      - grad_norm_mean
      - conflict_penalty

    Returns list of saved file paths.
    """
    if out_dir is None:
        out_dir = os.path.dirname(json_path)
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "r") as jf:
        logs = json.load(jf)
    if not logs:
        raise RuntimeError(f"No logs in {json_path}")

    epochs = [e.get("epoch", idx) for idx, e in enumerate(logs)]

    def _plot(key, ylabel, fname):
        vals = [e.get(key, np.nan) for e in logs]
        if all(np.isnan(vals)):
            return None
        plt.figure(figsize=(6,4))
        plt.plot(epochs, vals, marker='o', linewidth=1)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.4)
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        if show:
            plt.show()
        plt.close()
        return out_path

    saved = []
    mapping = [
        ("train_loss", "Train Loss", f"{os.path.basename(json_path).replace('.json','')}_train_loss.png"),
        ("val_utility", "Val Utility", f"{os.path.basename(json_path).replace('.json','')}_val_utility.png"),
        ("utility", "Utility (train avg)", f"{os.path.basename(json_path).replace('.json','')}_utility.png"),
        ("pi_mean", "pi_mean", f"{os.path.basename(json_path).replace('.json','')}_pi_mean.png"),
        ("grad_norm_mean", "grad_norm_mean", f"{os.path.basename(json_path).replace('.json','')}_grad_norm_mean.png"),
        ("conflict_penalty", "conflict_penalty", f"{os.path.basename(json_path).replace('.json','')}_conflict_penalty.png"),
        ("sparsity", "sparsity (mean pi)", f"{os.path.basename(json_path).replace('.json','')}_sparsity.png"),
    ]

    for key, ylabel, fname in mapping:
        out = _plot(key, ylabel, fname)
        if out:
            saved.append(out)

    return saved

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot common metrics from per-epoch json")
    parser.add_argument("json_path", help="path to per-epoch json")
    parser.add_argument("--out-dir", help="output directory for plots", default=None)
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    args = parser.parse_args()
    paths = plot_metrics_from_json(args.json_path, args.out_dir, args.show)
    print("Saved plots:", paths)

