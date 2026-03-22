import os
import random
from time import strftime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.model import calculate_achievable_rate, compute_conflict_penalty
from utils.baseline import eval_baseline

class Trainer:
    def __init__(self, model, optimizer, device=None, gpu_id=None, log_dir="./logs", grad_clip_config=None):
        self.model = model
        self.optimizer = optimizer

        # Auto-detect GPU if no device specified or device is "auto"
        if device is None or device == "auto":
            if torch.cuda.is_available():
                # Use specified gpu_id, default to 0 if not specified
                gpu_id = gpu_id if gpu_id is not None else 0
                if gpu_id >= torch.cuda.device_count():
                    print(f"[Trainer] Warning: gpu_id {gpu_id} >= available GPUs ({torch.cuda.device_count()}), using GPU 0")
                    gpu_id = 0
                device = torch.device(f"cuda:{gpu_id}")
                print(f"[Trainer] Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                device = torch.device("cpu")
                print("[Trainer] Using CPU (no GPU available)")
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.log_dir = log_dir
        if grad_clip_config is None:
            grad_clip_config = {
                'explosion_threshold': 100,
                'explosion_clip': 10.0,
                'normal_clip': 5.0
            }
        self.grad_clip_config = grad_clip_config

    def evaluate(self, model, dataset, device=None):
        if device is None:
            device = self.device
        model.eval()
        model.to(device)
        total_util = 0.0
        with torch.no_grad():
            np.random.shuffle(dataset)
            for state in dataset:
                x = state.vertex_features.to(device)
                edge_index = state.edge_index.to(device)
                edge_weights = state.edge_weights.to(device)
                pi, _ = model(x, edge_index, edge_weights, use_ste=False)
                rates = calculate_achievable_rate(pi, state, binary=True, detach_hard=True)
                # Q = x[:, 1]
                Q = 1
                util = torch.sum(Q * rates).item()
                total_util += util
        avg_util = total_util / len(dataset) if len(dataset) > 0 else 0.0
        return avg_util

    def train(self, num_epochs, train_dataset, val_dataset,
              lambda_sparsity=0.01, eval_interval=10,
              lambda_conflict=0.1, conflict_warmup_epochs=5,
              lr_scheduler=None):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        model.train()
        model.to(device)
        starttime = strftime("%Y-%m-%d_%H-%M")
        writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "train_"+starttime[:16]), flush_secs=60)

        # model graph (需要sample input，如果train_dataset不为空则记录)
        if len(train_dataset) > 0:
            try:
                sample_state = train_dataset[0]
                sample_x = sample_state.vertex_features.to(device)
                sample_edge_index = sample_state.edge_index.to(device)
                sample_edge_attr = sample_state.edge_weights.to(device)
                writer.add_graph(model, (sample_x, sample_edge_index, sample_edge_attr))
            except Exception:
                # 如果add_graph失败，跳过
                pass

        best_val_util = -float('inf')
        per_epoch_logs = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            random.shuffle(train_dataset)
            pi_means = []
            pi_stds = []
            grad_norms = []
            conflict_penalties = []
            utility_terms = []
            sparsity_penalties = []

            for state in train_dataset:
                x = state.vertex_features.to(device)
                edge_index = state.edge_index.to(device)
                edge_weights = state.edge_weights.to(device)
                # forward and backward (simplified, single path)
                pi, _ = model(x, edge_index, edge_weights, use_ste=False)
                pi_means.append(pi.mean().item())
                pi_stds.append(pi.std().item())
                rates = calculate_achievable_rate(pi, state, binary=False)  # 训练阶段使用软概率保持梯度
                
                # Q = x[:, 1]  # 队列长度
                Q = 1  # 先不加入考虑队列长度
                mean_pi = torch.mean(pi)
                # normalized sparsity penalty = lambda_sparsity * mean(pi)
                sparsity_penalty = lambda_sparsity * mean_pi

                # Warmup for conflict penalty
                if epoch < conflict_warmup_epochs:
                    warmup_factor = (epoch + 1) / conflict_warmup_epochs
                    current_penalty_scale = lambda_conflict * warmup_factor
                else:
                    current_penalty_scale = lambda_conflict
                conflict_penalty = compute_conflict_penalty(
                    pi, edge_index, edge_weights,
                    penalty_scale=current_penalty_scale,
                    normalize=True
                )

                utility_term = torch.sum(Q * rates)
                sparsity_penalties.append(sparsity_penalty.item())
                conflict_penalties.append(conflict_penalty.item())
                # use unified, normalized name 'sparsity_penalty' for the sparsity regularizer term
                loss = -utility_term + sparsity_penalty + conflict_penalty
                # loss = -utility_term
                
                optimizer.zero_grad()
                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
                grad_norms.append(total_norm.item())
                # clip grad
                explosion_threshold = self.grad_clip_config['explosion_threshold']
                explosion_clip = self.grad_clip_config['explosion_clip']
                normal_clip = self.grad_clip_config['normal_clip']
                if total_norm.item() > explosion_threshold: # 梯度爆炸：使用温和的clip，保留更多信息
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=explosion_clip)
                else: # 正常范围：使用标准clip
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=normal_clip)
                
                optimizer.step()
                epoch_loss += loss.item()
                utility_terms.append(utility_term.item())

            avg_loss = epoch_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
            avg_utility_term = np.mean(utility_terms) if utility_terms else 0.0
            avg_sparsity_penalty = np.mean(sparsity_penalties) if sparsity_penalties else 0.0
            avg_conflict_penalty = np.mean(conflict_penalties) if conflict_penalties else 0.0

            writer.add_scalar("train_loss", avg_loss, epoch)
            writer.add_scalar("pi_mean", np.mean(pi_means) if pi_means else 0.0, epoch)
            writer.add_scalar("pi_std", np.mean(pi_stds) if pi_stds else 0.0, epoch)
            # concise console print for quick per-epoch pi stats
            avg_pi_mean = float(np.mean(pi_means)) if pi_means else 0.0
            avg_pi_std = float(np.mean(pi_stds)) if pi_stds else 0.0
            print(f"[Epoch {epoch}] pi_mean={avg_pi_mean:.4f}, pi_std={avg_pi_std:.4f}, grad_norm={np.mean(grad_norms):.4f}")
            writer.add_scalar("grad_norm", np.mean(grad_norms) if grad_norms else 0.0, epoch)
            writer.add_scalar("utility_term", avg_utility_term, epoch)
            writer.add_scalar("sparsity_penalty", avg_sparsity_penalty, epoch)
            writer.add_scalar("conflict_penalty", avg_conflict_penalty, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("learning_rate", current_lr, epoch)

            # evaluate
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                eval_util = self.evaluate(model, val_dataset, device=device)
                # run expensive baselines only at epoch 0 (quick-run mode)
                results = {}
                # if epoch == 0:
                #     results = eval_baseline(val_dataset, baseline_names=['random', 'simple_greedy', 'local_greedy', 'greedy_spatial', 'fplinq'])
                # elif epoch == num_epochs - 1:
                #     results = eval_baseline(val_dataset, baseline_names=['random', 'simple_greedy', 'greedy_spatial', 'fplinq'])
                # else:
                #     results = {}
                writer.add_scalar("val_utility", eval_util, epoch)
                current_lr = optimizer.param_groups[0]['lr']
                # lr scheduler step
                if lr_scheduler is not None:
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # only call step(metric) if eval_util is a valid finite number
                        valid_metric = isinstance(eval_util, (int, float)) and not (np.isnan(eval_util) or np.isinf(eval_util))
                        if valid_metric:
                            lr_scheduler.step(eval_util)
                        else:
                            # skip stepping ReduceLROnPlateau when metric is invalid (avoid TypeError)
                            print("[Trainer] Warning: eval_util invalid, skipping ReduceLROnPlateau.step() this epoch")
                    else:
                        # other schedulers (StepLR, CosineAnnealingLR) use step() without args
                        lr_scheduler.step()
                    # update current_lr after potential scheduler change
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"📉 Learning rate updated to: {current_lr:.6f}")

                # update best validation utility and save checkpoint
                if eval_util > best_val_util:
                    best_val_util = eval_util
                    try:
                        models_dir = os.path.join(self.log_dir, "models")
                        os.makedirs(models_dir, exist_ok=True)
                        ckpt_path = os.path.join(models_dir, f"best_model_epoch{epoch}.pth")
                        torch.save(model.state_dict(), ckpt_path)
                        print(f"[Trainer] Saved best model to {ckpt_path}")
                    except Exception as e:
                        print(f"[Trainer] Failed to save best model: {e}")

                print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Utility: {avg_utility_term:.4f} | "
                      f"Sparsity: {avg_sparsity_penalty:.6f} | Conflict: {avg_conflict_penalty:.6f} | "
                      f"Val Utility: {eval_util:.2f} | Random: {results.get('random',0):.2f} | "
                      f"Simple: {results.get('simple_greedy',0):.2f} | LGS: {results.get('local_greedy',0):.2f} | "
                      f"GSp: {results.get('greedy_spatial',0):.2f} | FPLinQ: {results.get('fplinq',0):.2f} | LR: {current_lr:.6f}")
            else:
                print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Utility: {avg_utility_term:.4f} | LR: {current_lr:.6f}")

            epoch_log = {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "pi_mean": float(np.mean(pi_means)) if pi_means else None,
                "pi_std": float(np.mean(pi_stds)) if pi_stds else None,
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
                "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
                "utility": float(avg_utility_term),
                "sparsity_penalty": float(avg_sparsity_penalty),
                "conflict_penalty": float(avg_conflict_penalty),
                "learning_rate": float(current_lr),
                "val_utility": float(eval_util) if 'eval_util' in locals() else None
            }
            per_epoch_logs.append(epoch_log)

        writer.close()
        return {
            "model": model,
            "per_epoch_logs": per_epoch_logs,
            "best_val_util": best_val_util,
            "final_train_loss": float(avg_loss) if 'avg_loss' in locals() else None
        }


