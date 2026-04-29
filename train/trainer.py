import os
import random
from time import strftime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from models.model import calculate_achievable_rate
from utils.baseline import eval_baseline

class Trainer:
    def __init__(self, model, optimizer, device=None, gpu_id=None, log_dir="./logs",
                 grad_clip_config=None, use_power_control=False, use_queue=False):
        self.model = model
        self.optimizer = optimizer
        self.use_power_control = use_power_control
        self.use_queue = use_queue  # 是否启用队列感知训练（MWS加权loss）

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

    def evaluate(self, model, dataset, device=None, use_power_control=False, use_queue=False):
        if device is None:
            device = self.device
        model.eval()
        model.to(device)
        total_util = 0.0
        total_avg_alpha = 0.0
        with torch.no_grad():
            np.random.shuffle(dataset)
            for state in dataset:
                x = state.vertex_features.to(device)
                edge_index = state.edge_index.to(device)
                edge_weights = state.edge_weights.to(device)
                result = model(x, edge_index, edge_weights, use_ste=False)
                if use_power_control:
                    pi, alpha, _ = result
                    rates = calculate_achievable_rate(pi, state, binary=True, detach_hard=True, alpha=alpha)
                    total_avg_alpha += alpha.mean().item()
                else:
                    pi, _ = result
                    rates = calculate_achievable_rate(pi, state, binary=True, detach_hard=True)
                
                # 队列感知评估
                if use_queue:
                    if hasattr(state, 'queue_states'):
                        Q = state.queue_states.to(device)
                    else:
                        Q = torch.ones(state.num_vertices, device=device)
                    util = torch.sum(Q * rates).item()
                else:
                    util = torch.sum(rates).item()
                total_util += util
        avg_util = total_util / len(dataset) if len(dataset) > 0 else 0.0
        avg_alpha = total_avg_alpha / len(dataset) if len(dataset) > 0 else 1.0
        if use_power_control:
            return avg_util, avg_alpha
        return avg_util

    def train(self, num_epochs, train_dataset, val_dataset,
              eval_interval=10,
              lr_scheduler=None, use_power_control=False, use_queue=None):
        # 如果传入None，使用实例级别的use_queue设置
        if use_queue is None:
            use_queue = self.use_queue
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
            alpha_means = []
            grad_norms = []
            utility_terms = []

            for state in train_dataset:
                x = state.vertex_features.to(device)
                edge_index = state.edge_index.to(device)
                edge_weights = state.edge_weights.to(device)
                # forward and backward
                result = model(x, edge_index, edge_weights, use_ste=False)
                if use_power_control:
                    pi, alpha, _ = result
                    alpha_means.append(alpha.mean().item())
                    rates = calculate_achievable_rate(pi, state, binary=False, alpha=alpha)
                else:
                    pi, _ = result
                    alpha = None
                    rates = calculate_achievable_rate(pi, state, binary=False)

                pi_means.append(pi.mean().item())
                pi_stds.append(pi.std().item())
                
                # 队列感知MWS加权
                if self.use_queue:
                    # 从 vertex_features 提取归一化队列（第二维）
                    # vertex_features 格式: [SNR_dB, queue_normalized]
                    queue_norm = x[:, 1].clone()  # [num_links]
                    
                    # 转换回原始队列长度用于加权
                    # queue_normalized = log1p(queue) / 10 => queue = expm1(queue_normalized * 10) - 1
                    Q_raw = torch.expm1(queue_norm * 10.0) - 1.0
                    Q_raw = torch.clamp(Q_raw, min=0.0)
                    
                    # 或者直接从 state.queue_states 获取（更准确）
                    if hasattr(state, 'queue_states'):
                        Q_raw = state.queue_states.to(device)
                    
                    # MWS目标: 最大化 sum(Q_i * R_i)
                    utility_term = torch.sum(Q_raw * rates)
                else:
                    utility_term = torch.sum(rates)
                
                loss = -utility_term
                
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

            writer.add_scalar("train_loss", avg_loss, epoch)
            writer.add_scalar("pi_mean", np.mean(pi_means) if pi_means else 0.0, epoch)
            writer.add_scalar("pi_std", np.mean(pi_stds) if pi_stds else 0.0, epoch)
            # concise console print for quick per-epoch pi stats
            avg_pi_mean = float(np.mean(pi_means)) if pi_means else 0.0
            avg_pi_std = float(np.mean(pi_stds)) if pi_stds else 0.0
            print(f"[Epoch {epoch}] pi_mean={avg_pi_mean:.4f}, pi_std={avg_pi_std:.4f}, grad_norm={np.mean(grad_norms):.4f}")
            if use_power_control:
                avg_alpha_mean = float(np.mean(alpha_means)) if alpha_means else 1.0
                print(f"  alpha_mean={avg_alpha_mean:.4f}", end="")
            writer.add_scalar("grad_norm", np.mean(grad_norms) if grad_norms else 0.0, epoch)
            writer.add_scalar("utility_term", avg_utility_term, epoch)
            if use_power_control:
                avg_alpha_mean = float(np.mean(alpha_means)) if alpha_means else 1.0
                writer.add_scalar("alpha_mean", avg_alpha_mean, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("learning_rate", current_lr, epoch)

            # evaluate
            if (epoch + 1) % eval_interval == 0 or epoch == num_epochs - 1:
                eval_result = self.evaluate(model, val_dataset, device=device, use_power_control=use_power_control, use_queue=(use_queue or self.use_queue))
                if use_power_control:
                    eval_util, eval_alpha = eval_result
                else:
                    eval_util = eval_result
                    eval_alpha = 1.0
                # run expensive baselines only at epoch 0 (quick-run mode)
                results = {}
                # if epoch == 0:
                #     results = eval_baseline(val_dataset, baseline_names=['random', 'simple_greedy', 'local_greedy', 'greedy_spatial', 'fplinq'])
                # elif epoch == num_epochs - 1:
                #     results = eval_baseline(val_dataset, baseline_names=['random', 'simple_greedy', 'greedy_spatial', 'fplinq'])
                # else:
                #     results = {}
                writer.add_scalar("val_utility", eval_util, epoch)
                writer.add_scalar("val_alpha", eval_alpha, epoch)
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

                if use_power_control:
                    print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Utility: {avg_utility_term:.4f} | "
                          f"Alpha: {avg_alpha_mean:.4f} | "
                          f"Val Utility: {eval_util:.2f} | Val Alpha: {eval_alpha:.4f} | LR: {current_lr:.6f}")
                else:
                    print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Utility: {avg_utility_term:.4f} | "
                          f"Val Utility: {eval_util:.2f} | LR: {current_lr:.6f}")
            else:
                print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f} | Utility: {avg_utility_term:.4f} | LR: {current_lr:.6f}")

            epoch_log = {
                "epoch": epoch,
                "train_loss": float(avg_loss),
                "pi_mean": float(np.mean(pi_means)) if pi_means else None,
                "pi_std": float(np.mean(pi_stds)) if pi_stds else None,
                "alpha_mean": float(np.mean(alpha_means)) if alpha_means and any(alpha_means) else None,
                "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else None,
                "grad_norm_max": float(np.max(grad_norms)) if grad_norms else None,
                "utility": float(avg_utility_term),
                "learning_rate": float(current_lr),
                "val_utility": float(eval_util) if 'eval_util' in locals() else None,
                "val_alpha": float(eval_alpha) if 'eval_alpha' in locals() else None
            }
            per_epoch_logs.append(epoch_log)

        writer.close()
        return {
            "model": model,
            "per_epoch_logs": per_epoch_logs,
            "best_val_util": best_val_util,
            "final_train_loss": float(avg_loss) if 'avg_loss' in locals() else None
        }


