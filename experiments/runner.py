import random
import numpy as np
import torch
import os
import pickle as pkl

from utils.network_generate import NetworkGenerator
from models.model import GATScheduler
from utils.io import save_experiment_results, generate_dataset
from train.trainer import Trainer
from utils.baseline import eval_baseline
import json

class ExperimentRunner:
    def __init__(self, params, results_dir="./results", log_dir="./logs", gpu_id=None):
        self.params = params
        self.results_dir = results_dir
        self.log_dir = log_dir
        # Use gpu_id from CLI if provided, otherwise from config, default to 0
        self.gpu_id = gpu_id if gpu_id is not None else params.get("GPU_ID", 0)

    def run(self):
        p = self.params
        # set seed if provided
        seed = p.get("SEED", None)
        if seed is not None:
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        height_range = p.get("HEIGHT_RANGE", (50, 150))
        gen = NetworkGenerator(
            num_uavs=p.get("NUM_UAV", 50),
            area_width=p.get("AREA_WIDTH", 1000),
            height_min=height_range[0],
            height_max=height_range[1],
            uav_min_distance=p.get("MIN_DISTANCE", 20),
            tx_power=p.get("TX_POWER", 22),
            Gt=p.get("G_T", 0),
            Gr=p.get("G_R", 0),
            bandwidth=p.get("BANDWIDTH", 5e6),      # 5 MHz（与基线一致）
            carrier_f=p.get("CARRIER_FREQ", 2.4e9), # 2.4 GHz（与基线一致）
            noise_psd=p.get("NOISE_PSD", -169),     # -169 dBm/Hz（与基线一致）
            snr_gap_db=p.get("SNR_GAP_DB", 0)       # 0 dB = 无gap（与基线一致）
        )

        # CSI 模式：统一控制模型输入 + 速率计算
        csi_mode = p.get("CSI_MODE", "perfect")
        q_bits = p.get("Q_BITS", 8)
        print(f"[runner] CSI_MODE={csi_mode}, Q_BITS={q_bits}")

        # 队列感知开关（定义在使用点之前，与 include_queue_feature 联动）
        use_queue = p.get("USE_QUEUE", False)
        include_queue_feature = use_queue
        print(f"[runner] USE_QUEUE={use_queue}, include_queue_feature={include_queue_feature}")

        cache_path = p.get("DATASET_CACHE_PATH")

        # 初始化数据集变量
        train_set = None
        val_set = None
        test_set = None

        # 尝试从缓存加载数据集
        if cache_path and os.path.exists(cache_path):
            print(f"[runner] Loading cached datasets from {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    cached_datasets = pkl.load(f)
                full_train_set = cached_datasets['train_set']
                full_val_set = cached_datasets['val_set']
                full_test_set = cached_datasets['test_set']

                # sample according to requested sizes
                train_size = p.get("TRAIN_SIZE", 200)
                val_size = p.get("VAL_SIZE", 20)
                test_size = p.get("TEST_SIZE", 20)

                train_set = full_train_set[:train_size] if len(full_train_set) >= train_size else full_train_set
                val_set = full_val_set[:val_size] if len(full_val_set) >= val_size else full_val_set
                test_set = full_test_set[:test_size] if len(full_test_set) >= test_size else full_test_set

                print(f"[runner] Sampled from cache: train={len(train_set)}/{len(full_train_set)}, val={len(val_set)}/{len(full_val_set)}, test={len(test_set)}/{len(full_test_set)}")

                # 【安全检查】验证缓存数据的 include_queue_feature 与 link-level 队列语义
                if len(val_set) > 0:
                    cached_include_queue = getattr(val_set[0], 'include_queue_feature', None)
                    queue_shape_ok = True
                    if include_queue_feature:
                        q = getattr(val_set[0], 'queue_states', None)
                        queue_shape_ok = (
                            q is not None
                            and q.numel() == val_set[0].num_vertices
                            and val_set[0].vertex_features.size(1) >= 2
                        )
                    if cached_include_queue != include_queue_feature or not queue_shape_ok:
                        print(f"[runner] WARNING: Cached data include_queue_feature={cached_include_queue}, "
                              f"but current USE_QUEUE={use_queue}, queue_shape_ok={queue_shape_ok}. "
                              "Will regenerate.")
                        print(f"[runner] Expected vertex_features dim: {1 if not use_queue else 2}, "
                              f"cached dim: {val_set[0].vertex_features.size(1)}")
                        train_set = val_set = test_set = None
            except Exception as e:
                print(f"[runner] Failed to load cache {cache_path}: {e}, will generate new datasets")
                train_set = val_set = test_set = None

        if train_set is None or val_set is None or test_set is None:
            print("[runner] Generating new datasets...")
            train_set = generate_dataset(
                p.get("TRAIN_SIZE", 200), gen,
                batch_size=p.get("GEN_BATCH_SIZE", 50),
                csi_mode=csi_mode,
                q_bits=q_bits,
                include_queue_feature=include_queue_feature,
            )
            val_set = generate_dataset(
                p.get("VAL_SIZE", 20), gen,
                batch_size=p.get("GEN_BATCH_SIZE", 50),
                csi_mode=csi_mode,
                q_bits=q_bits,
                include_queue_feature=include_queue_feature,
            )
            test_set = generate_dataset(
                p.get("TEST_SIZE", 20), gen,
                batch_size=p.get("GEN_BATCH_SIZE", 50),
                csi_mode=csi_mode,
                q_bits=q_bits,
                include_queue_feature=include_queue_feature,
            )
            print(f"[runner] Generated datasets: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

            # save generated datasets if cache path is specified
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                datasets_to_save = {
                    'train_set': train_set,
                    'val_set': val_set,
                    'test_set': test_set
                }
                with open(cache_path, "wb") as f:
                    pkl.dump(datasets_to_save, f)
                print(f"[runner] Saved generated datasets to {cache_path}")

        # compute and print validation set link count statistics (avg and median) for quick reference
        val_counts = [int(s.links.size(0)) for s in val_set]
        val_avg = float(np.mean(val_counts)) if val_counts else 0.0
        val_median = float(np.median(val_counts)) if val_counts else 0.0
        print(f"[runner] Validation set link counts — avg: {val_avg:.1f}, median: {val_median:.1f}")

        # visualize network
        # sample_state = train_set[0]
        # fig_init = generator.visualize_network(sample_state, "uav_network_init.png")
        # fig_init.savefig("uav_network_init.png")
        # plt.show()
        
        # 功率控制开关
        use_power_control = p.get("USE_POWER_CONTROL", False)
        power_hidden_dim = p.get("POWER_HIDDEN_DIM", 16)
        print(f"[runner] USE_POWER_CONTROL={use_power_control}")

        # 队列感知开关（已在前面定义）
        # use_queue = p.get("USE_QUEUE", False)
        # include_queue_feature = use_queue
        print(f"[runner] USE_QUEUE={use_queue}")

        # Determine node input dimensionality.
        # 自动检测：如果 USE_QUEUE=True，则需要 2 维特征 [SNR, queue]
        # 如果 USE_QUEUE=False，则只需要 1 维特征 [SNR]
        if p.get("NODE_RAW_DIM") is not None:
            node_raw_dim = p.get("NODE_RAW_DIM")
        else:
            node_raw_dim = 2 if use_queue else 1

        model = GATScheduler(
            raw_node_dim=node_raw_dim,
            mlp_hidden_dims=p.get("MLP_HIDDEN_DIMS", [32,32]),
            embed_node_dim=p.get("NODE_EMBED_DIM", 32),
            edge_dim=p.get("EDGE_DIM", 1),
            gat_out_dim=p.get("GAT_OUT_DIM", 48),
            num_heads=p.get("NUM_HEADS", 4),
            T_iterations=p.get("NUM_GAT_LAYERS", 4),
            schedule_hidden_dim=p.get("SCHEDULE_HIDDEN", 32),
            use_power_control=use_power_control,
            power_hidden_dim=power_hidden_dim
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=p.get("LR", 1e-4))
        # build lr scheduler if requested
        lr_scheduler = None
        lr_type = p.get("LR_SCHEDULER_TYPE", None)
        lr_cfg = p.get("LR_SCHEDULER_CONFIG", {}) or {}
        if lr_type == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_cfg)
        elif lr_type == "StepLR":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_cfg)


        # Auto-detect GPU if no device specified in params
        default_device = None  # Let Trainer handle device detection
        device_param = p.get("DEVICE", default_device)
        trainer = Trainer(
            model, optimizer, device=device_param, gpu_id=self.gpu_id, log_dir=self.log_dir,
            grad_clip_config=p.get("GRAD_CLIP_CONFIG", None),
            use_power_control=use_power_control,
            use_queue=use_queue,
        )

        trained = trainer.train(
            num_epochs=p.get("EPOCHS", 20),
            train_dataset=train_set,
            val_dataset=val_set,
            eval_interval=p.get("EVAL_INTERVAL", 1),
            lr_scheduler=lr_scheduler,
            use_power_control=use_power_control,
            use_queue=use_queue,
        )

        # use trainer's evaluate
        test_result = trainer.evaluate(model, test_set, use_power_control=use_power_control, use_queue=use_queue)
        if use_power_control:
            test_util, test_alpha = test_result
        else:
            test_util = test_result

        # save results - use EXCEL_SHEET from config first, then fallback to experiment group
        excel_sheet = p.get("EXCEL_SHEET", None)
        if excel_sheet is None:
            # Fallback to experiment group based default
            experiment_group = p.get("EXPERIMENT_GROUP", "default")
            if experiment_group == "convergence_analysis":
                excel_sheet = "Sheet2"  # 收敛性分析实验
            elif experiment_group in ("csi_imperfection"):
                excel_sheet = "Sheet3"  # CSI 不完美性实验
            else:
                excel_sheet = "Sheet1"  # 默认参数选择实验
        info = save_experiment_results(trained, train_set, test_util, p, results_dir=self.results_dir, log_dir=self.log_dir, excel_sheet=excel_sheet)
        experiment_id = info.get("experiment_id", None)

        # optionally run baselines and save their results
        if p.get("EVAL_BASELINES", True):
            try:
                baseline_results = eval_baseline(test_set)
                # Always save baseline results inside the experiment directory
                experiment_dir = info.get("experiment_dir", self.results_dir) if 'info' in locals() else self.results_dir
                os.makedirs(experiment_dir, exist_ok=True)
                baseline_filename = f"{experiment_id}_baselines.json" if experiment_id else "baselines.json"
                baseline_path = os.path.join(experiment_dir, baseline_filename)
                with open(baseline_path, "w") as bf:
                    json.dump(baseline_results, bf, indent=2)
                print(f"[runner] Baseline results saved to {baseline_path}")
            except Exception as e:
                print(f"[runner] Failed to run baselines: {e}")

        return {"trained": trained, "test_util": test_util}

