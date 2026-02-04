from core.utils1.config import cfg  # isort: split

import os
import time
import random
from datetime import datetime
import numpy as np
import torch

from tensorboardX import SummaryWriter
from tqdm import tqdm

from core.utils1.datasets import create_dataloader, create_dual_branch_dataloader, create_video_dataloader
from core.utils1.earlystop import EarlyStopping
from core.utils1.eval import get_val_cfg, validate, validate_dual_branch, validate_video
from core.utils1.trainer import Trainer, DualBranchTrainer, VideoTrainer
from core.utils1.utils import Logger

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int):
    """设置随机种子，确保训练可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # 设置随机种子
    set_seed(cfg.seed)
    print(f"Random seed set to: {cfg.seed}")
    val_cfg = get_val_cfg(cfg, split="val", copy=True)
    cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
    # Also update optical_root for dual-branch/video-level training
    if cfg.optical_root:
        cfg.optical_root = os.path.join(cfg.optical_root, "train")

    # Choose dataloader based on mode
    if cfg.video_level:
        print("=" * 50)
        print("Video-Level Temporal Aggregation Mode Enabled")
        print(f"  - Frames per video: {cfg.num_frames}")
        print("=" * 50)
        data_loader = create_video_dataloader(cfg)
    elif cfg.dual_branch:
        print("=" * 50)
        print("Dual-Branch Frame-Level Fusion Mode Enabled")
        print("=" * 50)
        data_loader = create_dual_branch_dataloader(cfg)
    else:
        data_loader = create_dataloader(cfg)
    dataset_size = len(data_loader)

    log = Logger()
    log.open(cfg.logs_path, mode="a")

    # 输出详细的训练配置信息
    log.write("\n" + "=" * 70 + "\n")
    log.write("Training Configuration\n")
    log.write("=" * 70 + "\n")
    log.write(f"Experiment: {cfg.exp_name}\n")
    log.write(f"Training samples: {dataset_size * cfg.batch_size}\n")
    log.write(f"Batch size: {cfg.batch_size} | Epochs: {cfg.nepoch}\n")
    log.write(f"Optimizer: {cfg.optim} | Initial LR: {cfg.lr}\n")
    if cfg.video_level:
        log.write(f"Mode: Video-Level | Frames: {cfg.num_frames}\n")
        log.write(f"RGB arch: {cfg.arch_rgb} | Optical arch: {cfg.arch_optical}\n")
        log.write(f"Fusion: {cfg.fusion_type} | Feature dim: {cfg.feature_dim}\n")
        log.write(f"Transformer: {cfg.num_layers} layers, {cfg.num_heads} heads\n")
    elif cfg.dual_branch:
        log.write(f"Mode: Dual-Branch Frame-Level\n")
        log.write(f"RGB arch: {cfg.arch_rgb} | Optical arch: {cfg.arch_optical}\n")
        log.write(f"Fusion: {cfg.fusion_type} | Feature dim: {cfg.feature_dim}\n")
    else:
        log.write(f"Mode: Single-Branch | Arch: {cfg.arch}\n")
    log.write(f"Early stopping: {cfg.earlystop} (patience={cfg.earlystop_epoch})\n")
    log.write(f"Warmup: {cfg.warmup}\n")
    log.write(f"Random seed: {cfg.seed}\n")
    log.write("=" * 70 + "\n\n")

    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val"))

    # Choose trainer based on mode
    if cfg.video_level:
        trainer = VideoTrainer(cfg)
    elif cfg.dual_branch:
        trainer = DualBranchTrainer(cfg)
    else:
        trainer = Trainer(cfg)

    # Determine starting epoch
    # 注意：trainer在初始化时已经加载了checkpoint（如果continue_train=True）
    # 直接使用trainer.loaded_epoch，避免重复加载checkpoint
    start_epoch = 0
    if cfg.continue_train:
        start_epoch = trainer.loaded_epoch + 1
        log.write(f"[Resume] Loading checkpoint: {cfg.epoch}\n")
        log.write(f"[Resume] Checkpoint epoch: {trainer.loaded_epoch} -> Starting from epoch: {start_epoch}\n")
        log.write(f"[Resume] Total steps so far: {trainer.total_steps}\n")

    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=0.001, verbose=True)

    # Load early_stopping state if resuming training
    if cfg.continue_train and cfg.earlystop:
        earlystop_path = os.path.join(cfg.ckpt_dir, "earlystop_state.pth")
        if os.path.exists(earlystop_path):
            earlystop_state = torch.load(earlystop_path)
            early_stopping.best_score = earlystop_state.get("best_score")
            early_stopping.counter = earlystop_state.get("counter", 0)
            early_stopping.score_max = earlystop_state.get("score_max", -float('inf'))
            log.write(f"Loaded early_stopping state: best_score={early_stopping.best_score}, counter={early_stopping.counter}\n")

    for epoch in range(start_epoch, cfg.nepoch):
        epoch_start_time = time.time()
        epoch_start_datetime = datetime.now()
        epoch_iter = 0
        epoch_loss_sum = 0.0
        num_batches = 0

        # 获取当前学习率
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # Epoch开始信息
        log.write("\n" + "=" * 70 + "\n")
        log.write(f"Epoch {epoch}/{cfg.nepoch-1} | LR: {current_lr:.2e} | Total Steps: {trainer.total_steps}\n")
        log.write(f"Start Time: {epoch_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("=" * 70 + "\n")

        for data in tqdm(data_loader, dynamic_ncols=True, desc=f"Epoch {epoch}"):
            trainer.total_steps += 1
            epoch_iter += cfg.batch_size
            num_batches += 1

            trainer.set_input(data)
            trainer.optimize_parameters()

            # 累计loss用于计算epoch平均
            epoch_loss_sum += trainer.loss.item()
            train_writer.add_scalar("loss", trainer.loss, trainer.total_steps)

        # 计算epoch统计信息
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0

        # 记录训练集指标到tensorboard
        train_writer.add_scalar("epoch_loss", avg_loss, epoch)
        train_writer.add_scalar("learning_rate", current_lr, epoch)

        # 输出训练集统计信息
        log.write(f"[Train] Epoch {epoch} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s | Samples: {epoch_iter}\n")

        # 每个epoch结束后保存latest模型，确保可以随时断点恢复
        trainer.save_networks("latest", current_epoch=epoch)

        # 按频率保存epoch checkpoint（用于保留历史版本）
        if epoch % cfg.save_epoch_freq == 0:
            log.write(f"[Save] Checkpoint saved: model_epoch_{epoch}.pth\n")
            trainer.save_networks(epoch, current_epoch=epoch)

        # Validation
        val_start_time = time.time()
        trainer.eval()
        if cfg.video_level:
            val_results = validate_video(trainer.model, val_cfg)
        elif cfg.dual_branch:
            val_results = validate_dual_branch(trainer.model, val_cfg)
        else:
            val_results = validate(trainer.model, val_cfg)
        val_time = time.time() - val_start_time

        # 记录验证集指标到tensorboard
        val_writer.add_scalar("AP", val_results["AP"], epoch)
        val_writer.add_scalar("ACC", val_results["ACC"], epoch)
        val_writer.add_scalar("AUC", val_results["AUC"], epoch)
        val_writer.add_scalar("TPR", val_results["TPR"], epoch)
        val_writer.add_scalar("TNR", val_results["TNR"], epoch)
        val_writer.add_scalar("R_ACC", val_results["R_ACC"], epoch)
        val_writer.add_scalar("F_ACC", val_results["F_ACC"], epoch)

        # 输出验证集指标（格式化，便于论文使用）
        log.write(f"[Val]   Epoch {epoch} | "
                  f"AP: {val_results['AP']:.4f} | AUC: {val_results['AUC']:.4f} | ACC: {val_results['ACC']:.4f} | "
                  f"Time: {val_time:.1f}s\n")
        log.write(f"        TPR(Recall): {val_results['TPR']:.4f} | TNR(Specificity): {val_results['TNR']:.4f} | "
                  f"R_ACC(Real): {val_results['R_ACC']:.4f} | F_ACC(Fake): {val_results['F_ACC']:.4f}\n")

        # Epoch结束时间统计
        epoch_end_datetime = datetime.now()
        epoch_total_time = time.time() - epoch_start_time
        log.write(f"[Time]  End: {epoch_end_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Epoch Duration: {epoch_total_time:.1f}s (Train: {epoch_time:.1f}s + Val: {val_time:.1f}s)\n")

        # Early Stopping 检查
        lr_decayed_this_epoch = False  # 标记本epoch是否触发了LR衰减
        if cfg.earlystop:
            early_stopping(val_results["AUC"], trainer, epoch)

            # 输出early stopping状态
            best_auc = early_stopping.score_max if early_stopping.score_max != -float('inf') else 0
            log.write(f"[EarlyStop] Best AUC: {best_auc:.4f} | "
                      f"Patience: {early_stopping.counter}/{cfg.earlystop_epoch}\n")

            # Save early_stopping state
            earlystop_state = {
                "best_score": early_stopping.best_score,
                "counter": early_stopping.counter,
                "score_max": early_stopping.score_max,
            }
            torch.save(earlystop_state, os.path.join(cfg.ckpt_dir, "earlystop_state.pth"))

            if early_stopping.early_stop:
                if trainer.adjust_learning_rate():
                    lr_decayed_this_epoch = True  # 标记已触发LR衰减
                    new_lr = trainer.optimizer.param_groups[0]['lr']
                    log.write(f"[LR Decay] Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}\n")
                    # Preserve best score when creating new early stopping instance
                    old_best_score = early_stopping.best_score
                    old_score_max = early_stopping.score_max
                    early_stopping = EarlyStopping(patience=cfg.earlystop_epoch, delta=0.001, verbose=True)
                    early_stopping.best_score = old_best_score
                    early_stopping.score_max = old_score_max
                else:
                    log.write("\n" + "=" * 70 + "\n")
                    log.write(f"[Early Stop] Training finished at epoch {epoch}\n")
                    log.write(f"[Final] Best AUC: {best_auc:.4f}\n")
                    log.write("=" * 70 + "\n")
                    break

        # 只有当本epoch没有触发LR衰减时，才执行scheduler.step()
        # 因为手动LR衰减和scheduler.step()会冲突（scheduler会覆盖手动调整的LR）
        if cfg.warmup and not lr_decayed_this_epoch:
            trainer.scheduler.step()
        trainer.train()

    # 训练结束总结
    log.write("\n" + "=" * 70 + "\n")
    log.write("Training Completed!\n")
    log.write(f"Total epochs: {epoch + 1} | Total steps: {trainer.total_steps}\n")
    if cfg.earlystop:
        best_auc = early_stopping.score_max if early_stopping.score_max != -float('inf') else 0
        log.write(f"Best Validation AUC: {best_auc:.4f}\n")
    log.write("=" * 70 + "\n")
