import argparse
import os
import glob
from datetime import datetime
import numpy as np
import random
import time
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

from core.utils1.config import DefaultConfigs
from core.utils1.trainer import VideoTrainer, DualBranchTrainer, Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_system_mem_percent():
    try:
        import psutil
        return psutil.virtual_memory().percent
    except Exception:
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.dwMemoryLoad)
        except Exception:
            return None


def _wait_for_memory(max_pct, check_interval):
    if max_pct is None or max_pct <= 0:
        return
    while True:
        mem_pct = _get_system_mem_percent()
        if mem_pct is None or mem_pct < max_pct:
            return
        time.sleep(check_interval)


def load_model(model_path, cfg):
    """Load model based on configuration"""
    # Determine model type from config
    if cfg.video_level:
        trainer = VideoTrainer(cfg)
        print(f"Loading Video-level model from {model_path}")
    elif cfg.dual_branch:
        trainer = DualBranchTrainer(cfg)
        print(f"Loading Dual-branch model from {model_path}")
    else:
        trainer = Trainer(cfg)
        print(f"Loading Single-branch model from {model_path}")

    # Load checkpoint
    state_dict = torch.load(model_path, map_location=trainer.device)
    if "model" in state_dict:
        trainer.model.load_state_dict(state_dict["model"])
    else:
        trainer.model.load_state_dict(state_dict)

    trainer.model.eval()
    return trainer.model, trainer.device


def load_frames(frame_folder, num_frames=16, transform=None, start_idx=None, is_train=False):
    """
    Load frames from folder.
    与训练代码 VideoDataset._sample_paired_indices 保持一致。

    Args:
        frame_folder: Path to folder containing frames
        num_frames: Number of frames to load
        transform: Transform to apply to each frame
        start_idx: Starting index for consecutive frame sampling.
                   If None, use center sampling for test (matching training code).
        is_train: If True, use random sampling; if False, use center sampling.

    Returns:
        Tuple of (frames_tensor, start_idx_used)
    """
    frame_files = sorted(glob.glob(os.path.join(frame_folder, "*.jpg")) +
                        glob.glob(os.path.join(frame_folder, "*.png")))

    if len(frame_files) == 0:
        return None, None

    # Sample consecutive frames (与训练代码一致)
    if len(frame_files) >= num_frames:
        max_start = len(frame_files) - num_frames
        if start_idx is None:
            if is_train:
                # 训练模式：随机采样
                start_idx = np.random.randint(0, max_start + 1)
            else:
                # 测试模式：居中采样（与训练代码一致）
                start_idx = max_start // 2

        # Get consecutive frames
        frame_files = frame_files[start_idx:start_idx + num_frames]
    else:
        # Use all frames if not enough
        start_idx = 0

    frames = []
    for frame_path in frame_files:
        with Image.open(frame_path) as img:
            img = img.convert("RGB")
            if transform:
                img = transform(img)
            frames.append(img)

    # Pad if not enough frames (用最后一帧填充，与训练代码一致)
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return torch.stack(frames[:num_frames]), start_idx


def _find_image_dirs(root):
    """Find deepest directories that contain image files under root."""
    if not os.path.isdir(root):
        return []

    image_dirs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in filenames):
            image_dirs.append(dirpath)

    if not image_dirs:
        return []

    image_dirs_set = set(image_dirs)
    deepest = []
    for d in image_dirs:
        if not any(other != d and other.startswith(d + os.sep) for other in image_dirs_set):
            deepest.append(d)

    return sorted(deepest)


def _match_rgb_flow_dirs(rgb_root, flow_root):
    """Match RGB/flow directories by relative path."""
    if not os.path.isdir(rgb_root) or not os.path.isdir(flow_root):
        return []

    rgb_dirs = _find_image_dirs(rgb_root)
    flow_dirs = _find_image_dirs(flow_root)
    flow_map = {os.path.relpath(d, flow_root): d for d in flow_dirs}

    pairs = []
    for rgb_dir in rgb_dirs:
        rel = os.path.relpath(rgb_dir, rgb_root)
        flow_dir = flow_map.get(rel)
        if flow_dir:
            pairs.append((rgb_dir, flow_dir, rel))

    return pairs


def _collect_pairs_with_rel(split_path):
    """Collect matched (rgb_folder, flow_folder, rel) pairs from a split path."""
    if not os.path.isdir(split_path):
        return []

    pairs = []
    video_folders = [d for d in os.listdir(split_path)
                     if os.path.isdir(os.path.join(split_path, d))
                     and d not in ['rgb', 'flow']]

    if video_folders:
        for video_name in video_folders:
            rgb_root = os.path.join(split_path, video_name, "rgb")
            flow_root = os.path.join(split_path, video_name, "flow")
            for rgb_dir, flow_dir, rel in _match_rgb_flow_dirs(rgb_root, flow_root):
                # include video_name in rel for grouping
                pairs.append((rgb_dir, flow_dir, os.path.join(video_name, rel)))
    else:
        rgb_root = os.path.join(split_path, "rgb")
        flow_root = os.path.join(split_path, "flow")
        for rgb_dir, flow_dir, rel in _match_rgb_flow_dirs(rgb_root, flow_root):
            pairs.append((rgb_dir, flow_dir, rel))

    return pairs


def _collect_pairs_with_rel_from_roots(rgb_root, flow_root):
    """Collect matched (rgb_folder, flow_folder, rel) pairs from separate rgb/flow roots."""
    if not os.path.isdir(rgb_root) or not os.path.isdir(flow_root):
        return []
    return _match_rgb_flow_dirs(rgb_root, flow_root)


def _collect_pairs(split_path):
    """Collect matched (rgb_folder, flow_folder) pairs from a split path."""
    return [(rgb_dir, flow_dir) for rgb_dir, flow_dir, _rel in _collect_pairs_with_rel(split_path)]


def _group_key_from_rel(gen_name, rel):
    if not rel:
        return gen_name
    parent = os.path.basename(os.path.dirname(rel))
    if parent:
        return os.path.join(gen_name, parent)
    return gen_name


def _build_transforms(cfg):
    resize_size = 256
    crop_size = 224
    identity = transforms.Lambda(lambda x: x)
    crop = transforms.CenterCrop((crop_size, crop_size)) if cfg.aug_crop else identity
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]) if cfg.aug_norm else identity
    transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        crop,
        transforms.ToTensor(),
        norm
    ])
    return transform, transform

def _prepare_video_tensors(rgb_folder, flow_folder, cfg, transform_rgb, transform_flow):
    rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.jpg")) +
                       glob.glob(os.path.join(rgb_folder, "*.png")))
    flow_files = sorted(glob.glob(os.path.join(flow_folder, "*.jpg")) +
                        glob.glob(os.path.join(flow_folder, "*.png")))

    if len(rgb_files) == 0 or len(flow_files) == 0:
        return None, None

    available_frames = min(len(rgb_files), len(flow_files))
    num_frames = cfg.num_frames

    if available_frames >= num_frames:
        max_start = available_frames - num_frames
        start_idx = max_start // 2
    else:
        start_idx = 0

    rgb_frames, _ = load_frames(rgb_folder, num_frames, transform_rgb, start_idx=start_idx, is_train=False)
    if rgb_frames is None:
        return None, None

    flow_frames, _ = load_frames(flow_folder, num_frames, transform_flow, start_idx=start_idx, is_train=False)
    if flow_frames is None:
        return None, None

    return rgb_frames, flow_frames


def _infer_one(label, rgb_folder, flow_folder, cfg, model, device, use_video_level, group_key=None):
    if use_video_level:
        prob = test_video_level(model, device, None, cfg, rgb_folder=rgb_folder, flow_folder=flow_folder)
    else:
        prob = test_dual_branch(model, device, None, cfg, rgb_folder=rgb_folder, flow_folder=flow_folder)
    return label, prob, group_key


def _run_inference_video_level_batched(tasks, cfg, model, device, num_workers, batch_size, max_cpu_mem_pct, mem_check_interval, desc):
    y_true = []
    y_scores = []
    group_keys = []
    batch = []
    transform_rgb, transform_flow = _build_transforms(cfg)

    def load_task(task):
        label, rgb_folder, flow_folder, group_key = task
        rgb_frames, flow_frames = _prepare_video_tensors(rgb_folder, flow_folder, cfg, transform_rgb, transform_flow)
        if rgb_frames is None or flow_frames is None:
            return None
        return label, rgb_frames, flow_frames, group_key

    def flush_batch():
        if not batch:
            return
        labels = [b[0] for b in batch]
        rgb_batch = torch.stack([b[1] for b in batch]).to(device, non_blocking=True)
        flow_batch = torch.stack([b[2] for b in batch]).to(device, non_blocking=True)
        batch_group_keys = [b[3] for b in batch]
        with torch.no_grad():
            output = model(rgb_batch, flow_batch)
            probs = torch.sigmoid(output).view(-1).cpu().tolist()
        y_true.extend(labels)
        y_scores.extend(probs)
        group_keys.extend(batch_group_keys)
        batch.clear()

    if num_workers <= 1:
        for task in tqdm(tasks, desc=desc):
            _wait_for_memory(max_cpu_mem_pct, mem_check_interval)
            res = load_task(task)
            if res is None:
                continue
            batch.append(res)
            if len(batch) >= batch_size:
                flush_batch()
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            max_pending = max(1, num_workers * 4)
            pending = set()
            task_iter = iter(tasks)

            def submit_next():
                try:
                    task = next(task_iter)
                except StopIteration:
                    return False
                _wait_for_memory(max_cpu_mem_pct, mem_check_interval)
                pending.add(executor.submit(load_task, task))
                return True

            for _ in range(min(max_pending, len(tasks))):
                if not submit_next():
                    break

            with tqdm(total=len(tasks), desc=desc) as pbar:
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        res = fut.result()
                        if res is not None:
                            batch.append(res)
                            if len(batch) >= batch_size:
                                flush_batch()
                        pbar.update(1)
                        submit_next()

    flush_batch()
    return y_true, y_scores, group_keys


def _run_inference(tasks, cfg, model, device, use_video_level, num_workers, batch_size, max_cpu_mem_pct, mem_check_interval, desc):
    y_true = []
    y_scores = []
    group_keys = []

    if use_video_level and batch_size > 1:
        return _run_inference_video_level_batched(
            tasks, cfg=cfg, model=model, device=device,
            num_workers=num_workers, batch_size=batch_size,
            max_cpu_mem_pct=max_cpu_mem_pct, mem_check_interval=mem_check_interval,
            desc=desc
        )

    if num_workers <= 1:
        for label, rgb_folder, flow_folder, group_key in tqdm(tasks, desc=desc):
            _wait_for_memory(max_cpu_mem_pct, mem_check_interval)
            label, prob, group_key = _infer_one(label, rgb_folder, flow_folder, cfg, model, device, use_video_level, group_key)
            if prob is not None:
                y_true.append(label)
                y_scores.append(prob)
                group_keys.append(group_key)
        return y_true, y_scores, group_keys

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        max_pending = max(1, num_workers * 4)
        pending = set()
        task_iter = iter(tasks)

        def submit_next():
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            _wait_for_memory(max_cpu_mem_pct, mem_check_interval)
            label, rgb_folder, flow_folder, group_key = task
            pending.add(executor.submit(_infer_one, label, rgb_folder, flow_folder, cfg, model, device, use_video_level, group_key))
            return True

        for _ in range(min(max_pending, len(tasks))):
            if not submit_next():
                break

        with tqdm(total=len(tasks), desc=desc) as pbar:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    label, prob, group_key = fut.result()
                    if prob is not None:
                        y_true.append(label)
                        y_scores.append(prob)
                        group_keys.append(group_key)
                    pbar.update(1)
                    submit_next()

    return y_true, y_scores, group_keys


def test_video_level(model, device, video_folder, cfg, rgb_folder=None, flow_folder=None):
    """
    Test video-level model on a single video.
    Aligns with training: Resize(256) -> CenterCrop(224) if aug_crop, Normalize if aug_norm.
    """
    if rgb_folder is None or flow_folder is None:
        rgb_folder = os.path.join(video_folder, "rgb")
        flow_folder = os.path.join(video_folder, "flow")

    transform_rgb, transform_flow = _build_transforms(cfg)

    rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.jpg")) +
                       glob.glob(os.path.join(rgb_folder, "*.png")))
    flow_files = sorted(glob.glob(os.path.join(flow_folder, "*.jpg")) +
                        glob.glob(os.path.join(flow_folder, "*.png")))

    if len(rgb_files) == 0 or len(flow_files) == 0:
        return None

    available_frames = min(len(rgb_files), len(flow_files))
    num_frames = cfg.num_frames

    if available_frames >= num_frames:
        max_start = available_frames - num_frames
        start_idx = max_start // 2
    else:
        start_idx = 0

    rgb_frames, _ = load_frames(rgb_folder, num_frames, transform_rgb, start_idx=start_idx, is_train=False)
    if rgb_frames is None:
        return None

    flow_frames, _ = load_frames(flow_folder, num_frames, transform_flow, start_idx=start_idx, is_train=False)
    if flow_frames is None:
        return None

    rgb_frames = rgb_frames.unsqueeze(0).to(device)
    flow_frames = flow_frames.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(rgb_frames, flow_frames)
        prob = torch.sigmoid(output).item()

    return prob


def test_dual_branch(model, device, video_folder, cfg, rgb_folder=None, flow_folder=None):
    """
    Test dual-branch model on a single video.
    图像处理与训练代码 DualBranchDataset 保持一致：Resize(256) -> CenterCrop(224)
    帧采样与训练代码保持一致：测试时居中采样连续帧
    """
    if rgb_folder is None or flow_folder is None:
        rgb_folder = os.path.join(video_folder, "rgb")
        flow_folder = os.path.join(video_folder, "flow")

    # 与训练代码 DualBranchDataset 保持一致：统一使用 256->224
    transform_rgb, transform_flow = _build_transforms(cfg)

    # 获取帧文件列表
    rgb_files = sorted(glob.glob(os.path.join(rgb_folder, "*.jpg")) +
                       glob.glob(os.path.join(rgb_folder, "*.png")))
    flow_files = sorted(glob.glob(os.path.join(flow_folder, "*.jpg")) +
                        glob.glob(os.path.join(flow_folder, "*.png")))

    if len(rgb_files) == 0 or len(flow_files) == 0:
        return None

    # 取RGB和Optical的最小帧数，确保1:1对应（与训练代码一致）
    available_frames = min(len(rgb_files), len(flow_files))
    num_frames = getattr(cfg, 'num_frames', 16)

    # 计算居中采样的起始位置（测试模式，与训练代码一致）
    if available_frames >= num_frames:
        max_start = available_frames - num_frames
        start_idx = max_start // 2  # 居中采样
        selected_rgb = rgb_files[start_idx:start_idx + num_frames]
        selected_flow = flow_files[start_idx:start_idx + num_frames]
    else:
        # 帧数不足，使用所有可用帧
        selected_rgb = rgb_files[:available_frames]
        selected_flow = flow_files[:available_frames]

    probs = []
    for rgb_path, flow_path in zip(selected_rgb, selected_flow):
        with Image.open(rgb_path) as rgb_img:
            rgb_img = rgb_img.convert("RGB")
            rgb_tensor = transform_rgb(rgb_img).unsqueeze(0).to(device)

        with Image.open(flow_path) as flow_img:
            flow_img = flow_img.convert("RGB")
            flow_tensor = transform_flow(flow_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(rgb_tensor, flow_tensor)
            prob = torch.sigmoid(output).item()
            probs.append(prob)

    return np.mean(probs) if probs else None


def calculate_metrics(y_true, y_pred, y_scores):
    """Calculate comprehensive metrics for paper writing"""
    metrics = {}

    # Basic metrics
    metrics['ACC'] = accuracy_score(y_true, y_pred)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    if len(np.unique(y_true_arr)) > 1:
        metrics['AUC'] = roc_auc_score(y_true, y_scores)
        metrics['AP'] = average_precision_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    else:
        metrics['AUC'] = float('nan')
        metrics['AP'] = float('nan')
        fpr, tpr, thresholds = np.array([]), np.array([]), np.array([])

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    metrics['TPR'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
    metrics['TNR'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Precision, Recall, F1
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)

    # Real/Fake accuracy
    real_mask = y_true_arr == 0
    fake_mask = y_true_arr == 1
    metrics['R_ACC'] = accuracy_score(y_true_arr[real_mask], y_pred_arr[real_mask]) if real_mask.sum() > 0 else 0
    metrics['F_ACC'] = accuracy_score(y_true_arr[fake_mask], y_pred_arr[fake_mask]) if fake_mask.sum() > 0 else 0

    # TPR at different FPR thresholds (important for paper)
    if fpr.size > 0:
        for target_fpr in [0.01, 0.05, 0.10]:
            idx = np.where(fpr <= target_fpr)[0]
            if len(idx) > 0:
                metrics[f'TPR@FPR={target_fpr:.2f}'] = tpr[idx[-1]]
            else:
                metrics[f'TPR@FPR={target_fpr:.2f}'] = 0
    else:
        for target_fpr in [0.01, 0.05, 0.10]:
            metrics[f'TPR@FPR={target_fpr:.2f}'] = float('nan')

    return metrics, (fpr, tpr, thresholds)


def plot_roc_curve(fpr, tpr, auc, save_path):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(y_true, y_scores, ap, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'AP = {ap:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distribution(y_true, y_scores, save_path):
    """Plot score distribution for real vs fake"""
    real_scores = [score for label, score in zip(y_true, y_scores) if label == 0]
    fake_scores = [score for label, score in zip(y_true, y_scores) if label == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=50, alpha=0.6, label='Real', color='blue', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.6, label='Fake', color='red', density=True)
    plt.xlabel('Prediction Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    classes = ['Real', 'Fake']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_config_from_logs(exp_name):
    """Load training configuration from logs.txt"""
    import re
    import ast

    log_path = os.path.join("core/data/exp", exp_name, "logs.txt")
    if not os.path.exists(log_path):
        print(f"Warning: logs.txt not found at {log_path}")
        return None

    with open(log_path, 'r') as f:
        content = f.read()

    # Find the config dictionary in logs
    match = re.search(r"Config:\s*(\{.*?\})", content, re.DOTALL)
    if not match:
        print("Warning: Could not find config in logs.txt")
        return None

    try:
        config_str = match.group(1)
        # Fix common issues in config string
        config_str = config_str.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
        import json
        config_dict = json.loads(config_str)
        return config_dict
    except Exception as e:
        print(f"Warning: Could not parse config: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test deepfake detection model on video dataset')

    # Method 1: Use exp_name to auto-load config (Recommended)
    parser.add_argument("--exp_name", type=str, help="Experiment name to auto-load config from logs.txt")
    parser.add_argument("--epoch", type=str, default="best", help="Which checkpoint to test: 'best', 'latest', or epoch number")

    # Method 2: Manually specify model path and config
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint (alternative to --exp_name)")
    parser.add_argument("--test_video_path", type=str, required=True, help="Path to test video root folder")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--fake_only", type=lambda x: str(x).lower() == 'true', default=False,
                        help="If true, only test fake videos and skip real-vs-fake metrics")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Parallel workers for per-video inference (threads). Use 1 to disable parallelism.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for video-level inference. >1 can improve GPU utilization.")
    parser.add_argument("--max_cpu_mem_pct", type=float, default=90.0,
                        help="Throttle when system memory usage >= this percent. Set <=0 to disable.")
    parser.add_argument("--mem_check_interval", type=float, default=0.5,
                        help="Seconds between memory checks when throttling.")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use (e.g., --gpus 0 or --gpus 0 1). If not set, keep current CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--require_cuda", type=lambda x: str(x).lower() == 'true', default=False,
                        help="If true, exit when CUDA is not available.")

    # Model configuration (optional if using --exp_name)
    parser.add_argument("--arch_rgb", type=str, help="RGB branch architecture")
    parser.add_argument("--arch_optical", type=str, help="Optical branch architecture")
    parser.add_argument("--num_frames", type=int, help="Number of frames for video-level model")
    parser.add_argument("--dual_branch", type=lambda x: str(x).lower() == 'true', help="Use dual-branch model")
    parser.add_argument("--video_level", type=lambda x: str(x).lower() == 'true', help="Use video-level model")
    parser.add_argument("--fusion_type", type=str, help="Fusion type")
    parser.add_argument("--feature_dim", type=int, help="Feature dimension")
    parser.add_argument("--num_heads", type=int, help="Transformer heads")
    parser.add_argument("--num_layers", type=int, help="Transformer layers")
    parser.add_argument("--early_fusion", type=lambda x: str(x).lower() == 'true', help="Early fusion mode")
    parser.add_argument("--fused_dim", type=int, help="Fused dimension")

    args = parser.parse_args()

    # Load config from logs.txt if exp_name is provided
    if args.exp_name:
        print(f"Loading configuration from experiment: {args.exp_name}")
        loaded_config = load_config_from_logs(args.exp_name)

        if loaded_config:
            print("Successfully loaded training configuration!")
            # Set model path
            if not args.model_path:
                args.model_path = os.path.join("core/data/exp", args.exp_name, "ckpt", f"model_epoch_{args.epoch}.pth")

            # Auto-fill missing arguments from loaded config
            if args.arch_rgb is None:
                args.arch_rgb = loaded_config.get('arch_rgb', 'freqnet')
            if args.arch_optical is None:
                args.arch_optical = loaded_config.get('arch_optical', 'resnet50')
            if args.dual_branch is None:
                args.dual_branch = loaded_config.get('dual_branch', True)
            if args.video_level is None:
                args.video_level = loaded_config.get('video_level', True)
            if args.num_frames is None:
                args.num_frames = loaded_config.get('num_frames', 16)
            if args.fusion_type is None:
                args.fusion_type = loaded_config.get('fusion_type', 'gated')
            if args.feature_dim is None:
                args.feature_dim = loaded_config.get('feature_dim', 256)
            if args.num_heads is None:
                args.num_heads = loaded_config.get('num_heads', 8)
            if args.num_layers is None:
                args.num_layers = loaded_config.get('num_layers', 2)
            if args.early_fusion is None:
                args.early_fusion = loaded_config.get('early_fusion', False)
            if args.fused_dim is None:
                args.fused_dim = loaded_config.get('fused_dim', 512)
        else:
            print("Warning: Could not load config from logs.txt, using defaults or command-line arguments")

    # Validate required arguments
    if not args.model_path:
        parser.error("Either --exp_name or --model_path must be provided")

    # Set defaults if still None
    if args.arch_rgb is None:
        args.arch_rgb = "freqnet"
    if args.arch_optical is None:
        args.arch_optical = "resnet50"
    if args.dual_branch is None:
        args.dual_branch = True
    if args.video_level is None:
        args.video_level = True
    if args.num_frames is None:
        args.num_frames = 16
    if args.fusion_type is None:
        args.fusion_type = "gated"
    if args.feature_dim is None:
        args.feature_dim = 256
    if args.num_heads is None:
        args.num_heads = 8
    if args.num_layers is None:
        args.num_layers = 2
    if args.early_fusion is None:
        args.early_fusion = False
    if args.fused_dim is None:
        args.fused_dim = 512

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(g) for g in args.gpus])

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Please install a CUDA-enabled PyTorch build or disable --require_cuda.")

    # Create config
    cfg = DefaultConfigs()
    cfg.isTrain = False
    cfg.arch_rgb = args.arch_rgb
    cfg.arch_optical = args.arch_optical
    cfg.dual_branch = args.dual_branch
    cfg.video_level = args.video_level
    cfg.num_frames = args.num_frames
    cfg.fusion_type = args.fusion_type
    cfg.feature_dim = args.feature_dim
    cfg.num_heads = args.num_heads
    cfg.num_layers = args.num_layers
    cfg.early_fusion = args.early_fusion
    cfg.fused_dim = args.fused_dim
    set_seed(cfg.seed)

    print("="*80)
    print("Test Configuration:")
    print(f"  Model: {args.model_path}")
    print(f"  Test Data: {args.test_video_path}")
    print(f"  Mode: {'Video-level' if args.video_level else 'Dual-branch' if args.dual_branch else 'Single-branch'}")
    print(f"  RGB Arch: {args.arch_rgb}")
    print(f"  Optical Arch: {args.arch_optical}")
    if args.video_level:
        print(f"  Num Frames: {args.num_frames}")
        print(f"  Fusion: {'Early' if args.early_fusion else 'Late'}")
    print(f"  Fake Only: {args.fake_only}")
    print(f"  Num Workers: {args.num_workers}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max CPU Mem %: {args.max_cpu_mem_pct}")
    print(f"  Mem Check Interval: {args.mem_check_interval}s")
    if args.gpus is not None:
        print(f"  GPUs: {args.gpus}")
    print("="*80)

    # Load model
    model, device = load_model(args.model_path, cfg)
    print(f"Using device: {device}")

    if str(device).startswith("cuda") and args.num_workers > 1:
        print("Warning: CUDA detected. Using multiple threads may increase GPU memory usage and not speed up inference.")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(args.test_video_path, f"test_result_{timestamp}.txt")
    figure_dir = os.path.join(args.test_video_path, f"figures_{timestamp}")
    os.makedirs(figure_dir, exist_ok=True)

    # Open result file
    f = open(result_file, 'w')
    f.write(f"Deepfake Detection Test Results\n")
    f.write(f"{'='*80}\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {args.model_path}\n")
    f.write(f"Test Data: {args.test_video_path}\n")
    f.write(f"Classification Threshold: {args.threshold}\n")
    f.write(f"Architecture: RGB={args.arch_rgb}, Optical={args.arch_optical}\n")
    f.write(f"Mode: {'Video-level' if args.video_level else 'Dual-branch' if args.dual_branch else 'Single-branch'}\n")
    f.write(f"Fake Only: {args.fake_only}\n")
    f.write(f"Num Workers: {args.num_workers}\n")
    f.write(f"Batch Size: {args.batch_size}\n")
    f.write(f"Max CPU Mem %: {args.max_cpu_mem_pct}\n")
    f.write(f"Mem Check Interval: {args.mem_check_interval}s\n")
    if args.gpus is not None:
        f.write(f"GPUs: {args.gpus}\n")
    f.write(f"{'='*80}\n\n")

    # Test each generator
    generator_folders = [d for d in os.listdir(args.test_video_path)
                        if os.path.isdir(os.path.join(args.test_video_path, d))
                        and not d.startswith('.') and not d.startswith('figures_')
                        and not d.startswith('test_result_')]

    root_layout_type = None  # None | "split_inside" | "separate_roots"
    root_generators = set()

    # Layout A: test_path/rgb/1_fake/... and test_path/flow/1_fake/...
    root_rgb = os.path.join(args.test_video_path, "rgb")
    root_flow = os.path.join(args.test_video_path, "flow")
    if os.path.isdir(root_rgb) and os.path.isdir(root_flow):
        if os.path.isdir(os.path.join(root_rgb, "1_fake")) or os.path.isdir(os.path.join(root_rgb, "0_real")):
            root_layout_type = "separate_roots"
            for split in ["0_real", "1_fake"]:
                split_dir = os.path.join(root_rgb, split)
                if os.path.isdir(split_dir):
                    root_generators.update([
                        d for d in os.listdir(split_dir)
                        if os.path.isdir(os.path.join(split_dir, d))
                    ])

    # Layout B: test_path/1_fake/rgb/... and test_path/1_fake/flow/...
    if root_layout_type is None:
        root_fake_path = os.path.join(args.test_video_path, "1_fake")
        root_real_path = os.path.join(args.test_video_path, "0_real")
        if os.path.isdir(os.path.join(root_fake_path, "rgb")) and os.path.isdir(os.path.join(root_fake_path, "flow")):
            root_layout_type = "split_inside"
            root_generators.update([
                d for d in os.listdir(os.path.join(root_fake_path, "rgb"))
                if os.path.isdir(os.path.join(root_fake_path, "rgb", d))
            ])
        if os.path.isdir(os.path.join(root_real_path, "rgb")) and os.path.isdir(os.path.join(root_real_path, "flow")):
            root_layout_type = "split_inside"
            root_generators.update([
                d for d in os.listdir(os.path.join(root_real_path, "rgb"))
                if os.path.isdir(os.path.join(root_real_path, "rgb", d))
            ])

    if root_layout_type is not None and root_generators:
        generator_folders = sorted(root_generators)

    all_results = {}

    for gen_name in generator_folders:
        print(f"\n{'='*80}")
        print(f"Testing Generator: {gen_name}")
        print(f"{'='*80}")

        gen_path = args.test_video_path if root_layout_type is not None else os.path.join(args.test_video_path, gen_name)

        tasks = []
        if not args.fake_only:
            if root_layout_type == "separate_roots":
                real_rgb_root = os.path.join(root_rgb, "0_real")
                real_flow_root = os.path.join(root_flow, "0_real")
                for rgb_folder, flow_folder, rel in _collect_pairs_with_rel_from_roots(real_rgb_root, real_flow_root):
                    group = rel.split(os.sep)[0] if rel else ""
                    if group == gen_name:
                        tasks.append((0, rgb_folder, flow_folder, _group_key_from_rel(gen_name, rel)))
            else:
                real_path = os.path.join(gen_path, "0_real")
                for rgb_folder, flow_folder, rel in _collect_pairs_with_rel(real_path):
                    if root_layout_type == "split_inside":
                        group = rel.split(os.sep)[0] if rel else ""
                        if group != gen_name:
                            continue
                    tasks.append((0, rgb_folder, flow_folder, _group_key_from_rel(gen_name, rel)))

        if root_layout_type == "separate_roots":
            fake_rgb_root = os.path.join(root_rgb, "1_fake")
            fake_flow_root = os.path.join(root_flow, "1_fake")
            for rgb_folder, flow_folder, rel in _collect_pairs_with_rel_from_roots(fake_rgb_root, fake_flow_root):
                group = rel.split(os.sep)[0] if rel else ""
                if group == gen_name:
                    tasks.append((1, rgb_folder, flow_folder, _group_key_from_rel(gen_name, rel)))
        else:
            fake_path = os.path.join(gen_path, "1_fake")
            for rgb_folder, flow_folder, rel in _collect_pairs_with_rel(fake_path):
                if root_layout_type == "split_inside":
                    group = rel.split(os.sep)[0] if rel else ""
                    if group != gen_name:
                        continue
                tasks.append((1, rgb_folder, flow_folder, _group_key_from_rel(gen_name, rel)))

        if len(tasks) == 0:
            print(f"Warning: No valid videos found for {gen_name}, skipping...")
            continue

        y_true, y_scores, group_keys = _run_inference(
            tasks,
            cfg=cfg,
            model=model,
            device=device,
            use_video_level=cfg.video_level,
            num_workers=max(1, int(args.num_workers)),
            batch_size=max(1, int(args.batch_size)),
            max_cpu_mem_pct=args.max_cpu_mem_pct,
            mem_check_interval=args.mem_check_interval,
            desc=f"{gen_name} - Inference"
        )

        if len(y_true) == 0:
            print(f"Warning: No valid videos found for {gen_name}, skipping...")
            continue
        # Group by model (parent of video folder)
        group_map = {}
        for label, score, gkey in zip(y_true, y_scores, group_keys):
            if not gkey:
                gkey = gen_name
            if gkey not in group_map:
                group_map[gkey] = {"y_true": [], "y_scores": []}
            group_map[gkey]["y_true"].append(label)
            group_map[gkey]["y_scores"].append(score)

        for group_name, data in group_map.items():
            g_true = data["y_true"]
            g_scores = data["y_scores"]
            g_pred = [1 if score >= args.threshold else 0 for score in g_scores]

            if args.fake_only:
                avg_score = float(np.mean(g_scores)) if g_scores else 0.0
                fake_pass_rate = float(np.mean([1 if s >= args.threshold else 0 for s in g_scores])) if g_scores else 0.0
                acc = accuracy_score(g_true, g_pred) if g_true else 0.0
                if len(set(g_true)) > 1:
                    auc = roc_auc_score(g_true, g_scores)
                else:
                    auc = float('nan')

                metrics = {
                    'AVG_SCORE': avg_score,
                    'FAKE_PASS_RATE': fake_pass_rate,
                    'ACC': acc,
                    'AUC': auc,
                    'COUNT': len(g_scores)
                }
                all_results[group_name] = metrics

                f.write(f"\nGenerator: {group_name}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Total Videos: {len(g_scores)} (Fake Only)\n\n")
                f.write("Fake-only Metrics:\n")
                f.write(f"  AVG_SCORE (Mean Probability)   : {avg_score:.4f}\n")
                f.write(f"  FAKE_PASS_RATE (>= threshold)  : {fake_pass_rate:.4f}\n")
                f.write(f"  ACC (>= threshold)             : {acc:.4f}\n")
                f.write(f"  AUC (single-class -> NaN)      : {auc:.4f}\n")
                f.write("\n")

                print(f"\nResults for {group_name} (Fake-only):")
                print(f"  AVG_SCORE: {avg_score:.4f} | FAKE_PASS_RATE: {fake_pass_rate:.4f} | ACC: {acc:.4f} | AUC: {auc:.4f}")

                plot_score_distribution(g_true, g_scores,
                                       os.path.join(figure_dir, f"{group_name.replace(os.sep, '_')}_distribution.png"))
            else:
                metrics, (fpr, tpr, thresholds) = calculate_metrics(g_true, g_pred, g_scores)
                all_results[group_name] = metrics

                f.write(f"\nGenerator: {group_name}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Total Videos: {len(g_true)} (Real: {g_true.count(0)}, Fake: {g_true.count(1)})\n\n")
                f.write("Performance Metrics:\n")
                f.write(f"  AUC (Area Under ROC Curve)     : {metrics['AUC']:.4f}\n")
                f.write(f"  AP (Average Precision)         : {metrics['AP']:.4f}\n")
                f.write(f"  ACC (Accuracy)                 : {metrics['ACC']:.4f}\n")
                f.write(f"  TPR (True Positive Rate)       : {metrics['TPR']:.4f}\n")
                f.write(f"  TNR (True Negative Rate)       : {metrics['TNR']:.4f}\n")
                f.write(f"  FPR (False Positive Rate)      : {metrics['FPR']:.4f}\n")
                f.write(f"  FNR (False Negative Rate)      : {metrics['FNR']:.4f}\n")
                f.write(f"  Precision                      : {metrics['Precision']:.4f}\n")
                f.write(f"  Recall                         : {metrics['Recall']:.4f}\n")
                f.write(f"  F1 Score                       : {metrics['F1']:.4f}\n")
                f.write(f"  R_ACC (Real Accuracy)          : {metrics['R_ACC']:.4f}\n")
                f.write(f"  F_ACC (Fake Accuracy)          : {metrics['F_ACC']:.4f}\n")
                f.write(f"  TPR @ FPR=0.01                 : {metrics['TPR@FPR=0.01']:.4f}\n")
                f.write(f"  TPR @ FPR=0.05                 : {metrics['TPR@FPR=0.05']:.4f}\n")
                f.write(f"  TPR @ FPR=0.10                 : {metrics['TPR@FPR=0.10']:.4f}\n")
                f.write("\n")

                print(f"\nResults for {group_name}:")
                print(f"  AUC: {metrics['AUC']:.4f} | AP: {metrics['AP']:.4f} | ACC: {metrics['ACC']:.4f}")
                print(f"  TPR: {metrics['TPR']:.4f} | TNR: {metrics['TNR']:.4f} | F1: {metrics['F1']:.4f}")

                if fpr.size > 0 and not np.isnan(metrics['AUC']):
                    plot_roc_curve(fpr, tpr, metrics['AUC'],
                                  os.path.join(figure_dir, f"{group_name.replace(os.sep, '_')}_roc.png"))
                if not np.isnan(metrics['AP']):
                    plot_pr_curve(g_true, g_scores, metrics['AP'],
                                 os.path.join(figure_dir, f"{group_name.replace(os.sep, '_')}_pr.png"))
                plot_score_distribution(g_true, g_scores,
                                       os.path.join(figure_dir, f"{group_name.replace(os.sep, '_')}_distribution.png"))
                plot_confusion_matrix(g_true, g_pred,
                                    os.path.join(figure_dir, f"{group_name.replace(os.sep, '_')}_confusion_matrix.png"))

    # Calculate and write average metrics
    if all_results:
        f.write(f"\n{'='*80}\n")
        f.write("AVERAGE METRICS ACROSS ALL GENERATORS\n")
        f.write(f"{'-'*80}\n")

        avg_metrics = {}
        for metric_name in all_results[list(all_results.keys())[0]].keys():
            values = [results[metric_name] for results in all_results.values()]
            if any(isinstance(v, float) and np.isnan(v) for v in values):
                avg_metrics[metric_name] = float(np.nanmean(values))
            else:
                avg_metrics[metric_name] = float(np.mean(values))

        if args.fake_only:
            f.write(f"  AVG_SCORE (Mean Probability)   : {avg_metrics['AVG_SCORE']:.4f}\n")
            f.write(f"  FAKE_PASS_RATE (>= threshold)  : {avg_metrics['FAKE_PASS_RATE']:.4f}\n")
            f.write(f"  ACC (>= threshold)             : {avg_metrics['ACC']:.4f}\n")
            f.write(f"  AUC (single-class -> NaN)      : {avg_metrics['AUC']:.4f}\n")

            print(f"\n{'='*80}")
            print("Average Metrics (Fake-only):")
            print(f"  AVG_SCORE: {avg_metrics['AVG_SCORE']:.4f} | FAKE_PASS_RATE: {avg_metrics['FAKE_PASS_RATE']:.4f} | ACC: {avg_metrics['ACC']:.4f} | AUC: {avg_metrics['AUC']:.4f}")
            print(f"{'='*80}")
        else:
            f.write(f"  AUC (Area Under ROC Curve)     : {avg_metrics['AUC']:.4f}\n")
            f.write(f"  AP (Average Precision)         : {avg_metrics['AP']:.4f}\n")
            f.write(f"  ACC (Accuracy)                 : {avg_metrics['ACC']:.4f}\n")
            f.write(f"  TPR (True Positive Rate)       : {avg_metrics['TPR']:.4f}\n")
            f.write(f"  TNR (True Negative Rate)       : {avg_metrics['TNR']:.4f}\n")
            f.write(f"  FPR (False Positive Rate)      : {avg_metrics['FPR']:.4f}\n")
            f.write(f"  FNR (False Negative Rate)      : {avg_metrics['FNR']:.4f}\n")
            f.write(f"  Precision                      : {avg_metrics['Precision']:.4f}\n")
            f.write(f"  Recall                         : {avg_metrics['Recall']:.4f}\n")
            f.write(f"  F1 Score                       : {avg_metrics['F1']:.4f}\n")
            f.write(f"  R_ACC (Real Accuracy)          : {avg_metrics['R_ACC']:.4f}\n")
            f.write(f"  F_ACC (Fake Accuracy)          : {avg_metrics['F_ACC']:.4f}\n")
            f.write(f"  TPR @ FPR=0.01                 : {avg_metrics['TPR@FPR=0.01']:.4f}\n")
            f.write(f"  TPR @ FPR=0.05                 : {avg_metrics['TPR@FPR=0.05']:.4f}\n")
            f.write(f"  TPR @ FPR=0.10                 : {avg_metrics['TPR@FPR=0.10']:.4f}\n")

            print(f"\n{'='*80}")
            print("Average Metrics:")
            print(f"  AUC: {avg_metrics['AUC']:.4f} | AP: {avg_metrics['AP']:.4f} | ACC: {avg_metrics['ACC']:.4f}")
            print(f"  TPR: {avg_metrics['TPR']:.4f} | TNR: {avg_metrics['TNR']:.4f} | F1: {avg_metrics['F1']:.4f}")
            print(f"{'='*80}")

    f.close()

    print(f"\n✓ Results saved to: {result_file}")
    print(f"✓ Figures saved to: {figure_dir}/")
    print(f"\nGenerated figures for each generator:")
    print(f"  - ROC curve")
    print(f"  - Precision-Recall curve")
    print(f"  - Score distribution")
    print(f"  - Confusion matrix")


if __name__ == "__main__":
    main()
