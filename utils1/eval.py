import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from utils1.config import CONFIGCLASS
from utils1.utils import to_cuda


def get_val_cfg(cfg: CONFIGCLASS, split="val", copy=True):
    if copy:
        from copy import deepcopy

        val_cfg = deepcopy(cfg)
    else:
        val_cfg = cfg
    # Replace 'train' with the target split (e.g., 'val') in the path
    if 'train' in val_cfg.dataset_root:
        val_cfg.dataset_root = val_cfg.dataset_root.replace('train', split)
    else:
        val_cfg.dataset_root = os.path.join(val_cfg.dataset_root, split)
    # Also update optical_root for dual-branch/video-level validation
    if val_cfg.optical_root:
        if 'train' in val_cfg.optical_root:
            val_cfg.optical_root = val_cfg.optical_root.replace('train', split)
        else:
            val_cfg.optical_root = os.path.join(val_cfg.optical_root, split)
    val_cfg.datasets = cfg.datasets_test
    val_cfg.datasets_optical = cfg.datasets_optical_test if cfg.datasets_optical_test else cfg.datasets_test
    val_cfg.isTrain = False
    # val_cfg.aug_resize = False
    # val_cfg.aug_crop = False
    val_cfg.aug_flip = False
    val_cfg.serial_batches = True
    val_cfg.jpg_method = ["pil"]
    # Currently assumes jpg_prob, blur_prob 0 or 1
    if len(val_cfg.blur_sig) == 2:
        b_sig = val_cfg.blur_sig
        val_cfg.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_cfg.jpg_qual) != 1:
        j_qual = val_cfg.jpg_qual
        val_cfg.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_cfg

def validate(model: nn.Module, cfg: CONFIGCLASS):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    from utils1.datasets import create_dataloader

    data_loader = create_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            img, label, meta = data if len(data) == 3 else (*data, None)
            in_tens = to_cuda(img, device)
            meta = to_cuda(meta, device)
            predict = model(in_tens, meta).sigmoid()
            y_pred.extend(predict.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle empty validation set
    if len(y_true) == 0:
        print("Warning: Validation dataset is empty, skipping evaluation")
        return {"ACC": 0.0, "AP": 0.0, "AUC": 0.0, "TPR": 0.0, "TNR": 0.0, "R_ACC": 0.0, "F_ACC": 0.0}

    y_bin = y_pred > 0.5
    r_acc = accuracy_score(y_true[y_true == 0], y_bin[y_true == 0]) if (y_true == 0).sum() > 0 else 0.0
    f_acc = accuracy_score(y_true[y_true == 1], y_bin[y_true == 1]) if (y_true == 1).sum() > 0 else 0.0
    acc = accuracy_score(y_true, y_bin)
    ap = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    tn = ((y_true == 0) & (y_bin == 0)).sum()
    fp = ((y_true == 0) & (y_bin == 1)).sum()
    fn = ((y_true == 1) & (y_bin == 0)).sum()
    tp = ((y_true == 1) & (y_bin == 1)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    results = {
        "ACC": acc,
        "AP": ap,
        "AUC": auc,
        "TPR": tpr,
        "TNR": tnr,
        "R_ACC": r_acc,
        "F_ACC": f_acc,
    }
    return results


def validate_video(model: nn.Module, cfg: CONFIGCLASS):
    """
    Validation function for video-level model with temporal aggregation.
    """
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    from utils1.datasets import create_video_dataloader

    data_loader = create_video_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            rgb_video, optical_video, label = data
            rgb_video = to_cuda(rgb_video, device)
            optical_video = to_cuda(optical_video, device)
            predict = model(rgb_video, optical_video).sigmoid()
            y_pred.extend(predict.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle empty validation set
    if len(y_true) == 0:
        print("Warning: Validation dataset is empty, skipping evaluation")
        return {"ACC": 0.0, "AP": 0.0, "AUC": 0.0, "TPR": 0.0, "TNR": 0.0, "R_ACC": 0.0, "F_ACC": 0.0}

    y_bin = y_pred > 0.5
    r_acc = accuracy_score(y_true[y_true == 0], y_bin[y_true == 0]) if (y_true == 0).sum() > 0 else 0.0
    f_acc = accuracy_score(y_true[y_true == 1], y_bin[y_true == 1]) if (y_true == 1).sum() > 0 else 0.0
    acc = accuracy_score(y_true, y_bin)
    ap = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    tn = ((y_true == 0) & (y_bin == 0)).sum()
    fp = ((y_true == 0) & (y_bin == 1)).sum()
    fn = ((y_true == 1) & (y_bin == 0)).sum()
    tp = ((y_true == 1) & (y_bin == 1)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    results = {
        "ACC": acc,
        "AP": ap,
        "AUC": auc,
        "TPR": tpr,
        "TNR": tnr,
        "R_ACC": r_acc,
        "F_ACC": f_acc,
    }
    return results


def validate_dual_branch(model: nn.Module, cfg: CONFIGCLASS):
    """
    Validation function for dual-branch model.
    """
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    from utils1.datasets import create_dual_branch_dataloader

    data_loader = create_dual_branch_dataloader(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in data_loader:
            rgb, optical, label = data
            rgb = to_cuda(rgb, device)
            optical = to_cuda(optical, device)
            predict = model(rgb, optical).sigmoid()
            y_pred.extend(predict.flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Handle empty validation set
    if len(y_true) == 0:
        print("Warning: Validation dataset is empty, skipping evaluation")
        return {"ACC": 0.0, "AP": 0.0, "AUC": 0.0, "TPR": 0.0, "TNR": 0.0, "R_ACC": 0.0, "F_ACC": 0.0}

    y_bin = y_pred > 0.5
    r_acc = accuracy_score(y_true[y_true == 0], y_bin[y_true == 0]) if (y_true == 0).sum() > 0 else 0.0
    f_acc = accuracy_score(y_true[y_true == 1], y_bin[y_true == 1]) if (y_true == 1).sum() > 0 else 0.0
    acc = accuracy_score(y_true, y_bin)
    ap = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    tn = ((y_true == 0) & (y_bin == 0)).sum()
    fp = ((y_true == 0) & (y_bin == 1)).sum()
    fn = ((y_true == 1) & (y_bin == 0)).sum()
    tp = ((y_true == 1) & (y_bin == 1)).sum()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    results = {
        "ACC": acc,
        "AP": ap,
        "AUC": auc,
        "TPR": tpr,
        "TNR": tnr,
        "R_ACC": r_acc,
        "F_ACC": f_acc,
    }
    return results
