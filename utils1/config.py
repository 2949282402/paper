import argparse
import os
import sys
from abc import ABC
from typing import Type


class DefaultConfigs(ABC):
    ####### base setting ######
    gpus = [0]
    seed = 3407
    # arch = "resnet50"  # Original ResNet50 backbone
    # Supported architectures:
    #   - "resnet18", "resnet34", "resnet50", "resnet101", "resnet152" (with ImageNet pretrained weights)
    #   - "freqnet" (Frequency-aware network from AAAI 2024, no pretrained weights)
    # To use FreqNet for RGB branch, change to: arch = "freqnet"
    arch = "freqnet"  # Changed to FreqNet for RGB branch (used for single-branch mode)

    ####### dual-branch settings ######
    dual_branch = True  # Set to True to enable dual-branch fusion mode
    arch_rgb = "freqnet"  # Architecture for RGB branch
    arch_optical = "resnet50"  # Architecture for optical flow branch
    fusion_type = "gated"  # Fusion strategy: 'concat', 'add', 'bilinear', 'attention', 'gated'
    feature_dim = 256  # Feature dimension for each branch
    optical_root = ""  # Root directory for optical flow images (parallel to dataset_root)

    ####### video-level temporal aggregation settings ######
    video_level = True  # Set to True to enable video-level training with Transformer
    num_frames = 16  # Number of frames to sample per video
    num_heads = 8  # Number of attention heads in Transformer
    num_layers = 2  # Number of Transformer encoder layers
    early_fusion = False  # True: Frame-level fusion -> Transformer (recommended for deepfake)
                          # False: Separate Transformers -> Late fusion
    fused_dim = 512  # Fused feature dimension for early fusion

    datasets = ["zhaolian_train"]
    datasets_optical = []  # Optical flow dataset names (if different from RGB datasets)
                           # If empty, uses the same names as datasets
    datasets_test = ["adm_res_abs_ddim20s"]
    datasets_optical_test = []  # Optical flow dataset names for testing
    mode = "binary"
    class_bal = False
    batch_size = 64
    loadSize = 256
    cropSize = 224
    epoch = "latest"
    num_workers = 0  # Windows: use 0 to avoid multiprocessing issues; Linux: can use 4-20
    serial_batches = False
    isTrain = True

    # data augmentation
    rz_interp = ["bilinear"]
    # blur_prob = 0.0
    blur_prob = 0.1
    blur_sig = [0.5]
    # jpg_prob = 0.0
    jpg_prob = 0.1
    jpg_method = ["cv2"]
    jpg_qual = [75]
    gray_prob = 0.0
    aug_resize = True
    aug_crop = True
    aug_flip = True
    aug_norm = True

    ####### train setting ######
    warmup = False
    # warmup = True
    warmup_epoch = 3
    earlystop = True
    earlystop_epoch = 5
    optim = "adam"
    new_optim = False
    loss_freq = 400
    save_latest_freq = 2000
    save_epoch_freq = 20
    continue_train = False
    epoch_count = 1
    last_epoch = -1
    nepoch = 400
    beta1 = 0.9
    lr = 0.0001
    init_type = "normal"
    init_gain = 0.02
    pretrained = True

    # paths information
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_root = os.path.join(root_dir, "data")
    exp_root = os.path.join(root_dir, "data", "exp")
    _exp_name = ""
    exp_dir = ""
    ckpt_dir = ""
    logs_path = ""
    ckpt_path = ""

    @property
    def exp_name(self):
        return self._exp_name

    @exp_name.setter
    def exp_name(self, value: str):
        self._exp_name = value
        self.exp_dir: str = os.path.join(self.exp_root, self.exp_name)
        self.ckpt_dir: str = os.path.join(self.exp_dir, "ckpt")
        self.logs_path: str = os.path.join(self.exp_dir, "logs.txt")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def to_dict(self):
        dic = {}
        for fieldkey in dir(self):
            fieldvalue = getattr(self, fieldkey)
            if not fieldkey.startswith("__") and not callable(fieldvalue) and not fieldkey.startswith("_"):
                dic[fieldkey] = fieldvalue
        return dic


def args_list2dict(arg_list: list):
    assert len(arg_list) % 2 == 0, f"Override list has odd length: {arg_list}; it must be a list of pairs"
    return dict(zip(arg_list[::2], arg_list[1::2]))


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    elif v.lower() in ("true", "yes", "on", "y", "t", "1"):
        return True
    elif v.lower() in ("false", "no", "off", "n", "f", "0"):
        return False
    else:
        return bool(v)


def str2list(v: str, element_type=None) -> list:
    if not isinstance(v, (list, tuple, set)):
        v = v.lstrip("[").rstrip("]")
        v = v.split(",")
        v = list(map(str.strip, v))
        if element_type is not None:
            v = list(map(element_type, v))
    return v


def get_crop_size(arch: str) -> int:
    """
    Get the appropriate crop size based on the architecture.
    - FreqNet: 224x224 (original FreqNet design)
    - ResNet: 448x448 (original AIGVDet design)
    """
    if "freqnet" in arch.lower():
        return 224
    else:
        return 448


CONFIGCLASS = Type[DefaultConfigs]

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Basic settings
parser.add_argument("--gpus", default=[0], type=int, nargs="+", help="GPU IDs to use")
parser.add_argument("--exp_name", default="", type=str, help="Experiment name")
parser.add_argument("--ckpt", default="model_epoch_latest.pth", type=str, help="Checkpoint filename")
parser.add_argument("--seed", default=3407, type=int, help="Random seed")

# Architecture settings
parser.add_argument("--arch", default="freqnet", type=str, help="Single-branch architecture")
parser.add_argument("--arch_rgb", default="freqnet", type=str, help="RGB branch architecture")
parser.add_argument("--arch_optical", default="resnet50", type=str, help="Optical flow branch architecture")

# Mode settings
parser.add_argument("--dual_branch", default=False, type=str2bool, help="Enable dual-branch mode")
parser.add_argument("--video_level", default=False, type=str2bool, help="Enable video-level temporal modeling")
parser.add_argument("--early_fusion", default=True, type=str2bool, help="Use early fusion (frame-level fusion first)")
parser.add_argument("--fusion_type", default="concat", type=str, help="Fusion strategy: concat/add/bilinear/attention/gated")
parser.add_argument("--feature_dim", default=256, type=int, help="Feature dimension per branch")
parser.add_argument("--fused_dim", default=512, type=int, help="Fused feature dimension (early fusion)")

# Temporal settings
parser.add_argument("--num_frames", default=16, type=int, help="Frames to sample per video")
parser.add_argument("--num_heads", default=8, type=int, help="Transformer attention heads")
parser.add_argument("--num_layers", default=2, type=int, help="Transformer encoder layers")

# Dataset settings
parser.add_argument("--dataset_root", default="", type=str, help="RGB frames root directory")
parser.add_argument("--optical_root", default="", type=str, help="Optical flow root directory")
parser.add_argument("--datasets", default="", type=str, help="Training dataset names (comma-separated)")
parser.add_argument("--datasets_optical", default="", type=str, help="Optical flow dataset names (if different)")
parser.add_argument("--datasets_test", default="", type=str, help="Test dataset names")
parser.add_argument("--datasets_optical_test", default="", type=str, help="Optical flow test dataset names")

# Training settings
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--lr", default=0.0001, type=float, help="Learning rate")
parser.add_argument("--nepoch", default=400, type=int, help="Number of epochs")
parser.add_argument("--optim", default="adam", type=str, help="Optimizer: adam/sgd")
parser.add_argument("--beta1", default=0.9, type=float, help="Adam beta1")
parser.add_argument("--pretrained", default=True, type=str2bool, help="Use pretrained weights")

# Early stopping and warmup
parser.add_argument("--warmup", default=False, type=str2bool, help="Enable learning rate warmup")
parser.add_argument("--warmup_epoch", default=3, type=int, help="Warmup epochs")
parser.add_argument("--earlystop", default=True, type=str2bool, help="Enable early stopping")
parser.add_argument("--earlystop_epoch", default=5, type=int, help="Early stopping patience")

# Save settings
parser.add_argument("--save_epoch_freq", default=20, type=int, help="Save model every N epochs")
parser.add_argument("--save_latest_freq", default=2000, type=int, help="Save latest model every N steps")

# Continue training
parser.add_argument("--continue_train", default=False, type=str2bool, help="Continue training from checkpoint")
parser.add_argument("--epoch", default="latest", type=str, help="Which checkpoint to load: 'latest', 'best', or epoch number")
parser.add_argument("--epoch_count", default=1, type=int, help="Starting epoch count")
parser.add_argument("--last_epoch", default=-1, type=int, help="Last epoch for scheduler")

# Data augmentation
parser.add_argument("--blur_prob", default=0.1, type=float, help="Blur augmentation probability")
parser.add_argument("--jpg_prob", default=0.1, type=float, help="JPEG compression probability")
parser.add_argument("--aug_flip", default=True, type=str2bool, help="Enable horizontal flip")
parser.add_argument("--aug_norm", default=True, type=str2bool, help="Enable normalization")
parser.add_argument("--aug_crop", default=True, type=str2bool, help="Enable cropping")

# Other settings
parser.add_argument("--num_workers", default=0, type=int, help="DataLoader workers (Windows: use 0)")
parser.add_argument("--class_bal", default=False, type=str2bool, help="Class-balanced sampling")

args, _unknown = parser.parse_known_args()

# Load experiment config if exists
if args.exp_name and os.path.exists(os.path.join(DefaultConfigs.exp_root, args.exp_name, "config.py")):
    sys.path.insert(0, os.path.join(DefaultConfigs.exp_root, args.exp_name))
    from config import cfg
    cfg: CONFIGCLASS
else:
    cfg = DefaultConfigs()

# Apply command-line arguments to config
for key, value in vars(args).items():
    if hasattr(cfg, key):
        default_value = getattr(DefaultConfigs, key, None)
        # Only override if argument was explicitly provided (different from parser default)
        arg_default = parser.get_default(key)
        if value != arg_default or key in ['gpus', 'exp_name', 'ckpt']:
            setattr(cfg, key, value)

# Handle comma-separated list arguments
if args.datasets:
    cfg.datasets = [d.strip() for d in args.datasets.split(",")]
if args.datasets_optical:
    cfg.datasets_optical = [d.strip() for d in args.datasets_optical.split(",")]
if args.datasets_test:
    cfg.datasets_test = [d.strip() for d in args.datasets_test.split(",")]
if args.datasets_optical_test:
    cfg.datasets_optical_test = [d.strip() for d in args.datasets_optical_test.split(",")]

# Set GPU environment
cfg.gpus = args.gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(gpu) for gpu in cfg.gpus])
cfg.exp_name = args.exp_name
cfg.ckpt_path = os.path.join(cfg.ckpt_dir, args.ckpt)

# Override paths if specified
if args.dataset_root:
    cfg.dataset_root = args.dataset_root
if args.optical_root:
    cfg.optical_root = args.optical_root
