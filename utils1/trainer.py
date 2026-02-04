import os

import torch
import torch.nn as nn
from torch.nn import init

from utils1.config import CONFIGCLASS
from utils1.utils import get_network
from utils1.warmup import GradualWarmupScheduler
from networks.fusion import create_dual_branch_model, create_video_dual_branch_model, create_early_fusion_video_model


class BaseModel(nn.Module):
    def __init__(self, cfg: CONFIGCLASS):
        super().__init__()
        self.cfg = cfg
        self.total_steps = 0
        self.loaded_epoch = 0  # 存储加载的epoch，避免重复加载checkpoint
        self.isTrain = cfg.isTrain
        self.save_dir = cfg.ckpt_dir
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model: nn.Module
        self.model = None
        #self.model.load_state_dict(torch.load('./checkpoints/optical.pth'))
        self.optimizer: torch.optim.Optimizer

    def save_networks(self, epoch: int, current_epoch: int = None):
        save_filename = f"model_epoch_{epoch}.pth"
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epoch": current_epoch if current_epoch is not None else epoch,
        }

        # Save scheduler state if it exists
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()

        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, epoch: int):
        load_filename = f"model_epoch_{epoch}.pth"
        load_path = os.path.join(self.save_dir, load_filename)

        if epoch==0:
            # load_filename = f"lsun_adm.pth"
            load_path="checkpoints/optical.pth"
            print("loading optical path")
        else :
            print(f"loading the model from {load_path}")

        # print(f"loading the model from {load_path}")

        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=self.device)
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        self.model.load_state_dict(state_dict["model"])
        self.total_steps = state_dict["total_steps"]

        # Load epoch number if available
        self.loaded_epoch = state_dict.get("epoch", 0)

        if self.isTrain and not self.cfg.new_optim:
            self.optimizer.load_state_dict(state_dict["optimizer"])
            # move optimizer state to GPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)

            # Don't reset learning rate - use the one from checkpoint
            # If you want to use a different lr, set --new_optim True
            # for g in self.optimizer.param_groups:
            #     g["lr"] = self.cfg.lr

        # Load scheduler state if available and scheduler exists
        if self.isTrain and hasattr(self, 'scheduler') and self.scheduler is not None:
            if "scheduler" in state_dict:
                self.scheduler.load_state_dict(state_dict["scheduler"])
                print(f"Loaded scheduler state from checkpoint")

        return self.loaded_epoch

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net: nn.Module, init_type="normal", gain=0.02):
    def init_func(m: nn.Module):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f"initialization method [{init_type}] is not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"initialize network with {init_type}")
    net.apply(init_func)


class Trainer(BaseModel):
    def name(self):
        return "Trainer"

    def __init__(self, cfg: CONFIGCLASS):
        super().__init__(cfg)
        self.arch = cfg.arch
        self.model = get_network(self.arch, cfg.isTrain, cfg.continue_train, cfg.init_gain, cfg.pretrained)

        self.loss_fn = nn.BCEWithLogitsLoss()
        # initialize optimizers
        if cfg.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        elif cfg.optim == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError("optim should be [adam, sgd]")
        if cfg.warmup:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cfg.nepoch - cfg.warmup_epoch, eta_min=1e-6
            )
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, multiplier=1, total_epoch=cfg.warmup_epoch, after_scheduler=scheduler_cosine
            )
            self.scheduler.step()
        if cfg.continue_train:
            self.load_networks(cfg.epoch)
        self.model.to(self.device)

        

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def set_input(self, input):
        img, label, meta = input if len(input) == 3 else (input[0], input[1], {})
        self.input = img.to(self.device)
        self.label = label.to(self.device).float()
        for k in meta.keys():
            if isinstance(meta[k], torch.Tensor):
                meta[k] = meta[k].to(self.device)
        self.meta = meta

    def forward(self):
        self.output = self.model(self.input, self.meta)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def train(self):
        self.model.train()


class DualBranchTrainer(BaseModel):
    """
    Trainer for dual-branch model with RGB and optical flow fusion.
    """

    def name(self):
        return "DualBranchTrainer"

    def __init__(self, cfg: CONFIGCLASS):
        super().__init__(cfg)
        self.arch_rgb = cfg.arch_rgb
        self.arch_optical = cfg.arch_optical

        # Create dual-branch fusion model
        self.model = create_dual_branch_model(
            arch_rgb=cfg.arch_rgb,
            arch_optical=cfg.arch_optical,
            fusion_type=cfg.fusion_type,
            feature_dim=cfg.feature_dim,
            pretrained=cfg.pretrained
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize optimizer
        if cfg.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        elif cfg.optim == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError("optim should be [adam, sgd]")

        if cfg.warmup:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cfg.nepoch - cfg.warmup_epoch, eta_min=1e-6
            )
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, multiplier=1, total_epoch=cfg.warmup_epoch, after_scheduler=scheduler_cosine
            )
            self.scheduler.step()

        if cfg.continue_train:
            self.load_networks(cfg.epoch)

        self.model.to(self.device)
        print(f"DualBranchTrainer initialized:")
        print(f"  - RGB branch: {cfg.arch_rgb}")
        print(f"  - Optical branch: {cfg.arch_optical}")
        print(f"  - Fusion type: {cfg.fusion_type}")
        print(f"  - Feature dim: {cfg.feature_dim}")

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def set_input(self, input):
        """
        Set input for dual-branch model.
        Input format: (rgb_tensor, optical_tensor, label)
        """
        rgb, optical, label = input
        self.input_rgb = rgb.to(self.device)
        self.input_optical = optical.to(self.device)
        self.label = label.to(self.device).float()

    def forward(self):
        self.output = self.model(self.input_rgb, self.input_optical)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def train(self):
        self.model.train()


class VideoTrainer(BaseModel):
    """
    Trainer for video-level model with Transformer temporal aggregation.
    """

    def name(self):
        return "VideoTrainer"

    def __init__(self, cfg: CONFIGCLASS):
        super().__init__(cfg)
        self.arch_rgb = cfg.arch_rgb
        self.arch_optical = cfg.arch_optical
        self.early_fusion = getattr(cfg, 'early_fusion', True)

        # Choose between Early Fusion and Late Fusion
        if self.early_fusion:
            # Early Fusion: Frame-level fusion -> Single Transformer
            self.model = create_early_fusion_video_model(
                arch_rgb=cfg.arch_rgb,
                arch_optical=cfg.arch_optical,
                fusion_type=cfg.fusion_type,
                feature_dim=cfg.feature_dim,
                fused_dim=getattr(cfg, 'fused_dim', 512),
                num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
                pretrained=cfg.pretrained
            )
        else:
            # Late Fusion: Separate Transformers -> Fusion
            self.model = create_video_dual_branch_model(
                arch_rgb=cfg.arch_rgb,
                arch_optical=cfg.arch_optical,
                fusion_type=cfg.fusion_type,
                feature_dim=cfg.feature_dim,
                num_heads=cfg.num_heads,
                num_layers=cfg.num_layers,
                pretrained=cfg.pretrained
            )

        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize optimizer
        if cfg.optim == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
        elif cfg.optim == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise ValueError("optim should be [adam, sgd]")

        if cfg.warmup:
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, cfg.nepoch - cfg.warmup_epoch, eta_min=1e-6
            )
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, multiplier=1, total_epoch=cfg.warmup_epoch, after_scheduler=scheduler_cosine
            )
            self.scheduler.step()

        if cfg.continue_train:
            self.load_networks(cfg.epoch)

        self.model.to(self.device)
        fusion_mode = "Early Fusion (Frame-level)" if self.early_fusion else "Late Fusion (Separate Transformers)"
        print(f"VideoTrainer initialized:")
        print(f"  - Fusion mode: {fusion_mode}")
        print(f"  - RGB branch: {cfg.arch_rgb}")
        print(f"  - Optical branch: {cfg.arch_optical}")
        print(f"  - Fusion type: {cfg.fusion_type}")
        print(f"  - Feature dim: {cfg.feature_dim}")
        if self.early_fusion:
            print(f"  - Fused dim: {getattr(cfg, 'fused_dim', 512)}")
        print(f"  - Num frames: {cfg.num_frames}")
        print(f"  - Transformer: {cfg.num_layers} layers, {cfg.num_heads} heads")

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def set_input(self, input):
        """
        Set input for video-level model.
        Input format: (rgb_video, optical_video, label)
        rgb_video: [batch, num_frames, 3, H_rgb, W_rgb]
        optical_video: [batch, num_frames, 3, H_optical, W_optical]
        """
        rgb, optical, label = input
        self.input_rgb = rgb.to(self.device)
        self.input_optical = optical.to(self.device)
        self.label = label.to(self.device).float()

    def forward(self):
        self.output = self.model(self.input_rgb, self.input_optical)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def train(self):
        self.model.train()
