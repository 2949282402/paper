import os
from io import BytesIO
from random import choice, random

import cv2
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from scipy.ndimage import gaussian_filter
from torch.utils.data.sampler import WeightedRandomSampler

from utils1.config import CONFIGCLASS, get_crop_size

ImageFile.LOAD_TRUNCATED_IMAGES = True


def dataset_folder(root: str, cfg: CONFIGCLASS):
    if cfg.mode == "binary":
        return binary_dataset(root, cfg)
    if cfg.mode == "filename":
        return FileNameDataset(root, cfg)
    raise ValueError("cfg.mode needs to be binary or filename.")


def binary_dataset(root: str, cfg: CONFIGCLASS):
    identity_transform = transforms.Lambda(lambda img: img)

    rz_func = identity_transform

    # Get crop size based on architecture: FreqNet uses 224, ResNet uses 448
    # crop_size = 448  # Original: hardcoded 448 for ResNet
    crop_size = get_crop_size(cfg.arch)  # Modified: dynamic based on arch

    if cfg.isTrain:
        # crop_func = transforms.RandomCrop((448,448))  # Original
        crop_func = transforms.RandomCrop((crop_size, crop_size))  # Modified
    else:
        # crop_func = transforms.CenterCrop((448,448)) if cfg.aug_crop else identity_transform  # Original
        crop_func = transforms.CenterCrop((crop_size, crop_size)) if cfg.aug_crop else identity_transform  # Modified

    if cfg.isTrain and cfg.aug_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = identity_transform
    

    return datasets.ImageFolder(
        root,
        transforms.Compose(
            [
                rz_func,
                #change
                transforms.Lambda(lambda img: blur_jpg_augment(img, cfg)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if cfg.aug_norm
                else identity_transform,
            ]
        )
    )


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def blur_jpg_augment(img: Image.Image, cfg: CONFIGCLASS):
    img: np.ndarray = np.array(img)
    if cfg.isTrain:
        if random() < cfg.blur_prob:
            sig = sample_continuous(cfg.blur_sig)
            gaussian_blur(img, sig)

        if random() < cfg.jpg_prob:
            method = sample_discrete(cfg.jpg_method)
            qual = sample_discrete(cfg.jpg_qual)
            img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s: list):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s: list):
    return s[0] if len(s) == 1 else choice(s)


def gaussian_blur(img: np.ndarray, sigma: float):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg(img: np.ndarray, compress_val: int) -> np.ndarray:
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img: np.ndarray, compress_val: int):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {"cv2": cv2_jpg, "pil": pil_jpg}


def jpeg_from_key(img: np.ndarray, compress_val: int, key: str) -> np.ndarray:
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img: Image.Image, cfg: CONFIGCLASS) -> Image.Image:
    interp = sample_discrete(cfg.rz_interp)
    return TF.resize(img, cfg.loadSize, interpolation=rz_dict[interp])


def get_dataset(cfg: CONFIGCLASS):
    dset_lst = []
    for dataset in cfg.datasets:
        root = os.path.join(cfg.dataset_root, dataset)
        dset = dataset_folder(root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset: torch.utils.data.ConcatDataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))


def create_dataloader(cfg: CONFIGCLASS):
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    dataset = get_dataset(cfg)
    sampler = get_bal_sampler(dataset) if cfg.class_bal else None

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )


# ============ Dual-Branch Dataset for RGB + Optical Flow ============

class DualBranchDataset(torch.utils.data.Dataset):
    """
    Dataset for dual-branch training with RGB and optical flow images.

    Directory structure expected:
        rgb_root/
            0_real/
                video1/
                    frame_00000.png
                    frame_00001.png
                video2/
                    ...
            1_fake/
                ...
        optical_root/
            0_real/
                video1/
                    frame_00000.png  (optical flow visualization)
                    frame_00001.png
                ...
            1_fake/
                ...

    Each RGB frame is paired with corresponding optical flow frame by matching paths.
    """

    def __init__(self, rgb_root: str, optical_root: str, cfg: CONFIGCLASS):
        self.cfg = cfg
        self.rgb_root = rgb_root
        self.optical_root = optical_root

        # 先resize到256，再crop到224（标准做法）
        self.resize_size = 256
        self.crop_size = 224

        # Build transforms
        self.resize = transforms.Resize((self.resize_size, self.resize_size))

        # Common transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if cfg.aug_norm else transforms.Lambda(lambda x: x)

        # Collect all RGB image paths and labels
        self.samples = []
        self.targets = []

        # Walk through rgb_root to find all images
        for class_idx, class_name in enumerate(['0_real', '1_fake']):
            class_dir = os.path.join(rgb_root, class_name)
            if not os.path.exists(class_dir):
                continue

            for root, dirs, files in os.walk(class_dir):
                for fname in files:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rgb_path = os.path.join(root, fname)
                        # Construct corresponding optical flow path
                        rel_path = os.path.relpath(rgb_path, rgb_root)
                        optical_path = os.path.join(optical_root, rel_path)

                        # Only add if both files exist
                        if os.path.exists(optical_path):
                            self.samples.append((rgb_path, optical_path, class_idx))
                            self.targets.append(class_idx)

        print(f"DualBranchDataset: Found {len(self.samples)} paired samples")
        print(f"  - Resize: {self.resize_size} -> Crop: {self.crop_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rgb_path, optical_path, label = self.samples[index]

        # Load images
        rgb_img = Image.open(rgb_path).convert('RGB')
        optical_img = Image.open(optical_path).convert('RGB')

        # Resize to 256x256 first
        rgb_img = self.resize(rgb_img)
        optical_img = self.resize(optical_img)

        # Apply data augmentation (blur/jpeg) to RGB only
        if self.cfg.isTrain:
            rgb_img = blur_jpg_augment(rgb_img, self.cfg)

        # Apply synchronized flip (same flip for both)
        if self.cfg.isTrain and self.cfg.aug_flip:
            if random() > 0.5:
                rgb_img = TF.hflip(rgb_img)
                optical_img = TF.hflip(optical_img)

        # Apply synchronized crop (same position for both RGB and optical)
        if self.cfg.isTrain:
            # Random crop with same parameters for both
            i, j, h, w = transforms.RandomCrop.get_params(
                rgb_img, output_size=(self.crop_size, self.crop_size))
            rgb_img = TF.crop(rgb_img, i, j, h, w)
            optical_img = TF.crop(optical_img, i, j, h, w)
        else:
            # Center crop for testing
            if self.cfg.aug_crop:
                rgb_img = TF.center_crop(rgb_img, (self.crop_size, self.crop_size))
                optical_img = TF.center_crop(optical_img, (self.crop_size, self.crop_size))

        # Convert to tensor and normalize
        rgb_tensor = self.normalize(self.to_tensor(rgb_img))
        optical_tensor = self.normalize(self.to_tensor(optical_img))

        return rgb_tensor, optical_tensor, label


def get_dual_branch_dataset(cfg: CONFIGCLASS):
    """Get dual-branch dataset combining RGB and optical flow."""
    dset_lst = []
    # Use datasets_optical if specified, otherwise use same names as datasets
    optical_datasets = cfg.datasets_optical if cfg.datasets_optical else cfg.datasets
    for dataset, optical_dataset in zip(cfg.datasets, optical_datasets):
        rgb_root = os.path.join(cfg.dataset_root, dataset)
        optical_root = os.path.join(cfg.optical_root, optical_dataset)
        dset = DualBranchDataset(rgb_root, optical_root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def create_dual_branch_dataloader(cfg: CONFIGCLASS):
    """Create dataloader for dual-branch training."""
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    dataset = get_dual_branch_dataset(cfg)

    # Get balanced sampler if needed
    sampler = None
    if cfg.class_bal:
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        ratio = np.bincount(targets)
        w = 1.0 / torch.tensor(ratio, dtype=torch.float)
        sample_weights = w[targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )


# ============ Video-Level Dataset for Temporal Aggregation ============

class VideoDataset(torch.utils.data.Dataset):
    """
    Video-level dataset for temporal aggregation with Transformer.

    Each sample is a video with N frames (RGB + optical flow pairs).
    Returns tensors of shape [num_frames, C, H, W] for both RGB and optical flow.

    Directory structure after preprocessing:
        rgb_root/
            0_real/
                video1/
                    00000.jpg, 00001.jpg, ..., 0000{N-1}.jpg  (N frames)
            1_fake/
                ...
        optical_root/
            0_real/
                video1/
                    00000.jpg, 00001.jpg, ..., 0000{N-2}.jpg  (N-1 flow images)
                    # flow[i] = motion from frame[i] to frame[i+1]
            1_fake/
                ...

    RGB-Flow 1:1 Correspondence:
        - We sample T consecutive indices from [0, N-2] (limited by flow count)
        - RGB[i] pairs with flow[i] (flow[i] represents motion starting from frame[i])
        - Result: T RGB frames + T optical flow images (1:1)
    """

    def __init__(self, rgb_root: str, optical_root: str, cfg: CONFIGCLASS):
        self.cfg = cfg
        self.rgb_root = rgb_root
        self.optical_root = optical_root
        self.num_frames = cfg.num_frames  # Number of frames to sample per video

        # 先resize到256，再crop到224（标准做法）
        self.resize_size = 256
        self.crop_size = 224

        # Build transforms
        self.resize = transforms.Resize((self.resize_size, self.resize_size))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if cfg.aug_norm else transforms.Lambda(lambda x: x)

        # Collect videos (each video is a directory of frames)
        self.videos = []  # List of (video_rgb_dir, video_optical_dir, label)
        self.targets = []

        for class_idx, class_name in enumerate(['0_real', '1_fake']):
            rgb_class_dir = os.path.join(rgb_root, class_name)
            optical_class_dir = os.path.join(optical_root, class_name)

            if not os.path.exists(rgb_class_dir):
                continue

            # Each subdirectory in class_dir is a video
            for video_name in os.listdir(rgb_class_dir):
                video_rgb_path = os.path.join(rgb_class_dir, video_name)
                video_optical_path = os.path.join(optical_class_dir, video_name)

                if os.path.isdir(video_rgb_path) and os.path.exists(video_optical_path):
                    # Check if video has enough frames
                    optical_frames = sorted([f for f in os.listdir(video_optical_path)
                                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if len(optical_frames) >= 2:  # At least 2 flow images
                        self.videos.append((video_rgb_path, video_optical_path, class_idx))
                        self.targets.append(class_idx)

        print(f"VideoDataset: Found {len(self.videos)} videos")
        print(f"  - Frames per video: {self.num_frames}")
        print(f"  - Resize: {self.resize_size} -> Crop: {self.crop_size}")

    def __len__(self):
        return len(self.videos)

    def _sample_paired_indices(self, num_rgb_frames: int, num_optical_frames: int, num_frames: int) -> list:
        """
        Sample T consecutive indices ensuring RGB-flow 1:1 correspondence.

        After preprocessing:
        - RGB has N frames: 00000.jpg ~ 0000{N-1}.jpg
        - Optical flow has N-1 images: 00000.jpg ~ 0000{N-2}.jpg
        - flow[i] = motion from RGB[i] to RGB[i+1]

        Sampling strategy:
        - 取RGB和Optical的最小帧数作为可用帧数
        - 从 [0, available_frames - T] 中选取起始位置
        - 确保采样的T帧在RGB和Optical中都存在且一一对应

        Args:
            num_rgb_frames: Number of RGB frames (N)
            num_optical_frames: Number of optical flow images (N-1)
            num_frames: Target number of frames to sample (T)

        Returns:
            List of T consecutive indices for both RGB and optical flow
        """
        # 取RGB和Optical的最小帧数，确保采样的索引对两者都有效
        available_frames = min(num_rgb_frames, num_optical_frames)

        if available_frames >= num_frames:
            # 帧数足够：采样连续的T帧
            max_start = available_frames - num_frames  # 起始位置范围 [0, max_start]
            if self.cfg.isTrain:
                # 训练：随机起始位置
                start_idx = np.random.randint(0, max_start + 1)
            else:
                # 测试：居中位置
                start_idx = max_start // 2
            indices = list(range(start_idx, start_idx + num_frames))
        else:
            # 帧数不足：使用所有可用帧，末尾用最后一帧填充
            indices = list(range(available_frames))
            # 用最后一帧填充到T帧
            last_idx = available_frames - 1
            indices.extend([last_idx] * (num_frames - available_frames))

        return indices

    def __getitem__(self, index):
        video_rgb_path, video_optical_path, label = self.videos[index]

        # Get sorted frame lists
        rgb_frames = sorted([f for f in os.listdir(video_rgb_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        optical_frames = sorted([f for f in os.listdir(video_optical_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Sample paired indices (same indices for RGB and flow, ensuring 1:1)
        # 传入RGB和Optical的帧数，确保采样的索引对两者都有效
        indices = self._sample_paired_indices(len(rgb_frames), len(optical_frames), self.num_frames)

        # Decide flip and crop for this video (same for all frames)
        do_flip = self.cfg.isTrain and self.cfg.aug_flip and random() > 0.5

        # 为整个视频生成一个统一的crop参数（训练时随机，测试时居中）
        # 与DualBranchDataset保持一致：测试时只有aug_crop=True才进行crop
        if self.cfg.isTrain:
            # 随机crop参数（对整个视频使用相同的crop位置）
            do_crop = True
            crop_top = int(random() * (self.resize_size - self.crop_size))
            crop_left = int(random() * (self.resize_size - self.crop_size))
        else:
            # 测试模式：只有aug_crop=True时才进行居中crop
            do_crop = self.cfg.aug_crop
            crop_top = (self.resize_size - self.crop_size) // 2
            crop_left = (self.resize_size - self.crop_size) // 2

        rgb_tensors = []
        optical_tensors = []

        for idx in indices:
            # Use same index for RGB and optical flow (1:1 correspondence)
            rgb_fname = rgb_frames[idx]
            optical_fname = optical_frames[idx]

            # Load RGB frame
            rgb_path = os.path.join(video_rgb_path, rgb_fname)
            rgb_img = Image.open(rgb_path).convert('RGB')

            # Load optical flow frame
            optical_path = os.path.join(video_optical_path, optical_fname)
            optical_img = Image.open(optical_path).convert('RGB')

            # Resize to 256x256 first
            rgb_img = self.resize(rgb_img)
            optical_img = self.resize(optical_img)

            # Apply augmentation
            if self.cfg.isTrain:
                rgb_img = blur_jpg_augment(rgb_img, self.cfg)

            # Apply synchronized flip
            if do_flip:
                rgb_img = TF.hflip(rgb_img)
                optical_img = TF.hflip(optical_img)

            # Apply synchronized crop (same position for RGB and optical, same for all frames)
            if do_crop:
                rgb_img = TF.crop(rgb_img, crop_top, crop_left, self.crop_size, self.crop_size)
                optical_img = TF.crop(optical_img, crop_top, crop_left, self.crop_size, self.crop_size)

            # Convert to tensor and normalize
            rgb_tensor = self.normalize(self.to_tensor(rgb_img))
            optical_tensor = self.normalize(self.to_tensor(optical_img))

            rgb_tensors.append(rgb_tensor)
            optical_tensors.append(optical_tensor)

        # Stack frames: [num_frames, C, H, W]
        rgb_batch = torch.stack(rgb_tensors, dim=0)
        optical_batch = torch.stack(optical_tensors, dim=0)

        return rgb_batch, optical_batch, label


def get_video_dataset(cfg: CONFIGCLASS):
    """Get video-level dataset for temporal aggregation."""
    dset_lst = []
    # Use datasets_optical if specified, otherwise use same names as datasets
    optical_datasets = cfg.datasets_optical if cfg.datasets_optical else cfg.datasets
    for dataset, optical_dataset in zip(cfg.datasets, optical_datasets):
        rgb_root = os.path.join(cfg.dataset_root, dataset)
        optical_root = os.path.join(cfg.optical_root, optical_dataset)
        dset = VideoDataset(rgb_root, optical_root, cfg)
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)


def create_video_dataloader(cfg: CONFIGCLASS):
    """Create dataloader for video-level training with temporal aggregation."""
    shuffle = not cfg.serial_batches if (cfg.isTrain and not cfg.class_bal) else False
    dataset = get_video_dataset(cfg)

    sampler = None
    if cfg.class_bal:
        targets = []
        for d in dataset.datasets:
            targets.extend(d.targets)
        ratio = np.bincount(targets)
        w = 1.0 / torch.tensor(ratio, dtype=torch.float)
        sample_weights = w[targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
        shuffle = False

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(cfg.num_workers),
    )
