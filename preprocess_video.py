import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Allow importing RAFT and utils from the local "core" package.
sys.path.append("core")
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


# Default video file extensions to scan.
VIDEO_EXTS_DEFAULT = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"]

# Keep RAFT iterations aligned with demo.py (fixed at 20).
RAFT_ITERS = 20


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Preprocess videos into frames and optical flow images using RAFT"
    )
    parser.add_argument(
        "--path",
        default="video/000000.mp4",
        help="dataset for preprocessing (video file or directory)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="output root directory (auto creates rgb/ and flow/ subfolders, enables recursive + preserve_structure)",
    )
    parser.add_argument(
        "-for",
        "--folder_original_path",
        default="frame/000000",
        help="output root for RGB frames (label/video/00000.jpg)",
    )
    parser.add_argument(
        "-fop",
        "--folder_optical_flow_path",
        default="optical_result/000000",
        help="output root for flow images (label/video/00000.jpg)",
    )
    parser.add_argument("--model", help="restore checkpoint", default="raft_model/raft-things.pth")
    parser.add_argument("--use_cpu", action="store_true", help="force CPU (GPU used by default)")
    parser.add_argument("--small", action="store_true", help="use small RAFT model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--alternate_corr", action="store_true", help="use alt CUDA correlation")
    parser.add_argument(
        "--exts",
        default=",".join(VIDEO_EXTS_DEFAULT),
        help="video extensions, comma-separated",
    )
    parser.add_argument("--recursive", action="store_true", help="search input dir recursively")
    parser.add_argument(
        "--label",
        default="0_real",
        help="label to use when input dir has no 0_real/1_fake folders",
    )
    parser.add_argument(
        "--preserve_structure",
        action="store_true",
        help="preserve input subfolder structure under label/video",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default="ffmpeg",
        help="ffmpeg executable path (e.g. D:\\miniconda\\Library\\bin\\ffmpeg.exe)",
    )
    parser.add_argument(
        "--hwaccel",
        default="cuda",
        help="ffmpeg hardware accel (set empty string to disable)",
    )
    parser.add_argument(
        "--start_number",
        type=int,
        default=0,
        help="start index for extracted frames (default 0 -> 00000.png)",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="shard index (0-based) for parallel processing",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="total number of shards for parallel processing",
    )
    return parser.parse_args()


def list_videos(input_path: str, exts: list[str], recursive: bool) -> list[str]:
    """Collect videos from a path (single file or directory)."""
    if os.path.isfile(input_path):
        return [input_path]

    videos = []
    if recursive:
        for root, _, files in os.walk(input_path):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    videos.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_path):
            full = os.path.join(input_path, name)
            if os.path.isfile(full) and Path(name).suffix.lower() in exts:
                videos.append(full)
    return sorted(videos)


def collect_video_items(
    input_path: str,
    exts: list[str],
    recursive: bool,
    label_default: str,
    preserve_structure: bool,
) -> list[tuple[str, str, str]]:
    """Collect videos and map them to (video_path, label, video_id)."""
    # Case 1: input is a single file.
    if os.path.isfile(input_path):
        video_id = Path(input_path).stem
        return [(input_path, label_default, video_id)]

    # Case 2: input is a directory.
    label_dirs = []
    for label in ("0_real", "1_fake"):
        label_path = os.path.join(input_path, label)
        if os.path.isdir(label_path):
            label_dirs.append((label, label_path))

    items = []
    if label_dirs:
        # The input directory already follows the README label format.
        for label, label_path in label_dirs:
            videos = list_videos(label_path, exts, recursive)
            for video_path in videos:
                if preserve_structure:
                    rel = os.path.relpath(video_path, label_path)
                    video_id = os.path.splitext(rel)[0]
                else:
                    video_id = Path(video_path).stem
                items.append((video_path, label, video_id))
    else:
        # No label folders found; treat all videos as the given default label.
        videos = list_videos(input_path, exts, recursive)
        for video_path in videos:
            if preserve_structure:
                rel = os.path.relpath(video_path, input_path)
                video_id = os.path.splitext(rel)[0]
            else:
                video_id = Path(video_path).stem
            items.append((video_path, label_default, video_id))

    return items


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_image_tensor(image_path: str, device: torch.device) -> torch.Tensor:
    """Load an image from disk and convert to a torch tensor."""
    img = np.array(Image.open(image_path)).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)


def count_jpg_files(folder: str) -> int:
    """Count JPG files in a folder."""
    if not os.path.isdir(folder):
        return 0
    return len(list(Path(folder).glob("*.jpg")))


def is_video_processed(out_frames_dir: str, out_flow_dir: str) -> bool:
    """
    Check whether a video has been fully processed.

    Expected output:
    - RGB frames: N images (00000.jpg to 0000{N-1}.jpg)
    - Optical flow: N-1 images (00000.jpg to 0000{N-2}.jpg)
    - flow[i] = flow between frame[i] and frame[i+1]
    """
    frame_count = count_jpg_files(out_frames_dir)
    flow_count = count_jpg_files(out_flow_dir)
    return frame_count >= 2 and flow_count == frame_count - 1


def reset_output_dirs(out_frames_dir: str, out_flow_dir: str) -> None:
    """Remove partial outputs so the video can be reprocessed cleanly."""
    if os.path.isdir(out_frames_dir):
        shutil.rmtree(out_frames_dir)
    if os.path.isdir(out_flow_dir):
        shutil.rmtree(out_flow_dir)


@torch.no_grad()
def compute_flow_image(model: torch.nn.Module, img1: torch.Tensor, img2: torch.Tensor) -> np.ndarray:
    """Compute optical flow image (colorized) from two tensors."""
    # Pad to be divisible by 8 for RAFT.
    padder = InputPadder(img1.shape)
    img1, img2 = padder.pad(img1, img2)
    # RAFT returns (low_res_flow, upsampled_flow).
    _, flow_up = model(img1, img2, iters=RAFT_ITERS, test_mode=True)
    flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
    # Convert flow vectors to color image for visualization/storage.
    flow_img = flow_viz.flow_to_image(flow)
    return flow_img


def extract_frames_ffmpeg(
    video_path: str,
    out_frames_dir: str,
    ffmpeg_path: str,
    hwaccel: str,
    start_number: int,
) -> None:
    """Extract all frames with ffmpeg (GPU decode if available)."""
    ensure_dir(out_frames_dir)
    output_pattern = os.path.join(out_frames_dir, "%05d.jpg")

    cmd = [ffmpeg_path]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += [
        "-i",
        video_path,
        "-start_number",
        str(start_number),
        "-q:v",
        "2",
        "-y",
        output_pattern,
    ]

    # Run ffmpeg and stream output to console.
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg not found. Set --ffmpeg_path to the full exe path. ({exc})"
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for video: {video_path}")


def process_video(
    video_path: str,
    out_frames_dir: str,
    out_flow_dir: str,
    model: torch.nn.Module,
    device: torch.device,
    ffmpeg_path: str,
    hwaccel: str,
    start_number: int,
) -> None:
    """Extract all frames with ffmpeg and compute optical flow between adjacent frames."""
    if is_video_processed(out_frames_dir, out_flow_dir):
        print(f"skip processed: {video_path}")
        return

    if os.path.isdir(out_frames_dir) or os.path.isdir(out_flow_dir):
        reset_output_dirs(out_frames_dir, out_flow_dir)

    ensure_dir(out_frames_dir)
    ensure_dir(out_flow_dir)

    # 1) Extract all frames using ffmpeg (GPU decode if available).
    try:
        extract_frames_ffmpeg(
            video_path,
            out_frames_dir,
            ffmpeg_path,
            hwaccel,
            start_number,
        )
    except RuntimeError as exc:
        print(exc)
        return

    # 2) Load extracted frames from disk and compute flow.
    frame_files = sorted(Path(out_frames_dir).glob("*.jpg"))
    if len(frame_files) <= 1:
        print(f"{video_path} has <=1 frame; no optical flow generated")
        return

    pbar = tqdm(total=len(frame_files) - 1, desc=f"flow:{Path(video_path).name}", unit="pair")

    # Use torch.no_grad() to prevent gradient calculation and save memory
    with torch.no_grad():
        for idx in range(len(frame_files) - 1):
            img1_path = str(frame_files[idx])
            img2_path = str(frame_files[idx + 1])

            img1 = load_image_tensor(img1_path, device)
            img2 = load_image_tensor(img2_path, device)

            # Pad to be divisible by 8 for RAFT
            padder = InputPadder(img1.shape)
            img1, img2 = padder.pad(img1, img2)

            # Compute optical flow
            _, flow_up = model(img1, img2, iters=RAFT_ITERS, test_mode=True)

            # Remove padding to restore original dimensions
            flow = padder.unpad(flow_up)[0].permute(1, 2, 0).detach().cpu().numpy()

            # Convert flow vectors to color image for visualization/storage
            flow_img = flow_viz.flow_to_image(flow)
            # flow_viz produces RGB, but OpenCV expects BGR for writing
            flow_bgr = flow_img[:, :, ::-1]

            # Save flow with the same base name as the first frame
            flow_name = Path(img1_path).stem + ".jpg"
            flow_path = os.path.join(out_flow_dir, flow_name)
            cv2.imwrite(flow_path, flow_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Explicitly delete tensors to free up memory
            del img1, img2, flow_up, flow

            pbar.update(1)

    pbar.close()

    # Clear CUDA cache after processing each video
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def main() -> None:
    """Main entry: load model and process video(s)."""
    args = parse_args()

    # If --output is specified, auto-configure paths and enable recursive + preserve_structure
    if args.output:
        args.folder_original_path = os.path.join(args.output, "rgb")
        args.folder_optical_flow_path = os.path.join(args.output, "flow")
        args.recursive = True
        args.preserve_structure = True
        print(f"output: {args.output} (rgb/ and flow/ subfolders)")

    # Parse extensions list.
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    if not exts:
        exts = VIDEO_EXTS_DEFAULT

    # Decide device: use GPU by default if available.
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    print(f"device: {device}")

    # Enable cudnn benchmarking for faster fixed-size workloads on GPU.
    torch.backends.cudnn.benchmark = device.type == "cuda"

    # Load RAFT model once and reuse for all videos.
    # Removed DataParallel to save memory on single GPU setup
    model = RAFT(args)
    
    # Load state dict handling both DataParallel and standard checkpoints
    state_dict = torch.load(args.model, map_location=device)
    if "module." in list(state_dict.keys())[0]:
        # If checkpoint was saved with DataParallel, remove 'module.' prefix
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # Collect input videos with label mapping.
    items = collect_video_items(
        args.path,
        exts,
        args.recursive,
        args.label,
        args.preserve_structure,
    )
    if not items:
        print("no videos found")
        return

    # Apply sharding for parallel processing
    if args.num_shards > 1:
        items = [item for i, item in enumerate(items) if i % args.num_shards == args.shard]
        print(f"shard {args.shard}/{args.num_shards}: processing {len(items)} videos")

    # Process each video independently (keep all frames).
    for video_path, label, video_id in items:
        out_frames_dir = os.path.join(args.folder_original_path, label, video_id)
        out_flow_dir = os.path.join(args.folder_optical_flow_path, label, video_id)
        process_video(
            video_path,
            out_frames_dir,
            out_flow_dir,
            model,
            device,
            args.ffmpeg_path,
            args.hwaccel,
            args.start_number,
        )


if __name__ == "__main__":
    # Script entrypoint.
    main()
