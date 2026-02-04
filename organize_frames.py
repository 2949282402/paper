#!/usr/bin/env python3
"""
将视频帧按视频名称组织到对应文件夹中。

用法: python organize_frames.py /path/to/folder
"""

import argparse
import os
import shutil
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="将视频帧按视频名称组织到对应文件夹中"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="包含视频帧的文件夹路径"
    )
    return parser.parse_args()


def organize_frames(folder_path):
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误: 文件夹不存在: {folder_path}")
        return

    # 收集所有jpg文件并按视频名称分组
    video_frames = defaultdict(list)

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith('.jpg'):
            continue

        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        # 解析文件名: videoname_000000.jpg
        # 找到最后一个下划线，前面是视频名，后面是帧号
        name_without_ext = os.path.splitext(filename)[0]
        last_underscore = name_without_ext.rfind('_')

        if last_underscore == -1:
            print(f"跳过: 文件名格式不正确: {filename}")
            continue

        video_name = name_without_ext[:last_underscore]
        frame_number = name_without_ext[last_underscore + 1:]

        # 验证帧号是否为数字
        if not frame_number.isdigit():
            print(f"跳过: 帧号不是数字: {filename}")
            continue

        video_frames[video_name].append(filename)

    if not video_frames:
        print("未找到符合格式的jpg文件")
        return

    # 创建文件夹并移动文件
    for video_name, frames in video_frames.items():
        # 创建视频名称对应的文件夹
        video_folder = os.path.join(folder_path, video_name)
        os.makedirs(video_folder, exist_ok=True)

        # 移动文件
        for frame in frames:
            src = os.path.join(folder_path, frame)
            dst = os.path.join(video_folder, frame)
            shutil.move(src, dst)

        print(f"已移动 {len(frames)} 个文件到: {video_name}/")

    print(f"\n完成! 共处理 {len(video_frames)} 个视频的帧文件")


def main():
    args = parse_args()
    organize_frames(args.folder)


if __name__ == "__main__":
    main()
