#!/usr/bin/env python3
"""
Convert a FastUMI-style HDF5 dataset (as described in the FastUMI paper,
Section X-B) into a LeRobot v3 dataset.

Expected input HDF5 structure per episode:
    episode_<idx>.hdf5
    ├── observations/
    │   ├── images/
    │   │   └── <camera_name> (num_frames, H, W, 3) uint8
    │   └── qpos (num_timesteps, D)
    ├── action (num_timesteps, D)
    └── attributes/sim = False

Usage:
    python convert_fastumi_to_lerobot.py \
        --src /home/nuc8/FastUMI/dataset/pick_and_place_the_bottle \
        --repo-id yourname/pick_and_place_the_bottle \
        --fps 20 \
        --task "pick and place the bottle"

Install requirements first:
    pip install lerobot h5py numpy tqdm --break-system-packages
    (or inside a venv without --break-system-packages)
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def find_episode_files(src_dir):
    """Find all per-episode HDF5 files in src_dir (recursively)."""
    patterns = ["episode_*.hdf5", "episode_*.h5", "*.hdf5", "*.h5"]
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(src_dir, "**", pat), recursive=True)))
    # de-duplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    return unique_files


def inspect_first_episode(path):
    """Read camera names, state dim, and image shape from the first episode file."""
    with h5py.File(path, "r") as f:
        if "observations" not in f:
            raise ValueError(f"{path}: no 'observations' group found — check your HDF5 schema.")

        obs = f["observations"]
        if "images" not in obs:
            raise ValueError(f"{path}: no 'observations/images' group found.")

        camera_names = list(obs["images"].keys())
        if len(camera_names) == 0:
            raise ValueError(f"{path}: 'observations/images' has no camera datasets.")

        img_shape = obs["images"][camera_names[0]].shape[1:]  # (H, W, C)
        qpos_shape = obs["qpos"].shape[1:]  # (D,)
        action_shape = f["action"].shape[1:]  # (D,)

    return camera_names, img_shape, qpos_shape, action_shape


def convert(src_dir, repo_id, fps, task, push_to_hub, robot_type):
    episode_files = find_episode_files(src_dir)
    if not episode_files:
        print(f"No HDF5 files found under {src_dir}. Check the path or file naming.")
        sys.exit(1)

    print(f"Found {len(episode_files)} episode files.")

    camera_names, img_shape, qpos_shape, action_shape = inspect_first_episode(episode_files[0])
    print(f"Detected cameras: {camera_names}")
    print(f"Image shape: {img_shape}, qpos/state dim: {qpos_shape[0]}, action dim: {action_shape[0]}")

    # Build the LeRobot feature schema
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": qpos_shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": action_shape,
            "names": None,
        },
    }
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": img_shape,
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
    )

    for ep_path in tqdm(episode_files, desc="Converting episodes"):
        with h5py.File(ep_path, "r") as f:
            obs = f["observations"]
            qpos = obs["qpos"][:]
            action = f["action"][:]
            images = {cam: obs["images"][cam][:] for cam in camera_names}

            num_frames = qpos.shape[0]

            for t in range(num_frames):
                frame = {
                    "observation.state": qpos[t].astype(np.float32),
                    "action": action[t].astype(np.float32),
                    "task": task,
                }
                for cam in camera_names:
                    frame[f"observation.images.{cam}"] = images[cam][t]

                dataset.add_frame(frame)

        dataset.save_episode()

    dataset.finalize()
    print(f"Done. Dataset written locally under the LeRobot cache for repo_id='{repo_id}'.")

    if push_to_hub:
        dataset.push_to_hub()
        print("Pushed to Hugging Face Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FastUMI HDF5 dataset to LeRobot format")
    parser.add_argument("--src", required=True, help="Path to directory containing episode HDF5 files")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. 'yourname/pick_and_place_the_bottle'")
    parser.add_argument("--fps", type=int, default=20, help="Recording FPS (FastUMI paper sub-samples to 20 Hz)")
    parser.add_argument("--task", default="pick and place the bottle", help="Natural-language task description")
    parser.add_argument("--robot-type", default="unknown", help="Robot type string, e.g. 'xarm6', 'franka'")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the converted dataset to HF Hub")
    args = parser.parse_args()

    convert(
        src_dir=args.src,
        repo_id=args.repo_id,
        fps=args.fps,
        task=args.task,
        push_to_hub=args.push_to_hub,
        robot_type=args.robot_type,
    )
