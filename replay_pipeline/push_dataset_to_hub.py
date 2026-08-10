#!/usr/bin/env python3
"""
Push an already-converted local LeRobot dataset to the Hugging Face Hub.

Assumes the dataset was already created locally (e.g. via
convert_fastumi_to_lerobot.py without --push-to-hub) under the LeRobot cache
for the given --repo-id, or under an explicit --root.

Usage:
    python push_dataset_to_hub.py --repo-id <hf_user>/<task_name>
    python push_dataset_to_hub.py --repo-id <hf_user>/<task_name> --root /path/to/dataset --public
"""

import argparse
import inspect

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Push a local LeRobot dataset to the HF Hub")
    parser.add_argument("--repo-id", required=True, help="HF Hub repo id, e.g. 'yourname/task_name'")
    parser.add_argument(
        "--root",
        default=None,
        help="Local dataset root, if it differs from the default LeRobot cache path for --repo-id",
    )
    parser.add_argument("--public", action="store_true", help="Push as a public dataset (default: private)")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)

    print(f"num_episodes: {dataset.num_episodes}")
    print(f"num_frames:   {dataset.num_frames}")
    print(f"fps:          {dataset.fps}")

    # push_to_hub()'s kwargs vary across lerobot versions — only pass
    # 'private' if this installed version's signature actually accepts it.
    push_kwargs = {}
    sig = inspect.signature(dataset.push_to_hub)
    if "private" in sig.parameters:
        push_kwargs["private"] = not args.public
    else:
        print(
            "Installed lerobot's push_to_hub() has no 'private' argument — pushing without it "
            "(visibility will follow the Hub/org default)."
        )

    dataset.push_to_hub(**push_kwargs)

    print(f"Pushed to: https://huggingface.co/datasets/{dataset.repo_id}")


if __name__ == "__main__":
    main()
