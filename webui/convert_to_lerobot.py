#!/usr/bin/env python3
"""
Convert a synced FastUMI dataset (<dataset>/episode_N.hdf5, written by
convert_episodes.py) into a LeRobot v3 dataset folder. Run in the base env.

    python convert_to_lerobot.py --src /home/nuc8/FastUMI/dataset/UMI_2026-09-23_pen_nitya \\
        --out /home/nuc8/FastUMI/dataset/UMI_2026-09-23_pen_nitya/lerobot --task-name "pick up the pen"

Per frame it writes
    observation.state          T265 pose [x, y, z, qx, qy, qz, qw]   (observations/qpos)
    action                     the same pose                          (action)
    observation.images.<cam>   camera video, RGB, --width x --height  (observations/images/<cam>)
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import cv2
import h5py
import numpy as np

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

POSE_NAMES = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
# lerobot 0.6.0 defaults to AV1, which this env's torchcodec can't decode;
# H.264 matches replay_pipeline/d1_to_lerobot_stage2.py.
H264_ENCODER = RGBEncoderConfig(vcodec='h264', pix_fmt='yuv420p', g=2)
BLOCK = 32   # frames read from the HDF5 at once (~200 MB at 1080p)
# Written into the LeRobot dataset; push_to_hub.py puts it on the Hub dataset card.
SOURCE_META = 'fastumi_episodes.json'
DELETED_LOG = 'deleted_episodes.json'   # kept in the source dataset by the web UI


def episode_files(src, only):
    """[(episode number, path)] sorted by number."""
    files = []
    for path in glob.glob(os.path.join(src, 'episode_*.hdf5')):
        m = re.match(r'episode_(\d+)\.hdf5$', os.path.basename(path))
        if m and (not only or int(m.group(1)) in only):
            files.append((int(m.group(1)), path))
    return sorted(files)


def write_source_meta(src, out, numbers):
    """Record which recorded episode each LeRobot episode is, and what was deleted."""
    try:
        with open(os.path.join(src, DELETED_LOG)) as f:
            deleted = json.load(f).get('deleted', [])
    except (OSError, ValueError):
        deleted = []
    meta = {
        'source_dataset': os.path.basename(os.path.normpath(src)),
        'source_episodes': numbers,   # LeRobot episode_index i = recorded episode numbers[i]
        'deleted_episodes': deleted,
    }
    with open(os.path.join(out, 'meta', SOURCE_META), 'w') as f:
        json.dump(meta, f, indent=2)
    if deleted:
        print('deleted episodes (recorded in meta/{}): {}'.format(
            SOURCE_META, ', '.join(str(e['episode']) for e in deleted)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='synced dataset folder with episode_N.hdf5')
    ap.add_argument('--out', required=True, help='LeRobot folder to write')
    ap.add_argument('--repo-id', default=None, help='dataset id label (default: local/<folder name>)')
    ap.add_argument('--task-name', default=None, help='task text stored per frame (default: from the folder name)')
    ap.add_argument('--fps', type=int, default=None,
                    help='dataset fps (default: the fps Sync wrote into the episodes, 30)')
    ap.add_argument('--episodes', default='', help='only these episode numbers, e.g. 1,2,5')
    ap.add_argument('--no-video', action='store_true', help='pose only, no camera')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=360)
    ap.add_argument('--overwrite', action='store_true', help='delete --out first if it exists')
    args = ap.parse_args()

    only = {int(x) for x in args.episodes.split(',') if x.strip()} if args.episodes else None
    files = episode_files(args.src, only)
    if not files:
        sys.exit('ERROR: no episode_N.hdf5 files in {} (run Sync first)'.format(args.src))
    name = os.path.basename(os.path.normpath(args.src))
    repo_id = args.repo_id or 'local/' + name
    task_text = args.task_name or name.replace('_', ' ')

    with h5py.File(files[0][1], 'r') as f:
        cams = sorted(f['observations/images'].keys()) if not args.no_video else []
        dim = f['observations/qpos'].shape[1]
        synced_fps = f.attrs.get('fps')
    if args.fps is None:
        # Episodes synced before Sync recorded its rate kept every 3rd frame of 60 fps video.
        args.fps = int(synced_fps) if synced_fps is not None else 20
        if synced_fps is None:
            print('NOTE: these episodes were synced at 20 fps (before the 30 fps change). '
                  'Delete their episode_N.hdf5 and Sync again (raw kept) to get 30 fps.')
    names = POSE_NAMES if dim == len(POSE_NAMES) else None
    features = {
        'observation.state': {'dtype': 'float32', 'shape': (dim,), 'names': names},
        'action': {'dtype': 'float32', 'shape': (dim,), 'names': names},
    }
    for cam in cams:
        features['observation.images.' + cam] = {
            'dtype': 'video', 'shape': (args.height, args.width, 3), 'names': ['height', 'width', 'channels']}

    if os.path.exists(args.out):
        if not args.overwrite:
            sys.exit('ERROR: {} already exists (use --overwrite)'.format(args.out))
        shutil.rmtree(args.out)

    print('source   : {}'.format(args.src))
    print('output   : {}'.format(args.out))
    print('task     : "{}"   fps: {}   cameras: {}'.format(task_text, args.fps, ', '.join(cams) or 'none'))
    print('Converting {} episode(s)'.format(len(files)), flush=True)

    ds = LeRobotDataset.create(repo_id=repo_id, fps=args.fps, features=features, root=args.out,
                               robot_type='fastumi', use_videos=bool(cams),
                               rgb_encoder=H264_ENCODER if cams else None)
    for i, (_, path) in enumerate(files, 1):
        with h5py.File(path, 'r') as f:
            qpos = f['observations/qpos'][:].astype(np.float32)
            action = f['action'][:].astype(np.float32)
            images = {c: f['observations/images/' + c] for c in cams}
            n = min([len(qpos), len(action)] + [len(v) for v in images.values()])
            # A whole 1080p episode does not fit in RAM, and single-frame reads
            # decompress each 8-frame gzip chunk 8 times: read BLOCK frames at once.
            for start in range(0, n, BLOCK):
                block = {c: v[start:min(n, start + BLOCK)] for c, v in images.items()}
                for t in range(start, min(n, start + BLOCK)):
                    frame = {'observation.state': qpos[t], 'action': action[t], 'task': task_text}
                    for c, b in block.items():
                        img = cv2.resize(b[t - start], (args.width, args.height), interpolation=cv2.INTER_AREA)
                        frame['observation.images.' + c] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ds.add_frame(frame)
        ds.save_episode()
        print('[{}/{}] {}: {} frames'.format(i, len(files), os.path.basename(path), n), flush=True)
    ds.finalize()
    write_source_meta(args.src, args.out, [n for n, _ in files])
    print('Saved LeRobot dataset: {}'.format(args.out))


if __name__ == '__main__':
    main()
