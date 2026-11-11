#!/usr/bin/env python3
"""Get a LeRobot (v3.0) dataset ready for visualize_dataset.py: a Hugging Face
repo id (user/name) or a local LeRobot folder.

    python3 viz/fetch_lerobot.py Nityavedula/UMI_2026-11-05_T0_a1 --out <data_dir>/.viz_cache/hf/<user>__<name>

Runs in the conda base env (huggingface_hub, pyarrow); the viewer itself runs
in the FastUMI env, which has neither. Writes into --out:
    dataset.json      repo, fps, cameras, value names, episode list
    episode_<n>.json  the plotted values (observation.state, else action) and action
    episode_<n>.mp4   that episode's camera video (cameras side by side, 540p),
                      cut out of the LeRobot video file; none for a dataset without video
Episodes already in --out are kept unless the source changed.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

PREVIEW_HEIGHT = 540


def source_root(dataset):
    if os.path.isdir(dataset):
        return os.path.realpath(dataset)
    from huggingface_hub import snapshot_download
    print('Downloading {} from Hugging Face ...'.format(dataset), flush=True)
    return snapshot_download(dataset, repo_type='dataset')


def names(feature):
    """Value names of a feature: a list, or {'motors': [...]} in some datasets."""
    n = feature.get('names') or []
    if isinstance(n, dict):
        n = next(iter(n.values()), [])
    return [str(x) for x in n]


def cut_video(root, info, ep, cams, out):
    """ffmpeg: this episode's [from, to) of each camera file, side by side, H.264."""
    cmd = ['ffmpeg', '-y', '-loglevel', 'error']
    for cam in cams:
        src = os.path.join(root, info['video_path'].format(
            video_key=cam, chunk_index=int(ep['videos/{}/chunk_index'.format(cam)]),
            file_index=int(ep['videos/{}/file_index'.format(cam)])))
        start = float(ep['videos/{}/from_timestamp'.format(cam)])
        end = float(ep['videos/{}/to_timestamp'.format(cam)])
        cmd += ['-ss', '{:.4f}'.format(start), '-t', '{:.4f}'.format(end - start), '-i', src]
    scale = ['[{}:v]scale=-2:{}[v{}]'.format(i, PREVIEW_HEIGHT, i) for i in range(len(cams))]
    stack = ''.join('[v{}]'.format(i) for i in range(len(cams)))
    graph = ';'.join(scale) + (';{}hstack=inputs={}[out]'.format(stack, len(cams)) if len(cams) > 1
                               else ';[v0]null[out]')
    tmp = out + '.tmp.mp4'
    cmd += ['-filter_complex', graph, '-map', '[out]', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
            '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', help='Hugging Face repo id (user/name) or a local LeRobot folder')
    ap.add_argument('--out', required=True, help='viewer cache folder for this dataset')
    args = ap.parse_args()

    root = source_root(args.dataset)
    with open(os.path.join(root, 'meta', 'info.json')) as f:
        info = json.load(f)
    if not str(info.get('codebase_version', '')).startswith('v3'):
        sys.exit('Only LeRobot v3 datasets can be viewed (this one is {}).'.format(info.get('codebase_version')))
    feats = info['features']
    cams = [k for k, v in feats.items() if v.get('dtype') == 'video']
    shown = 'observation.state' if 'observation.state' in feats else 'action'

    data = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, 'data', '*', '*.parquet')))],
                     ignore_index=True)
    meta = pd.concat([pd.read_parquet(p) for p in
                      sorted(glob.glob(os.path.join(root, 'meta', 'episodes', '*', '*.parquet')))],
                     ignore_index=True).sort_values('episode_index')

    os.makedirs(args.out, exist_ok=True)
    stamp = os.path.join(args.out, 'source.txt')   # a different download: start the cache again
    old = open(stamp).read() if os.path.isfile(stamp) else None
    if old != root:
        for p in glob.glob(os.path.join(args.out, 'episode_*')):
            os.remove(p)
        with open(stamp, 'w') as f:
            f.write(root)

    if cams and not shutil.which('ffmpeg'):
        sys.exit('ffmpeg not found: needed to cut the episode videos.')
    episodes = [int(e) for e in meta['episode_index']]
    print('{}: {} episode(s), {} fps, cameras: {}'.format(
        args.dataset, len(episodes), info['fps'], ', '.join(cams) or 'none (plots only)'), flush=True)
    for i, (_, ep) in enumerate(meta.iterrows(), 1):
        n = int(ep['episode_index'])
        rows = data[data['episode_index'] == n].sort_values('frame_index')
        with open(os.path.join(args.out, 'episode_{}.json'.format(n)), 'w') as f:
            json.dump({'qpos': [list(map(float, v)) for v in rows[shown]],
                       'action': [list(map(float, v)) for v in rows['action']]}, f)
        video = os.path.join(args.out, 'episode_{}.mp4'.format(n))
        if cams and not os.path.isfile(video):
            cut_video(root, info, ep, cams, video)
        print('[{}/{}] episode_{} ready'.format(i, len(episodes), n), flush=True)

    with open(os.path.join(args.out, 'dataset.json'), 'w') as f:
        json.dump({'repo': args.dataset, 'fps': info['fps'], 'cameras': cams, 'shown': shown,
                   'labels': names(feats[shown]), 'episodes': episodes}, f)
    print('Done: {} episode(s) ready to view.'.format(len(episodes)), flush=True)


if __name__ == '__main__':
    main()
