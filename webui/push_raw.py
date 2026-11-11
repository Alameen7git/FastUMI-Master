#!/usr/bin/env python3
"""
Upload a dataset's raw recordings to the Hugging Face Hub (conda base env).

    python push_raw.py --repo-id <user>/umi_raw --dataset-dir <data_dir>/<dataset> [--public]

Uploads into <repo>/<dataset>/ only what Sync needs to rebuild the synced
episode_N.hdf5 files:
    raw/episode_N/{video.mp4, trajectory.csv, timestamps.csv}
    csv/, camera/, states.csv, deleted_episodes.json
The synced hdf5 files (10x bigger) and converted copies are left out. Files
already on the Hub are never deleted; unchanged files are skipped. The repo is
created private unless --public (an existing repo keeps its visibility).
After the upload every file is checked on the Hub (same path, same size).
"""
import argparse
import fnmatch
import os
import subprocess
import sys

from huggingface_hub import HfApi

INCLUDE = ['raw/*', 'csv/*', 'camera/*', 'states.csv', 'deleted_episodes.json']


def local_files(root):
    """Relative paths (/-separated) of the files the upload sends, with sizes."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, '/')
            if any(fnmatch.fnmatch(rel, p) for p in INCLUDE):
                out[rel] = os.path.getsize(os.path.join(d, f))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--public', action='store_true', help='create the repo public (default: private)')
    args = ap.parse_args()

    root = os.path.realpath(args.dataset_dir)
    name = os.path.basename(root)
    files = local_files(root)
    episodes = sorted({k.split('/')[1] for k in files if k.startswith('raw/')})
    if not episodes:
        sys.exit('ERROR: {} has no raw/episode_N folders to upload.'.format(root))
    total = sum(files.values())
    print('{}: {} raw episode(s), {} files, {:.2f} GB -> {}/{}/'.format(
        name, len(episodes), len(files), total / 1e9, args.repo_id, name), flush=True)

    hf = os.path.join(os.path.dirname(sys.executable), 'hf')
    cmd = [hf, 'upload', args.repo_id, root, name, '--repo-type', 'dataset',
           '--no-private' if args.public else '--private',
           '--commit-message', 'Raw data: {} ({} episodes)'.format(name, len(episodes))]
    for p in INCLUDE:   # fnmatch, like local_files(): raw/* covers raw/episode_N/*
        cmd += ['--include', p]
    code = subprocess.run(cmd).returncode
    if code:
        sys.exit('ERROR: hf upload failed (exit code {}).'.format(code))

    print('Checking the files on Hugging Face ...', flush=True)
    info = HfApi().dataset_info(args.repo_id, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    missing = [k for k in files if name + '/' + k not in remote]
    wrong = [k for k in files if name + '/' + k in remote and remote[name + '/' + k] != files[k]]
    for k in missing:
        print('ERROR: missing on the Hub: {}'.format(k), flush=True)
    for k in wrong:
        print('ERROR: size differs on the Hub: {}'.format(k), flush=True)
    if missing or wrong:
        sys.exit('ERROR: {} file(s) missing, {} with the wrong size.'.format(len(missing), len(wrong)))
    print('Verified: all {} files ({} episodes, {:.2f} GB) are on the Hub ({}).'.format(
        len(files), len(episodes), total / 1e9, 'private' if info.private else 'public'), flush=True)
    print('Pushed to: https://huggingface.co/datasets/{}/tree/main/{}'.format(args.repo_id, name), flush=True)


if __name__ == '__main__':
    main()
