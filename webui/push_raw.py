#!/usr/bin/env python3
"""
Upload a dataset's raw recordings to the Hugging Face Hub (conda base env).

    python push_raw.py --repo-id <user>/<dataset>_raw --dataset-dir <data_dir>/<dataset> [--public]

Uploads to the repo (one per dataset, next to the LeRobot repo <user>/<dataset>)
only what Sync needs to rebuild the synced episode_N.hdf5 files:
    raw/episode_N/{video.mp4, trajectory.csv, timestamps.csv}
    csv/, camera/, states.csv, deleted_episodes.json
The synced hdf5 files (10x bigger) and converted copies are left out. Files
already on the Hub are never deleted; unchanged files are skipped. The repo is
created private unless --public (an existing repo keeps its visibility).
After the upload every file is checked on the Hub (same path, same size), and
the repo gets a README.md (dataset card: what is in it, how to get it back),
rewritten on every push unless someone replaced it with their own.
"""
import argparse
import fnmatch
import os
import subprocess
import sys

from huggingface_hub import HfApi

INCLUDE = ['raw/*', 'csv/*', 'camera/*', 'states.csv', 'deleted_episodes.json']
CARD_MARK = '<!-- fastumi-raw-card: rewritten by webui/push_raw.py on every push -->'


def episode_number(name):
    try:
        return int(name.split('_')[1])
    except (IndexError, ValueError):
        return 1 << 30


def dataset_card(repo_id, name, files):
    """README.md for a raw-data repo. viewer: false - the Hub cannot preview mp4 + CSV folders."""
    eps = sorted({k.split('/')[1] for k in files if k.startswith('raw/')}, key=episode_number)
    nums = [episode_number(e) for e in eps]
    sizes = {e: sum(v for k, v in files.items() if k.startswith('raw/{}/'.format(e))) for e in eps}
    total = sum(files.values())
    rows = ['| {} | {:.0f} MB |'.format(e, sizes[e] / 1e6) for e in eps]
    return '\n'.join([
        '---', 'viewer: false', 'tags:', '- robotics', '- fastumi', '- raw-data', '---', CARD_MARK, '',
        '# {} — raw recordings'.format(name), '',
        'Raw FastUMI (EGI) recordings of the dataset `{}`: {} episode(s), {:.2f} GB.'.format(name, len(eps), total / 1e9),
        'Episode numbers: {}.'.format(', '.join(map(str, nums))), '',
        'These are the recordings as they were made, before Sync. The synced `episode_N.hdf5` files and the',
        'LeRobot dataset (`{}`, if it was pushed) are made from them.'.format(repo_id.rsplit('_raw', 1)[0]), '',
        '## What is in here', '',
        '| Path | What |', '|---|---|',
        '| `raw/episode_N/video.mp4` | GoPro video of the episode (60 fps) |',
        '| `raw/episode_N/trajectory.csv` | T265 poses (position + orientation) with timestamps |',
        '| `raw/episode_N/timestamps.csv` | timestamp of every video frame |',
        '| `csv/`, `camera/`, `states.csv` | session files written by the recorder |',
        '| `deleted_episodes.json` | episodes deleted after review, if any |', '',
        '## Getting the synced data back', '',
        'In the FastUMI web UI: **Sync → Raw data from Hugging Face**, type `{}`, press'.format(repo_id),
        '**Download & sync**. It downloads into `<data_dir>/{}/` and builds `episode_N.hdf5`.'.format(name), '',
        'By hand:', '', '```',
        'hf download {} --repo-type dataset --local-dir <data_dir>/{}'.format(repo_id, name),
        '```', 'then Sync `{}` in the web UI.'.format(name), '',
        '## Episodes', '', '| Episode | Size |', '|---|---|'] + rows + [''])


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
    print('{}: {} raw episode(s), {} files, {:.2f} GB -> {}'.format(
        name, len(episodes), len(files), total / 1e9, args.repo_id), flush=True)

    hf = os.path.join(os.path.dirname(sys.executable), 'hf')
    cmd = [hf, 'upload', args.repo_id, root, '.', '--repo-type', 'dataset',
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
    missing = [k for k in files if k not in remote]
    wrong = [k for k in files if k in remote and remote[k] != files[k]]
    for k in missing:
        print('ERROR: missing on the Hub: {}'.format(k), flush=True)
    for k in wrong:
        print('ERROR: size differs on the Hub: {}'.format(k), flush=True)
    if missing or wrong:
        sys.exit('ERROR: {} file(s) missing, {} with the wrong size.'.format(len(missing), len(wrong)))
    print('Verified: all {} files ({} episodes, {:.2f} GB) are on the Hub ({}).'.format(
        len(files), len(episodes), total / 1e9, 'private' if info.private else 'public'), flush=True)
    api = HfApi()
    if 'README.md' in remote:
        own = open(api.hf_hub_download(args.repo_id, 'README.md', repo_type='dataset')).read()
        write_card = CARD_MARK in own
    else:
        write_card = True
    if write_card:
        api.upload_file(path_or_fileobj=dataset_card(args.repo_id, name, files).encode(), path_in_repo='README.md',
                        repo_id=args.repo_id, repo_type='dataset', commit_message='Dataset card')
        print('Dataset card (README.md) written.', flush=True)
    else:
        print('README.md on the Hub was written by someone else: left as it is.', flush=True)
    print('Pushed to: https://huggingface.co/datasets/{}'.format(args.repo_id), flush=True)


if __name__ == '__main__':
    main()
