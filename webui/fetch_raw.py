#!/usr/bin/env python3
"""
Download a dataset's raw recordings from the Hugging Face Hub (conda base env):
the counterpart of push_raw.py.

    python fetch_raw.py --repo-id <user>/<dataset>_raw --dest <data_dir>/<dataset>

Puts raw/episode_N/, csv/, camera/, states.csv and deleted_episodes.json into
--dest, so Sync can rebuild the synced episode_N.hdf5. Files already there with
the right size are skipped; nothing local is deleted. Every file is checked
(same size as on the Hub) at the end.
"""
import argparse
import fnmatch
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

from push_raw import INCLUDE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--dest', required=True)
    args = ap.parse_args()

    info = HfApi().dataset_info(args.repo_id, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings
             if any(fnmatch.fnmatch(s.rfilename, p) for p in INCLUDE)}
    episodes = sorted({k.split('/')[1] for k in files if k.startswith('raw/')})
    if not episodes:
        sys.exit('ERROR: {} has no raw/episode_N folders. Is it a raw-data repo (pushed as "Raw recordings")?'.format(
            args.repo_id))
    total = sum(files.values())
    print('{}: {} raw episode(s), {} files, {:.2f} GB -> {}'.format(
        args.repo_id, len(episodes), len(files), total / 1e9, args.dest), flush=True)

    os.makedirs(args.dest, exist_ok=True)
    for i, (rel, size) in enumerate(sorted(files.items()), 1):
        path = os.path.join(args.dest, rel)
        if os.path.isfile(path) and os.path.getsize(path) == size:
            print('[{}/{}] {} already here'.format(i, len(files), rel), flush=True)
            continue
        hf_hub_download(args.repo_id, rel, repo_type='dataset', local_dir=args.dest)
        print('[{}/{}] {} downloaded ({:.0f} MB)'.format(i, len(files), rel, size / 1e6), flush=True)

    bad = [rel for rel, size in files.items()
           if not os.path.isfile(os.path.join(args.dest, rel)) or os.path.getsize(os.path.join(args.dest, rel)) != size]
    for rel in bad:
        print('ERROR: not downloaded correctly: {}'.format(rel), flush=True)
    if bad:
        sys.exit('ERROR: {} file(s) missing or the wrong size.'.format(len(bad)))
    print('Downloaded: all {} files ({} episodes, {:.2f} GB) are in {}.'.format(
        len(files), len(episodes), total / 1e9, args.dest), flush=True)


if __name__ == '__main__':
    main()
