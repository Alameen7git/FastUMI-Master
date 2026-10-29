#!/usr/bin/env python3
"""
Push a local LeRobot dataset folder to the Hugging Face Hub (base env).

Fallback used by the web UI when replay_pipeline/push_dataset_to_hub.py is not
in the checkout; same flags:

    python push_to_hub.py --repo-id <user_or_org>/<name> --root /path/to/dataset [--public]
"""
import argparse
import contextlib
import inspect
import json
import os

from huggingface_hub import CommitOperationDelete, HfApi
from huggingface_hub.errors import RevisionNotFoundError
from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDataset

KEEP_REMOTE = {'README.md', '.gitattributes'}


def card_description(root):
    """Dataset card text from meta/fastumi_episodes.json (written by convert_to_lerobot.py)."""
    try:
        with open(os.path.join(root, 'meta', 'fastumi_episodes.json')) as f:
            meta = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    nums = meta.get('source_episodes') or []
    lines = ['Recorded with FastUMI (EGI). Source dataset: `{}`.'.format(meta.get('source_dataset', '?')), '',
             'LeRobot episode_index → recorded episode number: ' +
             ', '.join('{}→{}'.format(i, n) for i, n in enumerate(nums)) + '.', '']
    deleted = meta.get('deleted_episodes') or []
    if deleted:
        lines += ['### Deleted episodes', '',
                  'These recorded episodes were deleted after review and are not in this dataset:', '',
                  '| Episode | Deleted at | Reason |', '|---|---|---|']
        lines += ['| {} | {} | {} |'.format(e.get('episode'), e.get('deleted_at', ''),
                                           (e.get('reason') or '—').replace('|', '/')) for e in deleted]
    else:
        lines.append('No episodes were deleted.')
    lines += ['', 'Details: [meta/fastumi_episodes.json](meta/fastumi_episodes.json).']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description='Push a local LeRobot dataset to the HF Hub')
    ap.add_argument('--repo-id', required=True)
    ap.add_argument('--root', default=None)
    ap.add_argument('--public', action='store_true', help='push as public (default: private)')
    args = ap.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    print('num_episodes: {}'.format(dataset.num_episodes))
    print('num_frames:   {}'.format(dataset.num_frames))
    print('fps:          {}'.format(dataset.fps))

    kwargs = {}
    if 'private' in inspect.signature(dataset.push_to_hub).parameters:
        kwargs['private'] = not args.public
    description = card_description(dataset.root)
    if description:
        kwargs['dataset_description'] = description
        print(description)
    if 'tag_version' in inspect.signature(dataset.push_to_hub).parameters:
        kwargs['tag_version'] = False   # tagged below, after the clean-up
    dataset.push_to_hub(**kwargs)
    remove_stale(dataset.repo_id, dataset.root)
    api = HfApi()
    with contextlib.suppress(RevisionNotFoundError):
        api.delete_tag(dataset.repo_id, tag=CODEBASE_VERSION, repo_type='dataset')
    api.create_tag(dataset.repo_id, tag=CODEBASE_VERSION, repo_type='dataset')
    print('Pushed to: https://huggingface.co/datasets/{}'.format(dataset.repo_id))


def remove_stale(repo_id, root):
    """push_to_hub only adds and overwrites files. When the dataset was converted
    again with fewer episodes, delete the Hub files it no longer has."""
    local = set()
    for dirpath, _, files in os.walk(root):
        for name in files:
            local.add(os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, '/'))
    api = HfApi()
    stale = [f for f in api.list_repo_files(repo_id, repo_type='dataset')
             if f not in local and f not in KEEP_REMOTE]
    if stale:
        print('Removing {} old file(s) from the Hub: {}'.format(len(stale), ', '.join(stale)))
        api.create_commit(repo_id, repo_type='dataset', commit_message='Remove files no longer in the dataset',
                          operations=[CommitOperationDelete(path_in_repo=f) for f in stale])


if __name__ == '__main__':
    main()
