import os
import sys
import json
import argparse
import shutil

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.json')

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)


def parse_episodes(raw):
    episodes = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        try:
            episodes.append(int(part))
        except ValueError:
            print(f'Warning: could not parse "{part}" as an episode number -- skipping.')
    return episodes


def list_existing_episode_numbers(data_path):
    numbers = set()
    for fname in os.listdir(data_path):
        if fname.startswith('episode_') and fname.endswith('.hdf5'):
            try:
                numbers.add(int(fname[len('episode_'):-len('.hdf5')]))
            except ValueError:
                pass
    raw_root = os.path.join(data_path, 'raw')
    if os.path.isdir(raw_root):
        for dname in os.listdir(raw_root):
            if dname.startswith('episode_'):
                try:
                    numbers.add(int(dname[len('episode_'):]))
                except ValueError:
                    pass
    return numbers


def compute_rename_mapping(remaining_sorted):
    return [(old, new) for new, old in enumerate(remaining_sorted) if old != new]


def apply_renames(data_path, rename_mapping):
    tmp_suffix = '.reindex_tmp'

    # Pass 1: move every episode being renamed to a temporary name, so a
    # target number that's currently occupied by another shifting episode
    # is never overwritten mid-process.
    for old, new in rename_mapping:
        hdf5_path = os.path.join(data_path, f'episode_{old}.hdf5')
        raw_dir = os.path.join(data_path, 'raw', f'episode_{old}')
        if os.path.exists(hdf5_path):
            os.rename(hdf5_path, hdf5_path + tmp_suffix)
        if os.path.isdir(raw_dir):
            os.rename(raw_dir, raw_dir + tmp_suffix)

    # Pass 2: move from temporary names to their final numbers.
    for old, new in rename_mapping:
        hdf5_path = os.path.join(data_path, f'episode_{old}.hdf5')
        raw_dir = os.path.join(data_path, 'raw', f'episode_{old}')
        final_hdf5 = os.path.join(data_path, f'episode_{new}.hdf5')
        final_raw = os.path.join(data_path, 'raw', f'episode_{new}')
        if os.path.exists(hdf5_path + tmp_suffix):
            os.rename(hdf5_path + tmp_suffix, final_hdf5)
        if os.path.isdir(raw_dir + tmp_suffix):
            os.rename(raw_dir + tmp_suffix, final_raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--episodes', type=str, required=True,
                         help='Comma-separated episode numbers, e.g. "3,7,12"')
    parser.add_argument('--dry-run', action='store_true',
                         help='Show what would be deleted without deleting anything')
    args = parser.parse_args()

    data_path = os.path.join(config['device_settings']['data_dir'], args.task)
    if not os.path.isdir(data_path):
        print(f'Task dataset folder not found: {data_path}')
        sys.exit(1)

    episode_ids = parse_episodes(args.episodes)
    if not episode_ids:
        print('No valid episode numbers given.')
        sys.exit(1)

    to_delete = []
    not_found = []

    for ep in episode_ids:
        hdf5_path = os.path.join(data_path, f'episode_{ep}.hdf5')
        raw_dir = os.path.join(data_path, 'raw', f'episode_{ep}')

        paths = []
        if os.path.exists(hdf5_path):
            paths.append(hdf5_path)
        if os.path.isdir(raw_dir):
            paths.append(raw_dir)

        if not paths:
            not_found.append(ep)
        else:
            to_delete.append((ep, paths))

    for ep in not_found:
        print(f'WARNING: episode {ep} not found in {data_path} -- skipping.')

    if not to_delete:
        print('\nNothing to delete.')
        sys.exit(0)

    print('\nThe following will be deleted:')
    for ep, paths in to_delete:
        for p in paths:
            print(f'  {p}')

    deleted_ids = {ep for ep, _ in to_delete}
    remaining = sorted(list_existing_episode_numbers(data_path) - deleted_ids)
    rename_mapping = compute_rename_mapping(remaining)

    if rename_mapping:
        print('\nAfter deletion, the remaining episodes will be renumbered to close the gap:')
        for old, new in rename_mapping:
            print(f'  episode_{old} -> episode_{new}')
    else:
        print('\nNo renumbering needed after deletion -- remaining episodes are already sequential.')

    if args.dry_run:
        print(f'\n[DRY RUN] Would delete {len(to_delete)} episode(s) and renumber {len(rename_mapping)} '
              f'remaining episode(s); {len(not_found)} not found/skipped. Nothing was actually changed.')
        sys.exit(0)

    confirm = input(f'\nDelete these {len(to_delete)} episode(s) and renumber '
                     f'{len(rename_mapping)} remaining episode(s) as shown above? [y/N]: ').strip().lower()
    if confirm != 'y':
        print('Aborted -- nothing deleted or renamed.')
        sys.exit(0)

    deleted_count = 0
    for ep, paths in to_delete:
        for p in paths:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        deleted_count += 1
        print(f'Deleted episode {ep}.')

    apply_renames(data_path, rename_mapping)
    for old, new in rename_mapping:
        print(f'Renamed episode {old} -> episode {new}.')

    print(f'\nSummary: {deleted_count} episode(s) deleted, {len(not_found)} skipped/not found.')
    if rename_mapping:
        print(f'Renumbered {len(rename_mapping)} remaining episode(s):')
        for old, new in rename_mapping:
            print(f'  episode_{old} -> episode_{new}')
    else:
        print('No renumbering was needed.')


if __name__ == '__main__':
    main()
