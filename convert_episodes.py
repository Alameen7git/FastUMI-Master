import os
import sys
import json
import argparse
import shutil
import time
import numpy as np
import pandas as pd
import cv2
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed

with open('config/config.json', 'r') as f:
    config = json.load(f)

cfg = config['task_config']


def convert_one_episode(args):
    episode_idx, raw_dir, data_path, camera_names, delete_raw = args
    video_path = os.path.join(raw_dir, 'video.mp4')
    traj_path  = os.path.join(raw_dir, 'trajectory.csv')
    ts_path    = os.path.join(raw_dir, 'timestamps.csv')

    dataset_path = os.path.join(data_path, f'episode_{episode_idx}.hdf5')
    if os.path.exists(dataset_path):
        return (episode_idx, 'skipped', 'already converted')

    t0 = time.time()
    try:
        timestamps = pd.read_csv(ts_path)
        downsampled = timestamps.iloc[::3].reset_index(drop=True)

        cap = cv2.VideoCapture(video_path)
        frames = []
        wanted = set(downsampled['Frame Index'].astype(int).tolist())
        frame_idx = 0
        target = len(wanted)
        collected = 0
        while collected < target:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in wanted:
                frames.append(frame)
                collected += 1
            frame_idx += 1
        cap.release()

        if len(frames) < target * 0.8:
            return (episode_idx, 'failed', f'only got {len(frames)}/{target} frames -- video may be corrupted')

        trajectory = pd.read_csv(traj_path)
        trajectory['Timestamp'] = trajectory['Timestamp'].astype(float)
        traj_ts = trajectory['Timestamp'].values

        qpos = []
        for _, row in downsampled.iterrows():
            idx = np.abs(traj_ts - row['Timestamp']).argmin()
            closest = trajectory.iloc[idx]
            qpos.append([closest['Pos X'], closest['Pos Y'], closest['Pos Z'],
                         closest['Q_X'], closest['Q_Y'], closest['Q_Z'], closest['Q_W']])

        tmp_path = dataset_path + '.tmp'
        with h5py.File(tmp_path, 'w', rdcc_nbytes=2 * 1024**2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            imgs = obs.create_group('images')
            for cam in camera_names:
                imgs.create_dataset(cam, data=np.array(frames, dtype=np.uint8),
                                     compression='gzip', compression_opts=1)
            root.create_dataset('observations/qpos', data=np.array(qpos))
            root.create_dataset('action', data=np.array(qpos))
        os.rename(tmp_path, dataset_path)

        # Raw data is preserved by default. Only delete if --delete-raw was passed.
        if delete_raw:
            shutil.rmtree(raw_dir, ignore_errors=True)

        elapsed = time.time() - t0
        return (episode_idx, 'done', f'{elapsed:.1f}s, {len(frames)} frames')

    except Exception as e:
        return (episode_idx, 'failed', str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--parallel', type=int, default=2,
                         help='Number of episodes to convert simultaneously (default 2, safe for 32GB RAM)')
    parser.add_argument('--delete-raw', action='store_true',
                         help='Delete raw video/csv files after successful conversion (default: keep raw data)')
    args = parser.parse_args()

    data_path = os.path.join(config['device_settings']['data_dir'], args.task)
    failed_log_path = os.path.join(data_path, 'failed_conversions.log')
    raw_root = os.path.join(data_path, 'raw')

    if not os.path.exists(raw_root):
        print(f'No raw episodes found at {raw_root}')
        sys.exit(1)

    episode_dirs = sorted(
        [d for d in os.listdir(raw_root) if d.startswith('episode_')],
        key=lambda x: int(x.split('_')[1])
    )
    if not episode_dirs:
        print('No raw episodes to convert.')
        sys.exit(0)

    print(f'Found {len(episode_dirs)} raw episode(s). Converting with {args.parallel} worker(s)...')
    if args.delete_raw:
        print('Raw data will be DELETED after successful conversion (--delete-raw passed).\n')
    else:
        print('Raw data will be KEPT after conversion (default). Pass --delete-raw to remove it.\n')

    jobs = []
    for d in episode_dirs:
        episode_idx = int(d.split('_')[1])
        raw_dir = os.path.join(raw_root, d)
        jobs.append((episode_idx, raw_dir, data_path, cfg['camera_names'], args.delete_raw))

    done_count = 0
    failed_count = 0
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(convert_one_episode, job): job[0] for job in jobs}
        for future in as_completed(futures):
            episode_idx, status, detail = future.result()
            if status == 'done':
                print(f'  Episode {episode_idx}: DONE ({detail})')
                done_count += 1
            elif status == 'skipped':
                print(f'  Episode {episode_idx}: SKIPPED ({detail})')
                skipped_count += 1
            else:
                print(f'  Episode {episode_idx}: FAILED -- {detail}')
                failed_count += 1
                from datetime import datetime
                with open(failed_log_path, 'a') as f:
                    f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Episode {episode_idx} | {detail}\n')

    print(f'\nConversion complete: {done_count} done, {skipped_count} skipped, {failed_count} failed')
    if failed_count > 0:
        print('Failed episodes were not deleted -- check their raw/ folders and redo if needed.')
        print(f'Failed episode details logged to: {failed_log_path}')


if __name__ == '__main__':
    main()