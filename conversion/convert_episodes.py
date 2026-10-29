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
    episode_idx, raw_dir, data_path, camera_names, keep_raw, fps = args
    video_path = os.path.join(raw_dir, 'video.mp4')
    traj_path  = os.path.join(raw_dir, 'trajectory.csv')
    ts_path    = os.path.join(raw_dir, 'timestamps.csv')

    dataset_path = os.path.join(data_path, f'episode_{episode_idx}.hdf5')
    if os.path.exists(dataset_path):
        return (episode_idx, 'skipped', 'already converted')

    t0 = time.time()
    try:
        timestamps = pd.read_csv(ts_path)
        # Keep every Nth frame so the output is ~fps (the camera records at 60,
        # so 30 fps = every 2nd frame). The real rate comes from the timestamps.
        ts = timestamps['Timestamp'].astype(float).values
        cam_fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else 60.0
        stride = max(1, int(round(cam_fps / fps)))
        out_fps = int(round(cam_fps / stride))
        downsampled = timestamps.iloc[::stride].reset_index(drop=True)

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
            root.attrs['fps'] = out_fps
            obs = root.create_group('observations')
            imgs = obs.create_group('images')
            for cam in camera_names:
                imgs.create_dataset(cam, data=np.array(frames, dtype=np.uint8),
                                     compression='gzip', compression_opts=1)
            root.create_dataset('observations/qpos', data=np.array(qpos))
            root.create_dataset('action', data=np.array(qpos))
        os.rename(tmp_path, dataset_path)

        if not keep_raw:
            shutil.rmtree(raw_dir, ignore_errors=True)

        elapsed = time.time() - t0
        return (episode_idx, 'done', f'{elapsed:.1f}s, {len(frames)} frames @ {out_fps} fps')

    except Exception as e:
        return (episode_idx, 'failed', str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--parallel', type=int, default=2,
                         help='Number of episodes to convert simultaneously (default 2, safe for 32GB RAM)')
    parser.add_argument('--fps', type=int, default=30,
                         help='Output frame rate (default 30; the camera records at 60)')
    parser.add_argument('--keep-raw', action='store_true',
                         help='Keep raw video/csv files after successful conversion (default: delete)')
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

    print(f'Found {len(episode_dirs)} raw episode(s). Converting with {args.parallel} worker(s) at {args.fps} fps...\n')

    jobs = []
    for d in episode_dirs:
        episode_idx = int(d.split('_')[1])
        raw_dir = os.path.join(raw_root, d)
        jobs.append((episode_idx, raw_dir, data_path, cfg['camera_names'], args.keep_raw, args.fps))

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
    if failed_count > 0:
        print(f'Failed episode details logged to: {failed_log_path}')


if __name__ == '__main__':
    main()
