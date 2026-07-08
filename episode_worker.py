#!/usr/bin/env python3
"""Stage 2 — background episode processor. Run as its own long-lived process/terminal,
independent of data_collection.py's lifecycle:

    python3 episode_worker.py

Polls dataset/<task>/raw/episode_<n>/manifest.json for status "pending" (written
atomically by data_collection.py once an episode's raw files are flushed), decodes
only the frames needed for the target sync rate straight out of the raw MJPEG blob
(by byte offset — no full sequential re-decode), matches them against the T265
trajectory log, and writes the final per-episode HDF5 in the existing schema
(episode_<n>.hdf5, observations/images/<cam>, observations/qpos, action).

Resilience:
  - manifest.json / episode_<n>.hdf5 are both written via tmp-file + os.rename,
    so a crash never leaves a half-written file visible to readers.
  - Raw files are only deleted (or left in place, per config) after the HDF5 is
    re-opened and its shapes are verified against the decoded frame count.
  - A failed episode is marked "failed" with its traceback in error.log next to
    the raw files and is never retried automatically or deleted -- it simply
    doesn't block the next manifest in the queue.
  - The claim lock records this process's PID; if a worker dies mid-episode,
    the next worker to start reclaims the stale lock (dead PID or lock older
    than stale_lock_minutes) instead of leaving the episode stuck forever.
"""
import concurrent.futures as cf
import csv
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

import cv2
import h5py
import numpy as np
import pandas as pd

import episode_manifest as em

with open('config/config.json', 'r') as f:
    config = json.load(f)

cfg = config['task_config']
worker_cfg = config.get('worker', {})

DATA_ROOT        = config['device_settings']['data_dir']
POLL_SEC         = worker_cfg.get('poll_interval_sec', 1.0)
NICE_LEVEL       = worker_cfg.get('nice_level', 15)
STALE_SEC        = worker_cfg.get('stale_lock_minutes', 10) * 60
KEEP_RAW         = worker_cfg.get('keep_raw_after_success', False)
PARALLEL_WORKERS = worker_cfg.get('parallel_workers', 4)
LOG_PATH         = os.path.join(DATA_ROOT, 'worker.log')

os.makedirs(DATA_ROOT, exist_ok=True)

# Set once per pool worker process by _init_worker (stays None in the parent,
# whose own log lines are tagged "main"). Each ProcessPoolExecutor worker gets
# a stable name like "ProcessPoolExecutor-0_2" -- the trailing index is a
# small, human-followable slot number, paired here with the OS pid.
_TAG = None


def log(msg):
    tag = _TAG or 'main'
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}][{tag}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def deprioritize_self():
    """Best-effort CPU/IO isolation so this process can never starve live capture,
    which matters here since capture and processing share one physical disk."""
    try:
        os.nice(NICE_LEVEL)
    except OSError as e:
        log(f'Could not renice worker: {e}')
    try:
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())],
                        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        log('ionice not found -- skipping I/O priority isolation')


def _init_worker():
    """ProcessPoolExecutor initializer -- runs once per pool worker process at
    spawn time (not per task), so nice/ionice apply to the process actually
    doing the decode/write work, and every log() line from this process is
    tagged with a stable slot number for the duration of the pool's life."""
    global _TAG
    name = mp.current_process().name  # e.g. 'ForkProcess-2' or 'ProcessPoolExecutor-0_2'
    match = re.search(r'(\d+)$', name)
    slot = match.group(1) if match else name
    _TAG = f'slot{slot}/pid{os.getpid()}'
    deprioritize_self()


def process_episode(manifest):
    t0 = time.time()
    episode_idx = manifest['episode_index']
    data_path   = manifest['data_dir']
    image_path  = manifest['image_out_dir']
    state_path  = manifest['state_path']
    stride      = manifest.get('frame_stride', 3)
    cam_names   = manifest['camera_names']
    blob_path   = manifest['video_blob_path']
    index_path  = manifest['index_csv_path']
    traj_path   = manifest['trajectory_csv_path']
    start_time  = manifest['start_time']

    # image_out_dir is shared across every episode of this task (set once by
    # data_collection.py, not per-episode), and frame indices restart at 0 for
    # each episode -- so each episode needs its own subdirectory. Sequentially
    # this was a silent same-name overwrite; under concurrent workers it would
    # be two processes writing the same path at once (corrupted JPEG on disk).
    ep_image_dir = os.path.join(image_path, f'episode_{episode_idx}')
    os.makedirs(ep_image_dir, exist_ok=True)

    index_df    = pd.read_csv(index_path)
    downsampled = index_df.iloc[::stride].reset_index(drop=True)
    log(f'episode {episode_idx}: decoding {len(downsampled)}/{len(index_df)} frames')

    frame_imgs = []
    with open(blob_path, 'rb') as blob_f:
        for _, row in downsampled.iterrows():
            blob_f.seek(int(row['Offset']))
            chunk = blob_f.read(int(row['Length']))
            frame = cv2.imdecode(np.frombuffer(chunk, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f'Failed to decode frame at offset {row["Offset"]} in {blob_path}')
            frame_idx = int(row['Frame Index'])
            # Raw bytes are already a JPEG -- write them straight through, no re-encode.
            with open(os.path.join(ep_image_dir, f'{frame_idx // stride}.jpg'), 'wb') as jf:
                jf.write(chunk)
            frame_imgs.append(frame)

    log(f'episode {episode_idx}: matching trajectory')
    trajectory = pd.read_csv(traj_path)
    if len(trajectory) == 0:
        raise ValueError('trajectory.csv is empty -- no T265 poses were logged for this episode')

    traj_ts  = trajectory['Timestamp'].astype(float).values
    frame_ts = downsampled['Timestamp'].values

    ins = np.searchsorted(traj_ts, frame_ts).clip(1, len(traj_ts) - 1)
    left_closer = np.abs(traj_ts[ins - 1] - frame_ts) <= np.abs(traj_ts[ins] - frame_ts)
    best_idx = np.where(left_closer, ins - 1, ins)

    qcols   = ['Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
    matched = trajectory.iloc[best_idx][qcols].values  # shape (N, 7)

    dataset_path = os.path.join(data_path, f'episode_{episode_idx}.hdf5')
    tmp_path = dataset_path + '.tmp'
    with h5py.File(tmp_path, 'w') as root:
        root.attrs['sim'] = False
        obs  = root.create_group('observations')
        imgs = obs.create_group('images')
        for cam in cam_names:
            imgs.create_dataset(cam, data=np.array(frame_imgs, dtype=np.uint8))
        root.create_dataset('observations/qpos', data=matched)
        root.create_dataset('action',            data=matched)
    os.rename(tmp_path, dataset_path)  # atomic -- never exposes a half-written episode_N.hdf5

    # Verify before touching anything else (STATE_PATH append, raw deletion).
    with h5py.File(dataset_path, 'r') as check:
        for cam in cam_names:
            if check[f'observations/images/{cam}'].shape[0] != len(frame_imgs):
                raise ValueError(f'Verification failed: {cam} frame count mismatch in {dataset_path}')
        if check['observations/qpos'].shape[0] != len(matched):
            raise ValueError(f'Verification failed: qpos row count mismatch in {dataset_path}')

    # Only append to the shared, cross-episode STATE_PATH after the HDF5 is
    # verified good, so a failed episode never leaves partial rows behind.
    with open(state_path, 'a', newline='') as f:
        w = csv.writer(f)
        for i, (ts_val, pq) in enumerate(zip(frame_ts, matched)):
            w.writerow([i, start_time, traj_ts[best_idx[i]], ts_val] + pq.tolist())

    elapsed = time.time() - t0
    log(f'episode {episode_idx}: wrote {dataset_path} ({len(frame_imgs)} frames, {elapsed:.1f}s)')
    return dataset_path


def handle_pending(ep_dir, manifest):
    pid = os.getpid()
    if not em.try_claim(ep_dir, pid, stale_after_sec=STALE_SEC):
        return  # owned by another live worker

    em.update_manifest(ep_dir, status=em.STATUS_PROCESSING)
    episode_idx = manifest['episode_index']
    log(f'episode {episode_idx}: claimed ({ep_dir})')

    try:
        process_episode(manifest)
    except Exception:
        tb = traceback.format_exc()
        em.log_error(ep_dir, tb)
        em.update_manifest(ep_dir, status=em.STATUS_FAILED, error=tb.strip().splitlines()[-1])
        log(f'episode {episode_idx}: FAILED -- {tb.strip().splitlines()[-1]} (see {ep_dir}/error.log)')
        em.release_claim(ep_dir)
        return

    em.update_manifest(ep_dir, status=em.STATUS_DONE)
    em.release_claim(ep_dir)

    if not KEEP_RAW:
        shutil.rmtree(ep_dir, ignore_errors=True)
    log(f'episode {episode_idx}: done')


def _log_future_result(fut, ep_dir, manifest):
    """handle_pending() already catches and logs every normal failure itself --
    this is just a safety net for anything that escapes it (e.g. the process
    getting killed) so a busted future can never vanish silently."""
    try:
        fut.result()
    except Exception:
        tb = traceback.format_exc().strip().splitlines()[-1]
        log(f'episode {manifest["episode_index"]}: worker process for {ep_dir} '
            f'raised unexpectedly: {tb}')


def drain_queue(task=None):
    """Process every episode currently pending, then return -- one-shot alternative
    to main()'s poll loop, used by run_worker.sh for on-demand runs. If task is
    given, only that task's queue is processed instead of every task's.

    Up to PARALLEL_WORKERS episodes are processed concurrently in separate OS
    processes; the pool's own internal queue bounds concurrency, so all pending
    episodes can be submitted up front."""
    if task is not None:
        valid_tasks = em.list_tasks(DATA_ROOT)
        if task not in valid_tasks:
            if valid_tasks:
                print(f'Error: unknown task "{task}". Valid tasks: {", ".join(valid_tasks)}',
                      file=sys.stderr)
            else:
                print(f'Error: unknown task "{task}" -- no tasks found under {DATA_ROOT}',
                      file=sys.stderr)
            sys.exit(1)

    deprioritize_self()
    pending = em.list_pending(DATA_ROOT, task=task)
    total = len(pending)
    scope = f'task "{task}"' if task else 'all tasks'
    if total == 0:
        log(f'Queue is empty for {scope} -- nothing to process.')
        return
    log(f'episode_worker (drain mode) started (pid={os.getpid()}, nice={NICE_LEVEL}, '
        f'parallel_workers={PARALLEL_WORKERS}), {total} pending episode(s) for {scope}')

    done_count = 0
    with cf.ProcessPoolExecutor(max_workers=PARALLEL_WORKERS, initializer=_init_worker) as ex:
        futures = {ex.submit(handle_pending, ep_dir, manifest): (ep_dir, manifest)
                   for ep_dir, manifest in pending}
        for fut in cf.as_completed(futures):
            ep_dir, manifest = futures[fut]
            _log_future_result(fut, ep_dir, manifest)
            done_count += 1
            print(f'Processed {done_count}/{total} (task "{manifest.get("task", "?")}", '
                  f'episode {manifest["episode_index"]})', flush=True)
    log(f'Queue drained -- {total} episode(s) processed for {scope}.')


def main():
    deprioritize_self()
    log(f'episode_worker started (pid={os.getpid()}, nice={NICE_LEVEL}, '
        f'parallel_workers={PARALLEL_WORKERS}, watching {DATA_ROOT})')

    in_flight = {}  # ep_dir -> (future, manifest)
    with cf.ProcessPoolExecutor(max_workers=PARALLEL_WORKERS, initializer=_init_worker) as ex:
        while True:
            for ep_dir, (fut, manifest) in list(in_flight.items()):
                if fut.done():
                    _log_future_result(fut, ep_dir, manifest)
                    del in_flight[ep_dir]

            pending = em.list_pending(DATA_ROOT)
            new_items = [(d, m) for d, m in pending if d not in in_flight]
            if new_items:
                log(f'queue depth: {len(pending)} ({len(new_items)} newly submitted, '
                    f'{len(in_flight)} already in flight)')
            for ep_dir, manifest in new_items:
                in_flight[ep_dir] = (ex.submit(handle_pending, ep_dir, manifest), manifest)

            time.sleep(POLL_SEC)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true',
                         help='Process everything currently pending, then exit (used by '
                              'run_worker.sh) instead of polling forever.')
    parser.add_argument('task', nargs='?', default=None,
                         help='With --once, only process this task\'s pending episodes '
                              '(default: all tasks).')
    args = parser.parse_args()
    try:
        drain_queue(args.task) if args.once else main()
    except KeyboardInterrupt:
        log('episode_worker stopped (KeyboardInterrupt)')
        sys.exit(0)
