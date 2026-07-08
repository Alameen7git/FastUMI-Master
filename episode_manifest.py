"""Shared schema and atomic filesystem-queue helpers for the raw-capture -> HDF5 handoff.

data_collection.py (Stage 1) finalizes an episode by writing manifest.json into
dataset/<task>/raw/episode_<n>/ once its raw video blob + index + trajectory CSV
are flushed to disk. episode_worker.py (Stage 2) polls for manifest.json files
with status "pending", claims them, and processes them into the final HDF5.

The manifest.json write is atomic (tmp file + os.rename on the same filesystem)
so the worker never observes a half-written manifest, and there is no message
broker or lock server involved — the filesystem is the queue, so state survives
either process crashing and restarting independently.
"""
import json
import os
import time

STATUS_PENDING    = 'pending'
STATUS_PROCESSING = 'processing'
STATUS_DONE       = 'done'
STATUS_FAILED     = 'failed'

MANIFEST_NAME  = 'manifest.json'
LOCK_NAME      = '.processing.lock'
ERROR_LOG_NAME = 'error.log'


def episode_raw_dir(data_path, episode_index):
    return os.path.join(data_path, 'raw', f'episode_{episode_index}')


def write_manifest(episode_dir, manifest):
    path = os.path.join(episode_dir, MANIFEST_NAME)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, path)  # atomic on the same filesystem
    return path


def read_manifest(episode_dir):
    with open(os.path.join(episode_dir, MANIFEST_NAME)) as f:
        return json.load(f)


def update_manifest(episode_dir, **fields):
    manifest = read_manifest(episode_dir)
    manifest.update(fields)
    write_manifest(episode_dir, manifest)
    return manifest


def try_claim(episode_dir, worker_pid, stale_after_sec=600):
    """Atomically claim an episode directory for processing.

    Returns True if claimed. If a lock already exists, reclaims it when the
    owning PID is no longer alive (worker crashed) or the lock has outlived
    stale_after_sec, so one dead worker can never wedge the queue forever.
    """
    lock_path = os.path.join(episode_dir, LOCK_NAME)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(worker_pid).encode())
        os.close(fd)
        return True
    except FileExistsError:
        pass

    try:
        with open(lock_path) as f:
            old_pid = int(f.read().strip())
    except (ValueError, OSError):
        old_pid = None

    alive = True
    if old_pid is not None:
        try:
            os.kill(old_pid, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True

    try:
        lock_age = time.time() - os.path.getmtime(lock_path)
    except OSError:
        lock_age = 0

    if not alive or lock_age > stale_after_sec:
        try:
            os.remove(lock_path)
        except OSError:
            pass
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(worker_pid).encode())
            os.close(fd)
            return True
        except FileExistsError:
            return False

    return False


def release_claim(episode_dir):
    try:
        os.remove(os.path.join(episode_dir, LOCK_NAME))
    except OSError:
        pass


def list_tasks(dataset_root):
    """Return the sorted list of task names under dataset_root (dirs with a raw/ subdir)."""
    if not os.path.isdir(dataset_root):
        return []
    return sorted(
        t for t in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, t, 'raw'))
    )


def list_pending(dataset_root, task=None):
    """Scan dataset_root/<task>/raw/<episode>/manifest.json for pending episodes, oldest first.
    If task is given, only that task's queue is scanned instead of every task."""
    pending = []
    if not os.path.isdir(dataset_root):
        return pending
    tasks = [task] if task is not None else sorted(os.listdir(dataset_root))
    for task in tasks:
        raw_dir = os.path.join(dataset_root, task, 'raw')
        if not os.path.isdir(raw_dir):
            continue
        for ep in sorted(os.listdir(raw_dir)):
            ep_dir = os.path.join(raw_dir, ep)
            if not os.path.isfile(os.path.join(ep_dir, MANIFEST_NAME)):
                continue
            try:
                manifest = read_manifest(ep_dir)
            except (json.JSONDecodeError, OSError):
                continue
            if manifest.get('status') == STATUS_PENDING:
                pending.append((ep_dir, manifest))
    pending.sort(key=lambda item: item[1].get('created_at', 0))
    return pending


def log_error(episode_dir, message):
    with open(os.path.join(episode_dir, ERROR_LOG_NAME), 'a') as f:
        f.write(message + '\n')
