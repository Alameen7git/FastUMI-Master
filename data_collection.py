import os
import json
import signal
import sys
import termios
import tty
import queue
import atexit
import shutil
import torch
import argparse
from time import sleep, time
import rospy

from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
import csv
import threading
from collections import deque

import episode_manifest as em

# ── Config ────────────────────────────────────────────────────────────────────
with open('config/config.json', 'r') as f:
    config = json.load(f)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ROBOT_TYPE = config['device_settings']['robot_type']
cfg = config['task_config']

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='test')
parser.add_argument('--num_episodes', type=int, default=5)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

# ── Paths ─────────────────────────────────────────────────────────────────────
data_path = os.path.join(config['device_settings']['data_dir'], task)
CSV_PATH  = os.path.join(data_path, 'csv/')
for p in (data_path, CSV_PATH):
    os.makedirs(p, exist_ok=True)

STATE_PATH = os.path.join(data_path, 'states.csv')
if not os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Index', 'Start Time', 'Trajectory Timestamp', 'Frame Timestamp',
            'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'
        ])

IMAGE_PATH    = os.path.join(data_path, 'camera/')
FRAME_TS_PATH = os.path.join(data_path, 'csv', 'frame_timestamps.csv')
FRAME_STRIDE  = config.get('sync', {}).get('frame_stride', 3)

# ── ROS ───────────────────────────────────────────────────────────────────────
rospy.init_node('video_trajectory_recorder', anonymous=True)

frame_width, frame_height = cfg['cam_width'], cfg['cam_height']

video_buffer      = deque()
trajectory_buffer = deque()
buffer_lock       = threading.Lock()

# ── Shared recording state ────────────────────────────────────────────────────
is_recording       = False
first_frame_ts     = None
first_time_judger  = False
start_time         = 0

# ── Keyboard control ──────────────────────────────────────────────────────────
# Keys: space = start/stop   r = redo   e = end session
key_queue    = queue.Queue()
session_done = threading.Event()

_stdin_fd     = sys.stdin.fileno()
_stdin_is_tty = os.isatty(_stdin_fd)
if _stdin_is_tty:
    _term_orig = termios.tcgetattr(_stdin_fd)
    atexit.register(termios.tcsetattr, _stdin_fd, termios.TCSADRAIN, _term_orig)

def keyboard_listener():
    """Reads single keypresses without blocking the main thread. On a real
    terminal, cbreak mode delivers each keystroke immediately without waiting
    for Enter. When stdin is a plain pipe -- e.g. a supervising process (a web
    UI) writing single control bytes instead of a human typing -- there's no
    such wait to avoid, since the caller already controls exactly what byte
    arrives and when, so cbreak mode is skipped entirely."""
    if _stdin_is_tty:
        tty.setcbreak(_stdin_fd)
    while not session_done.is_set():
        ch = sys.stdin.read(1)
        if not ch:
            break  # stdin closed (EOF) -- supervising process exited/disconnected
        if ch in (' ', 'r', 'R', 'e', 'E'):
            key_queue.put(ch.lower())

def sigint_handler(sig, frame):
    """Ctrl+C stops the current recording (treated as Space) instead of killing the process."""
    key_queue.put(' ')

def sigterm_handler(sig, frame):
    """External stop request (e.g. stop_collection.sh) -- treated as 'E' so any
    recording in progress is stopped and finalized (manifest written) before the
    process exits, instead of being killed mid-write."""
    key_queue.put('e')

signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigterm_handler)

def wait_key(valid):
    """Block until one of the valid keys is pressed. Returns the key or None if session ends."""
    while not session_done.is_set() and not rospy.is_shutdown():
        try:
            k = key_queue.get(timeout=0.1)
            if k in valid:
                return k
        except queue.Empty:
            continue
    return None

# ── ROS callbacks ─────────────────────────────────────────────────────────────
def video_callback(msg):
    """msg.data is the raw JPEG bytes straight from cam_capture_node -- no decode here."""
    global first_frame_ts, first_time_judger
    if not is_recording:
        return
    timestamp = msg.header.stamp.to_sec()
    with buffer_lock:
        if start_time < timestamp:
            video_buffer.append((msg.data, timestamp))
            if first_time_judger:
                first_frame_ts = timestamp
                first_time_judger = False

def trajectory_callback(msg):
    if not is_recording:
        return
    timestamp = msg.header.stamp.to_sec()
    with buffer_lock:
        if start_time < timestamp:
            p = msg.pose.pose
            trajectory_buffer.append((
                timestamp,
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
            ))

# ── Writer threads ────────────────────────────────────────────────────────────
def write_raw_thread(blob_file, index_writer, done):
    """Appends raw JPEG bytes to the episode blob and records (index, ts, offset,
    length) -- no decode, no re-encode, just a byte copy. The buffer pop happens
    under the lock but the (slower) file write does not, so this can never make
    video_callback wait on disk I/O."""
    frame_index = 0
    offset = 0
    while not rospy.is_shutdown():
        item = None
        with buffer_lock:
            if video_buffer:
                item = video_buffer.popleft()
            elif done.is_set():
                break
        if item is None:
            sleep(0.001)
            continue
        data, timestamp = item
        blob_file.write(data)
        length = len(data)
        index_writer.writerow([frame_index, timestamp, offset, length])
        offset += length
        frame_index += 1

def write_trajectory_thread(traj_writer, done):
    while not rospy.is_shutdown():
        item = None
        with buffer_lock:
            if trajectory_buffer:
                item = trajectory_buffer.popleft()
            elif done.is_set():
                break
        if item is None:
            sleep(0.001)
            continue
        traj_writer.writerow(item)

# ── Status printer (shows live frame count while recording) ───────────────────
def status_printer(frame_count_ref, stop_flag):
    t0 = time()
    while not stop_flag.is_set():
        elapsed = time() - t0
        with buffer_lock:
            n = len(video_buffer)
        print(f'\r  ● REC  {elapsed:5.1f}s   frames buffered: {n}   (Space=stop  R=redo  E=end)', end='', flush=True)
        sleep(0.5)
    print()  # newline after recording ends

# ── Finalize episode: write the manifest and hand off to the worker ──────────
def finalize_episode(episode_idx, ep_dir, blob_path, index_path, traj_path,
                      frame_ts_writer, captured_first_frame_ts):
    """No decoding, no HDF5 write here -- just flag the already-flushed raw files
    as ready for episode_worker.py to pick up. This is what keeps the gap between
    episodes down to milliseconds instead of ~30s."""
    frame_ts_writer.writerow([episode_idx, captured_first_frame_ts])

    manifest = {
        'episode_index': episode_idx,
        'task': task,
        'created_at': time(),
        'status': em.STATUS_PENDING,
        'video_blob_path': blob_path,
        'index_csv_path': index_path,
        'trajectory_csv_path': traj_path,
        'camera_names': cfg['camera_names'],
        'cam_width': frame_width,
        'cam_height': frame_height,
        'frame_stride': FRAME_STRIDE,
        'start_time': start_time,
        'first_frame_ts': captured_first_frame_ts,
        'data_dir': data_path,
        'state_path': STATE_PATH,
        'image_out_dir': IMAGE_PATH,
    }
    em.write_manifest(ep_dir, manifest)
    print(f'  Queued → {ep_dir}', flush=True)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rospy.Subscriber(cfg['ros']['video_topic'],      CompressedImage, video_callback,      queue_size=cfg['ros']['queue_size'])
    rospy.Subscriber(cfg['ros']['trajectory_topic'], Odometry,        trajectory_callback, queue_size=cfg['ros']['queue_size'])

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    print(f'\nFastUMI  |  Robot: {ROBOT_TYPE}  |  Task: {task}  |  Episodes: {num_episodes}')
    print('Controls:  Space = start / stop   R = redo episode   E = end session\n')

    completed = 0

    with open(FRAME_TS_PATH, 'a', newline='') as frame_ts_file:
        frame_ts_writer = csv.writer(frame_ts_file)
        frame_ts_writer.writerow(['Episode Index', 'Timestamp'])

        while completed < num_episodes:
            print(f'─── Episode {completed + 1}/{num_episodes}  ───')
            print('  Space = start   R = redo previous   E = end session')

            key = wait_key([' ', 'r', 'e'])
            if key == 'e' or session_done.is_set() or rospy.is_shutdown():
                print('\nSession ended.')
                break

            if key == 'r':
                if completed == 0:
                    print('  Nothing to redo — no episodes saved yet.\n')
                    continue
                prev_idx = completed - 1
                prev_dir = em.episode_raw_dir(data_path, prev_idx)
                if os.path.exists(os.path.join(prev_dir, em.LOCK_NAME)):
                    print(f'  ⚠ Episode {prev_idx} is already being processed by the worker — '
                          f'cannot redo safely. Delete its output manually if needed.\n')
                    continue
                completed = prev_idx
                print(f'  ↺ Redoing episode {completed + 1}...')
                shutil.rmtree(prev_dir, ignore_errors=True)
                hdf5_path = os.path.join(data_path, f'episode_{prev_idx}.hdf5')
                if os.path.exists(hdf5_path):
                    os.remove(hdf5_path)
                    print(f'  Deleted {hdf5_path}')
                print()
                continue

            # ── Start recording ──────────────────────────────────────────────
            is_recording      = True
            start_time        = rospy.Time.now().to_sec()
            first_time_judger = True

            ep_dir = em.episode_raw_dir(data_path, completed)
            os.makedirs(ep_dir, exist_ok=True)
            video_path = os.path.join(ep_dir, 'video.mjpeg')
            traj_path  = os.path.join(ep_dir, 'trajectory.csv')
            index_path = os.path.join(ep_dir, 'index.csv')

            done_flag   = threading.Event()
            stop_status = threading.Event()

            with open(video_path, 'wb')          as blob_f, \
                 open(traj_path, 'w', newline='') as traj_f, \
                 open(index_path, 'w', newline='') as index_f:

                traj_w  = csv.writer(traj_f)
                index_w = csv.writer(index_f)
                traj_w.writerow(['Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])
                index_w.writerow(['Frame Index', 'Timestamp', 'Offset', 'Length'])

                vt = threading.Thread(target=write_raw_thread,        args=(blob_f, index_w, done_flag))
                tt = threading.Thread(target=write_trajectory_thread, args=(traj_w, done_flag))
                st = threading.Thread(target=status_printer,          args=(None, stop_status))
                vt.start(); tt.start(); st.start()

                key = wait_key([' ', 'r', 'e'])

                # Stop recording — let callbacks know immediately
                is_recording = False
                # Signal writers to drain remaining buffer items then exit
                done_flag.set()
                vt.join(); tt.join()
                stop_status.set(); st.join()

                blob_f.flush()
                os.fsync(blob_f.fileno())
            # blob_f/traj_f/index_f are now closed — episode's raw files are finalized on disk

            if key == 'r':
                print('  ↺ Current take discarded. Press R again at the prompt to also redo the previous saved episode.\n')
                with buffer_lock:
                    video_buffer.clear()
                    trajectory_buffer.clear()
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue  # stay on same episode number, don't save

            if key == 'e' or session_done.is_set() or rospy.is_shutdown():
                print(f'\n  Finalizing episode {completed + 1}...')
                finalize_episode(completed, ep_dir, video_path, index_path, traj_path,
                                  frame_ts_writer, first_frame_ts)
                completed += 1
                print(f'  Done. ({completed}/{num_episodes} recorded — processing continues in background)')
                session_done.set()
                break

            # Space — finalize instantly, then move to next episode right away
            print(f'  Stopped. Finalizing episode {completed + 1}...')
            finalize_episode(completed, ep_dir, video_path, index_path, traj_path,
                              frame_ts_writer, first_frame_ts)
            completed += 1
            print(f'  Done. ({completed}/{num_episodes} recorded — processing continues in background)\n')

    session_done.set()
    print(f'\nAll done. {completed}/{num_episodes} episodes recorded. '
          f'Run episode_worker.py (if not already running) to finish processing.')
