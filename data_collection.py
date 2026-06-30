import os
import json
import signal
import sys
import termios
import tty
import queue
import torch
import cv2
import h5py
import argparse
from time import sleep, time
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import csv
from scipy.spatial.transform import Rotation as R
import threading
from collections import deque
import pandas as pd

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
data_path = os.path.join(config['device_settings']['data_dir'], 'dataset', task)
IMAGE_PATH = os.path.join(data_path, 'camera/')
CSV_PATH   = os.path.join(data_path, 'csv/')
for p in (data_path, IMAGE_PATH, CSV_PATH):
    os.makedirs(p, exist_ok=True)

STATE_PATH = os.path.join(data_path, 'states.csv')
if not os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Index', 'Start Time', 'Trajectory Timestamp', 'Frame Timestamp',
            'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'
        ])

VIDEO_PATH_TEMP      = os.path.join(data_path, 'camera', 'temp_video_n.mp4')
TRAJECTORY_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_trajectory.csv')
TIMESTAMP_PATH_TEMP  = os.path.join(data_path, 'csv', 'temp_video_timestamps.csv')
FRAME_TS_PATH        = os.path.join(data_path, 'csv', 'frame_timestamps.csv')

# ── ROS ───────────────────────────────────────────────────────────────────────
rospy.init_node('video_trajectory_recorder', anonymous=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width, frame_height = cfg['cam_width'], cfg['cam_height']

video_buffer      = deque()
trajectory_buffer = deque()
buffer_lock       = threading.Lock()
cv_bridge         = CvBridge()

# ── Shared recording state ────────────────────────────────────────────────────
is_recording       = False
first_frame_ts     = None
first_time_judger  = False
start_time         = 0

# ── Keyboard control ──────────────────────────────────────────────────────────
# Keys: space = start/stop   r = redo   e = end session
key_queue    = queue.Queue()
session_done = threading.Event()

def keyboard_listener():
    """Reads single keypresses without blocking the main thread."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not session_done.is_set():
            ch = sys.stdin.read(1)
            if ch in (' ', 'r', 'R', 'e', 'E'):
                key_queue.put(ch.lower())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def sigint_handler(sig, frame):
    """Ctrl+C stops the current recording (treated as Space) instead of killing the process."""
    key_queue.put(' ')

signal.signal(signal.SIGINT, sigint_handler)

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
    global first_frame_ts, first_time_judger
    if not is_recording:
        return
    frame = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    timestamp = msg.header.stamp.to_sec()
    with buffer_lock:
        if start_time < timestamp:
            video_buffer.append((frame, timestamp))
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
def write_video_thread(video_writer, ts_writer, done):
    frame_index = 0
    while not rospy.is_shutdown():
        with buffer_lock:
            if video_buffer:
                frame, timestamp = video_buffer.popleft()
                video_writer.write(frame)
                ts_writer.writerow([frame_index, timestamp])
                frame_index += 1
            elif done.is_set():
                break
        sleep(0.001)

def write_trajectory_thread(traj_writer, done):
    while not rospy.is_shutdown():
        with buffer_lock:
            if trajectory_buffer:
                traj_writer.writerow(trajectory_buffer.popleft())
            elif done.is_set():
                break
        sleep(0.001)

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

# ── Save episode to HDF5 (unchanged data format) ─────────────────────────────
def save_episode(episode_idx, video_path, frame_ts_writer):
    frame_ts_writer.writerow([episode_idx, first_frame_ts])

    timestamps = pd.read_csv(TIMESTAMP_PATH_TEMP)
    downsampled = timestamps.iloc[::3].reset_index(drop=True)
    cap = cv2.VideoCapture(video_path)

    data_dict = {'/observations/qpos': [], '/action': []}
    for cam in cfg['camera_names']:
        data_dict[f'/observations/images/{cam}'] = []

    print('  Processing frames...', flush=True)
    for _, row in downsampled.iterrows():
        frame_idx = row['Frame Index']
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(os.path.join(IMAGE_PATH, f"{int(frame_idx / 3)}.jpg"), frame)
            for cam in cfg['camera_names']:
                data_dict[f'/observations/images/{cam}'].append(frame)
    cap.release()

    print('  Matching trajectory...', flush=True)
    trajectory = pd.read_csv(TRAJECTORY_PATH_TEMP)
    trajectory['Timestamp'] = trajectory['Timestamp'].astype(float)

    for idx, row in downsampled.iterrows():
        closest = trajectory.iloc[(trajectory['Timestamp'] - row['Timestamp']).abs().argmin()]
        pos_quat = [
            closest['Pos X'], closest['Pos Y'], closest['Pos Z'],
            closest['Q_X'],   closest['Q_Y'],   closest['Q_Z'], closest['Q_W']
        ]
        data_dict['/observations/qpos'].append(pos_quat)
        data_dict['/action'].append(pos_quat)
        with open(STATE_PATH, 'a', newline='') as f:
            csv.writer(f).writerow(
                [idx, start_time, closest['Timestamp'], row['Timestamp']] + pos_quat
            )

    existing = len([n for n in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, n))])
    dataset_path = os.path.join(data_path, f'episode_{existing}.hdf5')

    with h5py.File(dataset_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs['sim'] = False
        obs = root.create_group('observations')
        imgs = obs.create_group('images')
        for cam in cfg['camera_names']:
            imgs.create_dataset(
                cam,
                data=np.array(data_dict[f'/observations/images/{cam}'], dtype=np.uint8),
                compression='gzip', compression_opts=4
            )
        root.create_dataset('observations/qpos', data=np.array(data_dict['/observations/qpos']))
        root.create_dataset('action',            data=np.array(data_dict['/action']))

    print(f'  Saved → {dataset_path}', flush=True)
    return dataset_path

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rospy.Subscriber(cfg['ros']['video_topic'],      Image,    video_callback,      queue_size=cfg['ros']['queue_size'])
    rospy.Subscriber(cfg['ros']['trajectory_topic'], Odometry, trajectory_callback, queue_size=cfg['ros']['queue_size'])

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    print(f'\nFastUMI  |  Robot: {ROBOT_TYPE}  |  Task: {task}  |  Episodes: {num_episodes}')
    print('Controls:  Space = start / stop   R = redo episode   E = end session\n')

    completed       = 0
    last_saved_path = None

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
                print(f'  ↺ Redoing episode {completed}...')
                if last_saved_path and os.path.exists(last_saved_path):
                    os.remove(last_saved_path)
                    print(f'  Deleted {last_saved_path}')
                completed -= 1
                last_saved_path = None
                print()
                continue

            # ── Start recording ──────────────────────────────────────────────
            is_recording      = True
            start_time        = rospy.Time.now().to_sec()
            first_time_judger = True
            video_path        = VIDEO_PATH_TEMP.replace('_n', f'_{completed}')
            video_writer      = cv2.VideoWriter(video_path, fourcc, 60, (frame_width, frame_height))
            done_flag         = threading.Event()
            stop_status       = threading.Event()

            with open(TRAJECTORY_PATH_TEMP, 'w', newline='') as traj_f, \
                 open(TIMESTAMP_PATH_TEMP,  'w', newline='') as ts_f:

                traj_w = csv.writer(traj_f)
                ts_w   = csv.writer(ts_f)
                traj_w.writerow(['Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])
                ts_w.writerow(['Frame Index', 'Timestamp'])

                vt = threading.Thread(target=write_video_thread,      args=(video_writer, ts_w, done_flag))
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

            video_writer.release()

            if key == 'r':
                print('  ↺ Current take discarded. Press R again at the prompt to also redo the previous saved episode.\n')
                with buffer_lock:
                    video_buffer.clear()
                    trajectory_buffer.clear()
                continue  # stay on same episode number, don't save

            if key == 'e' or session_done.is_set() or rospy.is_shutdown():
                print(f'\n  Saving episode {completed + 1} before ending...')
                last_saved_path = save_episode(completed, video_path, frame_ts_writer)
                completed += 1
                print(f'  Done. ({completed}/{num_episodes} saved)')
                session_done.set()
                break

            # Space — save episode
            print(f'  Stopped. Saving episode {completed + 1}...')
            last_saved_path = save_episode(completed, video_path, frame_ts_writer)
            completed += 1
            print(f'  Done. ({completed}/{num_episodes} completed)\n')

    session_done.set()
    print(f'\nAll done. {completed}/{num_episodes} episodes saved.')
