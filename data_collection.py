import os
import sys
import tty
import termios
import json
import torch
import cv2
import h5py
import argparse
from tqdm import tqdm
from time import sleep
import numpy as np
import pyrealsense2 as rs
import apriltag
import rospy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
import csv
from scipy.spatial.transform import Rotation as R
import threading
from collections import deque
from datetime import datetime
import pandas as pd
import shutil
import time

# Load configuration from config.json
with open('config/config.json', 'r') as f:
    config = json.load(f)

# Set environment variables
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"

# Set device
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

ROBOT_TYPE = config['device_settings']["robot_type"]
TASK_CONFIG = config['task_config']


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="test3")
parser.add_argument('--num_episodes', type=int, default=2)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

cfg = TASK_CONFIG
robot = ROBOT_TYPE

data_path = os.path.join(config['device_settings']["data_dir"], "dataset" ,str(task))
os.makedirs(data_path, exist_ok=True)

IMAGE_PATH = os.path.join(data_path, 'camera/')
os.makedirs(IMAGE_PATH, exist_ok=True)

CSV_PATH = os.path.join(data_path, 'csv/')
os.makedirs(CSV_PATH, exist_ok=True)

STATE_PATH = os.path.join(data_path, 'states.csv')
if not os.path.exists(STATE_PATH):
    with open(STATE_PATH, 'w') as csv_file2:
        csv_writer2 = csv.writer(csv_file2)
        csv_writer2.writerow(['Index', 'Start Time', 'Trajectory Timestamp', 'Frame Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])

VIDEO_PATH_TEMP = os.path.join(data_path, 'camera', 'temp_video_n.mp4')
TRAJECTORY_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_trajectory.csv')
TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'temp_video_timestamps.csv')
FRAME_TIMESTAMP_PATH_TEMP = os.path.join(data_path, 'csv', 'frame_timestamps.csv')

video_subscriber = None
trajectory_subscriber = None

# Initialize ROS node
rospy.init_node('video_trajectory_recorder', anonymous=True)

# Video writer parameters for 60 Hz recording
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width, frame_height = cfg['cam_width'], cfg['cam_height']

# Buffers for storing incoming data
video_buffer = deque()
trajectory_buffer = deque()

# Lock for thread synchronization
buffer_lock = threading.Lock()

# Initialize CvBridge for image conversion
cv_bridge = CvBridge()

# Variable to store the first frame's timestamp
first_frame_timestamp = None

# Flag to stop recording when user presses E
stop_recording_flag = False

# ── Live dropout detection ────────────────────────────────────────────────────
# Wall-clock time of the last message received on each topic.
# Updated inside buffer_lock in each callback — consistent with buffer writes.
# Read inside write_video / write_trajectory every 500ms — negligible cost.
_last_frame_time = 0.0   # updated by video_callback
_last_traj_time  = 0.0   # updated by trajectory_callback

# If a dropout is detected mid-episode, the reason is stored here so the
# main loop can print it after recording stops. Empty string = no fault.
_episode_fault   = ""

# Timeouts — how long without a message before flagging a dropout.
# GoPro at 60fps: normal gap ~16ms. 1.5s = ~90 missed frames — unambiguous.
# T265 at 200Hz:  normal gap ~5ms.  0.5s = ~100 missed poses — unambiguous.
_CAMERA_DROPOUT_TIMEOUT = 1.5
_T265_DROPOUT_TIMEOUT   = 0.5

# Gap scan thresholds — used in post-recording validation.
# Any single gap larger than these in the timestamp CSVs = dropout detected.
_MAX_VIDEO_GAP_S = 0.10   # 100ms = ~6 missed frames at 60fps
_MAX_TRAJ_GAP_S  = 0.05   # 50ms  = ~10 missed poses at 200Hz
_MIN_EPISODE_S   = 2.0    # episodes shorter than this are rejected


def get_key():
    """Read a single keypress without requiring Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wait_for_key(target):
    """Block until target key is pressed."""
    while True:
        if get_key().lower() == target.lower():
            break


def wait_for_keys(targets):
    """Block until one of the target keys is pressed. Returns the matched key (lowercase)."""
    targets_lower = [t.lower() for t in targets]
    while True:
        key = get_key().lower()
        if key in targets_lower:
            return key


def keyboard_stop_listener():
    """Background thread — sets stop flag when spacebar is pressed."""
    global stop_recording_flag
    wait_for_key(' ')
    stop_recording_flag = True
    print("\nStopping recording...")


def check_disk_space(path, min_gb=10):
    """Warn if free disk space is below threshold. Returns free space in GB."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_gb:
        print(f"  [WARNING] Low disk space: {free_gb:.1f} GB free (minimum recommended: {min_gb} GB)")
    return free_gb


def check_topic_health(timeout=3.0):
    """
    Check both the T265 trajectory topic and the camera image topic are
    actively publishing before allowing an episode to start. Returns True
    if both are alive, False otherwise (with details printed to terminal).
    """
    print("\n  Checking topic health...")
    topics_ok = True

    checks = [
        (config['task_config']['ros']['trajectory_topic'], Odometry, "T265 trajectory"),
        (config['task_config']['ros']['video_topic'], Image, "Camera feed"),
    ]

    for topic_name, msg_type, label in checks:
        received = {"ok": False, "stamp": None}

        def _cb(msg, received=received):
            received["ok"] = True
            received["stamp"] = time.time()

        sub = rospy.Subscriber(topic_name, msg_type, _cb, queue_size=1)
        start = time.time()
        while time.time() - start < timeout and not received["ok"]:
            rospy.sleep(0.05)
        sub.unregister()

        if received["ok"]:
            print(f"    [OK]   {label} ({topic_name}) is publishing")
        else:
            print(f"    [DEAD] {label} ({topic_name}) — no messages received in {timeout}s")
            print(f"           Check: is the node running? Is the cable connected?")
            topics_ok = False

    return topics_ok


def restart_node(fault_reason):
    """
    Automatically restart the ROS node that caused the dropout.
    Called when operator presses R after a dropout alert.
    Returns True if the topic recovered, False if still dead after retrying.
    """
    import subprocess
    import signal as _signal

    ROS_SETUP = "source /opt/ros/noetic/setup.bash"

    # Determine which node to restart based on fault reason
    if "GoPro" in fault_reason or "Camera" in fault_reason:
        topic   = config['task_config']['ros']['video_topic']
        label   = "GoPro (usb_cam)"
        kill_cmd = "rosnode kill /usb_cam 2>/dev/null; sleep 1"
        launch_cmd = (
            f"bash -c '{ROS_SETUP} && "
            f"roslaunch /home/nuc8/Downloads/FastUMI-Master-ur7e-dev-2.0/usb_cam-test.launch' &"
        )
        msg_type = Image
        recover_timeout = 10.0

    elif "T265" in fault_reason:
        topic   = config['task_config']['ros']['trajectory_topic']
        label   = "T265 (realsense)"
        kill_cmd = "rosnode kill /camera/realsense2_camera_manager 2>/dev/null; sleep 1"
        launch_cmd = (
            f"bash -c '{ROS_SETUP} && "
            f"roslaunch realsense2_camera rs_t265.launch' &"
        )
        msg_type = Odometry
        recover_timeout = 12.0

    else:
        print("  Cannot auto-restart — unknown fault source. Restart nodes manually.")
        return False

    print(f"\n  Restarting {label}...")

    # Kill the existing node
    try:
        subprocess.run(kill_cmd, shell=True, timeout=5)
    except Exception:
        pass

    # Launch the node in background
    try:
        subprocess.Popen(launch_cmd, shell=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  Failed to launch node: {e}")
        return False

    # Wait for the topic to come back
    print(f"  Waiting for {label} to recover (up to {recover_timeout:.0f}s)...")
    recovered = {"ok": False}

    def _cb(msg, recovered=recovered):
        recovered["ok"] = True

    sub = rospy.Subscriber(topic, msg_type, _cb, queue_size=1)
    start = time.time()
    while time.time() - start < recover_timeout and not recovered["ok"]:
        elapsed = time.time() - start
        print(f"\r  Waiting... {elapsed:.1f}s", end='', flush=True)
        rospy.sleep(0.5)
    sub.unregister()
    print()

    if recovered["ok"]:
        print(f"  {label} recovered successfully.")
        return True
    else:
        print(f"  {label} did not recover in {recover_timeout:.0f}s.")
        print(f"  Check the cable and press R to try again.")
        return False


    """Delete the saved video and HDF5 file for a given episode index, plus its
    extracted jpg frames. Used by the redo feature. Safe to call even if some
    files don't exist."""
    removed = []

    if os.path.exists(video_path):
        os.remove(video_path)
        removed.append(video_path)

    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        removed.append(hdf5_path)

    if removed:
        print(f"  Discarded episode {episode_idx}: removed {len(removed)} file(s)")
    else:
        print(f"  Nothing to discard for episode {episode_idx} (no files found)")


# Callback for video frames (60 Hz expected)
def video_callback(msg):
    global first_frame_timestamp, first_time_judger, _last_frame_time
    frame = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    timestamp = msg.header.stamp.to_sec()

    with buffer_lock:
        _last_frame_time = time.time()   # wall-clock time of this frame
        if start_time < timestamp:
            video_buffer.append((frame, timestamp))
            if first_time_judger:
                first_frame_timestamp = timestamp
                first_time_judger = False

# Callback for trajectory data (e.g., T265 at 200 Hz)
def trajectory_callback(msg):
    global _last_traj_time
    timestamp = msg.header.stamp.to_sec()
    with buffer_lock:
        _last_traj_time = time.time()    # wall-clock time of this pose
        if start_time < timestamp:
            pose = msg.pose.pose
            trajectory_buffer.append((timestamp, pose.position.x, pose.position.y, pose.position.z,
                                      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))

# Thread for writing video frames and timestamps
def write_video():
    global stop_recording_flag, _episode_fault
    frame_index = 0
    pbar = tqdm(desc='Recording frames', unit='frame')
    _last_check = time.time()

    while not rospy.is_shutdown() and not stop_recording_flag:
        with buffer_lock:
            if video_buffer:
                frame, timestamp = video_buffer.popleft()
                video_writer.write(frame)
                timestamp_writer.writerow([frame_index, timestamp])
                frame_index += 1
                pbar.update(1)

        # Dropout check every 500ms — negligible CPU cost.
        # Only check after recording has actually started (_last_frame_time > 0).
        now = time.time()
        if now - _last_check >= 0.5:
            _last_check = now
            with buffer_lock:
                gap = now - _last_frame_time if _last_frame_time > 0 else 0
            if gap > _CAMERA_DROPOUT_TIMEOUT:
                _episode_fault = f"GoPro dropped — no frame for {gap:.1f}s"
                stop_recording_flag = True
                break

        sleep(0.001)

    # Drain remaining buffer after stop
    with buffer_lock:
        while video_buffer:
            frame, timestamp = video_buffer.popleft()
            video_writer.write(frame)
            timestamp_writer.writerow([frame_index, timestamp])
            frame_index += 1

    pbar.close()
    print(f"Video done: {frame_index} frames")


# Thread for writing trajectory data to CSV
def write_trajectory():
    global stop_recording_flag, _episode_fault
    counter = 0
    _last_check = time.time()

    while not rospy.is_shutdown() and not stop_recording_flag:
        with buffer_lock:
            if trajectory_buffer:
                Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W = trajectory_buffer.popleft()
                trajectory_writer.writerow([Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W])
                counter += 1

        # T265 dropout check every 500ms
        now = time.time()
        if now - _last_check >= 0.5:
            _last_check = now
            with buffer_lock:
                gap = now - _last_traj_time if _last_traj_time > 0 else 0
            if gap > _T265_DROPOUT_TIMEOUT:
                _episode_fault = f"T265 dropped — no pose for {gap:.1f}s"
                stop_recording_flag = True
                break

        sleep(0.001)

    # Drain remaining buffer after stop
    with buffer_lock:
        while trajectory_buffer:
            Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W = trajectory_buffer.popleft()
            trajectory_writer.writerow([Timestamp, PosX, PosY, PosZ, Q_X, Q_Y, Q_Z, Q_W])
            counter += 1

    print(f"Trajectory done: {counter} samples")


def validate_episode(ts_path, traj_path, elapsed_time):
    """
    Scan the episode's timestamp CSVs for gaps that indicate a dropout.
    Called after recording stops, before HDF5 is written.
    Takes <100ms. Returns (True, None) if clean, (False, reason) if faulty.
    This is the second line of defence — catches anything the live check
    missed, e.g. a very brief glitch that recovered before the 500ms check.
    """
    try:
        # ── Video timestamps ──────────────────────────────────────────────
        ts = pd.read_csv(ts_path)
        if len(ts) < 2:
            return False, f"Too few frames: {len(ts)} (episode may not have started)"

        if elapsed_time < _MIN_EPISODE_S:
            return False, f"Episode too short: {elapsed_time:.1f}s (minimum {_MIN_EPISODE_S}s)"

        gaps = ts['Timestamp'].diff().dropna()
        max_gap = gaps.max()
        if max_gap > _MAX_VIDEO_GAP_S:
            worst_frame = int(gaps.idxmax())
            return False, f"Camera gap: {max_gap:.2f}s at frame {worst_frame}"

        # ── Trajectory timestamps ─────────────────────────────────────────
        traj = pd.read_csv(traj_path)
        if len(traj) < 10:
            return False, f"Too few trajectory samples: {len(traj)}"

        traj_gaps = traj['Timestamp'].diff().dropna()
        max_traj_gap = traj_gaps.max()
        if max_traj_gap > _MAX_TRAJ_GAP_S:
            return False, f"T265 gap: {max_traj_gap:.2f}s"

        return True, None

    except Exception as e:
        # If the scan itself fails (e.g. empty file), treat as faulty
        return False, f"Validation error: {e}"


# Main function to start recording
def start_recording():
    global stop_recording_flag, _episode_fault, _last_frame_time, _last_traj_time
    stop_recording_flag = False
    _episode_fault = ""          # clear any fault from previous episode
    _last_frame_time = 0.0       # reset so dropout check doesn't fire before
    _last_traj_time  = 0.0       # recording actually begins
    video_buffer.clear()
    trajectory_buffer.clear()

    video_thread = threading.Thread(target=write_video)
    trajectory_thread = threading.Thread(target=write_trajectory)
    keyboard_thread = threading.Thread(target=keyboard_stop_listener, daemon=True)

    video_thread.start()
    trajectory_thread.start()
    keyboard_thread.start()

    video_thread.join()
    trajectory_thread.join()

if __name__ == "__main__":
    start_time = 0
    first_time_judger = False
    first_frame_timestamp = None
    cv_bridge = CvBridge()
    video_subscriber = rospy.Subscriber(config['task_config']['ros']['video_topic'], Image, video_callback, queue_size=config['task_config']['ros']['queue_size'])
    trajectory_subscriber = rospy.Subscriber(config['task_config']['ros']['trajectory_topic'], Odometry, trajectory_callback, queue_size=config['task_config']['ros']['queue_size'])

    with open(FRAME_TIMESTAMP_PATH_TEMP, "a", newline='') as frame_timestamp_file:
        frame_timestamp_writer = csv.writer(frame_timestamp_file)
        frame_timestamp_writer.writerow(['Episode Index', 'Timestamp'])

        episode = 0

        while episode < num_episodes:
            print(f"\nEpisode {episode + 1}/{num_episodes} ready.")

            # Health checks run before any files are created for this episode
            while True:
                free_gb = check_disk_space(data_path, min_gb=10)
                topics_alive = check_topic_health(timeout=3.0)
                if topics_alive:
                    break
                print("\n  [BLOCKED] Cannot start — one or more topics are not publishing.")
                print("  Fix the issue above, then press R to retry, or Ctrl+C to quit.")
                wait_for_key('r')

            video_path = VIDEO_PATH_TEMP.replace("_n", f"_{episode}")
            video_writer = cv2.VideoWriter(video_path, fourcc, 60, (frame_width, frame_height))

            with open(TRAJECTORY_PATH_TEMP, 'w', newline='') as trajectory_file, \
                 open(TIMESTAMP_PATH_TEMP, 'w', newline='') as timestamp_file:

                trajectory_writer = csv.writer(trajectory_file)
                timestamp_writer = csv.writer(timestamp_file)

                trajectory_writer.writerow(['Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])
                timestamp_writer.writerow(['Frame Index', 'Timestamp'])

                first_time_judger = False

                redo_prompt = "  Press X to redo episode (instead of starting)" if episode > 0 else ""
                print(f"\n  All checks passed ({free_gb:.1f} GB free).")
                print(f"  Press SPACE to start recording.{(' ' + redo_prompt) if redo_prompt else ''}")

                if episode > 0:
                    key = wait_for_keys([' ', 'x'])
                else:
                    key = wait_for_keys([' '])

                if key == 'x':
                    # Discard the previous episode and re-record it at the same index
                    discard_episode_files(
                        episode - 1,
                        video_path=VIDEO_PATH_TEMP.replace("_n", f"_{episode - 1}"),
                        hdf5_path=os.path.join(data_path, f'episode_{episode - 1}.hdf5'),
                    )
                    episode -= 1
                    video_writer.release()
                    continue  # restart loop at the lower episode index

                start_time = rospy.Time.now().to_sec()
                first_time_judger = True
                print(f"Recording! Press SPACE to stop.")

                ep_start_wall = time.time()

                try:
                    start_recording()
                except Exception as e:
                    print(f"An error occurred: {e}")
                    raise
                finally:
                    elapsed_time = time.time() - ep_start_wall
                    video_writer.release()

                # ── Live dropout check ────────────────────────────────────
                if _episode_fault:
                    print(f"\n  ⚠  Episode {episode + 1} FAULT DETECTED")
                    print(f"     {_episode_fault}")
                    print(f"     Data not saved.\n")
                    print(f"     R = auto-restart the affected node and redo")
                    print(f"     E = end session")
                    key = wait_for_keys(['r', 'e'])
                    for p in [video_path, TRAJECTORY_PATH_TEMP, TIMESTAMP_PATH_TEMP]:
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except OSError:
                            pass
                    if key == 'e':
                        print("\nSession ended.")
                        break
                    # Auto-restart the node — if it fails, the pre-episode
                    # health check will block and tell the operator what to do
                    restart_node(_episode_fault)
                    continue

                # ── Post-recording gap scan ───────────────────────────────
                # Flush CSV writers before reading back
                timestamp_file.flush()
                trajectory_file.flush()

                print("  Validating episode data...")
                valid, reason = validate_episode(
                    TIMESTAMP_PATH_TEMP,
                    TRAJECTORY_PATH_TEMP,
                    elapsed_time
                )

                if not valid:
                    print(f"\n  ⚠  Episode {episode + 1} FAILED VALIDATION")
                    print(f"     {reason}")
                    print(f"     Data not saved.\n")
                    print(f"     R = redo this episode")
                    print(f"     E = end session")
                    key = wait_for_keys(['r', 'e'])
                    for p in [video_path, TRAJECTORY_PATH_TEMP, TIMESTAMP_PATH_TEMP]:
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except OSError:
                            pass
                    if key == 'e':
                        print("\nSession ended.")
                        break
                    continue

                # ── Save to HDF5 (atomic write) ───────────────────────────
                frame_timestamp_writer.writerow([episode, first_frame_timestamp])

                data_dict = {
                    '/observations/qpos': [],
                    '/action': [],
                }
                for cam_name in cfg['camera_names']:
                    data_dict[f'/observations/images/{cam_name}'] = []

                timestamps = pd.read_csv(TIMESTAMP_PATH_TEMP)
                downsampled_timestamps = timestamps.iloc[::3].reset_index(drop=True)
                cap = cv2.VideoCapture(video_path)

                for idx, row in tqdm(downsampled_timestamps.iterrows(), desc='Extracting Images'):
                    frame_idx = row['Frame Index']
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        filename = f"{int(frame_idx / 3)}.jpg"
                        cv2.imwrite(os.path.join(IMAGE_PATH, filename), frame)
                        for cam_name in cfg['camera_names']:
                            data_dict[f'/observations/images/{cam_name}'].append(frame)

                cap.release()

                trajectory = pd.read_csv(TRAJECTORY_PATH_TEMP)
                trajectory['Timestamp'] = trajectory['Timestamp'].astype(float)

                for idx, row in tqdm(downsampled_timestamps.iterrows(), desc='Extracting States'):
                    closest_idx = (np.abs(trajectory['Timestamp'] - row['Timestamp'])).argmin()
                    closest_row = trajectory.iloc[closest_idx]
                    pos_quat = [
                        closest_row['Pos X'], closest_row['Pos Y'], closest_row['Pos Z'],
                        closest_row['Q_X'], closest_row['Q_Y'], closest_row['Q_Z'], closest_row['Q_W']
                    ]
                    data_dict['/observations/qpos'].append(pos_quat)
                    data_dict['/action'].append(pos_quat)
                    with open(STATE_PATH, 'a', newline='') as csv_file2:
                        csv_writer2 = csv.writer(csv_file2)
                        csv_writer2.writerow([idx, start_time, closest_row['Timestamp'], row['Timestamp']] + pos_quat)

                max_timesteps = len(data_dict['/observations/qpos'])
                dataset_path = os.path.join(data_path, f'episode_{episode}.hdf5')
                tmp_path = dataset_path + '.tmp'   # atomic write — rename on success
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

                with h5py.File(tmp_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
                    root.attrs['sim'] = False
                    obs = root.create_group('observations')
                    image_grp = obs.create_group('images')
                    for cam_name in cfg['camera_names']:
                        image_grp.create_dataset(
                            cam_name,
                            data=np.array(data_dict[f'/observations/images/{cam_name}'], dtype=np.uint8),
                            compression='gzip',
                            compression_opts=4
                        )
                    root.create_dataset('observations/qpos', data=np.array(data_dict['/observations/qpos']))
                    root.create_dataset('action', data=np.array(data_dict['/action']))

                # Atomic rename — only appears as final file when fully written
                os.rename(tmp_path, dataset_path)
                print(f"  Episode {episode + 1} saved -> {dataset_path}")

            episode += 1

    print("All episodes completed successfully!")
