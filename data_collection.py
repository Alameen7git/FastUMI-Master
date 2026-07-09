import os
import json
import signal
import sys
import termios
import tty
import atexit
import queue
import shutil

# Optional — used for live CPU/memory display during recording and saving.
# Falls back to system load average (always available, zero extra cost) if
# psutil isn't installed, rather than breaking anything on its absence.
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import torch
import cv2
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

# ── Config ────────────────────────────────────────────────────────────────────
with open('config/config.json', 'r') as f:
    config = json.load(f)

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ROBOT_TYPE = config['device_settings']['robot_type']
cfg = config['task_config']

# How long (seconds) a feed can go silent during an active recording before
# it's treated as a dead node. Optional in config.json — defaults to 1.0s if
# not present, so this doesn't require touching existing config files.
NODE_TIMEOUT_S = cfg.get('node_timeout_s', 1.0)

# Content-based dead-feed detection, via frame-to-frame comparison: some
# HDMI capture cards (e.g. Elgato) keep publishing frames at full rate even
# when the source (GoPro) is unplugged or off — they just resend the same
# image over and over instead of stopping. Two confirmed failure modes on
# this actual hardware, both caught by the SAME check:
#   - GoPro battery dies -> Elgato repeats a solid black frame
#   - GoPro powered but cable pulled -> Elgato repeats a static "no signal"
#     card (has real internal texture — a single-frame brightness/variance
#     check alone does NOT catch this, confirmed by direct measurement)
# A real, connected camera never produces two pixel-identical frames in a
# row — sensor noise guarantees some difference every frame, even on a
# perfectly still scene. So instead of judging any one frame in isolation,
# this compares each frame to the previous one: if frames are repeating
# (near-zero difference) for a sustained stretch, the feed is dead
# regardless of what the repeated image looks like.
#
# DIFF_THRESH: below this frame-to-frame difference, two frames count as
# "the same." Validated in isolated testing (healthy footage, black
# screen, frozen card, flicker pattern, genuinely-still real scene) at
# 1.0 — comfortable headroom given real footage differences are typically
# much higher. Run check_temporal_diff.py against this specific rig to
# fine-tune if needed; not required before using this default.
TEMPORAL_DIFF_THRESH  = cfg.get('temporal_diff_thresh', 1.0)

# T265 equivalent of the above, applied to pose data instead of images.
# Measured on this actual rig with check_t265_temporal_diff.py: a healthy,
# connected T265 never reports the exact same pose twice — sensor/VIO
# noise guarantees drift even when physically still. Lowest observed real
# diff was ~0.000013; this threshold sits ~13x below that for safety
# margin, while staying far above the ~0.0 a genuinely frozen/dead T265
# would produce (repeating its last pose exactly). Note: this measures
# the healthy-noise floor directly, not an actual observed T265 failure —
# a true dead-T265 test hasn't been run on this rig, so the "frozen"
# assumption is inferred, not measured.
POSE_DIFF_THRESH = cfg.get('pose_diff_thresh', 0.000001)

# WINDOW/RATIO rather than "N identical frames in a row": a live incident
# showed a disconnected feed can flicker (occasional genuine frames mixed
# into an otherwise dead stream), which would reset any simple consecutive
# counter back to zero before it ever reached its threshold. Tracking the
# frozen-frame ratio over a rolling window is robust to that.
DIFF_WINDOW_FRAMES = cfg.get('diff_window_frames', 60)   # ~1s at 60Hz
DIFF_WINDOW_RATIO  = cfg.get('diff_window_ratio', 0.8)   # 80%+ frozen within the window -> flag

# Save-time backstop: after extraction, independently re-check the same
# frame-to-frame pattern among the frames actually about to be saved. A
# second layer — even if the live detector above ever has a gap, this
# catches it before the episode is written to disk rather than after.
SAVE_FROZEN_RATIO_WARN = cfg.get('save_frozen_ratio_warn', 0.5)

# Live CPU/memory display during recording and saving — helps confirm at a
# glance whether either phase is under real load, instead of only knowing
# after the fact from an external `top` snapshot. Can be turned off in
# config.json if it's ever unwanted; doesn't affect anything else if psutil
# isn't installed, it just falls back to system load average.
SHOW_RESOURCE_USAGE = cfg.get('show_resource_usage', True)
_this_process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
if HAS_PSUTIL:
    _this_process.cpu_percent(interval=None)  # prime — first call always returns 0.0, per psutil's own docs

def get_resource_line():
    if not SHOW_RESOURCE_USAGE:
        return ''
    if HAS_PSUTIL:
        cpu = _this_process.cpu_percent(interval=None)
        mem_mb = _this_process.memory_info().rss / (1024 * 1024)
        return f'CPU: {cpu:5.1f}%  MEM: {mem_mb:6.0f}MB'
    else:
        load1, _, _ = os.getloadavg()
        return f'system load: {load1:.2f} (install psutil for per-process CPU%)'

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='test')
parser.add_argument('--num_episodes', type=int, default=5)
args = parser.parse_args()
task = args.task
num_episodes = args.num_episodes

# ── Paths ─────────────────────────────────────────────────────────────────────
data_path = os.path.join(config['device_settings']['data_dir'], task)
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
is_recording      = False
first_frame_ts    = None
first_time_judger = False
start_time        = 0

# Node-health tracking: updated by the ROS callbacks every time a message
# arrives, regardless of recording state, so the watchdog always has a fresh
# baseline the moment recording starts.
last_video_msg_time = time()
last_traj_msg_time  = time()
node_watch_lock     = threading.Lock()

# Shared failure signal — set either by the timing-based watchdog thread
# (silence) or directly from video_callback (content-based frozen-feed
# check). Declared once at module level (not recreated per episode) so the
# callback can reach it directly; reset_node_health() clears it at the
# start of each new recording.
node_dead_flag    = threading.Event()
node_failure_info = {'source': None}
# Sliding window of recent frames' frozen/not-frozen classification, plus
# the previous frame's downsampled sample for comparison. All mutated
# under node_watch_lock (the same lock already used for last_video_msg_time)
# so reset_node_health() and video_callback can never interleave mid-update.
frozen_frame_window = deque(maxlen=DIFF_WINDOW_FRAMES)
prev_frame_sample    = None

# T265 equivalent of the two lines above, applied to pose instead of images.
frozen_pose_window = deque(maxlen=DIFF_WINDOW_FRAMES)
prev_pose_sample    = None

def reset_node_health():
    global prev_frame_sample, prev_pose_sample
    with node_watch_lock:
        frozen_frame_window.clear()
        prev_frame_sample = None
        frozen_pose_window.clear()
        prev_pose_sample = None
    node_dead_flag.clear()
    node_failure_info['source'] = None

# ── Keyboard control ──────────────────────────────────────────────────────────
# Keys during the "ready" prompt:        Space = start        E = end session
# Keys during an active recording:       Space = stop & save  R = discard take   E = stop, save, end session
# Keys during a node-failure pause:      C = restart this episode   E = end session
key_queue    = queue.Queue()
session_done = threading.Event()

def keyboard_listener():
    """Reads single keypresses without blocking the main thread."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    # BUG FIX: this thread's own `finally` block below is NOT a reliable way
    # to restore the terminal. It's a daemon thread that spends most of its
    # time blocked inside sys.stdin.read(1) — if the script exits (normally
    # or otherwise) while it's sitting in that blocking call, Python does
    # NOT wait for daemon threads before exiting, so this thread can be
    # killed mid-block, and its `finally` never runs. The terminal is then
    # left permanently in raw/cbreak mode after the script exits — which
    # looks exactly like "I can't type anything" in the next prompt, since
    # normal line-buffered keyboard input no longer works correctly.
    # atexit.register is guaranteed to run when the interpreter exits,
    # independent of what any thread is doing, so it's the reliable place
    # to put this — register it once, right after capturing the ORIGINAL
    # (pre-raw-mode) settings, before anything changes them.
    def _restore_terminal():
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
    atexit.register(_restore_terminal)

    try:
        tty.setcbreak(fd)
        while not session_done.is_set():
            # Interrupted reads (e.g. from the SIGINT handler firing mid-read)
            # just retry instead of potentially killing this daemon thread
            # silently, which would leave Space/R/E/C unresponsive with no
            # error shown.
            try:
                ch = sys.stdin.read(1)
            except (OSError, IOError):
                continue
            if ch in (' ', 'r', 'R', 'e', 'E', 'c', 'C'):
                key_queue.put(ch.lower())
    finally:
        _restore_terminal()  # best-effort immediate restore if this thread does get to exit cleanly

def sigint_handler(sig, frame):
    """
    Space means different things depending on state (start vs. stop), so
    Ctrl+C can't always mean "Space." Mid-recording it stops the current
    take (same as Space there); anywhere else (idle prompt, node-failure
    pause) it ends the session cleanly instead of risking an unintended
    action.
    """
    if is_recording:
        key_queue.put(' ')
    else:
        key_queue.put('e')

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

def drain_key_queue():
    """
    Discard any keypresses queued up during a previous phase (e.g. extra
    Space presses made while save_episode() was still running). Without
    this, a stale key sitting in the queue gets instantly consumed the
    moment the next wait_key() call starts — which is what caused
    recording to appear to start "on its own" right after a save finished.
    Call this immediately before every wait_key()/wait_key_or_node_failure()
    call that begins a new phase.
    """
    while True:
        try:
            key_queue.get_nowait()
        except queue.Empty:
            break

def wait_key_or_node_failure(valid, dead_flag, poll=0.1):
    """Same as wait_key, but also returns 'NODE_FAILURE' if the watchdog fires."""
    while not session_done.is_set() and not rospy.is_shutdown():
        if dead_flag.is_set():
            return 'NODE_FAILURE'
        try:
            k = key_queue.get(timeout=poll)
            if k in valid:
                return k
        except queue.Empty:
            continue
    return None

# ── ROS callbacks ─────────────────────────────────────────────────────────────
def video_callback(msg):
    global first_frame_ts, first_time_judger, last_video_msg_time, prev_frame_sample
    with node_watch_lock:
        last_video_msg_time = time()
    if not is_recording:
        return
    frame = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Content-based dead-feed check via frame-to-frame comparison: see the
    # comment on TEMPORAL_DIFF_THRESH above for why this replaced a
    # single-frame check (it missed a confirmed real failure mode — a
    # frozen "no signal" card with real internal texture). Sampling every
    # 20th pixel and converting to grayscale keeps this cheap enough to run
    # on every frame at 60Hz.
    sample = frame[::20, ::20].mean(axis=2)
    with node_watch_lock:
        if prev_frame_sample is not None:
            diff = np.abs(sample.astype(np.float32) - prev_frame_sample.astype(np.float32)).mean()
            is_frozen = diff < TEMPORAL_DIFF_THRESH
            frozen_frame_window.append(is_frozen)
        prev_frame_sample = sample
        window_full = len(frozen_frame_window) == frozen_frame_window.maxlen
        frozen_ratio = (sum(frozen_frame_window) / len(frozen_frame_window)) if frozen_frame_window else 0.0
    if window_full and frozen_ratio >= DIFF_WINDOW_RATIO and not node_dead_flag.is_set():
        node_failure_info['source'] = f'GoPro / video feed (frozen/repeated frames detected, {frozen_ratio*100:.0f}% of last {DIFF_WINDOW_FRAMES} frames)'
        node_dead_flag.set()

    timestamp = msg.header.stamp.to_sec()
    with buffer_lock:
        if start_time < timestamp:
            video_buffer.append((frame, timestamp))
            if first_time_judger:
                first_frame_ts = timestamp
                first_time_judger = False

def trajectory_callback(msg):
    global last_traj_msg_time, prev_pose_sample
    with node_watch_lock:
        last_traj_msg_time = time()
    if not is_recording:
        return
    p = msg.pose.pose
    pose = np.array([p.position.x, p.position.y, p.position.z,
                      p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])

    # T265 equivalent of the video frozen-feed check: a healthy T265 never
    # reports the exact same pose twice (sensor/VIO noise). See
    # POSE_DIFF_THRESH comment above for calibration details.
    with node_watch_lock:
        if prev_pose_sample is not None:
            pose_diff = np.linalg.norm(pose - prev_pose_sample)
            is_frozen_pose = pose_diff < POSE_DIFF_THRESH
            frozen_pose_window.append(is_frozen_pose)
        prev_pose_sample = pose
        pose_window_full = len(frozen_pose_window) == frozen_pose_window.maxlen
        frozen_pose_ratio = (sum(frozen_pose_window) / len(frozen_pose_window)) if frozen_pose_window else 0.0
    if pose_window_full and frozen_pose_ratio >= DIFF_WINDOW_RATIO and not node_dead_flag.is_set():
        node_failure_info['source'] = f'T265 / trajectory feed (frozen/repeated pose detected, {frozen_pose_ratio*100:.0f}% of last {DIFF_WINDOW_FRAMES} messages)'
        node_dead_flag.set()

    timestamp = msg.header.stamp.to_sec()
    with buffer_lock:
        if start_time < timestamp:
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

# ── Node-health watchdog ───────────────────────────────────────────────────────
def node_watchdog(stop_flag):
    """
    Runs only while a take is actively recording. Two independent checks feed
    the same shared node_dead_flag:
      1. Timing (here): if the GoPro or T265 topic goes silent for longer
         than NODE_TIMEOUT_S — genuine node/topic death.
      2. Content (in video_callback): if frames keep arriving on schedule but
         are suspiciously uniform — a capture card publishing blank "no
         signal" frames instead of stopping. Timing alone can't see that.
    Neither path attempts to restart anything — that's a manual step.
    """
    while not stop_flag.is_set():
        if node_dead_flag.is_set():
            return  # already flagged by the content-based check in video_callback
        now = time()
        with node_watch_lock:
            video_age = now - last_video_msg_time
            traj_age  = now - last_traj_msg_time
        if video_age > NODE_TIMEOUT_S:
            node_failure_info['source'] = 'GoPro / video feed (no messages received)'
            node_dead_flag.set()
            return
        if traj_age > NODE_TIMEOUT_S:
            node_failure_info['source'] = 'T265 / trajectory feed (no messages received)'
            node_dead_flag.set()
            return
        sleep(0.1)

# ── Status printer (shows live frame count while recording) ───────────────────
def status_printer(stop_flag):
    t0 = time()
    while not stop_flag.is_set():
        elapsed = time() - t0
        with buffer_lock:
            n = len(video_buffer)
        res = get_resource_line()
        res_part = f'   {res}' if res else ''
        print(f'\r  ● REC  {elapsed:5.1f}s   frames buffered: {n}{res_part}   (Space=stop  R=discard  E=end)', end='', flush=True)
        sleep(0.5)
    print()

def save_status_printer(stop_flag, label):
    """Same idea as status_printer, but for the save/processing phase — so a
    slow save (like the ~100s gzip compression we found earlier) shows live
    CPU/memory instead of just a silent wait with no feedback."""
    if not SHOW_RESOURCE_USAGE:
        return
    t0 = time()
    while not stop_flag.is_set():
        elapsed = time() - t0
        res = get_resource_line()
        if res:
            print(f'\r  ⚙ {label}  {elapsed:5.1f}s   {res}', end='', flush=True)
        sleep(1.0)
    print()

# ── Save episode to HDF5 (unchanged data format) ─────────────────────────────
def save_episode(episode_number, video_path, episode_start_time, frame_ts_writer):
    save_start_t = time()  # timing: overall save duration, printed at the end

    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(target=save_status_printer, args=(monitor_stop, 'Saving'))
    monitor_thread.start()
    def _stop_monitor():
        monitor_stop.set()
        monitor_thread.join()

    # Clear stale preview JPGs first: camera/images/ is one shared folder
    # across every episode. If this episode has fewer frames than whatever
    # was extracted last, old higher-numbered frames would otherwise survive
    # untouched. The real dataset (inside the .hdf5) isn't affected either
    # way — this only keeps the loose preview folder honest.
    for name in os.listdir(IMAGE_PATH):
        if name.endswith('.jpg'):
            os.remove(os.path.join(IMAGE_PATH, name))

    # Raw files are already complete on disk (written by write_video/write_trajectory
    # threads during recording). Just rename them to permanent per-episode names
    # and move on immediately -- conversion happens later via convert_episodes.py.
    raw_dir = os.path.join(data_path, 'raw', f'episode_{episode_number}')
    os.makedirs(raw_dir, exist_ok=True)

    final_video_path = os.path.join(raw_dir, 'video.mp4')
    final_traj_path  = os.path.join(raw_dir, 'trajectory.csv')
    final_ts_path    = os.path.join(raw_dir, 'timestamps.csv')

    shutil.move(video_path, final_video_path)
    shutil.move(TRAJECTORY_PATH_TEMP, final_traj_path)
    shutil.move(TIMESTAMP_PATH_TEMP, final_ts_path)

    frame_ts_writer.writerow([episode_number, first_frame_ts])

    print(f'  Episode {episode_number + 1} raw data saved -> {raw_dir}')
    print(f'  ({os.path.getsize(final_video_path) / 1e6:.1f} MB video)')
    _stop_monitor()
    return raw_dir

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rospy.Subscriber(cfg['ros']['video_topic'],      Image,    video_callback,      queue_size=cfg['ros']['queue_size'])
    rospy.Subscriber(cfg['ros']['trajectory_topic'], Odometry, trajectory_callback, queue_size=cfg['ros']['queue_size'])

    kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
    kb_thread.start()

    print(f'\nFastUMI  |  Robot: {ROBOT_TYPE}  |  Task: {task}  |  Episodes: {num_episodes}')
    print('Controls:  Space = start / stop take   R = discard current take (only while recording)   E = end session\n')

    completed = 0

    # Episode numbering: seeded once from disk at startup (same convention as
    # before, for compatibility with any existing dataset), then only ever
    # incremented. Redo can no longer delete a saved file, so there's nothing
    # that can desync this counter from what's actually on disk.
    next_episode_number = len([n for n in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, n))])

    with open(FRAME_TS_PATH, 'a', newline='') as frame_ts_file:
        frame_ts_writer = csv.writer(frame_ts_file)
        if os.path.getsize(FRAME_TS_PATH) == 0:
            frame_ts_writer.writerow(['Episode Index', 'Timestamp'])

        while completed < num_episodes:
            print(f'─── Episode {completed + 1}/{num_episodes}  ───')
            print('  Space = start   E = end session')

            drain_key_queue()  # discard anything left over from the previous phase
            key = wait_key([' ', 'e'])
            if key == 'e' or session_done.is_set() or rospy.is_shutdown():
                print('\nSession ended.')
                break

            # ── Start recording ──────────────────────────────────────────
            # reset_node_health() runs first, before is_recording flips to
            # True — otherwise there's a brief window where video_callback
            # could already be processing frames as "recording" while the
            # previous episode's leftover window/flag state hasn't been
            # cleared yet.
            reset_node_health()
            is_recording       = True
            start_time         = rospy.Time.now().to_sec()
            episode_start_time = start_time
            first_time_judger  = True
            video_path         = VIDEO_PATH_TEMP.replace('_n', f'_{next_episode_number}')
            video_writer       = cv2.VideoWriter(video_path, fourcc, 60, (frame_width, frame_height))
            done_flag          = threading.Event()
            stop_status        = threading.Event()

            with open(TRAJECTORY_PATH_TEMP, 'w', newline='') as traj_f, \
                 open(TIMESTAMP_PATH_TEMP,  'w', newline='') as ts_f:

                traj_w = csv.writer(traj_f)
                ts_w   = csv.writer(ts_f)
                traj_w.writerow(['Timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W'])
                ts_w.writerow(['Frame Index', 'Timestamp'])

                vt = threading.Thread(target=write_video_thread,      args=(video_writer, ts_w, done_flag))
                tt = threading.Thread(target=write_trajectory_thread, args=(traj_w, done_flag))
                st = threading.Thread(target=status_printer,          args=(stop_status,))
                wt = threading.Thread(target=node_watchdog,           args=(done_flag,))
                vt.start(); tt.start(); st.start(); wt.start()

                drain_key_queue()  # discard anything queued before recording actually began
                key = wait_key_or_node_failure([' ', 'r', 'e'], node_dead_flag)

                # Stop recording — let callbacks know immediately
                is_recording = False
                done_flag.set()
                vt.join(); tt.join(); wt.join()
                stop_status.set(); st.join()

            video_writer.release()

            if key == 'NODE_FAILURE':
                print(f"\n  ⚠  {node_failure_info['source']} — pausing.")
                print('  This take was discarded (nothing saved). Fix or restart the node')
                print('  manually, then press C to go back and re-record this episode.')
                print('  (Press E instead if you want to end the session.)')
                with buffer_lock:
                    video_buffer.clear()
                    trajectory_buffer.clear()
                drain_key_queue()  # discard anything queued while the failure message was printing
                resume_key = wait_key(['c', 'e'])
                if resume_key == 'e' or session_done.is_set() or rospy.is_shutdown():
                    print('\nSession ended.')
                    break
                print()
                continue  # back to the same episode's "ready" prompt

            if key == 'r':
                print('  ↺ Current take discarded.\n')
                with buffer_lock:
                    video_buffer.clear()
                    trajectory_buffer.clear()
                continue  # stay on same episode number, nothing was ever saved

            if key == 'e' or session_done.is_set() or rospy.is_shutdown():
                print(f'\n  Saving episode {completed + 1} before ending...')
                saved_path = save_episode(next_episode_number, video_path, episode_start_time, frame_ts_writer)
                if saved_path is not None:
                    next_episode_number += 1
                    completed += 1
                    print(f'  Done. ({completed}/{num_episodes} saved)')
                else:
                    print(f'  Episode not saved. ({completed}/{num_episodes} saved)')
                session_done.set()
                break

            # Space — save episode (final, no undo after this point, unless a
            # decode failure mid-save causes save_episode() to return None —
            # in that case nothing was written and this episode number is
            # re-recorded, same as a discarded take).
            print(f'  Stopped. Saving episode {completed + 1}...')
            saved_path = save_episode(next_episode_number, video_path, episode_start_time, frame_ts_writer)
            if saved_path is None:
                continue  # stay on same episode number, nothing was saved
            next_episode_number += 1
            completed += 1
            print(f'  Done. ({completed}/{num_episodes} completed)\n')

    session_done.set()
    print(f'\nAll done. {completed}/{num_episodes} episodes saved.')
