"""
FastUMI web UI — every command and path the UI runs lives here.

Each constant is tagged with where it came from:

  [CONFIRMED]  copied from an existing script in this repo (file named inline)
  [GUESS]      not verifiable from the repo — check it on the rig and edit

Edit freely; nothing else in webui/ hardcodes a command.
"""
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ── Web UI server ─────────────────────────────────────────────────────────────
UI_HOST = '127.0.0.1'
UI_PORT = int(os.environ.get('FASTUMI_UI_PORT', 5050))
RUN_DIR = os.path.join(REPO_ROOT, 'webui', '.run')   # pid/state files (gitignored)

# ── Environments ──────────────────────────────────────────────────────────────
# [CONFIRMED] start_collection.sh, replay_pipeline/make_replay_dataset.sh
ROS_SETUP   = '/opt/ros/noetic/setup.bash'


def _conda_base():
    """The conda install: $FASTUMI_CONDA_BASE, the active conda, or a usual place."""
    home = os.path.expanduser('~')
    exe = os.environ.get('CONDA_EXE', '')
    for base in (os.environ.get('FASTUMI_CONDA_BASE', ''),
                 os.path.dirname(os.path.dirname(exe)) if exe else '',
                 os.path.join(home, 'miniconda3'), os.path.join(home, 'anaconda3'),
                 os.path.join(home, 'miniforge3'), '/opt/conda'):
        if base and os.path.isfile(os.path.join(base, 'etc', 'profile.d', 'conda.sh')):
            return base
    return os.path.join(home, 'miniconda3')


CONDA_BASE  = _conda_base()   # /home/nuc8/miniconda3 on the rig
CONDA_SETUP = os.path.join(CONDA_BASE, 'etc', 'profile.d', 'conda.sh')
FASTUMI_ENV = 'FastUMI'   # MuJoCo / IK / OpenCV / data_collection.py
BASE_ENV    = 'base'      # LeRobot (Convert / Push)
FASTUMI_PY  = os.path.join(CONDA_BASE, 'envs', FASTUMI_ENV, 'bin', 'python3')   # [CONFIRMED] make_replay_dataset.sh

# Shell prefixes. Everything runs through `bash -c "<prefix> exec <cmd>"` so
# the process group we track is the real process, not a wrapper shell.
ROS_PREFIX     = 'source {ros} && '.format(ros=ROS_SETUP)
FASTUMI_PREFIX = 'source {conda} && conda activate {env} && source {ros} && '.format(
    conda=CONDA_SETUP, env=FASTUMI_ENV, ros=ROS_SETUP)

# ── Steps 1–3: ROS nodes ──────────────────────────────────────────────────────
# [CONFIRMED] start_collection.sh window [1/4]
ROS_CMD = 'roscore'
ROS_MASTER_PORT = 11311

# [CONFIRMED] start_collection.sh window [2/4]
T265_CMD = 'roslaunch realsense2_camera rs_t265.launch'

# [CONFIRMED] start_collection.sh window [3/4]. The launch file runs usb_cam
# (Elgato capture of the GoPro HDMI feed) plus an `image_view` node, which is
# the on-screen preview window that appears as soon as the node starts.
GOPRO_CMD = 'roslaunch {}'.format(os.path.join(REPO_ROOT, 'usb_cam-test.launch'))


def _load_config():
    with open(os.path.join(REPO_ROOT, 'config', 'config.json')) as f:
        return json.load(f)


CONFIG = _load_config()
DATA_DIR = CONFIG['device_settings']['data_dir']

# Readiness checks: a node is green only after one message actually arrives on
# its topic. [CONFIRMED] topic names come from config.json task_config.ros.
T265_TOPIC  = CONFIG['task_config']['ros']['trajectory_topic']   # /camera/odom/sample
GOPRO_TOPIC = CONFIG['task_config']['ros']['video_topic']        # /usb_cam/image_raw
TOPIC_CHECK_TIMEOUT_S = 4
# The camera is green only when the picture is live, not just when frames
# arrive: the Elgato capture card keeps sending frames with no GoPro attached.
# Thresholds are in webui/camera_check.py.
CAMERA_CHECK_SCRIPT = 'webui/camera_check.py'
# How long a node may take to become ready before it is marked red.
READY_TIMEOUT_S = {'ros': 15, 't265': 30, 'gopro': 30}

# ── Step 4: data collection ───────────────────────────────────────────────────
# [CONFIRMED] data_collection.py argparse: --task, --num_episodes.
# The script reads Space/R/E/C from its terminal via termios, so the UI runs it
# under a pseudo-terminal and forwards key presses from the browser.
COLLECT_SCRIPT = 'data_collection.py'
COLLECT_DEFAULT_EPISODES = 5
COLLECT_KEYS = {'space': ' ', 'r': 'r', 'e': 'e', 'c': 'c', 'x': 'x'}
# Graceful stop sends 'E' (the script saves any take in progress, then exits);
# after this many seconds it falls back to SIGTERM.
COLLECT_STOP_GRACE_S = 60

# Dataset folder names: UMI_<date>_T<task number>_<operator><number>, e.g.
# UMI_2026-09-23_T1_N1. "UMI" is fixed; the date is the day the
# recording starts; the number counts that operator's datasets for the task on
# that day (n1, n2, ...). On the Record page the operator adds to the latest
# one or starts the next number.
DATASET_NAME_FORMAT = 'UMI_{date}_T{task_id}_{operator}{num}'
DATASET_DATE_FORMAT = '%Y-%m-%d'


def _first_existing(*candidates):
    """Scripts moved between branches (webUI_test: repo root; ur7e-dev7.0:
    conversion/, viz/, replay_pipeline/). Use whichever exists."""
    for rel in candidates:
        if os.path.exists(os.path.join(REPO_ROOT, rel)):
            return rel
    return None


# ── Step 5: synchronize ───────────────────────────────────────────────────────
# [CONFIRMED] convert_episodes.py: for every raw/episode_N/ it keeps every 2nd
# video frame (60 -> 30 fps), matches each to the nearest T265 pose by timestamp and writes
# <task>/episode_N.hdf5. That HDF5 is what steps 6 and 7 read.
# Flags: --task, --parallel N, --fps N, --keep-raw (default of the script: DELETE raw).
# The camera records at 60 fps; --fps 30 keeps every 2nd frame.
SYNC_SCRIPT = _first_existing('conversion/convert_episodes.py', 'convert_episodes.py')
SYNC_DEFAULT_PARALLEL = 2
DATASET_FPS = 20   # frame rate of synced episodes and LeRobot datasets (60 fps camera / 3)
# Log of episodes deleted on the Delete page, kept in each dataset folder; it is
# copied into the LeRobot dataset and shown on the Hugging Face dataset card.
DELETED_EPISODES_FILE = 'deleted_episodes.json'

# ── Step 6: convert to LeRobot ────────────────────────────────────────────────
# webui/convert_to_lerobot.py (base env): synced <dataset>/episode_N.hdf5 ->
# LeRobot v3 folder <data_dir>/<dataset>/lerobot with the pose as state/action
# and the camera as video. Frames are stored BGR by convert_episodes.py and
# written RGB, resized to CONVERT_IMAGE_SIZE, H.264 (as in
# replay_pipeline/d1_to_lerobot_stage2.py on ur7e-dev7.0).
CONVERT_SCRIPT = 'webui/convert_to_lerobot.py'
CONVERT_IMAGE_SIZE = (640, 360)   # width, height (16:9 like the GoPro feed)
# Each dataset folder holds raw/ (the original recordings) and lerobot/ (the
# converted dataset). EXPORT_FILE (in the dataset folder) remembers the convert
# settings and the Hugging Face repo, so deleting episodes can redo both.
LEROBOT_SUBDIR = 'lerobot'
EXPORT_FILE = 'lerobot_export.json'

# Optional robot-replay output (joint angles via IK, UR7e), only when
# replay_pipeline/make_replay_dataset.sh is in the checkout (branch ur7e-dev7.0).
# [CONFIRMED] make_replay_dataset.sh --help.
REPLAY_SCRIPT = _first_existing('replay_pipeline/make_replay_dataset.sh')
REPLAY_SPACES = ['joint', 'cart-urbase', 'cart-baselink', 'cart-both', 'all']
CONVERT_SPACES = ['pose'] + (REPLAY_SPACES if REPLAY_SCRIPT else [])

# ── Step 7: visualize ─────────────────────────────────────────────────────────
# [CONFIRMED] visualize_dataset.py argparse: --port (default 8000), --no-browser.
# It browses every <data_dir>/<task>/episode_N.hdf5 (the step-5 output); it
# cannot read LeRobot folders.
VIZ_SCRIPT = _first_existing('viz/visualize_dataset.py', 'visualize_dataset.py')
VIZ_PORT = 8000
# After each Sync, encode that dataset's viewer videos in the background
# (visualize_dataset.py --prepare, niced) so View opens episodes instantly.
# Paused while recording so it can't take CPU from the recorder.
VIZ_PREPARE_AFTER_SYNC = True

# ── Task list (Record page) ───────────────────────────────────────────────────
# Operator-defined tasks, kept next to the data so they survive UI restarts.
TASKS_FILE = os.path.join(DATA_DIR, '.fastumi_tasks.json')

# ── Live camera preview (Record page) ─────────────────────────────────────────
# webui/camera_preview.py subscribes to the camera topic (ROS env), shrinks
# frames and serves them to the browser as MJPEG. It is an extra subscriber to
# the 1080p topic, so it only runs while the preview is open. Measured: ~45% of
# one core with a synthetic 30 fps 1080p rgb8 feed (mostly rospy receiving the
# 6 MB frames; a topic_tools throttle in front did not reduce the total).
PREVIEW_SCRIPT = 'webui/camera_preview.py'
PREVIEW_PORT = 8081
PREVIEW_TOPIC = GOPRO_TOPIC
PREVIEW_WIDTH = 640
PREVIEW_FPS = 10

# ── T265 home check (Record page) ─────────────────────────────────────────────
# webui/home_monitor.py follows the T265 pose and reports how far the UMI is
# from where the T265 started (its origin). Start the T265 with the UMI in the
# start holder: back in the holder it must read ~0, anything else is drift.
# It runs whenever the T265 is green and is restarted with it.
HOME_SCRIPT = 'webui/home_monitor.py'
HOME_PORT = 8082
HOME_OK_MM, HOME_OK_DEG = 20, 3        # green: at home
HOME_WARN_MM, HOME_WARN_DEG = 50, 6    # amber up to here, red beyond
# A step larger than this between two consecutive T265 poses (200 Hz) is the
# T265 snapping its estimate, not hand motion. T11_N1 (2026-11-03): hand motion
# <= ~17 mm/sample, jumps 239-357 mm. The panel stays red this long after one.
JUMP_MM_PER_SAMPLE = 30
JUMP_ALERT_S = 15

# ── Hugging Face Hub push ─────────────────────────────────────────────────────
# webui/push_to_hub.py takes the same flags as replay_pipeline/push_dataset_to_hub.py
# (--repo-id, --root, --public; default private) and also removes files the new
# copy no longer has, so a push after a delete leaves no old episodes on the Hub.
# Runs in the base env (LeRobot).
LEROBOT_PY = os.path.join(CONDA_BASE, 'bin', 'python3')   # [CONFIRMED] make_replay_dataset.sh
PUSH_SCRIPT = _first_existing('webui/push_to_hub.py', 'replay_pipeline/push_dataset_to_hub.py')
