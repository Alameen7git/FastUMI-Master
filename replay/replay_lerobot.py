#!/usr/bin/env python3
"""
Replay one episode of a LeRobot *joint* dataset (local folder or Hugging Face
repo, e.g. the output of make_replay_dataset.sh --space joint) on the UR7e.

The joint angles are already IK solutions, so this skips
validate_joint_trajectory.py: it takes `action[:, :6]` (radians), checks it
against the URDF joint position limits, writes it in the validated-trajectory
format replay_ur7e.py reads, and hands it to replay_ur7e.run() -- the same
spline retiming, --speed slowdown, typed RUN / CONFIRM_FAST gates and dry-run
default as replay.py. The gripper column is not replayed.

Usage (dry run -- no hardware):
    python3 replay/replay_lerobot.py Nityavedula/UMI_2026-11-05_T0_a1_ur7e_joint --episode 0

Usage (read the robot only -- connects, never moves it):
    python3 replay/replay_lerobot.py Nityavedula/UMI_2026-11-05_T0_a1_ur7e_joint --episode 0 \\
        --check --robot-ip 192.168.1.102

Usage (real hardware, once you've reviewed the dry run):
    python3 replay/replay_lerobot.py Nityavedula/UMI_2026-11-05_T0_a1_ur7e_joint --episode 0 \\
        --speed 0.1 --execute --robot-ip 192.168.1.102
"""
import argparse
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'replay'))
import replay_ur7e as rur


# Reading parquet and downloading from the Hub need pyarrow + huggingface_hub,
# which live in the conda base env with LeRobot; the FastUMI env has ur_rtde but
# neither of those. So this runs as a separate process and writes the needed
# columns to an npz.
_EXPORT = r'''
import glob, os, sys
import numpy as np, pandas as pd
dataset, out = sys.argv[1], sys.argv[2]
if os.path.isdir(dataset):
    root = dataset
else:
    from huggingface_hub import snapshot_download
    root = snapshot_download(dataset, repo_type='dataset', allow_patterns=['data/*', 'meta/*'])
files = sorted(glob.glob(os.path.join(root, 'data', '*', '*.parquet')))
if not files:
    sys.exit('No data/*/*.parquet in {}'.format(root))
d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
np.savez(out, episode_index=d['episode_index'].values, frame_index=d['frame_index'].values,
         timestamp=d['timestamp'].values, action=np.stack(d['action'].values))
'''


def load_frames(dataset, out_dir):
    """episode_index, frame_index, timestamp, action of every row, from a local LeRobot folder or a Hub repo id."""
    sys.path.insert(0, _ROOT)
    from webui import settings
    npz = os.path.join(out_dir, '_lerobot_frames.npz')
    if subprocess.call([settings.LEROBOT_PY, '-c', _EXPORT, dataset, npz]) != 0:
        sys.exit('Could not read {} with {}'.format(dataset, settings.LEROBOT_PY))
    with np.load(npz) as z:
        frames = {k: z[k] for k in z.files}
    os.remove(npz)
    return frames


# rtde_receive codes (UR RTDE guide)
ROBOT_MODES = {-1: 'no controller', 0: 'disconnected', 1: 'confirm safety', 2: 'booting', 3: 'power off',
               4: 'power on', 5: 'idle', 6: 'backdrive', 7: 'running', 8: 'updating firmware'}
SAFETY_MODES = {1: 'normal', 2: 'reduced', 3: 'protective stop', 4: 'recovery', 5: 'safeguard stop',
                6: 'system emergency stop', 7: 'robot emergency stop', 8: 'violation', 9: 'fault',
                10: 'validate joint id', 11: 'undefined', 12: 'automatic mode safeguard stop',
                13: 'system three position enabling stop'}


def check_robot(robot_ip, start_q):
    """Read-only: connect with rtde_receive (never moves the arm) and compare
    where the arm is with where the episode starts."""
    import rtde_receive
    print('Connecting (read-only) to {} ...'.format(robot_ip))
    try:
        r = rtde_receive.RTDEReceiveInterface(robot_ip)
    except RuntimeError as e:
        sys.exit('Could not connect to the robot at {}: {}'.format(robot_ip, e))
    cur = np.array(r.getActualQ())
    mode, safety = r.getRobotMode(), r.getSafetyMode()
    r.disconnect()
    print('Robot mode: {}   Safety: {}'.format(ROBOT_MODES.get(mode, mode), SAFETY_MODES.get(safety, safety)))
    print('Current robot joints (deg): {}'.format(np.degrees(cur).round(1).tolist()))
    print('Episode start (deg): {}'.format(np.degrees(start_q).round(1).tolist()))
    print('Largest joint difference: {:.1f} deg'.format(np.degrees(np.abs(start_q - cur)).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help='LeRobot joint dataset: local folder or Hub repo id (user/name)')
    parser.add_argument('--episode', type=int, required=True, help='episode_index in the LeRobot dataset (0-based)')
    parser.add_argument('--speed', type=float, default=0.1,
                        help='fraction of the safety-retimed speed (default 0.1 = 10%%); above 0.25 needs "CONFIRM_FAST"')
    parser.add_argument('--execute', action='store_true', help='connect to the real robot and move it (default: dry run)')
    parser.add_argument('--robot-ip', default=None, help='overrides config.json ur7e_hardware.robot_ip')
    parser.add_argument('--check', action='store_true',
                        help='only read the robot (read-only connection, no motion): mode, joints, distance to the start')
    parser.add_argument('--yes', action='store_true', help='skip the typed "RUN" confirmation (never skips CONFIRM_FAST)')
    parser.add_argument('--force', action='store_true', help='proceed even if a waypoint is outside the joint limits')
    parser.add_argument('--config', default=os.path.join(_ROOT, 'config', 'config.json'))
    parser.add_argument('--replay-dir', default=os.path.join(_ROOT, 'outputs', 'replay'))
    args = parser.parse_args()

    os.makedirs(args.replay_dir, exist_ok=True)
    frames = load_frames(args.dataset, args.replay_dir)
    rows = np.flatnonzero(frames['episode_index'] == args.episode)
    if not len(rows):
        sys.exit('Episode {} not found; the dataset has episodes {}'.format(
            args.episode, np.unique(frames['episode_index']).tolist()))
    rows = rows[np.argsort(frames['frame_index'][rows])]
    q = frames['action'][rows, :6].astype(np.float64)
    dt_s = np.diff(frames['timestamp'][rows].astype(np.float64))
    if len(q) < 2 or (dt_s <= 0).any():
        sys.exit('Episode {} has too few frames or non-increasing timestamps'.format(args.episode))

    with open(args.config) as f:
        urdf = json.load(f)['data_process_config']['urdf_path']
    if not os.path.isabs(urdf):
        urdf = os.path.join(_ROOT, urdf)
    pos_lo, pos_hi, _ = rur.load_urdf_joint_limits(urdf)
    over = (q < pos_lo) | (q > pos_hi)
    print('{} episode {}: {} frames, {:.1f}s, {} waypoint(s) outside joint limits'.format(
        args.dataset, args.episode, len(q), dt_s.sum(), int(over.any(axis=1).sum())))

    if args.check:
        with open(args.config) as f:
            robot_ip = args.robot_ip or json.load(f).get('ur7e_hardware', {}).get('robot_ip')
        if not robot_ip or robot_ip == 'TODO_FILL_IN_ROBOT_IP':
            sys.exit('No robot IP set. Pass --robot-ip <ip>.')
        check_robot(robot_ip, q[0])
        return

    name = os.path.basename(args.dataset.rstrip('/'))
    validated = os.path.join(args.replay_dir, '{}_episode_{}.hdf5'.format(name, args.episode))
    with h5py.File(validated, 'w') as f:
        f['joint_trajectory'] = q
        f['dt_s'] = dt_s
        f['diagnostics/reachable'] = np.ones(len(q), dtype=bool)
        f['violations/over_position_limit'] = over

    # replay_ur7e resolves the URDF path relative to the working directory
    os.chdir(_ROOT)
    rur.run(SimpleNamespace(validated_hdf5=validated, config=os.path.abspath(args.config),
                            execute=args.execute, robot_ip=args.robot_ip, yes=args.yes, force=args.force,
                            speed_scale=args.speed, output_dir=args.replay_dir))


if __name__ == '__main__':
    main()
