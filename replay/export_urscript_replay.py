#!/usr/bin/env python3
"""
Export ONE validated UR7e joint trajectory as a self-contained URScript
(.script) file for native pendant playback -- no ur_rtde, no network
connection, no running PC. Copy the file to a USB stick, load it in
PolyScope, and the controller plays the whole trajectory on its own.

Input is the output of validate_joint_trajectory.py (already IK-converted
from the recorded TCP pose and checked for reachability / joint limits), so
this tool does NOT re-run kinematics -- it only formats the already-validated
joint waypoints into URScript.

Same safety philosophy as replay_ur7e.py:
  - Refuses to export a trajectory with unresolved IK-reachability or
    joint-position-limit violations (pass --force only after reviewing
    validate_joint_trajectory.py's report yourself).
  - --speed-scale (default 0.1 = play back 10x slower than recorded) paces
    the servoj loop. Unlike replay_ur7e.py there is NO software velocity/
    acceleration clamp on the pendant, so smoothness relies entirely on the
    trajectory already being validated AND played slowly -- a conservative
    default matters more here, not less.

Usage:
    python3 export_urscript_replay.py outputs/validated_trajectories/episode_10.hdf5
    python3 export_urscript_replay.py outputs/validated_trajectories/episode_10.hdf5 \\
        --speed-scale 0.1 --task Pick_and_place_the_bottle
"""
import argparse
import json
import os
import sys
from datetime import datetime

import h5py
import numpy as np

# servoj tuning for pendant playback. t is the servo control/blocking time
# per call; lookahead_time and gain smooth the tracking. These match the
# ur7e_hardware block in config.json (lookahead/gain are read from there;
# SERVOJ_T is pendant-specific and not part of the ur_rtde streaming config).
SERVOJ_T = 0.1
JOINT_ORDER = ('shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3')


def format_q(q):
    """6-vector of radians -> URScript list literal '[v0, v1, ...]'."""
    return '[' + ', '.join(f'{v:.6f}' for v in q) + ']'


def build_urscript(joint_traj, step_sleep, sleeps, header_lines,
                   startup_v, startup_a, servoj_t, lookahead_time, gain):
    """Assemble the full self-contained .script text."""
    n = joint_traj.shape[0]
    uniform = sleeps is None  # scalar step_sleep vs per-waypoint list

    header = '\n'.join('# ' + line for line in header_lines)

    rows = []
    for i in range(n):
        sep = ',' if i < n - 1 else ''
        rows.append('  ' + format_q(joint_traj[i]) + sep)
    waypoints_block = 'waypoints = [\n' + '\n'.join(rows) + '\n]'

    start_pose = format_q(joint_traj[0])

    if uniform:
        sleep_setup = f'  step_sleep = {step_sleep:.6f}'
        sleep_call = '    sleep(step_sleep)'
    else:
        sleep_rows = []
        for i in range(n):
            sep = ',' if i < n - 1 else ''
            sleep_rows.append(f'  {sleeps[i]:.6f}{sep}')
        sleep_setup = 'sleeps = [\n' + '\n'.join(sleep_rows) + '\n]'
        sleep_call = '    sleep(sleeps[i])'

    return f"""{header}

def fastumi_replay():
  textmsg("FastUMI replay: easing to trajectory start pose")
  # Slow, conservative approach to the first waypoint so the arm never jumps
  # to the trajectory start -- it eases in under movej's own trapezoidal profile.
  movej({start_pose}, a={startup_a}, v={startup_v})

  textmsg("FastUMI replay: starting servoj playback")
{sleep_setup}
  {waypoints_block}

  n = {n}
  i = 0
  while i < n:
    # a, v (2nd/3rd args) are unused by servoj and passed as 0 per the URScript
    # spec; t/lookahead_time/gain control the servo tracking.
    servoj(waypoints[i], 0, 0, t={servoj_t}, lookahead_time={lookahead_time}, gain={gain})
{sleep_call}
    i = i + 1
  end

  stopj(2.0)
  textmsg("FastUMI replay: done")
end

fastumi_replay()
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('validated_hdf5', help='output of validate_joint_trajectory.py (single episode)')
    parser.add_argument('--speed-scale', type=float, default=0.1,
                         help='playback speed as a fraction of recorded (default 0.1 = 10x slower)')
    parser.add_argument('--config', default='config/config.json')
    parser.add_argument('--task', default=None, help='task name to record in the header (not stored in the validated file)')
    parser.add_argument('--servoj-t', type=float, default=SERVOJ_T, help=f'servoj control time t (default {SERVOJ_T})')
    parser.add_argument('--output', default=None, help='output .script path (default: outputs/urscript/<episode>_replay.script)')
    parser.add_argument('--force', action='store_true',
                         help='export even if the validated file has unresolved reachability/position violations')
    args = parser.parse_args()

    if args.speed_scale <= 0:
        print('ERROR: --speed-scale must be > 0.')
        sys.exit(1)

    with open(args.config) as f:
        hw = json.load(f).get('ur7e_hardware', {})
    startup_v = hw.get('startup_move_speed_rad_s', 0.03)
    startup_a = hw.get('startup_move_accel_rad_s2', 0.05)
    lookahead_time = hw.get('servo_lookahead_time_s', 0.2)
    gain = hw.get('servo_gain', 100)

    try:
        with h5py.File(args.validated_hdf5, 'r') as f:
            if 'joint_trajectory' not in f:
                print(f"'{args.validated_hdf5}' has no 'joint_trajectory' -- this is not a validated file.")
                print('Run validate_joint_trajectory.py on the raw episode first.')
                sys.exit(1)
            joint_traj = f['joint_trajectory'][:]
            dt_s = f['dt_s'][:]
            reachable = f['diagnostics/reachable'][:]
            over_pos_limit = f['violations/over_position_limit'][:]
    except OSError as e:
        print(f'Could not read {args.validated_hdf5}: {e}')
        sys.exit(1)

    hard_violation = bool((~reachable).any() or over_pos_limit.any())
    if hard_violation and not args.force:
        print('This validated trajectory has unresolved IK-reachability or joint-position-limit '
              'violations (see validate_joint_trajectory.py). Refusing to export.')
        print('Re-check the recording/calibration, or pass --force if you have reviewed and accept it.')
        sys.exit(1)

    n = joint_traj.shape[0]
    # Pace the servoj loop: sleep the original inter-waypoint dt, stretched by
    # 1/speed_scale. dt_s has n-1 intervals; sleep after the last waypoint
    # reuses the final interval (harmless). If dt is uniform (it is, when
    # measured from timestamps.csv) emit a single constant instead of a list.
    dt_per_wp = np.append(dt_s, dt_s[-1]) if len(dt_s) else np.full(n, 1.0 / 20.0)
    sleeps_full = dt_per_wp / args.speed_scale
    uniform = bool(np.allclose(sleeps_full, sleeps_full[0]))
    step_sleep = float(sleeps_full[0])
    sleeps = None if uniform else sleeps_full

    recorded_duration = float(dt_s.sum()) if len(dt_s) else 0.0
    playback_duration = recorded_duration / args.speed_scale

    stem = os.path.splitext(os.path.basename(args.validated_hdf5))[0]  # e.g. 'episode_10'
    episode_idx = stem.split('_')[1] if '_' in stem else '?'
    out_path = args.output or os.path.join('outputs', 'urscript', f'{stem}_replay.script')
    timestamp = datetime.now().isoformat(timespec='seconds')

    # ----- report before writing -----
    print(f'Episode:            {stem} (index {episode_idx})' +
          (f'   task: {args.task}' if args.task else '   task: (not recorded -- pass --task to embed)'))
    print(f'Source HDF5:        {os.path.abspath(args.validated_hdf5)}')
    print(f'Waypoints:          {n}')
    print(f'Speed scale:        x{args.speed_scale:.3f}  ({1.0 / args.speed_scale:.1f}x slower than recorded)')
    print(f'Recorded duration:  {recorded_duration:.2f} s')
    print(f'Est. playback:      {playback_duration:.2f} s  (sum of stretched sleeps; servoj execution adds a little)')
    print(f'servoj:             t={args.servoj_t}, lookahead_time={lookahead_time}, gain={gain}')
    print(f'per-waypoint sleep: {step_sleep:.4f} s' + ('' if uniform else ' (varies -- per-waypoint list embedded)'))
    print(f'Startup movej:      a={startup_a} rad/s^2, v={startup_v} rad/s (eases into start pose)')
    if hard_violation:
        print('WARNING: exporting despite validation violations (--force).')

    header_lines = [
        '=' * 60,
        'FastUMI -> UR7e servoj replay (native pendant playback)',
        '=' * 60,
        f'Generated:        {timestamp}',
        f'Source HDF5:      {os.path.abspath(args.validated_hdf5)}',
        f'Episode:          {stem} (index {episode_idx})',
        f'Task:             {args.task if args.task else "(not recorded)"}',
        f'Waypoints:        {n}',
        f'speed_scale:      {args.speed_scale:.3f}  ({1.0 / args.speed_scale:.1f}x slower than recorded)',
        f'Recorded dur.:    {recorded_duration:.2f} s   ->   playback ~{playback_duration:.2f} s',
        f'servoj:           t={args.servoj_t}, lookahead_time={lookahead_time}, gain={gain}',
        f'Startup movej:    a={startup_a} rad/s^2, v={startup_v} rad/s',
        f'Joint order:      {", ".join(JOINT_ORDER)}  (radians)',
        '',
        'Transfer this file to a USB drive and load it via PolyScope.',
        'No PC / network / ur_rtde required -- runs entirely on the controller.',
        'Ensure the workspace is clear and an operator is at the e-stop before running.',
        '=' * 60,
    ]

    script_text = build_urscript(joint_traj, step_sleep, sleeps, header_lines,
                                  startup_v, startup_a, args.servoj_t, lookahead_time, gain)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(script_text)

    print(f'\nWrote URScript -> {os.path.abspath(out_path)}')


if __name__ == '__main__':
    main()
