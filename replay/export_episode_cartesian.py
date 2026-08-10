#!/usr/bin/env python3
"""
Export ONE FastUMI episode's end-effector trajectory in Cartesian (task) space
to a plain CSV (and .npz) -- for inspection or to feed a Cartesian controller
directly, without building a whole LeRobot dataset.

Reuses the exact stage-1 conversion (analytic IK -> joints -> winding fix ->
FK -> TCP pose), so the exported Cartesian poses are the same motion as the
joint dataset and are guaranteed reachable. Emits either/both frames:

  urbase   : [x, y, z, Rx, Ry, Rz, gripper]   UR base frame, rotation vector
             (ur_rtde moveL/servoL native; meters + radians)
  baselink : [x, y, z, qx, qy, qz, qw, gripper]  ROS base_link, quaternion

RUN WITH THE FastUMI ENV PYTHON:

    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 export_episode_cartesian.py \
        --task PnP_block11 --episode 7 --frame both
    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 export_episode_cartesian.py \
        /home/nuc8/FastUMI/dataset/PnP_block11/episode_7.hdf5 --frame urbase

Options: --frame {urbase,baselink,both}, --point {tcp,flange}, --lock-branch N,
         --no-rewind, --out <path-prefix>.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT] + [os.path.join(_ROOT, _d) for _d in ('conversion','viz','replay_pipeline','replay','lib')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import data_processing_to_joint as dpj
import d1_to_lerobot_stage1 as stage1

HDR = {'urbase':   ['frame', 'x', 'y', 'z', 'Rx', 'Ry', 'Rz', 'gripper', 'reachable'],
       'baselink': ['frame', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper', 'reachable']}


def cartesian_from_joints(joints, gripper, ftcp, point):
    """Recompute the Cartesian pose from the (rewound) joints so we can honor the
    --point choice. TCP = flange + ftcp along tool +Z; flange = ftcp 0."""
    off = ftcp if point == 'tcp' else 0.0
    RZ = dpj._RZ180
    T = joints.shape[0]
    ur = np.zeros((T, 7)); bl = np.zeros((T, 8))
    for i in range(T):
        Tf = dpj._ARM.fk(joints[i], tool='flange')
        Rf, p = Tf[:3, :3], Tf[:3, 3] + off * Tf[:3, 2]
        ur[i, :3] = p
        ur[i, 3:6] = dpj.R.from_matrix(Rf).as_rotvec()
        ur[i, 6] = gripper[i]
        bl[i, :3] = RZ @ p
        bl[i, 3:7] = dpj.R.from_matrix(RZ @ Rf).as_quat()
        bl[i, 7] = gripper[i]
    return ur.astype(np.float32), bl.astype(np.float32)


def write_csv(path, header, arr, reachable):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(arr.shape[0]):
            w.writerow([i] + [f'{v:.6f}' for v in arr[i]] + [int(reachable[i])])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', nargs='?', default=None, help='path to episode_N.hdf5')
    ap.add_argument('--task', default=None, help='task subdir under config data_dir (with --episode)')
    ap.add_argument('--episode', dest='episode_idx', type=int, default=None, help='episode index within --task')
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--frame', default='both', choices=['urbase', 'baselink', 'both'])
    ap.add_argument('--point', default='tcp', choices=['tcp', 'flange'])
    ap.add_argument('--lock-branch', type=int, default=None, choices=range(8),
                    help='force one kinematic branch 0..7 (e.g. 5 = elbow-up)')
    ap.add_argument('--no-rewind', action='store_true', help='skip the rest-pose winding fix')
    ap.add_argument('--out', default=None, help='output path prefix (default: <episode>_cartesian)')
    args = ap.parse_args()

    config = json.load(open(args.config))
    if args.episode:
        ep = args.episode
    elif args.task and args.episode_idx is not None:
        ep = os.path.join(config['device_settings']['data_dir'], args.task, f'episode_{args.episode_idx}.hdf5')
    else:
        sys.exit('give an episode path, or --task NAME --episode N')
    if not os.path.isfile(ep):
        sys.exit(f'episode not found: {ep}')

    # rest-pose rewind is on by default in stage1; suppress it by temporarily
    # blanking rest_joints_deg when --no-rewind is requested.
    if args.no_rewind:
        config = json.loads(json.dumps(config))
        config.get('ur7e_hardware', {}).pop('rest_joints_deg', None)

    res = stage1.process_episode(ep, config, 0, lock_branch=args.lock_branch)
    ftcp = config['data_process_config']['distances']['flange_to_tcp']
    ur, bl = cartesian_from_joints(res['joints6'], res['gripper'], ftcp, args.point)
    reachable = res['reachable']
    T = ur.shape[0]

    prefix = args.out or (os.path.splitext(ep)[0] + '_cartesian')
    frames = ['urbase', 'baselink'] if args.frame == 'both' else [args.frame]
    print(f'{os.path.basename(ep)}: {T} frames | point={args.point} | '
          f'reachable {int(reachable.sum())}/{T}'
          f'{" | LOCKED branch %d" % args.lock_branch if args.lock_branch is not None else ""}')
    unreached = np.where(~reachable)[0]
    if unreached.size:
        print(f'  held (unreachable) frames: {unreached.tolist()}')
    saved = {}
    for fr in frames:
        arr = ur if fr == 'urbase' else bl
        csv_path = f'{prefix}_{fr}.csv'
        write_csv(csv_path, HDR[fr], arr, reachable)
        saved[fr] = arr
        print(f'  wrote {csv_path}')
    npz_path = f'{prefix}.npz'
    np.savez(npz_path, reachable=reachable, gripper=res['gripper'], **saved)
    print(f'  wrote {npz_path}')


if __name__ == '__main__':
    main()
