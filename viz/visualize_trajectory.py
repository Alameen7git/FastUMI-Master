#!/usr/bin/env python3
"""
Visualize one episode's converted trajectory as a static figure: the 3D
end-effector (TCP) path plus position, orientation and gripper over time, with
IK-unreachable frames marked. Uses the same verified conversion as the replay
converter (analytic IK -> joints -> winding fix -> FK -> TCP pose), so you're
looking at exactly what would be replayed.

RUN WITH THE FastUMI ENV PYTHON:

    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 visualize_trajectory.py \
        --task PnP_block11 --episode 7
    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 visualize_trajectory.py \
        /home/nuc8/FastUMI/dataset/PnP_block11/episode_7.hdf5 --lock-branch 5

Saves <episode>_trajectory.png. Options: --frame {baselink,urbase}, --point,
--lock-branch N, --no-rewind, --out.
"""
import argparse
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


def cartesian_and_euler(joints, gripper, ftcp, point, frame):
    """TCP position + euler(deg) per frame, in the requested frame."""
    off = ftcp if point == 'tcp' else 0.0
    RZ = dpj._RZ180
    T = joints.shape[0]
    pos = np.zeros((T, 3))
    euler = np.zeros((T, 3))
    for i in range(T):
        Tf = dpj._ARM.fk(joints[i], tool='flange')
        Rf, p = Tf[:3, :3], Tf[:3, 3] + off * Tf[:3, 2]
        if frame == 'baselink':
            p, Rf = RZ @ p, RZ @ Rf
        pos[i] = p
        euler[i] = dpj.R.from_matrix(Rf).as_euler('xyz', degrees=True)
    return pos, euler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', nargs='?', default=None, help='path to episode_N.hdf5')
    ap.add_argument('--task', default=None)
    ap.add_argument('--episode', dest='episode_idx', type=int, default=None)
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--frame', default='baselink', choices=['baselink', 'urbase'])
    ap.add_argument('--point', default='tcp', choices=['tcp', 'flange'])
    ap.add_argument('--lock-branch', type=int, default=None, choices=range(8))
    ap.add_argument('--no-rewind', action='store_true')
    ap.add_argument('--out', default=None)
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
    if args.no_rewind:
        config = json.loads(json.dumps(config))
        config.get('ur7e_hardware', {}).pop('rest_joints_deg', None)

    res = stage1.process_episode(ep, config, 0, lock_branch=args.lock_branch)
    ftcp = config['data_process_config']['distances']['flange_to_tcp']
    pos, euler = cartesian_and_euler(res['joints6'], res['gripper'], ftcp, args.point, args.frame)
    grip, reach = res['gripper'], res['reachable']
    T = pos.shape[0]
    bad = ~reach
    t = np.arange(T)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(15, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax3d.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=t, cmap='viridis', s=10)
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='gray', lw=0.6, alpha=0.6)
    ax3d.scatter(*pos[0], c='green', s=90, marker='o', label='start')
    ax3d.scatter(*pos[-1], c='black', s=90, marker='s', label='end')
    if bad.any():
        ax3d.scatter(pos[bad, 0], pos[bad, 1], pos[bad, 2], c='red', s=40, marker='x', label='unreachable')
    ax3d.set_title(f'TCP path ({args.frame} frame)  -- color = time')
    ax3d.set_xlabel('x (m)'); ax3d.set_ylabel('y (m)'); ax3d.set_zlabel('z (m)')
    ax3d.legend(fontsize=8)
    fig.colorbar(sc, ax=ax3d, shrink=0.6, label='frame')

    axp = fig.add_subplot(2, 2, 2)
    for k, lab in enumerate('xyz'):
        axp.plot(t, pos[:, k], label=lab)
    axp.set_title('TCP position vs frame'); axp.set_ylabel('m'); axp.legend(fontsize=8)

    axo = fig.add_subplot(2, 2, 3)
    for k, lab in enumerate(['roll', 'pitch', 'yaw']):
        axo.plot(t, euler[:, k], label=lab)
    axo.set_title('TCP orientation (euler xyz) vs frame'); axo.set_ylabel('deg'); axo.set_xlabel('frame'); axo.legend(fontsize=8)

    axg = fig.add_subplot(2, 2, 4)
    axg.plot(t, grip, color='purple', label='gripper (0=closed,1=open)')
    axg.set_ylim(-0.05, 1.05); axg.set_title('gripper openness vs frame')
    axg.set_xlabel('frame'); axg.legend(fontsize=8)

    for a in (axp, axo, axg):
        for i in np.where(bad)[0]:
            a.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.12)

    lock = f'  LOCKED branch {args.lock_branch}' if args.lock_branch is not None else ''
    fig.suptitle(f'{os.path.basename(ep)}  |  {T} frames  |  reachable {int(reach.sum())}/{T}  |  '
                 f'point={args.point}{lock}   (red = IK-unreachable, held)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = args.out or (os.path.splitext(ep)[0] + '_trajectory.png')
    fig.savefig(out, dpi=130)
    print(f'reachable {int(reach.sum())}/{T} | saved -> {out}')


if __name__ == '__main__':
    main()
