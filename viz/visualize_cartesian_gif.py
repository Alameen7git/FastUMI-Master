#!/usr/bin/env python3
"""
Animate one episode's CARTESIAN (task-space) conversion as a GIF: the recorded
footage on the left, and on the right the end-effector TCP POSE -- its 3D
position trail plus an RGB orientation triad (x=red, y=green, z=blue) -- moving
through space frame-by-frame. This visualizes the Cartesian data itself (the
[x,y,z, rotation, gripper] we export), NOT the joint-driven robot mesh.

Uses the same verified conversion (analytic IK -> joints -> winding fix -> FK ->
TCP pose), so this is exactly the Cartesian trajectory in the *_cartesian_*.csv
and the --space cart-* datasets.

RUN WITH THE FastUMI ENV PYTHON:

    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 visualize_cartesian_gif.py \
        --task PnP_block11 --episode 7

Saves <episode>_cartesian_sim.gif (+ .mp4). Options: --frame {baselink,urbase},
--point, --lock-branch N, --no-rewind, --gif-fps, --out, --azimuth/--elevation.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import h5py
import numpy as np

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT] + [os.path.join(_ROOT, _d) for _d in ('conversion','viz','replay_pipeline','replay','lib')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import data_processing_to_joint as dpj
import d1_to_lerobot_stage1 as stage1

CAM_W, CAM_H = 480, 360
TRIAD_LEN = 0.07  # metres


def cartesian_poses(joints, ftcp, point, frame):
    """Per-frame TCP position (3,) and rotation matrix (3,3) in the given frame."""
    off = ftcp if point == 'tcp' else 0.0
    RZ = dpj._RZ180
    T = joints.shape[0]
    pos = np.zeros((T, 3))
    rot = np.zeros((T, 3, 3))
    for i in range(T):
        Tf = dpj._ARM.fk(joints[i], tool='flange')
        Rf, p = Tf[:3, :3], Tf[:3, 3] + off * Tf[:3, 2]
        if frame == 'baselink':
            p, Rf = RZ @ p, RZ @ Rf
        pos[i], rot[i] = p, Rf
    return pos, rot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', nargs='?', default=None)
    ap.add_argument('--task', default=None)
    ap.add_argument('--episode', dest='episode_idx', type=int, default=None)
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--frame', default='baselink', choices=['baselink', 'urbase'])
    ap.add_argument('--point', default='tcp', choices=['tcp', 'flange'])
    ap.add_argument('--lock-branch', type=int, default=None, choices=range(8))
    ap.add_argument('--no-rewind', action='store_true')
    ap.add_argument('--gif-fps', type=int, default=12)
    ap.add_argument('--azimuth', type=float, default=-60)
    ap.add_argument('--elevation', type=float, default=20)
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
    pos, rot = cartesian_poses(res['joints6'], ftcp, args.point, args.frame)
    grip, reach = res['gripper'], res['reachable']
    T = pos.shape[0]
    with h5py.File(ep, 'r') as f:
        images = f['observations/images/front'][:]

    out_gif = args.out or (os.path.splitext(ep)[0] + '_cartesian_sim.gif')
    out_mp4 = os.path.splitext(out_gif)[0] + '.mp4'
    print(f'{os.path.basename(ep)}: {T} frames | frame={args.frame} point={args.point} | '
          f'reachable {int(reach.sum())}/{T}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    lo, hi = pos.min(0), pos.max(0)
    c = (lo + hi) / 2
    r = max((hi - lo).max(), 0.3) / 2 * 1.3
    fig = plt.figure(figsize=(5.2, 5.2))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    axis_cols = ['#e64533', '#37a637', '#3b6fe6']  # x,y,z

    writer, tmp = None, out_mp4 + '.raw.mp4'
    fps_play = float(res['fps'])
    for i in range(T):
        ax.clear()
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='lightgray', lw=0.8)      # full path faint
        ax.plot(pos[:i + 1, 0], pos[:i + 1, 1], pos[:i + 1, 2], color='#1f77b4', lw=1.8)  # trail
        p, Rm = pos[i], rot[i]
        for k in range(3):
            d = Rm[:, k] * TRIAD_LEN
            ax.plot([p[0], p[0] + d[0]], [p[1], p[1] + d[1]], [p[2], p[2] + d[2]],
                    color=axis_cols[k], lw=2.5)
        held = not reach[i]
        ax.scatter(*p, c=('red' if held else 'black'), s=45)
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(elev=args.elevation, azim=args.azimuth)
        ax.set_title(f'TCP pose ({args.frame})  frame {i}/{T}  grip={grip[i]:.2f}'
                     + ('  UNREACHABLE(held)' if held else ''), fontsize=9)
        canvas.draw()
        plot = np.asarray(canvas.buffer_rgba())[:, :, :3]
        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
        plot = cv2.resize(plot, (int(plot.shape[1] * (2 * CAM_H) / plot.shape[0]), 2 * CAM_H))

        cam = cv2.resize(images[i], (CAM_W, CAM_H))
        cam = cv2.copyMakeBorder(cam, 0, 2 * CAM_H - CAM_H, 0, 0, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(cam, 'REAL (recorded)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(cam, 'REAL (recorded)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        canvas_img = cv2.hconcat([cam, plot])
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps_play, (canvas_img.shape[1], canvas_img.shape[0]))
        writer.write(canvas_img)
    writer.release()

    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp,
                    '-c:v', 'libx264', '-profile:v', 'baseline', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', out_mp4], check=True)
    vf = f'fps={args.gif_fps},scale=820:-1:flags=lanczos'
    pal = out_gif + '.pal.png'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out_mp4,
                    '-vf', vf + ',palettegen=stats_mode=diff', pal], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out_mp4, '-i', pal,
                    '-lavfi', f'{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3', out_gif], check=True)
    os.remove(pal); os.remove(tmp)
    print('saved GIF ->', out_gif)
    print('saved MP4 ->', out_mp4)


if __name__ == '__main__':
    main()
