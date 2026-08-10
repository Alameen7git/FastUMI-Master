#!/usr/bin/env python3
"""
Simulate one recorded FastUMI episode on the UR7e in MuJoCo and save a
side-by-side GIF: the recorded camera footage on the left, the simulated arm
(driven by the SAME joint angles the real robot would execute) on the right.

A quick, safe dry-run: it recomputes the joints exactly like the replay
converter (analytic IK + the winding fix that re-anchors to the arm's rest
pose), so what you watch here is what the robot will do -- minus the physical
startup slew, which a kinematic sim can't show.

RUN WITH THE FastUMI ENV PYTHON (needs mujoco + the analytic IK):

    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 simulate_episode_gif.py \
        /home/nuc8/FastUMI/dataset/Pick_and_place_the_bottle/episode_5.hdf5

    # or by task + index (uses data_dir from config)
    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 simulate_episode_gif.py \
        --task Pick_and_place_the_bottle --episode 5

Outputs, next to the episode by default:
    <episode>_sim_vs_real.gif   (always)
    <episode>_sim_vs_real.mp4   (baseline H.264, unless --no-mp4)

Options: --out, --fps, --gif-fps, --width, --no-mp4, --no-rewind,
         --azimuth/--elevation/--distance (sim camera).
"""
import argparse
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
import data_processing_to_joint as dpj          # analytic IK + get_gripper_width
import validate_joint_trajectory as vjt          # convert_and_diagnose + rewind_to_rest + config load

SCENE = os.path.join(_ROOT, 'assets', 'mujoco', 'ur7e_scene.xml')
CAM_W, CAM_H, SIM_PX = 480, 270, 480


def compute_joints_and_gripper(episode_path, config, rewind, lock_branch=None):
    """Same joints the replay converter produces: analytic IK + rest-pose rewind."""
    dcfg = config['data_process_config']
    joints, reachable, _, _ = vjt.convert_and_diagnose(episode_path, config, lock_branch=lock_branch)
    turns = np.zeros(6, dtype=int)
    rest_deg = config.get('ur7e_hardware', {}).get('rest_joints_deg')
    if rewind and rest_deg is not None:
        joints, turns = vjt.rewind_to_rest(joints, np.radians(rest_deg))
    with h5py.File(episode_path, 'r') as f:
        images = f['observations/images/front'][:]
    gripper = dpj.get_gripper_width(images).astype(np.float64) / dcfg['distances']['gripper_max']
    T = joints.shape[0]
    if gripper.shape[0] != T:                    # guard against marker-dropout length drift
        g = np.full(T, gripper[-1] if gripper.size else 0.0)
        g[:min(T, gripper.shape[0])] = gripper[:min(T, gripper.shape[0])]
        gripper = g
    return joints, np.clip(gripper, 0, 1), images, reachable, turns


def episode_fps(episode_path, T, override):
    """HDF5-frame rate: N/duration from raw/<stem>/timestamps.csv (~20 Hz), the
    same value the converter uses. Falls back to 20 Hz if no timestamps."""
    if override:
        return float(override)
    import csv
    ts = os.path.join(os.path.dirname(os.path.abspath(episode_path)), 'raw',
                      os.path.splitext(os.path.basename(episode_path))[0], 'timestamps.csv')
    if os.path.exists(ts) and T > 1:
        rows = list(csv.reader(open(ts)))[1:]
        dur = float(rows[-1][1]) - float(rows[0][1])
        if dur > 0:
            return (T - 1) / dur
    return 20.0


def _draw_tcp_axes(scene, pos, mat, length=0.09, width=0.006):
    """Draw an RGB orientation triad (x=red, y=green, z=blue) at the TCP -- the
    Cartesian pose [x,y,z, rotation] the arm is realizing -- into the render scene."""
    import mujoco
    cols = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]
    for k in range(3):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(cols[k], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, pos, pos + length * mat[:, k])
        scene.ngeom += 1


def render(episode_path, out_gif, out_mp4, config, fps, gif_fps, width, rewind, cam_angles, keep_mp4,
           lock_branch=None, tcp_axes=False):
    import mujoco
    joints, gripper, images, reachable, turns = compute_joints_and_gripper(episode_path, config, rewind, lock_branch)
    T = joints.shape[0]
    dcfg = config['data_process_config']
    jn = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5']
    rw = ', '.join(f'{jn[i]}{t:+d}' for i, t in enumerate(turns) if t) or 'none'
    lockmsg = f' | LOCKED branch {lock_branch}' if lock_branch is not None else ''
    unreached = np.where(~reachable)[0]
    print(f'{T} frames @ {fps:.2f} fps | reachable {int(reachable.sum())}/{T} | rewind[{rw}]{lockmsg}')
    if unreached.size:
        print(f'  unreachable frames (held): {unreached.tolist()}')

    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=SIM_PX, width=SIM_PX)
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    model.site_pos[tcp] = [0, 0, dcfg['distances']['flange_to_tcp']]

    tmp = mujoco.MjData(model)
    pts = []
    for i in range(0, T, max(1, T // 20)):
        tmp.qpos[:6] = joints[i]
        mujoco.mj_forward(model, tmp)
        pts.append(tmp.site_xpos[tcp].copy())
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = cam_angles
    cam.lookat = np.mean(pts, axis=0)

    tmp_mp4 = out_mp4 + '.raw.mp4'
    writer = None
    for i in range(T):
        data.qpos[:6] = joints[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        if tcp_axes:
            _draw_tcp_axes(renderer.scene, data.site_xpos[tcp].copy(),
                           data.site_xmat[tcp].reshape(3, 3).copy())
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cam_img = cv2.resize(images[i], (CAM_W, CAM_H))
        canvas = np.zeros((max(CAM_H, SIM_PX), CAM_W + SIM_PX, 3), dtype=np.uint8)
        canvas[:CAM_H, :CAM_W] = cam_img
        canvas[:SIM_PX, CAM_W:] = sim
        sim_label = 'SIM: UR7e + TCP pose (Cartesian)' if tcp_axes else 'SIM: UR7e replaying computed joints'
        for txt, org in [('REAL (recorded)', (10, 24)),
                         (sim_label, (CAM_W + 10, 24)),
                         (f'frame {i}/{T}  gripper={gripper[i]:.2f}', (10, CAM_H - 12))]:
            cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
    writer.release()

    # mp4 (baseline H.264 -- plays even without the H.264-high codec plugin)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp_mp4,
                    '-c:v', 'libx264', '-profile:v', 'baseline', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', out_mp4], check=True)

    # GIF via a 2-pass palette (clean colors, opens anywhere)
    vf = f'fps={gif_fps},scale={width}:-1:flags=lanczos'
    pal = out_gif + '.pal.png'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out_mp4,
                    '-vf', vf + ',palettegen=stats_mode=diff', pal], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out_mp4, '-i', pal,
                    '-lavfi', f'{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3', out_gif], check=True)
    os.remove(pal)
    os.remove(tmp_mp4)
    print('saved GIF ->', out_gif)
    if keep_mp4:
        print('saved MP4 ->', out_mp4)
    else:
        os.remove(out_mp4)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('episode', nargs='?', default=None, help='path to episode_N.hdf5')
    ap.add_argument('--task', default=None, help='task subdir under config data_dir (with --episode)')
    ap.add_argument('--episode', dest='episode_idx', type=int, default=None, help='episode index within --task')
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--out', default=None, help='output .gif path (default: <episode>_sim_vs_real.gif)')
    ap.add_argument('--fps', type=float, default=None, help='playback fps (default: measured ~20)')
    ap.add_argument('--gif-fps', type=int, default=12, help='GIF frame rate (default 12)')
    ap.add_argument('--width', type=int, default=760, help='GIF width in px (default 760)')
    ap.add_argument('--no-mp4', action='store_true', help='only keep the GIF')
    ap.add_argument('--no-rewind', action='store_true', help='skip the rest-pose winding fix')
    ap.add_argument('--lock-branch', type=int, default=None, choices=range(8),
                    help='force one kinematic branch 0..7 every frame (e.g. 5 = elbow-up/S1). '
                         'Prevents silent elbow/shoulder/wrist flips; frames unreachable on that '
                         'branch are flagged + held instead of flipping.')
    ap.add_argument('--tcp-axes', action='store_true',
                    help='draw the TCP pose (RGB orientation triad) on the arm end-effector, so the '
                         'Cartesian [x,y,z,rotation] the arm realizes is shown on the mesh.')
    ap.add_argument('--azimuth', type=float, default=340)
    ap.add_argument('--elevation', type=float, default=-30)
    ap.add_argument('--distance', type=float, default=1.9)
    args = ap.parse_args()

    import json
    config = json.load(open(args.config))
    if args.episode:
        ep = args.episode
    elif args.task and args.episode_idx is not None:
        ep = os.path.join(config['device_settings']['data_dir'], args.task, f'episode_{args.episode_idx}.hdf5')
    else:
        sys.exit('give an episode path, or --task NAME --episode N')
    if not os.path.isfile(ep):
        sys.exit(f'episode not found: {ep}')

    stem = os.path.splitext(ep)[0]
    out_gif = args.out or (stem + '_sim_vs_real.gif')
    out_mp4 = os.path.splitext(out_gif)[0] + '.mp4'

    with h5py.File(ep, 'r') as f:
        T = f['observations/images/front'].shape[0]
    fps = episode_fps(ep, T, args.fps)

    render(ep, out_gif, out_mp4, config, fps, args.gif_fps, args.width,
           not args.no_rewind, (args.azimuth, args.elevation, args.distance),
           keep_mp4=not args.no_mp4, lock_branch=args.lock_branch, tcp_axes=args.tcp_axes)
