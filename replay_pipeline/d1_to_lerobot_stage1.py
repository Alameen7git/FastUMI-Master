#!/usr/bin/env python3
"""
STAGE 1 of the FastUMI-raw -> LeRobot(D2-schema) converter.  RUN WITH THE
FastUMI CONDA ENV (needs embodied_ai_ml analytic IK + cv2 ArUco):

    /home/nuc8/miniconda3/envs/FastUMI/bin/python3 d1_to_lerobot_stage1.py \
        --task Pick_and_place_the_bottle --out-dir outputs/d1_lerobot_intermediate

This stage does ONLY the parts that require the FastUMI env: it turns each raw
episode's Cartesian T265 pose into UR7e *joint angles* via the same analytic IK
we validated (validate_joint_trajectory.convert_and_diagnose), re-winds them to
the arm's physical rest pose (the wrist_1/wrist_3 "long way round" fix), derives
the 0-1 gripper openness from the ArUco markers, and finite-differences a joint
velocity.  It writes a small per-episode .npz of pure numerics.

Stage 2 (run in the base/lerobot env) reads those .npz files plus the original
HDF5 images and writes the actual LeRobot v3.0 dataset -- see
d1_to_lerobot_stage2.py.  The split exists only because the IK lives in a
py3.8 env and lerobot lives in a py3.13 env; they can't share a process.

Each .npz contains (T = frames in that episode):
    action        (T,7) float32  [j0..j5 rad, gripper_openness 0..1]  == command
    joints6       (T,6) float32  the six joint angles (radians)
    gripper       (T,)  float32  openness 0..1  (1 = open)
    velocity      (T,7) float32  d(joints)/dt ; gripper channel = 0 (matches D2)
    dt_s          (T-1,) float64 real per-frame dt
    fps           scalar float64 (1/mean dt)
    reachable     (T,)  bool     IK reachability per frame
    rewind_turns  (6,)  int      integer 360-deg turns added per joint
    any_violation bool           limit/vel/accel/jump/reachability flag
    src_hdf5      str            absolute path to the source episode (for images)
    episode_index int
"""
import argparse
import csv
import glob
import json
import os
import sys

import h5py
import numpy as np

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT] + [os.path.join(_ROOT, _d) for _d in ('conversion','viz','replay_pipeline','replay','lib')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import data_processing_to_joint as dpj          # analytic IK + get_gripper_width
import validate_joint_trajectory as vjt          # convert_and_diagnose + rewind_to_rest

FALLBACK_FPS = 20.0


def hdf5_frame_dt(episode_path, T):
    """Per-frame dt for the *subsampled HDF5* stream (T frames), NOT the raw
    camera rate.  raw/<stem>/timestamps.csv is full-rate (~60 Hz) with many more
    rows than the HDF5's T frames; validate_joint_trajectory.episode_dt divides
    by that raw row count and so reports ~60 Hz.  The HDF5 frames span the same
    wall-clock window, so the honest per-HDF5-frame dt is duration/(T-1), i.e.
    fps = (T-1)/duration ~ 20 Hz -- the same value visualize_episode_mujoco uses.
    """
    dataset_dir = os.path.dirname(os.path.abspath(episode_path))
    stem = os.path.splitext(os.path.basename(episode_path))[0]
    ts_path = os.path.join(dataset_dir, 'raw', stem, 'timestamps.csv')
    if os.path.exists(ts_path) and T > 1:
        with open(ts_path) as f:
            rows = list(csv.reader(f))[1:]
        duration = float(rows[-1][1]) - float(rows[0][1])
        if duration > 0:
            return np.full(T - 1, duration / (T - 1)), True
    return np.full(max(T - 1, 1), 1.0 / FALLBACK_FPS), False


def process_episode(episode_path, config, episode_index, lock_branch=None):
    dcfg = config['data_process_config']
    hw = config.get('ur7e_hardware', {})

    # 1) Cartesian pose -> joint angles (the exact validated conversion path).
    joint_traj, reachable, pos_err_mm, rot_err_deg = vjt.convert_and_diagnose(
        episode_path, config, lock_branch=lock_branch)
    T = joint_traj.shape[0]

    # 2) Re-wind to the arm's real rest pose so the startup move doesn't slew a
    #    joint ~360 deg the long way (the confirmed "wrist dived to the ground" bug).
    rest_deg = hw.get('rest_joints_deg')
    rewind_turns = np.zeros(6, dtype=int)
    if rest_deg is not None:
        joint_traj, rewind_turns = vjt.rewind_to_rest(joint_traj, np.radians(rest_deg))

    # 3) Gripper openness (0..1) from the ArUco markers, same as the batch pipeline.
    with h5py.File(episode_path, 'r') as f:
        images = f['observations/images/front'][:]
    gripper = dpj.get_gripper_width(images).astype(np.float64) / dcfg['distances']['gripper_max']
    # get_gripper_width tries to return length T but can drift on marker dropouts.
    if gripper.shape[0] != T:
        if gripper.shape[0] < T:
            pad = np.full(T - gripper.shape[0], gripper[-1] if gripper.size else 0.0)
            gripper = np.concatenate([gripper, pad])
        else:
            gripper = gripper[:T]
    gripper = np.clip(gripper, 0.0, 1.0)

    # 4) action = [6 joints, gripper]; velocity = d(joints)/dt with gripper vel = 0.
    action = np.concatenate([joint_traj, gripper[:, None]], axis=1).astype(np.float32)

    # 5) Cartesian TCP pose per frame, derived from FK of the SAME joints so the
    #    Cartesian and joint datasets describe the identical motion (and held/
    #    unreachable frames carry the held pose). TCP = flange + flange_to_tcp
    #    along the flange local +Z. Emitted in two frames:
    #      cart_urbase  (T,7): [x,y,z, Rx,Ry,Rz, gripper]  UR base frame (DH/module),
    #                          rotation vector -- native for ur_rtde moveL/servoL.
    #      cart_baselink(T,8): [x,y,z, qx,qy,qz,qw, gripper]  ROS base_link (REP-103),
    #                          quaternion -- for MoveIt/ROS Cartesian control.
    #    base_link = _RZ180 @ module (the same relation _to_module_frame inverts).
    ftcp = dcfg['distances']['flange_to_tcp']
    RZ = dpj._RZ180
    cart_urbase = np.zeros((T, 7), dtype=np.float32)
    cart_baselink = np.zeros((T, 8), dtype=np.float32)
    for i in range(T):
        Tf = dpj._ARM.fk(joint_traj[i], tool='flange')     # flange pose, module frame
        Rf, pf = Tf[:3, :3], Tf[:3, 3]
        p_tcp = pf + ftcp * Rf[:, 2]                        # TCP = flange + offset along tool +Z
        cart_urbase[i, :3] = p_tcp
        cart_urbase[i, 3:6] = dpj.R.from_matrix(Rf).as_rotvec()
        cart_urbase[i, 6] = gripper[i]
        p_bl = RZ @ p_tcp
        q_bl = dpj.R.from_matrix(RZ @ Rf).as_quat()         # xyzw
        cart_baselink[i, :3] = p_bl
        cart_baselink[i, 3:7] = q_bl
        cart_baselink[i, 7] = gripper[i]

    dt, dt_measured = hdf5_frame_dt(episode_path, T)
    vel_joints = np.zeros((T, 6))
    if T > 1:
        vel_joints[1:] = np.diff(joint_traj, axis=0) / dt[:, None]
    velocity = np.concatenate([vel_joints, np.zeros((T, 1))], axis=1).astype(np.float32)
    fps = 1.0 / float(np.mean(dt)) if T > 1 else 20.0

    any_violation = bool((~reachable).any())

    return dict(
        action=action,
        joints6=joint_traj.astype(np.float32),
        gripper=gripper.astype(np.float32),
        cart_urbase=cart_urbase,
        cart_baselink=cart_baselink,
        velocity=velocity,
        dt_s=dt,
        fps=fps,
        reachable=reachable,
        rewind_turns=rewind_turns,
        any_violation=any_violation,
        src_hdf5=os.path.abspath(episode_path),
        episode_index=episode_index,
        dt_measured=dt_measured,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task', default=None, help='task subdir under config data_dir')
    ap.add_argument('--src', default=None, help='explicit dataset dir (overrides --task)')
    ap.add_argument('--episodes', default=None,
                    help='comma-separated episode indices (default: all episode_*.hdf5 in the dir)')
    ap.add_argument('--config', default='config/config.json')
    ap.add_argument('--out-dir', default='outputs/d1_lerobot_intermediate')
    ap.add_argument('--lock-branch', type=int, default=None, choices=range(8),
                    help='force one kinematic branch 0..7 every frame (e.g. 5 = elbow-up). '
                         'Prevents silent config flips; unreachable frames on that branch are '
                         'flagged (any_violation) and hold the previous joints.')
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    src_dir = args.src or os.path.join(config['device_settings']['data_dir'], args.task)
    if not os.path.isdir(src_dir):
        sys.exit(f'source dir not found: {src_dir}')

    if args.episodes:
        idxs = [int(x) for x in args.episodes.split(',')]
        paths = [os.path.join(src_dir, f'episode_{i}.hdf5') for i in idxs]
    else:
        paths = sorted(glob.glob(os.path.join(src_dir, 'episode_*.hdf5')),
                       key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[1]))
        # skip any *_validated / *_mujoco stray files (defensive)
        paths = [p for p in paths if os.path.basename(p).split('_')[1].split('.')[0].isdigit()]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'source: {src_dir}\nepisodes: {len(paths)}\nout: {args.out_dir}\n')

    manifest = []
    for ep_i, path in enumerate(paths):
        if not os.path.isfile(path):
            print(f'  (missing: {path}) -- skipped')
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        res = process_episode(path, config, ep_i, lock_branch=args.lock_branch)
        out = os.path.join(args.out_dir, f'{stem}.npz')
        np.savez(out, **{k: v for k, v in res.items()})
        rewound = ', '.join(
            f'{n}{t:+d}' for n, t in zip(['j0', 'j1', 'j2', 'j3', 'j4', 'j5'], res['rewind_turns']) if t)
        print(f'  {stem}: T={res["action"].shape[0]:4d}  fps={res["fps"]:.2f}  '
              f'gripper[{res["gripper"].min():.2f},{res["gripper"].max():.2f}]  '
              f'reach={int(res["reachable"].sum())}/{res["action"].shape[0]}  '
              f'rewind[{rewound or "none"}]  '
              f'{"VIOLATIONS" if res["any_violation"] else "ok"}  -> {os.path.basename(out)}')
        manifest.append(dict(stem=stem, npz=os.path.basename(out), src=res['src_hdf5'],
                             T=int(res['action'].shape[0]), fps=float(res['fps']),
                             any_violation=res['any_violation']))

    with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as f:
        json.dump(dict(source_dir=src_dir, episodes=manifest), f, indent=2)
    print(f'\nwrote {len(manifest)} episode(s) + manifest.json to {args.out_dir}')


if __name__ == '__main__':
    main()
