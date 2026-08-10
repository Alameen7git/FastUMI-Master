#!/usr/bin/env python3
"""
Replay FastUMI HDF5 episodes (TCP position+quaternion trajectories) on a
simulated UR5e arm in MuJoCo, via per-timestep inverse kinematics -- so you
can see how well a real arm could track a recorded handheld demonstration.

The dataset's observations/qpos is the raw T265 pose in the T265's own
sensor frame, not robot-base frame. This reuses the exact same
sensor-frame -> robot-flange-in-base-frame transform already calibrated in
convert_one_episode_v2.py (config.json's data_process_config), so IK targets
are physically meaningful poses, not raw sensor coordinates.

Usage:
    python3 mujoco_replay_episode.py --task nestest --episode 2
    python3 mujoco_replay_episode.py --task nestest            # all episodes in the task
    python3 mujoco_replay_episode.py --all-tasks               # every task under data_dir
    python3 mujoco_replay_episode.py --task nestest --episode 2 --live
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import cv2
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

MJCF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets/mujoco_menagerie/universal_robots_ur5e/scene.xml')
JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
ACTUATOR_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
SITE_NAME = 'attachment_site'

# IK tuning -- same spirit as convert_one_episode_v2.py's ikpy-based solve
# (which used a 20mm position-error threshold to flag/reseed bad frames),
# adapted here for MuJoCo's own Jacobian instead of ikpy's chain.
IK_MAX_ITERS = 150
IK_POS_TOL_M = 5e-4
IK_ROT_TOL_RAD = 1e-3
IK_DAMPING = 1e-2
POSITION_ERROR_WARN_MM = 20.0
JOINT_LIMIT_MARGIN_DEG = 2.0
SMOOTHNESS_JUMP_DEG = 15.0

RENDER_W, RENDER_H = 640, 480


def transcode_to_h264(intermediate_path, final_path):
    """cv2.VideoWriter's mp4v (MPEG-4 Part 2) mp4s fail to open in most
    browsers/players. Re-encode to H.264 the same way visualize_dataset.py's
    render_video() already does, then drop the intermediate."""
    subprocess.run(
        ['ffmpeg', '-y', '-loglevel', 'error', '-i', intermediate_path,
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', final_path],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    os.remove(intermediate_path)


def build_base_transform(base_position, base_orientation):
    t = np.array([base_position['x'], base_position['y'], base_position['z']])
    rot = R.from_euler('xyz',
                        [base_orientation['roll'], base_orientation['pitch'], base_orientation['yaw']],
                        degrees=True)
    return t, rot


def raw_qpos_to_flange_targets(qpos, offset_x, offset_z, base_t, base_rot, flange_to_tcp):
    """Raw T265 qpos (T, 7) [x,y,z,qx,qy,qz,qw] -> per-frame (position, quat[xyzw])
    of the robot FLANGE in base frame. Mirrors convert_one_episode_v2.py exactly."""
    T = qpos.shape[0]
    positions = np.zeros((T, 3))
    quats_xyzw = np.zeros((T, 4))
    for i in range(T):
        x, y, z = qpos[i, 0], qpos[i, 1], qpos[i, 2]
        qx, qy, qz, qw = qpos[i, 3], qpos[i, 4], qpos[i, 5], qpos[i, 6]
        x -= offset_x
        z += offset_z
        pos_base = base_t + base_rot.apply(np.array([x, y, z]))
        r_base = base_rot * R.from_quat([qx, qy, qz, qw])
        z_axis_base = r_base.apply([0, 0, 1])
        positions[i] = pos_base - flange_to_tcp * z_axis_base
        quats_xyzw[i] = r_base.as_quat()
    return positions, quats_xyzw


def solve_ik(model, data, site_id, target_pos, target_quat_wxyz, q_init, qpos_idx, dof_idx):
    """Damped least-squares IK against a MuJoCo site, warm-started from q_init.
    Returns (qpos_solution, pos_error_mm, rot_error_deg, iterations_used)."""
    data.qpos[qpos_idx] = q_init
    mujoco.mj_forward(model, data)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    rot_err_vec = np.zeros(3)
    it = 0

    for it in range(IK_MAX_ITERS):
        pos_err = target_pos - data.site_xpos[site_id]

        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, data.site_xmat[site_id].reshape(9))
        neg_cur = np.zeros(4)
        mujoco.mju_negQuat(neg_cur, cur_quat)
        quat_err = np.zeros(4)
        mujoco.mju_mulQuat(quat_err, target_quat_wxyz, neg_cur)
        mujoco.mju_quat2Vel(rot_err_vec, quat_err, 1.0)

        if np.linalg.norm(pos_err) < IK_POS_TOL_M and np.linalg.norm(rot_err_vec) < IK_ROT_TOL_RAD:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        err = np.concatenate([pos_err, rot_err_vec])
        JJt = J @ J.T + (IK_DAMPING ** 2) * np.eye(6)
        dq = J.T @ np.linalg.solve(JJt, err)

        data.qpos[qpos_idx] = data.qpos[qpos_idx] + dq
        mujoco.mj_forward(model, data)

    final_pos_err_mm = np.linalg.norm(target_pos - data.site_xpos[site_id]) * 1000.0
    final_rot_err_deg = np.degrees(np.linalg.norm(rot_err_vec))
    return data.qpos[qpos_idx].copy(), final_pos_err_mm, final_rot_err_deg, it + 1


def run_episode(model, hdf5_path, dcfg, out_dir, live=False):
    with h5py.File(hdf5_path, 'r') as f:
        qpos_raw = f['observations/qpos'][:]
        fps = float(f.attrs.get('fps', dcfg.get('target_fps', 20.0)))
    n_frames = qpos_raw.shape[0]

    base_t, base_rot = build_base_transform(dcfg['base_position'], dcfg['base_orientation'])
    positions, quats_xyzw = raw_qpos_to_flange_targets(
        qpos_raw, dcfg['offset']['x'], dcfg['offset']['z'], base_t, base_rot, dcfg['distances']['flange_to_tcp']
    )

    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES])
    actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTUATOR_NAMES])
    qpos_idx = model.jnt_qposadr[joint_ids]
    dof_idx = model.jnt_dofadr[joint_ids]
    jnt_lo = model.jnt_range[joint_ids, 0]
    jnt_hi = model.jnt_range[joint_ids, 1]

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    q = model.key_qpos[key_id][qpos_idx].copy() if key_id >= 0 else np.zeros(6)

    dt = 1.0 / fps
    n_sub = max(1, round(dt / model.opt.timestep))

    ik_pos_errors = np.zeros(n_frames)
    ik_rot_errors = np.zeros(n_frames)
    joint_limit_hit = np.zeros(n_frames, dtype=bool)
    joint_traj = np.zeros((n_frames, 6))

    renderer, video_writer, viewer, video_path, tmp_video_path = None, None, None, None, None
    if live:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
    else:
        renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        video_path = os.path.join(out_dir, f'{os.path.splitext(os.path.basename(hdf5_path))[0]}_ur5e.mp4')
        tmp_video_path = video_path + '.raw.mp4'
        video_writer = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (RENDER_W, RENDER_H))

    for i in range(n_frames):
        target_quat_xyzw = quats_xyzw[i]
        target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0],
                                      target_quat_xyzw[1], target_quat_xyzw[2]])

        q_sol, pos_err_mm, rot_err_deg, _ = solve_ik(
            model, data, site_id, positions[i], target_quat_wxyz, q, qpos_idx, dof_idx
        )
        # These joints all have a +/-2pi (or +/-pi, for the elbow) range, but a
        # serial revolute joint's pose is periodic in each joint's angle -- an
        # unwrapped DLS solve has no reason to prefer the small equivalent
        # angle over a wound-up one, and warm-starting from the raw previous
        # solution lets small per-frame corrections accumulate past a full
        # turn over a long trajectory. Unwrap relative to the PREVIOUS frame's
        # angle (nearest equivalent), not to a fixed (-pi, pi] range -- wrapping
        # to a fixed range still produces a fake ~360deg jump whenever the true
        # angle happens to cross that range's boundary between two frames.
        q_sol = q + (np.mod((q_sol - q) + np.pi, 2 * np.pi) - np.pi)
        q_clipped = np.clip(q_sol, jnt_lo, jnt_hi)
        clamped_by_limit = bool(np.any(np.abs(q_clipped - q_sol) > 1e-9))
        data.qpos[qpos_idx] = q_clipped
        q = q_clipped

        joint_traj[i] = q

        if clamped_by_limit:
            # solve_ik's pos_err_mm/rot_err_deg reflect the unconstrained
            # solution -- a joint limit clip just moved qpos, so recompute the
            # error actually achieved at the clipped pose. Otherwise this
            # would silently under-report tracking error on exactly the
            # frames where a real, joint-limited arm would struggle most.
            mujoco.mj_forward(model, data)
            pos_err_mm = np.linalg.norm(positions[i] - data.site_xpos[site_id]) * 1000.0
            cur_quat = np.zeros(4)
            mujoco.mju_mat2Quat(cur_quat, data.site_xmat[site_id].reshape(9))
            neg_cur = np.zeros(4)
            mujoco.mju_negQuat(neg_cur, cur_quat)
            quat_err = np.zeros(4)
            mujoco.mju_mulQuat(quat_err, target_quat_wxyz, neg_cur)
            rot_err_vec = np.zeros(3)
            mujoco.mju_quat2Vel(rot_err_vec, quat_err, 1.0)
            rot_err_deg = np.degrees(np.linalg.norm(rot_err_vec))

        ik_pos_errors[i] = pos_err_mm
        ik_rot_errors[i] = rot_err_deg

        margin = np.radians(JOINT_LIMIT_MARGIN_DEG)
        joint_limit_hit[i] = bool(np.any((q - jnt_lo < margin) | (jnt_hi - q < margin)))

        data.ctrl[actuator_ids] = q
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        if renderer is not None:
            renderer.update_scene(data)
            frame = renderer.render()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if viewer is not None:
            viewer.sync()
            if not viewer.is_running():
                break

    if video_writer is not None:
        video_writer.release()
        transcode_to_h264(tmp_video_path, video_path)
    if viewer is not None:
        viewer.close()

    deltas_deg = np.degrees(np.abs(np.diff(joint_traj, axis=0))).max(axis=1) if n_frames > 1 else np.array([0.0])
    n_ik_fail = int(np.sum(ik_pos_errors > POSITION_ERROR_WARN_MM))
    n_joint_limit = int(joint_limit_hit.sum())
    jump_frames = np.where(deltas_deg > SMOOTHNESS_JUMP_DEG)[0]

    print(f'--- {os.path.basename(hdf5_path)} ---')
    print(f'  frames: {n_frames}   fps: {fps:.2f}')
    print(f'  IK position error (mm): min={ik_pos_errors.min():.2f} max={ik_pos_errors.max():.2f} '
          f'mean={ik_pos_errors.mean():.2f}')
    print(f'  IK orientation error (deg): min={ik_rot_errors.min():.3f} max={ik_rot_errors.max():.3f} '
          f'mean={ik_rot_errors.mean():.3f}')
    print(f'  IK failures (pos err > {POSITION_ERROR_WARN_MM:.0f}mm): {n_ik_fail}/{n_frames}')
    print(f'  frames within {JOINT_LIMIT_MARGIN_DEG:.0f} deg of a joint limit: {n_joint_limit}/{n_frames}')
    print(f'  trajectory smoothness: max frame-to-frame joint jump = {deltas_deg.max():.1f} deg/frame, '
          f'{len(jump_frames)} frame(s) exceed {SMOOTHNESS_JUMP_DEG:.0f} deg/frame')
    if video_path is not None:
        print(f'  video saved -> {video_path}')
    print()

    return dict(hdf5=hdf5_path, n_frames=n_frames, pos_err=ik_pos_errors, rot_err=ik_rot_errors,
                n_ik_fail=n_ik_fail, n_joint_limit=n_joint_limit, max_jump_deg=float(deltas_deg.max()))


def find_task_dirs(data_dir):
    """Every subdir under data_dir that has at least one episode_*.hdf5 -- same
    idea as visualize_dataset.py's scan_dataset()."""
    tasks = []
    if not os.path.isdir(data_dir):
        return tasks
    for name in sorted(os.listdir(data_dir)):
        task_dir = os.path.join(data_dir, name)
        if os.path.isdir(task_dir) and not name.startswith('.'):
            if glob.glob(os.path.join(task_dir, 'episode_*.hdf5')):
                tasks.append(task_dir)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', default=None, help='Task subdir name under config data_dir, e.g. "nestest"')
    parser.add_argument('--data-dir', default=None, help='Explicit dataset dir (overrides --task/config data_dir)')
    parser.add_argument('--all-tasks', action='store_true', help='Replay every task under config data_dir')
    parser.add_argument('--episode', type=int, default=None, help='Single episode index (default: all in the dir)')
    parser.add_argument('--config', default='config/config.json')
    parser.add_argument('--output-dir', default='outputs/mujoco_replay')
    parser.add_argument('--live', action='store_true',
                         help='Launch an interactive viewer instead of saving video (single episode only)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    dcfg = config['data_process_config']
    base_data_dir = config['device_settings']['data_dir']

    if args.data_dir:
        task_dirs = [args.data_dir]
    elif args.all_tasks:
        task_dirs = find_task_dirs(base_data_dir)
        if not task_dirs:
            print(f'No task directories with episode_*.hdf5 found under {base_data_dir}')
            sys.exit(1)
    elif args.task:
        task_dirs = [os.path.join(base_data_dir, args.task)]
    else:
        print('Specify --task <name>, --data-dir <path>, or --all-tasks.')
        sys.exit(1)

    episode_paths = []
    for task_dir in task_dirs:
        if args.episode is not None:
            episode_paths.append(os.path.join(task_dir, f'episode_{args.episode}.hdf5'))
        else:
            episode_paths.extend(sorted(
                glob.glob(os.path.join(task_dir, 'episode_*.hdf5')),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[1])
            ))

    if args.live and len(episode_paths) > 1:
        print('--live only supports one episode at a time; pass --task/--data-dir with --episode N.')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)

    results = []
    for path in episode_paths:
        if not os.path.isfile(path):
            print(f'  (missing: {path})')
            continue
        results.append(run_episode(model, path, dcfg, args.output_dir, live=args.live))

    if len(results) > 1:
        print('=== summary across all episodes ===')
        for r in results:
            print(f"  {os.path.basename(r['hdf5']):20s} "
                  f"pos_err(mean)={r['pos_err'].mean():6.2f}mm  "
                  f"ik_failures={r['n_ik_fail']:3d}/{r['n_frames']:<4d} "
                  f"joint_limit_hits={r['n_joint_limit']:3d}  "
                  f"max_jump={r['max_jump_deg']:5.1f}deg")


if __name__ == '__main__':
    main()
