#!/usr/bin/env python3
"""
Convert a recorded FastUMI episode (TCP position+quaternion) to a UR7e joint
trajectory via the same analytic IK data_processing_to_joint.py uses, then
validate it before it goes anywhere near the physical arm:

  - IK reachability: ik_nearest failing on a frame is flagged, not silently
    papered over (data_processing_to_joint.py's own fallback -- hold the
    previous joints -- still runs, but here it's reported as a real failure).
  - FK cross-check: the analytic IK is closed-form, so a solved frame should
    reproduce its target pose to ~1e-16 m/rad. A nonzero residual on a frame
    that wasn't flagged unreachable means something upstream (frame
    convention, base calibration) is off for that frame specifically.
  - Joint position/velocity limits, parsed from assets/ur7e_robot.urdf (the
    velocity limit is the generic ur_description default carried in that
    file, i.e. the same number the URDF already uses elsewhere in this repo).
  - Joint acceleration, against config.json's ur7e_hardware.joint_accel_limit_rad_s2
    (NOT in the URDF -- a placeholder; verify against the pendant before
    trusting it on hardware).
  - Frame-to-frame joint jumps, using the episode's real inter-frame dt
    (raw/timestamps.csv sibling if present, else a documented 20 Hz fallback).

Writes <episode>_validated.hdf5 (joint_trajectory, dt_s, per-frame
diagnostics, and a `violations` group any downstream script -- e.g.
replay_ur7e.py -- can check before moving actual hardware) and a PNG plot of
joint angle / velocity / acceleration vs. time with violations marked.

Usage:
    python3 validate_joint_trajectory.py path/to/episode_10.hdf5
    python3 validate_joint_trajectory.py --task Pick_and_place_the_bottle --episode 10
    python3 validate_joint_trajectory.py --task Pick_and_place_the_bottle   # all episodes
"""
import argparse
import csv
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT] + [os.path.join(_ROOT, _d) for _d in ('conversion','viz','replay_pipeline','replay','lib')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
import data_processing_to_joint as dpj

JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

POSITION_LIMIT_MARGIN_DEG = 2.0   # flag a frame this close to its URDF position limit
JUMP_WARN_DEG = 15.0              # frame-to-frame joint jump considered a discontinuity
FALLBACK_FPS = 20.0               # documented assumption when no raw/timestamps.csv is found


def load_urdf_joint_limits(urdf_path):
    """{joint_name: (lower_rad, upper_rad, velocity_rad_s)} from the URDF's own <limit> tags."""
    tree = ET.parse(urdf_path)
    limits = {}
    for j in tree.findall('.//joint'):
        name = j.get('name')
        if name in JOINT_NAMES:
            lim = j.find('limit')
            limits[name] = (float(lim.get('lower')), float(lim.get('upper')), float(lim.get('velocity')))
    missing = [n for n in JOINT_NAMES if n not in limits]
    if missing:
        raise ValueError(f"URDF at {urdf_path} is missing <limit> for joint(s): {missing}")
    return np.array([limits[n][0] for n in JOINT_NAMES]), \
        np.array([limits[n][1] for n in JOINT_NAMES]), \
        np.array([limits[n][2] for n in JOINT_NAMES])


def episode_dt(episode_path, n_frames):
    """Real per-frame dt from a sibling raw/<episode>/timestamps.csv, else a
    constant documented fallback. Mirrors visualize_episode_mujoco.py's
    _episode_fps so both scripts agree on timing for the same episode."""
    dataset_dir = os.path.dirname(os.path.abspath(episode_path))
    stem = os.path.splitext(os.path.basename(episode_path))[0]
    ts_path = os.path.join(dataset_dir, 'raw', stem, 'timestamps.csv')
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            rows = list(csv.reader(f))[1:]
        t0, t1 = float(rows[0][1]), float(rows[-1][1])
        dt = np.full(n_frames - 1, (t1 - t0) / (len(rows) - 1))
        return dt, True
    return np.full(n_frames - 1, 1.0 / FALLBACK_FPS), False


def _unwrap_towards(q, ref):
    """Add the per-joint multiple of 2*pi that lands q nearest ref, so a raw
    branch solution stays continuous with the previous frame."""
    q = np.asarray(q, dtype=np.float64)
    return q + 2.0 * np.pi * np.round((np.asarray(ref, dtype=np.float64) - q) / (2.0 * np.pi))


def convert_and_diagnose(episode_path, config, lock_branch=None):
    """Same TCP-pose -> joint-angle conversion as data_processing_to_joint.py
    (reuses its functions directly, so this validates exactly what the batch
    pipeline would produce), plus per-frame reachability/FK diagnostics.

    lock_branch: if None (default), frame 0 seeds on _PREFERRED_BRANCH and every
    later frame uses ik_nearest (stays on the previous branch while reachable,
    else falls back to any branch -- which can silently flip elbow/shoulder/wrist
    when the demo grazes the arm's reach limit). If an int 0..7, EVERY frame is
    forced onto that kinematic branch via ik_branch: a frame is either solved on
    that branch (continuous, same posture) or flagged unreachable (previous joints
    held) -- never a silent flip. Frames unreachable on the locked branch mean the
    demanded pose has no solution in that arm configuration."""
    dcfg = config['data_process_config']
    base_x, base_y, base_z = dcfg['base_position']['x'], dcfg['base_position']['y'], dcfg['base_position']['z']
    base_roll, base_pitch, base_yaw = np.deg2rad(
        [dcfg['base_orientation']['roll'], dcfg['base_orientation']['pitch'], dcfg['base_orientation']['yaw']])
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = dpj.R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]

    with h5py.File(episode_path, 'r') as f:
        qpos_data = f['observations/qpos'][:]
    T = qpos_data.shape[0]

    normalized_qpos = np.copy(qpos_data)
    for i in range(T):
        x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]
        x -= dcfg['offset']['x']
        z += dcfg['offset']['z']
        x, y, z, qx, qy, qz, qw = dpj.remap_t265_to_robot(x, y, z, qx, qy, qz, qw)
        x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, _, _, _ = dpj.transform_to_base_quat(
            x, y, z, qx, qy, qz, qw, T_base_to_local)
        ori = dpj.R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()
        pos = np.array([x_base, y_base, z_base])
        pos += dcfg['offset']['x'] * ori[:, 2]
        pos -= dcfg['offset']['z'] * ori[:, 0]
        normalized_qpos[i, :] = [pos[0], pos[1], pos[2], qx_base, qy_base, qz_base, qw_base]

    joint_traj = np.zeros((T, 6))
    reachable = np.zeros(T, dtype=bool)
    pos_err_mm = np.zeros(T)
    rot_err_deg = np.zeros(T)

    init = np.array(dcfg['start_qpos'][2:8], dtype=np.float64)
    for i in range(T):
        pose = normalized_qpos[i]
        direction, quaternion = dpj.calculate_new_pose(
            pose[0], pose[1], pose[2], pose[3:], dcfg['distances']['flange_to_tcp'])
        T_target = dpj._to_module_frame(direction, quaternion)

        if lock_branch is not None:
            # Force this exact branch every frame; unwrap toward the previous
            # frame for continuity. No solution on this branch -> unreachable, hold.
            q_solved = dpj.ik_branch(dpj._ARM, T_target, branch=int(lock_branch), tool='flange')
            reachable[i] = q_solved is not None
            q = _unwrap_towards(q_solved, init) if q_solved is not None else np.asarray(init, dtype=np.float64)
        elif i == 0:
            q = dpj.seed_joint_angles(direction, quaternion, init)
            # frame 0 is seeded on a fixed branch (ik_branch), not ik_nearest --
            # reachable there iff seed_joint_angles didn't have to fall back.
            reachable[i] = dpj.ik_branch(dpj._ARM, T_target, branch=dpj._PREFERRED_BRANCH, tool='flange') is not None
        else:
            q_solved = dpj.ik_nearest(dpj._ARM, T_target, init, tool='flange')
            reachable[i] = q_solved is not None
            q = np.asarray(q_solved, dtype=np.float64) if q_solved is not None else np.asarray(init, dtype=np.float64)

        achieved = dpj._ARM.fk(q, tool='flange')
        pos_err_mm[i] = np.linalg.norm(achieved[:3, 3] - T_target[:3, 3]) * 1000.0
        rel_rot = dpj.R.from_matrix(achieved[:3, :3].T @ T_target[:3, :3])
        rot_err_deg[i] = np.degrees(rel_rot.magnitude())

        joint_traj[i] = q
        init = q

    return joint_traj, reachable, pos_err_mm, rot_err_deg


def rewind_to_rest(joint_traj, rest_rad):
    """Re-express each joint in the same revolution the real arm rests in.

    The analytic IK emits joints in a principal-ish winding chosen by its
    seed/unwrap, which can sit a full +/-360 deg away from where the physical
    arm actually is at the start pose (confirmed on hardware: wrist_1/wrist_3
    were a full turn off). Commanding that winding makes the startup move wind
    a joint ~360 deg "the long way".

    Fix: for each joint independently, add the integer multiple of 2*pi that
    lands frame 0 nearest ``rest_rad`` (the pendant home reading), and apply
    that SAME offset to every frame. The per-frame path is already continuous
    (ik_nearest unwraps toward the previous frame), so a constant per-joint
    offset preserves the exact motion -- it only chooses the turn number -- and
    re-anchors the trajectory to the arm's real starting revolution.

    Returns (rewound_traj, turns) where turns[j] is the integer 360-deg turns
    added to joint j.
    """
    rest_rad = np.asarray(rest_rad, dtype=np.float64).reshape(6)
    turns = np.round((rest_rad - joint_traj[0]) / (2.0 * np.pi))
    return joint_traj + (turns * 2.0 * np.pi)[None, :], turns.astype(int)


def validate(episode_path, config, out_prefix):
    dcfg = config['data_process_config']
    hw = config.get('ur7e_hardware', {})
    urdf_lo, urdf_hi, urdf_vel = load_urdf_joint_limits(dcfg['urdf_path'])
    accel_limit = np.full(6, hw.get('joint_accel_limit_rad_s2', np.pi))

    joint_traj, reachable, pos_err_mm, rot_err_deg = convert_and_diagnose(episode_path, config)
    T = joint_traj.shape[0]

    # Re-wind to the physical rest pose if one is configured. Done before the
    # limit/velocity checks so validation reflects exactly what the arm will
    # execute. Velocity/accel/jumps are unchanged by a constant offset; the
    # joint-position-limit check is the one that now sees the real winding.
    rest_deg = hw.get('rest_joints_deg')
    rewind_turns = None
    if rest_deg is not None:
        joint_traj, rewind_turns = rewind_to_rest(joint_traj, np.radians(rest_deg))

    dt, dt_measured = episode_dt(episode_path, T)

    vel = np.diff(joint_traj, axis=0) / dt[:, None]                     # (T-1, 6)
    accel = np.diff(vel, axis=0) / dt[1:, None]                          # (T-2, 6)
    jump_deg = np.degrees(np.abs(np.diff(joint_traj, axis=0))).max(axis=1)  # (T-1,)

    margin = np.radians(POSITION_LIMIT_MARGIN_DEG)
    near_pos_limit = (joint_traj - urdf_lo[None, :] < margin) | (urdf_hi[None, :] - joint_traj < margin)
    over_pos_limit = (joint_traj < urdf_lo[None, :]) | (joint_traj > urdf_hi[None, :])
    over_vel_limit = np.abs(vel) > urdf_vel[None, :]
    over_accel_limit = np.abs(accel) > accel_limit[None, :]
    ik_failed = ~reachable
    big_jump = jump_deg > JUMP_WARN_DEG

    any_violation = bool(
        over_pos_limit.any() or over_vel_limit.any() or over_accel_limit.any()
        or ik_failed.any() or big_jump.any()
    )

    print(f"--- {os.path.basename(episode_path)} ---")
    print(f"  frames: {T}   dt: {'measured from raw/timestamps.csv' if dt_measured else f'FALLBACK {FALLBACK_FPS:.0f} Hz (no raw/timestamps.csv found)'}")
    if rest_deg is None:
        print("  winding: NOT re-wound (no ur7e_hardware.rest_joints_deg in config) -- "
              "joints may sit a full turn from the arm's rest pose; startup move could wind the long way.")
    else:
        rewound = [i for i, t in enumerate(rewind_turns) if t != 0]
        jn = ['base', 'shoulder', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
        if rewound:
            det = ', '.join(f"{jn[i]} {rewind_turns[i]:+d} turn" for i in rewound)
            print(f"  winding: re-wound to rest pose -> {det}")
        else:
            print("  winding: re-wound to rest pose -> no shift needed (already on the rest revolution)")
        start_err_deg = np.degrees(joint_traj[0] - np.radians(rest_deg))
        print(f"  start vs rest (deg): {np.round(start_err_deg, 1).tolist()}  (small = frame 0 is near the pendant home)")
    print(f"  IK reachability: {int(ik_failed.sum())}/{T} frame(s) FAILED (target unreachable -- held previous joints)")
    if ik_failed.any():
        print(f"    frames: {np.where(ik_failed)[0].tolist()}")
    print(f"  FK cross-check residual (should be ~0 on reachable frames): "
          f"pos max={pos_err_mm.max():.4f}mm  rot max={rot_err_deg.max():.4f}deg")
    print(f"  joint position: {int(over_pos_limit.any(axis=1).sum())}/{T} frame(s) OUTSIDE URDF limits, "
          f"{int(near_pos_limit.any(axis=1).sum())}/{T} within {POSITION_LIMIT_MARGIN_DEG:.0f} deg of a limit")
    print(f"  joint velocity: {int(over_vel_limit.any(axis=1).sum())}/{max(T - 1, 1)} frame(s) exceed URDF's "
          f"{np.degrees(urdf_vel[0]):.0f} deg/s limit")
    print(f"  joint acceleration: {int(over_accel_limit.any(axis=1).sum())}/{max(T - 2, 1)} frame(s) exceed "
          f"{np.degrees(accel_limit[0]):.0f} deg/s^2 (config placeholder -- verify against pendant)")
    print(f"  frame-to-frame jumps: max={jump_deg.max():.1f} deg/frame, "
          f"{int(big_jump.sum())} frame(s) exceed {JUMP_WARN_DEG:.0f} deg/frame")
    print(f"  => {'VIOLATIONS FOUND -- do not replay on hardware without review' if any_violation else 'no violations'}")
    print()

    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    _save_plot(out_prefix + '.png', joint_traj, vel, accel, dt, urdf_lo, urdf_hi, urdf_vel, accel_limit,
               ik_failed, big_jump)
    _save_hdf5(out_prefix + '.hdf5', joint_traj, dt, dt_measured, reachable, pos_err_mm, rot_err_deg,
               jump_deg, over_pos_limit, over_vel_limit, over_accel_limit, any_violation)
    print(f"  validated trajectory -> {out_prefix}.hdf5")
    print(f"  plot                 -> {out_prefix}.png")
    return any_violation


def _save_plot(path, joint_traj, vel, accel, dt, urdf_lo, urdf_hi, urdf_vel, accel_limit, ik_failed, big_jump):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = joint_traj.shape[0]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    t_vel = (t[:-1] + t[1:]) / 2.0
    t_accel = (t_vel[:-1] + t_vel[1:]) / 2.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for j, name in enumerate(JOINT_NAMES):
        axes[0].plot(t, np.degrees(joint_traj[:, j]), label=name)
        axes[1].plot(t_vel, np.degrees(vel[:, j]))
        axes[2].plot(t_accel, np.degrees(accel[:, j]))
        axes[0].axhline(np.degrees(urdf_lo[j]), color='gray', lw=0.5, ls='--')
        axes[0].axhline(np.degrees(urdf_hi[j]), color='gray', lw=0.5, ls='--')
    axes[1].axhline(np.degrees(urdf_vel[0]), color='red', lw=0.8, ls='--', label='URDF limit')
    axes[1].axhline(-np.degrees(urdf_vel[0]), color='red', lw=0.8, ls='--')
    axes[2].axhline(np.degrees(accel_limit[0]), color='red', lw=0.8, ls='--', label='config limit')
    axes[2].axhline(-np.degrees(accel_limit[0]), color='red', lw=0.8, ls='--')

    if ik_failed.any():
        for i in np.where(ik_failed)[0]:
            axes[0].axvline(t[i], color='crimson', lw=1.2, alpha=0.6)
    if big_jump.any():
        for i in np.where(big_jump)[0]:
            axes[0].axvline(t[i], color='orange', lw=1.0, alpha=0.5)

    axes[0].set_ylabel('joint angle (deg)')
    axes[0].legend(fontsize=7, ncol=3, loc='upper right')
    axes[1].set_ylabel('joint velocity (deg/s)')
    axes[2].set_ylabel('joint accel (deg/s^2)')
    axes[2].set_xlabel('time (s)')
    fig.suptitle('crimson = IK unreachable frame, orange = >%.0f deg/frame jump' % JUMP_WARN_DEG)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _save_hdf5(path, joint_traj, dt, dt_measured, reachable, pos_err_mm, rot_err_deg, jump_deg,
               over_pos_limit, over_vel_limit, over_accel_limit, any_violation):
    with h5py.File(path, 'w') as f:
        f.create_dataset('joint_trajectory', data=joint_traj)
        f.create_dataset('dt_s', data=dt)
        f.attrs['dt_measured_from_timestamps'] = dt_measured
        f.attrs['any_violation'] = any_violation
        diag = f.create_group('diagnostics')
        diag.create_dataset('reachable', data=reachable)
        diag.create_dataset('pos_err_mm', data=pos_err_mm)
        diag.create_dataset('rot_err_deg', data=rot_err_deg)
        diag.create_dataset('jump_deg', data=jump_deg)
        viol = f.create_group('violations')
        viol.create_dataset('over_position_limit', data=over_pos_limit)
        viol.create_dataset('over_velocity_limit', data=over_vel_limit)
        viol.create_dataset('over_acceleration_limit', data=over_accel_limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', nargs='?', default=None, help='path to episode_N.hdf5')
    parser.add_argument('--task', default=None, help='task subdir under config data_dir, e.g. Pick_and_place_the_bottle')
    parser.add_argument('--episode-idx', type=int, default=None, help='single episode index within --task')
    parser.add_argument('--config', default='config/config.json')
    parser.add_argument('--output-dir', default='outputs/validated_trajectories')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.episode:
        episode_paths = [args.episode]
    elif args.task:
        task_dir = os.path.join(config['device_settings']['data_dir'], args.task)
        if args.episode_idx is not None:
            episode_paths = [os.path.join(task_dir, f'episode_{args.episode_idx}.hdf5')]
        else:
            episode_paths = sorted(
                glob.glob(os.path.join(task_dir, 'episode_*.hdf5')),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[1]))
    else:
        print('Specify an episode path, or --task <name> [--episode-idx N].')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    any_violation_overall = False
    for path in episode_paths:
        if not os.path.isfile(path):
            print(f'  (missing: {path})')
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        out_prefix = os.path.join(args.output_dir, stem)
        any_violation_overall |= validate(path, config, out_prefix)

    sys.exit(1 if any_violation_overall else 0)


if __name__ == '__main__':
    main()
