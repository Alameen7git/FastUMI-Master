#!/usr/bin/env python3
"""Reconstruct one recorded episode as a MuJoCo simulation and play it back
next to the recorded camera feed, to sanity-check that the IK-solved joint
trajectory actually reproduces the demonstrated motion.

Reads urdf_path / base_position / base_orientation / offset / flange_to_tcp /
start_qpos from config/config.json -- the same values data_processing_to_joint.py
uses -- so a different task recorded with the same robot/gripper/handheld rig
needs no extra input, just point it at that task's episode file. If the robot
itself changes, assets/mujoco/ur7e_scene.xml needs rebuilding to match the new
URDF (mesh/body names); everything else is read from config automatically.

Usage:
    python3 visualize_episode_mujoco.py <path/to/episode_N.hdf5> [--out out.mp4]
                                         [--step 2] [--azimuth 340]
                                         [--elevation -30] [--distance 1.9]
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_processing_to_joint as dpj

GRIPPER_YAW_FIX_DEG = 0.0  # rotation about gripper local Z-axis; disabled for isolated remap testing
SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'mujoco', 'ur7e_scene.xml')

# Robotiq 2F-140 finger_joint: 0 rad = fully open, ~0.7 rad = fully closed
# (matches the gripper_closed_position=0.695 default in the source URDF's
# ros2_control macro). The other 5 finger joints mirror this one 1:1 via
# fixed sign relationships (see FINGER_MIMIC_SIGNS below) -- the real
# hardware couples them through a four-bar linkage; the URDF/MJCF tree here
# can't express that closed loop, so we just set all 6 qpos directly instead
# of relying on physics/equality constraints (this script never steps the
# simulation, only calls mj_forward for rendering a given pose).
FINGER_JOINT_CLOSED = 0.7
FINGER_MIMIC_SIGNS = {
    'finger_joint': 1.0,
    'right_outer_knuckle_joint': -1.0,
    'left_inner_knuckle_joint': -1.0,
    'right_inner_knuckle_joint': -1.0,
    'left_inner_finger_joint': 1.0,
    'right_inner_finger_joint': 1.0,
}

CAM_W, CAM_H = 480, 270
SIM_PX = 480


def compute_joint_trajectory(episode_path, config):
    base_x, base_y, base_z = config['base_position']['x'], config['base_position']['y'], config['base_position']['z']
    base_roll, base_pitch, base_yaw = np.deg2rad([
        config['base_orientation']['roll'], config['base_orientation']['pitch'], config['base_orientation']['yaw']])
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()
    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]

    with h5py.File(episode_path, 'r') as f:
        qpos_data = f['action'][:]  # use action (poses), not observations/qpos (leader state)
        images = f['observations/images/front'][:]

    # Gripper openness (0=open, 1=closed) from the same ArUco marker tracking
    # data_processing_to_joint.py uses -- not stored in the raw episode itself.
    gripper_open_width = dpj.get_gripper_width(np.array(images))
    gripper_frac_open = np.clip(gripper_open_width / config['distances']['gripper_max'], 0.0, 1.0)
    gripper_theta = (1.0 - gripper_frac_open) * FINGER_JOINT_CLOSED  # 0=open .. FINGER_JOINT_CLOSED=closed

    N = qpos_data.shape[0]
    raw_t265_pos = np.copy(qpos_data[:, 0:3])  # raw, pre-transform T265 position (local/odom frame)
    normalized_qpos = np.copy(qpos_data)
    for i in range(N):
        x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]
        x -= config['offset']['x']
        z += config['offset']['z']
        x, y, z, qx, qy, qz, qw = dpj.remap_t265_to_robot(x, y, z, qx, qy, qz, qw)
        x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, _, _, _ = dpj.transform_to_base_quat(
            x, y, z, qx, qy, qz, qw, T_base_to_local)
        ori = R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()
        pos = np.array([x_base, y_base, z_base])
        pos += config['offset']['x'] * ori[:, 2]
        pos -= config['offset']['z'] * ori[:, 0]
        normalized_qpos[i, :] = [pos[0], pos[1], pos[2], qx_base, qy_base, qz_base, qw_base]

    joint_traj = []
    init = np.array(config['start_qpos'][2:8])  # fallback only; frame 0 uses seed_joint_angles below
    for i in range(N):
        pose = normalized_qpos[i]
        direction = np.array(pose[:3])
        q = np.array(pose[3:])
        # gripper-frame fix: rotate about local approach (Z) axis
        q = (R.from_quat(q) * R.from_euler('z', GRIPPER_YAW_FIX_DEG, degrees=True)).as_quat()
        direction, quaternion = dpj.calculate_new_pose(
            direction[0], direction[1], direction[2], q, config['distances']['flange_to_tcp'])
        if i == 0:
            full = dpj.seed_joint_angles(direction, quaternion, init)
        else:
            full = dpj.cartesian_to_joints(direction, quaternion, init)
        init = full  # full is already 6 arm joints
        joint_traj.append(full)

    return np.array(joint_traj), images, raw_t265_pos, gripper_theta


def _episode_fps(episode_path, N):
    """Look up the real per-frame time from the sibling raw/<episode>/timestamps.csv
    if present (accurate); otherwise fall back to a rough constant."""
    dataset_dir = os.path.dirname(os.path.abspath(episode_path))
    stem = os.path.splitext(os.path.basename(episode_path))[0]
    ts_path = os.path.join(dataset_dir, 'raw', stem, 'timestamps.csv')
    if os.path.exists(ts_path):
        import csv
        with open(ts_path) as f:
            rows = list(csv.reader(f))[1:]
        raw_n = len(rows)
        t0 = float(rows[0][1])
        t1 = float(rows[-1][1])
        duration = t1 - t0
        # HDF5 stores a subsampled subset of the raw camera frames
        return N / duration, True
    return 20.0, False  # documented fallback assumption, not measured


def render(episode_path, out_path, step, azimuth, elevation, distance):
    import mujoco

    config = dpj.config
    joint_traj, images, raw_t265_pos, gripper_theta = compute_joint_trajectory(episode_path, config)
    N = joint_traj.shape[0]
    print(f'{N} frames, solving done.')

    fps, measured = _episode_fps(episode_path, N)
    print(f"Episode fps: {fps:.3f} ({'measured from raw timestamps.csv' if measured else 'FALLBACK ASSUMPTION -- no raw/timestamps.csv found'})")

    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=SIM_PX, width=SIM_PX)

    tcp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    model.site_pos[tcp_site] = [0, 0, config['distances']['flange_to_tcp']]

    gripper_qadr = {
        name: model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in FINGER_MIMIC_SIGNS
    }

    def set_gripper(data, theta):
        for name, sign in FINGER_MIMIC_SIGNS.items():
            data.qpos[gripper_qadr[name]] = sign * theta

    # Auto-center on this episode's own TCP centroid -- base_position/
    # base_orientation changes can shift which region of the world the robot
    # actually operates in, and a hardcoded lookat silently frames the wrong spot.
    tmp_data = mujoco.MjData(model)
    tcp_positions = []
    for i in range(0, N, max(1, N // 20)):
        tmp_data.qpos[:6] = joint_traj[i]
        mujoco.mj_forward(model, tmp_data)
        tcp_positions.append(tmp_data.site_xpos[tcp_site].copy())
    lookat = np.mean(tcp_positions, axis=0)
    print(f'Auto lookat (episode TCP centroid): {lookat}')

    cam = mujoco.MjvCamera()
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat = lookat

    writer = None
    frame_indices = list(range(0, N, step))
    print(f'Rendering {len(frame_indices)} composite frames...')

    for count, i in enumerate(frame_indices):
        data.qpos[:6] = joint_traj[i]
        set_gripper(data, gripper_theta[i])
        mujoco.mj_forward(model, data)
        tcp_pos = data.site_xpos[tcp_site].copy()
        renderer.update_scene(data, camera=cam)
        sim_img_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        cam_img = cv2.resize(images[i], (CAM_W, CAM_H))
        canvas_h = max(CAM_H, SIM_PX)
        composite = np.zeros((canvas_h, CAM_W + SIM_PX, 3), dtype=np.uint8)
        composite[:CAM_H, :CAM_W] = cam_img
        composite[:SIM_PX, CAM_W:CAM_W + SIM_PX] = sim_img_bgr

        cv2.putText(composite, 'red = FORWARD (base +X)', (CAM_W + 10, SIM_PX - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 220), 2, cv2.LINE_AA)
        cv2.putText(composite, 'blue = RIGHT (base -Y)', (CAM_W + 10, SIM_PX - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 110, 30), 2, cv2.LINE_AA)

        elapsed_s = i / fps
        rp = raw_t265_pos[i]
        lines = [
            f'frame {i}  t={elapsed_s:.2f}s',
            f'T265 raw (m):  x={rp[0]:+.4f} y={rp[1]:+.4f} z={rp[2]:+.4f}',
            f'TCP sim (m):   x={tcp_pos[0]:+.4f} y={tcp_pos[1]:+.4f} z={tcp_pos[2]:+.4f}',
        ]
        for li, line in enumerate(lines):
            y = 30 + li * 26
            cv2.putText(composite, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(composite, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, 12, (composite.shape[1], composite.shape[0]))
        writer.write(composite)
        if count % 20 == 0:
            print(f'  frame {count}/{len(frame_indices)}')

    writer.release()
    print('Saved to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('episode', help='path to episode_N.hdf5')
    parser.add_argument('--out', default=None, help='output mp4 path (default: <episode>_mujoco.mp4)')
    parser.add_argument('--step', type=int, default=2, help='render every Nth frame (default 2)')
    parser.add_argument('--azimuth', type=float, default=340)
    parser.add_argument('--elevation', type=float, default=-30)
    parser.add_argument('--distance', type=float, default=1.9)
    args = parser.parse_args()

    out_path = args.out or os.path.splitext(args.episode)[0] + '_mujoco.mp4'
    render(args.episode, out_path, args.step, args.azimuth, args.elevation, args.distance)
