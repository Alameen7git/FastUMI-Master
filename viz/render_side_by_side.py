#!/usr/bin/env python3
"""
Render a real-episode-vs-simulated-UR5e side-by-side comparison video: the
actual recorded camera footage on the left, the MuJoCo IK replay (with an
RGB xyz axis marker drawn on the robot's end effector) on the right.

Usage:
    python3 render_side_by_side.py --task nestest --episode 2
"""
import argparse
import json
import os

import cv2
import h5py
import mujoco
import numpy as np

import mujoco_replay_episode as mre

AXIS_LEN_M = 0.12
AXIS_WIDTH_M = 0.008
AXIS_COLORS = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # x=red, y=green, z=blue

REAL_LABEL = 'REAL'
SIM_LABEL = 'SIM (UR5e, IK replay)'
OUT_HEIGHT = 480


def draw_axis_marker(scene, pos, mat):
    """Draw an RGB xyz triad (x=red, y=green, z=blue) at `pos`, oriented by the
    3x3 rotation matrix `mat` (each column is one axis direction)."""
    for axis_i in range(3):
        direction = mat[:, axis_i]
        start = pos
        end = pos + AXIS_LEN_M * direction
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                             np.array(AXIS_COLORS[axis_i], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, AXIS_WIDTH_M, start, end)
        scene.ngeom += 1


def label(frame, text):
    frame = frame.copy()
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def resize_to_height(frame, height):
    h, w = frame.shape[:2]
    new_w = int(round(w * height / h))
    return cv2.resize(frame, (new_w, height))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--config', default='config/config.json')
    parser.add_argument('--output-dir', default='outputs/mujoco_replay')
    parser.add_argument('--camera-key', default='front')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    dcfg = config['data_process_config']
    data_dir = os.path.join(config['device_settings']['data_dir'], args.task)
    hdf5_path = os.path.join(data_dir, f'episode_{args.episode}.hdf5')

    with h5py.File(hdf5_path, 'r') as f:
        qpos_raw = f['observations/qpos'][:]
        real_images = f[f'observations/images/{args.camera_key}'][:]
        fps = float(f.attrs.get('fps', dcfg.get('target_fps', 20.0)))
    n_frames = qpos_raw.shape[0]

    base_t, base_rot = mre.build_base_transform(dcfg['base_position'], dcfg['base_orientation'])
    positions, quats_xyzw = mre.raw_qpos_to_flange_targets(
        qpos_raw, dcfg['offset']['x'], dcfg['offset']['z'], base_t, base_rot, dcfg['distances']['flange_to_tcp']
    )

    model = mujoco.MjModel.from_xml_path(mre.MJCF_PATH)
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, mre.SITE_NAME)
    joint_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in mre.JOINT_NAMES])
    actuator_ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in mre.ACTUATOR_NAMES])
    qpos_idx = model.jnt_qposadr[joint_ids]
    dof_idx = model.jnt_dofadr[joint_ids]
    jnt_lo = model.jnt_range[joint_ids, 0]
    jnt_hi = model.jnt_range[joint_ids, 1]

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    q = model.key_qpos[key_id][qpos_idx].copy() if key_id >= 0 else np.zeros(6)

    dt = 1.0 / fps
    n_sub = max(1, round(dt / model.opt.timestep))

    renderer = mujoco.Renderer(model, height=mre.RENDER_H, width=mre.RENDER_W)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'episode_{args.episode}_side_by_side.mp4')
    tmp_path = out_path + '.raw.mp4'

    writer = None
    for i in range(n_frames):
        target_quat_xyzw = quats_xyzw[i]
        target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0],
                                      target_quat_xyzw[1], target_quat_xyzw[2]])

        q_sol, _, _, _ = mre.solve_ik(model, data, site_id, positions[i], target_quat_wxyz, q, qpos_idx, dof_idx)
        q_sol = q + (np.mod((q_sol - q) + np.pi, 2 * np.pi) - np.pi)
        q_sol = np.clip(q_sol, jnt_lo, jnt_hi)
        data.qpos[qpos_idx] = q_sol
        q = q_sol

        data.ctrl[actuator_ids] = q
        for _ in range(n_sub):
            mujoco.mj_step(model, data)

        renderer.update_scene(data)
        draw_axis_marker(renderer.scene, data.site_xpos[site_id].copy(), data.site_xmat[site_id].reshape(3, 3).copy())
        sim_frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)

        # HDF5 images are already stored in OpenCV's native BGR order (see
        # data_collection.py's cv_bridge.imgmsg_to_cv2(..., 'bgr8') and
        # convert_episodes.py reading frames straight from cv2.VideoCapture) --
        # no color conversion needed here.
        real_frame = real_images[i]

        real_resized = resize_to_height(real_frame, OUT_HEIGHT)
        sim_resized = resize_to_height(sim_frame, OUT_HEIGHT)

        composite = cv2.hconcat([label(real_resized, REAL_LABEL), label(sim_resized, SIM_LABEL)])

        if writer is None:
            h, w = composite.shape[:2]
            writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        writer.write(composite)

    if writer is not None:
        writer.release()
        mre.transcode_to_h264(tmp_path, out_path)

    print(f'Saved side-by-side comparison -> {out_path}')


if __name__ == '__main__':
    main()
