#!/usr/bin/env python3
"""Render an episode using the ANCHORED / relative-motion method:
frame 0 is pinned to the known real start joints, and every later frame follows
the T265's relative motion from there (no base_position, no offset, no branch
guessing). Light cleanup: interpolate across any unreachable frames, unwrap to
remove wrist wraps. Side-by-side T265 camera + MuJoCo, same layout as
visualize_episode_mujoco.
"""
import argparse, os, sys
import numpy as np, h5py, cv2
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_processing_to_joint as dpj
import visualize_episode_mujoco as vem
from kinematics import ik_nearest

arm = dpj._ARM


def _pose4(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def anchored_joints(episode_path, start_joints):
    """Pure relative-motion IK anchored to start_joints, with cleanup."""
    M4 = np.eye(4); M4[:3, :3] = dpj._T265_TO_ROBOT
    T_start = arm.fk(start_joints, tool='flange')
    q = h5py.File(episode_path, 'r')['action'][:]
    P0 = _pose4(*q[0, 0:7])
    J = [start_joints]
    prev = start_joints
    reachable = [True]
    for i in range(1, len(q)):
        dW = _pose4(*q[i, 0:7]) @ np.linalg.inv(P0)
        EE = (M4 @ dW @ np.linalg.inv(M4)) @ T_start
        jj = ik_nearest(arm, EE, prev, tool='flange')
        if jj is None:
            J.append(prev); reachable.append(False)
        else:
            J.append(jj); prev = jj; reachable.append(True)
    J = np.array(J)
    reachable = np.array(reachable)
    # interpolate across unreachable frames (they were held = flat)
    for j in range(6):
        good = np.where(reachable)[0]
        J[:, j] = np.interp(np.arange(len(J)), good, J[good, j])
    J = np.unwrap(J, axis=0)  # remove any 2pi wrist wraps -> continuous
    return J, reachable


def render(episode_path, out_path, start_joints, step, az, el, dist):
    import mujoco
    config = dpj.config
    J, reachable = anchored_joints(episode_path, start_joints)
    N = len(J)
    print(f'{N} frames | interpolated {np.sum(~reachable)} unreachable | '
          f'max jump {np.abs(np.diff(np.rad2deg(J),axis=0)).max():.0f} deg')
    images = h5py.File(episode_path, 'r')['observations/images/front'][:]

    model = mujoco.MjModel.from_xml_path(vem.SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=vem.SIM_PX, width=vem.SIM_PX)
    tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    model.site_pos[tcp] = [0, 0, config['distances']['flange_to_tcp']]

    tmp = mujoco.MjData(model); pts = []
    for i in range(0, N, max(1, N // 20)):
        tmp.qpos[:6] = J[i]; mujoco.mj_forward(model, tmp); pts.append(tmp.site_xpos[tcp].copy())
    cam = mujoco.MjvCamera(); cam.azimuth = az; cam.elevation = el; cam.distance = dist
    cam.lookat = np.mean(pts, axis=0)

    writer = None
    for i in range(0, N, step):
        data.qpos[:6] = J[i]; mujoco.mj_forward(model, data)
        tcp_pos = data.site_xpos[tcp].copy()
        renderer.update_scene(data, camera=cam)
        sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cam_img = cv2.resize(images[i], (vem.CAM_W, vem.CAM_H))
        H = max(vem.CAM_H, vem.SIM_PX)
        comp = np.zeros((H, vem.CAM_W + vem.SIM_PX, 3), dtype=np.uint8)
        comp[:vem.CAM_H, :vem.CAM_W] = cam_img
        comp[:vem.SIM_PX, vem.CAM_W:] = sim
        tag = 'ANCHORED to real start pose' + ('  [interp]' if not reachable[i] else '')
        for li, line in enumerate([f'frame {i}/{N}', tag,
                                   f'TCP z={tcp_pos[2]:+.3f}m']):
            y = 30 + li * 26
            cv2.putText(comp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(comp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 12, (comp.shape[1], comp.shape[0]))
        writer.write(comp)
    writer.release()
    print('Saved', out_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('episode')
    p.add_argument('--out', default=None)
    p.add_argument('--start-joints-deg', type=float, nargs=6,
                   default=[93.71, -54.77, 125.42, -250.60, -95.39, -184.31])
    p.add_argument('--step', type=int, default=2)
    p.add_argument('--azimuth', type=float, default=340)
    p.add_argument('--elevation', type=float, default=-30)
    p.add_argument('--distance', type=float, default=1.9)
    a = p.parse_args()
    out = a.out or os.path.splitext(a.episode)[0] + '_anchored.mp4'
    render(a.episode, out, np.deg2rad(a.start_joints_deg), a.step, a.azimuth, a.elevation, a.distance)
