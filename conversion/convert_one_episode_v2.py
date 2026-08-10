"""
Single-episode converter: FastUMI HDF5 (T265 pose + images) -> UR7e joint angles + gripper width.

Output per frame: [joint1, joint2, joint3, joint4, joint5, joint6, gripper_width]  (7 values, matches state_dim/action_dim=7)

Usage:
    python convert_one_episode.py --config config.json --episode episode_0.hdf5 --output episode_0_joint.hdf5
"""

import argparse
import json

import cv2
import h5py
import numpy as np
import ikpy.chain
from scipy.spatial.transform import Rotation as R

POSITION_ERROR_THRESHOLD_MM = 20.0
RESEED_PERTURBATION_STD_RAD = 0.3


def build_base_transform(base_position, base_orientation):
    t = np.array([base_position["x"], base_position["y"], base_position["z"]])
    rot = R.from_euler(
        "xyz",
        [base_orientation["roll"], base_orientation["pitch"], base_orientation["yaw"]],
        degrees=True,
    )
    return t, rot


def get_aruco_detector(aruco_dict_name):
    dict_id = getattr(cv2.aruco, aruco_dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def solve_ik(chain, target_frame, seed):
    """Solve IK from one seed pose, returning the full joint solution and its errors."""
    full_angles = chain.inverse_kinematics_frame(
        target_frame, initial_position=seed, orientation_mode="all"
    )
    achieved_frame = chain.forward_kinematics(full_angles)
    rel_rot = R.from_matrix(achieved_frame[:3, :3].T @ target_frame[:3, :3])
    orientation_error_deg = np.degrees(rel_rot.magnitude())
    pos_diff = achieved_frame[:3, 3] - target_frame[:3, 3]
    position_error_mm = np.linalg.norm(pos_diff) * 1000.0
    z_error_mm = abs(pos_diff[2]) * 1000.0
    return full_angles, orientation_error_deg, position_error_mm, z_error_mm


def reseed_candidates(prev_solution, start_qpos, elbow_joint_index, real_joint_indices, rng):
    """Alternative seed poses to try when a frame's IK solution has high position error."""
    perturbed = prev_solution.copy()
    perturbed[real_joint_indices] += rng.normal(
        scale=RESEED_PERTURBATION_STD_RAD, size=len(real_joint_indices)
    )

    elbow_bent = prev_solution.copy()
    elbow_bent[elbow_joint_index] = start_qpos[elbow_joint_index]

    return [
        ("start_qpos", start_qpos.copy()),
        ("perturbed_prev", perturbed),
        ("elbow_bent", elbow_bent),
    ]


def compute_gripper_width(image, detector, marker_id_0, marker_id_1,
                           marker_min, marker_max, gripper_max, prev_width=None):
    """Returns physical gripper width for one frame, using Eq. 7 from the FastUMI paper."""
    corners, ids, _ = detector.detectMarkers(image)

    center0, center1 = None, None
    if ids is not None:
        ids_flat = ids.flatten()
        for c, i in zip(corners, ids_flat):
            center = c[0].mean(axis=0)
            if i == marker_id_0:
                center0 = center
            elif i == marker_id_1:
                center1 = center

    if center0 is not None and center1 is not None:
        pixel_dist = np.linalg.norm(center0 - center1)
    elif center0 is not None or center1 is not None:
        # Only one marker found -> mirror it about image center as a rough estimate
        # (paper's fallback strategy; simple approximation here)
        pixel_dist = prev_width if prev_width is not None else (marker_min + marker_max) / 2
        found_one = True
    else:
        # No markers detected -> hold previous value (paper: "imputed value inserted")
        pixel_dist = prev_width if prev_width is not None else (marker_min + marker_max) / 2

    width = (pixel_dist - marker_min) / (marker_max - marker_min) * gripper_max
    width = float(np.clip(width, 0, gripper_max))
    return width, pixel_dist


def process_episode(config, episode_path, output_path, camera_key="front"):
    dcfg = config["data_process_config"]

    my_chain = ikpy.chain.Chain.from_urdf_file(dcfg["urdf_path"], base_elements=["world"])
    n_links = len(my_chain)
    print(f"ikpy chain length: {n_links}")

    active_mask = [False] * n_links
    real_joint_indices = list(range(3, 9))
    for idx in real_joint_indices:
        active_mask[idx] = True
    my_chain.active_links_mask = active_mask

    start_qpos = np.array(dcfg["start_qpos"], dtype=float)
    if len(start_qpos) != n_links:
        raise ValueError(
            f"start_qpos has {len(start_qpos)} entries but URDF chain needs {n_links}."
        )

    offset_x = dcfg["offset"]["x"]
    offset_z = dcfg["offset"]["z"]
    flange_to_tcp = dcfg["distances"]["flange_to_tcp"]
    marker_min = dcfg["distances"]["marker_min"]
    marker_max = dcfg["distances"]["marker_max"]
    gripper_max = dcfg["distances"]["gripper_max"]

    detector = get_aruco_detector(dcfg["aruco_dict"])
    marker_id_0 = dcfg["marker_id_0"]
    marker_id_1 = dcfg["marker_id_1"]

    base_t, base_rot = build_base_transform(dcfg["base_position"], dcfg["base_orientation"])

    with h5py.File(episode_path, "r") as f:
        qpos = f["observations/qpos"][:]  # (T, 7)
        images = f[f"observations/images/{camera_key}"][:]  # (T, H, W, 3)
        T = qpos.shape[0]
        print(f"Episode has {T} frames.")

    elbow_joint_index = real_joint_indices[2]
    rng = np.random.default_rng(0)

    output = np.zeros((T, 7))  # 6 joints + 1 gripper width
    orientation_errors_deg = np.zeros(T)
    position_errors_mm = np.zeros(T)
    z_errors_mm = np.zeros(T)
    reseed_events = []
    prev_solution = start_qpos.copy()
    prev_pixel_dist = None

    for i in range(T):
        x, y, z = qpos[i, 0], qpos[i, 1], qpos[i, 2]
        qx, qy, qz, qw = qpos[i, 3], qpos[i, 4], qpos[i, 5], qpos[i, 6]

        x -= offset_x
        z += offset_z

        local_pos = np.array([x, y, z])
        pos_base = base_t + base_rot.apply(local_pos)

        r_local = R.from_quat([qx, qy, qz, qw])
        r_base = base_rot * r_local

        z_axis_base = r_base.apply([0, 0, 1])
        flange_pos = pos_base - flange_to_tcp * z_axis_base

        target_frame = np.eye(4)
        target_frame[:3, :3] = r_base.as_matrix()
        target_frame[:3, 3] = flange_pos

        full_angles, orientation_error_deg, position_error_mm, z_error_mm = solve_ik(
            my_chain, target_frame, prev_solution
        )

        if position_error_mm > POSITION_ERROR_THRESHOLD_MM:
            original_error_mm = position_error_mm
            best = (full_angles, orientation_error_deg, position_error_mm, z_error_mm)
            best_label = "original"

            for label, seed in reseed_candidates(
                prev_solution, start_qpos, elbow_joint_index, real_joint_indices, rng
            ):
                candidate = solve_ik(my_chain, target_frame, seed)
                if candidate[2] < best[2]:
                    best = candidate
                    best_label = label

            full_angles, orientation_error_deg, position_error_mm, z_error_mm = best
            reseed_events.append({
                "frame": i,
                "original_error_mm": original_error_mm,
                "best_error_mm": position_error_mm,
                "winning_candidate": best_label,
                "resolved": position_error_mm <= POSITION_ERROR_THRESHOLD_MM,
            })

        joints = np.array(full_angles)[real_joint_indices]
        prev_solution = full_angles

        orientation_errors_deg[i] = orientation_error_deg
        position_errors_mm[i] = position_error_mm
        z_errors_mm[i] = z_error_mm

        gripper_width, pixel_dist = compute_gripper_width(
            images[i], detector, marker_id_0, marker_id_1,
            marker_min, marker_max, gripper_max, prev_width=prev_pixel_dist
        )
        prev_pixel_dist = pixel_dist

        output[i, :6] = joints
        output[i, 6] = gripper_width

    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("observations/qpos", data=output)
        f_out.create_dataset("action", data=output)  # mirrors qpos, per FastUMI convention
        f_out.create_dataset("diagnostics/orientation_error_deg", data=orientation_errors_deg)
        f_out.create_dataset("diagnostics/position_error_mm", data=position_errors_mm)
        f_out.create_dataset("diagnostics/z_error_mm", data=z_errors_mm)
        f_out.create_dataset("diagnostics/reseeded_frames", data=np.array([e["frame"] for e in reseed_events]))
        f_out.create_dataset("diagnostics/reseed_original_error_mm", data=np.array([e["original_error_mm"] for e in reseed_events]))
        f_out.create_dataset("diagnostics/reseed_best_error_mm", data=np.array([e["best_error_mm"] for e in reseed_events]))
        f_out.create_dataset("diagnostics/reseed_resolved", data=np.array([e["resolved"] for e in reseed_events]))

    print(f"Saved (T={T}, dim=7) to {output_path}")
    print("First frame [6 joints + gripper]:", output[0])
    print("Last frame  [6 joints + gripper]:", output[-1])
    print(
        "Orientation error (deg): "
        f"min={orientation_errors_deg.min():.4f} "
        f"max={orientation_errors_deg.max():.4f} "
        f"mean={orientation_errors_deg.mean():.4f}"
    )
    print(
        "Position error (mm): "
        f"min={position_errors_mm.min():.4f} "
        f"max={position_errors_mm.max():.4f} "
        f"mean={position_errors_mm.mean():.4f}"
    )
    print(
        "Z-position error (mm): "
        f"min={z_errors_mm.min():.4f} "
        f"max={z_errors_mm.max():.4f} "
        f"mean={z_errors_mm.mean():.4f}"
    )

    if reseed_events:
        resolved = sum(1 for e in reseed_events if e["resolved"])
        print(f"Re-seeded {len(reseed_events)} frame(s) ({resolved} brought back under "
              f"{POSITION_ERROR_THRESHOLD_MM:.0f}mm):")
        for e in reseed_events:
            status = "OK" if e["resolved"] else "STILL HIGH"
            print(f"  frame {e['frame']}: {e['original_error_mm']:.1f}mm -> "
                  f"{e['best_error_mm']:.1f}mm via '{e['winning_candidate']}' [{status}]")
    else:
        print(f"No frames exceeded the {POSITION_ERROR_THRESHOLD_MM:.0f}mm re-seed threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--camera_key", default="front")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    process_episode(config, args.episode, args.output, camera_key=args.camera_key)
