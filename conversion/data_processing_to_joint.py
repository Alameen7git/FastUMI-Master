import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import sys
import types
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json

# Load the configuration from the config.json file
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, 'config', 'config.json'), 'r') as config_file:
    config = json.load(config_file)
config = config["data_process_config"]

# Extract configuration values
START_QPOS = config["start_qpos"] # Initial joint positions for the robot (values specific to your robot's configuration)
PI = np.pi

# --- Analytic UR7e IK (embodied_ai_ml.kinematics) -------------------------
# Replaces ikpy's iterative solver, which had no way to detect or recover
# from converging to a bad local minimum near joint-limit boundaries --
# roughly 11/25 episodes had frames off by up to tens of cm as a result.
# This module computes the UR closed-form analytic solution (all reachable
# branches, exact algebra) and picks the branch nearest the previous frame's
# joints, so there is no seed-dependent divergence. Verified to reproduce
# every recorded target to ~1e-16 m across 5 test episodes (0 failures).
#
# The package targets Python >=3.12 and its own __init__.py eagerly imports
# unrelated heavy deps (lerobot, etc.) not installed in this (3.8) env; the
# kinematics submodule itself only needs numpy+scipy, so we stub the parent
# package in sys.modules to import just that submodule without triggering
# the rest of the package.
_EMBODIED_AI_ML_SRC = "/home/nuc8/embodied_ai_ml-main/src"  # machine-specific path -- this machine has no /dev/ prefix
if _EMBODIED_AI_ML_SRC not in sys.path:
    sys.path.insert(0, _EMBODIED_AI_ML_SRC)
if "embodied_ai_ml" not in sys.modules:
    _stub = types.ModuleType("embodied_ai_ml")
    _stub.__path__ = [os.path.join(_EMBODIED_AI_ML_SRC, "embodied_ai_ml")]
    sys.modules["embodied_ai_ml"] = _stub
from embodied_ai_ml.kinematics import ArmModel, ik_nearest, ik_branch  # noqa: E402

_ARM = ArmModel.ur7e()
# Fixed 180-degree yaw between this module's DH-canonical base frame and our
# URDF base_link (REP-103) convention -- verified by FK cross-check (matches
# our ikpy/URDF tool0 chain to ~0.3mm, i.e. within DH-table rounding) rather
# than a joint-angle-convention difference.
_RZ180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

# Kinematic branch (4*shoulder + 2*elbow + wrist, see embodied_ai_ml.kinematics.ik_ur)
# this robot/mounting/workspace naturally reaches in. Derived on THIS machine
# from frame 0 of dataset/Pick_and_place_the_bottle/episode_1.hdf5, after
# re-deriving base_position/base_orientation from a real UR pendant Joint
# Position reading taken at that episode's home pose (Base=93.71, Shoulder=
# -54.77, Elbow=125.42, Wrist1=-250.60, Wrist2=-95.39, Wrist3=-184.31 deg).
# All 8 branches were tested against that same pendant reading (mod-360-aware
# comparison); branch 5 matched to within 0.05 degrees per joint (residual
# consistent with the pendant display's 2-decimal rounding), every other
# branch was off by 6-180 degrees on at least one joint. Used only to seed
# frame 0; ik_nearest (branch-locked, nearest-to-previous-frame) takes over
# for every subsequent frame.
_PREFERRED_BRANCH = 5


def _to_module_frame(position, quaternion):
    """Our base_link-frame (position, xyzw quat) -> module's (4, 4) target."""
    T = np.eye(4)
    T[:3, :3] = _RZ180 @ R.from_quat(quaternion).as_matrix()
    T[:3, 3] = _RZ180 @ np.asarray(position, dtype=np.float64)
    return T


def seed_joint_angles(position, quaternion, fallback):
    """Frame-0 seed on ``_PREFERRED_BRANCH``, or ``fallback`` if unreachable there."""
    T_target = _to_module_frame(position, quaternion)
    q = ik_branch(_ARM, T_target, branch=_PREFERRED_BRANCH, tool='flange')
    if q is None:
        print("Warning: preferred branch unreachable at frame 0, falling back")
        return np.asarray(fallback, dtype=np.float64)
    return q


print(f"Analytic UR7e IK ready (embodied_ai_ml.kinematics), reach@0={_ARM.reach_at_zero():.4f}m")

# T265/UMI local frame vs the robot frame this pipeline's base_position/base_orientation
# calibration is expressed in -- determined empirically by the user comparing expected
# vs actual robot response: T265 +X(fwd)->Robot +Y, T265 +Y(left)->Robot -X, T265 +Z(up)->
# Robot +Z. Applied to BOTH the local position (before adding to base_position, which was
# previously a raw component-wise add across mismatched axis labels) and the local
# orientation (right-multiplied so it composes correctly with base_rot in transform_to_base_quat).
_T265_TO_ROBOT = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # test: sign-flipped fit, A->B=+Y (away from base), B->C=+X


def remap_t265_to_robot(x, y, z, qx, qy, qz, qw):
    pos = _T265_TO_ROBOT @ np.array([x, y, z], dtype=np.float64)
    m = R.from_quat([qx, qy, qz, qw]).as_matrix() @ _T265_TO_ROBOT.T
    qx, qy, qz, qw = R.from_matrix(m).as_quat()
    return pos[0], pos[1], pos[2], qx, qy, qz, qw


# Load predefined ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config["aruco_dict"]))
# The other machine's opencv-contrib-python==4.6.0.66 build segfaulted on
# DetectorParameters() and needed the legacy DetectorParameters_create()
# factory instead. This machine's opencv-contrib-python==4.13.0 has already
# dropped that legacy factory entirely (AttributeError) and DetectorParameters()
# itself works fine here (verified directly) -- machine-specific, not ported.
parameters = cv2.aruco.DetectorParameters()



def calculate_new_pose(x, y, z, quaternion, distance):
    """
    Calculate a new pose by translating along the negative Z-axis of the given pose.
    """
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    z_axis = rotation_matrix[:, 2]
    new_position = np.array([x, y, z]) - distance * z_axis
    return [new_position[0], new_position[1], new_position[2]], quaternion


def cartesian_to_joints(position, quaternion, initial_joint_angles):
    """
    Convert a Cartesian tool0 pose to the 6 UR7e joint angles via the exact
    analytic IK, picking the solution nearest ``initial_joint_angles`` (a
    6-vector) to keep the trajectory continuous frame-to-frame.
    """
    T_target = _to_module_frame(position, quaternion)
    q = ik_nearest(_ARM, T_target, initial_joint_angles, tool='flange')
    if q is None:
        # Should not happen for reachable pick-and-place targets; keep the
        # previous joints rather than silently propagating a bad solution.
        print("Warning: target unreachable, holding previous joint angles")
        return np.asarray(initial_joint_angles, dtype=np.float64)
    return q


def get_gripper_width(img_list):
    """
    Calculate gripper width from detected ArUco markers in the images.
    """
    distances = []
    distances_index = []
    current_frame = 0
    frame_count = len(img_list)

    for i in range(img_list.shape[0]):
        gray = cv2.cvtColor(img_list[i, :, :, :], cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        current_frame += 1
        if ids is not None:
            
            marker_centers = []
            for idx, marker_id in enumerate(ids.flatten()):
                if marker_id in [config["marker_id_0"], config["marker_id_1"]]:
                    marker_corners = corners[idx][0]
                    center = np.mean(marker_corners, axis=0).astype(int)
                    marker_centers.append(center)

            if len(marker_centers) >= 2:
                distance = np.linalg.norm(marker_centers[0] - marker_centers[1])
                distances.append(distance)
                distances_index.append(current_frame)
            elif len(marker_centers) == 1:
                distance = abs(gray.shape[1] / 2 - marker_centers[0][0]) * 2
                distances.append(distance)
                distances_index.append(current_frame)

    distances = np.array(distances)
    distances_index = np.array(distances_index)
    distances = ((distances - config["distances"]["marker_min"]) / (config["distances"]["marker_max"] - config["distances"]["marker_min"]) * config["distances"]["gripper_max"]).astype(np.int16).clip(0, config["distances"]["gripper_max"])

    new_distances = []
    for i in range(len(distances) - 1):
        if i == 0:
            if distances_index[i] == 1:
                new_distances.append(distances[0])
                continue
            else:
                for _ in range(distances_index[0]):
                    new_distances.append(distances[0])
        else:
            if distances_index[i + 1] - distances_index[i] == 1:
                new_distances.append(distances[i])
            else:
                for k in range(distances_index[i + 1] - distances_index[i]):
                    interpolated_distance = int(
                        k * (distances[i + 1] - distances[i]) /
                        (distances_index[i + 1] - distances_index[i]) +
                        distances[i])
                    new_distances.append(interpolated_distance)
    new_distances.append(distances[-1])
    if len(new_distances) < frame_count:
        for _ in range(frame_count - len(new_distances)):
            new_distances.append(distances[-1])

    return np.array(new_distances)

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    T_base_r = np.dot(T_local[:3, :3], T_base_to_local[:3, :3])
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler(
        'xyz', degrees=False)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base


def normalize_ik_and_save_hdf5(args):
    """
    Normalize input data and save processed HDF5 files.
    """
    input_file, output_file = args
    base_x, base_y, base_z = config["base_position"]["x"], config["base_position"]["y"], config["base_position"]["z"] # Initial position of the robot's base in 3D space (in meters)
    base_roll, base_pitch, base_yaw = np.deg2rad([config["base_orientation"]["roll"], config["base_orientation"]["pitch"], config["base_orientation"]["yaw"]]) # Initial orientation of the robot's base in 3D space (in roll, pitch, yaw format) (in degrees)
    rotation_base_to_local = R.from_euler('xyz', [base_roll, base_pitch, base_yaw]).as_matrix()

    T_base_to_local = np.eye(4)
    T_base_to_local[:3, :3] = rotation_base_to_local
    T_base_to_local[:3, 3] = [base_x, base_y, base_z]

    try:
        with h5py.File(input_file, 'r') as f_in:
            action_data = f_in['action'][:]
            qpos_data = f_in['observations/qpos'][:]
            normalized_qpos = np.copy(qpos_data)

            for i in range(normalized_qpos.shape[0]):
                x, y, z, qx, qy, qz, qw = normalized_qpos[i, 0:7]
                x -= config["offset"]["x"]
                z += config["offset"]["z"]
                x, y, z, qx, qy, qz, qw = remap_t265_to_robot(x, y, z, qx, qy, qz, qw)
                x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, _, _, _ = transform_to_base_quat(
                    x, y, z, qx, qy, qz, qw, T_base_to_local)
                ori = R.from_quat([qx_base, qy_base, qz_base, qw_base]).as_matrix()
                pos = np.array([x_base, y_base, z_base])
                pos += config["offset"]["x"] * ori[:, 2]
                pos -= config["offset"]["z"] * ori[:, 0]
                x_base, y_base, z_base = pos
                normalized_qpos[i, :] = [x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base]

            joint_angles = []
            action_data = np.copy(normalized_qpos)
            image_data = f_in['observations/images/front'][:]
            qpos_data = normalized_qpos
            data = np.array(action_data)

            # START_QPOS carries 2 fixed placeholder entries on each end (legacy
            # ikpy 10-link chain layout); only the middle 6 are used, and only
            # as a last-resort fallback -- frame 0's real seed comes from
            # seed_joint_angles (_PREFERRED_BRANCH) below.
            initial_joint_angles = np.array(START_QPOS[2:8], dtype=np.float64)
            for i in range(len(data)):
                pose = data[i]
                direction = np.array(pose[:3])
                q = np.array(pose[3:])
                direction, quaternion = calculate_new_pose(
                    direction[0], direction[1], direction[2], q, config["distances"]["flange_to_tcp"])
                if i == 0:
                    six_dof_joint_angles = seed_joint_angles(direction, quaternion, initial_joint_angles)
                else:
                    six_dof_joint_angles = cartesian_to_joints(
                        direction, quaternion, initial_joint_angles)
                initial_joint_angles = six_dof_joint_angles
                joint_angles.append(six_dof_joint_angles)

            joint_angles = np.array(joint_angles)

            image_data = np.array(image_data)
            gripper_open_width = get_gripper_width(image_data)
            gripper_open_width = gripper_open_width / config["distances"]["gripper_max"]

            gripper_width = gripper_open_width.reshape(-1, 1)
            new_joint_angles = np.concatenate(
                (joint_angles, gripper_width), axis=1)

            with h5py.File(output_file, 'w') as f_out:
                f_out.create_dataset('action', data=new_joint_angles)
                observations_group = f_out.create_group('observations')
                images_group = observations_group.create_group('images')

                max_timesteps = f_in['observations/images/front'].shape[0]
                cam_hight = f_in['observations/images/front'].shape[1]
                cam_width = f_in['observations/images/front'].shape[2]

                images_group.create_dataset(
                    'front',
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4)
                images_group['front'][:] = f_in['observations/images/front'][:]

                observations_group.create_dataset('qpos', data=new_joint_angles)

                print(f"Processed and saved: {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":
    input_dir = config["input_dir"]
    output_dir = config["output_joint_dir"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = [f for f in os.listdir(input_dir) if f.endswith('.hdf5')]
    args_list = []
    for f in file_list:
        input_file = os.path.join(input_dir, f)
        output_file = os.path.join(output_dir, f)
        args_list.append((input_file, output_file))
    print("Starting parallel processing...")

    for _a in tqdm(args_list, total=len(args_list), desc="Processing files"):
        normalize_ik_and_save_hdf5(_a)

    print("Processing completed.")
