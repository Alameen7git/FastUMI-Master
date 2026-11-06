"""Rotation tooling for robot-learning pose targets (pure numpy + scipy).

Provides conversions between SO(3) representations, with the **continuous 6D
representation** of Zhou et al., "On the Continuity of Rotation
Representations in Neural Networks" (CVPR 2019) as the headline output. The
6D form is the first two columns of the rotation matrix; it is decoded back
to SO(3) by Gram-Schmidt. Unlike quaternions (double cover, antipodal
discontinuity) and Euler/RPY (gimbal lock, +/-pi wrap), it is continuous and
is the de-facto standard regression target for modern manipulation / VLA
policies.

A "pose9" is the 9-vector ``[x, y, z, r11, r21, r31, r12, r22, r32]`` =
position (3) followed by the 6D rotation (the two columns stacked). This is
the per-tool, per-arm cartesian label this module emits.

Conventions
-----------
* Rotation matrices are column-major in the usual sense: ``R[..., :, i]`` is
  the i-th basis column. ``R @ v`` rotates the vector ``v``.
* Quaternions are scalar-last ``[x, y, z, w]`` (scipy convention).
* RPY are extrinsic-xyz Euler angles ``[roll, pitch, yaw]`` such that
  ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` — matching ``pinocchio.rpy.matrixToRpy``.

Every function accepts arbitrary leading batch dimensions.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "matrix_to_rot6d",
    "rot6d_to_matrix",
    "matrix_to_quat",
    "quat_to_matrix",
    "matrix_to_rpy",
    "rpy_to_matrix",
    "pose9_from_matrix",
    "matrix_from_pose9",
    "POSE9_ELEMENT_NAMES",
]

# Element names for a single-arm pose9 feature (used in lerobot feature specs).
POSE9_ELEMENT_NAMES = ["x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32"]

_EPS = 1e-12


def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalise along the last axis, guarding against zero-length vectors."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, _EPS)


# ---------------------------------------------------------------------------
# 6D continuous representation (Zhou et al. 2019)
# ---------------------------------------------------------------------------


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3) -> (..., 6)``: the first two columns of ``R`` stacked.

    Returns ``[R[:,0], R[:,1]]`` concatenated, i.e.
    ``[r11, r21, r31, r12, r22, r32]``.
    """
    R = np.asarray(R, dtype=np.float64)
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return np.concatenate([c0, c1], axis=-1)


def rot6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """``(..., 6) -> (..., 3, 3)``: Gram-Schmidt decode to a valid rotation.

    The two input 3-vectors need not be orthonormal; the output is always in
    SO(3) (orthonormal, det = +1).
    """
    r6 = np.asarray(r6, dtype=np.float64)
    a1 = r6[..., 0:3]
    a2 = r6[..., 3:6]
    b1 = _normalize(a1)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = _normalize(a2 - dot * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


# ---------------------------------------------------------------------------
# Quaternion (scalar-last xyzw, scipy convention)
# ---------------------------------------------------------------------------


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3) -> (..., 4)`` quaternion ``[x, y, z, w]``."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    q = Rotation.from_matrix(flat).as_quat()
    return q.reshape(R.shape[:-2] + (4,))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """``(..., 4) -> (..., 3, 3)`` from quaternion ``[x, y, z, w]``."""
    q = np.asarray(q, dtype=np.float64)
    flat = q.reshape(-1, 4)
    R = Rotation.from_quat(flat).as_matrix()
    return R.reshape(q.shape[:-1] + (3, 3))


# ---------------------------------------------------------------------------
# Roll-pitch-yaw (extrinsic xyz; matches pinocchio.rpy.matrixToRpy)
# ---------------------------------------------------------------------------


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3) -> (..., 3)`` extrinsic-xyz ``[roll, pitch, yaw]``."""
    R = np.asarray(R, dtype=np.float64)
    flat = R.reshape(-1, 3, 3)
    rpy = Rotation.from_matrix(flat).as_euler("xyz")
    return rpy.reshape(R.shape[:-2] + (3,))


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """``(..., 3) -> (..., 3, 3)`` from extrinsic-xyz ``[roll, pitch, yaw]``."""
    rpy = np.asarray(rpy, dtype=np.float64)
    flat = rpy.reshape(-1, 3)
    R = Rotation.from_euler("xyz", flat).as_matrix()
    return R.reshape(rpy.shape[:-1] + (3, 3))


# ---------------------------------------------------------------------------
# pose9 pack / unpack
# ---------------------------------------------------------------------------


def pose9_from_matrix(T: np.ndarray) -> np.ndarray:
    """``(..., 4, 4) -> (..., 9)``: ``[x, y, z, rot6d]``."""
    T = np.asarray(T, dtype=np.float64)
    pos = T[..., :3, 3]
    rot6 = matrix_to_rot6d(T[..., :3, :3])
    return np.concatenate([pos, rot6], axis=-1)


def matrix_from_pose9(p9: np.ndarray) -> np.ndarray:
    """``(..., 9) -> (..., 4, 4)``: inverse of :func:`pose9_from_matrix`."""
    p9 = np.asarray(p9, dtype=np.float64)
    pos = p9[..., :3]
    R = rot6d_to_matrix(p9[..., 3:9])
    T = np.zeros(p9.shape[:-1] + (4, 4), dtype=np.float64)
    T[..., :3, :3] = R
    T[..., :3, 3] = pos
    T[..., 3, 3] = 1.0
    return T
