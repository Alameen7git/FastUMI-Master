"""Pure-numpy forward kinematics for a UR-family 6-DOF arm.

``ArmModel`` builds the kinematic chain from a standard-DH table and computes
batched forward kinematics with plain numpy matrix products — no pinocchio
dependency. (The pinocchio-backed draft, ``ur_kinematics.URArm``, is retained
as an independent cross-validation backend; ``test_fk.py`` asserts the two
agree to ~1e-12.)

Named tool frames (e.g. a gripper fingertip or camera mount) are registered
as rigid 4x4 transforms relative to the flange and selected by name in
:meth:`ArmModel.fk`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .dh import UR7E_DH, DHParams, get_dh, load_dh_from_voraus_json
from .rotations import pose9_from_matrix

__all__ = ["ArmModel"]

FLANGE = "flange"


def _dh_joint_transforms(dh: Sequence[DHParams], q: np.ndarray) -> np.ndarray:
    """Per-joint 4x4 transforms ``T_i(q_i)`` for a batch of configs.

    ``q`` has shape ``(..., n)``; returns ``(..., n, 4, 4)`` where entry ``i``
    is ``Rz(q_i + offset_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i)`` (closed form).
    """
    q = np.asarray(q, dtype=np.float64)
    n = len(dh)
    batch = q.shape[:-1]
    T = np.zeros(batch + (n, 4, 4), dtype=np.float64)
    for i, p in enumerate(dh):
        phi = q[..., i] + p.theta_offset_rad
        c, s = np.cos(phi), np.sin(phi)
        ca, sa = np.cos(p.alpha_rad), np.sin(p.alpha_rad)
        T[..., i, 0, 0] = c
        T[..., i, 0, 1] = -s * ca
        T[..., i, 0, 2] = s * sa
        T[..., i, 0, 3] = p.a * c
        T[..., i, 1, 0] = s
        T[..., i, 1, 1] = c * ca
        T[..., i, 1, 2] = -c * sa
        T[..., i, 1, 3] = p.a * s
        T[..., i, 2, 1] = sa
        T[..., i, 2, 2] = ca
        T[..., i, 2, 3] = p.d
        T[..., i, 3, 3] = 1.0
    return T


class ArmModel:
    """Batched pure-numpy FK for a 6-DOF revolute arm defined by a DH table."""

    def __init__(
        self,
        dh: Sequence[DHParams],
        name: str = "ur_arm",
        tools: dict[str, np.ndarray] | None = None,
    ) -> None:
        dh = tuple(dh)
        if len(dh) != 6:
            raise ValueError(f"expected 6 DH entries, got {len(dh)}")
        self.dh = dh
        self.name = name
        # name -> (4,4) transform from flange to tool. "flange" is identity.
        self.tools: dict[str, np.ndarray] = {FLANGE: np.eye(4)}
        if tools:
            for tname, T in tools.items():
                self.add_tool(tname, T)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def ur7e(cls, tools: dict[str, np.ndarray] | None = None) -> "ArmModel":
        """UR7e from the checked-in DH constants."""
        return cls(UR7E_DH, name="ur7e", tools=tools)

    @classmethod
    def from_dh_name(cls, name: str, tools: dict[str, np.ndarray] | None = None) -> "ArmModel":
        return cls(get_dh(name), name=name, tools=tools)

    @classmethod
    def from_voraus_json(cls, path, name: str | None = None, tools: dict[str, np.ndarray] | None = None) -> "ArmModel":
        from pathlib import Path

        dh = load_dh_from_voraus_json(path)
        return cls(dh, name=name or Path(path).stem, tools=tools)

    def add_tool(self, name: str, T_flange_tool: np.ndarray) -> None:
        """Register a tool frame given its rigid 4x4 transform from the flange."""
        T = np.asarray(T_flange_tool, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"tool transform for '{name}' must be (4, 4), got {T.shape}")
        # Copy so later mutation of the caller's array can't silently change FK.
        self.tools[name] = T.copy()

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def fk(self, q: np.ndarray, tool: str = FLANGE) -> np.ndarray:
        """``(..., 6) -> (..., 4, 4)``: base-to-``tool`` homogeneous transform."""
        if tool not in self.tools:
            raise KeyError(f"unknown tool '{tool}'; registered: {sorted(self.tools)}")
        q = np.asarray(q, dtype=np.float64)
        if q.shape[-1] != 6:
            raise ValueError(f"q must have last dim 6, got {q.shape}")
        Tj = _dh_joint_transforms(self.dh, q)  # (..., 6, 4, 4)
        T = Tj[..., 0, :, :]
        for i in range(1, 6):
            T = T @ Tj[..., i, :, :]
        T = T @ self.tools[tool]
        return T

    def fk_pose9(self, q: np.ndarray, tool: str = FLANGE) -> np.ndarray:
        """``(..., 6) -> (..., 9)``: position + 6D rotation of ``tool``."""
        return pose9_from_matrix(self.fk(q, tool=tool))

    def fk_pos(self, q: np.ndarray, tool: str = FLANGE) -> np.ndarray:
        """``(..., 6) -> (..., 3)``: just the ``tool`` position."""
        return self.fk(q, tool=tool)[..., :3, 3]

    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        """``(..., 6) -> (..., 6, 3)``: joint-axis origins in the base frame.

        Entry ``i`` is the origin of joint frame ``i`` (the proximal axis
        location): entry 0 is the base origin, entry ``i>0`` is the origin
        after composing joint transforms ``1..i``. This matches the
        pinocchio draft's ``URArm.joint_positions`` (``oMi`` convention) and
        is intended for plotting the kinematic chain.
        """
        q = np.asarray(q, dtype=np.float64)
        Tj = _dh_joint_transforms(self.dh, q)
        out = np.empty(q.shape[:-1] + (6, 3), dtype=np.float64)
        out[..., 0, :] = 0.0  # base origin (joint-1 axis)
        T = Tj[..., 0, :, :]
        for i in range(1, 6):
            out[..., i, :] = T[..., :3, 3]
            T = T @ Tj[..., i, :, :]
        return out

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def reach_at_zero(self) -> float:
        """Base-to-flange distance at q=0 (sanity check; UR7e ~0.852 m)."""
        return float(np.linalg.norm(self.fk_pos(np.zeros(6))))

    def __repr__(self) -> str:
        return f"ArmModel(name={self.name!r}, tools={sorted(self.tools)}, reach@0={self.reach_at_zero():.4f}m)"
