"""Joint-vector layouts for single-arm and bimanual UR rigs.

Datasets store a flat joint vector per frame. This module describes how that
vector decomposes into per-arm 6-DOF joint blocks and gripper scalars, so FK
labelling and IK can slice the right six joints for each arm:

* single-arm 7-dim: ``[j0..j5, gripper]``
* bimanual 14-dim:  ``[right_j0..j5, right_gripper, left_j0..j5, left_gripper]``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ArmSlice",
    "JointLayout",
    "SINGLE_ARM_7",
    "BIMANUAL_14",
]


@dataclass(frozen=True)
class ArmSlice:
    """One arm within a flat joint vector."""

    name: str
    joint_start: int  # index of this arm's joint 0
    gripper_index: int | None  # index of this arm's gripper scalar, or None

    @property
    def joint_slice(self) -> slice:
        return slice(self.joint_start, self.joint_start + 6)


@dataclass(frozen=True)
class JointLayout:
    """Decomposition of a flat joint vector into arms."""

    name: str
    dim: int
    arms: tuple[ArmSlice, ...]

    @property
    def arm_names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.arms)

    def _get_arm(self, arm: str | None) -> ArmSlice:
        if arm is None:
            if len(self.arms) != 1:
                raise ValueError(
                    f"layout '{self.name}' has {len(self.arms)} arms; specify arm " f"(one of {self.arm_names})"
                )
            return self.arms[0]
        for a in self.arms:
            if a.name == arm:
                return a
        raise KeyError(f"unknown arm '{arm}' for layout '{self.name}'; have {self.arm_names}")

    def arm_q(self, q: np.ndarray, arm: str | None = None) -> np.ndarray:
        """``(..., dim) -> (..., 6)``: the six joint angles for ``arm``."""
        q = np.asarray(q)
        if q.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim} for layout '{self.name}', got {q.shape}")
        return q[..., self._get_arm(arm).joint_slice]

    def grippers(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """``(..., dim) -> {arm: (...,)}`` gripper scalar per arm (skips None)."""
        q = np.asarray(q)
        out: dict[str, np.ndarray] = {}
        for a in self.arms:
            if a.gripper_index is not None:
                out[a.name] = q[..., a.gripper_index]
        return out

    @classmethod
    def infer(cls, dim: int) -> "JointLayout":
        """Pick the layout matching a flat-vector dimension (7 or 14)."""
        for layout in (SINGLE_ARM_7, BIMANUAL_14):
            if layout.dim == dim:
                return layout
        raise ValueError(f"no known joint layout for dim {dim} (expected 7 or 14)")


SINGLE_ARM_7 = JointLayout(
    name="single_arm_7",
    dim=7,
    arms=(ArmSlice(name="arm", joint_start=0, gripper_index=6),),
)

BIMANUAL_14 = JointLayout(
    name="bimanual_14",
    dim=14,
    arms=(
        ArmSlice(name="right", joint_start=0, gripper_index=6),
        ArmSlice(name="left", joint_start=7, gripper_index=13),
    ),
)
