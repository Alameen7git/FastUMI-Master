"""Closed-form analytic inverse kinematics for UR-structured 6-DOF arms.

UR-family arms (UR3/5/7e/10/16, and any arm whose DH table normalises to the
same structure) admit a closed-form IK with at most **8 solutions**,
enumerated by three binary branch choices:

* **shoulder** (S): the base joint can reach the wrist centre "forward" or
  "backward" across the shoulder-offset cylinder,
* **elbow** (E): elbow on either side of the shoulder-wrist chord,
* **wrist** (W): wrist joint 5 folded to either side.

``branch_id = 4*S + 2*E + W`` (0..7). The bits are *deterministic functions
of the joint vector* (see :func:`classify_branch`), so recorded joint data can
be labelled with its branch and a policy can be trained to predict it —
turning IK at inference into branch-conditioned solution *selection* rather
than seed-dependent iteration.

DH normalisation
================

The textbook closed form assumes the canonical UR table::

    alpha = [+90, 0, 0, +90, -90, 0] deg,  theta offsets all 0,
    a     = [0, a2, a3, 0, 0, 0],          d = [d1, 0, 0, d4, d5, d6]

The tables used by our robot stack (voraus_deploy) carry 180-degree theta
offsets and flipped alpha/a signs. The two are *identical maps* q -> SE(3):
a row with offset pi factors as a canonical row times a trailing ``Rz(pi)``::

    Rz(q+pi) Tz(d) Tx(a) Rx(alpha)  =  Rz(q) Tz(d) Tx(-a) Rx(-alpha) Rz(pi)

and the trailing ``Rz(pi)`` propagates into the next row's offset.
:meth:`URGeometry.from_dh` walks the chain carrying that rotation, emits the
canonical row sequence, validates the UR structure, and numerically verifies
the equivalence at construction — so an unsupported table fails loudly
instead of producing mirrored solutions. (For the UR7e voraus table the
carry cancels at joint 6; a residual carry is kept as ``flange_rz``.)

Closed form (canonical frame; full derivation)
==============================================

Let ``T = [R | p]`` be the target base->flange pose and ``z6 = R[:,2]``.
With ``D = d2+d3+d4`` (the lateral offset between the joint-1 axis and the
wrist plane; translations along the parallel 2-3-4 axes commute and sum):

1. Wrist centre ``o5 = p - d6*z6``. Joint axes 2,3,4 are parallel to
   ``z1 = (sin q1, -cos q1, 0)`` and every chain offset along them sums to
   ``D``, hence ``o5 . z1 = D``:  with ``psi = atan2(o5_y, o5_x)`` and
   ``rho = |o5_xy|``: ``rho*sin(q1 - psi) = D`` giving
   ``q1 = psi + asin(D/rho)`` (S=0) or ``psi + pi - asin(D/rho)`` (S=1).
2. ``p . z1 = D + d6*(z6 . z1)`` and ``z6 . z1 = cos q5``, hence
   ``cos q5 = (p_x sin q1 - p_y cos q1 - D)/d6``, ``q5 = +/- acos(.)``.
3. Expressing ``z1`` in frame 6 gives ``(s5 c6, -s5 s6, c5)``; equating to
   ``R^T z1`` yields ``q6 = atan2(-g1*sign(s5), g0*sign(s5))`` with
   ``g0 = R00 s1 - R10 c1``, ``g1 = R01 s1 - R11 c1``. Undefined when
   ``sin q5 ~ 0`` (wrist singularity) — ``q6`` is then free and set from
   ``q6_if_singular``.
4. With q1, q5, q6 fixed: ``T14 = T01^-1 T T56^-1 T45^-1`` reduces to a
   planar 2R chain: ``T14[:2,3] = Rot(q2) . (a2 + a3 c3, a3 s3)`` so
   ``cos q3 = (x^2+y^2-a2^2-a3^2)/(2 a2 a3)``, ``q3 = +/- acos(.)``,
   ``q2 = atan2(y,x) - atan2(a3 s3, a2 + a3 c3)``, and orientation closure
   ``q2+q3+q4 = atan2(T14[1,0], T14[0,0])`` fixes ``q4``.

Solutions are exact to machine precision (verified FK(sol) == target at
~1e-10 over randomised round-trips, including all 8 branches).

Limitations
===========

* Valid for the *nominal* DH model only (which is what drives the robots).
* At branch boundaries (q3 or q5 ~ 0/pi, |D/rho| ~ 1) coincident branches
  are deduplicated and the branch label uses documented >=0 tie-breaks.
* Targets with ``rho < |D|`` or out of reach return zero solutions
  (never NaNs).
* Within ~1e-8 rad of a fold (q3 or q5 at 0/pi) the extracted angle is only
  accurate to ~sqrt(machine eps) because ``acos`` is ill-conditioned at
  |cos| ~ 1 — pose reproduction degrades from ~1e-12 to ~1e-7 there. This is
  inherent to every closed-form UR IK, not a defect of this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .arm import ArmModel, _dh_joint_transforms
from .dh import DHParams
from .layouts import JointLayout
from .rotations import matrix_from_pose9

__all__ = [
    "URGeometry",
    "ik",
    "ik_branch",
    "ik_nearest",
    "classify_branch",
    "branch_bits",
    "refine",
    "pose_action_to_joint_action",
]

_STRUCT_TOL = 1e-9  # tolerance for DH structure checks (radians / metres)
_CLAMP_TOL = 1e-9  # |cos| may exceed 1 by float noise before clamping
_SINGULAR_TOL = 1e-8  # sin(angle) below this => branch pair coincides
_HALF_PI = np.pi / 2.0


def _wrap(x):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _rz4(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    return T


def _se3_inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (rotation transpose, exact)."""
    Ti = np.eye(4)
    Rt = T[:3, :3].T
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ T[:3, 3]
    return Ti


# ---------------------------------------------------------------------------
# Geometry extraction / DH normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class URGeometry:
    """Canonical UR IK parameters extracted from a (possibly offset) DH table.

    ``a2``/``a3`` are signed as in the canonical convention (negative for the
    UR7e). ``D = d2+d3+d4``. ``flange_rz`` is a residual constant rotation
    about the flange z-axis if the table's offset carry does not cancel:
    ``FK_table(q) = FK_canonical(q) @ Rz(flange_rz)``.
    """

    d1: float
    a2: float
    a3: float
    D: float
    d5: float
    d6: float
    flange_rz: float
    canonical_dh: tuple[DHParams, ...]

    @classmethod
    def from_dh(cls, dh: tuple[DHParams, ...]) -> "URGeometry":
        return _geometry_from_dh(tuple(dh))

    def row_transform(self, index: int, q: float | np.ndarray) -> np.ndarray:
        """Canonical single-row transform ``T_{index,index+1}(q)`` (0-based)."""
        row = self.canonical_dh[index]
        q = np.asarray(q, dtype=np.float64)
        return _dh_joint_transforms((row,), q[..., None])[..., 0, :, :]


@lru_cache(maxsize=8)
def _geometry_from_dh(dh: tuple[DHParams, ...]) -> URGeometry:
    if len(dh) != 6:
        raise ValueError(f"UR analytic IK needs a 6-row DH table, got {len(dh)}")

    # --- Normalise: absorb {0, pi} theta offsets via the Rz(pi) carry. ---
    carry = 0.0
    rows: list[tuple[float, float, float]] = []  # (a, alpha, d)
    for i, p in enumerate(dh):
        off = float(_wrap(p.theta_offset_rad + carry))
        if abs(off) < _STRUCT_TOL:
            rows.append((p.a, p.alpha_rad, p.d))
            carry = 0.0
        elif abs(abs(off) - np.pi) < _STRUCT_TOL:
            rows.append((-p.a, -p.alpha_rad, p.d))
            carry = np.pi
        else:
            raise ValueError(
                f"DH row {i + 1}: effective theta offset {np.rad2deg(off):.3f} deg is not "
                "0 or 180 — table cannot be normalised to the canonical UR form"
            )
    flange_rz = carry

    a = [r[0] for r in rows]
    alpha = [r[1] for r in rows]
    d = [r[2] for r in rows]

    # --- Validate the canonical UR structure. ---
    expected_alpha = [_HALF_PI, 0.0, 0.0, _HALF_PI, -_HALF_PI, 0.0]
    for i, (got, want) in enumerate(zip(alpha, expected_alpha)):
        if abs(got - want) > _STRUCT_TOL:
            raise ValueError(
                f"DH row {i + 1}: normalised alpha {np.rad2deg(got):.3f} deg != "
                f"{np.rad2deg(want):.0f} deg — not a UR-structured arm "
                "(expected alpha pattern [+90, 0, 0, +90, -90, 0])"
            )
    for i in (0, 3, 4, 5):
        if abs(a[i]) > _STRUCT_TOL:
            raise ValueError(f"DH row {i + 1}: normalised a={a[i]:.6f} != 0 — not UR-structured")
    if abs(a[1]) < _STRUCT_TOL or abs(a[2]) < _STRUCT_TOL:
        raise ValueError("DH rows 2/3: link lengths a2, a3 must be non-zero")
    if abs(d[5]) < _STRUCT_TOL:
        raise ValueError("DH row 6: d6 must be non-zero (needed for the wrist closed form)")

    # d2, d3 commute along the parallel 2-3-4 axes and fold into D = d2+d3+d4.
    D = d[1] + d[2] + d[3]
    canonical = (
        DHParams(a=0.0, alpha_rad=_HALF_PI, d=d[0], theta_offset_rad=0.0),
        DHParams(a=a[1], alpha_rad=0.0, d=0.0, theta_offset_rad=0.0),
        DHParams(a=a[2], alpha_rad=0.0, d=0.0, theta_offset_rad=0.0),
        DHParams(a=0.0, alpha_rad=_HALF_PI, d=D, theta_offset_rad=0.0),
        DHParams(a=0.0, alpha_rad=-_HALF_PI, d=d[4], theta_offset_rad=0.0),
        DHParams(a=0.0, alpha_rad=0.0, d=d[5], theta_offset_rad=0.0),
    )
    geom = URGeometry(
        d1=d[0],
        a2=a[1],
        a3=a[2],
        D=D,
        d5=d[4],
        d6=d[5],
        flange_rz=flange_rz,
        canonical_dh=canonical,
    )

    # --- Numeric self-check: original table FK == canonical FK @ Rz(carry). ---
    probe = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, -1.1, 1.7, -2.0, -1.4, 2.6],
            [-2.4, 0.8, -0.6, 1.2, 2.9, -0.9],
            [1.6, 1.6, -2.8, 0.4, -0.2, -3.1],
        ]
    )
    T_orig = ArmModel(dh, name="orig").fk(probe)
    T_canon = ArmModel(canonical, name="canon").fk(probe) @ _rz4(flange_rz)
    err = float(np.max(np.abs(T_orig - T_canon)))
    if err > 1e-9:
        raise ValueError(
            f"DH normalisation self-check failed (max FK deviation {err:.3e}) — "
            "table is not equivalent to the canonical UR form"
        )
    return geom


# ---------------------------------------------------------------------------
# Branch classification (deterministic label from a joint vector)
# ---------------------------------------------------------------------------


def branch_bits(arm: ArmModel, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(..., 6) -> (S, E, W)`` boolean arrays, the three branch bits of ``q``.

    Tie-breaks (measure-zero singular boundaries): a wrapped angle of exactly
    0 classifies as bit 0; ``|wrap(q1 - psi)| <= pi/2`` classifies as S=0.
    """
    geom = URGeometry.from_dh(arm.dh)
    q = np.asarray(q, dtype=np.float64)
    T = arm.fk(q)  # flange pose; Rz(flange_rz) does not move p or z6
    p = T[..., :3, 3]
    z6 = T[..., :3, 2]
    o5 = p - geom.d6 * z6
    psi = np.arctan2(o5[..., 1], o5[..., 0])
    delta = _wrap(q[..., 0] - psi)
    S = np.abs(delta) > _HALF_PI
    E = _wrap(q[..., 2]) < 0.0
    W = _wrap(q[..., 4]) < 0.0
    return S, E, W


def classify_branch(arm: ArmModel, q: np.ndarray) -> np.ndarray:
    """``(..., 6) -> (...,)`` int branch id ``4*S + 2*E + W`` in 0..7."""
    S, E, W = branch_bits(arm, q)
    return (4 * S.astype(np.int64) + 2 * E.astype(np.int64) + W.astype(np.int64)).astype(np.int64)


# ---------------------------------------------------------------------------
# Analytic IK
# ---------------------------------------------------------------------------


def _flange_target(arm: ArmModel, T_target: np.ndarray, tool: str) -> np.ndarray:
    """Convert a base->tool target into a canonical base->flange target."""
    if tool not in arm.tools:
        raise KeyError(f"unknown tool '{tool}'; registered: {sorted(arm.tools)}")
    T_target = np.asarray(T_target, dtype=np.float64)
    if T_target.shape != (4, 4):
        raise ValueError(f"T_target must be (4, 4), got {T_target.shape}")
    return T_target @ _se3_inv(arm.tools[tool])


def ik(
    arm: ArmModel,
    T_target: np.ndarray,
    tool: str = "flange",
    q6_if_singular: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """All analytic IK solutions for a base->``tool`` target pose.

    Args:
        arm: the arm model (DH table must normalise to the UR structure).
        T_target: (4, 4) target pose of ``tool`` in the arm base frame.
        tool: which registered tool frame ``T_target`` refers to.
        q6_if_singular: value assigned to q6 at the wrist singularity
            (``sin q5 ~ 0``), where q6 is geometrically free.

    Returns:
        ``(solutions, branch_ids)`` with shapes (K, 6) and (K,), K <= 8,
        sorted by branch id. Joint angles are wrapped to [-pi, pi).
        K = 0 (empty arrays) when the target is unreachable.
    """
    geom = URGeometry.from_dh(arm.dh)
    T06 = _flange_target(arm, T_target, tool) @ _rz4(-geom.flange_rz)

    R = T06[:3, :3]
    p = T06[:3, 3]
    o5 = p - geom.d6 * R[:, 2]
    rho = float(np.hypot(o5[0], o5[1]))
    psi = float(np.arctan2(o5[1], o5[0]))

    sols: list[np.ndarray] = []
    bids: list[int] = []

    # Shoulder: rho*sin(q1 - psi) = D.
    ratio = geom.D / rho if rho > 0.0 else np.inf
    if abs(ratio) > 1.0 + _CLAMP_TOL:
        return np.empty((0, 6)), np.empty((0,), dtype=np.int64)
    ratio = float(np.clip(ratio, -1.0, 1.0))
    shoulder_degenerate = abs(np.cos(np.arcsin(ratio))) < _SINGULAR_TOL

    for S in (0, 1):
        if S == 1 and shoulder_degenerate:
            continue  # both shoulder branches coincide
        q1 = psi + np.arcsin(ratio) if S == 0 else psi + np.pi - np.arcsin(ratio)
        s1, c1 = np.sin(q1), np.cos(q1)

        # Wrist: cos q5 from the z1-component of p.
        c5 = (p[0] * s1 - p[1] * c1 - geom.D) / geom.d6
        if abs(c5) > 1.0 + _CLAMP_TOL:
            continue
        c5 = float(np.clip(c5, -1.0, 1.0))
        wrist_degenerate = abs(np.sin(np.arccos(c5))) < _SINGULAR_TOL

        for W in (0, 1):
            if W == 1 and wrist_degenerate:
                continue  # q5 = 0 or pi: both wrist branches coincide
            q5 = float(np.arccos(c5)) if W == 0 else -float(np.arccos(c5))
            s5 = np.sin(q5)

            # q6 from z1 expressed in frame 6: (s5 c6, -s5 s6, c5).
            if abs(s5) < _SINGULAR_TOL:
                q6 = float(q6_if_singular)
            else:
                g0 = R[0, 0] * s1 - R[1, 0] * c1
                g1 = R[0, 1] * s1 - R[1, 1] * c1
                q6 = float(np.arctan2(-g1 * np.sign(s5), g0 * np.sign(s5)))

            # Reduce to the planar 2R chain: T14 = T01^-1 T06 T56^-1 T45^-1.
            T01 = geom.row_transform(0, q1)
            T45 = geom.row_transform(4, q5)
            T56 = geom.row_transform(5, q6)
            T14 = _se3_inv(T01) @ T06 @ _se3_inv(T56) @ _se3_inv(T45)
            x, y = float(T14[0, 3]), float(T14[1, 3])

            c3 = (x * x + y * y - geom.a2**2 - geom.a3**2) / (2.0 * geom.a2 * geom.a3)
            if abs(c3) > 1.0 + _CLAMP_TOL:
                continue
            c3 = float(np.clip(c3, -1.0, 1.0))
            elbow_degenerate = abs(np.sin(np.arccos(c3))) < _SINGULAR_TOL
            q234 = float(np.arctan2(T14[1, 0], T14[0, 0]))

            for E in (0, 1):
                if E == 1 and elbow_degenerate:
                    continue  # straight/folded elbow: branches coincide
                q3 = float(np.arccos(c3)) if E == 0 else -float(np.arccos(c3))
                q2 = float(np.arctan2(y, x) - np.arctan2(geom.a3 * np.sin(q3), geom.a2 + geom.a3 * np.cos(q3)))
                q4 = q234 - q2 - q3
                sol = _wrap(np.array([q1, q2, q3, q4, q5, q6]))
                # E/W bits come from the *wrapped* solution, not the loop
                # indices: at an exact fold (q3 or q5 == pi) acos returns pi,
                # which wraps to -pi and classifies as bit 1 — deriving the id
                # from the solution keeps ik() consistent with
                # classify_branch by construction, even on that knife-edge.
                bids.append(4 * S + 2 * int(sol[2] < 0.0) + int(sol[4] < 0.0))
                sols.append(sol)

    if not sols:
        return np.empty((0, 6)), np.empty((0,), dtype=np.int64)
    order = np.argsort(bids, kind="stable")
    return np.stack(sols)[order], np.asarray(bids, dtype=np.int64)[order]


def ik_branch(
    arm: ArmModel,
    T_target: np.ndarray,
    branch: int,
    tool: str = "flange",
    q6_if_singular: float = 0.0,
) -> np.ndarray | None:
    """The solution on a specific branch (0..7), or None if unreachable there."""
    sols, bids = ik(arm, T_target, tool=tool, q6_if_singular=q6_if_singular)
    hit = np.nonzero(bids == int(branch))[0]
    return sols[hit[0]] if len(hit) else None


def _unwrap_towards(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Shift each angle by multiples of 2*pi to land nearest ``q_ref``."""
    return q + 2.0 * np.pi * np.round((q_ref - q) / (2.0 * np.pi))


def ik_nearest(
    arm: ArmModel,
    T_target: np.ndarray,
    q_seed: np.ndarray,
    tool: str = "flange",
    prefer_branch_of_seed: bool = True,
) -> np.ndarray | None:
    """The IK solution closest to ``q_seed`` (6-vector, radians).

    Each candidate is first unwrapped joint-wise toward the seed (so encoder
    windings beyond +/-pi, e.g. q6 ~ -3.15 in our datasets, are preserved),
    then ranked by L2 distance. With ``prefer_branch_of_seed`` the search is
    restricted to the seed's kinematic branch when that branch is reachable —
    the guard against silent elbow/shoulder/wrist flips during a rollout —
    falling back to all branches otherwise. Returns None when the target is
    unreachable.
    """
    q_seed = np.asarray(q_seed, dtype=np.float64).reshape(6)
    sols, bids = ik(arm, T_target, tool=tool, q6_if_singular=float(q_seed[5]))
    if len(sols) == 0:
        return None
    if prefer_branch_of_seed:
        seed_bid = int(classify_branch(arm, q_seed))
        mask = bids == seed_bid
        if mask.any():
            sols = sols[mask]
    candidates = np.stack([_unwrap_towards(s, q_seed) for s in sols])
    dists = np.linalg.norm(candidates - q_seed, axis=1)
    return candidates[int(np.argmin(dists))]


# ---------------------------------------------------------------------------
# Optional numerical polish (pure numpy; for future calibrated models)
# ---------------------------------------------------------------------------


def refine(
    arm: ArmModel,
    q0: np.ndarray,
    T_target: np.ndarray,
    tool: str = "flange",
    iters: int = 10,
    tol: float = 1e-12,
) -> np.ndarray:
    """Damped Gauss-Newton polish of ``q0`` toward ``T_target``.

    With the nominal DH model the analytic solutions are already exact; this
    exists for future *calibrated* models whose FK deviates slightly from the
    table. Uses a central-difference Jacobian on the 6D pose error
    ``[position; rotation-vector]``.
    """
    from scipy.spatial.transform import Rotation

    T_goal = np.asarray(T_target, dtype=np.float64)
    q = np.asarray(q0, dtype=np.float64).reshape(6).copy()

    def err(qv: np.ndarray) -> np.ndarray:
        T = arm.fk(qv, tool=tool)
        e_p = T[:3, 3] - T_goal[:3, 3]
        e_r = Rotation.from_matrix(T[:3, :3] @ T_goal[:3, :3].T).as_rotvec()
        return np.concatenate([e_p, e_r])

    h = 1e-6
    for _ in range(iters):
        e = err(q)
        if np.linalg.norm(e) < tol:
            break
        J = np.empty((6, 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = h
            J[:, j] = (err(q + dq) - err(q - dq)) / (2.0 * h)
        q = q - np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
    return q


# ---------------------------------------------------------------------------
# Inference adapter: pose-space action -> joint-space action
# ---------------------------------------------------------------------------


def pose_action_to_joint_action(
    pose9: np.ndarray,
    q_current: np.ndarray,
    arm: ArmModel,
    layout: JointLayout,
    tool: str = "flange",
    gripper_passthrough: np.ndarray | None = None,
) -> np.ndarray | None:
    """Convert a pose9-per-arm action into a flat joint-vector action.

    Args:
        pose9: (9 * n_arms,) pose9 blocks ordered as ``layout.arms``.
        q_current: (layout.dim,) current joint vector — the IK seed.
        arm: arm model applied to every arm slice (same hardware per arm).
        tool: tool frame the pose9 targets refer to.
        gripper_passthrough: optional gripper values in arm order; defaults
            to carrying the grippers over from ``q_current``.

    Returns:
        (layout.dim,) joint action, or None if IK fails for any arm —
        callers should treat None as "hold position".
    """
    pose9 = np.asarray(pose9, dtype=np.float64).reshape(len(layout.arms), 9)
    q_current = np.asarray(q_current, dtype=np.float64).reshape(layout.dim)
    out = q_current.copy()
    for i, arm_slice in enumerate(layout.arms):
        T_target = matrix_from_pose9(pose9[i])
        q_arm = ik_nearest(arm, T_target, q_current[arm_slice.joint_slice], tool=tool)
        if q_arm is None:
            return None
        out[arm_slice.joint_slice] = q_arm
        if arm_slice.gripper_index is not None and gripper_passthrough is not None:
            out[arm_slice.gripper_index] = float(gripper_passthrough[i])
    return out
