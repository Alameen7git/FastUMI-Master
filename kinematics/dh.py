"""Denavit-Hartenberg parameters for UR-family arms.

Holds the :class:`DHParams` dataclass, a loader for ``voraus_deploy`` robot
config JSON, and checked-in UR7e constants so the module does not depend on a
sibling-repo path at runtime (training / vast workers do not have
``~/repo/voraus_deploy``). A unit test asserts the constants match the voraus
JSON when that repo is present.

Convention (standard / "distal" DH):

    T_i = Rz(theta_i + theta_offset_i) @ Tz(d_i) @ Tx(a_i) @ Rx(alpha_i)

where ``theta_i`` is the joint encoder reading and ``theta_offset_i`` is a
constant offset. ``alpha`` and ``theta`` are stored in degrees in the voraus
JSON and converted to radians on load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DHParams",
    "load_dh_from_voraus_json",
    "UR7E_DH",
    "UR3E_DH",
    "get_dh",
]


@dataclass(frozen=True)
class DHParams:
    """One row of a standard-DH table (lengths in metres, angles in radians)."""

    a: float
    alpha_rad: float
    d: float
    theta_offset_rad: float

    @classmethod
    def from_voraus_entry(cls, entry: dict) -> "DHParams":
        dh = entry["DH"]
        return cls(
            a=float(dh["a"]),
            alpha_rad=float(np.deg2rad(float(dh["alpha"]))),
            d=float(dh["d"]),
            theta_offset_rad=float(np.deg2rad(float(dh["theta"]))),
        )


def load_dh_from_voraus_json(path: str | Path) -> list[DHParams]:
    """Parse the ``Axes`` array from a voraus_deploy robot config JSON."""
    p = Path(path).expanduser()
    with p.open() as f:
        cfg = json.load(f)
    return [DHParams.from_voraus_entry(e) for e in cfg["Axes"]]


# ---------------------------------------------------------------------------
# Checked-in UR7e constants
# ---------------------------------------------------------------------------
#
# Source: ~/repo/voraus_deploy/robots/UR/UR_UR7E/UR_UR7E.json (RobotConfigVersion 4),
# read 2026-06-11. Values below mirror that file exactly (alpha / theta_offset
# converted from the stored degrees). The test ``test_dh.py`` re-derives these
# from the JSON when the voraus repo is present, so drift is caught on dev
# machines.
#
#   joint   a (m)     alpha (deg)   d (m)      theta_off (deg)
#   1       0.0       -90           0.1625     180
#   2       0.425       0           0.0          0
#   3       0.3922      0           0.0          0
#   4       0.0       -90           0.133        0
#   5       0.0       +90           0.0997       0
#   6       0.0         0           0.0996     180
UR7E_DH: tuple[DHParams, ...] = (
    DHParams(a=0.0, alpha_rad=float(np.deg2rad(-90.0)), d=0.1625, theta_offset_rad=float(np.deg2rad(180.0))),
    DHParams(a=0.425, alpha_rad=0.0, d=0.0, theta_offset_rad=0.0),
    DHParams(a=0.3922, alpha_rad=0.0, d=0.0, theta_offset_rad=0.0),
    DHParams(a=0.0, alpha_rad=float(np.deg2rad(-90.0)), d=0.133, theta_offset_rad=0.0),
    DHParams(a=0.0, alpha_rad=float(np.deg2rad(90.0)), d=0.0997, theta_offset_rad=0.0),
    DHParams(a=0.0, alpha_rad=0.0, d=0.0996, theta_offset_rad=float(np.deg2rad(180.0))),
)

# ---------------------------------------------------------------------------
# Checked-in UR3e constants
# ---------------------------------------------------------------------------
#
# Source: voraus_deploy robots/UR/UR_UR3E/UR_UR3E.json (RobotConfigVersion 4).
# Same UR 6R structure / convention as UR7e (alpha [-90,0,0,-90,+90,0],
# 180-deg theta-offsets on joints 1 & 6), shorter links.
#
#   joint   a (m)      alpha (deg)   d (m)       theta_off (deg)
#   1       0.0        -90           0.15185     180
#   2       0.24355      0           0.0           0
#   3       0.2132       0           0.0           0
#   4       0.0        -90           0.13105       0
#   5       0.0        +90           0.08535       0
#   6       0.0          0           0.0921      180
UR3E_DH: tuple[DHParams, ...] = (
    DHParams(a=0.0, alpha_rad=float(np.deg2rad(-90.0)), d=0.15185, theta_offset_rad=float(np.deg2rad(180.0))),
    DHParams(a=0.24355, alpha_rad=0.0, d=0.0, theta_offset_rad=0.0),
    DHParams(a=0.2132, alpha_rad=0.0, d=0.0, theta_offset_rad=0.0),
    DHParams(a=0.0, alpha_rad=float(np.deg2rad(-90.0)), d=0.13105, theta_offset_rad=0.0),
    DHParams(a=0.0, alpha_rad=float(np.deg2rad(90.0)), d=0.08535, theta_offset_rad=0.0),
    DHParams(a=0.0, alpha_rad=0.0, d=0.0921, theta_offset_rad=float(np.deg2rad(180.0))),
)

_DH_REGISTRY: dict[str, tuple[DHParams, ...]] = {
    "ur7e": UR7E_DH,
    "ur3e": UR3E_DH,
}


def get_dh(name: str) -> tuple[DHParams, ...]:
    """Look up a checked-in DH table by name (``"ur7e"`` | ``"ur3e"``)."""
    key = name.lower()
    if key not in _DH_REGISTRY:
        raise KeyError(f"unknown DH table '{name}'; known: {sorted(_DH_REGISTRY)}")
    return _DH_REGISTRY[key]
