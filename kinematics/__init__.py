"""Analytic UR-family kinematics (FK, 6D rotations, closed-form UR IK).

Vendored from ``embodied_ai_ml`` v1.4.8, ``src/embodied_ai_ml/kinematics/``
(copied 2026-08-07 from /home/nuc8/dev/embodied_ai_ml-main, an unpacked source
tree with no git history).

Why vendored: data_processing_to_joint.py depends on this for every frame's IK,
but upstream lives outside any repo on a single machine, so a clone elsewhere
failed at import. Upstream also targets Python >=3.12 and its package
``__init__`` eagerly imports lerobot/torch, neither of which is installed here
(Python 3.8) -- the old workaround was a ``sys.modules`` stub to import this
submodule without triggering the rest of the package. Copying these five
modules removes both problems: they need only numpy + scipy.

Deliberately NOT copied: ``labeller.py`` (PoseLabeller), the only module
requiring torch and ``embodied_ai_ml.actions.blocks``. Nothing in this project
uses it. The remaining modules import each other relatively and reach nothing
outside this directory.

Local edits: none. Keep it that way -- if upstream needs changing, change it
upstream and re-copy, so this stays a clean snapshot.
"""

from .arm import ArmModel
from .dh import UR7E_DH, DHParams, get_dh, load_dh_from_voraus_json
from .ik_ur import classify_branch, ik, ik_branch, ik_nearest
from .layouts import BIMANUAL_14, SINGLE_ARM_7, JointLayout

__all__ = [
    "ArmModel",
    "DHParams",
    "UR7E_DH",
    "get_dh",
    "load_dh_from_voraus_json",
    "JointLayout",
    "SINGLE_ARM_7",
    "BIMANUAL_14",
    "ik",
    "ik_branch",
    "ik_nearest",
    "classify_branch",
]
