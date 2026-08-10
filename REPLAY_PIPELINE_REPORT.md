# Replaying FastUMI recordings on the UR7e — what we built

## The goal
You record demonstrations with the **handheld FastUMI rig** (a camera + a T265
tracker). You want the **UR7e arm** to replay those demonstrations. Your robot's
replay setup reads **LeRobot datasets** (like the `WS04_*` dataset, "D2"). The
handheld recordings ("D1") are in a totally different format, so they can't be
replayed directly. This pipeline converts D1 → a D2-style LeRobot dataset that
drops straight into your existing replay.

## The core problem, in one line
The handheld recording stores **where the gripper tip was in space** (a position
and a rotation). The robot needs **joint angles** (how far to turn each of its 6
motors). Those are not the same thing — you have to *compute* the joint angles.

## What the pipeline does (4 steps)

1. **Pose → joint angles (inverse kinematics).**
   For every frame we solve "what 6 joint angles put the gripper exactly there?"
   using the solver that matches your real UR7e's conventions.

2. **Winding fix ("don't take the long way round").**
   A motor at −250° and at +110° aim the arm identically, but the robot would
   physically spin almost a full turn to get to the wrong one — this once sent a
   joint diving toward the floor. We re-anchor every trajectory to the arm's real
   resting pose so the first move is short and safe.

3. **Gripper openness.**
   The handheld rig has little markers on the gripper. We measure how far apart
   they appear in the video and turn that into an openness value from **0 (closed)
   to 1 (open)** — the same 0–1 scale the robot expects.

4. **Repackage into the LeRobot format.**
   We write everything out with the **exact same columns** your D2 replay reads
   (`action` = 6 joints + gripper, plus the matching state/lead/follower/velocity
   columns), as a LeRobot **v3.0** dataset folder.

## Why there are two steps / two Pythons
The IK solver only runs in your **FastUMI** conda env; the LeRobot writer only
runs in the **base** conda env. They can't share one process, so the tool runs
stage 1 in one, hands off a small intermediate file, then runs stage 2 in the
other. The wrapper script hides this — you run **one** command.

## Cameras — deliberately left out
Your robot replays **joints**, not pictures, so the dataset is written
**joints-only by default** (no video). This makes it tiny and fully portable,
and it avoids a broken video-decoding library on this machine. The original
handheld view had only one camera anyway; D2's other two camera streams simply
don't exist in a handheld recording. (Use `--with-video` if you ever want the
one camera included as `cam_high`.)

---

## How to run it

**Convert a whole task in one command:**
```bash
cd /home/nuc8/Downloads/FastUMI-Master
./replay_pipeline/make_replay_dataset.sh --task Pick_and_place_the_bottle
```
Output folder: `<dataset>/Pick_and_place_the_bottle_lerobot` (joints-only, ~1 MB).

**Useful options:**
- `--src /path/to/dataset` — convert a dataset by path instead of a config task name
- `--out /path/to/output` — choose the output folder
- `--fps 30` — set the fps your replay expects (default 20 = the recording's real rate)
- `--episodes 1,5,9` — convert only some episodes
- `--with-video` — also include the front camera as `cam_high`
- `--help` — full option list

## Moving it to the robot
1. Copy the whole output folder (e.g. `Pick_and_place_the_bottle_lerobot/`) to the
   robot's machine.
2. Point your existing replay at it as a local dataset (`root=<pasted path>`).
3. **Run nothing else there** — no conversion, no extra setup.

## Before the first hardware replay — 3 checks + go slow
1. **Same arm & home pose.** The joints were solved for this rig and re-wound to
   this task's home. Confirm the arm starts near
   `[93.7, −54.8, 125.4, −250.6, −95.4, −184.3]°`.
2. **fps.** If your replay reads fps from the dataset, keep 20; if it assumes 30,
   convert with `--fps 30` (otherwise it plays 1.5× fast).
3. **Gripper direction.** `action`'s last value is `1 = open`. Confirm that
   matches how your replay drives the Robotiq Hand-E.

**Always** run the first replay at low speed (~10%) with a hand on the e-stop.
The simulation (see below) is a free dry-run but it sets joint angles directly,
so it can't show the real startup slew.

## Checking a conversion in simulation (optional)
To eyeball an episode before hardware — recorded footage next to the MuJoCo UR7e
driven by the *computed* joints:
```bash
/home/nuc8/miniconda3/envs/FastUMI/bin/python3 viz/visualize_episode_mujoco.py \
    /home/nuc8/FastUMI/dataset/Pick_and_place_the_bottle/episode_5.hdf5
```

---

## What was produced
- `replay_pipeline/d1_to_lerobot_stage1.py` — stage 1: pose → joints + gripper + winding fix (FastUMI env)
- `replay_pipeline/d1_to_lerobot_stage2.py` — stage 2: joints → LeRobot v3.0 dataset (base env)
- `replay_pipeline/make_replay_dataset.sh` — one-command wrapper around both stages
- `Pick_and_place_the_bottle_lerobot/` — the ready-to-copy dataset (25 episodes, 2862 frames)

## Honest limitations
- The joints are **computed** from a handheld demo, so they're only as accurate as
  the rig's base calibration. Sanity-check each conversion (all 25 here were
  100% reachable, smooth, and started at the arm's home).
- Duplicated columns (lead / follower-commanded / follower-achieved) are all set
  to the one computed trajectory — a handheld demo has no separate leader and
  follower arms to record, so these are reconstructed, not independently measured.
