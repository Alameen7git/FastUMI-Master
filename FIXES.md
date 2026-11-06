# Fixes — UMI → UR7e replay pipeline

Changes made 2026-11-05 / 06 while converting the T0 bottle pick-and-place datasets
(`UMI_2026-11-05_T0_a1`, `UMI_2026-11-06_T0_N1`) for replay on the UR7e + Robotiq 2F-140.
All of them are committed on branch `t265-home-check` (folder `FastUMI-Master-homecheck`, based
on `ur7e-dev7.0` @ `12c13f8`), local only, not pushed. The web UI home-check changes in this
folder are not committed yet: they overlap with newer `origin/ur7e-dev7.0` commits and need merging.

Verified on hardware: `UMI_2026-11-06_T0_N1_ep235_gripOC_offsetfix_joint` (episodes 2, 3, 5)
replayed correctly on the UR7e on 2026-11-06.

| # | Fix | Type | Confidence |
|---|---|---|---|
| 1 | Tool tilt turned sideways | Bug | Certain |
| 2 | T265 → tip offset on the wrong axis | Bug | Very likely |
| 3 | Gripper calibration | Wrong calibration | Certain |
| 4 | Gripper jitter | Weak estimate | Certain |
| 5 | Open/close gripper | Design choice (option) | — |

---

## 1. Tool tilt turned sideways (bug)

**Symptom.** When the UMI tilted up after picking (looking forward to place), the robot tool in
the sim turned sideways instead of tilting up. Up to ~90° off during placing; correct at the start.

**Cause.** `remap_t265_to_robot` converted the T265 *position* into robot axes
(`_T265_TO_ROBOT`, a 90° turn about vertical) but applied the T265 *rotation* without the same
conversion. Positions and rotations were in frames 90° apart.

**Fix.** `conversion/data_processing_to_joint.py` — `remap_t265_to_robot(..., ref_quat=None)`:
with the episode's frame-0 quaternion, the rotation since frame 0 is expressed in robot axes
(`M · R_t · R_0ᵀ · Mᵀ · R_0 · Mᵀ`). Frame 0 is unchanged, so the start pose and base calibration
are unchanged. Callers pass `ref_quat=qpos[0, 3:7]`:
- `viz/validate_joint_trajectory.py` (`convert_and_diagnose`, used by the replay pipeline)
- `conversion/data_processing_to_joint.py` (batch conversion)

**Evidence.** Pure maths (frame consistency) — no numbers were tuned. Sim and real replay match the
recorded motion.

## 2. T265 → UMI-tip offset applied along the wrong axis (bug)

**Symptom.** On the robot the fingers pressed into the tray at the pick (too low) and the bottle
was placed ~10 cm too high.

**Cause.** The tip position is the T265 position plus the measured offset (`offset.x` = 0.146 m
along the tool, `offset.z` = 0.159 m sideways). The sideways part was applied along the flange **x**
axis; on this rig the UMI sits 90° about the tool axis from the flange (the same reason the 2F-140
had to be turned 90° in the sim), so it belongs on the flange **y** axis. The error grows with how
far the UMI has turned since the start: small at the pick (pointing down as at the start), large at
the place (tilted ~75°).

**Fix.** `viz/validate_joint_trajectory.py` — `convert_and_diagnose`: sideways offset along `ori[:, 1]`,
anchored at frame 0 (`+ offset.z · (ori0[:, 1] − ori0[:, 0])`) so the start pose is unchanged.

**Effect** (lowest tool point, robot base frame; table is 5 cm below the base):

| Episode | Pick: before → after | Place: before → after |
|---|---|---|
| N1 ep 2 | −14 → +25 mm | 406 → 276 mm |
| N1 ep 3 | −33 → +10 mm | 401 → 266 mm |
| N1 ep 5 | −11 → +26 mm | 416 → 281 mm |

**Evidence.** Of the four possible axes (±x, ±y) only −y fixes both observations (pick too low *and*
place too high); it also matches the 90° gripper mount. Inferred, not measured — see open items.

**Note.** The batch script `conversion/data_processing_to_joint.py` computes this offset but then
writes the un-offset position (`normalized_qpos[i, :] = [x_base, ...]`), so it is not affected;
the replay pipeline uses `validate_joint_trajectory.convert_and_diagnose`.

## 3. Gripper calibration (wrong values)

**Symptom.** Gripper read fully open (1.00) in every frame of the T0_a1 data.

**Cause.** `marker_min` / `marker_max` (116 / 552 px) were from an older tag setup; with the current
tags almost every distance was above 552 px → clipped to fully open.

**Fix.** `config/config.json`: `marker_min` **320**, `marker_max` **835** (px, 1920×1080), measured
with the live camera (fully closed 320–332, fully open 803–835, both tags seen 96%).

## 4. Gripper jitter (weak estimate)

**Symptom.** Gripper flickered open/closed in the sim.

**Cause.** In frames where only one tag is seen (~25%), `get_gripper_width` guessed the width from
that tag's distance to the image centre. Those guesses scatter ±0.11 of full range.

**Fix.** `conversion/data_processing_to_joint.py` — `get_gripper_width` rewritten:
1. width from frames where **both** tags are seen;
2. gaps up to `GRIPPER_MAX_GAP` (15) frames: linear interpolation;
3. longer gaps: the old one-tag guess;
4. `GRIPPER_MEDIAN` (5)-frame median filter.

Always returns exactly one value per frame. Jumps > 0.05 per frame: 96 → 16 (the rest are real
open/close moves); value while holding the bottle unchanged (0.66).

## 5. Open/close gripper (design choice, optional)

**Why.** The robot's gripper is position-controlled (stack: 0–1, 1 = open, converted to the Robotiq
register by the voraus service). The UMI value while holding the bottle is the bottle's width
(~0.7), so the 2F-140 would stop at ~95 mm and not grip. The robot's own recordings
(`WS04_T000.0_Jul29_T1`) command ~0.0 while holding.

**Change.** New option `--gripper-binary THRESH` in `replay_pipeline/make_replay_dataset.sh` and
`replay_pipeline/d1_to_lerobot_stage1.py` (`binarize_gripper`): 0.0 below THRESH, 1.0 above, with
±0.05 hysteresis and runs shorter than 5 frames removed. Default off (continuous 0–1).
Used with `--gripper-binary 0.9`.

**Trade-off.** Partial-opening information is lost; 0.9 is a chosen threshold.

---

## How the verified dataset was made

```bash
cd replay_pipeline
./make_replay_dataset.sh --src /home/nuc8/FastUMI/dataset/UMI_2026-11-06_T0_N1 \
    --out /home/nuc8/FastUMI/dataset/UMI_2026-11-06_T0_N1_ep235_gripOC_offsetfix_joint \
    --episodes 2,3,5 --with-video --space all --gripper-binary 0.9
```

Output is LeRobot v3.0, 20 fps, `action` = 6 UR7e joints (rad) + gripper (1 = open).

## Other changes in this folder

- `config/config.json`: `base_position` / `base_orientation` refitted so the T265 start maps onto
  the pendant home pose `[63.31, −110.96, 115.32, −97.77, −92.77, −206.36]°` (single-point fit).
- `conversion/data_processing_to_joint.py`: analytic IK from the bundled `kinematics/` (from
  `ur7e-dev6.1`, identical to embodied_ai_ml v1.4.8; external checkout only as fallback);
  ArUco `DetectorParameters` works on both old and new OpenCV.
- Live T265 home check in the web UI; YUYV camera format (true 60 fps → 20 fps synced).

## Open items

- **Confirm fix 2 physically:** measure where the T265 sits relative to the UMI fingertips
  (which side, 15.9 cm) and check it lies along the robot flange's y axis.
- **`flange_to_tcp` = 0.244 m** equals the 2F-140 fingertip when fully closed (pad centre is
  191–215 mm, fingertip 223–247 mm from the flange). Check which UMI point the `offset` refers to.
- **Gripper on the stack** did not move during the first replays; that is on the robot side
  (`/action_004/right_UR7/gripper_states`), not the dataset.
- **Tag visibility:** both tags are seen in only 54–67% of frames during collection (96% when
  calibrating); move the tags so a finger or the bottle doesn't hide one.
- **fps:** UMI datasets are 20 fps; the robot's own recordings are 30 fps.
- Port these fixes to the main folder / commit — not done yet.
