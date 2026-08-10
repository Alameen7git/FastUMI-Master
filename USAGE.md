# FastUMI → UR7e — File Usage Guide

Every script grouped by purpose. Click a heading to expand it. Each entry says
**what it does**, **which Python env**, and an **example command**.

**Folder layout**
```
FastUMI-Master/
├── (root)            data collection + infra (wired into the dashboard/systemd)
├── conversion/       raw FastUMI → joints / TCP / other formats
├── replay_pipeline/  D1 raw → LeRobot dataset for robot replay
├── replay/           execute / export a trajectory to the robot
├── viz/              visualization, validation, simulation
├── lib/              imported-only helper libraries
├── config/           config.json (all calibration)
├── assets/           MuJoCo models, URDFs, fonts
├── dashboard/        collection web dashboard
└── systemd/          service units
```

**Two Python environments** (can't be mixed in one process):
- **FastUMI** — `/home/nuc8/miniconda3/envs/FastUMI/bin/python3` — MuJoCo, analytic IK (`embodied_ai_ml`), OpenCV (most processing, replay-prep, viz).
- **base** — `/home/nuc8/miniconda3/bin/python3` — LeRobot (dataset writing / Hub upload).

> **Always run from the repo root** (`/home/nuc8/Downloads/FastUMI-Master`). Scripts read `config/` + `assets/` relative to the current directory, and cross-folder imports assume root is the working dir. Example: `python3 viz/simulate_episode_gif.py ...`.

---

<details>
<summary>📹 <b>Data collection</b> (root) — record handheld demos</summary>

Kept at repo root because the dashboard + systemd launch/track them by name.

| File | What it does |
|---|---|
| `data_collection.py` | Records demos (GoPro + T265) into per-episode HDF5. |
| `data_collection.sh` | Shell entry point (launches ROS nodes + `data_collection.py`). |
| `start_collection.sh` / `stop_collection.sh` | Start / gracefully stop a run. |
| `calibrate_markers.py` | Check/calibrate the ArUco gripper markers. |

```bash
./start_collection.sh
python3 data_collection.py --task PnP_block11 --num_episodes 25
./stop_collection.sh
```
</details>

<details>
<summary>🔄 <b>conversion/</b> — raw FastUMI → joints / TCP / other</summary>

| File | What it does | Env |
|---|---|---|
| `conversion/data_processing_to_joint.py` | **Core library.** Cartesian T265 pose → UR7e joints + gripper (analytic IK). Imported by most tools; not run directly. | FastUMI |
| `conversion/data_processing_to_tcp.py` | Raw pose → TCP-in-base-frame. | FastUMI |
| `conversion/data_processing_tcp_to_dp.py` | TCP → diffusion-policy (zarr) format (needs `zarr`). | FastUMI |
| `conversion/convert_episodes.py` | Batch driver over a task (`--task`, `--parallel`, `--keep-raw`). | FastUMI |
| `conversion/convert_one_episode_v2.py` | Single episode → joints+gripper HDF5 (`--episode`, `--output`). | FastUMI |
| `conversion/convert_fastumi_to_lerobot.py` | Generic FastUMI → LeRobot (state = raw pose). | base |

```bash
/home/nuc8/miniconda3/envs/FastUMI/bin/python3 conversion/convert_episodes.py --task PnP_block11
```
</details>

<details>
<summary>🤖 <b>replay_pipeline/</b> — D1 raw → LeRobot dataset for robot replay <i>(this session)</i></summary>

| File | What it does | Env |
|---|---|---|
| `replay_pipeline/make_replay_dataset.sh` | **One-command wrapper** around both stages (switches envs for you). | (both) |
| `replay_pipeline/d1_to_lerobot_stage1.py` | Pose → joints + gripper + winding-fix + Cartesian → per-episode `.npz`. | FastUMI |
| `replay_pipeline/d1_to_lerobot_stage2.py` | `.npz` → LeRobot v3.0 dataset (joint or Cartesian schema). | base |
| `replay_pipeline/push_dataset_to_hub.py` | Upload a finished dataset to the HF Hub (`--repo-id`, `--root`). | base |

```bash
# joint-space replay dataset
./replay_pipeline/make_replay_dataset.sh --task PnP_block11

# Cartesian, both frames (ur_rtde + ROS)
./replay_pipeline/make_replay_dataset.sh --task PnP_block11 --space cart-both

# one episode, lock elbow-up config, include camera
./replay_pipeline/make_replay_dataset.sh --task PnP_block11 --episodes 7 --lock-branch 5 --with-video
```
Flags: `--space {joint,cart-urbase,cart-baselink,cart-both,all}`, `--episodes 1,5,9`, `--lock-branch N`, `--fps`, `--with-video`, `--out`, `--help`.
</details>

<details>
<summary>🦾 <b>replay/</b> — execute / export a trajectory to the robot</summary>

| File | What it does | Env |
|---|---|---|
| `replay/replay.py` | One-command `ur_rtde` replay (task+episode). Dry-run by default; `--execute` + `--robot-ip` + typed confirm to move hardware. | FastUMI |
| `replay/replay_ur7e.py` | Lower-level replay of a validated trajectory, `--speed-scale` retiming + dry-run. | FastUMI |
| `replay/export_urscript_replay.py` | Export one validated episode as a `.script` (URScript) for the pendant. | FastUMI |
| `replay/export_episode_cartesian.py` | Export one episode's Cartesian TCP waypoints to CSV + npz (both frames). | FastUMI |

```bash
PY=/home/nuc8/miniconda3/envs/FastUMI/bin/python3
$PY replay/replay.py --task PnP_block11 --episode 7 --speed 0.1          # safe dry-run
$PY replay/export_episode_cartesian.py --task PnP_block11 --episode 7 --frame both
```
</details>

<details>
<summary>👁️ <b>viz/</b> — visualization, validation, simulation</summary>

| File | What it does | Output |
|---|---|---|
| `viz/validate_joint_trajectory.py` | Check limits/velocity/reachability/winding; flags violations. | `.hdf5` + PNG |
| `viz/simulate_episode_gif.py` | MuJoCo arm replaying computed joints next to the video. `--tcp-axes` overlays the Cartesian pose. | GIF + MP4 |
| `viz/visualize_cartesian_gif.py` | Pure **Cartesian** animation: TCP pose (trail + orientation triad), no mesh. | GIF + MP4 |
| `viz/visualize_trajectory.py` | Static 3D path + position/orientation/gripper plots. | PNG |
| `viz/visualize_episode_mujoco.py` | MuJoCo episode reconstruction next to camera feed. | MP4 |
| `viz/mujoco_replay_episode.py` | Replay on the UR5e menagerie model. | MP4 |
| `viz/render_side_by_side.py` | Real-vs-sim comparison video (UR5e). | MP4 |
| `viz/render_episode_report.py` / `render_dataset_report.py` | Self-contained HTML report (one episode / whole task). | HTML |
| `viz/visualize_dataset.py` | Local web dashboard to browse a dataset (`--port`). | web |
| `viz/publish_test_output_trajectory.py` | Publish a joint trajectory as ROS JointState for RViz. | — |

```bash
PY=/home/nuc8/miniconda3/envs/FastUMI/bin/python3
$PY viz/validate_joint_trajectory.py --task PnP_block11 --episode-idx 7   # check before hardware
$PY viz/simulate_episode_gif.py       --task PnP_block11 --episode 7      # arm vs video
$PY viz/simulate_episode_gif.py       --task PnP_block11 --episode 7 --tcp-axes  # + Cartesian pose
$PY viz/visualize_cartesian_gif.py    --task PnP_block11 --episode 7      # pure Cartesian
$PY viz/visualize_trajectory.py       --task PnP_block11 --episode 7      # static plots
```
</details>

<details>
<summary>🧩 <b>lib/</b> — imported-only helper libraries</summary>

| File | What it does |
|---|---|
| `lib/replay_buffer.py` | UMI/diffusion-policy replay-buffer (zarr) library. |
| `lib/imagecodecs_numcodecs.py` | Vendored numcodecs image codecs for the zarr store. |
| `lib/odom_to_path.py` | ROS odometry → nav_msgs/Path helper. |
</details>

<details>
<summary>⚙️ <b>Config, assets & infrastructure</b></summary>

| Path | What it is |
|---|---|
| `config/config.json` | All calibration: base pose, `flange_to_tcp`, ArUco/gripper distances, rest joints, hardware limits. Every tool reads this. |
| `assets/mujoco/` | UR7e MuJoCo model (`ur7e_scene.xml` + meshes) for the analytic-IK sims. |
| `assets/mujoco_menagerie/` | UR5e model used by `viz/mujoco_replay_episode.py` / `render_side_by_side.py`. |
| `assets/*.urdf` | UR7e URDFs (joint limits parsed by `validate_joint_trajectory.py`). |
| `assets/fonts/` | Fonts for the HTML reports. |
| `dashboard/` | Web dashboard app (`app.py` + static frontend). |
| `systemd/` | Service units. |
| `*.launch` (root) | ROS launch files. |
</details>

<details>
<summary>📄 <b>Docs</b> (root)</summary>

| File | What it is |
|---|---|
| `README.md` | Upstream FastUMI readme. |
| `REPLAY_PIPELINE_REPORT.md` | Plain-English writeup of the D1→replay pipeline. |
| `USAGE.md` | This file. |
</details>

---

## Typical end-to-end flow
```bash
PY=/home/nuc8/miniconda3/envs/FastUMI/bin/python3

# 1. sanity-check an episode's conversion
$PY viz/validate_joint_trajectory.py --task PnP_block11 --episode-idx 7
$PY viz/simulate_episode_gif.py       --task PnP_block11 --episode 7

# 2. build the replay dataset (joint space)
./replay_pipeline/make_replay_dataset.sh --task PnP_block11

# 3. copy <task>_lerobot/ to the robot machine and run your existing replay there
```
</content>
