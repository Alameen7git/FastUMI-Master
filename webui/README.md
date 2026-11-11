# FastUMI operator web UI

Local Flask UI for the whole collection workflow, as five numbered steps in the top bar:
**1 Machines** (ROS / T265 / camera) → **2 Record** (`data_collection.py`) → **3 Sync** →
**4 Convert** (LeRobot) → **5 View**, plus **Delete** and **Robot** (replay an episode on the UR7e). Each step has a light in the top bar, one big button and
a one-line instruction saying what to do next. Commands and extra options sit under the
*Details* / *Advanced* folds. The debug log is a drawer behind the yellow **LOG** tab on the
right edge. It starts closed, shows a red count when new errors arrive, and closes with Esc.

```bash
./bin/fastumi-ui start          # blocks; opens http://localhost:5050
./bin/fastumi-ui stop           # from another terminal
./bin/fastumi-ui status
```

`Ctrl+C`, closing the terminal, `kill` and `stop` all run the same shutdown as **Stop All**
(Collection → GoPro → T265 → ROS, plus sync/convert/dashboard). Each process runs in its
own process group, so roslaunch children and the `image_view` preview window go with it.
If the server itself crashes, the next `start`/`stop` kills whatever it left running
(tracked in `webui/.run/pgids.json`).

**Desktop icon**: after cloning, run once:

    ./bin/fastumi-ui install

It puts a **FastUMI Control** icon on the desktop and in the app menu, pointing at wherever this
checkout lives (`bin/fastumi-ui.desktop` is the template; `@REPO@` is filled in). The icon opens a
terminal running `fastumi-ui start`, which opens the browser; closing that terminal window stops
everything. If the icon shows a warning, right-click it → *Allow Launching*. Moved the repo? Run
`install` again. `./bin/fastumi-ui uninstall` removes the icons. The UI needs no internet (no web fonts or CDN scripts); only
**Push** to the Hugging Face Hub does.

The server runs in the `FastUMI` conda env (Flask is installed there). Override with
`FASTUMI_UI_PY=/path/to/python3`, and the port with `FASTUMI_UI_PORT`. Conda is looked for in
`~/miniconda3`, `~/anaconda3`, `~/miniforge3` and `/opt/conda`; set `FASTUMI_CONDA_BASE` if it is
elsewhere. The machine also needs ROS Noetic, the `FastUMI` env (Flask, h5py, OpenCV) and LeRobot
in the conda `base` env (Convert / Push).

## Operator features

- **Swapping the capture card**: the camera does not depend on `/dev/elgato` (that udev name
  only matched one card model). When the camera starts, the UI uses whichever Elgato card is plugged
  in, any model or serial, and shows it under the Camera light. While the camera is on, unplugging
  the card stops it, and plugging a card in (the same one or another) starts it again by itself;
  pressing Start with no card plugged in waits for one. This is skipped while recording, and after
  you press Stop. Force a device with `FASTUMI_CAPTURE_DEVICE=/dev/videoN`; other vendors go in
  `CAPTURE_VENDORS` in `settings.py`.
- **60 fps needs the card on USB 3**: on USB 3 the camera asks the card for uncompressed YUYV
  (no JPEG decoding, measured 59.95 fps while recording); on USB 2 it falls back to MJPEG, which
  this card's large JPEG frames limit to ~50 fps. The log line when the camera starts says which
  (`… USB 3 at /dev/video0, sending YUYV`). Force one with `FASTUMI_PIXEL_FORMAT=yuyv|mjpeg`.
  Check a recording with `python3 viz/check_fps.py [dataset]`.
- **GoPro icons in the picture**: the battery / SD-card icons come from the GoPro's HDMI output
  in *Monitor* mode. Set the GoPro's HDMI output to *Live* (clean feed); nothing in software can
  remove them once they are in the picture.
- **Deleting episodes** (Delete page, with a progress bar like Convert's): type the dataset, then episode numbers as the viewer
  shows them (`3, 7, 10-12`) and an optional reason. Both `raw/episode_N/` and `episode_N.hdf5`
  are removed, and the deletion is logged in `<dataset>/deleted_episodes.json`. If the dataset
  was converted, it is then converted again automatically (same settings), and if it was on the
  Hub it is pushed again to the same repo; files the new copy no longer has are removed from the
  Hub. Convert copies the log into the LeRobot dataset as `meta/fastumi_episodes.json`, with which
  recorded episode each LeRobot `episode_index` is, and Push puts a *Deleted episodes* table on
  the Hub dataset card. The recorder never reuses a deleted number. Refused while Record, Sync,
  Convert or Push is running on that dataset; a second delete restarts the running update.
- **Dataset name after recording**: once a session saves at least one episode, its dataset name is
  typed in on Sync, Convert, Push, View and Delete (remembered in `webui/run/last_recorded.json`, so
  also after a UI restart). A name the operator typed or cleared is left alone.
- **Viewer videos are made ahead of time**: when Sync finishes, `visualize_dataset.py --prepare
  <dataset>` encodes that dataset's viewer videos in the background (niced, paused while
  recording and resumed after), so View opens those episodes at once. Episodes not prepared yet
  are encoded on first open: frames are read by a pool of worker processes and piped into
  ffmpeg at 960x540 (a 208-frame episode: ~5 s, was ~16 s). While one episode is watched, the next
  one is encoded. Set `VIZ_PREPARE_AFTER_SYNC = False` in `settings.py` to turn the background step off.
- **Viewer hot keys** (View page): Space play / pause, → +10 s, ← −5 s, ↓ next episode, ↑ previous
  episode (keeps playing if it was). They work on the View page without clicking into the viewer
  (passed to it with postMessage), but not while typing in a box. A short label shows what happened.
- **T265 graph** (View page): two panels on a time axis in seconds: position X/Y/Z in metres (T265
  odometry frame, from where the camera started) and orientation as a quaternion qx/qy/qz/qw (no unit).
- **Hugging Face datasets** (View page): type `user/name` and press ⤓ Download & view.
  `viz/fetch_lerobot.py` (conda base env) downloads the LeRobot v3 dataset and cuts each episode's
  video and values into `<data_dir>/.viz_cache/hf/<user>__<name>/`; the viewer lists it by its repo
  id, next to the local datasets. Joint-angle datasets are drawn as joints (rad) + gripper. A private
  dataset needs this NUC to be logged in to Hugging Face. ⤓ Download again picks up a newer version.
- **Ending a session early**: ■ Finish (or E) asks first. During a take it offers *Save episode N
  & end*, *Don't save it & end* (sends X to `data_collection.py`, which throws the take away) or
  *Keep recording*; between takes it only confirms. Episodes already saved are always kept.

- **Dataset names**: every recording goes to `UMI_<date>_T<task number>_<operator><number>`, e.g.
  `UMI_2026-09-23_T1_N1` (`DATASET_NAME_FORMAT` in settings.py). The number counts that
  operator's datasets for that task on that day. When one exists, the Record page offers
  *Add to n1* (the latest, chosen by default, also after *Record more episodes*) or *Start a
  new dataset: n2*. Operator names can't end with a digit, so the number is never ambiguous.
- **Tasks** (Record → Tasks): add the tasks you plan to record before a session. The operator
  types each task's number after a fixed *T* (whole or decimal: 7, 4.1; each number used once;
  the last used and highest numbers are shown under the box), a name, an optional description (stored in the LeRobot
  dataset as the task text) and a default episode count. On the Record page the operator picks
  a task from the dropdown, or chooses *Type a new task name* (a typed task needs its number
  typed under the name too, and is added to the list when recording starts). Entries can be edited or removed at any time; a task's name is locked
  only if an old folder is named exactly like it (new folders use the task number). Removing a task only removes the
  list entry; recordings stay. Saved in `<data_dir>/.fastumi_tasks.json`, with the dataset
  folders recorded for each task. Folders recorded before the naming rule (named exactly like
  the task) are still counted.
- **Next-step buttons**: when recording ends there is a *Next: Synchronize* button; when sync ends
  there is a *Next: Convert* button; each carries the task name along.
- **Machine lost during recording**: if ROS, T265 or the camera stops (the process dies, or
  `data_collection.py`'s own feed watchdog fires), the take is stopped and a panel names the
  machine with a *Restart* button. *Continue recording* (key C) re-records that episode once all
  machines are green; *Save & finish* (key E) keeps the episodes saved so far and ends the session.
- **Live camera**: a dropdown on the Record page shows a reduced preview of the camera topic
  (`webui/camera_preview.py`, port 8081). About half a CPU core while open; closed by default.
- **Convert + push**: `webui/convert_to_lerobot.py` (base env) turns the synced HDF5 files into
  a LeRobot v3 folder `<dataset>/lerobot` (next to `<dataset>/raw`, the original recordings;
  older conversions in `<dataset>_lerobot` are still recognised): state and action are the T265 pose
  `[x, y, z, qx, qy, qz, qw]`, the camera is H.264 video at 640×360 RGB. One toggle uploads the
  result to the Hugging Face Hub (private unless *public* is ticked). The Hub name is not typed:
  it is `<logged-in user>/<dataset name>`, or the repo the dataset was pushed to before
  (remembered in `<dataset>/lerobot_export.json` with the convert settings). The robot-replay outputs of
  `replay_pipeline/make_replay_dataset.sh` (joint angles etc.) appear under Advanced only
  on checkouts that have it (branch `ur7e-dev7.0`).
- **Robot page (replay on the UR7e)**: plays one episode of a LeRobot *joint-angle* dataset (local
  `<dataset>/lerobot_joint`, or a Hugging Face repo id) on the arm with `replay/replay_lerobot.py`, in
  three steps: **1 Dry run** (no robot: joint limits, duration at the chosen speed, a plot), **2 Check
  robot** (read-only connection: robot mode, safety mode, current joints next to the episode start, to
  compare with the pendant; never moves the arm) and **3 Run on robot**. The server refuses the run
  until a dry run and a check of the same episode, speed and IP passed, and the operator ticked that the
  joints match the pendant and typed RUN (and CONFIRM_FAST above 25%). ■ Stop sends Ctrl+C, which
  stops the arm (also during the slow move to the start); the e-stop is still the real stop. Several
  robots with the same IP: connect the NUC only to the one you want (the page explains). The gripper is
  not replayed. The page also has the how-to for making a replayable dataset (below).
- **Converting for the robot** (Convert → Advanced): *Output type* "Robot replay: joint angles" writes
  `<dataset>/lerobot_joint` (pushes go to `<user>/<dataset>_joint`). *Calibration file* picks the
  `config/*.json` it is converted with: the robot base position / orientation and home pose for that
  task. A task recorded with the base somewhere else gets its own copy,
  `config/calib_<dataset>.json`, which is picked by itself when it exists. Episodes with unreachable
  poses (`VIOLATIONS` in stage 1) are listed as problems.
- **Top-bar lights**: green = on (machines, recorder, viewer), amber = working (sync, convert,
  push), red = problem. A finished sync / convert / push goes back to off; its page shows the
  result.
- **Push**: a separate page uploads any LeRobot dataset folder.
- **Push raw**: on the Push page, *Raw recordings* uploads `<dataset>/raw/` (+ `csv/`, `camera/`,
  `states.csv`, `deleted_episodes.json`) with `webui/push_raw.py` to a private repo
  `<user>/<dataset>_raw`, next to the LeRobot repo `<user>/<dataset>`. The synced `episode_N.hdf5` are
  left out (about 10x bigger; Sync rebuilds them from `raw/`). Unchanged files are skipped, nothing on
  the Hub is deleted, and every file is checked on the Hub afterwards. The last raw upload is kept in
  `lerobot_export.json` (`raw_hub`).
- **Sync from Hugging Face**: on the Sync page, *Raw data from Hugging Face* takes a
  `<user>/<dataset>_raw` repo (your repos are listed), downloads it into `<data_dir>/<dataset>/` with
  `webui/fetch_raw.py` (files already there are skipped, every file is size-checked) and then syncs it.

## What each step runs

All commands live in [`settings.py`](settings.py) and are tagged `[CONFIRMED]` (copied
from a script in this repo) or `[GUESS]`.

| Step | Command | Green when |
|---|---|---|
| 1 ROS | `roscore` | `:11311` accepts connections |
| 2 T265 | `roslaunch realsense2_camera rs_t265.launch` | a message arrives on `/camera/odom/sample` |
| 3 GoPro | `roslaunch usb_cam-test.launch` (usb_cam + `image_view` preview) | `webui/camera_check.py` sees a live picture on `/usb_cam/image_raw` (not blank, not frozen) |
| 4 Collection | `data_collection.py --task UMI_<date>_T<task>_<operator><number> --num_episodes <n>` in a pseudo-terminal | the script prints its first prompt |
| 5 Synchronize | `convert_episodes.py --task <dataset> --parallel N [--keep-raw]` | running / exited 0 |
| 6 Convert | `webui/convert_to_lerobot.py --src <dataset> --out <dataset>/lerobot` (base env, + push) | running / exited 0 |
| Push | `push_dataset_to_hub.py --repo-id <user/name> --root <folder> [--public]` (base env) | running / exited 0 |
| Robot | `replay/replay_lerobot.py <dataset> --episode N --speed S [--check \| --execute --yes] --robot-ip IP` (FastUMI env; reads the dataset with the base env) | running / exited 0 |
| 7 Visualize | `visualize_dataset.py --port 8000 --no-browser`, embedded in an iframe | `/api/dataset` answers |

Notes:

- `data_collection.py` reads SPACE / R / E / C from its terminal (termios), not from a
  window. The UI forwards the on-page buttons (or those keys pressed on the Collect page)
  into its pseudo-terminal. **End session** sends `E`, so a take in progress is saved.
- Step 5 is `convert_episodes.py`: it pairs every 2nd video frame (60 → 30 fps, `--fps 30`; the rate is stored in each file) with the nearest T265 pose
  by timestamp and writes `episode_N.hdf5`. That script deletes `raw/episode_N/` by
  default; the UI passes `--keep-raw` unless you untick it.
- The Elgato capture card keeps sending frames when no GoPro is plugged in (or it is off), so
  the camera light checks the picture itself: green needs a grey-level spread of at least 4 and
  a change of at least 0.3 between frames 0.3 s apart (a live GoPro measured ~61 and ~3). A
  camera that goes blank or frozen turns red within ~12 s, and green again by itself once the
  picture is back.
- Script paths are resolved for both the flat layout and the `conversion/` / `viz/` /
  `replay_pipeline/` layout. Converting again overwrites `<dataset>/lerobot`, so the UI asks first.
- Step 7's dashboard reads the synced HDF5 files, not LeRobot folders. LeRobot outputs are
  listed next to the task they came from.
- While a collection session is running, the T265/GoPro health probes pause, because
  `data_collection.py` has its own feed watchdog.
- The GoPro preview window needs a desktop session (`DISPLAY`). Start the UI from a
  desktop terminal, not over plain SSH.
