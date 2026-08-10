#!/usr/bin/env bash
#
# make_replay_dataset.sh
# ----------------------
# ONE command that turns a raw FastUMI dataset (handheld T265 pose + video, the
# "D1" format) into a replay-ready LeRobot v3.0 dataset (joint angles + gripper,
# the "D2" format your robot's replay setup reads).
#
# It just orchestrates the two-stage converter, because the inverse-kinematics
# solver and the LeRobot writer live in two different conda envs that can't
# share one Python process:
#     stage 1  (FastUMI env)  raw pose  -> UR7e joints + gripper + winding fix
#     stage 2  (base env)     joints    -> LeRobot v3.0 dataset folder
#
# USAGE
#   ./make_replay_dataset.sh --task Pick_and_place_the_bottle
#   ./make_replay_dataset.sh --src /path/to/dataset --out /path/to/output_lerobot
#   ./make_replay_dataset.sh --task my_task --fps 30 --with-video
#
# COMMON OPTIONS
#   --task NAME        task subdir under the dataset root in config/config.json
#   --src DIR          explicit dataset dir (use instead of --task)
#   --out DIR          output dataset folder (default: <src>_lerobot)
#   --fps N            dataset fps written to the folder (default: 20 = D1's rate).
#                      Set to whatever your replay assumes (e.g. 30) if it doesn't
#                      read fps from the dataset.
#   --task-name "STR"  natural-language task string stored per frame
#   --repo-id ID       LeRobot repo id label (default: embodied-ai/<task>)
#   --episodes 1,2,3   only these episode indices (default: all)
#   --with-video       include the front camera as observation.images.cam_high
#                      (default: OFF -- joints only, smallest + most portable)
#   --lock-branch N    force one kinematic branch 0..7 every frame (e.g. 5 = elbow-up)
#                      so the arm can't silently flip configuration. Frames the
#                      demo pushes out of that branch's reach are flagged + held.
#   --space MODE       output representation (default: joint):
#                        joint          6 joints + gripper (D2 schema, joint replay)
#                        cart-urbase    TCP [x,y,z,Rx,Ry,Rz,grip], UR base frame (ur_rtde)
#                        cart-baselink  TCP [x,y,z,qx,qy,qz,qw,grip], ROS base_link
#                        cart-both      both Cartesian folders (<out>_urbase / _baselink)
#                        all            joint + both Cartesian folders
#   -h | --help        show this help
#
# The two Python interpreters are auto-detected but can be overridden with the
# FASTUMI_PY and LEROBOT_PY environment variables.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTUMI_PY="${FASTUMI_PY:-/home/nuc8/miniconda3/envs/FastUMI/bin/python3}"
LEROBOT_PY="${LEROBOT_PY:-/home/nuc8/miniconda3/bin/python3}"
CONFIG="${CONFIG:-$HERE/../config/config.json}"

TASK="" ; SRC="" ; OUT="" ; FPS="20" ; TASK_NAME="" ; REPO_ID="" ; EPISODES="" ; WITH_VIDEO=0 ; LOCK_BRANCH="" ; SPACE="joint"
INTERMEDIATE="$HERE/../outputs/d1_lerobot_intermediate"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)       TASK="$2"; shift 2;;
    --src)        SRC="$2"; shift 2;;
    --out)        OUT="$2"; shift 2;;
    --fps)        FPS="$2"; shift 2;;
    --task-name)  TASK_NAME="$2"; shift 2;;
    --repo-id)    REPO_ID="$2"; shift 2;;
    --episodes)   EPISODES="$2"; shift 2;;
    --with-video) WITH_VIDEO=1; shift;;
    --lock-branch) LOCK_BRANCH="$2"; shift 2;;
    --space)       SPACE="$2"; shift 2;;
    -h|--help)    tail -n +2 "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "unknown option: $1 (try --help)"; exit 1;;
  esac
done

if [[ -z "$TASK" && -z "$SRC" ]]; then
  echo "ERROR: pass --task NAME or --src DIR  (see --help)"; exit 1
fi

# Derive names/paths from whatever was given.
if [[ -z "$SRC" ]]; then
  DATA_ROOT="$("$LEROBOT_PY" -c "import json;print(json.load(open('$CONFIG'))['device_settings']['data_dir'])")"
  SRC="${DATA_ROOT%/}/$TASK"
fi
[[ -z "$TASK" ]] && TASK="$(basename "$SRC")"
[[ -z "$OUT" ]] && OUT="${SRC%/}_lerobot"
[[ -z "$TASK_NAME" ]] && TASK_NAME="$(echo "$TASK" | tr '_' ' ')"
[[ -z "$REPO_ID" ]] && REPO_ID="embodied-ai/$(echo "$TASK" | tr 'A-Z' 'a-z')"

for PY in "$FASTUMI_PY" "$LEROBOT_PY"; do
  [[ -x "$PY" ]] || { echo "ERROR: python not found: $PY (set FASTUMI_PY / LEROBOT_PY)"; exit 1; }
done

echo "=============================================================="
echo " source dataset : $SRC"
echo " output folder  : $OUT"
echo " repo id / task : $REPO_ID  |  \"$TASK_NAME\""
echo " fps / video    : $FPS  |  $([[ $WITH_VIDEO -eq 1 ]] && echo 'cam_high included' || echo 'joints only (no video)')"
echo " space          : $SPACE"
echo "=============================================================="

# Start from a clean intermediate so a partial (--episodes) run can't pick up
# leftover .npz files from a previous full run.
rm -rf "$INTERMEDIATE"

echo; echo ">>> STAGE 1/2  (FastUMI env)  raw pose -> joints + gripper + winding"
S1=( "$FASTUMI_PY" "$HERE/d1_to_lerobot_stage1.py" --src "$SRC" --config "$CONFIG" --out-dir "$INTERMEDIATE" )
[[ -n "$EPISODES" ]] && S1+=( --episodes "$EPISODES" )
[[ -n "$LOCK_BRANCH" ]] && S1+=( --lock-branch "$LOCK_BRANCH" )
"${S1[@]}"

# Stage 1 always computes joints AND both Cartesian representations, so stage 2
# just selects which to write. --space chooses the output dataset(s):
#   joint (default) | cart-urbase | cart-baselink | cart-both | all
run_stage2 () {  # $1 = space, $2 = out dir
  echo; echo ">>> STAGE 2  (base env)  writing space=$1 -> $2"
  local S2=( "$LEROBOT_PY" "$HERE/d1_to_lerobot_stage2.py"
             --intermediate-dir "$INTERMEDIATE" --out-root "$2"
             --repo-id "$REPO_ID" --task "$TASK_NAME" --fps "$FPS" --space "$1" --overwrite )
  [[ $WITH_VIDEO -eq 0 ]] && S2+=( --no-video )
  "${S2[@]}"
}

OUTS=()
case "$SPACE" in
  joint)          run_stage2 joint         "$OUT";                 OUTS+=("$OUT");;
  cart-urbase)    run_stage2 cart-urbase   "$OUT";                 OUTS+=("$OUT");;
  cart-baselink)  run_stage2 cart-baselink "$OUT";                 OUTS+=("$OUT");;
  cart-both)      run_stage2 cart-urbase   "${OUT}_urbase";        run_stage2 cart-baselink "${OUT}_baselink";
                  OUTS+=("${OUT}_urbase" "${OUT}_baselink");;
  all)            run_stage2 joint         "$OUT";                 run_stage2 cart-urbase "${OUT}_cart_urbase";
                  run_stage2 cart-baselink "${OUT}_cart_baselink";
                  OUTS+=("$OUT" "${OUT}_cart_urbase" "${OUT}_cart_baselink");;
  *) echo "ERROR: --space must be joint | cart-urbase | cart-baselink | cart-both | all"; exit 1;;
esac

echo
echo "=============================================================="
echo " DONE.  Replay-ready dataset(s):"
for o in "${OUTS[@]}"; do echo "   - $o"; done
echo " Copy the folder(s) to the robot's machine and point your"
echo " replay at it (root=<pasted path>). Nothing else to run there."
echo "=============================================================="
