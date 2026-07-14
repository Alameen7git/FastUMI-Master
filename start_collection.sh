#!/bin/bash
# FastUMI — data collection launcher
# Opens 4 separate terminal windows. Press Enter in each to start its process.
#
# Usage:
#   ./start_collection.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/noetic/setup.bash"
CONDA_SETUP="/home/nuc8/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="FastUMI"

TMP="$(mktemp -d)"

# ── [1] roscore ───────────────────────────────────────────────────────────────
cat > "$TMP/1_roscore.sh" <<EOF
#!/bin/bash
trap '' INT
while true; do
    clear
    echo "╔══════════════════════════════════════════════╗"
    echo "║  [1/4]  roscore  (ROS master)                ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "  Command:  roscore"
    echo "  Start this FIRST."
    echo "  (close the window to exit)"
    echo ""
    read -p "  Press Enter to start... "
    source $ROS_SETUP
    roscore
    echo ""
    echo "  roscore stopped. Press Enter to restart..."
    read
done
EOF

# ── [2] T265 ──────────────────────────────────────────────────────────────────
cat > "$TMP/2_t265.sh" <<EOF
#!/bin/bash
trap '' INT
while true; do
    clear
    echo "╔══════════════════════════════════════════════╗"
    echo "║  [2/4]  RealSense T265                       ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "  Command:  roslaunch realsense2_camera rs_t265.launch"
    echo "  Start AFTER roscore is running."
    echo "  (close the window to exit)"
    echo ""
    read -p "  Press Enter to start... "
    source $ROS_SETUP
    roslaunch realsense2_camera rs_t265.launch
    echo ""
    echo "  T265 stopped. Press Enter to restart..."
    read
done
EOF

# ── [3] USB camera ────────────────────────────────────────────────────────────
cat > "$TMP/3_camera.sh" <<EOF
#!/bin/bash
trap '' INT
while true; do
    clear
    echo "╔══════════════════════════════════════════════╗"
    echo "║  [3/4]  USB Camera                           ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "  Command:  roslaunch $DIR/usb_cam-test.launch"
    echo "  Start AFTER roscore is running."
    echo "  (close the window to exit)"
    echo ""
    read -p "  Press Enter to start... "
    source $ROS_SETUP
    roslaunch "$DIR/usb_cam-test.launch"
    echo ""
    echo "  Camera stopped. Press Enter to restart..."
    read
done
EOF

# ── [4] data_collection (asks task + episodes interactively, loops on retry) ──
# FIX: the previous version accepted whatever was typed at the "Task name"
# prompt via ${TASK:-test} — but that only falls back to "test" if NOTHING
# is typed. A single accidental Space keypress before Enter produces a
# literal space character, which bash does NOT treat as empty, so it was
# silently accepted as a real (garbage) task name — and if that folder
# already existed, it would silently reuse/mix into an existing dataset.
# Now: loops until a name is entered that is (a) non-empty after trimming
# whitespace, and (b) does not already exist as a dataset folder.
cat > "$TMP/4_collect.sh" <<EOF
#!/bin/bash
trap '' INT
source $CONDA_SETUP
conda activate $CONDA_ENV
source $ROS_SETUP
cd "$DIR"

# Read data_dir from config.json so the duplicate-name check looks in the
# actual dataset location, not a guessed/hardcoded path.
DATA_DIR="\$(python3 -c "import json; print(json.load(open('config/config.json'))['device_settings']['data_dir'])" 2>/dev/null)"

while true; do
    clear
    echo "╔══════════════════════════════════════════════╗"
    echo "║  [4/4]  Data Collection                      ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "  Env:  $CONDA_ENV"
    echo "  Start LAST — after roscore, T265, and camera are all up."
    echo "  (close the window to exit)"
    echo ""

    # Loop until a valid, non-duplicate task name is entered. No default —
    # an empty/whitespace-only entry re-prompts instead of silently
    # falling back to something, so a stray keypress can never accidentally
    # commit to a name.
    while true; do
        read -p "  Task name (required, must be new): " TASK
        # Trim leading/trailing whitespace
        TASK="\$(echo -n "\$TASK" | sed 's/^[[:space:]]*//;s/[[:space:]]*\$//')"

        if [ -z "\$TASK" ]; then
            echo "  ⚠  Task name cannot be empty. Try again."
            continue
        fi

        if [ -n "\$DATA_DIR" ] && [ -d "\$DATA_DIR/dataset/\$TASK" ]; then
            echo "  ⚠  A dataset named '\$TASK' already exists at \$DATA_DIR/dataset/\$TASK"
            echo "     Choose a different name to avoid mixing into existing data."
            continue
        fi

        break
    done

    # Episode count: default-on-Enter is fine here (low risk compared to
    # the task name), but still reject non-positive-integer input rather
    # than silently accepting garbage.
    while true; do
        read -p "  Num episodes [default: 5]: " NUM_EPISODES
        NUM_EPISODES="\${NUM_EPISODES:-5}"
        if ! [[ "\$NUM_EPISODES" =~ ^[1-9][0-9]*\$ ]]; then
            echo "  ⚠  Enter a positive whole number."
            continue
        fi
        break
    done

    echo ""
    echo "  Command:  python3 data_collection.py --task \$TASK --num_episodes \$NUM_EPISODES"
    echo ""
    read -p "  Press Enter to start (or Ctrl+C to cancel and re-enter)... "

    python3 data_collection.py --task "\$TASK" --num_episodes "\$NUM_EPISODES"

    echo ""
    echo "  Stopped (exit code: \$?). Press Enter to run again..."
    read
done
EOF

chmod +x "$TMP"/*.sh

# Record each wrapper script's path so stop_collection.sh can find and close
# exactly these 4 terminal windows later. $TMP is unique per run (mktemp -d),
# so this never matches any other terminal — including whichever one
# stop_collection.sh itself gets run from.
SESSION_FILE="/tmp/.fastumi_collection_terminals"
printf '%s\n' \
    "$TMP/1_roscore.sh" \
    "$TMP/2_t265.sh" \
    "$TMP/3_camera.sh" \
    "$TMP/4_collect.sh" \
    > "$SESSION_FILE"

# ── Open 4 gnome-terminal windows ─────────────────────────────────────────────
gnome-terminal --title="FastUMI | 1 roscore"    -- bash "$TMP/1_roscore.sh" &
sleep 0.3
gnome-terminal --title="FastUMI | 2 T265"       -- bash "$TMP/2_t265.sh" &
sleep 0.3
gnome-terminal --title="FastUMI | 3 USB camera" -- bash "$TMP/3_camera.sh" &
sleep 0.3
gnome-terminal --title="FastUMI | 4 collect"    -- bash "$TMP/4_collect.sh" &

echo "4 terminals opened."
echo "Start order: [1] roscore → [2] T265 → [3] camera → [4] collect"
