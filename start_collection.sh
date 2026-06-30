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
cat > "$TMP/4_collect.sh" <<EOF
#!/bin/bash
trap '' INT
source $CONDA_SETUP
conda activate $CONDA_ENV
source $ROS_SETUP

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

    read -p "  Task name    [default: test]: " TASK
    TASK="\${TASK:-test}"

    read -p "  Num episodes [default: 5]:    " NUM_EPISODES
    NUM_EPISODES="\${NUM_EPISODES:-5}"

    echo ""
    echo "  Command:  python3 data_collection.py --task \$TASK --num_episodes \$NUM_EPISODES"
    echo ""
    read -p "  Press Enter to start... "

    cd "$DIR"
    python3 data_collection.py --task "\$TASK" --num_episodes "\$NUM_EPISODES"

    echo ""
    echo "  Stopped (exit code: \$?). Press Enter to run again..."
    read
done
EOF

chmod +x "$TMP"/*.sh

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
