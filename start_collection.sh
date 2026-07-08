#!/bin/bash
# FastUMI — data collection launcher
# Opens 4 separate terminal windows. Press Enter in each to start its process.
# Episode processing (Stage 2) is NOT started here -- run ./run_worker.sh
# manually once you're done collecting.
#
# Usage:
#   ./start_collection.sh

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_SETUP="/opt/ros/noetic/setup.bash"
CONDA_SETUP="/home/nuc8/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="FastUMI"

TMP="$(mktemp -d)"
echo "$TMP" > "$DIR/.fastumi_session"

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
source $CONDA_SETUP
conda activate $CONDA_ENV
source $ROS_SETUP

while true; do
    clear
    echo "╔══════════════════════════════════════════════╗"
    echo "║  [3/4]  Camera (compressed capture)          ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "  Command:  python3 cam_capture_node.py"
    echo "  Publishes raw MJPEG on /usb_cam/image_raw/compressed -- no decode."
    echo "  Start AFTER roscore is running."
    echo "  (close the window to exit)"
    echo ""
    read -p "  Press Enter to start... "
    cd "$DIR"
    python3 cam_capture_node.py
    echo ""
    echo "  Camera stopped. Press Enter to restart..."
    read
done
EOF

# ── [4] data_collection ───────────────────────────────────────────────────────
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
gnome-terminal --title="FastUMI | 3 camera"     -- bash "$TMP/3_camera.sh" &
sleep 0.3
gnome-terminal --title="FastUMI | 4 collect"    -- bash "$TMP/4_collect.sh" &

echo "4 terminals opened."
echo "Start order: [1] roscore → [2] T265 → [3] camera → [4] collect"
echo "When you're done collecting, run ./run_worker.sh to process the queue."
