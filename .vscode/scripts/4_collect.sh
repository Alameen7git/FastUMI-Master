#!/bin/bash
trap '' INT
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/nuc8/miniconda3/etc/profile.d/conda.sh
conda activate FastUMI
source /opt/ros/noetic/setup.bash

while true; do
    clear
    echo "╔══════════════════════════════════════════════╗"
    echo "║  [4/4]  Data Collection                      ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "  Env:  FastUMI"
    echo "  Start LAST — after roscore, T265, and camera are all up."
    echo "  (close the terminal to exit)"
    echo ""

    read -p "  Task name    [default: test]: " TASK
    TASK="${TASK:-test}"

    read -p "  Num episodes [default: 5]:    " NUM_EPISODES
    NUM_EPISODES="${NUM_EPISODES:-5}"

    echo ""
    echo "  Command:  python3 data_collection.py --task $TASK --num_episodes $NUM_EPISODES"
    echo ""
    read -p "  Press Enter to start... "

    cd "$DIR"
    python3 data_collection.py --task "$TASK" --num_episodes "$NUM_EPISODES"

    echo ""
    echo "  Stopped (exit code: $?). Press Enter to run again..."
    read
done
