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
    echo "  (close the terminal to exit)"
    echo ""
    read -p "  Press Enter to start... "
    source /opt/ros/noetic/setup.bash
    roscore
    echo ""
    echo "  roscore stopped. Press Enter to restart..."
    read
done
