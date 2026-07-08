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
    echo "  (close the terminal to exit)"
    echo ""
    read -p "  Press Enter to start... "
    source /opt/ros/noetic/setup.bash
    roslaunch realsense2_camera rs_t265.launch
    echo ""
    echo "  T265 stopped. Press Enter to restart..."
    read
done
