#!/bin/bash
trap '' INT
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
while true; do
    clear
    echo "╔══════════════════════════════════════════════╗"
    echo "║  [3/4]  USB Camera                           ║"
    echo "╚══════════════════════════════════════════════╝"
    echo ""
    echo "  Command:  roslaunch $DIR/usb_cam-test.launch"
    echo "  Start AFTER roscore is running."
    echo "  (close the terminal to exit)"
    echo ""
    read -p "  Press Enter to start... "
    source /opt/ros/noetic/setup.bash
    roslaunch "$DIR/usb_cam-test.launch"
    echo ""
    echo "  Camera stopped. Press Enter to restart..."
    read
done
