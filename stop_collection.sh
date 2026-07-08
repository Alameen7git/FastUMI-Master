#!/bin/bash
# FastUMI — stop all data collection nodes

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  FastUMI — Stopping all nodes                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

stop_proc() {
    local label="$1"
    local pattern="$2"
    if pkill -SIGTERM -f "$pattern" 2>/dev/null; then
        echo "  [stopped]  $label"
    else
        echo "  [not running]  $label"
    fi
}

# 1. Stop data collection first (graceful — lets it flush buffers)
stop_proc "data_collection.py" "data_collection.py"
sleep 1.5

# 2. Stop roslaunch sessions (T265 + USB camera nodes)
stop_proc "roslaunch (T265 / camera)" "roslaunch"
sleep 1

# 3. Stop roscore / rosmaster
stop_proc "roscore" "roscore"
pkill -SIGTERM -f "rosmaster" 2>/dev/null
pkill -SIGTERM -f "rosout"    2>/dev/null
sleep 0.5

# 4. Force-kill anything still alive
pkill -SIGKILL -f "data_collection.py" 2>/dev/null
pkill -SIGKILL -f "roslaunch"          2>/dev/null
pkill -SIGKILL -f "rosmaster"          2>/dev/null
pkill -SIGKILL -f "rosout"             2>/dev/null

echo ""
echo "  Done. All FastUMI nodes stopped."
echo ""
