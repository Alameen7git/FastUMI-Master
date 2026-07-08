#!/bin/bash
# FastUMI — stop all data collection nodes and close their terminal windows

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_FILE="$DIR/.fastumi_session"

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

# Poll until no process matches $1, up to $2 seconds. Returns 1 on timeout.
wait_for_exit() {
    local pattern="$1"
    local timeout="$2"
    local waited=0
    while pgrep -f "$pattern" >/dev/null 2>&1; do
        sleep 0.2
        waited=$((waited + 1))
        [ "$waited" -ge $((timeout * 5)) ] && return 1
    done
    return 0
}

# 1. Stop data collection first. SIGTERM is caught by data_collection.py and
#    treated like pressing "E" -- it finalizes (writes the manifest for) any
#    episode currently being recorded before the process exits, so in-progress
#    data is never lost. Wait for it to actually exit before moving on.
stop_proc "data_collection.py" "data_collection.py"
wait_for_exit "data_collection.py" 15 \
    || echo "  ⚠ data_collection.py did not exit in time -- forcing"

# 2. Stop the camera capture node and T265 launch
stop_proc "cam_capture_node.py" "cam_capture_node.py"
stop_proc "roslaunch (T265)"    "roslaunch"
sleep 1

# 3. Stop roscore / rosmaster
stop_proc "roscore" "roscore"
pkill -SIGTERM -f "rosmaster" 2>/dev/null
pkill -SIGTERM -f "rosout"    2>/dev/null
sleep 0.5

# 4. Force-kill anything still alive
pkill -SIGKILL -f "data_collection.py"  2>/dev/null
pkill -SIGKILL -f "cam_capture_node.py" 2>/dev/null
pkill -SIGKILL -f "roslaunch"           2>/dev/null
pkill -SIGKILL -f "rosmaster"           2>/dev/null
pkill -SIGKILL -f "rosout"              2>/dev/null

# 5. Close the 4 terminal windows opened by start_collection.sh.
#    Each one is running "bash $TMP/N_*.sh" (a while-loop wrapper around the
#    real command); killing that wrapper -- not just the process inside it --
#    makes the shell exit, and the terminal profile's exit-action ("close")
#    closes the window instead of leaving a dead "press enter" shell open.
#    $TMP is a unique mktemp dir created by this session, so this can never
#    match the terminal this script itself is running in.
if [ -f "$SESSION_FILE" ]; then
    TMP="$(cat "$SESSION_FILE")"
    if [ -n "$TMP" ] && [ -d "$TMP" ]; then
        pkill -SIGTERM -f "$TMP/" 2>/dev/null
        sleep 0.5
        pkill -SIGKILL -f "$TMP/" 2>/dev/null
        rm -rf "$TMP"
        echo "  [closed]   4 terminal windows"
    fi
    rm -f "$SESSION_FILE"
else
    echo "  [skip]     no active session found -- terminal windows left as-is"
fi

echo ""
echo "  Done. All FastUMI nodes stopped."
echo ""
