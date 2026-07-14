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

# 0. data_collection.py only registers a handler for SIGINT (not SIGTERM) —
#    that's what actually closes the video writer / CSVs and ends the
#    current episode cleanly. Give it that chance first, so the SIGTERM
#    below (which it can't act on gracefully) isn't the first thing it sees.
if pkill -SIGINT -f "data_collection.py" 2>/dev/null; then
    echo "  [signaled]  data_collection.py (SIGINT — graceful episode save)"
    for i in $(seq 1 20); do
        pgrep -f "data_collection.py" >/dev/null 2>&1 || break
        sleep 0.25
    done
fi

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

# 5. Close the 4 terminal windows start_collection.sh opened.
#    Each one is running a wrapper script (a "press Enter to restart" loop)
#    recorded by start_collection.sh at launch, in $SESSION_FILE. Killing
#    that wrapper's shell process makes gnome-terminal close the window
#    (this profile's "When command exits" is set to close automatically).
#    This never touches the terminal this script is run from — that
#    terminal is never running one of the tracked wrapper scripts.
SESSION_FILE="/tmp/.fastumi_collection_terminals"
if [ -f "$SESSION_FILE" ]; then
    echo "  Closing collection terminals..."
    while IFS= read -r WRAPPER; do
        [ -z "$WRAPPER" ] && continue
        # Extra guard specifically for terminal [4]: don't close it until
        # data_collection.py is actually gone, so window-closing can never
        # race ahead of the episode finalizing (steps 0/1/4 above should
        # already have it stopped by this point — this just double-checks).
        if [[ "$WRAPPER" == *4_collect.sh ]]; then
            for i in $(seq 1 20); do
                pgrep -f "data_collection.py" >/dev/null 2>&1 || break
                sleep 0.25
            done
        fi
        pkill -SIGTERM -f "$WRAPPER" 2>/dev/null
    done < "$SESSION_FILE"
    sleep 0.3
    # Fallback for any wrapper still alive (e.g. stuck at a "Press Enter"
    # prompt in a way SIGTERM didn't clear).
    while IFS= read -r WRAPPER; do
        [ -z "$WRAPPER" ] && continue
        pkill -SIGKILL -f "$WRAPPER" 2>/dev/null
    done < "$SESSION_FILE"
    rm -f "$SESSION_FILE"
    echo "  Closed 4 terminal windows."
else
    echo "  No session file found — terminal windows (if any) were not auto-closed."
    echo "  (They'll be tracked next time you run ./start_collection.sh.)"
fi
echo ""
