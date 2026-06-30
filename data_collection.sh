#!/bin/bash

# Default values
TASK="test"
NUM_EPISODES=2

# Help function
function show_help {
    echo "Usage: $0 [OPTIONS]"
    echo "Record robot trajectory data"
    echo ""
    echo "Options:"
    echo "  -t, --task          Task name (default: test)"
    echo "  -n, --num-episodes  Number of episodes to record (default: 2)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Note: config is always read from config/config.json"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -n|--num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute the Python script
python3 data_collection.py \
    --task "$TASK" \
    --num_episodes "$NUM_EPISODES"