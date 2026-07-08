#!/bin/bash
# FastUMI — process everything currently sitting in the episode queue, then exit.
# Run this manually whenever you want (typically after you're done collecting
# for the session) -- it does NOT run alongside live capture.
#
# Usage:
#   ./run_worker.sh              # process pending episodes across all tasks
#   ./run_worker.sh <task_name>  # process pending episodes for one task only

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SETUP="/home/nuc8/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="FastUMI"

source "$CONDA_SETUP"
conda activate "$CONDA_ENV"

cd "$DIR"
python3 episode_worker.py --once "$@"
