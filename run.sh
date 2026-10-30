#!/bin/bash
# Run one notebook headless on the server: ./run.sh queue_wave1.ipynb
# The executed copy goes to runs/<name>.ipynb and the log to runs/<name>.log; models and results are pushed to HF by the notebooks.
set -e
cd "$(dirname "$0")"
source ~/env/bin/activate
mkdir -p runs
name=$(basename "$1" .ipynb)
papermill "$1" "runs/$name.ipynb" --log-output 2>&1 | tee "runs/$name.log"
