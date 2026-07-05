#!/usr/bin/env bash
# Run the lightweight_train.py smoke test in the background with a log file,
# so a single terminal session (no split panes) is enough to launch it,
# walk away, and check on it later with `tail -f` instead of babysitting.
#
# Usage (from anywhere, run on the GPU node in the yuval_rl conda env):
#   bash mini_entropy_srt/run_smoke_test.sh
#
# Then:
#   tail -f <printed log path>      # watch live output
#   ps -p <printed PID>             # check if it's still running
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
git pull

mkdir -p logs
LOG_FILE="logs/smoke_test_$(date +%Y%m%d_%H%M%S).log"

# expandable_segments reduces the odds of an OOM caused by allocator
# fragmentation rather than genuinely insufficient memory -- the kind of
# failure where nvidia-smi shows free memory but a single allocation still
# fails to find a contiguous block.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -u -m mini_entropy_srt.lightweight_train \
    --n_steps 3 \
    --n_rollouts 2 \
    --base_max_tokens 256 \
    --escalated_max_tokens 512 \
    --eval_every 1 \
    --n_eval_questions 5 \
    --debug_steps 3 \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started smoke test in the background."
echo "  PID: $PID"
echo "  log: $LOG_FILE"
echo ""
echo "Watch it live:   tail -f $LOG_FILE"
echo "Check it's alive: ps -p $PID"
echo "GPU usage:        nvidia-smi"
