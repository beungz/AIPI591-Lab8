#!/bin/bash
# Record inference videos from the newest checkpoint of a run (run inside WSL).
# Usage: bash run_inference_wsl.sh <logdir> <task> [episodes] [output_dir]
#   e.g. bash run_inference_wsl.sh ~/logdir/dreamer_walker dmc_walker_walk 3 ~/videos/walker
set -e
LOGDIR="${1:?logdir required}"
TASK="${2:?task required}"
EPISODES="${3:-3}"
OUT="${4:-$HOME/videos}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$HERE/inference_video.py" "$HOME/lab8/inference_video.py"
cd "$HOME/lab8"
source .venv/bin/activate
export MUJOCO_GL=egl

# Newest timestamped checkpoint folder (skips the 'latest' pointer file)
CKPT=$(ls -1d "$LOGDIR"/ckpt/2* | sort | tail -n 1)
echo "Using checkpoint: $CKPT"

python inference_video.py \
  --checkpoint "$CKPT" \
  --task "$TASK" \
  --episodes "$EPISODES" \
  --output "$OUT"
