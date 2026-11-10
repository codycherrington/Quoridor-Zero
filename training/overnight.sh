#!/usr/bin/env bash
# Idempotent launcher for a long training run: resumes from the newest checkpoint if training
# isn't already running, and makes sure the autosave loop is up. Safe to call repeatedly.
#   training/overnight.sh <branch> <stop-time-ISO8601>
set -u
branch="${1:?branch}"
until="${2:?stop time}"
cd "$(dirname "$0")/.."
if ! pgrep -f "autosave.sh" > /dev/null; then
  setsid nohup training/autosave.sh "$branch" >> training/runs/autosave.log 2>&1 < /dev/null &
  echo "started autosave"
fi
if pgrep -f "training/train.ts" > /dev/null; then
  echo "training already running"
  exit 0
fi
if [ "$(date -u +%s)" -ge "$(date -u -d "$until" +%s)" ]; then
  echo "past stop time $until; not starting"
  exit 0
fi
latest=$(ls training/runs/gen*.json 2> /dev/null | sort | tail -1)
setsid nohup node training/train.ts ${latest:+--resume "$latest"} --iters 1000 --games 192 --sims 200 \
  --steps 400 --blend 0.2 --gate-games 48 --until "$until" >> training/runs/stdout.log 2>&1 < /dev/null &
echo "started training from ${latest:-scratch}"
