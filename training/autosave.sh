#!/usr/bin/env bash
# Commits and pushes public/models/model.json whenever training promotes a new generation,
# so an overnight run survives the machine going away. Usage: training/autosave.sh <branch>
set -u
branch="${1:?branch}"
cd "$(dirname "$0")/.."
while true; do
  sleep 300
  if ! git diff --quiet -- public/models/model.json; then
    gen=$(head -c 400 public/models/model.json | grep -o '"generation":[0-9]*' | cut -d: -f2)
    git add public/models/model.json
    git commit -q -m "Overnight training: model generation ${gen}" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0168pZgSWoZPSeTVqYrt3eVX"
    for i in 1 2 3 4; do git push -q origin "$branch" && break; sleep $((2 ** i)); done
    echo "$(date -u +%H:%M) pushed generation ${gen}"
  fi
done
