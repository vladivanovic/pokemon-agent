#!/usr/bin/env bash
# Supervised Von driver: restart on ANY exit so a crash never costs hours.
#
# Why this exists: the 13:12 crash (empty criteria -> SystemExit) left the
# game frozen on the Route 3 Youngster's dialog for 3.2 hours because
# nothing restarted the driver. The in-code fixes make that specific crash
# impossible, but a supervisor is the general answer - the driver process
# should never be a single point of failure for a multi-hour run.
#
# Backoff: 5s after a clean-ish exit, growing to 60s if the driver dies
# repeatedly (so a hard config error does not spin the CPU forever).
set -u

REPO=/home/vlad/pokemon-agent
VENV=/home/vlad/hermes_venv
# Configurable so the same supervisor covers both the live VLM run and a
# rules-planner DATA-COLLECTION run (teacher labels for the next retrain).
PLANNER="${PLANNER:-vlm}"
PLAN_INTERVAL="${PLAN_INTERVAL:-15}"
LOG="${VON_LOG:-$REPO/v2/runs/von_vlm.log}"
SUPLOG="${VON_SUPLOG:-$REPO/v2/runs/von_supervisor.log}"

cd "$REPO" || exit 1

export POKEMON_SKIP_PREFLIGHT=1
export POKEMON_PLANNER_URL=http://127.0.0.1:8002/v1
export POKEMON_PLANNER_MODEL=planner

backoff=5
attempt=0

while true; do
    attempt=$((attempt + 1))
    start_ts=$(date +%s)
    echo "[supervisor $(date '+%F %T')] start attempt=$attempt backoff=${backoff}s" \
        >> "$SUPLOG"
    "$VENV/bin/pokemon-agent" play \
        --host localhost --port 8765 \
        --scorer von --no-vision \
        --stall-seconds 45 --hermes-turns 8 --turn-timeout 240 \
        --planner "$PLANNER" --plan-interval "$PLAN_INTERVAL" \
        >> "$LOG" 2>&1
    rc=$?
    ran=$(( $(date +%s) - start_ts ))
    # A run that stayed up a while is healthy - reset the backoff. Only a
    # FAST death (crash loop) grows it, so a config error cannot make the
    # supervisor spin, while a long healthy run restarts immediately.
    if [ "$ran" -ge 300 ]; then
        backoff=5
    elif [ "$backoff" -lt 60 ]; then
        backoff=$((backoff * 2))
    fi
    echo "[supervisor $(date '+%F %T')] driver exited rc=$rc after ${ran}s" \
         "- restarting in ${backoff}s" >> "$SUPLOG"
    sleep "$backoff"
done
