#!/usr/bin/env bash
# Von/Hermes agent log rotation: daily, 5-day retention.
# Does NOT touch SB3 checkpoints or training logs - those are managed separately.
#
# Cron: 0 * * * * (hourly). Daily pass at DAILY_HOUR (default 04:00 UTC).
# No-agent, SILENT on success. Prints only on real problems.
set -uo pipefail

LOG_DIR="/home/vlad/pokemon-agent/v2/runs"
ARCHIVE_DIR="$LOG_DIR/archive"
TRACE="$HOME/.pokemon-agent/flight_recorder.jsonl"

AGENT_KEEP_DAYS=5          # agent logs: 5 days
DAILY_HOUR="04"            # UTC hour for the daily rotation
TS=$(date -u +"%Y%m%d_%H%M%S")
DAY=$(date -u +"%Y%m%d")
HOUR="${FORCE_HOUR:-$(date -u +"%H")}"   # FORCE_HOUR lets us test the daily pass
NOTES=""

mkdir -p "$ARCHIVE_DIR" || { echo "housekeeping: cannot create dirs"; exit 1; }

rotate_small() { # whole file -> archive, then truncate in place
    local f="$1" base; [ -s "$f" ] || return 0; base=$(basename "$f")
    gzip -c "$f" > "$ARCHIVE_DIR/${base}.${DAY}.gz" \
        || { NOTES="$NOTES
  ERROR gzip failed for $base"; return 0; }
    : > "$f"
}

# --- hourly: do nothing (agent logs are small) -------------------------------

# --- daily: rotate the agent logs, keep 5 days -------------------------------
if [ "$HOUR" = "$DAILY_HOUR" ]; then
    rotate_small "$LOG_DIR/autopilot_qwen65k.log"
    rotate_small "$LOG_DIR/serve_fix.log"
    rotate_small "$TRACE"
    # keep only the newest archive of each agent log per day
    for pat in autopilot_qwen65k.log serve_fix.log flight_recorder.jsonl; do
        n=$(ls -1t "$ARCHIVE_DIR/$pat".*.gz 2>/dev/null | wc -l)
        if [ "$n" -gt 1 ]; then
            ls -1t "$ARCHIVE_DIR/$pat".*.gz | tail -n +2 | xargs -r rm -f
        fi
    done
fi

# --- retention: drop agent archives older than 5 days ------------------------
find "$ARCHIVE_DIR" -name "*.gz" ! -name "*.tail.gz" -mtime +"$AGENT_KEEP_DAYS" -delete 2>/dev/null

# disk guard
USE=$(df --output=pcent /home 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${USE:-}" ] && [ "$USE" -ge 85 ]; then
    NOTES="$NOTES
  DISK ${USE}% full on /home"
fi

if [ -n "$NOTES" ]; then
    echo "pokemon-agent log rotation:${NOTES}
  archive=$(du -sh "$ARCHIVE_DIR" 2>/dev/null | cut -f1) disk=${USE}%"
fi
exit 0