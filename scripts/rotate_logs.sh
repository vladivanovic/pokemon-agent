#!/usr/bin/env bash
# Hourly rotation + 12h retention for the pokemon-agent logs.
#
# Designed for a no_agent cron job: SILENT on success, prints only when
# something needs a human (rotation error, or disk/archive pressure).
set -uo pipefail

LOG_DIR="/home/vlad/pokemon-agent/v2/runs"
ARCHIVE_DIR="$LOG_DIR/archive"
TRACE="$HOME/.pokemon-agent/flight_recorder.jsonl"
RETENTION_MIN=720                                  # 12 hours
BIG_LOG_CAP=$((40 * 1024 * 1024))                  # keep only last 40MB of huge logs
ARCHIVE_SOFT_CAP_MB=3000
DISK_ALERT_PCT=85
TS=$(date -u +"%Y%m%d_%H%M%S")
NOTES=""

mkdir -p "$ARCHIVE_DIR" || { echo "log-rotate: cannot create $ARCHIVE_DIR"; exit 1; }

rotate_small() { # archive whole file, then truncate in place
    local f="$1" base
    [ -s "$f" ] || return 0
    base=$(basename "$f")
    if ! gzip -c "$f" > "$ARCHIVE_DIR/${base}.${TS}.gz"; then
        NOTES="$NOTES
  ERROR gzip failed for $base"; return 0
    fi
    : > "$f"
}

rotate_big() { # archive only the tail so an hourly gzip of a 900MB training
               # log cannot itself become the disk problem
    local f="$1" base
    [ -s "$f" ] || return 0
    base=$(basename "$f")
    tail -c "$BIG_LOG_CAP" "$f" | gzip -c > "$ARCHIVE_DIR/${base}.${TS}.tail.gz" || return 0
    : > "$f"
}

rotate_small "$LOG_DIR/autopilot_qwen65k.log"
rotate_small "$LOG_DIR/serve_fix.log"
rotate_small "$TRACE"
for f in "$LOG_DIR"/train_*.log; do [ -e "$f" ] && rotate_big "$f"; done

# retention: drop archives older than 12h
find "$ARCHIVE_DIR" -name "*.gz" -mmin +"$RETENTION_MIN" -delete 2>/dev/null

# hard cap on the archive dir
ARCH_MB=$(du -sm "$ARCHIVE_DIR" 2>/dev/null | cut -f1); ARCH_MB=${ARCH_MB:-0}
if [ "$ARCH_MB" -gt "$ARCHIVE_SOFT_CAP_MB" ]; then
    n=$(find "$ARCHIVE_DIR" -name '*.gz' | wc -l)
    find "$ARCHIVE_DIR" -name "*.gz" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | head -n $(( n / 2 )) | cut -d' ' -f2- | xargs -r rm -f
    NOTES="$NOTES
  archive dir was ${ARCH_MB}MB - dropped oldest half"
fi

# disk guard
USE=$(df --output=pcent /home 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${USE:-}" ] && [ "$USE" -ge "$DISK_ALERT_PCT" ]; then
    NOTES="$NOTES
  DISK ${USE}% full on /home"
fi

# only speak when there is something worth saying
if [ -n "$NOTES" ]; then
    echo "pokemon-agent log rotation:${NOTES}
  runs=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1) archive=${ARCH_MB}MB disk=${USE}%"
fi
exit 0
