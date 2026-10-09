#!/bin/bash
set -u

# Collect all data for the status report
JST_TIME=$(TZ=Asia/Tokyo date '+%H:%M')

# SB3 Training
SB3_PID=$(ps aux | grep "baseline_fast_v2.py" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$SB3_PID" ]; then
    # Get latest step
    STEP=$(grep -aoE "step:[[:space:]]+[0-9]+" /home/vlad/pokemon-agent/v2/runs/train_*.log 2>/dev/null | tail -1 | awk '{print $NF}' || echo "N/A")
    
    # Get latest checkpoint
    CHECKPOINT=$(ls -t /home/vlad/pokemon-agent/v2/runs/checkpoints/poke_*_steps.zip 2>/dev/null | head -1 || ls -t /home/vlad/pokemon-agent/v2/runs/poke_*_steps.zip 2>/dev/null | head -1)
    if [ -n "$CHECKPOINT" ]; then
        CHECKPOINT_NAME=$(basename "$CHECKPOINT")
        CHECKPOINT_TIME=$(TZ=Asia/Tokyo date -r "$CHECKPOINT" '+%H:%M' 2>/dev/null || echo "N/A")
        CHECKPOINT_STEPS=$(echo "$CHECKPOINT_NAME" | sed -E 's/poke_([0-9]+)_steps\.zip/\1/')
    else
        CHECKPOINT_NAME="N/A"
        CHECKPOINT_TIME="N/A"
        CHECKPOINT_STEPS="N/A"
    fi
    
    # SB3 Trend Analysis: compare last 2 episode reward means
    SB3_TREND="➡️ stable"
    REWARD_LINES=$(grep -E "ep_rew_mean" /home/vlad/pokemon-agent/v2/runs/train_*.log 2>/dev/null | tail -2)
    if [ -n "$REWARD_LINES" ]; then
        REW1=$(echo "$REWARD_LINES" | head -1 | sed -E 's/.*ep_rew_mean[[:space:]]*[|]*[[:space:]]*([0-9.]+).*/\1/')
        REW2=$(echo "$REWARD_LINES" | tail -1 | sed -E 's/.*ep_rew_mean[[:space:]]*[|]*[[:space:]]*([0-9.]+).*/\1/')
        if [ -n "$REW1" ] && [ -n "$REW2" ]; then
            if (( $(echo "$REW2 > $REW1 + 5" | bc -l 2>/dev/null || echo 0) )); then
                SB3_TREND="📈 improving"
            elif (( $(echo "$REW2 < $REW1 - 5" | bc -l 2>/dev/null || echo 0) )); then
                SB3_TREND="📉 declining"
            fi
        fi
    fi
    
    SB3_STATUS="✅ running (PID $SB3_PID) · $SB3_TREND"
else
    STEP="N/A"
    CHECKPOINT_NAME="N/A"
    CHECKPOINT_TIME="N/A"
    CHECKPOINT_STEPS="N/A"
    SB3_STATUS="❌ not running"
    SB3_TREND="N/A"
fi

# Autopilot
AUTOPILOT_PID=$(ps aux | grep "pokemon-agent play" | grep -v grep | grep -v SCREEN | grep -v bash | awk '{print $2}' | head -1)
if [ -n "$AUTOPILOT_PID" ]; then
    # Query state endpoint
    STATE_JSON=$(curl -s -m 5 http://localhost:8765/state 2>/dev/null || echo '{}')
    
    TURN=$(curl -s -m 5 http://localhost:8765/stats 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('turn', 'N/A'))" 2>/dev/null || echo "N/A")
    MODE=$(curl -s -m 5 http://localhost:8765/stats 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('mode', 'N/A'))" 2>/dev/null || echo "N/A")
    PLAYER_NAME=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('player',{}).get('name','N/A'))" 2>/dev/null || echo "N/A")
    HP=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('party',[{}])[0].get('hp','N/A'))" 2>/dev/null || echo "N/A")
    MAX_HP=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('party',[{}])[0].get('max_hp','N/A'))" 2>/dev/null || echo "N/A")
    LEVEL=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('party',[{}])[0].get('level','N/A'))" 2>/dev/null || echo "N/A")
    
    # Get actual species name from party data (not hardcoded Bulbasaur)
    SPECIES_NAME=$(echo "$STATE_JSON" | python3 -c "
import sys,json
d=json.load(sys.stdin)
party = d.get('party',[])
if party:
    p = party[0]
    name = p.get('name','').strip()
    if name:
        print(name)
    else:
        sp = p.get('species')
        if sp is not None:
            print(sp)
        else:
            print('Unknown')
else:
    print('No party')
" 2>/dev/null || echo "Unknown")
    
    POS_X=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('player',{}).get('position',{}).get('x','N/A'))" 2>/dev/null || echo "N/A")
    POS_Y=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('player',{}).get('position',{}).get('y','N/A'))" 2>/dev/null || echo "N/A")
    POKEDEX=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Yes' if d.get('flags',{}).get('has_pokedex') else 'No')" 2>/dev/null || echo "N/A")
    PARCEL=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print('Yes' if d.get('flags',{}).get('has_oaks_parcel') else 'No')" 2>/dev/null || echo "N/A")
    MONEY=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('player',{}).get('money','N/A'))" 2>/dev/null || echo "N/A")
    BADGES=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('player',{}).get('badge_count','N/A'))" 2>/dev/null || echo "N/A")
    FRAME_COUNT=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('metadata',{}).get('frame_count','N/A'))" 2>/dev/null || echo "N/A")
    IN_BATTLE=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('battle',{}).get('in_battle',False))" 2>/dev/null || echo "false")
    
    if [ "$IN_BATTLE" = "True" ] || [ "$IN_BATTLE" = "true" ]; then
        ENEMY_SPECIES=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('battle',{}).get('enemy',{}).get('species','Unknown'))" 2>/dev/null || echo "Unknown")
        ENEMY_LEVEL=$(echo "$STATE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('battle',{}).get('enemy',{}).get('level','?'))" 2>/dev/null || echo "?")
        BATTLE_TEXT="$ENEMY_SPECIES Lv$ENEMY_LEVEL"
    else
        BATTLE_TEXT="not in battle"
    fi
    
    # Get uptime
    UPTIME=$(ps -o etime= -p "$AUTOPILOT_PID" 2>/dev/null | awk '{print $1}' || echo "N/A")
    
    AUTOPILOT_STATUS="✅ running (PID $AUTOPILOT_PID)"
else
    TURN="N/A"
    MODE="N/A"
    PLAYER_NAME="N/A"
    HP="N/A"
    MAX_HP="N/A"
    LEVEL="N/A"
    SPECIES_NAME="N/A"
    POS_X="N/A"
    POS_Y="N/A"
    POKEDEX="N/A"
    PARCEL="N/A"
    MONEY="N/A"
    BADGES="N/A"
    FRAME_COUNT="N/A"
    BATTLE_TEXT="N/A"
    UPTIME="N/A"
    AUTOPILOT_STATUS="❌ not running"
fi

# 30-second Video (replaces screenshot)
VIDEO_PATH="/tmp/pokemon_30s_$(date +%s).mp4"
curl -s -m 40 -X POST http://localhost:8765/screenrecord \
     -H "Content-Type: application/json" \
     -d '{"duration":30,"fps":20,"scale":2}' \
     > "$VIDEO_PATH" 2>/dev/null || true

if [ -f "$VIDEO_PATH" ] && [ -s "$VIDEO_PATH" ]; then
    VIDEO_MEDIA="MEDIA: $VIDEO_PATH"
else
    VIDEO_MEDIA="(video unavailable)"
fi

# --- Hourly Log Analysis ---
ANALYSIS=""
if [ -f "/home/vlad/.pokemon-agent/flight_recorder.jsonl" ]; then
    ANALYSIS=$(python3 << 'PYEOF'
import json, sys, os
from datetime import datetime, timezone, timedelta

path = "/home/vlad/.pokemon-agent/flight_recorder.jsonl"
now = datetime.now(timezone.utc)
cutoff = now - timedelta(hours=1)

turns = []
battle_turns = []
stuck_events = []
interact_count = 0
move_count = 0
flee_count = 0
heal_count = 0
route_count = 0
explore_count = 0
maps_seen = set()
positions = []
here_max = 0
stuck_max = 0
prev_pos = None
stuck_streak = 0
freeze_max = 0
freeze_pos = None

with open(path) as f:
    for line in f:
        try:
            d = json.loads(line)
        except:
            continue
        try:
            ts = datetime.fromisoformat(d.get("ts", "")).replace(tzinfo=timezone.utc)
        except:
            continue
        if ts < cutoff:
            continue
        turns.append(d)
        if d.get("battle", {}).get("in_battle"):
            battle_turns.append(d)
        c = d.get("choice", "")
        if c == "interact":
            interact_count += 1
        elif c.startswith("move_"):
            move_count += 1
        elif c == "flee":
            flee_count += 1
        elif c == "heal_center" or c.startswith("heal_route_"):
            heal_count += 1
        elif c.startswith("route_"):
            route_count += 1
        elif c.startswith("explore_") or c.startswith("travel_") or c.startswith("goto_"):
            explore_count += 1
        m = d.get("map")
        if m:
            maps_seen.add(m)
        pos = d.get("pos")
        if pos:
            positions.append(tuple(pos))
            here = d.get("here", 0)
            stuck = d.get("stuck", 0)
            # Only a warp tile is a legitimate place to sit for many turns
            # (stepping onto/off a door). heal_center CAN genuinely wedge, so
            # it is NOT excluded - that mistake hid a real Viridian City wedge.
            on_warp = d.get("door", False)
            if here > here_max and not on_warp:
                here_max = here
            if stuck > stuck_max:
                stuck_max = stuck
            if prev_pos == pos:
                stuck_streak += 1
                if stuck_streak > freeze_max:
                    freeze_max = stuck_streak
                    freeze_pos = pos
            else:
                stuck_streak = 0
            prev_pos = pos

if not turns:
    print("ANALYSIS: no turns in the last hour - agent may have been idle or trace not started")
    sys.exit(0)

# Progress metrics
unique_positions = len(set(positions))
total_turns = len(turns)

# Stuck detection
stuck_warning = ""
if here_max >= 15:
    stuck_warning += f" ⚠️ TILE_STUCK: max turns on one tile={here_max}"
if stuck_max >= 8:
    stuck_warning += f" ⚠️ STUCK_COUNTER: max stuck={stuck_max}"
if freeze_max >= 8:
    stuck_warning += (f" ⚠️ POSITION_FROZEN: {freeze_max} consecutive turns "
                      f"at {freeze_pos}")

# Battle behavior
battle_flee_rate = ""
if battle_turns:
    bflee = sum(1 for d in battle_turns if d.get("choice") == "flee")
    bmove = sum(1 for d in battle_turns if d.get("choice", "").startswith("move_"))
    btotal = len(battle_turns)
    battle_flee_rate = f"  battle: {btotal} turns, flee={bflee}, move={bmove}"
    if bflee > 0 and bmove == 0:
        battle_flee_rate += " ⚠️ ONLY_FLEEING"
    elif bflee == 0 and bmove == 0 and btotal > 3:
        battle_flee_rate += " ⚠️ NO_ACTION_IN_BATTLE"

# Map progress
map_progress = ""
if len(maps_seen) == 1:
    map_progress = f"  map: STUCK on {list(maps_seen)[0]} (no map change)"
elif len(maps_seen) > 1:
    map_progress = f"  maps: {len(maps_seen)} ({', '.join(sorted(maps_seen))})"

# Directional progress
pos_progress = ""
if positions:
    x0, y0 = positions[0]
    x1, y1 = positions[-1]
    dx, dy = x1 - x0, y1 - y0
    dist = int((dx*dx + dy*dy)**0.5)
    pos_progress = f"  position drift: ({x0},{y0}) -> ({x1},{y1}) dist={dist}"

# Action mix
action_mix = f"  actions: interact={interact_count}, move={move_count}, flee={flee_count}, heal={heal_count}, route={route_count}, explore={explore_count}"

# Verdict
verdict = "✅ APPROPRIATE"
warnings = []
if "⚠️" in stuck_warning:
    verdict = "⚠️ NEEDS ATTENTION"
    warnings.append("stuck")
if "ONLY_FLEEING" in battle_flee_rate:
    verdict = "⚠️ NEEDS ATTENTION"
    warnings.append("battle_flee")
if len(maps_seen) == 1 and total_turns > 30:
    verdict = "⚠️ NEEDS ATTENTION"
    warnings.append("no_map_progress")
if interact_count > total_turns * 0.5:
    verdict = "⚠️ NEEDS ATTENTION"
    warnings.append("interact_heavy")

summary = (
    f"ANALYSIS (last hour): {total_turns} turns, {unique_positions} unique tiles, "
    f"battle_turns={len(battle_turns)} | {verdict}\n"
    f"  max_here={here_max} max_stuck={stuck_max} max_freeze={freeze_max}"
    f"{stuck_warning}{battle_flee_rate}{map_progress}{pos_progress}\n"
    f"{action_mix}"
)
print(summary)
PYEOF
)
fi

# Output the report
cat << REPORT
Cronjob Response: simulation_status_updates
(job_id: cfd59a58bdac)
-------------

All checks complete. Status summary composed:

**⚡ Pokemon Red — Hourly Status ($JST_TIME JST)**

**🤖 SB3 Training** (\`pokemon_train_12core\`) — $SB3_STATUS
- Episode step ~$STEP / $CHECKPOINT_STEPS
- Latest checkpoint: \`$CHECKPOINT_NAME\` (saved $CHECKPOINT_TIME) · approx total train time on Orin CUDA
- Trend: $SB3_TREND (last ep_rew_mean comparison)

**🎮 Laya/Hermes Autopilot** — $AUTOPILOT_STATUS
- Turn $TURN, $MODE mode; actively polling the emulator server
- Player \`$PLAYER_NAME\` · **$SPECIES_NAME Lv$LEVEL** · **HP $HP/$MAX_HP** — currently $BATTLE_TEXT
- Map pos ($POS_X,$POS_Y) · Pokédex $POKEDEX · Oak's parcel $PARCEL · \$$MONEY · $BADGES badges
- Emulator: $FRAME_COUNT frames, uptime ~$UPTIME

**📹 30-second Video** (20fps, 2x scale, MP4)
$VIDEO_MEDIA

**🧠 Hourly Log Analysis**
$ANALYSIS
REPORT