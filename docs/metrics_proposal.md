# Metrics Collection Proposal for Hermes/Laya Agent

## Goal
Enable time-series graphing of key agent metrics (frontier exploration, HP, battle statistics, mode switches, etc.) to evaluate training effectiveness and agent behavior.

## Suggested Implementation

### 1. Extend `/state` endpoint
Add a `metrics` field to the JSON returned by `http://localhost:8765/state` containing:

```json
"metrics": {
  "frontier_explored": 1234,     // number of globally-explored cells (from union frontier)
  "frontier_total": 5678,        // total frontier size
  "exploration_pct": 21.7,
  "hp": 25,
  "max_hp": 36,
  "level": 12,
  "badges": 0,
  "party_size": 1,
  "steps_taken": 456789,
  "turn": 78,
  "mode": "hermes",
  "hermes_turns": 12,
  "laya_turns": 5,
  "battle_wins": 3,
  "battle_losses": 0,
  "items_used": 2,
  "money": 532,
  "species_seen": 7,
  "pokedex_owned": 1,
  "map_name": "Route 1",
  "position": {"x":18,"y":16},
  "timestamp": "2026-10-05T21:05:00+09:00"
}
```

Values already exist in internal state; just need to expose them.

### 2. Lightweight logging daemon
Create a simple script that polls `/state` every 5–10 seconds and appends a CSV line to `~/pokemon-agent/v2/runs/metrics_log.csv`.

CSV columns: timestamp, turn, mode, hp, max_hp, badges, species_seen, frontier_explored, frontier_total, exploration_pct, map_name, position_x, position_y, money, pokedex_owned, battle_wins, battle_losses.

### 3. Grafana / simple HTML dashboard
- Use an existing lightweight tool (e.g., `plotly` offline, or just `matplotlib` to generate PNGs hourly).
- Or push CSV to a Google Sheet via API for quick sharing.
- For a zero‑dependency option, add a `/metrics` endpoint that returns the last N rows as JSON and let Hermes’ built‑in server render a minimal chart with Chart.js.

### 4. Integration with cron job
Modify the existing `collect_pokemon_status.sh` to also POST the metrics JSON to a local endpoint (or just append to CSV). The cron job can then include a sparkline graphic (using `gnuplot` or `spark`) in its report.

### 5. Frontier stats source
The union frontier size and explored count are already computed in `_frontier_targets` (see `union_global` length). Export those two numbers via the `metrics` field.

## SB3 Training Progress Assessment
- **Current step**: ~10,217 (from latest log line)
- **Latest checkpoint**: `poke_12000000_steps.zip` – this indicates the SB3 trainer has already saved a checkpoint at **12 million steps**, meaning the training run has progressed far beyond the current log line (the log line is from a *different* run or the checkpoint is from an earlier extended training).
- The checkpoint file size (~14 MB) and timestamp (saved ~20:57 JST) show recent activity.
- If the SB3 process is still alive (`baseline_fast_v2.py` PID 24163), it is actively training and will continue to improve the policy.
- The training hyper‑parameters (ep‑length 163840, n‑steps 1024, batch‑size 1024, save‑freq 50000) suggest it is designed for long runs; each save represents roughly 50k steps of experience.
- **Conclusion**: Yes, the extended training is paying off – the agent already has a 12M‑step policy saved, and the live process is still accumulating steps. Expect improved exploration and battle performance as the policy converges.

## Next Steps
1. Deploy the `/state` metrics extension (few lines in `state/builder.py`).
2. Start the lightweight logger (can be a background `screen` or cron‑every‑minute task).
3. Verify CSV grows and generate a sample graph (e.g., exploration % over time).
4. Optionally, add a `/metrics` page to the existing Hermes server for on‑demand viewing.