# Logging & tracing

Three layers, so a question about any turn is one query away instead of a grep
through millions of characters.

## 1. Flight recorder — `~/.pokemon-agent/flight_recorder.jsonl`

One JSON object per decision, written by the autopilot. This is the primary
troubleshooting tool.

Fields of interest:

| field | meaning |
|---|---|
| `turn`, `ts` | when |
| `choice` | the key the scorer/policy picked (`move_0`, `route_toward_north`, `flee`, …) |
| `actions` | the button presses actually dispatched |
| `options` | every option that was on the table that turn |
| `policy_driven` | true when the deterministic battle policy chose it (not the scorer) |
| `battle.active_moves` | raw PP from the **battle struct** (`wBattleMon`) — what the policy scores |
| `battle.party_moves` | raw PP from the **party struct** — the durable copy |
| `battle.enemy` | `[species, level, hp, max_hp]` |
| `here` / `stuck` | turns on this tile / turns without moving |

Common queries:

```bash
# every battle turn with the PP both sources reported
jq -c 'select(.battle.in_battle) | {turn,choice,opts:.options,
        active:.battle.active_moves, party:.battle.party_moves,
        enemy:.battle.enemy}' ~/.pokemon-agent/flight_recorder.jsonl

# why did it flee? (the option set tells you)
jq -c 'select(.choice=="flee") | {turn,options,hp:.battle.my_hp,
        enemy:.battle.enemy}' ~/.pokemon-agent/flight_recorder.jsonl

# longest stalls on one tile
jq -r 'select(.here>8) | "\(.turn) \(.map) \(.pos) here=\(.here)"' \
    ~/.pokemon-agent/flight_recorder.jsonl

# where did the policy disagree with the scorer?
jq -c 'select(.policy_driven==false and .choice|startswith("move_")) |
       {turn,choice,options}' ~/.pokemon-agent/flight_recorder.jsonl
```

Path override: `POKEMON_TRACE=/some/path.jsonl`.

## 2. Human log — `v2/runs/autopilot_qwen65k.log`

INFO level, one line per turn. `urllib3`/`requests` are pinned to WARNING
because their DEBUG output logged every `/state` request and buried the real
decisions.

## 3. Rotation — hourly, 12 hour retention

`~/.hermes/scripts/rotate_pokemon_logs.sh` (copy: `scripts/rotate_logs.sh`),
run by the `pokemon-log-rotation` cron job (`0 * * * *`, no-agent).

- small logs (`autopilot_*.log`, `serve_fix.log`, `flight_recorder.jsonl`) are
  gzipped whole into `v2/runs/archive/` then truncated in place;
- big training logs (`train_*.log`, 100s of MB) are only tail-archived (last
  40 MB) so an hourly gzip of a 900 MB file cannot become the disk problem;
- archives older than 12 h are deleted; a hard cap drops the oldest half if the
  archive dir exceeds 3 GB;
- the job is **silent on success** and only prints if something needs a human
  (gzip failure, archive pressure, disk ≥ 85%).

Truncating in place is safe: writers use `>>` (O_APPEND), so the next write
lands at offset 0 of the emptied file.

### Disk note

The logs were never the real disk consumer — `v2/runs/checkpoints/` (SB3 model
saves) held ~8.5 GB vs ~0.09 GB of logs. If `/home` fills up, look at
checkpoints first, not logs.
