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

## 3. Rotation — the agent logs only

`~/.hermes/scripts/rotate_pokemon_logs.sh` (copy: `scripts/rotate_logs.sh`),
run by the `pokemon-log-rotation` cron job (`0 * * * *`, no-agent).

Scope: **autopilot + serve + flight recorder only.**

- rotated **daily** at 04:00 UTC, gzipped into `v2/runs/archive/`;
- **5 days** of agent-log history retained, then deleted;
- the job is **silent on success** and only prints if something needs a human
  (gzip failure, disk ≥ 85%);
- `FORCE_HOUR=04 ./rotate_pokemon_logs.sh` runs the daily pass out of band for
  testing.

### Deliberately out of scope

SB3 training is **not** touched by this cron: neither `v2/runs/checkpoints/`
nor `train_*.log`. Culling those has to be an explicit, separate decision —
a misplaced "tidy up" could delete a resume point or a milestone.

`v2/runs/checkpoints/` is the real disk consumer (~8.6 GB vs ~5 MB of agent
logs). If `/home` fills up, look there first — but prune it on purpose, not by
a routine cron.

Truncating in place is safe: writers use `>>` (O_APPEND), so the next write
lands at offset 0 of the emptied file.

## 4. Hourly analysis (cron `simulation_status_updates`, :05)

`~/.hermes/scripts/collect_pokemon_status.sh` (copy:
`scripts/collect_pokemon_status.sh`) collects SB3 + autopilot status and emits a
30-second video, then runs a rule-based screen over the last hour of the flight
recorder: action mix, battle behaviour, map progress, position drift, and stuck
detection (max turns on a tile, max stuck counter, longest position freeze).

The cron then hands that data to an agent that **investigates before judging**:
it queries the flight recorder and the autopilot log, and says whether
Von/Hermes is behaving appropriately, naming the specific turns and the likely
cause when it is not. It ends with the video's `MEDIA:` line.

Design notes:

- only genuine warp tiles are excluded from the stuck metric — an earlier
  version also excluded `heal_center`, which **masked a real Viridian City
  wedge** (agent cycling `heal_center` for 130+ of 272 turns, 45-72 turns on one
  tile at (23,12));
- the screen is deliberately crude and cheap. It exists to *raise* suspicions;
  the agent's investigation is what resolves them.
