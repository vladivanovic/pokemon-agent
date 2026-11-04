# Datasets: what exists, what was decided

Durable record of dataset facts for this project, so they survive
context turnover. Last updated 2026-11-04.

## PokeDreamer — real in-game Red trajectory set

- Location: `~/pokemon-data/pokedreamer` (82 MB, transition `.npz`
  files).
- Discussed in Discord as a genuine in-game Red trajectory
  collection. **Never trained on** — the earlier image-based plan was
  dropped when 77/3,528 frames were found missing; the `.npz`
  trajectories remain usable in principle.

## PokeBot waypoint data — movement only

- Extracted from bouletmarc/PokeBot `data/red/paths.lua`
  (`scripts/convert_pokebot_routes.py` → `data/routes_red.json`,
  197 segments / 118 maps). **Movement waypoints only, no battle
  data.** Verified live: the waypoint executor walked Mt Moon 1F.

## No public "LLM plays Pokemon battles" dataset

- None was ever found. **Smogon Showdown replay logs are the only
  public source** of turn-by-turn battle data across gens.
- Agreed verdict (Vlad, 2026-10/11): **oracle-synthesis from verified
  trajectories makes an external battle dataset unnecessary** — the
  same trick as the Metasploit tool-calling set. The
  supervisor-dataset builder is the implementation.

## battles.jsonl pilot — binned

- A 343-example battle-advisor pilot (raw + chat formats) built by the
  other AI's tooling lived at
  `pokemon-agent/training/data/battles.jsonl` (+ `battles_chat.jsonl`).
- Binned 2026-10-28 during the repo reset cleanup; it was untracked
  and is regenerable by `scripts/build_supervisor_dataset.py`. It was
  a format test, not a training set.

## Supervisor training data status

- Current set: 800 oracle-synthesized examples
  (`dataset/supervisor/`, 760/40 split, 12 target maps).
- Known gap (battery v2, 2026-11-04): no *recovery* rows — failure
  events in the context paired with a different validated target.
  Sources for the next round (waiting for go): the battery's
  stock-recovery positives already in
  `~/workspace/pokemon-play/supervisor_log.jsonl`, plus a recovery
  oracle in the synthesizer.
