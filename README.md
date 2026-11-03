# pokemon-agent

A Pokemon Red player built **deterministic-first**: exact ROM data does the
driving, RAM tells the truth, and a language model — if one is used at all —
only supervises.

This replaces the previous design (kept under `legacy/`), in which a 6,679-line
driver arbitrated between a per-turn VLM, a learned map that kept poisoning
itself, and a neural option-scorer. The failure analysis that motivated the
rewrite is summarised at the bottom.

## Architecture

```
        ┌─────────────┐   proposes goals (JSON)   ┌──────────────┐
        │  planner    │ ─────────────────────────▶ │  validator   │
        │ rules / LLM │                            │ propose-and- │
        └─────────────┘                            │ verify       │
                                                   └──────┬───────┘
                                                          │ ok only
        ┌─────────────┐   BFS paths, warps, macros ┌──────▼───────┐
        │  world      │ ◀───────────────────────── │  executor    │
        │ (ROM truth) │                            │ event-driven │
        └─────────────┘                            └──────┬───────┘
        ┌─────────────┐   badges, party, position         │ presses
        │  state      │ ◀── RAM reads ──────────── ┌──────▼───────┐
        │             │                            │  emulator    │
        └─────────────┘                            │ PyBoy / sim  │
                                                   └──────────────┘
        Episodic memory (memory.py) records every outcome, so a goal that
        failed twice in an unchanged situation is suppressed, not retried.
```

- **`agent/world.py`** — static world model from `data/maps/` (246 maps:
  walkability, connections, warp tables, extracted from the ROM by
  `data/maps/build_walkability.py`). Never learned, never mutated at
  runtime. Tile-level BFS inside a map; map-level routing across
  connections/warps with capability gates (Surf to sea routes and the
  Power Plant, Fly between towns). `missing_capabilities()` explains *why*
  somewhere is unreachable.
- **`agent/state.py`** — game state read from WRAM (map, position, badges,
  money, party HP/levels/moves, battle flag) using pret/pokered addresses.
  Capabilities (HM use) are derived from party moves + badges.
- **`agent/goals.py`** — goals as typed values, plus the story spine as an
  ordered checklist whose completion is judged from RAM + visited-map
  memory, never from planner opinion. `validate_goal()` is the
  propose-and-verify gate.
- **`agent/planner.py`** — `ChecklistPlanner` (deterministic; the game is
  drivable with no model at all) and `LLMPlanner` (any OpenAI-compatible
  endpoint, e.g. llama.cpp serving Qwen locally; low temperature, small
  budget, JSON goals, checklist fallback on any failure). The context a
  planner needs is a few thousand tokens of game state — not an agent
  framework's 64K of scaffolding, which is why this talks to the model
  directly instead of through Hermes. Hermes' role is the outer loop:
  supervision, research, reporting.
- **`agent/executor.py`** — event-driven loop: observe → handle battle →
  propose → validate → walk. Positional stuck detection fails a goal
  *with its state signature* so identical retries get suppressed.
- **`agent/emu.py`** — the emulator seam (`press`, `read_state`,
  save/load) plus `SimEmulator`, which runs movement/warps/connections
  against the real world data so the whole loop is testable with no ROM.

## Run it

```bash
python3 -m unittest discover -s tests   # 23 tests
python3 run_demo.py                     # simulated opening: Pallet -> Pewter
```

The demo walks the actual opening of the game — starter from Oak's Lab,
Oak's Parcel from Viridian Mart, delivery, then north to Pewter City — in
530 simulated button presses.

## On a real ROM (PyBoy)

`agent/pyboy_emu.py` is the hardware backend the wiring notes used to
sketch: tile-step press semantics (hold until the RAM coordinates
change), state reads that settle out warp-transition desyncs before
trusting a position, screenshot/save/load, a dialog signal calibrated
against the real text box, and `wait_free()` for riding out scripted
scenes. `run_real.py` drives the whole opening unattended:

```bash
python3 run_real.py --rom /path/to/pokered.gbc --workdir run
```

The ROM is not in this repo; building pret/pokered from source (RGBDS)
produces a byte-identical retail Red (SHA1
ea9bcae617fdf159b045185467ae58b2e4a48b9a), which is what this was
verified against on 2026-11-03. The run, from a fresh boot: title
screen → Oak's intro with both names typed on the letter grid (VLAD /
GARY) → bedroom → the Oak intercept cutscene → Squirtle picked → rival
battle won → Route 1 → Viridian Mart parcel scene → back to Oak's Lab,
at roughly 2,000–2,800 fps headless (about 35–45× real-time; the whole
opening takes well under a minute).

Cutscenes are macros, not executor work, by design: the intro/naming
sequence (a fresh boot is deterministic, so it runs on counted presses
at a measured cadence — no RAM signal marks the naming screens), the
Oak intercept, and the starter pick (the lab's Poke Balls are inert
until the intercept arms them, and the pick crosses a full-screen
PokeDex page that is not a text box). Each macro uses only verified
backend primitives and hands control back to the Executor.

For a local model, point `LLMPlanner` at the llama.cpp server
(`http://localhost:8080`, model name of the served Qwen). Nothing else
changes: proposals are validated identically whoever makes them.

## Known approximations (read before trusting)

- The walkability grids were regenerated from the pokered source after
  the original extraction was found to have the passability test
  inverted (the engine's `coll_tiles` lists are *passable* tiles — see
  `data/maps/build_walkability.py`). Two maps (teleporter floors) use
  an open-grid fallback. Tile-PAIR collisions beyond ledges are not
  modelled; on hardware the stuck detector is the safety net.
- NPCs are transient obstacles the static grid cannot see. The executor
  handles them two ways: a tile that refuses entry is excluded and the
  walk replans around it, and a battle or dialog that interrupts a walk
  aborts the walk (never recorded as a blocked tile) so `run()` can
  handle the interruption first. A determined NPC standing in a one-tile
  corridor can still stall a goal until it wanders off.
- The story checklist judges some steps by *arrival* (reaching the
  Mart = parcel picked up; reaching the Lab = parcel delivered). The
  real game gates progress on the event scenes themselves: Viridian's
  north road stays blocked ("You can't go through here") until Oak's
  Pokedex handover scene has actually played, which the arrival-only
  delivery does not trigger. Playing that scene (walk to Oak, face him,
  run the dialog) is the known next step toward Pewter on hardware.
- The Mart's exit mat occasionally needs a second attempt to fire
  (the runner's next round exits cleanly); the retrigger logic is the
  least-hardened part of the hardware path.
- CAVERN-tileset maps are marked fully open by the data generator (their
  tile ids don't map through the collision lists); interiors there rely
  on warp chains.
- Battles are a policy seam: the shipped `FightPolicy` mashes A
  (Fight → first move), which wins the early game and nothing cleverer.
  Dialog handling is real but shallow: advance the box while one is
  verifiably open, never blind-mash (that way lies examining the SNES).

## Why the rewrite (short version)

The old system's loops had four generators: symmetric goal-text 2-cycles,
competing movement producers, a learned map corrupting itself faster than
purge tools could clean it, and a labeler that guaranteed *some*
executable goal for any model output. Its VLM could never have been
trained as set up — labels depended on state the training prompt never
showed, three divergent labelers existed despite a "single source of
truth" docstring, and the committed dataset covered 4 maps with zero
samples past Viridian. Every one of those failure modes is designed out
here: goals are typed, the map is ROM truth, state comes from RAM, and
failures are remembered.

`legacy/` holds the previous implementation (itself built on a fork of
NousResearch's pokemon-agent) for reference. It is not imported by the
new package.
