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
python3 -m unittest discover -s tests   # 22 tests
python3 run_demo.py                     # simulated opening: Pallet -> Pewter
```

The demo walks the actual opening of the game — starter from Oak's Lab,
Oak's Parcel from Viridian Mart, delivery, then north to Pewter City — in
~360 simulated button presses.

## Wiring it to a real emulator (Orin)

Implement three verbs against PyBoy and pass the object to `Executor`:

- `press(button)` → `pyboy.button(...)` / tick
- `read_state()` → `agent.state.read_state(...)` over a reader backed by
  PyBoy memory (addresses in `state.py`; `ADDR_PARTY_MONS` is a named
  constant so it can be recalibrated in one place against the live game)
- `save_state()` / `load_state()` → PyBoy state files; snapshot before any
  planner-chosen venture (e.g. a legendary hunt) so failures are reversible.

For a local model, point `LLMPlanner` at the llama.cpp server
(`http://localhost:8080`, model name of the served Qwen). Nothing else
changes: proposals are validated identically whoever makes them.

## Known approximations (read before trusting)

- The walkability JSON stores one quadrant sample per tile; `world.py`
  collapses each 2×2 block (walkable iff any sampled quadrant is passable)
  and forces warp tiles walkable. See the `world.py` docstring. On
  hardware, live RAM collision is the final arbiter and the stuck
  detector is the safety net.
- Ledge tiles are recorded but one-way movement is **not** enforced in
  the static layer (the extracted ledge set also flags fences and city
  entrances; enforcing it seals Viridian City).
- CAVERN-tileset maps are marked fully open by the data generator (their
  tile ids don't map through the collision lists); interiors there rely
  on warp chains.
- Battles and dialog are seams, not implementations: `FightPolicy` is
  where a battle state machine plugs in, and dialog detection is a
  backend-supplied signal.

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
