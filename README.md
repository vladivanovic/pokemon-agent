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
python3 -m unittest discover -s tests   # the test suite
python3 run_demo.py                     # simulated opening: Pallet -> Pewter
```

The demo walks the actual opening of the game — starter from Oak's Lab,
Oak's Parcel from Viridian Mart, delivery, then north to Pewter City — in
530 simulated button presses, against the simulated backend (no ROM
needed).

## Play the whole game (real ROM)

`play_full_game.py` plays the **entire game in one process** — power-on
→ intro → all eight badges → Elite Four → Champion → Hall of Fame — by
sequencing the same ported leg functions in `run_real.py` that produced
the completed save line ([`docs/supervisor-training.md`](docs/supervisor-training.md)
§§1–28). It checkpoints after every phase and retries a failed phase
from its start checkpoint (max 2 by default, then it stops and reports —
it never skips a phase silently).

Prerequisites: Python ≥ 3.10 and PyBoy:

```bash
pip install -e .[pyboy]        # or simply: pip install pyboy
```

You also need a Pokémon Red ROM, which is **not distributed with this
repo**: supply your own, or build one from pret/pokered (RGBDS) — that
build is byte-identical to retail Red (SHA1
ea9bcae617fdf159b045185467ae58b2e4a48b9a), which is what everything here
was verified against.

```bash
export POKERED_ROM=/path/to/pokered.gbc   # or pass --rom each time

python3 play_full_game.py                    # the full game, headless
python3 play_full_game.py --web 8000         # + the live viewer (below)
python3 play_full_game.py --list-phases      # the 34-phase table
python3 play_full_game.py --out run --resume # continue an interrupted run
python3 play_full_game.py --from-phase silph --state some.state
python3 play_full_game.py --directives dir.jsonl  # + live redirects (below)
```

Checkpoints land in `run/checkpoints/NN-phase.state`, with a running
`run/progress.log` carrying RAM-verified badges/party/money per phase.
Save states stay out of git; they are plain files and copy between
machines byte-for-byte.

### Live viewer

`--web PORT` serves a small page (Python stdlib only, no extra
dependencies): the live game screen, map/badges/party/dex, the current
phase, and Start/Stop controls. It binds **localhost by default**; to
watch an Orin run from another machine on your LAN, add
`--host 0.0.0.0`. Add `--web-manual` to wait for the page's Start
button instead of starting immediately. (The NousResearch-era
dashboard under `legacy/` is wired to the legacy driver and kept for
reference; this viewer is the supported one — see `web_viewer.py` for
why.)

### Redirecting a live run

`--directives FILE` opens a channel through which anything outside
the process — Hermes, another LLM, a script, or you — can interrupt
and redirect a run in progress. The channel is an append-only JSON
Lines file; each line is one directive:

```bash
# send the agent somewhere else (a goal, in the repo's typed format)
echo '{"goal": {"target_map_name": "ViridianCity", "reason": "detour"}}' >> dir.jsonl
# hold the run at the next safe point / let it continue
echo '{"action": "pause"}'  >> dir.jsonl
echo '{"action": "resume"}' >> dir.jsonl
# end the run cleanly (checkpoints run/stopped.state first)
echo '{"action": "stop"}'   >> dir.jsonl
```

A goal names its target by exact map name (`target_map_name`, the
form the LLM planner emits) or by id (`target_map`), with an optional
`"target_xy": [x, y]`, `"reason"` and `"optional"` — the same shape
`agent/supervisor_log.py` writes for supervisor proposals. With
`--web` on, the same channel has a second door: the viewer page
carries a redirect form (goal box + Pause/Resume/Stop buttons), and
`POST /directive` accepts the same JSON directly; without an explicit
`--directives`, web runs default the file to `run/directives.jsonl`.

The rules are the propose-and-verify rules, applied to outsiders
exactly as to the built-in planner:

- **Nothing executes unvalidated.** A goal directive is parsed into a
  typed `Goal` and run through `validate_goal` against the live RAM
  state. Accepted, it pre-empts the storyline, executes through the
  executor, and the run resumes — the checklist re-derives the next
  story step from RAM, so a redirect that moved the player is
  absorbed like a detour. Rejected, it is logged *with the
  validator's reasons* and the run continues undisturbed.
- **Granularity is phase boundaries and executor rounds.** The
  runner services the file before every phase and between rounds of
  the planner-driven stretches (the opening, the Vermilion loop);
  the ported leg functions are atomic, so a directive that lands
  mid-leg is picked up when the leg ends. A redirect is an inserted
  validated goal, never mid-leg surgery.
- **Pause holds; stop saves.** `pause` blocks at the next service
  point until `resume` or `stop`; goals sent while paused queue and
  run, in order, after the resume. `stop` saves `run/stopped.state`
  and halts with exit code 0 — pick it up again with
  `--from-phase <phase> --state run/stopped.state`.
- **Exactly once.** Consumed lines are tracked by a byte offset
  persisted in `run/directives.offset`, so a `--resume` restart does
  not replay old directives.

Every directive lands in `run/progress.log` and in a per-run audit
(`run/directives-audit.jsonl`); goal directives also appear in the run's
`run/supervisor_log.jsonl`, tagged `planner: "external"` with the
validator's verdict and the execution outcome — the same records the
built-in supervisor produces. This composes with `--supervisor`
rather than competing with it: the supervisor is the *built-in*
proposer (a local model suggesting goals at planner rounds), the
directives channel is the *external* proposer — both feed the same
validator and the same executor. For an agent like Hermes, steering
a run is: append a line (or POST), then watch `/state.json` and
`run/progress.log` for the verdict and the outcome.

### Running a single leg

Every storyline leg is a function in `run_real.py`, and the pattern the
game was built with is: load a save state, build an executor, call the
leg, verify from RAM.

```python
emu = PyBoyEmulator(rom)
emu.load_state("some.state"); emu.tick(30)
ex = Executor(emu, default_world(), ChecklistPlanner(),
              EpisodicMemory(), battle_policy=SmartFightPolicy())
rock_tunnel_leg(emu, ex)   # ...or any other leg
```

`run_real.py`'s own `main()` still plays the opening standalone
(through the Vermilion hooks):

```bash
python3 run_real.py --rom /path/to/pokered.gbc --workdir run
```

### On an Orin (AGX / Nano)

The deterministic agent is light — PyBoy plus the world model — and
ran the full stack on an AGX Orin 64GB in November 2026 (the
supervisor live run, docs §8). An Orin Nano Super is fine for it too.
The only heavy optional piece is the supervisor LLM
(`--supervisor adapter|stock`, applied to the opening planner); it is
**off by default** — the deterministic line is the product. The
scripts set the headless SDL dummy drivers themselves; no display is
needed.

### Honesty note

Each phase is individually proven against the real ROM, and most
segments were cold-replayed from checkpoints during the build; the
phase chain in `play_full_game.py` is that same code, with the save
line's decisions baked in (Squirtle, Fire Stone → Arcanine, Hitmonlee,
Surf + Strength on Lapras, the Route 16 Snorlax caught — Master Ball
never thrown — and the League team). What has *not* yet been done is a
single-process power-on-to-credits soak of the runner itself; that is
the remaining hardening step, and the checkpoint/retry design exists
so a soak — or your own run — stops loudly at a phase boundary
instead of corrupting a save line. Three segments are hand-written
glue rather than single ported functions (the Fly leg's journeys:
`fly-journey`, `snorlax-journey`, `teach-fly`), assembled from the
same primitives along the legs' documented routes; treat their first
full-run outing as part of the soak.

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
battle won → Route 1 → Viridian Mart parcel scene → back to Oak's Lab
→ **the Pokedex handover scene** → Viridian's north road → Route 2 →
Viridian Forest → Pewter City → **the Boulder Badge** (the first
Brock attempt blacked out; the agent re-ran the journey, levelled on
the way, and won the rematch — all unattended). The executor phase
runs at roughly 2,300 fps headless (about 40× real-time; the full
boot-to-badge run takes about 90 seconds of emulation).

Cutscenes are macros, not executor work, by design: the intro/naming
sequence (a fresh boot is deterministic, so it runs on counted presses
at a measured cadence — no RAM signal marks the naming screens), the
Oak intercept, the starter pick (the lab's Poke Balls are inert until
the intercept arms them, and the pick crosses a full-screen PokeDex
page that is not a text box), the Pokedex handover (talk to Oak from
(5, 3); verified by the EVENT_GOT_POKEDEX flag itself, index 37 —
event constants have `const_skip` gaps, so never count them naively),
and the Brock fight (walk the gym's aisle, ride the Jr. Trainer's
line-of-sight battle, talk to Brock from (4, 2); verified by the
badge bit in RAM). Each macro uses only verified backend primitives
and hands control back to the Executor.

For a local model, point `LLMPlanner` at the llama.cpp server
(`http://localhost:8080`, model name of the served Qwen). Nothing else
changes: proposals are validated identically whoever makes them.

Every `LLMPlanner` round is also logged (proposal, verdict, outcome)
to `~/workspace/pokemon-play/supervisor_log.jsonl` — that log, plus
scenarios synthesized from the world model, is the training set for a
fine-tuned supervisor. The full pipeline (log → dataset → LoRA on the
Orin → stuck-state battery eval) is in
[`docs/supervisor-training.md`](docs/supervisor-training.md);
builders/trainers live in `scripts/`.

## Known approximations (read before trusting)

- The walkability grids were regenerated from the pokered source after
  the original extraction was found to have the passability test
  inverted (the engine's `coll_tiles` lists are *passable* tiles — see
  `data/maps/build_walkability.py`). Two maps (teleporter floors) use
  an open-grid fallback. Ledge jumps ARE modelled: the generator emits
  the overworld ledge-face tiles from the engine's `LedgeTiles` table
  (faces only — flagging the tile you *stand on* was the old disaster),
  and the pathfinder treats a face as a one-way pass-through onto the
  landing tile. That is what makes Route 4 crossable: its ledge drops
  are the only way down to Cerulean, not a shortcut. Other tile-PAIR
  collisions are not modelled; on hardware the stuck detector is the
  safety net.
- NPCs are transient obstacles the static grid cannot see. The executor
  handles them two ways: a tile that refuses entry is excluded and the
  walk replans around it, and a battle or dialog that interrupts a walk
  aborts the walk (never recorded as a blocked tile) so `run()` can
  handle the interruption first. A determined NPC standing in a one-tile
  corridor can still stall a goal until it wanders off.
- The story checklist judges some steps by *arrival* (reaching the
  Mart = parcel picked up; reaching the Lab = parcel delivered), and
  `run_real.py` pairs the two gated ones with scene macros (the
  Pokedex handover, the Brock fight) that verify the real event — see
  above. Any future gated step will need the same treatment: arrival
  alone does not move the game's flags.
- Some warp tiles are *land-only*: pressing into them from an adjacent
  tile never fires them (the Viridian Forest south gate's (4, 0) is
  one; its sibling (5, 0) is the real door). When a walk's target is a
  warp tile that refuses, the executor retries via sibling warps to
  the same destination before reporting stuck.
- After the Boulder Badge the checklist targets Cerulean Gym, and
  the agent has now walked the whole route on hardware: Route 3 →
  Mt. Moon (west door in, B1F → B2F detour → out the B1F east exit
  chamber) → Route 4's ledge drops → Cerulean City → into the gym.
  Two scene beats on that route have macros in `run_real.py`: the
  Super Nerd's fossil choice on B2F (the fossils are objects
  blocking the only door to the exit ladder; take the Helix, he
  takes the Dome) alongside the parcel handover and Brock. What
  stands between the agent and the Cascade Badge now is only a
  Misty fight macro of the Brock kind.
- The Mart's exit mat occasionally needs a second attempt to fire
  (the runner's next round exits cleanly); the retrigger logic is the
  least-hardened part of the hardware path.
- CAVERN-tileset maps decode with the same rule as everywhere else.
  They were marked fully open for a while, on a verdict reached while
  the passable list was still being misread as a blocker list — and
  the phantom-open grids did real damage: the planner drew straight
  lines through Mt. Moon's rock and planned a direct B1F walk to the
  exit chamber, when the true route detours through B2F. The decoded
  grids agree with every cave cell the agent has ever stood on. As a
  safety net for mixed edge cells the sample can still get wrong, a
  refused tile on a CAVERN map is *learned* as a wall for the rest of
  the session (outdoors, a refusal is usually an NPC and stays
  transient), and forgetting is layered/fail-soft so a false wall
  can never seal a map in the model. A warp whose destination map is
  not in the data at all (the Underground Path interior) is treated
  as a dead end.
- Battles are a policy seam. The sim default (`FightPolicy`) still
  mashes A, but the hardware runner uses `SmartFightPolicy`: it reads
  the battle menus' cursor state from RAM and fights with the lead
  mon's last *damaging* move that has PP (for Squirtle: Bubble, then
  Water Gun — the type-advantage move A-mashing never reaches). Two
  live-learned subtleties are baked in: the cursor bytes are stale
  outside live menus, so the policy probes once per menu before
  trusting them; and a status move must never be the target, or a
  PP-dry battle can stall forever (Tail Whip vs a Harden-only
  Metapod). There is still no type chart, no items, no switching —
  Brock fell to levels and a 4×-effective move, not tactics.
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
