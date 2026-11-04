# Supervisor training pipeline

The deterministic stack plays Pokemon Red with no model at all. The
only place a model earns its keep is *supervision*: proposing the next
typed goal (and, later, recovering from stuck states). This pipeline
trains exactly that role — and it starts by **measuring** the stock
model, because the run logs are the training set.

```
LLMPlanner (stock model, validator in front)
    │  every proposal + verdict + outcome
    ▼
~/workspace/pokemon-play/supervisor_log.jsonl      (agent/supervisor_log.py)
    │  scripts/build_supervisor_dataset.py
    ▼
dataset/supervisor/{train,eval}.jsonl              (gitignored)
    │  scripts/train_supervisor_lora.py  (on the Orin, hermes_venv)
    ▼
LoRA adapter ──► serve behind LLMPlanner ──► validator still decides
```

## 1. Log (automatic when the LLM planner is in use)

`LLMPlanner` appends every round to the supervisor log: the exact
context the model saw, its proposal as typed JSON, the validator's
verdict and problems, whether the planner used the proposal or fell
back to the checklist, and — linked by `record_id`, via the executor's
outcome hook — whether the returned goal executed and succeeded.

Default path: `~/workspace/pokemon-play/supervisor_log.jsonl`.
Configure with `LLMPlanner(..., log_path=...)`, disable with
`log_path=None`. ChecklistPlanner runs never log.

## 2. Build the dataset

```bash
python3 scripts/build_supervisor_dataset.py --count 800 --seed 42
# -> dataset/supervisor/train.jsonl + eval.jsonl (95/5 split)
```

Two sources, one format — chat messages where the user turn is the
exact context dict `LLMPlanner` sends at inference time and the
assistant turn is `{"target_map_name": ..., "reason": ...}`:

- **Scenario synthesis (the oracle).** For each story step, states are
  synthesized along the route to it using the ROM world model —
  progression prefixes (badges, visits, HM moves a real run would
  have), plausible start maps and party compositions. The
  ChecklistPlanner proposes the goal; `goals.validate_goal` must
  accept it against its own state or the row is dropped. This is the
  same trick as the Metasploit tool-calling set: trajectories from a
  verified oracle, not scraped human play.
- **The supervisor log.** Proposals the validator accepted *and* the
  executor completed (`decision == "llm"`, outcome executed +
  success). Real distribution, verified labels. Rejected and failed
  proposals stay in the log as analysis material; they are never
  trained on as positives.

Rows are deduped on the full (user, assistant) payload, shuffled with
the seed, and split 95/5. The builder prints pool size, per-source
counts, and target-map coverage. Useful flags: `--variants` (state
diversity per step/start), `--log PATH` / `--no-log`, `--out DIR`,
`--eval-frac`.

## 3. Train (on the Orin)

Target stack is Hermes' `hermes_venv` (transformers 5.17 / TRL 1.15 /
PEFT 0.21.2 / torch 2.14.1+cu132). Base model `Qwen/Qwen3.5-4B` —
the same 4B class as the Metasploit run, so it can share serving.

```bash
# one-time: fetch the base weights (~8.8 GB) to models/Qwen3.5-4B
python3 scripts/train_supervisor_lora.py --download

# train: LoRA r=32, bf16, lr 1e-4, 2 epochs, max_len 4096,
# checkpoints every epoch under training_output/qwen_supervisor/
python3 scripts/train_supervisor_lora.py \
    --data-dir dataset/supervisor \
    --out training_output/qwen_supervisor
```

Supervisor examples are compact (a state summary in, one goal JSON
out), so 4096 tokens is generous and **no length ladder is needed** —
the ladder in the Metasploit run exists because those trajectories
*are* the skill; here they aren't. Training is assistant-only by
default (`--no-completion-only` opts out). Serving note from the
Metasploit pre-flight: the hybrid `qwen3_5` arch needs vLLM ≥ 0.32
(or the Mjolnir image) — the local vLLM 0.28 cannot serve it.

## 4. Evaluate before trusting it

Two gates, neither optional:

- **Held-out split.** `eval.jsonl` is never trained on; the trainer
  reports eval loss per epoch. Loss is a sanity check, not the gate.
- **Stuck-state battery.** Scripted scenarios (SimEmulator save
  states: mid-route, post-badge, low-HP, detour-eligible) where the
  fine-tuned supervisor proposes goals behind the *same* validator as
  production. Measure: valid-goal rate, and — the one that matters —
  did executing the goal actually advance the checklist step. Compare
  against the stock base model on identical states; the fine-tune
  only replaces the stock model if it wins there. This mirrors the
  Metasploit 4-test battery: a small functional gate aimed at the
  known failure modes, not a vanity metric.

A fine-tune that can't beat the stock model behind the validator
doesn't ship. The validator stays in front either way — training
improves proposals, it never gets a vote on what executes.

## 5. Orin run notes (2026-10-28): blockers and fixes

First real train on the Orin. The pipeline now runs end to end here;
what had to change, and why:

- **TRL 1.15 dropped `warmup_ratio`.** `SFTConfig` rejects it; replaced
  with `warmup_steps=24` (0.03 × the 800-example set). Committed.
- **fp32 AdamW states don't fit.** 4B base + AdamW keeps ~32 GB of
  optimizer state alone; the Orin OOM-killed the run twice (once at
  epoch 2, once at start). `optim="paged_adamw_8bit"` + bitsandbytes
  fits in ~16 GB total.
- **Broken Pillow poisoning imports.** The Hermes gateway exports
  `PYTHONPATH=~/.hermes/installs/<env>/venv/lib/python3.14/site-packages`
  ahead of everything, and that env's Pillow has no working `_imaging`
  for python 3.12. Any python that imports PIL here must run with
  `env -u PYTHONPATH`. (Also fixed by force-installing Pillow 12.3 into
  that env; do both.)
- **The Hermes worker cgroup OOM-kills background jobs at ~5 GB.** Two
  training runs died with exit 137 while the system had 40+ GB free.
  Long-running or memory-heavy jobs must launch OUTSIDE the worker
  cgroup: `systemd-run --user --unit=<name> ...` (verified: training
  peaked ~16 GB, no kills).
- **Base weights were already local.** The Metasploit project trained
  the same Qwen3.5-4B; use `--model-dir /home/vlad/models/Qwen3.5-4B`
  instead of re-downloading (~8.8 GB, both shards verified).
- **Epoch count.** The oracle task is deterministic; after one epoch
  eval loss was ~4e-06 with 100% token accuracy. Train with
  `--epochs 1` — the second epoch adds nothing but OOM risk.

Exact working launch (from the repo root):

    systemd-run --user --unit=pokemon-lora-train \
      --working-directory=$PWD \
      env -u PYTHONPATH /home/vlad/hermes_venv/bin/python3 \
      scripts/train_supervisor_lora.py \
      --model-dir /home/vlad/models/Qwen3.5-4B \
      --data-dir dataset/supervisor \
      --out training_output/qwen_supervisor --epochs 1

Dataset was built with: `scripts/build_supervisor_dataset.py --count
800 --seed 42` (scenario pool 1560 → 800 examples, 12 distinct target
maps, no supervisor-log rows yet). Output adapter: 260 MB r=32 under
`training_output/qwen_supervisor/final/` (gitignored - weights never
in git). Serving caveat unchanged: the hybrid qwen3_5 arch needs
vLLM >= 0.32, but transformers-direct works on the Orin and is what
the battery uses.

## 6. Stuck-state battery v2 results (2026-11-04): KEEP STOCK

Battery v2 (`scripts/stuck_state_battery.py`, 40 scenarios: 12
midroute / 8 postbadge / 8 lowhp / 12 control; both contenders through
`goals.validate_goal`; full rounds in the supervisor log at
`~/workspace/pokemon-play/supervisor_log.jsonl`, 160 records). Harness
anomalies from v1 (greedy brace parse, thinking preamble) fixed before
this run - the scores below are model behavior.

| family    | metric   | stock | adapter |
|-----------|----------|-------|---------|
| control   | valid    | 100%  | 100%    |
| control   | unstick  | 100%  | 100%    |
| control   | advance  | 58%   | 100%    |
| midroute  | valid    | 42%   | 0%      |
| midroute  | unstick  | 42%   | 0%      |
| postbadge | valid    | 50%   | 0%      |
| postbadge | unstick  | 50%   | 0%      |
| lowhp     | valid    | 50%   | 0%      |
| lowhp     | unstick  | 50%   | 0%      |

**Verdict: NO SHIP** (the rule: the adapter ships only if it beats
stock on unstick without losing validity or controls - it loses
unstick outright).

Failure classes, dominant first:
1. **Adapter: 28/28 stuck scenarios suppressed.** It always proposes
   the story step's goal - the exact goal that already failed twice -
   because the training set contains only failure-free oracle
   trajectories. Failure events in the context are out of
   distribution; the adapter ignores them and recites the checklist.
   Perfect checklist follower (control advance 100%), zero recovery.
2. **Stock: 15/28 suppressed** the same way, but 13/28 proposed a
   different valid goal - its base-model variety is what unsticks.
3. Controls: the stock returned a valid-but-wrong target 42% of the
   time (advance 58%); the adapter was exactly right 100%.

Path forward (waiting for go): the next training round must include
recovery rows - failure events in the context paired with a DIFFERENT
validated target. Sources: this battery's stock-recovery positives
(13, already in the log) plus a recovery oracle for the synthesizer.
Stock stays the default; logging stays on.

## 7. Round 2 (2026-11-04): recovery rows trained, battery 3 = SHIP

Round-2 dataset (`dataset/supervisor_r2/`, builder flags `--count 800
--seed 42 --recovery 240 --battery-train 12 --battery-holdout 4
--out dataset/supervisor_r2`): 800 failure-free originals + 240
recovery rows (midroute/postbadge/lowhp, failure events in context,
priority oracle: heal → Zapdos detour → retreat; every row validated
and different from the failed goal) + 12 real battery-stock recovery
rows. Mechanical audit: **0 of 240 recovery rows target the failed
goal**. 4 real rows held out (`battery_holdout.jsonl`) to audit
synthetic against real; trained **from base**, 1 epoch - continuing
from the v1 adapter would have started from its checklist collapse.

Seed arithmetic (corrected - the earlier draft said 13): the
supervisor log holds **31 battery-stock successes** (25 from battery
v2 - 13 stuck-family recoveries + 12 controls - and 6 from the first
battery run). Round 2 trained on 12 of that 31-row pool and held out
4; the remaining 15 rows stay in the log, unused.

Battery 3 (same 40 scenarios, same harness/scoring; stock and v1 rows
from battery v2, adapter v2 run fresh):

| family    | metric   | stock | adapter v1 | adapter v2 |
|-----------|----------|-------|------------|------------|
| midroute  | valid/unstick | 42% | 0%  | **83%** |
| postbadge | valid/unstick | 50% | 0%  | **100%** |
| lowhp     | valid/unstick | 50% | 0%  | **75%** |
| control   | valid/unstick | 100% | 100% | **100%** |
| control   | advance  | 58%   | 100% | **100%** |

**Verdict: SHIP.** v2 beats stock on unstick in every stuck family
(86% avg vs 47%) without losing validity, and controls hold at v1's
level (100%, advance included). Wiring v2 as the default supervisor is
a separate next go. Raw generations are not in git; the battery v2
rows are in the supervisor log at
`~/workspace/pokemon-play/supervisor_log.jsonl` (round results in
`training_output/*/adapter_v2_battery_results.json`, gitignored).

## 8. Live run (2026-11-05): PyBoy real ROM from probe138 chain head

Setup: chain head `probe138/after-surge.state` transferred from Vlad
(sha256 2f0d8da6a01fe46cf7281a3d93b1a94090faed217f62361a441fd7242a2b50
44 verified, 178,108 bytes), working copy `after-surge-work.state`
booted on PyBoy real ROM (Red, SGB-enhanced). Start verified from RAM:
VermilionGym (5,2), badges Boulder/Cascade/Thunder, party levels
31/15/21 (Wartortle/Oddish-Cut/Diglett as specified). Supervisor =
wired default (adapter v2, inline transformers-direct), logging on.
Budget: 25 goals / 45 min wall.

Run: 26 supervisor invocations, **all served inline:adapter - stock and
checklist never took over at the serving level** (the checklist did
step in 19 times inside LLMPlanner after validator rejections, the
designed inner chain). Outcomes: 1 executed goal (VermilionCity - the
executor walked out of the gym to the city, the run's only progress),
the rest failed against two executor limits: "no journey (portal-level
route) to target" - **the Diglett's Cave LAST_MAP exit-chaining gap,
live exactly as predicted** - on the supervisor's sensible
ViridianCity proposals, and "already at destination" on a same-map
proposal. Final RAM: VermilionCity (12,20), badges unchanged, party
unchanged (no battles). End: goal budget consumed; the run stopped at
the executor's movement limits, not the supervisor's - the proposals
were valid, the world model could not route them. The Diglett's Cave
chaining fix is the follow-up work this run points at.

## 9. LAST_MAP exit chaining fixed (2026-11-05, probe141)

Root cause: `journey()` propagated wLastMap with the engine's
generic rule only (WarpFound sets it when warping from an outside
map). But the Diglett's Cave lobbies - and the Underground Path
lobbies - assign wLastMap in their **map scripts** on entry:
`scripts/DiglettsCaveRoute2.asm` -> ROUTE_2,
`scripts/DiglettsCaveRoute11.asm` -> ROUTE_11, same pattern in
UndergroundPathRoute5/6/7/7Copy/8; Route22Gate assigns it
position-dependently (y<4 -> ROUTE_23, else ROUTE_22). Under the
generic rule the far lobby's LAST_MAP exit resolved back to the
entry-side map, Route 2 was never generated as a journey state,
and every through-cave journey returned None - the §8 failure.

Fix (`agent/world.py`): a `LAST_MAP_SCRIPT_MAPS` table mirroring
those scripts; journey seeds and updates wLastMap by script
assignment when the entered map has one (the tile-aware rule for
Route22Gate), the generic rule otherwise. `route()` untouched.
Tests 59/59 (4 new: through-cave journeys both directions, cold
start inside the cave, a lobby exiting to its own outside map,
ordinary building-exit regression).

Live (probe141, real ROM, RAM-verified): from a copy of the chain
head (original untouched, hash verified), exit gym, cut the gym
tree from the yard side (stand (15,19) facing north - the Surge
macro approaches from the city side), heal, then ONE continuous
executor journey Vermilion -> Route 11 -> Diglett's Cave ->
Route 2: the 6 planned steps above, 3 wild Diglett battles fought
en route via the run loop's battle/dialog recovery, arrival
Route 2 (12,10), party unharmed.

Still open (separate follow-up): Vermilion -> ViridianCity still
returns None. Route 2's cave-mouth pocket (y 2-21) is sealed at
tile level by the Cut-tree tile class, which tile-level journeys
don't model - the same class as the Vermilion gym tree. The
supervisor's ViridianCity proposals would route once the journey
can pass the far side of Route 2 (or once a Flash leg targets
Route 2 itself).

## 10. Cut-tree traversal (2026-11-05): trees are journey steps, not walls

The §9 remainder is closed. Tile-level journeys now model Cut
trees as capability-gated ACTION steps.

Identification (data, not hardcodes): the engine checks ONE tile
id per tileset when Cut is used (engine/overworld/cut.asm UsedCut:
$3D on OVERWORLD, $50 on GYM; $52 grass is walkable already and
deliberately excluded). build_walkability.py emits `cut_trees`
per map - the cell whose bottom-left quadrant sample (the value
wTileInFrontOfPlayer takes when facing that cell) equals the
tileset's cut id. 35 face cells across 16 maps, matching the
game's known trees (Route 2 x6, Fuchsia x4, Celadon Gym x3 - the
GYM rule is real: Erika's gym grows them indoors).

Model (agent/world.py): a tree is a 2x2 BLOCK (the engine's
CutTreeBlockSwaps replaces it whole); GameMap.cut_blocks maps
every block cell to its face. In the tile BFS a block's
non-walkable cells are solid while the tree stands (a block cell
whose own tile is passable ground stays ground - the engine's
collision is per-tile); with the "cut" capability the block is
entered only through its face, and from the face the rest of the
block is passable. journey() splits a ("cut", map, face) step out
of any walk path that enters a tree, stance first, in path order -
a journey may fell several trees. Without the capability,
journey_missing_capabilities / missing_capabilities (tile-aware
form) name "cut", and validate_goal rejects a Cut-sealed goal
with that diagnosis instead of waving it through to a stuck.

Execution (agent/executor.py, agent/pyboy_emu.py,
agent/emu.py): the Executor owns _cut_done - faces cut this map
visit, pruned the moment another map is current (trees respawn
on map reload, engine semantics; even a gatehouse hop resets the
previous map's record). A ("cut", ...) step - and, belt-and-
braces, any walk path stepping onto a standing face - runs
_cut_tree: face the tree, then the backend's field_cut verb with
the cutter and submenu index derived from RAM party data (first
mon knowing Cut; its field-move rank in its own slot order,
pokered GetMonFieldMoves). PyBoyEmulator.field_cut is the
generalised gym-tree macro (START -> POKEMON -> mon -> CUT,
menus steered by RAM cursor bytes); SimEmulator.field_cut models
the same contract (facing + face cell + mon knows Cut) so the
whole loop is testable without a ROM.

Tests: 67/67 (8 new: data identification + block expansion, tile
seal/open, Vermilion -> Viridian journey with ordered cut steps,
capability diagnosis, validator reject/accept, sim end-to-end
executor run felling both Route 2 trees, no-Cut blocked run).

Live (probe142, real ROM, RAM-verified): from probe141/
route2.state, ONE executor ride Route 2 -> Viridian City on the
planned 8 steps - cut (15,22) out of the cave pocket, gatehouse,
cut (12,68) at the south end, cross - arriving ViridianCity
(17,4), party unharmed, both cuts performed by the executor from
its journey steps (no hand-driven cutting). The supervisor's §8
ViridianCity proposals are now executable end to end.

Read-only aside (same probe, RAM wPokedexOwned): 4 species owned
- Squirtle, Wartortle, Oddish, Diglett. The Route 2 aide's HM05
Flash needs >= 10, so a catching detour is a decision for Vlad
before any Rock Tunnel leg.
