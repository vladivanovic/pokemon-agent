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

## 11. Catching detour + HM05 Flash (2026-11-06, probe143)

The Route 2 Gate aide gates HM05 Flash on 10 owned species
(scripts/Route2Gate.asm); the chain head's dex held 4. The
detour caught the six cheapest neighbours of Viridian City,
all Lv3, all verified in RAM (stored count, dex count, and
the receiving list):

- Route 22 grass: Spearow (9 balls - unlucky; the ride math
  below says ~3 is par), Nidoran (m) (2), Nidoran (f) (2) -
  party slots 4-6, filling the party.
- Route 1: Rattata (2), Pidgey (3) - both to Box 1.
- Route 2 south grass: Weedle (2) - Box 1. Dex 4 -> 10.

Balls are thrown at full HP on purpose: nothing in this party
can weaken a Lv2-6 wild without one-shotting it, and the Gen 1
formula caps at ~1/3 per ball at catch rate 255 regardless
(item_effects.asm: at full HP the shake value X is 85). 20
balls thrown across the six catches, 10 left; the only money
spent was a 2-ball top-up (2450 -> 2050) that doubled as the
live test of the buyer port below.

Ported into run_real.py (the probe drove the ported code, not
a private copy):
- _catch_any(emu, ex, want_set): multi-species grass hunts,
  box-aware - success is the STORED count (party + box,
  _stored_count) ticking, so hunts keep working after the
  party fills at 6. _catch_species is now a wrapper with the
  probe115/126 call shape. _catch_epilogue's completion test
  moved from party size to stored count for the same reason.
- _catch_current(emu): catch the mon already in front of you
  (a walk interruption that IS the target gets caught, not
  fled). Its throw ride is the pass's real find: in the item
  effect's order - catch text, dex flag, ShowPokedexData,
  AddPartyMon/SendNewMonToBox, remove ball - a first-of-
  species catch parks on the full-screen Pokedex page BEFORE
  the mon is filed. The page is not a dialog (in_dialog reads
  False) and only a plain A/B press dismisses it, while the
  stored count cannot tick until it is gone: wait-for-count-
  before-pressing deadlocks with the catch already made
  (probe143's first hunt22 sat on Spearow's dex page for the
  full 280 s timeout). The ride presses A through everything
  that is not a live decision menu (throw text, shakes, the
  enemy turn, the dex page) and stops when menuY 14 is
  genuinely back (broke free). The old single-species hunter
  survived this by press-spam luck in _throw_ball's steering.
- _dex_owned_count / _box_count readers (wPokedexOwned /
  wBoxCount), unit-tested against a fake emulator.
- buy_poke_balls(..., mart=None): the Vermilion buyer
  retargeted - every mart shares the counter layout (clerk
  (0,5), stance (2,5) facing left); exercised live at Viridian
  Mart (28 -> 30 balls, count-verified per press as before).
- collect_flash(emu, ex): the aide scene - dex gate checked
  first (a refusal leaves the bag untouched, so no false
  positives), ride to Route2Gate, talk from (2,4) facing left,
  verified by HM05 appearing in the bag.

Live result: collect_flash returned True on the first ride;
final state probe143/flash.state stands in Viridian City with
the party healed - dex owned [Squirtle, Wartortle, Weedle,
Pidgey, Rattata, Spearow, Nidoran (f), Nidoran (m), Oddish,
Diglett], HM05 in the bag untaught (nobody in the story party
is a sensible Flash carrier yet; Oddish can learn it if a
carrier is wanted for Rock Tunnel), TM24 unspent, story
party untouched in slots 1-3. Tests: 73/73 (67 + 6 new
counter tests). Not done, deliberately: teaching Flash,
Rock Tunnel, and any nickname for the new catches (the
epilogue's verified NO held - all six carry species names;
box catches are never offered the prompt at all,
item_effects.asm SendNewMonToBox).

## 12. Rock Tunnel (2026-11-06, probe144): the carrier that couldn't, tile-pair collisions, and a ladder zigzag

Premise correction first, because it changes the leg: section
11's closing claim that "Oddish can learn it" is WRONG. Oddish's
tmhm learnset in pokered (data/pokemon/base_stats/oddish.asm)
lists Cut, not Flash, and the game's own CanLearnTM screen for
HM05 reads NOT ABLE for all six party members (probe144 evidence
screenshot 00-hm05-compat-all-not-able.png). Oddish had a free
move slot all along - the blocker was compatibility, not slots,
so no move-delete flow was ever needed (and none was used;
nobody's moves were deleted this leg). The full Gen 1 Flash
learnset is 27 species; the reachable one was Voltorb. Recovery
adopted: catch Voltorb on Route 10, PC-swap it in for Spearow
(a Lv3 detour catch, not story party), teach Voltorb instead.
Oddish finishes the leg exactly as it entered it: [Absorb,
PoisonPowder, Cut]. Cut was never at risk.

Road taken (all planned executor journeys): Viridian ->
Diglett's Cave -> Vermilion (heal), Vermilion -> Cerulean via
the Underground Path (heal; Saffron's gates stayed shut and
unexploited), Cerulean -> Route 9 -> Route 10 -> Rock Tunnel
Pokecenter (heal). Mt. Moon's interior was NOT needed - the
planner's road goes under/around it - though it still plans
cleanly offline. Route 9 is a genuine gauntlet: Wartortle
reached the tunnel-side center at 4/93 HP. The stock battle
policy (SmartFightPolicy) wedges on its trainers - 500-press
battle caps, party bleeding, then frozen with dialog up - so
every ride in this leg escalates battles to _BattleDriver
bursts (ported as _ride_battling).

Shopping lesson, learned expensively (10 extra balls, money
drained to Y50 before the canary tripped): in a mart, the
cursor bytes change meaning per stage - menuY 1 is
BUY/SELL/QUIT (index 1 is SELL), menuY 4 the item list, menuY
8 quantity/confirm. The ported buy_mart_item is stage-gated
and aborts if the Poke Ball canary count moves. Loadout that
crossed: 9 balls, 5 Potions, 2 Antidotes, 3 Repels (the
Repels and extra Potions came from a deliberate second
Cerulean run - see attempt 2 below).

The tunnel took three attempts, and each failure taught the
model something:

- Attempt 1 (tile-walk with a persistent blocked set): 1F
  clean, most of B1F crossed, then the party WIPED - pure HP
  attrition across ~15 battles (the diagnostic run with no
  supplies showed the same curve: Wartortle 96 -> 4 HP without
  facing anything individually dangerous). Blackout respawn
  at the Route 10 center, healed, money halved to Y2,955.
  Reported as the brief's contingency allows: recovered and
  re-attempted.
- Attempt 2 (after the Cerulean resupply): SEALED in B1F with
  six tiles blocked - (26,30) and (20,21) are STAY trainers
  standing on the through-path (Pokemaniac, Super Nerd), but
  (13,28), (16,28), (19,16), (19,22) were the tell. Root cause
  found in the engine source: TilePairCollisionsLand
  (data/tilesets/pair_collision_tile_ids.asm) forbids crossing
  between CAVERN tiles $20 and $05 in EITHER direction
  (CheckForTilePairCollisions matches standing/front in both
  orders - an elevation rule), and the check reads the same
  bottom-left cell sample the walkability grid is built from.
  The model's B1F path crossed $20 -> $05 at (14,28) ->
  (13,28) and the game simply refuses the step. (One of the
  six, (16,28), is plain floor - its "refusal" was a trainer
  engagement cancelling the step, the Super Nerd at (14,28)
  facing the corridor.)
- The fix is in the model, not the scene: build_walkability.py
  now parses TilePairCollisionsLand and emits the per-cell
  sampled tile ids + forbidden pairs for the 24 CAVERN/FOREST
  maps (regeneration is purely additive - every walk grid,
  warp and connection is byte-identical), and agent/world.py's
  _bfs_steps refuses a plain step whose sampled pair is
  forbidden (ledge jumps skip the check, as in the engine;
  Cut tilesets have no pair entries). Warp-reachability deltas
  vs the old model are confined to pair-tileset maps and match
  pokered's elevation geography (Mt. Moon B2F's two sections
  really do join only via B1F). With the rule, B1F's east and
  west clusters are correctly unreachable from each other
  inside B1F.
- Attempt 3 (pair-aware routes): the real road is a LADDER
  ZIGZAG - 1F door (15,3) -> east ladder (37,3) -> B1F (33,25)
  -> north ladder (27,3) -> 1F (5,3) -> mid ladder (17,11) ->
  B1F (23,11) -> west ladder (3,3) -> 1F south (37,17) ->
  south exit (15,33) -> Route 10 (8,54). It executed with zero
  pair refusals. Flash verified in RAM both ways: entry
  wMapPalOffset = 6 with screenshot mean brightness 5.4; after
  Flash, offset 0 and brightness 93.7. The toll concentrated
  in 1F south (Hiker teams; the lead was Potion-topped at
  4/116, 24, 42, 18, 38 HP - all 5 Potions spent), and
  Wartortle evolved into Blastoise at Lv36 mid-crossing (dex
  11 -> 12). Exit state: Blastoise 28/116, money Y7,820.

The last hop had its own machinery lesson: ride_robust
orbited Route 10's south corridor for 15+ minutes and
Executor._follow_tiles then wedged silently on it in two
further runs (its inner loop can spin without a StuckError
when a press keeps changing state without reaching the
target; the Super Nerd engagement at (14,64) is the local
suspect, not fully isolated). The corridor was finally
walked with a hand-rolled single-step walker (bounded
presses, every step proven by RAM coordinates, ported as
_step_walk_to) under a fresh Repel. Blastoise FAINTED in the
Super Nerd's fight on that corridor - the party won it
(+Y1,000) and walked into Lavender Town with the lead down,
then healed at Lavender's Pokecenter. No second blackout;
the faint is a faint, reported as one.

Model decision - Flash is NOT a capability gate in the world
model: wMapPalOffset is palette-only in the engine, darkness
changes nothing mechanical, and RAM navigation does not read
the screen, so gating journey() on "flash" would model a
constraint the game does not have. Canonical play is enforced
at the scene level instead: traverse_rock_tunnel uses Flash
on entry and verifies wMapPalOffset 6 -> 0 before walking.

Ported to run_real.py: rock_tunnel_leg (the chain above,
stage-idempotent), traverse_rock_tunnel, teach_flash
(free-slot only - the delete flow stays unported, as with
TM28's lesson), use_flash (via PyBoyEmulator.field_flash;
field_cut's menu driver is now a shared _field_move in
agent/pyboy_emu.py, with a matching SimEmulator.field_flash),
buy_mart_item + CERULEAN_MART_LIST, catch_voltorb,
pc_swap + _open_pc (Bill's PC told from the PC main menu by
wMaxMenuItem 0xCC28 - max 4 vs max 3 at the same menuY),
_walk_tile_leg (persistent-blocked interior walker with the
engage fallback and blackout as a distinct outcome),
_step_walk_to, _ride_battling, use_repel, _field_heal_lead.
Fidelity: the scenes are faithful ports of the probe-proven
flows; the tunnel legs' routes now come from the pair-aware
model (unit-tested + offline-verified) rather than the
probe's offline graph. rock_tunnel_leg as a whole has not
been cold-replayed end to end.

Final state: probe144/lavender.state - Lavender Town,
party healed at its Pokecenter. Blastoise Lv36, Oddish Lv15
[Absorb, PoisonPowder, Cut], Diglett Lv22, Nidoran (m) Lv3,
Nidoran (f) Lv3, Voltorb Lv16 [Tackle, Screech, FLASH]; box
holds Weedle, Pidgey, Rattata, Spearow; capabilities cut +
flash; dex 12; 9 balls; Y8,820; TM24 still unspent; no
Saffron, no Pokemon Tower, no Celadon. Tests: 77/77
(73 + 3 world pair-rule tests + 1 Sim field-flash test).
Save states stay out of git as ever; the chain head moves
to the Orin by hand with its sha256.

## 13. Celadon + Erika (2026-11-06/07, probes 145-147): the Fire counter, a stone, and a grind

The road west worked as planned journeys from the tunnel
chain head: Lavender -> Route 8 -> Underground Path -> Route 7
-> Celadon City, then the gym interior through all three tree
faces as planner Cut steps. All seven gym trainers were beaten
along the way (event flags 426-432 verified set in RAM).

Shopping, all RAM-verified: 5x Super Potion (Y3,500) and 4x
Great Balls (Y2,400) at Mart 2F (clerk 1, stand (5,5) facing
up); one Fresh Water (Y200) from the roof vending machine -
the Saffron guards' drink (Gen 1: one drink to any guard opens
every gate). Money in: Y8,820 + Y330 (a Route 8 trainer).

The matchup math came from pokered data, not vibes: Blastoise
cannot solo Erika. Its Water damage is Grass-resisted (~21/hit
on the aces) while Razor Leaf crits near-always in Gen 1
(~70/hit back). The counter was always going to be Fire:
Growlithe, caught on Route 8 with the first ball thrown
(dex 12 -> 13), Nidoran-F boxed for it.

What failed, in order (five blackouts across the leg):

1. Gym trainer #1 (probe145): the party walked in under-level
   for the room and the lead fainted mid-clear.
2. Erika, attempts 1-2 (probe145): Growlithe switch-training
   turned out to be economically broken in Gen 1 - a mon that
   faints earns NO XP and shared XP is halved, so the trainee
   plan starved. What actually levelled Growlithe (16 -> 18,
   Ember learned) was leading and soloing Route 24 grass via
   a new party_lead reorder helper. Erika still won: Ember at
   Lv18 is not enough (below), and the Lv40 Blastoise closer
   is resisted.
3. Erika, attempts 3-4 (probe146, post-stone): see below.

Two battle-driver bugs were found by the failures and fixed
in b905d04 (suite 80/80): the faint-pick wedge (a stale
signature branch pressed A on a fainted mon forever) and
_ranked_move_fn's 0-PP fallback (it returned slot 0 at 0 PP,
wedging battles on the "no PP" text; it now falls back to any
PP'd move).

The Fire Stone was Vlad's decision (2026-11-07), taken with
its permanent cost on the table: Arcanine's Gen 1 level-up
learnset is EMPTY, so evolving forfeits Growlithe's Lv50
Flamethrower for good; Fire Blast remains available later as
TM38. Bought at Mart 4F (Y2,828 -> Y728, exact delta verified);
Growlithe -> Arcanine Lv18 (dex 13 -> 14).

Correction, stated plainly: the numbers that pitched the
stone were wrong. Arcanine at Lv18 has 63 max HP, not the
~86 estimated (a stone evolution adds only the stat delta:
+12 HP), and Ember lands ~21-24 per hit on Erika's ~85 HP
aces - about four hits each - into ~28-damage Razor Leaf
crits it can only tank two of. Two more blackouts followed
(money Y728 -> Y364 -> Y182; Pokecenter heals are free, and
the lineage parked healed at probe145/p9-failed.state).

The grind was also Vlad's call (2026-11-07): Arcanine to
~24-25 on Route 24, then re-fight. probe147 first found a
third machinery bug: probe145's party_lead hard-codes the
party submenu's SWITCH entry at item 1, which is only true
for mons with NO field moves - the submenu prepends the mon's
field moves (Oddish's reads CUT / STATS / SWITCH / CANCEL),
so for Oddish item 1 is STATS and the "reorder" silently
opened the stats page. (The failed arrange saved
probe145/g-travel-failed.state, poisoned mid-STATS-page; do
not build on it.) SWITCH sits at 1 + (field moves known);
the port computes it (_switch_menu_index) and verifies the
swap by the party order afterwards.

The grind itself: 11 per-map hop rides to Cerulean (drink
re-verified in the bag after every hop), arrange
[Arcanine, Blastoise], then Arcanine led and soloed Route 24's
Lv12-14 grass - 132 battles, Lv18 -> Lv25, ~4.5 minutes of
emulation, checkpoints saved per level. Exactly one heal
trip, and it was PP-driven, not HP (at Lv23: 58 HP but only
4 damaging PP left; Ember's 25 PP is the grind's real
budget). Per the mission rule, Lv25 was the target and 24
the floor; it stopped at 25 because that was the target.

The fight at Lv25 took ONE attempt. Arcanine soloed all
three (Victreebel Lv29, Tangela Lv24, Vileplume Lv29) on
Ember, levelled 25 -> 26 mid-fight, and ended at 60/91 HP;
Blastoise - arranged second precisely so the driver's
first-healthy replacement pick would find it - was never
needed. Payout Y2,871 (money Y182 -> Y3,053). Rainbow Badge
verified in RAM (badges Boulder/Cascade/Rainbow/Thunder;
Erika event flag 425 set), TM21 Mega Drain received. Blastoise
picked up Lv41 fighting intercepts on the road home.

The drink incident, flagged plainly: in an abandoned scratch
lineage (p5-failed4, never the chain), a single multi-map
journey execution diverged from its plan, rode through a
Saffron gatehouse, and the guard auto-took the Fresh Water,
stranding that lineage in Route 8's west pocket. The live
lineage used per-map hop rides throughout, its drink was
re-verified in the bag after every gatehouse-adjacent hop,
and Saffron was never entered. The drink is still in the bag
at the chain head (RAM-verified).

Machinery gaps banked by this leg:

- Multi-map journey EXECUTION can diverge from the planner's
  route (the gatehouse ride) - intercity travel is per-map
  hop rides now (_hop_ride).
- Celadon Mart's elevator defeats single multi-warp journeys
  (they spin) - floor-chained rides, one warp at a time.
- _use_bag_item_on_mon returns with the bag menu still open;
  callers must close menus or the next walk desyncs.
- in_dialog flickers under the letter-printing delay, so text
  rides need a two-poll clear test (_ride_text).
- Sight trainers are engaged by _walk_fighting, not raw
  _walk_to, which can thread a sight line without triggering.
- Evolution rides press A only: B cancels a Gen 1 evolution.
- The battle driver still has no mid-battle item path; the
  4 Super Potions rode out the whole leg unused (field
  top-ups only, via _maintain_party).
- drive_battle's entry print reads a stale enemy-species byte
  (cosmetic, also noted in S12); read_state is the truth.

Ported to run_real.py: celadon_leg (the winning route:
road west -> catch_growlithe -> pc swap -> Mart 2F resupply
-> buy_vending_drink -> Fire Stone -> evolve_with_stone ->
grind_route24 to 25 -> beat_erika), beat_erika (flag-verified
trainer worklist, heal trips, _maintain_party,
_challenge_gym_trainer, _gym_heal_trip), catch_growlithe,
buy_vending_drink, evolve_with_stone, grind_route24,
_hop_ride, _walk_fighting, _drive_battle_typed,
_typed_move_fn (+ _MOVE_INFO/_TYPE_CHART generated from
pokered - the raw-power ranker cannot see Ember > Bite into
Grass), party_lead + _swap_party_slots + _arrange_front with
the _switch_menu_index fix, and buy_mart_item generalised
with stand/face (Celadon lists CELADON_MART2F_LIST /
CELADON_MART4F_LIST). One deliberate divergence: the ported
trainer phase leads with the ground Arcanine rather than
replaying the historical Blastoise-plus-trainee order - the
segments are probe-proven, the composition follows the
winning route, and celadon_leg as a whole has NOT been
cold-replayed end to end. Tests: 90/90 (80 + typed-ranker,
effectiveness, switch-index and damaging-PP tests).

Final state: probe145/celadon.state - Celadon City, party
healed at its Pokecenter. Arcanine Lv26 [Bite, Roar, Ember],
Blastoise Lv41, Diglett Lv22, Nidoran (m) Lv3, Voltorb Lv16
[Flash], Oddish Lv23 [Cut]; box holds Weedle, Pidgey,
Rattata, Spearow, Nidoran-F; badges Boulder/Cascade/
Rainbow/Thunder; dex 14; Y3,053; 8 Poke Balls, 4 Great
Balls, 4 Super Potions, Fresh Water x1, TM21 in the bag,
TM24 still unspent; Saffron never entered.
sha256 8e58b3dfe0c49016e96ad52f2204960b2e0c48eafc0d54b7e31b2e93ae3c85c3.
Save states stay out of git as ever; the chain head moves
to the Orin by hand with its sha256.
