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

## 14. Rocket Hideout (2026-11-07, probe148): learned-graph navigation, the elevator, and Giovanni

The Game Corner poster route needs no coins: beat the Rocket
guarding the poster (Raticate/Zubat Lv20, +Y600), ride his
scripted exit, then examine the poster — its text_asm sets
EVENT_FOUND_ROCKET_HIDEOUT (441) only after the "Hey! A switch
behind the poster!?" text is advanced through, and swaps in
the staircase block. Ported as `rocket_hideout` /
`beat_giovanni_hq` in run_real.py (faithful ports of the
probe flow, not cold-replayed end to end).

Navigation is the section's real content. The shipped indoor
grids are not reliable at cell granularity in the FACILITY
tileset: a block's four quadrants mix floor and machine tiles,
so a per-cell sample lies both ways (B2F's strip: (25,8) and
(26,8) are floor, (24,8) is a wall, whatever the grid says),
B4F's grid is the all-open coherence fallback, and B2F/B3F
arrow tiles relocate the player by script (per-map
map_coord_movement tables: RLE joypad lists played in reverse
db order with input ignored). The hideout is therefore
navigated empirically by `explore_nav`: one press, settle
until the spin runs out, read the landing cell from RAM, and
record the edge (cell, dir) -> landing | wall | warp in a
per-map graph. BFS plans over observed edges only; frontier
steps expand the graph toward the target, using the model
grid only as a prior (model-wall +6, arrow cell +10, never
speculate onto a warp cell). Arrow rides are deterministic,
so a recorded ride replays exactly. Recorded walls may be
re-probed once per call at +80: doors change state, and B4F's
office door was first recorded shut, then walked through
after its guards fell — the door blocks swap via the map
script's load callback, which has run by the time the second
guard's battle ends. One honest failure: the B2F Nugget
(16,8) was attempted twice and abandoned — stepping off the
strip drops into the south-west arrow pocket, whose learned
component is a dead end for the stances around the ball;
optional loot, not worth the maze. A general spinner model
for the world model would need the map_coord_movement tables
compiled in as forced-move edges (entry cell + simulated
joypad RLE -> landing cell, chainable), plus per-quadrant
(tile-level, not block-level) passability for FACILITY
interiors — the learned graph is the empirical stopgap.

The elevator protocol, cracked in one attempt from a saved
cab state: panel bg at (1,1), stance (1,2) facing up (the
(2,1) stance is itself a door warp); the floor menu is
steered by wCurrentMenuItem (B1F/B2F/B4F = 0/1/2); choosing
a floor rewrites the door warps' destinations, and stepping
north out lands on the matching floor's door cell (B4F:
(25,15)). It refuses without the LIFT KEY in the bag.

Floor notes. B1F: no arrows; the NE stairs hop (23,2) avoids
all five Rockets — only Rocket5 (28,18) was fought, on the
way out, because he guards the elevator room's door. B3F's
two Rockets both fell to arrow-ride sight lines during the
crossing. B4F's west side: Rocket3 (11,2) drops the LIFT KEY
ball at (10,2) when talked to *after* his battle (event
1702); TM02 Razor Wind (9,4) and an HP Up (10,12) were the
cheap pickups. The office guards Rocket1 (23,12) / Rocket2
(26,12) have sight range 0 — they fight only when talked to.
Giovanni (25,3) stands behind a 2x2 desk: cells (24-25,4-5)
are wall tiles (block $47), and FACILITY's only counter tile
($12) is not among them, so there is no talk-over — he is
talked to from the north room at (26,3), reached around the
desk's west side. His Kangaskhan fainted Blastoise, which
had entered at 81/139 after a no-heal hideout run; Arcanine
closed the fight (event 1703). The SILPH SCOPE ball at
(25,2) was taken from his own cell (25,3) once he left.

Two tooling bugs found and fixed on the way. (1)
`_swap_party_slots` steered the START menu to item 3 (the
trainer card) — the port of probe147's helper drifted from
its source, which uses item 1 (POKEMON); every live swap
failed until `START_MENU_POKEMON = 1` was pinned with a
test. (2) Event-id counting from event_constants.asm is a
trap twice over: the file uses const_def/const_next (not
const_event), and a bare `const_skip` still advances the
counter. The hideout ids used here are anchored on
EVENT_67F = 1663 by construction and on wEventFlags diffs
between probe saves (B2F trainer = 1665; B3F = 1681/1682;
B4F = 1698/1699/1700, door 1701, key drop 1702, Giovanni
1703).

Chain: probe148/scope.state — Celadon City outside the PC,
party healed (Blastoise Lv43, Arcanine Lv27, Diglett Lv22,
NidoranM Lv3, Voltorb Lv16 Flash, Oddish Lv23 Cut), money
Y12,194, bag: 8 Poke Balls, 4 Great Balls, 4 Super Potions,
Fresh Water x1 (unspent, Saffron untouched), LIFT KEY,
SILPH SCOPE, TM02 + TM21 + TM24 (all unspent), HP Up.
Tests 97/97.
sha256 df1ebb8ec263e707094123273d5dee464ce2f5c46e0863d31f62cad202d125c6.
Save states stay out of git as ever; the chain head moves
to the Orin by hand with its sha256.

## 15. Pokemon Tower (2026-11-07, probe149): fleeing wilds, a miscounted event block, and the Poké Flute

Travel first, because it is the discipline that keeps paying:
Celadon -> Lavender went as six per-map hop rides through the
Underground Path (Route 7 -> UP Route 7 -> WE path -> UP
Route 8 -> Route 8 -> Lavender), the Fresh Water re-verified
in the bag after every hop, Saffron never entered, money
untouched (no battles at all on the road).

The climb's new machinery is `flee_wild`, threaded through
`_drive_battle_typed` / `_walk_fighting` / `explore_nav` /
`talk_fight_event` with the default OFF everywhere. Wild
battles are told from trainer battles by wIsInBattle
(0xD057: 1 wild, 2 trainer); a wild battle is fled with
`_run_away` (two attempts, then fight). In the probe every
wild battle was escaped - 72 fled on 5F alone, where the
learned-graph approaches step a long time in Gastly grass -
and none had to be fought. Gen 1 type trivia the typed
ranker needed no help with, but worth writing down: Bite /
Scratch / Cut / the 100-power Normal move are all NORMAL
and bounce off Ghost; Water, Ember and Dig all hit
Gastly's Ghost/Poison fine.

Event ids bit us again, differently. The Tower block was
first computed 2 too high (rival at 241) because
event_constants.asm line 67 is `const_next $F0 - 2` - an
EXPRESSION, which a plain-number parser silently skips.
The beaten rival exposed it: his save's wEventFlags diff
flips exactly one bit, 239. With expressions evaluated
(and const_def's optional increment honoured), the whole
file reproduces every anchor (441, 1663 EVENT_67F, 1703
Giovanni) AND the empirical 239. Tower ids: rival 239;
3F 241/242/243; 4F 249/250/251; 5F 258-261; purified zone
263; 6F 265/266/267; Marowak 271; 7F Rockets 273/274/275;
rescued-2 279; flute 296; rescued 1231 (the Fuji pair is
split across the file - _2 sits in the Tower block, the
main flag 900 lines later). NUM_EVENTS is $A00, so the
flag array is 320 bytes; bits past it in a wide diff are
toggleable-object flags (the rival hiding, etc.), not
events.

The fights, briefly. Rival (trigger (14,6)/(15,5); Rival2
#4: Pidgeotto 25 / Growlithe 23 / Exeggcute 22 / Kadabra
20 / Wartortle 25) paid Y1,625 and cut Blastoise to 87/139.
The Channelers are the leg's real cost - thirteen of them,
all single/double Gastly 22-24 or Haunter 22-23 - and
their Night Shade / status attrition ground Blastoise from
87 to 16/142 across 3F/4F (levelling it to 44 on the way).
Sight lines do the scheduling: on 5F one talk-fight's
approach set two further trainer flags, and on 6F all
three flags fell in a single navigation sweep. The 5F
purified zone (cells (10-11, 8-9), heal on entry, resets
on leaving, reusable) is therefore used FIRST on arrival,
again before the 6F stairs - the probe's original
trainers-first order was revised mid-leg for exactly this.
On 7F the three Rockets (Zubat/Zubat/Golbat 25; Koffing/
Drowzee 26; Zubat/Rattata/Raticate/Zubat 23) dropped
Blastoise to 15/148 but never threatened a blackout.

Two traps documented for the next legs. (1) The ghost
MAROWAK's scripted battle reads as a WILD battle in RAM
(wIsInBattle == 1) - a flee-aware driver would run from
it, and running counts as not-defeated (the script shoves
the player right and re-arms the (10,16) trigger).
`beat_ghost_marowak` drives it with flee_wild off and
compares ball counts before/after (thrown balls are
dodged by script; none were thrown). Lv30 Marowak took
one Water Gun. (2) 6F's only southern corridor in the
world model runs through (6,8) - the Rare Candy ball's
cell. Item balls are objects, not tiles, so the grid
reads the cell open and `_walk_to` wedges on the ball;
taking the candy (bag-verified) clears the corridor. The
model has no notion of object-occupied cells; that is the
general gap.

Fuji: talking to him on 7F sets 1231 + 279 and his script
warps the player to MrFujisHouse, where Fuji (3,1) hands
over the POKE FLUTE (bag count 1, event 296). Exit south,
heal in Lavender.

Chain: probe149/flute.state - Lavender Town, party healed
(Blastoise Lv46, Arcanine Lv27, Diglett Lv22, NidoranM
Lv3, Voltorb Lv16 Flash, Oddish Lv23 Cut), money Y25,009,
bag: POKE FLUTE, SILPH SCOPE, LIFT KEY, RARE CANDY, HP Up,
TM02 + TM21 + TM24 (all unspent), 8 Poke Balls, 4 Great
Balls, 4 Super Potions, Fresh Water x1 (unspent). Dex 14.
Ports: `pokemon_tower` + `beat_tower_rival` +
`beat_ghost_marowak` + `rescue_mr_fuji` - and for once the
port is better than faithful: it was cold-replayed
end-to-end from probe148/scope.state in 345 s and reached
the identical end state (same money; Arcanine Lv28 rather
than 27 on battle RNG). Tests 101/101.
sha256 629b3a4f26cfacef8a9cc4bc5ca4126b9408a5ead2d99e150a2c3b57c97f58bd.
Save states stay out of git as ever; the chain head moves
to the Orin by hand with its sha256.

## 16. Saffron + Silph Co. (2026-11-07, probes 150/151): the bag cap, a two-lead grind, and the office that wasn't where the map said

Gates (probe150): the Route 8 guard takes the Fresh Water
(wStatusFlags1 bit 6). Mr. Psychic hands over TM29 (event
944). The journey rides strand ON gatehouse warp cells
without firing them - the crossing has to be walked by
hand, and the Route 8 gate's warp tile is the door's
BOTTOM cell (8,10); (8,9) is a dead floor tile at the
building front that eats a whole debugging cycle if you
trust the warp table over the game.

The Card Key (5F) taught the leg's first lesson: Gen 1's
bag holds 20 item STACKS, and the pickup silently fails
at the cap. The probe's bag was 20/20. Fix, as played:
Saffron Mart, sell the Nugget (+¥5,000) and the Antidote
stack (+¥200) - shop menus are stage-gated on menuY
signatures (1 = BUY/SELL/QUIT, 4 = the list) and the
money delta is the only trustworthy receipt. Then the
key room via the (9,15) pad bounce to 9F and back.

The rival (7F west room, trigger (3,2)/(3,3), event
1856; Pidgeot 37 / Growlithe 38 / Exeggcute 35 /
Alakazam 35 / Blastoise 40) stopped the first pass
cold: two losses from Blastoise ~103/155 starts - the
typed driver plays correctly, it just loses - and an
in-battle potion-support wrapper wedged silently once
(the field menu or nothing, ever since). Vlad chose
the grind over the item-policy build and over skipping
ahead.

The grind (probe151): lead-and-solo, because Gen 1
switch-training pays nothing to a fainted participant
and splits XP - the grinder leads every fight it takes
XP from. Blastoise 48 -> 52 on Silph's own trainers
(finite, best XP + money density, and the 9F nurse is a
free in-building heal); Arcanine 27 -> 38 was the real
deficit and outgrew the building: 4F + 11F east, then
the Lavender-side route pools - Route 8, Route 10,
Route 12 (gate corridor crossed on foot; the Snorlax
at (10,62) woken with the Poke Flute from the bag menu
and beaten with no ball thrown), Route 13, and two
Route 14 fights to close the last 2,639 XP. Lessons
that cost real time: a whiteout is invisible to an
all-zero-HP check (Gen 1 auto-heals at the respawn -
detect it by the money/map signature after every
battle); the executor's planner stalls on ~85-cell
walks and on NPC bodies in chokepoints (the Route 12
pier is a waypoint chain; Fisher4's beaten body seals
his 1-wide dock cell; the (5,49) shelf is the detour);
Route 12 <-> 13 is a SOUTH map connection (offset
-20), Route 13 <-> 14 a WEST one at row 10, and the
block grid lies about half-block dock tiles. Route 12's
Fisher5 (ev1160) sits in a tile-sealed west-shore
pocket - one-way ledge in, no walk-out, Dig is
cave-only in Gen 1 - and is skipped by design; Route 6
and Route 11 were never needed. Grind totals: no
whiteouts after the detector was added (one before it,
in a discarded lineage), money ¥37,484 -> ¥81,570.

Third rival attempt: from the nurse at 100%, far-side
circuit, nobody chipped, no top-off spent. Blastoise
Lv52 soloed the team and fainted on the ace (+5,181
XP); Arcanine Lv38 closed it at 81/130. Event 1856.

Lapras: the 7F worker's gift routes to the box (party
full; nobody deposited or released). GivePokemon stops
at the NICKNAME screen, which is not a dialog state -
dialog closers wedge on it and the player reads as
glued to (2,5). START confirms an empty name; then
BIT_GOT_LAPRAS sets. Verified three ways: the status
bit, dex 14 -> 15, Lapras in the box.

Giovanni (event 1935): the compiled office geography
was wrong and the block grid nearly kept it wrong -
the 11F middle room (Giovanni (6,9), President (7,5))
is entered ONLY through the (6,12) card-key door from
the SW room, via the (5,7) pad chain and the west
column; the east column / south corridor route
dead-ends against it. Rocket1 (3,16, ev1924) guards
the SW room and is fought on sight. Blastoise took
8 HP across both fights and levelled to 53. The
President's only stance is (6,5), reached from (5,5) -
a dummy "inaccessible" warp tile that can be stood on,
which the executor's planner treats as a terminal, so
the office is walked by hand in both directions. The
Master Ball (event 1933, bag count 1) lands as the
bag's 20th stack - the cap the leg started with.

Chain: probe151/silph.state - Saffron City, healed
(Blastoise Lv53, Arcanine Lv38, Diglett Lv22, NidoranM
Lv3, Voltorb Lv16 Flash, Oddish Lv23 Cut), money
¥89,189, bag 20 stacks incl. CARD KEY and MASTER BALL,
Lapras Lv15 in the box, dex 15.
sha256 6b33101c55c8ee2251b34ca09ab1cb4ad04a2d35e2bde280e504bba71067ca28.
Ports: `saffron_gates` + `silph_co` with separable
beats (`silph_card_key`, `silph_grind`, `silph_rival`,
`silph_lapras`, `silph_giovanni`, `silph_master_ball`)
and the general machinery - `_silph_dfs` (the
model-free navigator that generalises the explore_nav
story for FACILITY interiors), `_silph_cab_to` /
`_silph_elevator`, the Mart sale, the gate/pier/border
crossings. NOT cold-replayed end-to-end, unlike the
Tower port: the grind alone ran ~2h of wall time
across a dozen probe scripts with hand-built
crossings, so the port is a faithful transcription
and the probes' saved intermediates are the proof
chain. Tests 104/104.

## 17. Saffron Gym / Sabrina (2026-11-08, probe152): the bag cap again, a learned warp graph, and the settle that lied

Bag first, again: Sabrina's post-battle script awards
TM46 only if the bag has room - and the Silph leg had
ended at exactly 20 stacks. This time the fix was PC
item storage, not a sale: the LIFT KEY (dead weight
since the Hideout) went into Bill's PC at Saffron
(wNumBoxItems $D53A). The storage menu shares its
signature with the PC main menu (topY 2, max item 3);
it is told apart by the cursor resetting to 0 when the
"Accessed Item Storage System." text clears - and A
presses must stop the instant it does, or the next one
selects WITHDRAW. Verified in RAM both directions:
bag 20 -> 19, storage [Potion, Lift Key].

Saffron City defeats the stock rides: journey() returns
None for Pokecenter<->Gym and the executor spins. The
probe crossed the city on the tile grid with
RAM-verified single steps, and the Pokecenter door has
its own gremlin - a wandering girl NPC parks on the
door stance (9,30); entry is an NPC shuffle from
(10,31). Inside, heal_at_pokecenter works unchanged.

The gym is one map (178), a 3x3 of rooms joined by 30
warp pads = 15 reciprocal pairs. The graph was learned
empirically (243 cells, 946 edges) and the compiled
table from SaffronGym.asm's warp events agrees with
all 38 teleports the learner recorded - the table now
lives in run_real as SAFFRON_GYM_PADS and the port
plans over it directly.

Two exploration bugs worth keeping. First, the settle:
a sight trainer's walk-up runs under wJoyIgnore with
no battle and no dialog yet, so a settle that breaks
on the first quiet check races the fade and records
the step as a WALL. Channeler1 "didn't exist" for a
whole pass that way; her sight column (10,3) was
reachable all along. The strong settle - three
consecutive quiet checks, joy-ignore clear - is now
the gym standard. Second, explore()'s frontier
fallback: when navigation to a frontier cell fails,
it marks ALL that cell's untried directions as walls,
poisoning cells it never tested. Purge rule: delete
None edges whose target was never actually visited.

The sweep (all seven trainers, events 866-872, sight
range 3 facing down; teams in the port's table): the
first pass fainted Blastoise to Psychic1's party and
cost the only Super Potion; Channeler1, found late,
dropped it to 31/178 again. Two heal trips. Sabrina
herself (Kadabra 38 / Mr. Mime 37 / Venomoth 38 /
Alakazam 43, the $FF custom-moves entry) fell to a
Lv55 Blastoise solo: 51/178 left, Hydro Pump dry,
Water Gun 25->17. Marsh Badge, TM46 in the freed bag
slot, money JPY 97,956. Chain head
probe152/sabrina.state.

Port (beat_sabrina + deposit_item + _gym_navigate +
_saffron_grid_walk): COLD-REPLAYED end-to-end from
silph.state on the third attempt. Attempts one and
two failed informatively - a mid-sweep blackout the
recovery-less sweep could not absorb, then a Sabrina
blackout stalling the scene ride in the respawn map -
and the port now carries the brief's own contingency
as code: heal-and-re-enter recovery inside the sweep,
and one Sabrina re-attempt after a blackout. The
replay's path cost more than the probe's (money
halved by blackouts; Oddish took XP and an
Acid-for-Sleep-Powder swap from a level-up prompt);
the chain head remains the probe's state. Tests
108/108.

## 18. Fighting Dojo (2026-11-08, probe153): the King's flag range, a frozen Pokédex page, and a species gate

The mini-leg Vlad unlocked by making the dojo's one
permanent decision himself: Hitmonlee over Hitmonchan.
The case, as presented to him with Gen 1 numbers: Lee
runs everything off Attack 120 / Speed 87 (Double Kick
now, Hi Jump Kick later, and Gen 1's crash-on-miss
penalty is a token 1 HP); Chan's famous elemental
punches are Special-based in this generation and run
off its 65 Special, not its 105 Attack - the coverage
is a trap, and what remains is a slower, sturdier
fighter. He picked the kicker.

The dojo (map 177, 10x12; Saffron door warp (26, 3),
entered from (26, 4) on the Sabrina leg's grid walk)
is one open mat. Its script holds a trap for the
unwary verifier: the Karate King's post-battle script
runs SetEventRange(EVENT_BEAT_KARATE_MASTER ..
EVENT_BEAT_FIGHTING_DOJO_TRAINER_3), so beating him
first would light all four Black Belt flags unread.
The belts were fought first and flagged individually
(850-853, parser re-validated on every anchor -
37/239/441/864/865/1231/1663/1703 - before 848-855
were trusted). In the event, the room did the ordering
itself: entering from the (4, 11) mat, the straight
walk north up column x=4 toward BB1's sight cell
crossed BB4's row 7, BB2's row 6 and BB3's row 5
sight lines first, and each belt engaged and fell on
its own row - Mankey/Mankey/Primeape 31, Machop/
Machoke 32, Primeape 36, Machop/Mankey/Primeape 31,
payouts totalling JPY 3,250, Blastoise levelling
55 -> 56 mid-column. The King is not a trainer-header
entry: the map's default script engages him when the
player stands on (4, 3), left of his (5, 3), and the
trigger fired first try (his Hitmonlee 37 +
Hitmonchan 37 - the player previews both gifts in
his party - paid JPY 925 and set 849). Across all
five fights Blastoise took zero damage: a clean OHKO
sweep, no items used (the bag's Super Potion count
was already 0 after the Sabrina sweep, so
_maintain_party never even fired).

The pick. Lee's ball is the west ball at (4, 1),
Chan's the east at (5, 1) - but the port does not
trust the cell. The ball's text runs DisplayPokedex
before its Yes/No, and DisplayPokedex leaves the
species in wCurPartySpecies ($CF91, from the ROM's
own sym file): the probe read 0x2B (Hitmonlee) from
RAM, with the dex page itself on screenshot -
"HITMONLEE, No. 106, KICKING" - before any
confirmation was pressed. The first ride then failed
instructively: the dex page waits on
WaitForTextScrollButtonPress, and the Lapras ride
(START presses + close_dialog) never dismissed it,
because close_dialog only presses while in_dialog
and the dex screen is not a dialog state - the
before/after screenshots were byte-identical, a
frozen page. Raw A presses dismiss it (probe153e);
from there the established pattern held (A confirms
YES at the take prompt, interleaved START clears
GivePokemon's nickname screen with the default
name). Verified three ways from RAM, as with
Lapras: EVENT_GOT_HITMONLEE (854) and
EVENT_DEFEATED_FIGHTING_DOJO (848) set,
EVENT_GOT_HITMONCHAN (855) clear, owned dex
15 -> 16, species 0x2B physically in the box (Lee,
Lapras, and the five catching-detour species),
party list unchanged - and a fourth, visual check:
the room screenshot shows (4, 1) empty and Chan's
ball still sitting at (5, 1), untouched. The other
ball's text would now only say "better not get
greedy"; it was never talked to.

Chain head probe153/dojo.state - Saffron City at
(26, 4), outside the dojo door: money JPY 102,131
(97,956 + 3,250 + 925, arithmetic exact), party at
full HP (Blastoise Lv56; no heal stop was needed),
bag unchanged at 19 stacks, Lift Key still filed in
PC item storage next to its Potion.

Port (fighting_dojo(choice="hitmonlee") +
take_hitmon / take_hitmonlee + _dojo_walk /
_dojo_ride_scene / _dojo_fight / dojo entry-exit):
the choice is a parameter with this line's value,
and take_hitmon carries the species gate as code -
it returns False without touching the Yes/No if
wCurPartySpecies is not the chosen species, and
refuses outright if the other gift's flag is set.
The brief's blackout contingency (heal, re-enter,
resume once) is in the umbrella, unused. COLD
REPLAY from probe152/sabrina.state passed on the
first attempt: fighting_dojo returned True and the
end state matches the probe chain on every invariant
- same money to the yen, dex 16, same flags, same
box, same party, same (26, 4) doorstep. The single
difference is Blastoise's HP, 151/181 against the
probe's untouched 181/181: battle RNG, not path
divergence. Tests 112/112 (four new dojo-constant
tests: event ids, species/gate addresses, the gift
table's west/east geometry, the belt table).

Gap notes: the dojo is a rare interior where the
stock world grid plus a column walk suffices - no
learned graph, no DFS (contrast Silph's components
and the gym's pads). The reusable artefact is the
gate, not the geography: any scripted object that
previews its gift through DisplayPokedex - the
fossils, the other Hitmon, any future one-of-two -
can be identified from wCurPartySpecies before the
player commits, and that check now exists as a
pattern. What the world model still lacks is any
representation of a one-of-two choice as a state
edge; the dojo's two balls are, to the planner, two
identical item pickups, and only the port's
parameter knows one of them ends the other.

## 19. Fuchsia leg (2026-11-09, probe154): a one-way gate, a zoo with a step budget, and the party screen that ate a battle driver

Bag prep first, and it bit. The leg adds HM03, HM04, the
Gold Teeth and TM06 to a 19-stack bag (cap 20), so three
spare TMs - TM02, TM21, TM34 - went into Saffron PC item
storage beside the Potion and the Lift Key. The §17
deposit's menu signature (MAX==3, topY!=0) turned out to
be stale-byte fragile in lineages that had already used
a PC: the bytes linger from the last session, the check
passes while the screen still shows "VLAD turned on
the PC.", and the steering misses. One intermediate
probe routine with a neighbouring wrong predicate
(withdraw demands the dialog flag down; the storage
menu keeps it up) pressed A in the live storage menu
and WITHDREW the Potion and the Lift Key into the bag.
Both incidents were caught by RAM verification in the
same phase and fully repaired there: the working
predicates are main menu = MAX==3 AND topY==2 AND not
in dialog, storage = the cursor reset 1->0 observed
BEFORE pressing A. End state verified stack-by-stack:
bag 19 -> 16, storage exactly {Potion, Lift Key, TM34,
TM21, TM02}, TM24/TM29/TM46 kept in the bag.
deposit_item in run_real.py now carries the fix.

The road south was quiet except for its doors. Gate
warp cells are DIRECTIONAL - they fire only when
stepped onto heading INTO the building. Route 8's
gatehouse strands every stock ride on the door's top
cell (1, 9), sliding along the front; the crossing is
a stance at (0, 10) and a step EAST onto the door
(the mirror image of §16's westbound crossing, which
worked for the same reason). Route 12's Fisher5
(event 1160) was guarded at every checkpoint and
never engaged; the pier followed §16's waypoints.
Standing trainers on Routes 14/15 and a few wilds
paid JPY 1,925 and a Blastoise level (56 -> 57) on the
way in. One honesty note on the guard: a fresh
const-parse of event_constants.asm puts Fisher5 at
692, and flags 686-692 (all seven R12 trainers) read
False in the final chain even though §16 records most
of them beaten in the grind. The 1160 anchor is the
numbering the codebase itself uses and has validated
behaviourally across §16/§17; the discrepancy is
recorded here rather than resolved - fresh parses
have been wrong before (§15's expression trap), and
the guard that matters (never enter his pocket) held
under either numbering, since the route never goes
there.

The Safari Zone broke the world model twice. Its
walkable() follows per-tile gfx data and splits
Center into disconnected pieces; Gen 1's real rule
(CheckTilePassable) tests ONE tile - the destination
cell's BOTTOM-LEFT tile - against the tileset's
passable list, plus tile-PAIR collisions on top.
Even with correct grids, Center's west doors open
only into its own sealed NW component: the route to
the West area rings the whole zone, Center -> East ->
North -> West (~430 steps of the 500 budget in
practice). The step counter wSafariSteps (0xD70D) is
stored BIG-ENDIAN - 0x01F4 at entry, gate-interior
steps not counted - and the balls live at 0xDA47.
One entry, JPY 500: paid at the counter from (4, 2)
facing right (Gen 1 counter-talk), ring walked with
a replanning BFS over grids compiled from pokered's
.blk data, Gold Teeth lifted at West (19, 7), HM03
SURF from the Secret House guru at West (3, 3) - his
event flag (2176) sets at the END of the script, so
the text is ridden fully before the flag is read.
Six encounters, all RUN from through the Safari
menu, balls 30 throughout: Doduo 26, Exeggcute 23,
NidoranM 22, Paras 23, Nidorino 30, Venomoth 32 -
nothing caught (the catching decision is Vlad's; the
zone re-enters for JPY 500). Exit was a deliberate
burn of the last 37 steps to ejection - items kept,
auto-returned to the gate. The Warden's exchange
verified both ways in RAM: teeth 1 -> 0, HM04
STRENGTH in the bag, events 569/568 set. Neither HM
was taught to anyone.

Fuchsia Gym's invisible walls are honest collision
tiles and the model grid sees them fine; the gym's
real traps were social and mechanical. Engagement:
stepping into a stance can start a sight trainer's
walk-up, and a DIRECTION press during it aborts the
engagement - ride it pressing only A; and a talk's
first A must be pressed blind, because waiting for
the dialog before pressing A never opens it (the
press IS the talk). The stock battle driver then
wedged twice on phases this gym serves constantly:
the shift prompt's party screen with a text box open
("... is already out!") - its stale-signature branch
mashes A, which re-selects the active mon and loops
forever - and move rejection (a Disable'd Hydro
Pump), same A-mashing shape. The fix is a hardened
loop: the party screen gets unconditional B priority
(B backs out whether or not a text box is up; after
a dozen B's it really switches), and the move pick
rotates when a confirmation spends no PP. With it,
the six trainers (events 602-607, object order
confirmed by three flags before the rest were
trusted) fell in seconds each. Koga (Koffing 37 /
Muk 39 / Koffing 37 / Weezing 43): Blastoise walled
the first three; the Weezing's Selfdestruct fainted
Blastoise from full line duty AND itself - as Koga's
last mon the double KO still pays: Soul Badge (6
badge bits), event 601, TM06 Toxic into the freed
bag slot (event 600), all verified from RAM. Two
mid-gym heal trips; heal_at_pokecenter FAILS in
Fuchsia (its counter logic never completes there) -
the port heals by riding in and talking to the nurse
directly.

Chain head probe154/fuchsia.state, sha256
6e30c19239abf2012ce88716f43251df145e75d1dceea2b2e
ec2a297b250c669a - Fuchsia Pokecenter, party at full
HP: Blastoise Lv58, Arcanine Lv39, Diglett Lv22,
NidoranM Lv3, Voltorb Lv16 (Flash), Oddish Lv23
(Cut); money JPY 115,288; dex 16; bag 19 stacks
incl. HM03 + HM04 + TM06; box unchanged (Lapras,
Hitmonlee + 5); PC storage: Potion, Lift Key, TM02,
TM21, TM34. No blackouts all leg.

Port (fuchsia_road + safari_zone_hms +
give_teeth_get_strength + beat_koga): the Safari
engine is the reusable core - SAFARI_GRIDS (grids
compiled from pokered .blk under the bottom-left
rule, embedded as bitsets), _safari_plan/_sz_nav
(ring routing over the areas + world-model warp
graph), _safari_steps (big-endian), _safari_run_
battle (RUN-only; catching stays a policy, not a
default), ejection as a legitimate exit when the
budget won't cover the ring back. _drive_battle_
hardened carries the party-screen B priority and
the PP-stall rotation. deposit_item's PC-menu
predicate is fixed. COLD REPLAY passed: fuchsia_
road, safari_zone_hms and give_teeth_get_strength
replayed uninterrupted from probe153/dojo.state with
probe-identical money at every checkpoint (JPY
104,056 / 103,556; the replay's five RUN-from
encounters were Exeggcute 27, Paras 23, Nidorino 30,
Chansey 26, Nidorina 30). beat_koga failed its first
two replay attempts - a sight-engage dialog starving
_step_to of A presses, then a beaten trainer's
walked-up sprite blocking the planned cell invisibly
to the tile grid - and the port now carries both
fixes (dialog-aware stepping, learned sprite blocks)
plus the probe's heal trips as umbrella recovery
(_fuchsia_gym_heal_trip, lead under 40% or fainted);
with them, beat_koga replayed from the same run's
Warden state to an end state identical on every
invariant: JPY 115,288, the same party at the same
levels, Soul Badge, TM06. Tests 121/121
(nine new: BE step counter, ball counter, grid
dimensions and stood-on cells, ring-route plans over
the real grids + pokered warp tables, ball cells
never routed through, Fuchsia item/event constants).

Gap notes: the bottom-left-tile rule is the general
FOREST-tileset truth - the world model's walkable()
is wrong for every forest map, not just the Safari
Zone (Viridian Forest included), and pair collisions
still have to be learned by stepping. The Safari
Zone remains a one-visit place in the port: the
engine enters, routes and leaves, but a catching
expedition needs a throw/bait/rock policy and a
target list, both Vlad's to set. The event-id
numbering split (codebase evparse vs a fresh const
parse, Fisher5 1160 vs 692) wants a diffing session
against wEventFlags on a known state before any
future leg trusts a fresh parse for a NEW flag.

## 20. Lapras Surf + Safari rare hunt (2026-11-09, probe155): the HM a star cannot learn, one-way geography, and the zoo again

Job A was Vlad's call over the lean presented to him:
Surf goes to the boxed Lapras, not over Blastoise's
Bubble. At the Fuchsia PC, TM24/TM29/TM46 were deposited
first (the bag sat at 19 stacks and the hunt prep needed
room for two rods), then NidoranM (Lv3) was deposited
and Lapras Lv15 withdrawn. Lapras knew Water Gun and
Growl - two moves - so per pokered's LearnMove the HM03
teach could only fill the first empty slot: the
forget-flow runs solely at four full slots, and the port
asserts exactly that (the ONLY legal change is the move
appended in the first free slot). Verified from RAM:
Lapras [Water Gun, Growl, SURF], in the party at slot 5,
Blastoise's four move bytes byte-identical (Hydro Pump /
Tail Whip / Bubble / Water Gun). The port is
teach_hm_to_boxed over _pc_swap_with_heal - stock
pc_swap ends in heal_at_pokecenter, which §19 proved
fails in Fuchsia - and its cold replay from
probe154/fuchsia.state reproduced the probe's end state
exactly (party, money, bag, box all compared
field-by-field).

Job C died on a ROM fact. Vlad's follow-up decision -
teach HM04 Strength to Arcanine replacing Roar - is
engine-impossible: Arcanine's tmhm bitfield in
data/pokemon/base_stats/arcanine.asm (the very data
CanLearnTM reads) contains no HMs at all, and neither
does Growlithe's. Verified three ways: the bitfield
itself, a behavioural teach attempt with the identical
flow that had just taught Lapras (Arcanine's moves came
back unchanged), and the HM04 teach-flow frames. The
coordinator declined the fallback of writing Strength
into Arcanine's struct: Vlad had authorised a struct
edit only to REMOVE Roar after a legitimate teach, and
adding a move the engine would never produce is a
brighter legitimacy line - it would taint a save line
whose ports and cold replays are all engine-verified.
Arcanine stays Bite/Roar/Ember. The verified
alternatives are Vlad's to pick: Lapras can learn
Strength into its free fourth slot right now (making
it the two-HM carrier: Water Gun/Growl/Surf/Strength);
the boxed Hitmonlee can learn it; Blastoise can, at
the price of a replacement decision he already
declined once. Until he chooses, Strength is on
nobody: Surf alone reaches Cinnabar, but the Seafoam
crossing wants both.

Job B's rods ran into geography. The Good Rod is a free
gift in Fuchsia (Guru's brother, map 164) and was
collected and bag-verified - and it cannot produce
Dratini anywhere (its only group is Goldeen/Poliwag
Lv10). Dratini needs the Super Rod (the four Safari
areas share its group: Dratini/Krabby/Psyduck/Slowpoke
Lv15, 12.5% per cast). The brief's detour - hop rides
Fuchsia -> R15 -> R14 -> R13 -> R12 house - is one-way-
locked in reverse, engine-verified: that road is a
one-way descent from Lavender. Route 13's plateau and
its pier strip are separate components joined only by
ledge jumps downward; Route 14's north strip is the
same trap; the world model's and even the compiled
engine-rule grids agree there is no eastbound path, the
executor's own tile_path concurs, and the one live
_executor ride spun replanning against the ledges. There
is no Bicycle in the bag for Cycling Road, and the only
feasible route (Surf R19/R21 to Pallet, Diglett's Cave,
R11, the R12 gate, the pier) needs Surf-step machinery
the executor does not have - water maps are capability-
gated in the world model but no surf step exists in the
executor. Making that trip is a scope decision for
Vlad, so Dratini was left unfished and the four entries
went to grass. Fisher5 was never engaged; nothing went
near Route 16's Snorlax.

The hunt itself (four entries, Y500 each, money
115,288 -> 113,288; dex 16 -> 22). Encounter data was
compiled from pokered (grass slot 8 = 4.3%, slot 9 =
1.2%): East holds Kangaskhan (s8) and Scyther (s9),
North Chansey (s8) and Tauros (s9), West Tauros (s8)
and Kangaskhan (s9), Center Scyther (s8) and Chansey
(s9). Species were read by INTERNAL id; a naive parse
of pokemon_constants.asm that ignores const_skip shifts
every later id (it produced Chansey 0x26, Dratini 0x48,
a "Tauros" that is Diglett) - the const_skip-aware
parse validates against all 16 species constants this
codebase already carried and against §19's raw Chansey
sighting byte 0x28. The Safari battle menu's cursor
bytes were measured live: left column 1, right column
13, no wrap (BALL=(0,1), BAIT=(0,13), ROCK=(1,1),
RUN=(1,13)) - _run_away's stale 15 is never reached,
which is harmless for RUN but wrong for aimed picks.

Entries: E1 East - PARAS 22, DODUO 26, EXEGGCUTE 25,
NIDORINO 33 caught; Kangaskhan twice fled. E2 North -
RHYHORN 26 caught; Tauros 28 seen and fled. E3 East
again (West proved unreachable: Center's fence-maze gap
is mis-modelled in SAFARI_GRIDS and the world model
alike, and _sz_nav's warp-cell dance dies at North
(38,12) on the ring) - KANGASKHAN 25 caught on its
fifth encounter of the entry. E4 North - no catches;
Tauros twice more, a Chansey in an earlier transit,
Scyther once in a discarded crash run. All catches
filed in the BOX (party full; SendNewMonToBox files
with no nickname prompt) and verified from RAM: dex
count, box contents, species. Final box additions:
Paras, Doduo, Exeggcute, Nidorino, Rhyhorn, Kangaskhan.

Two hunt mechanics notes. First, the exit: when the
step counter hits zero the PA announcement is a dialog,
and an unridden dialog freezes the player solid -
entry 1's first burn loop pressed directions for 600
iterations against an open text box. The hunt settle
rides dialogs (close_dialog) as well as battles; the
rescue from the mid-entry checkpoint (saved after
every catch - the other lesson: a dead process loses
its log, so battle records now print as they complete)
ejected cleanly. Second, bait: the compiled formula
(engine/battle/core.asm - after each action, flee iff
a random byte < speed*2, /4 while the bait factor at
0xCCE9 is up, *2 while the escape factor at 0xCCE8 is
up; bait halves the catch rate, rock doubles it) says
bait-first is right for the rares. The live hunt
disagreed in pattern if not in law: every baited
target fled after exactly one ball (10 of 11
encounters) and all six catches were unbaited
first-ball throws. The formula is engine-true and the
factor byte was read live, so the pattern is recorded
as an open question - bait-duration rolls and the
deterministic input cadence correlating the RNG are
the candidate causes - not as a mechanics claim. A
future hunt should A/B it deliberately.

Port: teach_hm_to_boxed, _pc_swap_with_heal,
_teach_hm_free_slot (strict free-slot assertion),
get_good_rod, safari_rare_hunt with _safari_hunt_entry
(stock-settle transits - the hunt settle breaks
_sz_nav's warp dance; pacing hunts), _safari_hunt_battle
(bait/ball policy above), _safari_menu_pick,
_safari_grass_pair over the compiled SAFARI_GRASS
bitsets, and the target/catch/speed tables. Fidelity:
Job A cold-replayed exactly; the hunt port is a
faithful procedure (Safari RNG does not reproduce
catches). No Job C code exists, deliberately. Tests
128/128 (7 new). Chain head probe155/safari-hunt.state:
Fuchsia PC, party healed, Lapras aboard with Surf,
dex 22, money Y113,288. Open for Vlad: the Strength
carrier, the Super Rod trip (and with it Dratini), and
Route 16's Snorlax.

## 21. Lapras Strength (2026-11-10, probe156): verification first, and the five-minute leg

The smallest leg of the run, run under the rule the
previous one bought: verify before you offer, and
verify again before you teach. Vlad's re-decision
(2026-11-10) put HM04 on Lapras - Strength into the
free 4th slot, no replacement, after the Arcanine
pick proved engine-impossible in S20.

Both pre-checks passed before a single input was
pressed. First, the ROM data: Lapras's tmhm learnset
in pokered's data/pokemon/base_stats/lapras.asm - the
very bitfield CanLearnTM reads - lists SURF and
STRENGTH. Second, the live save: a fresh RAM read of
probe155/safari-hunt.state showed Lapras in party
slot 6 holding exactly [Water Gun, Growl, Surf,
empty], Blastoise at [Hydro Pump, Tail Whip, Bubble,
Water Gun] and Arcanine at [Bite, Roar, Ember].

The teach itself was the bag flow behind
_teach_hm_free_slot, held to the same strict-append
assertion: the only legal change is Strength landing
in the first empty slot. It did - Lapras now runs
[Water Gun, Growl, Surf, Strength] with Strength at
its full 15 PP. Everything else is byte-identical to
the start state: both other movesets, the party
order, money Y113,288, the 17-stack bag (HM04 still
x1 - an HM is never consumed), dex 22, the box.

Media (probe156/media): the fast dialog ride's frame
burst froze on the bag list - at headless cadence the
screenshots lag the text - so the teach was re-driven
slowly on a scratch copy to catch the text frames.
The catch was worth it: alongside "Booted up an HM!",
the party ABLE screen shows Blastoise and Lapras ABLE
and Arcanine NOT ABLE - the S20 incompatibility
finding rendered in the game's own UI - over the
words "LAPRAS learned STRENGTH!".

Port: teach_strength in run_real.py - the pupil is
located by the Surf it already carries, its live
moveset is read before any input, a four-move or
Surf-less pupil stops the step untouched, and the
teach runs through _teach_hm_free_slot. Cold replay
from safari-hunt.state through the port alone ended
field-identical to the probe's chain head (party,
money, bag, box, dex all compared). Tests 131/131
(3 new: TestTeachStrength parses both learnsets out
of pokered - Lapras holds SURF + STRENGTH, Arcanine
holds no HMs - so the S20 finding is now a regression
test, not just a story). Chain head
probe156/lapras-strength.state: Fuchsia PC, party
healed, Lapras the two-HM carrier (Surf + Strength),
dex 22, money Y113,288. Cinnabar needs only Surf;
the Seafoam crossing wants both, and now has both.
Still Vlad's calls: the Super Rod trip (and Dratini),
Route 16's Snorlax, the TM shelf, Rare Candy / HP Up.

## 22. The Grand Tour (2026-11-10, probe157): surf machinery built and proven - and Route 20 is sealed

Vlad's chosen route: surf around Seafoam to Cinnabar (Blaine),
on to Pallet and the Super Rod, home to Fuchsia, then fish
Dratini in the Safari Zone. Phase 0 landed in full. The
crossing did not - because the geography the route assumes
does not exist in Gen 1. This section records both halves,
and the tour is PARKED on Vlad's call at the Seafoam shore.

SURF MACHINERY (built, live-proven). Engine facts compiled
from pokered: there is NO A-button water prompt in Gen 1;
boarding is the party-menu SURF field move. The tile in
front must pass IsNextTileShoreOrWater (sampled tile
$14/$32/$48); the engine steps the player onto the water
itself and sets wWalkBikeSurfState (0xD700) to 2. While
surfing, stepping onto a land-passable cell auto-disembarks;
onto $14/$48 (and $32 off SHIP_PORT) you keep surfing.
Build: the generator now emits the sampled `tiles` and the
tileset passable list (`coll`) for ALL 247 maps (every
pre-existing field byte-identical, verified by field-diff);
world.py gains GameMap.surf_cells, GameMap.tile_walkable
(engine-true passability from tiles x coll - the walk grid
cannot be trusted where it is the open-grid coherence
fallback: Route 20, Pokemon Mansion 3F, Rocket Hideout B4F,
SS Anne 1F Rooms, Celadon Mansion Roof, Lance's Room),
World.surf_path (land U surf BFS: warp-destination rule,
ledge hops, TilePairCollisionsWater pairs) and
World.surf_crossings (journey()'s arrival arithmetic);
pyboy_emu gains field_surf (the _field_move macro);
run_real gains the surf core (surfing / _surf_carrier /
_surf_board / _surf_step / _surf_walk_to / surf_to):
one RAM-verified step at a time, wilds fled, trainers
fought, capability gated on the live party read (Lapras).

Two live traps banked. (1) TRAINER SIGHT-LOCK: a step can
be refused in all four directions with no battle - the
trainer has spotted the player and the engagement script
stalls while directions keep being pressed; _refusal_recover
idles with neutral input, taps A, and resolves the battle
that fires (R19's Swimmer4, whose LIVE sprite sits at
(9, 13), two tiles south of his .asm object coords - a
tall-map sprite-loading offset). (2) A fled battle can
leave its empty text box standing, swallowing every later
direction press; the steppers drain it (close_dialog).

PROOF HOP (probe157a; re-run byte-identical): Fuchsia PC ->
out -> Route 19 via the ledge staircase -> beach stance
(9, 9) -> BOARD (surf byte 2, standing on (9, 10)) -> swim
to (9, 16) -> disembark (byte 0, on foot at (9, 9)).
Swimmer4 beaten on the walk back (+150 yen). R19->R20
crossing also proven (landed surfing at R20 (95, 4)-ish;
the held crossing press carries a few tiles into the new
map - surf_to verifies the map, not the arrival cell).

THE BLOCKER: ROUTE 20 HAS NO SURFACE PASSAGE PAST SEAFOAM.
With engine-true passability, the island (x43-62), $3A
shoal columns at x43/x55/x62, the row-1 $3A band (the row-0
"all-water northern channel" is a sealed border strip) and
the southern strip split at x62 cut Route 20 into two
basins that share NO cell:
- East basin (from R19): the island's north shore, the west
  cave's yard, the x44-45 channel, the west pool - a closed
  system, reachable and fully navigable, going nowhere west.
- West basin (from Cinnabar): the southern strip (rows
  14-16) and the lagoon's east pool (x56-61, rows 12-16).
Verified three ways: tile data + BFS region computation;
unit tests (test_route20_has_no_surface_passage_past_seafoam);
LIVE - a westward press at (63, 14) bumps and
wTileInFrontOfPlayer reads $3A. (One live read briefly
suggested $14 there; it was a stale facing byte, one step
behind - re-measured under a settled face.) The $11 plateau
patches are decorative mesas ($11 is not in Overworld_Coll)
- not standable; both cave entrances sit in ledge bands and
are reached from their own yards. No ledge hop bridges the
basins (every face's back cell is unstandable mesa), there
are no Strength objects on the surface, and the land detour
is closed too (the Fuchsia->Lavender road is one-way-locked
northbound (S20), and Cycling Road needs the Bicycle we do
not have).

The only through-route is Seafoam's interior. A 1F-only
transit exists on the compiled grid: west door (4-5, 17) ->
west corridor -> the (6-7, 11) gap -> the central chamber
-> east rows 12-15 -> east door (26-27, 17). The two 1F
boulders are off that path; no Strength is needed; Articuno
is on B4F and untouched by it. The brief's fence - do not
enter the Seafoam cave system - was written for a surface
route that does not exist, so rather than improvise past
an explicit fence the tour parked: retreat to Fuchsia,
heal, save probe157/blocked-at-seafoam.state (sha256
4682fcdb62aabb80...; deltas vs the chain head: +150 yen,
Swimmer4's beaten flag). CHAIN HEAD UNCHANGED:
probe156/lapras-strength.state.

VLAD'S CALL: (a) authorise the 1F-only transit (the cave
lobby only - no B1F, no Articuno, no loot) and resume the
tour as planned; (b) authorise a fuller Seafoam crossing
if the 1F transit proves impassable live; (c) abandon the
Cinnabar route - noting Blaine, the Super Rod and Dratini
all hang off it (Pallet is unreachable any other way:
the R21 surf starts at Cinnabar, and the land roads are
one-way-locked or Bike-gated).

PORT STATE: the surf machinery, tile_walkable and the
generator's tiles/coll emission are in and tested
(142/142, 11 new). The tour functions - mansion_secret_key,
beat_blaine (quiz answers and gate events 680-685 compiled),
super_rod_run, safari_dratini (Super Rod group 6: Dratini /
Krabby / Psyduck / Slowpoke, all Lv15, Dratini 12.5% per
cast; unbaited-ball policy per the S20 anomaly) - are
written into run_real S22 from compiled pokered ground
truth (Mansion switch doors with ReplaceTileBlock b=Y,
c=X semantics and per-cell gate-block patterns) but are
UNPROVEN: the leg stopped before their first live step.
No cold replay past the block was possible; the
deterministic spans that did run (the proof hop) replay
byte-identical from lapras-strength.state. Media:
probe157/media (beach stance, first surf, disembarked).

## 23. The Grand Tour, completed (2026-11-10/11, probe158 + replay probe159): the fuller Seafoam crossing, three compiled "truths" that weren't, and the NPC the map cannot see

Vlad authorised the fuller Seafoam crossing ("go fuller
Seafoam crossing", 2026-11-10) and added the Vermilion
BIKE VOUCHER to the route mid-tour. The tour ran end to
end from probe156/lapras-strength.state with NO blackout
at any point. NEW CHAIN HEAD: probe158/grand-tour.state
(sha256 db0d713cb3c9cb84c7c7bdba15fe88d54d8688cf1bf08-
8ddae1c9df76df64f2f) - Fuchsia PC, healed, 7 badges,
Pokedex 26, money 136,251 yen, bag 20 stacks incl. BIKE
VOUCHER + SUPER ROD. Ported and pushed (9b5f53d, suite
151/151, 9 new pins) and cold-replayed in full (below).

SEAFOAM, FLOOR BY FLOOR. First, a correction owed to
S22: its "no surface passage" claim was OVERSTATED. The
island's north plateau IS a surface route - sail R20's
north channel, disembark at (61, 5), walk west to the
WEST door (48, 5). S22's BFS was surf-only and missed
the disembark-and-walk link; R20's two surf basins still
share no cell, the EAST door's yard is sealed from the
east (the $3A shoal column x62 rows 10-16 + row 17),
and a Fuchsia->Cinnabar crossing still needs the full
boulder cascade - but "Route 20's surface is sealed"
was wrong as stated. Cave entry is the west door ->
SF1 (4, 17); exiting a cave door lands on the cell in
FRONT of it (west door -> (48, 6), east door -> (58, 10))
and the exit pattern is: stand on the door cell, keep
pressing OUT until the map flips.

Boulder machinery (all RAM-verified): Strength via
Lapras's party-menu field move; the activation bit is
wSpriteStateData1 0xD728 bit 0, re-checked before every
push; a push takes TWO direction contacts (the first
only sets the tried-flag); boulder sprite k's cell =
wSpriteStateData2 0xC200+16k, (byte[+5]-4, byte[+4]-4);
hole cells are WALKER HAZARDS (surf_path treats them as
walkable floor) - SF1 {(17,6),(24,6)}, B1F {(18,6),
(23,6)}, B2F {(19,6),(22,6)}, B3F {(3,16),(6,16)} are
blocked in every cave leg. A hole push gets AT MOST TWO
contacts - a third could step the player onto the hole.
The cascade, in order (event flags are the verdicts):
1F boulder1 UP x4 from (18,11)..(18,8) to (18,6), LEFT
from (19,6) into hole (17,6) = event 1294; 1F boulder2
UP from (26,8), LEFT from (27,6), LEFT from (26,6) into
hole (24,6) = 1295. B1F boulder2 RIGHT from (21,6) into
(23,6) = 2497 (reached via SF1 (25,3)); B1F boulder1
RIGHT from (16,6) into (18,6) = 2496. B2F boulder1
RIGHT from (17,6) into (19,6) = 2504. B2F boulder2 is
the long way: B2F -> B1F -> B2F -> B3F peninsula ->
the current ride -> B4F lagoon -> up and around ->
LEFT from (24,6) into hole (22,6) = 2505 (SEAFOAM3:
the pool calms).

The currents are a one-way conveyor, not swimmable
water: pre-cascade there is NO free swimming in the
B3F pool. Boarding is pair-rule-gated by the SHORE
tile: the B3F peninsula (15, 7) is $15, the ONLY
boardable shore into the pool - $05 raised-floor shores
refuse ("No SURFing on LAPRAS here!", screenshot-
verified; the CAVERN pair {$14,$05} collides). Boarding
at the peninsula steps onto trigger (15, 8) and the
DefaultScript rides you to B4F (20, 15), surfing. B4F's
south door is ENTRY-ONLY pre-cascade (the DefaultScript
drift ejects you north - the prep notes had the
condition backwards; it fires when SEAFOAM3 is NOT
set). Post-cascade the finish is: B2F -> B3F -> B4F
(25,4) -> B3F (8,6) -> peninsula -> board -> CALM CHECK
(300 frames, still at (15,8)) -> surf east -> disembark
(23,9) -> climb out -> SF1 east pocket -> (26,17) door
-> R20 yard (58,10) -> surf to Cinnabar. ARTICUNO:
never engaged, never sighted (the B4F legs stay in the
lagoon and the north/east land, out of its island's
sight). No Seafoam catches; cave wilds fled, water
trainers fought by the walker. Cinnabar checkpoint
money 115,123 yen - the replay matched it to the yen.

MANSION. The S22 route was impossible as modelled, and
the reason is pure switch logic: the 1F stairs room's
two doors need OPPOSITE switch states in sequence
((12,6) open-when-OFF, then (10,8) open-when-ON) with
the only 1F statue west of both. The real route fell
out of an offline (map, cell, switch) graph search
that included the 3F hole drop: enter OFF -> (5,10) ->
2F -> (6,1) -> 3F -> statue ON -> hole (16,14) -> FALL
to the 1F strip at the same coords -> stairs (21,23)
-> B1F -> statue OFF -> statue ON -> key ball (5,13)
-> OFF -> ON -> exit (23,22) -> 1F east doors (26,27).
Two data corrections: B1F has TWO statues, (20,3)
stance (20,4) AND (18,25) stance (18,26) - the port
knew only the first; and hidden-event hole indices 1-2
on 3F land on 1F while index 3 lands on 2F (a miss is
recoverable by climbing back). The S22 YesNo guess is
SETTLED, against the guesser: _yesno_pick's return
value is UNRELIABLE (its (13,0) tilemap detector can
miss the open window) - in the live run it returned
False on a toggle that had in fact succeeded. The
event FLAG is the only verdict; _mansion_press_statue
no longer early-returns on it. SECRET KEY verified in
bag from RAM; the run also picked up CARBOS from the
1F (18,21) ball (unused - a permanent stat change is
Vlad's call). The mansion switch flag reads False
again once back on Cinnabar (per-visit behaviour).

BLAINE. Two more S22 corrections, both proven live.
(1) The quiz gate events are 681..686, NOT 680..685:
hidden-event gate index i sets event 680+i (FlagAction
counts from EVENT_CINNABAR_GYM_GATE0_UNLOCKED=680 with
the index itself). Proof: machine (15,7) set event 681.
The door pairing shifts with it: (9,3)<->681, (6,3)<->
682, (6,6)<->683, (3,8)<->684, (2,6)<->685, (2,3)<->686.
Quiz answers and machine coords were correct as
compiled. (2) The ceremony races a bare drain: with
the gates fixed, all six quizzes passed in one round
and Blaine was beaten (payout 4,653 = 99x47 proves the
fight), yet EV_BEAT_BLAINE sat unset because the
script's badge/TM38 sequence starts a beat after the
battle - beat_blaine now rides A until flag + TM38
land. Verified from RAM: 7 badge bits, EV_BEAT_BLAINE
(665), EV_GOT_TM38 (664), TM38 in bag, money 134,286
yen (replay: identical). All six gym trainers were
also beaten along the way (events 667-672).

SURF DEPARTURE FIX. surf_to failed Cinnabar -> R21:
Cinnabar's north exit cells are LAND shore cells (the
crossing water is R21's first row), so the walker
delivers you to the beach on foot and a bare surf step
cannot board. surf_to now boards (party-menu SURF)
whenever a leg starts un-surfed at a shore. Pallet
landfall (7, 14); Mum's heal used the (5, 2) -> talk
(5, 3) stance and cured Tentacool poison picked up on
R21 (HP ticking 44 -> 43 indoors - poison walks with
you). _mom_heal hardened to match (stance order +
fighting-walker exit; its old exit stage had failed
live after the heal landed).

VERMILION (Vlad's addendum). The tour's bag math broke
exactly as predicted: 20 stacks at Blaine, voucher +
rod = 22. TM06 and TM38 were deposited at the
Vermilion PC (bag 20 -> 18; both retrievable from PC
storage) - TM deposits only, as authorised. Fan Club:
the chairman at (3, 1) sits behind a table block; the
working stance is (2, 1) facing RIGHT (the compiled
(3, 2) is a table cell). BIKE VOUCHER (0x2D) verified
in bag from RAM. The Super Rod followed on R12
(Fishing Guru stance (1, 4); Fisher5's flag asserted
False at every checkpoint), bag exactly 20.

THE NPC THE MAP CANNOT SEE (the ride home). The R13
fence maze wedged both riders for ~20 minutes apiece
in the live run, and the first port replay wedged
again for 18. The root cause is not the maze: static
trainers are invisible to the tile model. R13's Bird
Keeper stands at (12, 4) - already beaten in this
lineage, so a permanent obstacle - and every shortest
path west funnels through his cell; _walk_fighting
burned 446 s bumping to (13, 4) and stalling. Worse,
R14's entry has a sealed pocket: crossing R13 at row 4
lands at (16, 6), a 4-cell pocket whose only exit cell
(15, 6) is occupied by another STAY Bird Keeper, while
row 10 lands in the open map (reachable region 400+).
_ride_or_cross now: tries candidate boundary rows
nearest-first, paths with LIVE NPC cells blocked
(sprite-table reads, the same 0xC200 addressing as the
boulders), walks with _surf_walk_to (it replans around
refused cells), and MOBILITY-TESTS each landing - the
reachable region from the landing cell, live NPCs
excluded, must reach 40 cells, else step back and try
the next row. Result: R13 -> R14 in 4 s, R14 -> R15 in
3 s, R15 -> Fuchsia in 3 s (against ~40 minutes of
wedging before the fix).

DRATINI. The port's stance search picked (15, 14) - a
shore segment fenced off by posts and unreachable in
its own passability grid (the search never checked
reachability); _safari_pond_stance now filters to
anchor-reachable cells. The second attempt caught
Dratini and then CRASHED in the bonus-grass phase
(_safari_settle's "safari RUN failed" inside _sz_nav)
and lost the catch with the session - the bonus block
is now wrapped so it can never cost the mission. The
live catch (probe158o3, stance (15, 7)): ONE entry,
FIRST cast, one unbaited ball - DRATINI Lv15. The
burn-out exit walk then engaged the standing hunt
policy: CHANSEY Lv23 CAUGHT (bait 1, balls 1 - a
long-standing target), VENONAT Lv22 and PARASECT Lv30
caught, SCYTHER Lv23 appeared once and fled. Tauros
never appeared. Final Pokedex 26. Media caveat: the
encounter/throw frames caught the bite-transition
wipe; the RAM/log evidence is conclusive.

COLD REPLAY (probe159): the whole port pipeline
re-driven from lapras-strength.state - seafoam_
crossing True in 69 s (money 115,123, identical),
mansion_secret_key True in 24 s (116,573, identical),
beat_blaine True in 32 s (134,286, identical),
super_rod_run True stage-wise + in legs (voucher,
rod, Fisher5 untouched, 136,751 at R12, identical),
safari_dratini True in 128 s - replay RNG differs, as
it must: Dratini fled 5 casts in entry 1 and was
caught in entry 2 (Slowpoke/Krabby/Psyduck joined the
replay lineage; dex 26 by a different road). Battle
RNG is not byte-identical and never claimed to be;
every deterministic span matched to the yen.

OPEN DECISIONS (Vlad's, untouched): Articuno - catch
call unmade, never engaged or sighted. Route 16
Snorlax - decision MADE 2026-11-10: CATCH (balls first,
Master Ball holstered, KO only if balls run out); it
gates the Fly leg, which also cashes the Bike Voucher
at Cerulean's Bike Shop for Cycling Road. Safari
leftovers: Chansey is now CAUGHT; Tauros and Scyther
still outstanding; Safari West still unreachable (grid
mis-model, _sz_nav dies at North (38,12)). TM shelf:
PC storage now holds TM02/21/24/29/34/46 + TM06 + TM38;
none taught, ever; Rare Candy, HP Up and the Mansion
Carbos unspent. Ahead: Giovanni's Viridian Gym, the
final rival fights, Victory Road, the Elite Four.

## 24. The Fly leg (probe160, port-verified probe161)

Vlad's orders (2026-11-10): collect Fly ("you don't want
to have to walk everywhere"), and the Route 16 Snorlax
gets CAUGHT - balls first, Master Ball holstered, KO
only if the balls run out. From the grand tour's chain
head (probe158/grand-tour.state, Fuchsia PC, Blastoise
Lv61 - it levelled on the tour).

ROUTE: the brief's "surface only" R19/R20 surf fails at
the plateau shore transitions, so the leg rode the
tour's plateau route into Seafoam's west door and made
a CAVE TRANSIT (the fuller-crossing precedent Vlad
authorised for the tour; Articuno and B4F untouched, no
boulder pushes - the cascade flags were already set).
A pure 1F transit does not exist; the working line is
1F -> B1F -> B2F -> B1F (ladder reset) -> B2F south ->
B3F -> the stilled pool -> B2F east -> B1F east -> 1F
east pocket -> out. Two engine truths fell out:
TilePairCollisionsLand is UNORDERED, and surf_path did
not apply the LAND pair table to on-foot steps at all -
Seafoam B2F's west column is sealed by the CAVERN
$20/$05 pair row. agent/world.py surf_path now refuses
land-pair-forbidden steps between two non-surf cells
(the same rule _bfs_steps applies), and the offline
floor-component analysis (SF1 2, B1F 3, B2F 4, B3F 6
ladder-linked components) is what made the transit
routable. Do NOT block live boulder sprite cells in
cave pathing: fallen boulders read as phantom sprites
and sealing (18, 10) walls off SF1's corridor.

Then the long way round on proven hops: Pallet -> R1 ->
Viridian -> R2 -> Viridian Forest -> Pewter (heal) ->
R3 -> Mt Moon -> R4 -> Cerulean (heal). Money +210 from
one trainer a ride fought: 136,251 -> 136,461.

THE BICYCLE (Cerulean Bike Shop): the bag was at 20
stacks and the shop script's GiveItem runs BEFORE the
voucher is removed, so at 20 stacks the exchange aborts
on the bag-full branch (verified live - the brief's
stack-neutral assumption is wrong). HM05 Flash was
deposited at the Cerulean PC first (Voltorb knows the
move; the item is retrievable; storage now also holds
HM05). The clerk at (6, 2) is talked to from (6, 4)
facing UP across the counter row. Verified from RAM:
BICYCLE x1, voucher consumed, EV_GOT_BICYCLE (192) set.

EAST/SOUTH: R9 -> R10 -> Rock Tunnel (Flash) ->
Lavender -> R8 gate -> Saffron -> R7 gate -> Celadon
(heal). Two gate lessons: hop rides strand on gate door
cells (the warp does not fire when a ride stops ON the
cell; the R8 crossing walks the door's BOTTOM cell
(8, 10) then steps through), and the R7 gate is crossed
on foot at (19, 10) -> Route7Gate (5, 4) -> (1, 4) ->
the west segment (11, 10), then a battling ride.

THE SNORLAX (Route 16 (26, 10); the stance EAST of it
is (27, 10) - the brief's (26, 10) is the Snorlax's own
cell, and an early walker "stuck" at (27, 10) was being
blocked by the Snorlax itself). Checkpoint saved before
the wake (never overwritten): Great x4, Poke x8,
Master x1, Flute x1 verified in the bag. Final battle:
flute wake -> switch to Oddish -> Sleep Powder (3-turn
sleep) -> switch to Blastoise -> Bubble CRIT 141 -> 68
(a crit at full HP cannot kill: 68 left) -> GREAT BALL
#1 -> CAUGHT, asleep at 68/141. Verified from RAM:
stored 23 -> 24, dex 26 -> 27, Snorlax Lv30 in the BOX
(party full), EV_BEAT (1225) set, balls after: Great
x3, Poke x8, Master Ball x1 NEVER THROWN. Party took
zero damage. Attempts disclosure: six earlier attempts
from the same checkpoint were discarded - all driver /
policy defects, no game failures (a policy that threw
at full HP into Rest windows and never entered the
Powder phase; two menu-steering wedges; a tooling abort
seconds after launch; a cursor assert after 2 throws;
throw-entry gates that failed on the wake turn). The
checkpoint rule was written for a crit accident; the
only crit was the planned full-HP Bubble. Gen 1 facts
banked: sleep does NOT break on damage; Rest's
self-sleep is exactly 2 turns; Snorlax's AI spams Rest
(heal + sleep) and Amnesia; it outspeeds Oddish, so
Powder must land before a Rest; Blastoise always acts
first. Battle-menu mechanics for the port: the main
menu cursor is (top_y 14, item, col) with FIGHT at
(0, 9) and PKMN at (0, 15); _advance_battle leaves it
on FIGHT; the move menu is top_y 12 with a 1-based
cursor byte; the party pick needs a confirming A on
the SWITCH/STATS/CANCEL submenu; text pages need A
presses (a passive wait stalls on the wake text); and
_use_bag_item cannot be used for the Poke Flute - it
verifies success by consumption and key items are
never consumed.

THE FLY HOUSE: row 9 is a continuous hedgerow fence
with NO gap; the gatehouse's south corridor is the
Cycling Road bike-check (a guard behind a counter at
trigger cells, NOT a passage) and Gate2F is a dead-end
viewing room. The real entrance is the CUT TREE at
(34, 9), whose 2x2 block straddles the fence between
the road (34, 10) and the east yard (34, 8) - the
brief's "through the Cut trees" made literal. Yard west
-> the gate's NORTH corridor (doors (24, 4)/(24, 5);
cross on row 2 - the gambler stands on row 3) -> out
at (17, 4) into the house plot -> door (7, 5) -> the
girl at (2, 3) facing RIGHT, talked to from (1, 3).
Bag was 19 stacks after the catch, so HM02 fit with no
Celadon deposit: HM02 x1 in bag, EV_GOT_HM02 (1230)
set, 20 stacks. The return trip re-cuts the tree (map
reload regrows it).

TEACH FLY - VERIFY FIRST: Pidgey's pokered tmhm
includes FLY (checked in the data AND pinned by
TestFlyLeg); its live moveset was [Gust] - one move,
Lv3. At the Celadon PC: pc_swap DIGLETT out / PIDGEY
in, teach via _teach_hm_free_slot's strict-append:
[Gust] -> [Gust, FLY]; Blastoise / Arcanine / Lapras
move bytes verified unchanged; HM02 not consumed.
End arrangement per the brief: PIDGEY STAYS IN THE
PARTY as the fly carrier, DIGLETT STAYS BOXED
(reversible at any PC). Two heal traps banked:
_heal_at_pc EXITS the Pokecenter afterwards, and a
full-HP party makes heal/ride calls no-op - check the
map, not the return value, before PC work.

THE FLIGHT: Celadon street -> START -> POKEMON ->
Pidgey -> FLY. The backend's _field_move cannot fly:
its B-press backout cancels the Fly map, so fly_to is
hand-driven. The map always opens on PALLET TOWN and
UP cycles the VISITED flyable towns - the live order
(read off the header) is Pallet, Viridian, Pewter,
Cerulean, LAVENDER, VERMILION, Celadon, Saffron,
FUCHSIA, Cinnabar. Counted presses are NOT reliable
(one run double-stepped and landed Cinnabar); the
steering signal is the header text itself, decoded
from the 0x9C00 tilemap (Gen 1 font A=0x80, space
0x7F; wWhichTownMapLocation at 0xCD3D does NOT track
the fly cursor in this build - it sat at 0xFF through
every press). Landing verified from RAM: Fuchsia City
(7) at (19, 28). Healed at the Fuchsia PC.

CHAIN HEAD: probe160/fly.state, sha256
3cf144f5079f7d463d01e5acfe0e68bef380dbc328e93d22
11a88fd4235936b7. Party (all full HP): Blastoise 61,
Arcanine 39, Voltorb 16, Oddish 23, Lapras 15,
Pidgey 3 (Gust/Fly). Box 18 (incl. Snorlax, Diglett).
Money 136,461. Dex 27. Badges 7. Bag 20 stacks incl.
Bicycle, HM02, Great x3, Poke x8, Master x1.

PORT (probe161): bicycle_exchange, snorlax_catch,
route16_fly_house, teach_fly, fly_to + _step_to_live
(the greedy live stepper for cells the model cannot
path) into run_real.py; the surf_path land-pair fix
into agent/world.py; TestFlyLeg (6 tests: wiring,
constants, Pidgey learnset parse, the Master-Ball
source pin, the town cycle order, the header decode)
- suite 157/157. FIDELITY: this leg IS cold-replayed,
through the ported code - probe161 re-ran every ported
function from the probe160 checkpoints and reproduced
every outcome: bicycle_exchange True (after two fixes
the verification caught: deposit_item needs the caller
inside the PC, and the shop stance needs a live-step
settle + an initiating A), snorlax_catch CAUGHT again
on ONE Great Ball (the catch is RNG; the policy and
mechanics are identical and verified), route16_fly_
house True, teach_fly True ([Gust, FLY], Diglett
boxed), fly_to Fuchsia True (map 7).

OPEN DECISIONS (Vlad's, untouched): Articuno - catch
call unmade, never engaged or sighted. Safari
leftovers: Tauros and Scyther still outstanding;
Safari West still unreachable. TM shelf: PC storage
holds TM02/21/24/29/34/46/06/38 + HM05 + Lift Key +
Potion; none taught, ever; Rare Candy, HP Up and the
Mansion Carbos unspent. Party arrangement: Pidgey in /
Diglett boxed (the Fly-leg brief's end state; Vlad can
re-shuffle anytime). Ahead: Giovanni's Viridian Gym
(Earth Badge), the final rival fights, Victory Road,
the Elite Four.

## 25. The Viridian leg (probe162, port-verified probe163)

Vlad's order (2026-11-11): the Earth Badge, #8, the
last one. From the Fly leg's chain head (probe160/
fly.state, Fuchsia PC, Blastoise Lv61, bag at 20
stacks). Fences held throughout: no Route 22 exit,
no Victory Road, no catches, no TM teaching, no
stat items, Articuno untouched.

FLIGHT + DEPOSIT: exit Fuchsia PC -> fly_to Viridian
(landing (23, 26), RAM-verified) -> Viridian PC:
deposit GOOD ROD (0x4D, superseded by the Super Rod)
verified from RAM - bag 20 -> 19 stacks, Good Rod x1
in PC storage. The deposit is the Sabrina lesson
applied BEFORE the fight: Giovanni's script gives
TM27 Fissure in the post-battle ceremony, and at 20
stacks the bag-full branch would have bounced it
(the script does allow a re-talk retry via
EVENT_GOT_TM27, but the badge run should not need
it). The gym's REVIVE item ball at (16, 9) was
deliberately NOT collected for the same bag
discipline.

THE ARROW FLOOR, COMPILED: the gym script's default
handler looks the player's cell up in
ViridianGymArrowTilePlayerMovement (scripts/
ViridianGym.asm) and, on a hit, simulates the pad
(StartSimulatingJoypadStates) with wJoyIgnore set -
12 triggers: (19,11) UP 9, (19,1) LEFT 8, (18,2)
DOWN 9, (11,2) RIGHT 6, (16,10) DOWN 2, (4,6) DOWN 7,
(5,13) RIGHT 8, (4,14) RIGHT 9, (0,15) UP 8,
(1,15) UP 6, (13,16) LEFT 6, (13,17) LEFT 12.
probe162e calibrated the semantics live: a slide is
up to N steps in the arrow direction, stopped early
by walls/sprites, triggers chain, ~48 frames/step -
(11,2) RIGHT 6 landed exactly (17,2); (18,2) DOWN 9
landed exactly (18,11). One anomaly banked: in
probe162c a DOWN 9 entered in immediate succession
after the (11,2) ride stopped 2 steps early at
(18,9); entered cold it runs the full 9. The ported
navigator therefore never trusts a predicted
endpoint - it re-reads the live cell after every
action and re-plans (BFS over the compiled grid +
slides, sprite cells read LIVE from
wSpriteStateData2 via _boulder_cell).

THE WALK-UP TRAP (probe162b's failure, fixed in
162c): gym trainers are STAY sprites, but a sight-
engaged trainer WALKS UP to the player and remains
on his walk-up cell after the battle. T5 (Blackbelt,
(10,1)) engaged at (10,5) and stayed on (10,4),
sealing the only column north; a static-sprite BFS
bumped it 160 times and died. The fix is threefold:
live sprite blocking, a learned set fed by
no-progress steps, and - if the road seals anyway -
exit + re-enter (sprites reset home; beaten trainers
stay beaten and do not walk up again; the re-entry
fallback was armed but not needed in 162c).

TRAINERS: the door-to-Giovanni road crosses exactly
two sight lines. T4 CooltrainerM2 (13,5), Rhyhorn 43
(event 86): Blastoise took zero damage. T5
Blackbelt3 (10,1), Machoke/Machop/Machoke 38 (event
87): zero damage, and the XP levelled Blastoise
61 -> 62 (203 HP) before Giovanni. Payouts +2,455.
The other six were never engaged. NOTE for flag
readers: beating Giovanni runs the script's
SetEventRange TRAINER_0..7, so all eight trainer
flags read TRUE after the badge although only two
were fought (verified: all False at entry, all True
in badge.state, money consistent with two fights).
The same script sets EVENT_2ND_ROUTE22_RIVAL_BATTLE
and EVENT_ROUTE22_RIVAL_WANTS_BATTLE - the game has
armed the Route 22 rival for the League leg; we did
not go west.

GIOVANNI: compiled team (parties.asm $FF format)
Rhyhorn 45 / Dugtrio 42 / Nidoqueen 44 / Nidoking 45
/ Rhydon 50, plus the LoneMove FISSURE on mon index
4 (Rhydon, move slot 3; special_moves.asm entry 8).
The OHKO threat is theoretical against this lead:
Gen 1 OHKO moves fail when the user is slower than
the target, and Rhydon (Spd ~60) never outspeeds a
Lv62 Blastoise. Blastoise soloed: 203 -> 127 HP
(76 damage across the whole gym run), Hydro Pump
5/5 spent, Bubble x12, Water Gun untouched; no other
party member took a point. Payout +4,950. Ceremony
ridden to completion and verified from RAM: 8 badge
bits (Earth present), EVENT_BEAT_VIRIDIAN_GYM_
GIOVANNI (81), EVENT_GOT_TM27 (80), TM27 x1 in the
bag (20 stacks again). Zero blackouts - the
two-blackout stop rule never came into play.

EXIT + FINISH: out on foot (replanning navigator),
healed at the Viridian PC. Chain head probe162/
viridian.state (sha256 174661d2...345e): Viridian
City (23, 28), all full HP - Blastoise 62, Arcanine
39, Voltorb 16, Oddish 23, Lapras 15, Pidgey 3;
money 143,866 (+7,405 on the leg); dex 27; badges 8;
bag 20 stacks incl. TM27, Master Ball x1 (never
thrown), Great x3, Poke x8. PC storage now also
holds the Good Rod.

PORT: viridian_gym + _gym_go / _gym_step / _gym_bfs /
_gym_slide_end / _gym_sprite_cells / _gym_settle /
_gym_exit in run_real.py (the full leg: PC exit,
flight, deposit, gym, two-blackout stop rule, heal).
Tests: TestViridianLeg x7 (arrow table parsed back
out of pokered's script and compared trigger-for-
trigger; Giovanni's party parsed from parties.asm;
slide endpoints on the compiled grid incl. a
sprite-blocked lane; the door->stance road exists;
fence pins: no Route22 / VictoryRoad in the source,
deposit wired) - suite 164/164. FIDELITY: this leg
IS cold-replayed through the ported code - probe163
ran viridian_gym from probe160/fly.state and the
endpoint matched the live chain head on every field
(map/position (23,28), money 143,866 to the yen,
party species/levels/HP/moves/PP, bag stack-for-
stack, box, dex 27, 8 badges, TM27 held, Good Rod
in storage, all event flags).

OPEN DECISIONS (Vlad's, untouched): the League leg
is next per the roadmap - Route 22 rival (now armed
by Giovanni's script), Victory Road, the Elite Four
+ Champion - and MOLTRES sits inside Victory Road
in Gen 1, so its catch call is Vlad's to make
before that leg runs. Still open: Articuno; Safari
Tauros/Scyther + West; TM shelf / Rare Candy /
HP Up / Carbos; party arrangement (Pidgey in,
Diglett boxed).

## 26. The road to Indigo — STOPPED at the Route 22 rival (probe164)

Vlad's kickoff (2026-11-11) carried one explicit ruling:
Moltres is LEFT ALONE. The leg ran prep + the rival and
stopped under the standing stop rule — two genuine
blackouts to the rival. Route 23, Victory Road and the
Plateau were never entered; Moltres/Articuno untouched;
chain head unchanged (probe162/viridian.state).

Prep (verified, reusable): the bag held ZERO healing
items, and Viridian Mart sells no potions at all (Poke
Ball / Antidote / Parlyz Heal / Burn Heal — the Gen 1
quirk). So: deposited S.S. TICKET + HELIX FOSSIL +
SILPH SCOPE at Viridian PC (bag 20 -> 17, all
retrievable), flew to Fuchsia, bought SUPER POTION x6 +
REVIVE x2 + FULL HEAL x2 at Fuchsia Mart (clerk across
the counter from stance (2,5) — buy_mart_item's default;
stance (1,5) is the counter tile itself and pathing
refuses it), flew back. Total spend 8,400 (700x6 +
1500x2 + 600x2, to the yen). Checkpoint:
probe164/prepped.state sha256 5be2f20e...37b.

The rival (compiled from Rival2Data's Route 22 set,
Squirtle line — matches the brief's expectation except
Exeggcute is Growlithe in this starter row): PIDGEOT 47 /
RHYHORN 45 / GYARADOS 45 / GROWLITHE 47 / ALAKAZAM 50 /
VENUSAUR 53. Trigger cells (29,4)/(29,5); event 1318
(beat) / 1319 (wants battle, verified armed at start).

Two driver defects were found and fixed first, neither
a blackout (both runs aborted in-battle, party intact,
disclosed per the Snorlax precedent): (1) the shift
prompt — when a trainer's mon faints, _advance_battle's
A presses answer YES to "Will VLAD change POKEMON?" and
open the party list (top_y 1, in_dialog True), where A
re-selects the active mon forever; ONE B press declines
cleanly (proven live, probe164 dbg3). (2) A silent-spin
branch: with Venusaur asleep and Arcanine dead, the
policy ordered Water Gun from Oddish — a move it does
not know — failing silently for 215 turns (a
no-progress watchdog now guards the driver pattern).

The policy (PP economy: Hydro Pump capped 2/2/2,
Water Gun/Bubble otherwise, Arcanine on Bite duty vs
Growlithe, hard rule "Gyarados/Alakazam are Blastoise's
fights", Oddish Sleep Powder plan for Venusaur) then
lost twice, fairly: Gyarados is the tax — Dragon Rage
is a flat 40 and pump #1 missed in every run (80% acc),
so Blastoise reached Alakazam at 63 HP and died to one
Psychic on the switch turn; Arcanine (134 -> 67 on a
crit Flamethrower switching into Growlithe) died to the
next; the Lv3-23 cascade cannot hurt Alakazam (Voltorb
Tackle = 1 dmg). The near-miss run (different RNG
stream) reached Venusaur with Blastoise at 21 HP and
the sleep landed — the fight is a weighted coin flip
on crits and pump accuracy, not a clean win. Per the
stop rule the leg halts here; grinding or further
attempts are Vlad's call. Post-blackout state preserved
at probe164/dbg4-end.state (Viridian, healed, money
67,733 — halved twice from the post-prep 135,466 on the
attempt line; the chain line is untouched at 143,866).

No port this leg: route22_rival does not exist until a
policy actually beats him. Battle-driver lessons banked
above (shift-prompt B-decline; move-membership checks;
watchdog) belong in any future trainer-battle driver.

## 27. The road to Indigo — completed (2026-11-12/13, probe165, port + cold replay probe166): the Silph playbook again, and the switch that was a coordinate swap

Vlad's call (2026-11-12) was the Silph precedent,
verbatim: "Grind first, then re-fight him." Executed
from probe164/prepped.state (the S26 prep checkpoint:
Viridian, healed, the 8,400-yen stock in the bag); the
chain head before it (probe162/viridian.state,
143,866 yen) is untouched.

**The grind (probe165a/b).** Venue picked from compiled
pokered pools: Pokemon Mansion 1F (Koffing 30/32,
Ponyta 28-34, Growlithe 34, Grimer 30, Weezing 37,
Muk 39 — ~610 avg solo XP/fight, the richest unfought
pool reachable without passing the rival; the Power
Plant tops out lower per fight). Lead-and-solo,
per-level checkpoints, HP gate 55% / damaging-PP gate
10, free Cinnabar PC heal trips (no item spend; the
bought stock stays reserved for the road):
Blastoise 62 -> 66 in 73 fights, 1 heal trip;
Arcanine 39 -> 48 in ~97 fights over two runs (the
first died on a Mansion-exit bug at Lv45 — the street
door is the WEST room's south mats (4-7,27), not the
(26,27) strip door — resumed from checkpoint), 3 heal
trips on the resume. Zero blackouts, zero spend.
Lessons banked: a probe file named `inspect.py`
shadows the stdlib and poisons imports; `fly_to`
needs a `wait_free` settle after loading a state (the
probe166 boundary race, below).

**The re-fight (probe165c) — won on attempt 1.** The
two S26 driver fixes rode along (B-decline on the
shift prompt; moveset-aware cascade + watchdog), plus
the policy delta that mattered: **Alakazam is
Arcanine's fight.** The S26 hard rule switched
Blastoise in and fed it a switch-turn Psychic at
63 HP; at Lv48 Arcanine stays in and Bites (Alakazam's
Defense is paper): Bite 123 -> 56 -> KO, Arcanine
untouched. Gyarados: Hydro Pump cap 2 — both pumps
hit, Blastoise took ZERO damage-dealt back (the S26
tax simply never landed). Venusaur slept first try
(Oddish Sleep Powder) and took Ember x3 from
Arcanine. Payout +3,445 -> 138,911 yen. Side effect,
reported not chosen: Oddish participated, levelled
to 24 and evolved into Gloom on the road (dex
27 -> 28).

**Route 23 (probe165d).** Viridian heal, the gate,
all five badge guards passed at their object
coordinates, one surf stretch (12,103) -> (11,70),
ending in front of the VR door at (4,32).

**Victory Road (probe165e + the endgame drives).**
1F: the "switch" is a boulder switch — BOULDER1
(5,15) in 18 pushes onto (17,13) [event 2327]; the
opened gate block's passable row is its NORTH row.
2F switch 1: (4,14) west/south onto (1,16) [event
1336]; gate 1; the long loop to the (23,7) ladder.
Fight toll in VR: Blastoise levelled 66 -> 67 in
trainer fights and the typed driver burned Hydro
Pump to 0 PP; trainer payouts +2,545 -> 141,456 yen.

Then the routing problem that ate the middle of the
leg. The design, read from the scripts:
`Route23SetVictoryRoadBoulders` (runs on Route 23
entry) hides 2F's BOULDER3 and resets both 2F switch
events and the 3F hole event; 3F's hole handler is
what shows BOULDER3 again (HideObject on the 3F
boulder that falls, ShowObject on the 2F one, event
1638). So switch 2's boulder does not exist until a
3F boulder falls through the hole at (23,15) — the
hole drop is mandatory, a branch state that tried to
push BOULDER3 directly was chasing a ghost.

Reaching the hole took three unlocks, each a lesson:

1. **The push-aware save-state BFS lied.** A BFS over
   (player cell, boulder cells) from the 3F landing
   exhausted at 406 states and "proved" the south
   unreachable — its step edges silently dropped real
   steps (held-input drift / frozen-player loads), so
   the exhaustion was over a subgraph. The engine is
   the only oracle: a hand-driven column-12 walk
   (rows 6 -> 11) walked straight into the band's
   north strip. A BFS whose edges are sampled steps
   must verify its step model before its verdicts
   are believed.
2. **dbmapcoord stores (y, x).** 3F's switch is
   scripted as `dbmapcoord 3, 5`, long read as cell
   (3,5)-the-wall or as a red herring. The macro emits
   `db \2, \1` and CheckBoulderCoords compares
   (Y, X) — so the switch cell is (3,5) after all:
   ordinary floor in the NW room by the (2,0) warp.
   Its boulder is B1 (22,3), which rides the top band
   west (row 1, leapfrogging with rows 0/2 as the
   return lane), drops into the NW room and pushes
   east (2,5) -> (3,5). The gate (block (3,5) -> $1d,
   cells (6-7,10-11)) opens the west platform; the
   (17,5) tile-21 stairs are the platform massif's
   only other door (pair (5,42) seals its edges), and
   (1,9)'s matching stairs drop into the south band.
3. **Tile 5 is a one-way texture, not a wall.** The
   south band was "sealed" by three separate wrong
   reads — B2's alcove plug (its second push is
   genuinely refused, destination tile 5), the west
   platform's (1,6) -> (1,7) refusal (pair (5,32)),
   and gate rows — while the real path ran through
   two tile-21 stairs cells and the gate the B1
   switch opens. Pair-collision forbids (5,32),
   (5,42), (5,65), (5,33); stairs (21) pair with
   everything, which is exactly how the floors stack.

The drop itself: B4 pushed east from stance (21,15)
onto (23,15); the fall and the dungeon warp process
when the PLAYER steps onto the hole cell after it —
the player lands on 2F at (22,16) with BOULDER3
visible at (23,16), pushed west x14 onto switch 2
(9,16) [event 1343]. Exit via gate 2's north row,
ladder (25,14) -> 3F (27,15), the east column,
ladder (26,8) -> 2F pocket, door (29,7) -> Route 23
(14,31) — the door warp needed an off-and-back-on
step. **Moltres (2F (11,5)) was never engaged** —
the route never enters its room or its sight; the
whole VR traverse from the 3F landing ran with zero
battles and money constant at 141,456 yen.

**Indigo Plateau.** North edge from (18,32), lobby
entered — and `heal_at_pokecenter` FAILS there: its
standard-layout assumption does not fit the Plateau
lobby. The nurse (sprite at (7,5)) is talked to
across her counter from (7,7); healed manually,
party full HP/PP (Hydro Pump back to 5). New chain
head: probe165/indigo.state, sha256 54a0d6c5…f47 —
Indigo Plateau (9,6), Blastoise 67 / Arcanine 48 /
Voltorb 16 / Gloom 24 / Lapras 15 / Pidgey 3, money
141,456 yen, dex 28, badges 8, bag 20 stacks (Master
Ball never thrown; the prep stock untouched: Super
Potion x6, Revive x2, Full Heal x2). The leg ends
here by design: the League is the next leg, gated
on Vlad's team and item decisions.

**Port.** route22_prep, mansion_grind, route22_rival,
route23_passage and victory_road (plus _ride_menu,
_battle_enemy, _exit_building and the VR machinery)
are in run_real.py; the manual lobby heal is encoded
as driven, with the heal_at_pokecenter failure noted
in the docstring. Suite 174/174 (10 new tests:
pokered switch-coord parses for VR2F/VR3F, push-table
chain checks, fence pins). Cold replay (probe166, from probe162/viridian.state
through the ported functions only): prep spend exact
(8,400 yen); grind True (Blastoise 66 in 71 fights /
1 trip, Arcanine 48 in 104 fights / 10 trips, 0
blackouts); the rival fight won identically (event
1318 set, money 138,911 yen exact, party untouched) —
but the ported verdict first read the event flag
inside the fight and raced the post-battle script
that sets it during the exit scene; the verdict now
reads after the scene ride. Route 23 passage True,
endpoint exact. Victory Road traversed and healed to
a state field-identical to the probe chain head —
after the same class of race in the heal verification
(the nurse restores member by member) was fixed with
a settle-retry verify; the re-run returns True and
lands at Indigo Plateau (9,6), money 141,456 yen.
Two read-too-early races, both in verification reads,
both fixed in the port; no gameplay logic changed.

## 28. The League — COMPLETE (2026-11-13, probe167): the grind, the gauntlet, the Champion, and the species table that lied

Vlad's call: "Grind a proper League team first (Lapras
and a third attacker up), then run it." The storyline is
done: all five fights won, the Hall of Fame reached on
screen, and the game landed back at Pallet Town (5,6) —
chain head probe167/post-league.state, sha256
e7660ae15c7e237074afb2e1909ed6c96bd9b41f803df153db026f378f264352,
money 125,855 yen, dex 28, badges 8. Zero blackouts in
the gauntlet.

**Phase 1 — the grind (all Pokemon Mansion 1F, Cinnabar
PC heal loop, lead-and-solo, per-level moveset audits,
zero blackouts in every segment, 0 yen spent).** The
battle driver's stock form A-mashes learn prompts and
replaces slot 1 — it had already cost Bite in the early
game — so the grind ran on a prompt-aware driver: screen
text decoded from the tilemap, a tripwire on every A
press, and deliberate answers from a per-species plan.
LAPRAS 15 -> 50 (two stays: 116 fights for Lv19 -> 35
and 175 for Lv35 -> 50, plus the short Lv15 -> 19
opening): SING accepted over WATER GUN at 16 (deliberate
this time — it is the team's sleep source, because of
the discovery below), MIST declined at 20, BODY SLAM
over GROWL at 25, CONFUSE RAY declined at 31, ICE BEAM
over BODY SLAM at 38 (NOT over Strength — Strength is
the party's only copy and the Victory Road return still
needed it for boulders), HYDRO PUMP declined at 46.
Final: [Sing, Ice Beam, Surf, Strength], 214 HP.
SNORLAX 30 -> 48 (263 fights): BODY SLAM auto-appended
into the free slot at 35, HARDEN declined at 41,
DOUBLE-EDGE over HEADBUTT at 48. Final: [Double-Edge,
Amnesia, Rest, Body Slam]. ARCANINE 48 -> 52 (54
fights; its Gen 1 learnset is empty — zero prompt
risk). Blastoise stayed 67: PP, not levels, is his
constraint.

**Discovery — Gloom does NOT know Sleep Powder.** The
live moveset at Indigo read [Acid, PoisonPowder, Cut,
Stun Spore]: Sleep Powder was auto-replaced by Acid at
the Lv24 evolution level-up during probe165's rival
fight (the stock driver's slot-1 mash again). Gen 1 has
no move relearner; it cannot be restored. Mitigation
adopted: Lapras's Sing is the sleep source (55% vs Sleep
Powder's 75%); Gloom's Stun Spore remains as paralysis
utility. Disclosed to Vlad in the leg report.

**Phase 2 — assembly.** Two more corrections to the
brief's assumptions, both from pokered: Gen 1's Indigo
Plateau Lobby has NO PC (nurse, mart clerk, guide,
Cooltrainer, link receptionist — no PC object), so the
final six were assembled at the Viridian PC; and there
is no land route from Indigo Plateau back into Victory
Road 1F (the plateau ride lands on Route 23's north
strip; the VR door pockets are sealed from the east), so
Lapras's second grind stayed at the Mansion and the
final crossing was flown with Pidgey aboard, the bench
swap done at Viridian on the way back. Deposited at
Viridian: Poke Ball x8, Great Ball x3, Repel x2, Super
Rod, TM27, HM01, HM02 (bag 20 -> 13 stacks; the stat
items — HP Up, Rare Candy, Carbos — stayed untouched in
the bag, as did the Master Ball, never thrown). Bench
slot: Voltorb over Pidgey (40 HP death-switch vs 15,
Screech lottery; Fly is useless inside the gauntlet).
Final six: Blastoise 67, Lapras 50, Snorlax 48,
Arcanine 52, Gloom 24, Voltorb 16. Stock-up at the
Indigo mart, to the yen: Full Restore x8 (3,000),
Max Potion x4 (2,500), Revive +6 -> 8 (1,500), Full
Heal +4 -> 6 (600) = 45,400 yen; money 141,456 ->
96,056. Pre-League checkpoint: probe167/pre-league.state
(sha256 a70352fd...).

**Phase 3 — the gauntlet** (field-menu care between
fights; no mid-battle item path, per Vlad's standing
decline). LORELEI: Snorlax soloed all five — Amnesia x2
wall (her Dewgong Rested mid-fight), one Rest at 59/231,
Double-Edge finish vs her Lapras; Snorlax to 49; payout
+5,544 -> 101,600. BRUNO: Lapras Surf one-shot Onix #1,
Snorlax took the Hitmon tax (236 -> 62), Blastoise's
Water Gun economy covered the rest; Machamp Surfed out;
+5,742 -> 107,342. AGATHA: Lapras Surfed Gengar-56,
Blastoise Water-Gunned Golbat and Haunter, Lapras back
for Arbok and the Lv60 Gengar (Surf 151 -> 35 -> KO);
Lapras to 51; +5,940 -> 113,282. LANCE: the bloodbath —
Lapras, Blastoise AND Snorlax all fainted (Dragonite's
Hyper Beams); Arcanine's Bite closed the last Dragonite
from 105 HP; +6,138 -> 119,420; the corridor care spent
3 Revives and 4 Full Restores to put everyone back at
full. CHAMPION: the redo's lines won it in 23 turns —
Lapras Ice Beamed Pidgeot, Arcanine Bit Alakazam (never
let Lapras face it), Blastoise Water-Gunned Rhydon and
ground Gyarados down, Arcanine Bit Venusaur 142 -> 0
for the win while Blastoise and Lapras fainted;
+6,435 -> 125,855. Hall of Fame verified on screen
(media: hall-of-fame.png); landing Pallet Town (5,6).

**The species table that lied (the leg's costliest
bug).** The probe runner's species constants came from
a parse of pokered's pokemon_constants.asm that did not
count the const_skip slots (the MissingNos). Every id
after the first skips was low by the skip count —
Pidgeot 0x7C for the true 0x97, Alakazam 0x7A for 0x95,
Venusaur 0x7F for 0x9A, Dragonite likewise. Every policy
branch keyed on those species silently never fired, in
both Champion attempts and (via Dragonite) in Lance's
fight: Blastoise Water-Gunned a Dragonite it was
supposed to Hydro Pump, Lapras Surf'd an Alakazam she
should never have faced. The first Champion attempt
ended one mon from the win with the whole bench dead;
its post-mortem (a ('switch', None) decision then
crashed the runner on a format string) produced the
engine's targetless-switch coercion. run_real.py's
hand-written SP_ constants were always correct; the
port's regression test recounts the constants from
pokered WITH the skips and pins all 27 League ids.

**Driver defects found and fixed along the way** (all
in the port): corridor care saved a state with the bag
menu still open, wedging every later walk (menus eat
directional presses) — _close_all_menus now runs after
every item use and again before every room transition
(party_lead's flow can leave the START menu open the
same way); a 0-PP move selection beeps and wedges the
move list forever — the decide-level PP guard
substitutes or switches, and the send path verifies by
PP/HP deltas; a stray ride-A answered a shift prompt
YES once, opening the party list mid-fight — the ride
no longer A-presses while a two-option menu is live,
and a party list found with a living active mon is
cancelled as the accident it is; the executor's walker
routes its first E4-room step through the leader's
cell, so room movement uses a step-verified BFS
(_step_walk) on the model's grid (Agatha's (5,1) door
approach is unreachable behind her — (4,1) works; the
world model also carries twelve maps named LancesRoom,
ids 105-117 — the game's real room, live-verified, is
113, pinned in _LEAGUE_ROOM_IDS); the field-menu item
helper flakes intermittently with nothing consumed —
corridor care retries once; the PKMN column steer races
the menu opening animation — the switch opening retries
whole; the slot confirm A can be eaten — it is verified
against the list still being open and re-pressed; and
a gauntlet wipe cannot be detected by party HP (the
blackout warp HEALS the party) — elite_four declares a
blackout on the halved money and waking outside the
room just fought.

**Port.** league_grind (prompt-aware Mansion grind +
learn plans), league_assemble (Viridian deposits +
bench swap + ported Route 23/VR return + the 45,400-yen
stock-up) and elite_four (fight engine, policies,
corridor care, room machinery, ending ride) are in
run_real.py. Suite 180/180 (6 new tests: wired
callables, the const_skip species recount, learn-plan
final movesets, Champion policy lines on stub snaps,
room/pump tables, fence pins). **Cold replay**
(probe167r, from pre-league.state through the ported
elite_four only): four attempts. Attempts 1-2 died in
Bruno on the menu races above (Lorelei's payout exact
both times: 101,600). Attempt 3 cleared Lorelei, Bruno
(107,342) and Agatha (113,282) — all exact — then died
on the START-menu transition into Lance's room.
Attempt 4, with all fixes in: Lorelei, Bruno, Agatha
and Lance ALL beaten with payouts exact to the yen
(101,600 / 107,342 / 113,282 / 119,420), then the
Champion fight was LOST — a genuine wipe on that RNG
line (Venusaur left standing at ~48 HP with Voltorb
the last mon out), detected and reported by the new
blackout check. The Champion is a marginal fight: the
probe's line won it with three fainted and Arcanine at
50 HP; the replay's line lost it by roughly one mon.
No fifth attempt was rolled — re-rolling until a win
would be RNG shopping, not verification. The shipped
chain head remains the probe run's post-league.state.

**Left open in the save** (all Vlad's calls): Articuno
(never sighted) and Moltres (decided: leave alone) —
the birds stay unengaged; Safari leftovers (Tauros and
Scyther never caught, Safari West unreachable, the bait
anomaly of S20); the TM shelf (8 TMs in storage) and
the unspent stat items (Rare Candy, HP Up, Carbos — all
still in the bag); dex completion beyond 28; and the
mid-battle item path, declined three times and not
re-offered. The main storyline — eight badges, the
Elite Four, the Champion, the Hall of Fame — is
COMPLETE.

## 29. Post-game leg 1 (probe168, 2026-11-13 + resumed 2026-11-17): the dex audit, two failed Safari hunts, and the sweep that took the dex from 28 to 65

Vlad's call after the League: keep going — Pokédex
completion and the Safari leftovers. This leg was
INTERRUPTED: the coordinator process was killed by a
runtime interruption on 2026-11-13 at ~16:31 JST,
mid-sweep, and never reported. The leg was resumed on
2026-11-17 from its own on-disk artifacts — the audit
ledger, the hunt log, and `work.state` (verified from
RAM at the resume: Celadon Mansion roof, ¥112,355,
dex 29) — not restarted. What follows is the full
account of both halves.

### The audit (Phase 1, pre-interruption)

`agent/dex_audit.py` (ported this leg from the probe's
`audit_data.py`; its output is byte-identical to the
probe's `ledger.json`) compiles every species'
reachability from pokered sources — encounter tables,
rod groups, prizes, trades, gifts — and closes over
level/item evolutions (trade evolutions deliberately
do not propagate). Verdict for a solo Red cartridge
with this run's one-way choices already spent
(Squirtle starter, Helix fossil, Hitmonlee):
**ceiling 124** = 123 obtainable + 1 of the 3
choice-locked Eeveelutions. The 25 unobtainable:
11 Blue exclusives (incl. the Pinsir/Magmar lines'
bases), 4 trade evolutions counted as evolutions,
the Kabuto line, Hitmonchan, Mew, and the 6 unchosen
starter-line species. Anomalies: none — every
unobtainable species has a stated reason. The full
ledger lives in the goal workspace
(`hidden_files/dex-ledger.json`).

### Safari hunt 1 (pre-interruption, 6 entries, ¥3,000) — CLOSED, failed

West WAS reached (the §28 "unreachable" note is
superseded). 40 encounters logged. **Tauros: 4
sightings, all fled. Scyther: 4 sightings, all fled.
Neither caught.** The bait A/B experiment completed:
**unbaited 11/23 caught vs baited 5/17 — unbaited
wins**; the target arm was set to U for the rest of
the programme. One new species fell out of the hunt:
Nidorina (dex 28→29). Per-target detail: each of the
8 target sightings took exactly 1 ball before the
mon fled (1 of Tauros's and 1 of Scyther's were
baited-arm; the rest unbaited).

### The sweep (resumed 2026-11-17) — dex 29 → 65

In chain order, each step RAM-verified with a named
checkpoint state in probe168:

- **Celadon (s2-celadon):** Eevee collected from the
  mansion roof (dex 30) — collected, NOT evolved; the
  either/or is Vlad's. Gloom→Vileplume and a boxed
  Exeggcute→Exeggutor by Leaf Stone; Slowpoke from
  the Celadon pond (Good Rod).
- **North loop (s2b-karp → s3-north):** the ¥500
  Magikarp bought; Mt. Moon yielded Zubat, Geodude,
  Clefairy and the 1F Moon Stone ball; Cerulean Gym
  water (Super Rod group 3) yielded Goldeen and
  Psyduck; the Pewter Museum back room gave the Old
  Amber; Route 3 gave Jigglypuff; Viridian Forest
  gave Caterpie and Metapod.
- **Cinnabar (s4a-omanyte → s4b-cinnabar):** Helix
  Fossil revived to Omanyte, Old Amber revived to
  Aerodactyl; the south-beach Super Rod stance
  (10,13) produced Poliwag (Good Rod), Staryu,
  Horsea and Shellder; Water Stones closed Starmie
  and Cloyster.
- **The grind (s5-grind):** switch-training at the
  Pokémon Mansion under the prompt-aware driver —
  Butterfree, Gyarados, Pidgeotto, Poliwhirl,
  Poliwrath (the third Water Stone), Kakuna,
  Beedrill, Raticate, Fearow, Electrode, Dugtrio,
  Dragonair, Dodrio, Venomoth, and Nidoking by Moon
  Stone. Final dex **65**.

Spends, to the yen. Leg start ¥125,855.
Pre-interruption: hunt 1 fees ¥3,000 + stones
(2 Leaf + 3 Water @ ¥2,100) ¥10,500 = ¥13,500
(balance at resume ¥112,355 ✓). This window: Poké
Balls ¥4,000 ×3 (Cerulean restocks), Magikarp ¥500,
trainer income +¥275, hunt 2 fees ¥2,500 (see
below). **Final money ¥97,630** — every figure
reconciles against the stage dumps.

### Safari hunt 2 (2026-11-17, 6 more entries authorised by Vlad) — CLOSED, failed

Policy per the authorisation and the hunt-1 verdict:
unbaited ball-only throws, rocks holstered, owned
dupes fled, stop early only if BOTH targets fell.
84 encounters logged. **Scyther: 4 sightings, 37
balls thrown (1 + 12 + 12 + 12 — three encounters
ran to the per-encounter ball cap), all fled.
Tauros: 0 sightings. Neither caught.** The budget
is spent; a further authorisation is Vlad's call.
Two honest complications: (1) entry 5 was cut short
by a service restart on the VM — its partial
encounters are in `hunt2_log.json`, but its ¥500 fee
died with the process, so the surviving chain paid
5 fees (¥2,500) for 6 entry sessions; no seventh
session was run to "make it up". (2) Across both
hunts the combined target record is Tauros 4
sightings / 4 balls and Scyther 8 sightings / 41
balls, zero catches — at a 4.3% encounter rate and
Gen 1 Safari flee mechanics, these two are a
long-tail grind, and the ledger notes Scyther has
an alternative source: Game Corner prize, 5,500
coins.

### What the sweep taught (all live-verified)

- **The nickname screen is a trap.** Fossil revival
  (any GivePokemon) asks "give a nickname?" AFTER the
  box write; A-mashing answers YES and lands on the
  nickname keyboard, which is not a dialog — every
  movement executor then spins forever. Answer NO
  with B presses until the dialog clears. The same
  root cause explained the pre-interruption "missing
  Eevee" mystery: `_GivePokemon` sets the dex bit
  BEFORE the box write lands, so a dex-only
  completion check snapshots mid-gift.
- **Evolution B-cancel.** The guarded advance's
  faint-B (skip the "hurt by poison" text) keeps
  landing while `menu_top_y` is stale at 12 through
  the EndOfBattle evolution window — and B there
  CANCELS the evolution. The grind variant presses B
  only while the enemy is alive. Two grinds first
  "succeeded" with zero evolutions before this was
  understood; their checkpoints were poisoned and
  deleted.
- **The learn prompt is a state, not a text.** It
  fires inside the battle-end sequence; once the
  YES/NO box replaces the prompt text, a text watcher
  is blind and the next A forgets slot 0 (the §28
  Sleep Powder pattern, repeated on Venonat's Lv30
  Stun Spore before the fix). The shipped classifier
  (`_learn_prompt_state` in run_real, ported this
  leg) reads screen text + menu bytes and names four
  states — decline / confirm-abandon (declining is
  TWO menus) / cancel-forget / evolving — and every
  advance step in the grind checks it before pressing
  anything. Venonat's Lv30 audit then passed with
  Tackle intact.
- **Learnset audits are engine-exact.** One learnset
  check per level-up, by the pre-evolution form, at
  exactly the final level gained (multi-level gains
  skip intermediate levels permanently — Pidgey lost
  its Lv5 Sand-Attack this way); an evolution at
  level L triggers a second check by the new form at
  exactly L (Gyarados learned Bite on evolving).
- **Disable can lock the driver's favourite move.**
  A wild Muk disabled Water Gun — `_last_damaging_move`'s
  pick — and the finish loop waited forever at the
  move list. `_pick_battle_move` (ported) excludes
  `wPlayerDisabledMove` (0xD06D). Two sibling stalls
  fell out of the same grind: the Gen 1 battle-menu
  cursor PERSISTS across turns (steer it back to
  FIGHT instead of pressing A onto ITEM), and a stray
  A can open the bag mid-battle (B out; Oak's "not
  the time" text loops on A forever).
- Mart inventories differ: Cinnabar sells no Poké
  Balls. Encounter deserts are real (MtMoonB1F's
  ladder room ~1 battle/600 paced steps vs ~19/300
  on the main floor). The executor's planner can
  spin at 99% CPU on some building exits — kill and
  reroute beats waiting.

### Port + state

Ported into the repo this leg: `agent/dex_audit.py`
(ledger builder, output verified identical),
`_pick_battle_move` / `_disabled_move_id` /
`_learn_prompt_state` in run_real.py, and 14 tests
(suite 223 → **237, all green**). The probe-side
machinery that did not generalise (the learned-graph
`live_nav`, the mansion DFS router, `pace_catch`,
the PC box-rotation helpers) stays in probe168 with
this section as its documentation.

**Final chain head: `probe168/s6-hunt2-e6.state`,
sha256 f2d0554204eda9f1fab5299e9478e4cf87089982a762c7aade1ff2fa58c9d840**
— Fuchsia Pokécenter, dex 65, money ¥97,630, badges
8. Eevee sits unevolved in box 3. Unspent as ordered:
Master Ball, Rare Candy, HP Up, Carbos; no TMs
taught; no new Safari entries beyond the two
authorised budgets.

### Left for leg 2 (58 obtainable + the Eevee choice)

Easy: Pidgeot (L36 grind), Ekans, Mankey→Primeape,
Abra, Machop, Tentacool→Tentacruel, Gastly, Drowzee,
Onix, Cubone, the Mansion commons (Koffing, Ponyta,
Grimer), Krabby (Cerulean group 3), the three
in-game trades (Farfetch'd, Mr. Mime, Jynx —
counterpart species owned or trivially re-caught),
Omastar (L40), and ONE of Clefable/Nidoqueen (one
Moon Stone remains). Grindy: Dragonite (L55),
Rapidash, Haunter. Location blocks: Cerulean Cave
(Arbok, Raichu, Golbat, Kadabra, Magneton, Hypno,
Ditto, Wigglytuff, Marowak, Rhydon), Seafoam wilds
(Golduck, Slowbro, Seel, Dewgong, Seadra — the
islands' wilds are not fenced, only Articuno is),
Power Plant (Pikachu, Magnemite, Electabuzz, and
the Zapdos static — deferred, not fenced).
Gated behind Vlad's decisions: the Eeveelution;
Porygon (9,999 coins) and Scyther's prize
alternative (5,500 coins); the second Moon Stone
for whichever of Clefable/Nidoqueen loses the first;
any further Safari entries for Tauros/Scyther; and
the fenced three — Articuno, Moltres, Mewtwo —
which stay unengaged unless Vlad says otherwise.

## 30. Post-game leg 2 (probe169, 2026-11-18): the easy block, the Moon Stone pair, Jolteon by Vlad's call, and Cerulean Cave — dex 65 → 98

Vlad's call (2026-11-18): "Kick off leg 2 — the easy
block + Cerulean Cave". Scope was exactly the "Easy"
list and the "Cerulean Cave" block of
`leg2-remaining.md`; the Grindy list, Seafoam, the
Power Plant, Game Corner prizes, Safari entries and
all TMs stayed out of scope, and the fenced birds +
Mewtwo stayed unengaged. Mid-leg Vlad settled the
Eevee either/or himself — "Jolteon — Thunder Stone
it" — and that one evolution was folded in. Chain:
`probe168/s6-hunt2-e6.state` (dex 65, ¥97,630) →
`probe169/s9-ditto.state` (sha256 131cd7f2…c71f,
dex 98, ¥645, on Cerulean Cave B1F). Every catch,
evolution and trade below was verified in RAM; saves
were taken per stage and never overwritten.

### The easy block (S0–S8): dex 65 → 87

Prep (S0): current box moved 3 → 9 (box 3 was full);
Ultra Balls ×30 + Greats topped to 20 for ¥46,800.

- KRABBY (S1) — Cerulean City's outdoor pond stance is
  unreachable by the compiled navigation (the grid
  marks the shore unwalkable and the DFS dies on the
  (9,14)→(9,16) ledge slide), so it was fished from
  the Cerulean GYM pool instead (Super Rod group 3 —
  same pool, probe168 precedent). First cast.
- ABRA ×2 (S1) — Route 24 grass pair. One is trade
  fuel, one stays (the chain rule: never trade the
  last specimen of a chain with unregistered members).
- MANKEY (S2a) — Route 5's grass is sealed inside
  one-way ledge enclosures the navigation cannot
  enter; Route 6's Red table is slot-identical
  (pokered-verified), so Mankey came from Route 6.
- MR. MIME (S2c) — Route 2 house, TRADE_FOR_MARCEL,
  gave 1 Abra. The trade driver (`_do_trade`) polls
  the screen text until stable — trade dialogs render
  over ~240 ticks and acting early desyncs the flow.
- EKANS + POLIWHIRL (S4a) — Route 10 grass / Super
  Rod group 5. Poliwhirl was already registered in
  leg 1; this specimen was the Jynx fuel.
- MACHOP + ONIX (S4b) — Rock Tunnel 1F. Onix, the
  block's 4.3% long tail, arrived at encounter 33 of
  the same pacing session as Machop (enc 7). Not
  parked.
- FARFETCH'D (S3) — Vermilion house, gave a Spearow
  caught on Route 11 (Spearow was already registered;
  pure fuel). DROWZEE from the same Route 11 pair.
- JYNX (S4b) — Cerulean house, TRADE_FOR_LOLA, gave
  the Route 10 Poliwhirl.
- GASTLY + CUBONE (S5) — Pokémon Tower. **The Silph
  Scope was sitting in PC item storage**: without it
  the Tower ghosts stay unidentified "GHOST" and one
  throw call steered onto HM03/Oak's scolding text
  and burned all 15 Great Balls. Withdraw the scope
  before any Tower work. Restock before the Tower:
  ¥33,000 at Fuchsia.
- TENTACOOL (S6) — Route 21 surf (100% Tentacool
  water). Docking back into Cinnabar uses the west
  water strip, not the planner.
- PONYTA / KOFFING / GRIMER (S7a) — Pokémon Mansion
  1F entrance corridor pair (8,26)/(9,26); Weezing
  and Muk were fled (Grindy list, out of scope).
- Grinds (S7b/c, `switch_grind_one`, standing
  DECLINE on every move replacement, per-level
  moveset audits all clean): PIDGEOTTO 18→36
  PIDGEOT (125 fights), MANKEY 28 PRIMEAPE (76),
  TENTACOOL 30 TENTACRUEL (111), OMANYTE 40 OMASTAR
  (129). The grinder requires the pupil to LEAD the
  party; calling it from inside a PC fails silently.

### The Moon Stone pair (S2c + S8): both evolutions done

**The brief's "one MOON_STONE remains in the bag"
was wrong** — verified absent from both the bag and
PC item storage at leg start. Both stones used were
uncollected world pickups, checked against pokered
event flags: Route 2 at (13,54), flag idx 25 (picked
up S2c), and Rocket Hideout B2F at (1,11), flag idx
125 (S8a; the B2F arrow floor defeats `_dfs` and
`_walk_fighting` — `explore_nav` is arrow-aware and
walked it cleanly). Route 2's stone → CLEFABLE;
the Hideout stone → NIDOQUEEN. (The third stone,
Mt. Moon 1F idx 104, was leg 1's Nidoking stone.)
Save immediately after one-shot pickups and
evolutions, BEFORE the heal/exit steps — a heal
assert cost two runs this leg.

### Jolteon (S10): Vlad's mid-leg decision

Eevee (box 3) → JOLTEON. Thunder Stone verified
from pokered before travelling: CeladonMart4FClerk
sells it, price ¥2,100 (`bcd3 2100`). Funds were
¥395 after the cave restock, so the stone was
funded by selling re-buyable vendor stock (1× FULL
RESTORE ¥1,500 + 1× MAX POTION ¥1,250 at Cerulean
Mart — the sell flow is the buy flow mirrored:
menuY 1, steer to SELL, bag index, quantity/confirm,
verify on the money delta). The fenced items were
never sold. Stone bought, Eevee withdrawn from
box 3, evolved, dex 89 → 90, screenshot taken, and
the party restored exactly (Jolteon filed in
box 10). Vaporeon/Flareon are now final as
unobtainable-by-choice.

### Cerulean Cave (S9–S16): dex 87 → 98, all ten

Encounter tables compiled from pokered per floor.
Entry is hand-driven — planner rides to
CeruleanCave1F spin forever from every origin (the
surf/portal leg): fly Cerulean → Route 24 → walk
(4,31) → surf the river down → the pond channel to
(6,12) → disembark (5,12) → door at (4,11).

- 1F: GOLBAT (enc 1), HYPNO, MAGNETON, ARBOK,
  KADABRA (no Teleport escape), RAICHU.
- 2F: RHYDON, MAROWAK, WIGGLYTUFF.
- B1F: DITTO. **Mewtwo was never engaged** — B1F
  work stayed on the west side (pacing pairs at
  x ≤ 22; the static sprite at (27,13) was never
  approached or interacted with).

The "Magneton curse": after Golbat filled the last
slot of box 9 (20/20), every catch was refused with
"The POKéMON BOX is full!" — no ball consumed, the
post-throw ride misread the refusal as "broke free",
and hours of desync theories followed (menu scroll
aliasing, stale species bytes — the stale-byte read
WAS real and is now double-read in `_hunt_paired`).
**Check box headroom before any hunt; a full box
fails silently.** Box switched to the empty box 10
and the curse lifted. Also from this floor: the
enemy species byte is stale during battle intros.

Surf inside the cave is pair-collision gated
(pokered `TilePairCollisionsWater`): in the CAVERN
tileset, standing on tile $05 facing $14 water is
refused ("No SURFing on LAPRAS here!") — the whole
north shore of 1F is $05. The only boardable
entrance-side cell is (25,9) ($15, facing down); the
harness `field_surf` macro also completes without
taking in the cave, so `_board_surf` drives the
party menu by its cursor bytes. B1F connects ONLY
to 1F (0,6), and the direct floor route to it is
walled by the CAVERN $20/$05 land pair-collision;
the working route is a ladder loop — surf to 1F's
west floor (landing (11,13)), (3,11) up to 2F, west
across 2F, (1,3) down onto 1F's $20 platform, (0,7),
up onto (0,6). The planner and the mansion DFS both
refuse these shore/platform regions; `_press_walk`
BFS-walks the raw compiled grid with verified
single steps, and `_step_warp` tolerates battles
eating a warp step.

The cave's ball economy drove two restock detours.
The 1F Nugget (5,0) sold for ¥5,000; surplus Revives
and Super Potions went too (S11: +¥10,850, Ultras
topped to 27 for ¥10,800). The 11-ball Rhydon fight
left the party with three faints (the next hunt
then hung in a battle the catch driver couldn't
advance) — heal trips are cheap insurance (S12).
For the endgame the bag was slimmed at the PC
(S13: heals/rods/coin case deposited; the deposited
MAX POTIONs and FULL HEALs were later withdrawn and
sold in S15a, +¥5,550 → 5 more Ultras for ¥6,000),
because the careful thrower's worst case grows with
bag depth (Ultra at index 12 made single throws
take minutes). The typed driver's flee can also
churn for minutes against cave mons; `_flee_manual`
is fully bounded and replaced it in `_hunt_paired`.

DITTO (rate 35, the cave's lowest) took five
encounters across four runs. The Sing detour was a
dead end twice over: the FIGHT move menu remembers
its cursor in BOTH axes (it opened on the bottom
row from Lapras's last-used move, so "Sing" was
Ice Beam until the column was steered too), and a
transformed Ditto can Sing back. The plain catch —
Blastoise leading, careful thrower, no switching —
took one ball in the spy debug and one attempt in
the real run (enc 8). Lesson: debug the thrower
against the actual target before inventing tactics.

### Spends (to the yen)

Start ¥97,630 → end ¥645. Purchases ¥116,700:
S0 balls ¥46,800; S5 restock ¥33,000; S9 restock
¥18,000; Thunder Stone ¥2,100; S11 Ultra top-up
¥10,800; S15a Ultra top-up ¥6,000. Sales income
¥19,150: FULL RESTORE ¥1,500 + MAX POTION ¥1,250
(S10); Nugget ¥5,000 + 5× REVIVE ¥3,750 + 6× SUPER
POTION ¥2,100 (S11); 3× MAX POTION ¥3,750 + 6×
FULL HEAL ¥1,800 (S15a). Trainer winnings observed
in RAM during travel: +¥565. Ledger balances
exactly. Final balls: Ultra 5, Great 3, Poké 0
(deposited), Master Ball 1 — never thrown. Rare
Candy / HP Up / Carbos unspent, still in the bag.
In PC item storage at leg end: COIN CASE, SUPER
ROD, GOOD ROD, POKE BALL ×3 (deposited S13).

### Port and what remains

Ported into `run_real.py` (suite 237/237 green):
`_specimen_count`, `_npc_cells`, `_do_trade`,
`_flee_manual`, `_throw_ball_careful`,
`_catch_current_careful`, `_hunt_paired`,
`_board_surf`, `_press_walk`, `_step_warp`,
`_SPECIES_NAMES_169`, with the cave-route and
box-headroom lessons in the section header.

Dex 98 / 124. Remaining (leg 3): the Grindy list
(Dragonite L55, Rapidash, Haunter, Weezing /
Machoke / Graveler), Seafoam wilds (Golduck,
Slowbro, Seel, Dewgong, Seadra — Articuno still
fenced), Power Plant (Pikachu, Magnemite,
Electabuzz; Zapdos deferred, not fenced), and the
decision-gated items (Porygon / Scyther prizes,
further Safari entries for Tauros/Scyther). The
fenced three — Articuno, Moltres, Mewtwo — stay
unengaged unless Vlad says otherwise.

## 31. Post-game leg 3 (probe170, 2026-11-18): the grindy set, Seafoam, the Power Plant — dex 98 → 112, and the funding saga that nearly ate the leg

Vlad's call (2026-11-18): "Kick off leg 3 — grindy
set + Seafoam + Power Plant wilds". Scope was
exactly the Grindy section + Seafoam wilds + Power
Plant wilds of `leg2-remaining.md`. Articuno stayed
fenced (Seafoam B4F never entered), Zapdos stayed
out of scope (deferred, not fenced — never
approached or interacted with), Mewtwo and Moltres
unengaged, no Safari entries, no Game Corner, no TM
teaching, Rare Candy / HP Up / Carbos unspent,
Master Ball still never thrown. Chain:
`probe169/s9-ditto.state` (dex 98, ¥645, Cerulean
Cave B1F) → `probe170/s5-done.state` (sha256
8e3ab527…c94f39dffc0bb0, dex 112, ¥2,011, Fuchsia
City). Every catch and evolution below was verified
in RAM; saves were taken per stage and never
overwritten.

### S0 — out of the cave, box prep

The B1F exit rode the s15a ladder/surf route (the
Max Revive at (18,1) was collected on the way out —
it funds S1; the Ultra Ball at (16,9) was skipped:
its south ledge pocket's return crossing defeats
the tile-BFS walker, v1 died at (9,7)). Clefable
was deposited and the current box switched 10 → 4
(empty) so the leg's catches and deposits had
headroom (§30's full-box lesson applied up front).

### S1 — funding: the Elite Four money tap that wouldn't

The leg started with ¥645 against a restock that
needed ~¥30k of balls and care. The classic Gen 1
money tap — an Elite Four re-run — turned into the
leg's longest fight, because the stock gauntlet
driver had a real defect: `_league_menu_move`
hardcodes switch fallbacks that never check the
target is alive (lance: Blastoise vs Gyarados →
"Lapras if alive else Snorlax"; champion and bruno
have the same shape). With the fallback fainted,
the party list refuses the switch, the
accidental-list cancel fires, and the fight
re-decides the identical switch until the turn cap
— S1 died at Bruno on exactly this loop, and S1c
died at Lance on its twin. §28's win had simply
never had the fallback fainted at those moments.

The fix (ported, see below): `_league_guard_switch`
validates every menu-path switch decision in
`_league_decide` — a non-viable target is replaced
by the best living non-active mon via `_league_pick`,
or by the active mon's strongest PP'd damaging move.

With the guard in, the runs went deep but the
Champion kept winning the war of attrition (the
corridor-care chain can only Revive + top up between
rooms; there is no mid-battle item path, by the
player's standing choice). The run log, honestly:

- S1 (unguarded): died at Bruno, switch loop.
  Nothing saved. S1b: died in a mart approach.
- S1c → S1e (bootstrap, ¥145 after selling the cave
  Max Revive +¥2,500 and buying Revive ×2 −¥3,000):
  Lorelei→Lance paid ¥23,364 gross; Champion
  blackout halved ¥23,509 → **¥11,754 banked**.
- S1f (care basket −¥10,000: Revive ×4, Super
  Potion ×4, Full Heal ×2): payouts again ¥23,364;
  Champion blackout → **¥12,559 banked**.
- S1g (side branch: Full Restore ×3 from the Indigo
  lobby's own mart — its list is ULTRA_BALL,
  GREAT_BALL, FULL_RESTORE, MAX_POTION, FULL_HEAL,
  REVIVE, MAX_REPEL, clerk at (0,5) behind the (1,5)
  counter, standard stand (2,5)): payouts ¥23,364;
  Champion blackout → ¥13,461 (a side branch, not an
  ancestor of the final head).
- S1h (canonical line, Full Restore ×4): Lorelei
  and Bruno banked (¥11,845), then the Agatha fight
  ABORTED on a party-menu open failure with the
  battle live (Gengar vs a sleeping, confused
  Lapras at 52/245) — saved mid-battle.
- S1j resumed that exact battle with a hand-mirrored
  elite_four loop tail (the stock loop can only
  start from the lobby): Agatha and Lance beaten
  (¥23,923), Champion blackout number five →
  **¥11,961 banked**.

Five Champion blackouts against §28's single win
settled it: the gauntlet is a coin flip weighted by
attrition, and each cycle's halving eats the gross.
Funding closed lean instead (S2).

### S2 — the lean restock

Sell the 34 Poké Balls, buy Ultra Balls to 14 and
Great Balls to 10 at Fuchsia Mart: ¥11,961 → ¥361,
which the hunts then lived within (below). Two
mart lessons: the SELL quantity selector lives in
the menuY-4 list row ("POKé BALL ×01 ¥100"), not at
menuY 8 — the first sell_many waited for a stage
that never comes, its blind A-ride wandered into
BUY and spent ¥11,000 in-memory; the process died
on its assert before any save, so the accident had
zero persistent effect. The fixed sell_qty steers
the quantity with UP taps inside the list row,
verifies the ×34 readout on screen, confirms once,
and rides only dialog text.

### S3 — the grindy set (Mansion switch-grind)

Pupils switch-trained at Pokémon Mansion 1F
(Blastoise the fighter, prompt-aware driver,
DECLINE every move replacement, per-level moveset
audits — all audits passed):

- HAUNTER — Gastly L20→25 (box 9), 21 fights after
  one wedged attempt was fled. Dex 99.
- WEEZING — Koffing L30→35, 53 fights. Dex 100.
- MACHOKE — Machop L15→28, 54 fights. Dex 101.
- RAPIDASH — Ponyta L34→40, 82 fights. Dex 102.
- GRAVELER — Geodude L8→25 (box 3), 41 fights.
  Dex 103.
- DRAGONITE — Dragonair L30→55 (box 1), in level
  bands (35/40/43/46/49/52/55) with a save per band:
  67+95+65+76+94+99+112 ≈ 608 fights. Dex 104.
  (Machoke / Graveler / Haunter stop at the
  mid-stage by design — their finals are
  trade-locked.)

The grind machinery had its own wedge:
`_finish_fight` (probe168/sweep_lib.py) steers the
battle menu back to FIGHT with `_steer_byte` on the
COLUMN byte — but _steer_byte presses up/down,
which toggles the ROW byte, so from the
post-switch PKMN cell (0,15) the two steers fight
forever and the fight dies on its 120 s cap (a
Mansion Grimer's Disable on the chosen move is the
other trigger: the loop re-selects the disabled
move's row and eats the "disabled!" text forever).
The durable pattern is flee-and-retry: when a grind
attempt aborts mid-battle, flee the wedged battle
(B out to the main menu, then `_run_away`) and
re-enter the grind fresh — every pupil completed
within two attempts of any wedge. The column steer
itself is fixed in sweep_lib (left/right presses).
Blastoise went L84 → L93 across the leg's fighting.

Mid-band-45 the S3d process was killed externally
(no traceback; VM memory pressure) — the banded
saves bounded the loss to three levels and S3d2
resumed from the band-40 save.

### S4 — Seafoam wilds (Articuno fenced, B4F never entered)

Travel is the grand tour's proven legs
(probe160a2): exit Fuchsia PC → `surf_to` Route 19
→ `surf_to` Route 20 → `_surf_walk_to` the west
door (48,5) → 1F landing (4,17). Hand-rolled
beelines all wedged on a swimmer's spot-dialog at
Route 20 (15,5) — in_dialog set, no battle, no
movement; surf_to's surf_path routing simply does
not hit it. Catches were Sing-assisted (Lapras
leads; one Sing attempt per target encounter via
`_use_move_at_menu`, then the careful thrower) to
fit the ball budget:

- SEEL L30 — 1F pace cells (5,8)/(6,8), Great
  Balls (2 spent on the floor). Dex 105.
- GOLDUCK L38 — 1F, the table's last slot; Ultra
  ×2. Dex 106.
- DEWGONG L38 + SEADRA L37 — B1F (4,12)/(5,12) via
  the (7,5) ladder; Great ×5 for the pair.
  Dex 107, 108.
- SLOWBRO — B2F (3,3)/(4,3) via the B1F (4,2)
  ladder; Great ×1. Dex 109.

Exit reversed the ladders and surf_to'd home.

### S5 — Power Plant wilds (Zapdos untouched)

Route prep produced the leg's worst discovery:
**Gloom and Voltorb are lost.** The Cerulean east
exit is blocked by the Cut tree at (32,13) (tile
0x55, measured live), and the team's only Cut user
was Gloom. A full census — the in-game PC withdraw
lists for all twelve boxes, plus the party — has
neither Gloom nor Voltorb anywhere. They were in
the party at leg 1's start (probe168/held.json:
Gloom L24, Voltorb L16) and were lost during leg
1/2's heavy box reorganization; the Pokédex owned
flags are unaffected (both species still count in
the 112), but the specimens — and with them the
team's only Cut and Flash users — are gone. No
fence was crossed to replace them (teaching an HM
to a new mon is a permanent-decision class the leg
was not authorised to spend).

The plant was therefore reached the long way:
Fly Lavender → ride Route 10 south segment →
Rock Tunnel in reverse (dark — `_walk_tile_leg`
ladder chain (37,17)→(23,11)→(5,3)→(33,25)→exit
(15,3), the ported traverse's own legs without its
Flash/Repel gates) → Route 10 north segment →
`_surf_walk_to` from the bank at (13,28) down the
river (the channel is x=16–17; surf_path covers
the whole leg, (13,28)→door = 105 steps) → the
plant door (6,39). The hunts paced at (12,28)/
(13,28) in the south-central room; Zapdos's room
was never entered and its sprite never approached;
no item ball was touched:

- MAGNEMITE + PIKACHU — the 2 remaining Great
  Balls could not carry two species (and a
  ball-less Sing attempt per encounter made the
  Great hunt crawl at ~90 s/encounter), so the
  commons were hunted with Ultra Balls directly:
  ×6 for the pair. Dex 110, 111.
- ELECTABUZZ — Ultra ×1 (asleep). Dex 112.

The S5b process was killed by a service restart
mid-hunt; the hunts were restarted from the
plant-entry save (S5c/S5d) with no loss beyond
time.

### Money trail (to the yen)

¥645 start → Max Revive sale +2,500 → Revive ×2
−3,000 → ¥145 → E4 payouts +23,364, Champion
blackout → ¥11,754 → care basket −10,000 → ¥1,754
→ E4 payouts +23,364, Champion blackout → ¥12,559
→ Full Restore ×4 −12,000 → ¥559 → Lorelei + Bruno
+11,286 → ¥11,845 → resumed Agatha + Lance
+12,078 → ¥23,923 → Champion blackout → ¥11,961 →
Poké Ball ×34 sale +3,400 → ¥15,361 → Ultra ×9
−10,800 → ¥4,561 → Great ×7 −4,200 → ¥361 →
trainer winnings on the S4/S5 travel legs +1,650 →
**¥2,011 final**. Balls: Ultra 5 +9 −2 (Golduck)
−6 (plant commons) −1 (Electabuzz) = ×5; Great
3 +7 −2 (Seel) −5 (B1F pair) −1 (Slowbro) = ×2.
Final party: Lapras 58, Blastoise 93, Pidgeot 36,
Seel 30, Golduck 38, Dewgong 38 (Seadra, Slowbro,
Pikachu, Magnemite, Electabuzz and the evolved
pupils are in box 4; Dragonite in box 4).

### Port and what remains

Ported into `run_real.py` (suite 241/241 green,
+4 tests): `_league_guard_switch` — validates
menu-path switch targets in `_league_decide`
(substitutes `_league_pick`'s best living
non-active mon, else attacks with the strongest
PP'd damaging move). Probe-side fixes kept with
the probe libs: sweep_lib's battle-menu column
steer (left/right), the flee-and-retry grind
wrapper, the Sing-assisted catch wrapper, the
reverse Rock Tunnel ladder chain. One tooling
caveat for future legs: the probe-side .state SRAM
box scanner (probe170/scan_boxes.py, MBC-register
poking) reads bank-3 boxes ≥10 as empty — the
in-game PC enumeration is the authoritative census
(it found box 10 intact when the scanner showed
zero).

Dex 112 / 124. Remaining (leg 4): twelve
obtainable species. Free to take: MUK (Grimer
L38 grind — a Grimer L30 sits in box 9; also a
1.2% Mansion 1F wild), KINGLER (Krabby L28 grind,
box 9), SEAKING (Goldeen L33 grind, box 3),
TANGELA (Route 21 grass, 5.1% — a wild catch, not
a trade). Vlad-gated: TAUROS and SCYTHER (Safari
entries need fresh authorisation; Scyther
alternatively 5,500 Game Corner coins), PORYGON
(9,999 coins), ZAPDOS (deferred, not fenced — a
go-ahead would un-park it), and LICKITUNG (Route
18 Gate trade, gives a SLOWBRO — the chain holds
exactly one, caught this leg, so a second Slowbro
from Seafoam B2F makes the trade free). Fenced
unless Vlad says otherwise: ARTICUNO, MOLTRES,
MEWTWO.

## 32. Post-game leg 4 (probe171, 2026-11-18): the free four, the Cut/Flash pair restored, and the throw that left the bag open — dex 112 → 116

Vlad's kickoff: "Kick off leg 4 — free four + re-catch
the Cut/Flash pair". Scope A was the four species
obtainable from owned stock or a plain wild catch;
scope B undid §31's quiet loss — Gloom (the Cut user)
and Voltorb (the Flash user) had vanished during the
leg 1/2 box reorganization, leaving the chain with no
Cut and no Flash. Vlad's kickoff explicitly authorised
the two HM re-teaches (the named exception to the
no-HM-teach fence, these species/moves only).

**The free four.** All three boxed grinds ran on the
Mansion switch-grind with DECLINE-all replacements and
per-level audits; every evolution was RAM-verified and
the specimen deposited back to its home box.

- MUK: Grimer L30 (box 9) → L38. Final moves
  [Pound, Disable, Poison Gas, Minimize], filed box 9.
- KINGLER: Krabby L15 (box 9) → L28. Final moves
  [Bubble, Leer, ViceGrip, Guillotine], filed box 9.
- SEAKING: Goldeen L15 (box 3) → L33. Final moves
  [Peck, Tail Whip, Supersonic, Horn Attack], filed
  box 3. (The first Goldeen attempt hit the grind
  driver's prompt-abort after 107 fights; the
  flee-and-retry shell finished it.)
- TANGELA: wild, Route 21. The route's only grass is
  a 6×6 island patch at x 8–13, y 4–9 (north end);
  the working landing is surf to (10,10) then one step
  up onto the grass at (10,9). Caught L30
  [Constrict, Bind, Absorb], 3 Ultra Balls in the
  successful hunt, filed box 3 at close-out.

Blastoise, the grind fighter, went 93 → 96 across the
leg. Dex asserted at each step: 112 → 113 → 114 →
115 → 116.

**The bag-list wedge (the leg's real find).** The
first Tangela hunts failed in a way that looked like
legendary bad luck: every Tangela "got away" after
exactly one ball, however many balls the catch was
allowed. Instrumented throws (probe171/dbg_throw*)
isolated it: after a ball is selected in the battle
ITEM list, the list menu STAYS OPEN (wMenuBoxTopY=4,
cursor resting on the thrown ball, a dialog pending)
while the throw resolves. The post-throw ride in
_catch_current_careful only presses A in that state,
the game ignores A there, the ride times out, and the
catch reports failure — the hunt then flees a live
battle. One B press closes the list back to the
battle menu (verified live). Two companion findings:
in_dialog reads True even at the settled battle menu,
so "menu live" can only be proven by _advance_battle's
cursor-wiggle test, never by the dialog flag; and the
party-list cursor is ADDR_MENU_ITEM alone (in that
menu ADDR_MENU_SCROLL holds the party count, not a
scroll offset — a summed getter reads garbage, which
broke the first Flash-test steer). The fix is ported:
_catch_current_careful's ride now B-closes a leftover
list (top_y 4) and proves the menu with
_advance_battle, and the same B-close branch was added
to the _catch_any and _throw_ball_ordered rides, which
share the throw-through-the-bag mechanism. With the
fix, Tangela took 2 balls in the verification run
(breakout, B-close, catch). The wedge also explains
past legs' ball spend on commons — attempts were
silently costing one ball + a flee each. Separately,
_hunt_paired referenced _PACE_DIRS, which the §30
port never brought over from probe169's catch_lib (a
NameError on first use); the constant is now defined
in run_real.py with a regression test. Suite 242/242.

**The Cut/Flash restore.** HM01 and HM05 were both
found in PC item storage (×1 each; HM05 had sat there
since the storyline's Fly leg) and withdrawn in S1.
Compatibility was verified from pokered base_stats
before teaching, per the verify-first rule: ODDISH's
tmhm list ends in CUT, VOLTORB's in FLASH.

- ODDISH: caught fresh on Route 24 grass, L12,
  [Absorb] — both remaining Great Balls spent across
  the hunt. Cut taught into the free slot:
  [Absorb, Cut]. NOTHING was replaced. Functional
  test: the executor cut Cerulean's east tree
  (32,13) and the party's arrival on Route 9 was
  RAM-verified (media: s4-tree-before/after.png).
- VOLTORB: caught fresh on Route 10 grass at the
  proven (8,7) patch, L16, [Tackle, Screech] — 1
  Ultra Ball. (The first S5 run hunted with the
  Great Ball stack S4 had just emptied; every throw
  failed instantly on the missing item. The script
  now asserts its hunt ball exists.) Flash taught
  into the free slot: [Tackle, Screech, Flash].
  NOTHING was replaced. Functional test: Rock
  Tunnel 1F frame brightness 67.79 (dark) → 134.02
  after using Flash from Voltorb's party sub-menu
  (sub-menu index 0 — the engine puts field moves
  first; the normalised screen text hides the cursor
  glyph, so selection was verified by consequence,
  media: s5-tunnel-dark/lit.png).

HMs are not consumed: HM01 and HM05 are both still
in the bag at leg end. Neither utility specimen was
ground further. Oddish/Voltorb are already dex-owned
species, so the count did not move for them.

**Money trail.** Start ¥2,011. One purchase all leg:
Ultra Balls topped up to 6 at Fuchsia Mart in S1,
−¥1,200 → ¥811, where it stayed. No sales, no E4
re-run (§31's attrition lesson respected). Ball
consumption in the persistent line: Ultra 6 → 2
(3 on the Tangela hunt, 1 on Voltorb), Great 2 → 0
(both on Oddish).

**Close-out state.** Healed at Fuchsia, Tangela
filed, census taken (SRAM scan + WRAM mirror for the
current box): 92 boxed across 12 boxes + 5 in party.
The current box at leg end is box 3 (Seaking and
Tangela there; Muk/Kingler box 9; the chain's only
Slowbro still unique in box 4 — the second Slowbro
was NOT caught, per the Lickitung fence). Two
standing-behaviour notes: heal_at_pokecenter leaves
by the exit mat, so PC work after a heal must
re-enter first; and s5-flash-lit.state boots with
the START menu open, so a following fly_to needs the
menus closed first (its START press otherwise
toggles the menu shut and its waits fail).

FINAL leg-4 chain head: probe171/s6-final.state,
sha256
b7e897a7ee03ddc10061bbd240774dbc8ee4dfccb5d86a56e909d29c43f39239
— Fuchsia Pokécenter, dex 116, ¥811, badges 8.
Party: Lapras 58 [Sing, Ice Beam, Surf, Strength],
Blastoise 96, Pidgeot 36 [Gust, Fly, Quick Attack,
Whirlwind], Oddish 12 [Absorb, Cut], Voltorb 16
[Tackle, Screech, Flash] — the utility pair rides
with the party again, all healed.

Dex 116 / 124. Remaining (leg 5) is exactly the
gated eight: LICKITUNG (Route 18 Gate trade for a
Slowbro — Vlad's separate call; the trade is only
free once a second Slowbro exists, and catching one
is bundled with that call), TAUROS and SCYTHER
(Safari — two budgets already spent, fresh
authorisation needed; Scyther alternatively 5,500
Game Corner coins), PORYGON (9,999 Game Corner
coins), ZAPDOS (deferred, not fenced — a go-ahead
un-parks it; its Power Plant room was again not
entered this leg), and the fenced three: ARTICUNO,
MOLTRES, MEWTWO.
