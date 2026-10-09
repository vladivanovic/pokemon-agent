# Pokemon Agent Roadmap

## North Star
Generalize the Red/Blue agent stack to any Gen 1/2 (and beyond) game with
minimal per-game hardcoding. The layers we built for Red split into
**portable** (gameplay logic, reusable as-is) and **ROM-specific** (memory
pointers, data tables, per-game reversal).

## Where we are (Oct 2026)

### Portable layers (work on any game unchanged)
- `MapGraph` — global navigation, portal learning from live warp reads,
  bidirectional edges, empirical edge verification (fake-adjacency
  blacklisting), door-avoiding BFS, geographic fallback. The warp-table
  *format* (count + entries) is the same across Gen 1/2.
- Heal decision layer (`_needs_heal` / `_heal_options`), route layer,
  arrival guard, interact-spam guard, oscillation window.
- Gen 1 battle policy (`gen1_battle.py`) — deterministic, compiled from
  pokered. The *approach* (compile from ROM data) is portable; the
  *tables* are per-game.
- Scorer abstraction (`BaseScorer`: Laya/Von/Expert) — pluggable via
  `--scorer`, `POKEMON_SCORER`, or config.
- SB3 exploration union (`explore_map_union_rank*.json`) — a learned map
  prior that seeds any new run.

### ROM-specific (per-game reversal required)
- `memory/red.py` — RAM pointers (party, map id, position, bag, money).
  Gen 2 (Gold/Silver) reuses most structures at shifted addresses
  (party ~$DA43 area, map id ~$D35E area — verify against pret/pokegold).
- `gen1_data.py` — moves, type chart, sprites, MAP_NAMES. Regenerate from
  pokegold's `moves.asm`, `type_matchups.asm`, `sprite_constants.asm`,
  `map_constants.asm`.
- `pc_centers.json` — Pokecenter door coordinates. Re-extract from
  pokegold's `map_objects` warp_event lines.

## Kanto advantage (Gold/Silver)
Gold's endgame returns to Kanto — the Red map data, portal table, and
pc_centers all carry over directly for that half. Johto is the only
net-new region. This makes Gold the cheapest second game: ~40% of the
world is already known.

---

## Phase 1 — VL data collection (Red, now)
The current stack *is* the labeled dataset. Every autopilot turn logs
`(state, action)` where the action is known-good (Von scored it, the
dispatch executed it, the game state confirmed it).

- Decision trace: `~/.pokemon-agent/decision_trace.jsonl`
- What to capture per turn: the raw frame (160x144 PNG or the model
  frame 72x80x3), the minimal state (map, pos, HP, PP, menu/dialog
  flags), and the chosen action.
- Storage: one JSONL per session under `v2/runs/vl_dataset/`.

## Phase 2 — Gold playthrough with the Red stack
Run the same autopilot on pokegold. Expect clumsy menus/dialog (Gen 2
has phone calls, day/night, held items) but working traversal (the
graph learns Gold's portals live). Capture the same dataset.

## Phase 3 — Qwen2-VL distillation
Train on `(frame + minimal state, action)` pairs. The model learns
"press A on nurse", "walk toward door", "FIGHT -> move 1" *visually*.

### Concept pseudocode

```python
# 1. Dataset assembly — one row per autopilot turn
def build_vl_row(turn_log):
    return {
        "image": frame_png(turn_log),          # 160x144 resized to 448x448
        "text": minimal_state_prompt(turn_log),
        #   "Map: Viridian Forest. HP 15/59. PP 0. Dialog: none.
        #    Battle: wild Rattata 12/20. Menu: none. Options: ..."
        "action": turn_log["chosen"],           # "route_B8" / "move_1" / ...
        "scorer_p": turn_log["p"],              # confidence filter >= 0.5
        "outcome": turn_log["next_pos_delta"],  # did it work?
    }

# 2. Training — LoRA on Qwen2-VL-2B (fits AGX Orin 64GB)
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="bfloat16")
# LoRA r=16, alpha=32 on vision+language projections
# loss: next-token on the action string, masked to the completion
# filter: only rows where outcome == "worked" (behavior cloning on success)

# 3. Deployment — VLScorer as a BaseScorer
class VLScorer(BaseScorer):
    """One forward pass: frame + minimal state -> action + confidence."""
    def decide(self, state, criteria):
        prompt = minimal_state_prompt(state)
        image = state_frame(state)
        out = model.generate(image, prompt, max_new_tokens=8)
        action, p = parse_action(out)        # "route_B8" 0.87
        if action in criteria and p >= self.min_p:
            return {action: criteria[action]}, p
        return {}, 0.0                       # fall to Von

# 4. Supervisor idea — Qwen2-VL watches Von, interjects when unsure
#    (the user's "supervisor" concept):
class VLSupervisor:
    """Von decides; VL reviews every Nth turn or on low confidence."""
    def review(self, state, von_choice, von_p):
        if von_p >= 0.8:
            return von_choice               # trust the fast scorer
        vl_action, vl_p = VLScorer().decide(state)
        if vl_p > von_p:
            return vl_action                # VL overrides on uncertainty
        return von_choice
```

### Key details for the Qwen2-VL integration
- Model GGUF exists at `/home/vlad/models/Qwen2-VL-2B-*.gguf` (downloaded,
  unsloth quant). For training use the HF transformers checkpoint, not the
  GGUF; llama.cpp serves the GGUF for inference.
- llama.cpp multimodal: `llama-server -m Qwen2-VL-2B.gguf --mmproj
  mmproj-Qwen2-VL-2B-f16.gguf` — the mmproj file is the vision encoder.
- AGX Orin 64GB: 2B bf16 LoRA training fits (~14GB weights + ~8GB
  optimizer). 7B would need QLoRA (4-bit) and more patience.
- Prompt shape: keep the minimal state text SHORT (<200 tokens). The
  frame carries the visual detail; the text only carries what the frame
  cannot show (HP numbers, PP, flags, menu state).
- Success filter: behavior-clone only on actions that *worked* (position
  advanced, dialog advanced, battle won). Failed actions become negative
  examples or are dropped.

## Phase 4 — VL as the supervisor (the user's idea)
Von is fast (~200ms) but shallow (ModernBERT-base, no visual context).
Qwen2-VL is slow (~2-5s on Orin) but sees everything. The natural split:

- **Von decides every turn** (fast path, no visual needed for routine
  walking/routing).
- **VL reviews on:**
  - low Von confidence (p < 0.5),
  - repeated action (interact spam, oscillation),
  - state change the graph cannot express (new dialog type, new menu).
- **VL learns from Von's mistakes**: when Von's action fails (no
  transition, no progress), that `(frame, failed_action)` pair becomes a
  correction example for the next VL fine-tune.

This is online imitation learning with a slow teacher and a fast
student — the student (Von) keeps getting better, and the teacher (VL)
only pays attention where the student is wrong.

## Phase 5 — Silver/Crystal/Emerald
Once Gold works, only the tileset and RAM offsets change. The graph,
scorer, heal/route layers, and VL policy carry over.

## Immediate next actions
1. Monitor the reshaped-reward run (`train_1009_reshaped.log`) — does
   `badge` reward now pull the agent toward Pewter?
2. Wire `get_pokeball_count()` into a catch incentive (buy balls at the
   Mart when money > 500 and balls == 0; throw ball in battle when
   enemy HP < 30% and wild).
3. Start logging the VL dataset from the live autopilot.
4. Gate↔Route 2 maze traversal: per-map explored-cell memory for big
   interiors (Viridian Forest 24 rows) — the 10x9 window cannot see the
   whole maze.
