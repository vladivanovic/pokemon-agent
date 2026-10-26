# Pokemon Agent — Battle Model Training

Goal: decide whether a small local model (Qwen3.5-0.8B) is worth fine-tuning to
play Gen 1 battles, before spending GPU time.

**Current verdict (2026-10-10): NOT WORTH IT YET.** See "Evidence" below.

## Model benchmark: 0.8B vs 2B (2026-10-10)

`bench_models.py` runs the same prompts against llama.cpp on both models.
**The 0.8B is both faster and more accurate** on the prompts we actually use.

| prompt size | 0.8B | 2B | winner |
|---|---|---|---|
| battle decision (~150 tok) | **2/3 correct**, 0.2-0.4 s | 0/3 correct, 0.2-0.4 s | 0.8B |
| tool/JSON call (~150 tok) | **3/3 valid JSON**, 0.2-0.6 s | 2/3, 0.4-0.8 s | 0.8B |
| tool call, 518 tok | 0.6 s (34 tok/s) | 1.1 s (20 tok/s) | 0.8B (1.7x) |
| tool call, 8,724 tok | **2.9 s**, malformed JSON | 5.0 s, malformed JSON | 0.8B (1.7x) |

The 2B's battle answers were not a caching artifact: it returns `move_2`
(Leech Seed, **power 0**) with `cache_prompt` both on and off. The 0.8B
returns `move_0` (Tackle) consistently.

### The real timeout fix is a grammar, not a smaller model

At 8,724 prompt tokens **both** models emit malformed tool JSON (the 2B even
hallucinates fields like `"new_pos"` / `"status"`). Shrinking the model does
not fix that. Constraining decoding does:

```json
"response_format": {"type": "json_schema", "json_schema": {
    "name": "act", "strict": true, "schema": {
        "type": "object",
        "properties": {"actions": {"type": "array", "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string", "enum": ["walk_up","walk_down",
                "walk_left","walk_right","press_a","press_b",
                "wait_30","wait_60"]}}},
        "required": ["actions"]}}}
```

Result at 8,724 tokens with the schema: **valid JSON from both** (0.8B 4.5 s,
2B 6.4 s). Schema-valid does not mean semantically right - both picked
`walk_right` for a "walk north" instruction, so the prompt still matters - but
it eliminates the malformed-JSON retry loop that was burning the budget.

**Recommendation:** switch the escalation model to 0.8B *and* constrain the
tool-call schema. Either alone is a partial fix; together they remove both the
latency and the retry-loop cause.

## Schema-constrained scorer (`POKEMON_SCORER=llm`)

Step 2 of the plan. `pokemon_agent/llm_direct.py` calls llama.cpp directly
with a **per-turn json_schema whose enum is the criteria key set**, so the
model cannot return an option that was not offered. The `hermes chat`
subprocess path cannot take a decoder constraint, which is why it retried
malformed output until the turn timed out.

```bash
POKEMON_SCORER=llm POKEMON_LLM_URL=http://127.0.0.1:8001/v1 \
POKEMON_LLM_MODEL=Qwen3.5-0.8B \
python3 -m pokemon_agent.cli play --scorer llm ...
```

`test_llm_scorer.py` covers the cases that actually wedged the agent:

```
  [OK ] leave_building_F9     want=[leave_building_F9, leave_toward_down]  0.5s
  [OK ] shop_buy              want=[route_H1, shop_buy]                   0.4s
  [OK ] advance_text          want=[advance_text]                         0.6s
  [OK ] route_toward_north    want=[route_A2, route_toward_north]         0.5s
  4/4 correct
```

### Where the boundary is

| task | engine | why |
|---|---|---|
| battles | **deterministic Gen 1 policy** | 100% oracle agreement over 134 decisions, and free |
| navigation / doors / menus / dialog / shopping | **constrained LLM** | opaque option keys need the descriptions |

The LLM does **not** reliably reason about type effectiveness from bare
numbers - it picks GRASS (0.25x vs BUG/POISON) over NORMAL (1.0x). That case
is in `INFO_CASES`, deliberately excluded from pass/fail. Battles stay on the
policy: more accurate *and* no GPU cost.

## Files

| File | Purpose |
|---|---|
| `gen1_oracle.py` | Parses `pret/pokered` → type chart, 165 moves, 151 base stats. Ground truth for the ROM in the emulator. |
| `evaluate_headroom.py` | The gate. Scores the live policy against the oracle on recorded battles. |
| `build_dataset.py` | Turn `flight_recorder.jsonl` battles into SFT pairs (oracle-labelled). |
| `bench_models.py` | 0.8B vs 2B on battle + tool prompts (accuracy, JSON validity, tok/s). |
| `test_llm_scorer.py` | Regression test for `POKEMON_SCORER=llm` navigation decisions. |
| `train_sft.py` | LoRA SFT. **Do not run until the gate passes.** |
| `data/` | Generated datasets (gitignored). |

## The gate

Run these two, in order:

```bash
python3 gen1_oracle.py          # self-check must print PASS
python3 evaluate_headroom.py    # prints agreement + a VERDICT
python3 build_dataset.py        # only if headroom is real
```

Train only when **both** hold:
1. `evaluate_headroom.py` reports **>5k move decisions** and **>15% disagreement**.
2. `data/battles.jsonl` has **>5k pairs** with a spread of enemy species and
   at least some **trainer/gym** battles.

## Evidence (2026-10-10, after the mapping fixes)

`evaluate_headroom.py` over 7,285 recorded turns:

```
battle turns   : 323
move decisions : 134
  oracle agrees: 134 (100.0%)
  DISAGREES    : 0 (0.0%)
non-move turns : 189 (flee/switch/item)
enemy species  : Kakuna 163, Weedle 110, Caterpie 18, Metapod 16, Pikachu 8, ...
```

**The current deterministic policy is already oracle-optimal on move choice.**
The apparent Vine Whip misuse (32 picks) is explained: every one of them
happened with Tackle PP at 0 — the correct fallback.

Also verified: the **untrained** base Qwen3.5-0.8B already answers `Tackle` for
a Weedle/PP=17 prompt, i.e. it agrees with the oracle without any SFT.

## Why the data is not ready

1. **Volume.** 343 pairs total (134 fights + 209 flee). SFT on this overfits
   instantly and generalises to nothing.
2. **No signal.** 100% agreement means there is nothing to imitate.
3. **Domain.** Every recorded battle is `type: wild`, fought against two
   species (Weedle/Kakuna) in Viridian Forest. Zero trainer battles.
4. **The agent flees 59% of fights** (no Poke Balls in the bag). Run-vs-fight
   is the real open question, and the oracle does not cover it.

The places an LLM could add value — gym leaders, the rival, status/switch
decisions — have **no data at all**.

## Platform notes (NVIDIA AGX Orin, aarch64)

- `Qwen3_5ForCausalLM` loads text-only in **1.3 s**, **752M params** (bf16).
  The full checkpoint is multimodal (12-layer vision tower); loading the
  CausalLM head drops it, keeping the VL weights untouched for future
  navigation work.
- **No `causal_conv1d` / `flash-linear-attention` wheels for aarch64** — the
  hybrid gated-delta-rule layers fall back to reference PyTorch and run at
  ~**4 tok/s**, far too slow for in-battle use. The llama.cpp Qwen3.5-2B GGUF
  on port 8000 is the faster path today.
- `peft`, `datasets`, `trl` install cleanly. `bitsandbytes` does not (not
  needed — 0.8B fits in bf16 with LoRA).

## Ground truth, not scraped data

Everything the oracle needs is in the local `pret/pokered` clone. Do **not**
train on Gen 7 sources: Gen 1 differs, and the differences are exactly the
battles we care about:

```
POISON -> BUG     SUPER_EFFECTIVE   (neutral in Gen 2+)
BUG    -> POISON  SUPER_EFFECTIVE   (not-very-effective in Gen 2+)
GHOST  -> PSYCHIC NO_EFFECT         (the famous 0-damage "super effective" bug)
```

## Next step to make training worthwhile

Collect data where the policy is actually weak:
- Give the agent Poke Balls (Viridian Mart, $200 each) so it **fights** instead
  of fleeing — that alone multiplies battle data.
- Push through to **trainer battles** (Brock, rival, route trainers) and record
  those specifically.
- Re-run the gate. If disagreement stays near 0 on trainer battles too, the
  honest answer is that no battle model is needed.
