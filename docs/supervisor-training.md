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
