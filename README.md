# Pokemon Red Agent Project

A Pokemon Red agent with a **two-tier architecture**: a VLM planner decides
WHAT to do, a deterministic driver decides HOW to move.

```
VLM planner (--planner vlm, every 15 turns)   "Traverse Mt. Moon south to north"
                  |
                  v
Von scorer  (--scorer von, every turn)         BFS to door, walk right, ledge hop,
                                               Gen 1 battle policy (~1ms/turn)
```

## Architecture: planner vs scorer (pick ONE per layer)

There are two decision layers. Each has swappable implementations - they are
NOT duplicates of each other:

**Layer 1 - High-level planner** (`--planner`): what should the agent want?
| Option | What it does | Speed |
|--------|--------------|-------|
| `--planner vlm` | Fine-tuned Qwen2.5-VL-7B LoRA reads screen+RAM, picks a goal from a closed 16-goal vocabulary | ~2s, every 15 turns |
| `--planner rules` | Deterministic rules (`planner_rules.py`) - same logic that labels the training data | ~0ms |
| *(absent)* | Legacy: Hermes LLM escalation on stall | slow, per-stall |

**Layer 2 - Low-level scorer** (`--scorer`): given the goal, what button now?
| Option | What it does | Speed |
|--------|--------------|-------|
| `--scorer von` | Deterministic: BFS over learned walkable grid, portal routing, empirical ledge hops, exact Gen 1 battle policy | ~1ms/turn |
| `--scorer laya` | Neural fast scorer - candidate actions scored by a small local model | ~50ms/turn |
| `--scorer llm` | Local llama.cpp server picks from offered options per turn | ~2-4s/turn |

**Key point:** Von and Laya are NOT both running, and are not the same thing.
**Von is the agent** (the whole driver: map graph, stall guards, flight
recorder, heal logic). **Laya is one scorer backend you could plug into it.**
Von's scorer is strictly more capable today (Laya has no map graph, no ledge
hops, no battle policy) - we run `--scorer von` and keep `--scorer laya`
available as an experiment.

Also available: `--scorer expert:von` (Von + fallback), and legacy `--laya`
(which just selects Laya as the scorer; redundant with `--scorer laya`).

## Components

1. **Emulator Server** (`pokemon-agent serve`)
   - Runs the Pokemon Red ROM and exposes a HTTP API (`localhost:8765`) for state/control.
   - Also serves `/screenshot` (latest frame) and `/screenrecord` (MP4 clips
     sampled from the published framebuffer - does NOT pause the emulator).

2. **Play Loop** (`pokemon-agent play`) - the two-tier agent
   ```bash
   pokemon-agent play --host localhost --port 8765 \
       --scorer von --planner vlm --plan-interval 15 \
       --stall-seconds 45 --hermes-turns 8 --turn-timeout 240
   ```
   - `--planner vlm` requires a llama.cpp server with the planner LoRA
     (`POKEMON_PLANNER_URL`, default `http://127.0.0.1:8002/v1`); falls back
     to `--planner rules` behavior per-turn if the server is down.
   - `--plan-interval N`: planner refresh cadence in turns (default 15).
   - `--scorer von`: deterministic movement+battle (recommended).
   - `--laya` is accepted but ignored: Von is the only driver now.

### Engine layout (which file does what)

| File | Lines | Role |
|------|-------|------|
| `pokemon_agent/von.py` | ~4,000 | **The engine AND the source of truth.** Primitives (BFS, ledges, doors, heal, battle, screen stitching) + the driver + the router + the goal-termination and desync-recovery layers. |
| `pokemon_agent/autopilot.py` | ~90 | Back-compat shim: re-exports `VonDriver`/`HermesDriver`/`PlannerDriver`/`run_autopilot` and the BFS primitives so the CLI, tests and `vlm_brain.py` keep importing the same names. |
| `pokemon_agent/planner_client.py` | — | Tier-1 VLM planner client (closed goal vocab). |
| `pokemon_agent/planner_rules.py` | — | Tier-1 deterministic rules planner (offline labeler + fallback). |

The legacy monolith (`autopilot-von-on-rails.py` / `von_rails.py`) and the
slice build (`build_von.py`) were retired on 2026-10-17 - `von.py` is
edited directly now, and the pre-split history lives in git.

### Design rules (why the engine looks like this)

1. **The planner owns strategy; the engine owns topology.** There is no
   `_infer_goal` tower, no per-map `if cur_map == ...` overrides, no compass
   locks, no stall cooldowns and no pinball detectors. Adding a map must not
   require an engine change.
2. **One decision shape.** The router computes a concrete walk path and
   offers it as the **sole** option (`route_to_waypoint`); its handler
   executes that path verbatim. A single option cannot oscillate - this is
   structural, not a threshold.
3. **Three generic routing tiers**, no per-map code:
   1. a **learned portal** toward the destination (decoded from
      `portals.json`; sharper every session),
   2. a **BFS to the boundary** in the graph's compass direction over the
      stitched map,
   3. a **frontier step** toward unexplored space, for winding maps and
      boundaries the stitched map has not learned yet.
   The chosen target cell is **sticky** until reached or unreachable.
4. **The map corrects itself.** When the agent stops moving with a target
   committed, the first step of that path is marked unwalkable and the target
   is dropped (`_waypoint_stall_check`). Stitched grids lie; observation wins.
5. **Battles stay deterministic** (`gen1_battle`): an exact type chart at
   ~0ms beats a language model at ~2s/turn, always.
6. **Naming convention that matters:** the map graph speaks **compass**
   (north/south/east/west) while the BFS and actions speak **button**
   directions (up/down/left/right). Translating between them
   (`COMPASS_TO_WALK`) is the difference between a working route hop and a
   handler that silently falls through.
7. **Portal decoding:** a persisted portal stores the player's position when
   the door was seen (`player_x`/`player_y`) plus the door's **screen** cell
   (`col`/`row`). The door is at
   `(player_x + col - PLAYER_COL, player_y + row - PLAYER_ROW)`. All four
   Mt Moon 1F portals on Route 4 decode to the same door at `(18, 5)`.
8. **The free-walk explorer is EXPERIMENTAL — do not let it outrank a route.**
   It was written when the map was unknown and there was no planner: exploring
   was the only way forward. Neither premise holds now (ROM-native decode of
   all 221 maps + cross-map BFS + a planner that supplies a destination), so
   it is no longer load-bearing. It is still a **third unsupervised producer**
   competing with `route_to_waypoint`, and it is the component that repeatedly
   walked Von into the Pewter Museum.
   **Planned change (deliberately deferred):** gate it behind planner mode —
   `planner=vlm` + a resolved destination offers `route_to_waypoint` ONLY;
   `planner=vlm` with no route should re-resolve or hold rather than wander;
   `planner=rules` ("Von only") keeps free-walk as its designed behaviour.
   We are keeping it live for now because an exploratory walker is expected to
   be useful on a fresh playthrough before a map has been seen. See also
   `warp-cell discipline` below — doors must never be entered by an aimless
   single step.
9. **Cell authority: `static_open OR physically walked`.** The ROM decode
   (`static_walk.json`) is the walkability authority; the live camera-window
   fold (`map_walk.json`) may only ADD cells the player has actually crossed.
   The fold is otherwise more permissive than the ROM and produces false
   OPENS — Pewter City had 56 of them, a 5x3 building footprint at x=15-19
   y=1-3 that BFS will happily route *through*. The `OR walked` half is
   essential: the ROM decode has its own known false WALLS (Pewter rows
   32-33), and physically walking a cell must always be able to open it.

3. **Planner training** (`train_planner_lora.py` + `convert_pokedreamer.py`)
   - Data: PokeDreamer NPZ (auto-labeled from RAM state via
     `planner_rules.derive_goal` - NO manual labeling) + TAS replay with RAM
     capture (`tas_replay.py`) + live self-play (`collect_live.py` writes
     `dataset/live/` while the agent plays).
   - Class-balanced conversion is mandatory: raw PokeDreamer is ~82% dialog
     frames (`--max-per-goal 800`).
   - Qwen2.5-VL-7B LoRA; serve with llama.cpp:
     ```bash
     # convert PEFT adapter to GGUF for llama.cpp
     python /path/to/llama.cpp/convert_lora_to_gguf.py \
         out/planner_lora/adapter \
         --base /home/vlad/models/Qwen2.5-VL-7B-Instruct \
         --outfile out/planner_lora/planner_lora.gguf --outtype f16
     # serve (GGUF base model + mmproj + LoRA)
     llama-server -m <Qwen2.5-VL-7B-BF16.gguf> --mmproj <mmproj-F16.gguf> \
         --lora out/planner_lora/planner_lora.gguf \
         --port 8002 --ctx-size 4096 --n-gpu-layers 999
     ```

4. **SB3 Training** (`baseline_fast_v2.py`, retired but kept)
   - PPO policy training; produced 0 badges at 100M steps - kept for reference.
   - Run from `/home/vlad/pokemon-agent/v2/`.

## Setup & Execution

### 1. Prepare the ROM
You need a Pokemon Red (USA) ROM file named `PokemonRed.gb`.  
To build your own ROM from source, follow the instructions in the pret/pokered repository:
https://github.com/pret/pokered

Place the built ROM at:
```
/home/vlad/pokemon-agent/PokemonRed.gb
```
(or adjust paths in the commands below.)

### 2. Start the Emulator Server
In a terminal/screen session:
```bash
cd /home/vlad/pokemon-agent
source /home/vlad/hermes_venv/bin/activate
pokemon-agent serve --rom PokemonRed.gb --port 8765
```

### 2b. Start the VLM Planner Server (pick ONE backend)

The planner LoRA (`out/planner_lora_v2/adapter`, Qwen2.5-VL-7B base) is
served over an OpenAI-compatible API. Two supported backends:

**vLLM (recommended on AGX Orin)** - serves the HF base + LoRA adapter
directly, no GGUF conversion, full VL fidelity:

```bash
cd /home/vlad
/home/vlad/vllm_env/bin/vllm serve /home/vlad/models/Qwen2.5-VL-7B-Instruct \
    --enable-lora \
    --lora-modules planner=/home/vlad/pokemon-agent/out/planner_lora_v2/adapter \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.35 \
    --port 8002 --host 127.0.0.1
```

Notes:
- `--gpu-memory-utilization 0.35` is required on the Orin (24GB free at
  rest of 61GB shared; 0.55 fails with a memory error)
- startup takes ~4 minutes (CUDA graph capture, 102 graphs)
- the LoRA serves under the name `planner`; the base under its path
- verify: `curl http://127.0.0.1:8002/v1/models`

**llama.cpp with the LoRA (PROVEN 2026-10-23 — the working path):**

The 7B + LoRA CANNOT run as a Hermes background process: Hermes worker
scopes are cgroup-capped at 4GB (`memory.max=4294967296`) and the 7B needs
~14GB anon — the kernel OOM-kills it mid-load (`constraint=CONSTRAINT_MEMCG`,
exit 137) regardless of 57GB free. Foreground terminals work but die at the
terminal timeout. The fix is a **detached systemd user service**:

```bash
# 1. Convert the PEFT adapter to GGUF (note: --outfile, NOT --out;
#    --out is ambiguous with --outtype and errors out)
python3 /home/vlad/llama.cpp/convert_lora_to_gguf.py \
    --base /home/vlad/models/Qwen2.5-VL-7B-Instruct \
    --outfile /home/vlad/models/planner_lora_tas.gguf \
    out/planner_lora_tas_balanced/adapter

# 2. Install the service (file: ~/.config/systemd/user/llama7b.service)
systemctl --user daemon-reload && systemctl --user start llama7b
# verify: systemctl --user is-active llama7b  +  curl :8002/health
# expect ~33GB RSS (BF16 7B + mmproj + LoRA + KV); fits the 61GB unified pool
# free page cache first if tight:  sudo /home/vlad/drop_cache.sh
```

Service unit (serves base BF16 GGUF + mmproj + our LoRA on :8002):
```ini
[Service]
ExecStart=/home/vlad/llama.cpp/build/bin/llama-server \
  -m /home/vlad/models/models--unsloth--Qwen2.5-VL-7B-Instruct-GGUF/snapshots/68bb8bc4b7df5289c143aaec0ab477a7d4051aab/Qwen2.5-VL-7B-Instruct-BF16.gguf \
  --mmproj /home/vlad/models/models--unsloth--Qwen2.5-VL-7B-Instruct-GGUF/snapshots/68bb8bc4b7df5289c143aaec0ab477a7d4051aab/mmproj-BF16.gguf \
  --lora /home/vlad/models/planner_lora_tas.gguf \
  -c 8192 -ngl 99 -fa on -t 8 -tb 8 -np 1 --reasoning off \
  --host 0.0.0.0 --port 8002 --sleep-idle-seconds -1
Restart=on-failure
OOMScoreAdjust=0
```

Verified live: real screenshot + RAM state -> exact vocabulary goal in ~3s
("Travel north toward Pewter City..."), zero prose drift. A Qwen2-VL-2B
GGUF can substitute for smoke-testing the HTTP plumbing, but it carries
ZERO of the LoRA training (adapters are architecture-specific).

The driver picks the server up via env (defaults match):
```bash
export POKEMON_PLANNER_URL="http://127.0.0.1:8002/v1"
export POKEMON_PLANNER_MODEL="planner"
```

Full-stack spin-up from scratch (the session checklist):
```bash
# terminal 1 - game server (screen instance / background)
pokemon-agent serve --rom pokeblue.gbc --host 0.0.0.0 --port 8765 \
    --data-dir /home/vlad/.pokemon-agent
# pick the save: curl :8765/games  ->  POST /games/{sid}/load (arms latest save)
# boot it:       curl -X POST :8765/control -d '{"state":"running"}'
# NOTE: /action takes {"actions": ["walk_up", ...]} — an ARRAY named actions.
```
```bash
# terminal 2 - Von driver with the trained VLM as the WHAT
POKEMON_PLANNER_URL=http://127.0.0.1:8002/v1 POKEMON_PLANNER_MODEL=planner \
pokemon-agent play --host localhost --port 8765 --scorer von --no-vision \
    --planner vlm --plan-interval 10 --stall-seconds 45 \
    --hermes-turns 8 --turn-timeout 240
```

Known blocker (2026-10-23): Viridian City's stitched-graph route to Pewter
runs through a phantom gap at local (35,30)->(36,31); the y=31 wall
(x28-40) is real geometry, the collision learner re-poisons those cells
every run, and BFS wedges in a 2-tile loop (`BFS-FAIL (35,29)->(36,32)
seen_n=1132`). Purging `map_walk.json['Viridian City']` (backup:
`map_walk.json.pre_viridian_unstick_*`) frees the agent temporarily but it
re-wedges. Game event flags (parcel delivered, pokeDex held) are NOT the
cause. Fix pending: correct/ground-truth the Viridian portal records.

**Deterministic post-filter**: the planner client (`planner_client.py`)
veto goals whose preconditions are already satisfied (delivered parcel,
healthy party heal goals, wrong-city early-route goals) and snaps the
VLM's raw choice to a valid vocab line - the VLM decides, the filter
validates, the BFS executes. The static route decode
(`pokemon_agent/static_decode.py`) provides ROM-exact walkability for
221 maps; `scripts/validate_bfs.py` verifies checkpoint cells and
`scripts/calibrate_static.py` scores the decode against the position
log (currently 1733/1734 real positions agree).

### 3. Start the Play Loop (two-tier: VLM planner + Von scorer)
In another terminal/screen session:
```bash
cd /home/vlad/pokemon-agent
source /home/vlad/hermes_venv/bin/activate
export POKEMON_PLANNER_URL="http://127.0.0.1:8002/v1"   # planner LoRA server
pokemon-agent play --host localhost --port 8765 \
    --scorer von --planner vlm --plan-interval 15 \
    --stall-seconds 45 --hermes-turns 8 --turn-timeout 240
```
Without a planner server running, use `--planner rules` (deterministic) or
omit `--planner` for legacy Hermes-escalation behavior.

### 4. (Optional) Start SB3 Training
In a third terminal/screen session:
```bash
cd /home/vlad/pokemon-agent/v2
source /home/vlad/hermes_venv/bin/activate
python baseline_fast_v2.py \
  --num-cpu 12 \
  --ep-length 163840 \
  --n-steps 1024 \
  --batch-size 1024 \
  --save-freq 50000 \
  --device cuda \
  --rom ../PokemonRed.gb
```

## Notes

- The agent has been modified to use the `nvidia/nemotron-3.5-lightning-30b-a3b` model to avoid API streaming stalls encountered with larger models.
- Hermes turn timeout increased to 360 seconds to allow full 8‑call turns to complete.
- Temporary/test scripts in the project root have been cleaned up; only essential files remain.
- For troubleshooting, check logs:
  - Hermes agent: `~/.hermes/logs/agent.log`
  - Pokemon agent play: `/home/vlad/pokemon-agent/v2/runs/autopilot_*.log`
  - SB3 training: `/home/vlad/pokemon-agent/v2/runs/train_*.log`

## GPU PyTorch on JetPack 7.2 (Orin sm_87) — the undocumented path

JetPack 7.2 (L4T R39, Ubuntu 24.04, CUDA 13.2, Python 3.12) has NO working
PyTorch in NVIDIA's apt repos, and the official Jetson redist indexes
(`developer.download.nvidia.com/compute/redist/jp/vXX`) do not carry JetPack 7
wheels yet. The plain `pip install torch` wheel reports
`torch.cuda.is_available() == True` but silently falls back to CPU — its CUDA
kernel list has no `sm_87` (Orin's compute capability). Verify with:

```python
import torch
print(torch.cuda.get_arch_list())   # bad: ['sm_80','sm_90','sm_100',...] (no sm_87)
# + warning: "No published PyTorch CUDA builds ... support this GPU"
```

`torch.cuda.is_available() == True` is NOT proof — only a real kernel launch is.

**The one that works** is the prerelease `cu132` wheel from PyTorch's own
index (cu132 matches the JetPack 7.2 system CUDA 13.2; the aarch64 build
includes sm_87):

```bash
pip install --pre torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cu132 --upgrade
```

Gotchas:

- `--pre` is REQUIRED — without it pip finds nothing and you conclude no wheel
  exists. It is a prerelease, invisible to normal pip listing.
- Use `--extra-index-url` (NOT `--index-url`) so numpy etc. still resolve from
  PyPI.
- Python must be 3.12 (JetPack 7.2 default). No cp313 wheel exists for this
  stack yet.
- If a later `import torch`-adjacent module fails with a missing `lib*.so`,
  install the NVIDIA CUDA repo packages (NVPL, cuDSS) per the reference:
  https://github.com/iuliaferoli/jetson-jp7.2-install
- **torchaudio conflict**: the newest torchaudio in user site-packages
  (`~/.local/...`) may be built against a different CUDA (e.g. cu130 vs
  cu132) and will hard-fail EVERY import of transformers/peft with
  "PyTorch and TorchAudio were compiled with different CUDA versions". If you
  don't need torchaudio, move it aside:
  ```bash
  mkdir -p ~/.local/lib/python3.12/site-packages_disabled
  mv ~/.local/lib/python3.12/site-packages/torchaudio* \
     ~/.local/lib/python3.12/site-packages_disabled/
  ```
  (No cu132 torchaudio exists yet; restore from `site-packages_disabled/` when
  it does or if a project needs audio.)
- Packages that pin torch (e.g. `vllm==2.13.0`) will warn about the version
  mismatch — harmless unless you actually use them.

**Verify after install** (the only test that counts — a real kernel launch):

```python
import torch
print(torch.__version__)                    # expect 2.14.1+cu132 (or newer cu132)
print(torch.cuda.get_arch_list())           # must still list sm-capable archs
x = torch.randn(2048, 2048, device='cuda'); y = x @ x; torch.cuda.synchronize()
print('GPU OK', y.shape)                    # no "no kernel image" error = pass
```

Measured speedup on AGX Orin 64GB (Qwen2.5-VL-7B LoRA, bs=2): ~24.6s/step CPU
→ ~12-24s/step GPU (unified memory caps the advantage; inference benefits are
larger than training).

Enjoy watching the agent explore Kanto!