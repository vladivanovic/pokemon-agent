# Pokemon Red Agent Project

This repository contains the setup and scripts to run a Pokemon Red agent using:
- **Laya** (fast heuristic explorer) for local decision-making
- **Hermes Agent** (LLM-powered planner) for high-level strategy
- **SB3 (Stable Baselines3)** PPO training for reinforcement learning
- Custom Gym environment (`RedGymEnv_v2`) interfacing with Pokemon Red via `pokemon-agent serve`

## Components

1. **Emulator Server** (`pokemon-agent serve`)
   - Runs the Pokemon Red ROM and exposes a HTTP API (`localhost:8765`) for state/control.
   - Required for both Laya/Hermes play and SB3 training.

2. **Play Loop** (`pokemon-agent play`)
   - Combines Laya (fast, local) and Hermes (LLM, slower) agents.
   - Laya runs every turn; Hermes escalates after 45 seconds of no progress.
   - Configured via environment variables:
     - `POKEMON_HERMES_MODEL`: e.g., `nvidia/nemotron-3.5-lightning-30b-a3b`
     - `POKEMON_HERMES_PROVIDER`: e.g., `nvidia`
   - Flags:
     - `--host`, `--port`: API endpoint
     - `--laya`: enable Laya
     - `--stall-seconds 45`: wait before escalating to Hermes
     - `--hermes-turns 8`: max Hermes turns per escalation
     - `--turn-timeout 360`: seconds to wait for a Hermes turn (increased to accommodate slower LLMs)

3. **SB3 Training** (`baseline_fast_v2.py`)
   - Trains a PPO policy using the custom Gym environment.
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

### 3. Start the Play Loop (Laya + Hermes)
In another terminal/screen session:
```bash
cd /home/vlad/pokemon-agent
source /home/vlad/hermes_venv/bin/activate
export POKEMON_HERMES_MODEL="nvidia/nemotron-3.5-lightning-30b-a3b"
export POKEMON_HERMES_PROVIDER="nvidia"
pokemon-agent play --host localhost --port 8765 --laya --stall-seconds 45 --hermes-turns 8 --turn-timeout 360
```

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

Enjoy watching the agent explore Kanto!