# validate_env.py
from pathlib import Path
import numpy as np
from red_gym_env_v2 import (RedGymEnv, PARTY_COUNT, BADGES, IN_BATTLE,
                            EVENT_FLAGS_START, EVENT_FLAGS_END)

env = RedGymEnv({
    "gb_path": "../PokemonRed.gb", "init_state": "../init.state",
    "session_path": Path("runs/_probe"), "headless": True,
    "max_steps": 100000, "reward_scale": 0.5, "explore_weight": 1.0,
})
obs, _ = env.reset()
fails = []

def chk(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fails.append(name)

chk("obs in observation_space", env.observation_space.contains(obs))
for k, v in obs.items():
    sub = env.observation_space[k]
    chk(f"  dtype {k}", v.dtype == sub.dtype, f"{v.dtype} vs {sub.dtype}")
chk("terminate_on_wipe is False", env.terminate_on_wipe is False)
chk("explore_map shape", env.get_explore_map().shape ==
    (env.coords_pad * 4, env.coords_pad * 4))
chk("event bits length", len(env.read_event_bits()) ==
    (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8)

start_coords = len(env.seen_coords)
for _ in range(3000):
    env.step(env.action_space.sample())

hp = env.read_hp_fraction()
chk("seen_coords grows", len(env.seen_coords) > start_coords + 20,
    f"{start_coords} -> {len(env.seen_coords)}")
chk("hp_fraction in [0,1]", 0.0 <= hp <= 1.0, f"{hp:.3f}")
chk("badges 0..8", 0 <= env.get_badges() <= 8)
chk("pokedex 0..151", 0 <= env.get_pokedex_owned() <= 151,
    str(env.get_pokedex_owned()))
chk("party 0..6", 0 <= env.read_m(PARTY_COUNT) <= 6)
chk("in_battle sane", env.read_m(IN_BATTLE) in (0, 1, 2))
chk("map_progress >= 0", env.max_map_progress >= 0)
chk("no death miscount", env.died_count == 0 or env.party_max_hp_sum() > 0,
    f"deaths={env.died_count}")

r = env.get_game_state_reward()
chk("all rewards finite", all(np.isfinite(v) for v in r.values()))
print("\nreward terms:")
for k, v in r.items():
    print(f"  {k:14s} {v:9.3f}{'   <-- ZERO' if v == 0 else ''}")
print(f"  {'stuck (inst)':14s} {env.get_instantaneous_reward():9.3f}")

print(f"\n{len(fails)} failure(s): {fails}" if fails else "\nall checks passed")
env.close()