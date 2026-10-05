from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from red_gym_env_v2 import RedGymEnv
from tensorboard_callback import TensorboardCallback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true",
                   help="Render env 0 in an SDL2 window (throttles that worker)")
    p.add_argument("--num-cpu", "--num_cpu", dest="num_cpu", type=int,
                   default=min(8, os.cpu_count() or 1),
                   help="Parallel environments (default: min(8, core count))")
    p.add_argument("--ep-length", type=int, default=2048 * 80,
                   help="Episode truncation length (max_steps), in agent steps")
    p.add_argument("--n-steps", type=int, default=1024,
                   help="PPO rollout length per env — independent of episode length")
    p.add_argument("--save-freq", type=int, default=50_000,
                   help="Checkpoint every N per-env steps")
    p.add_argument("--sess-id", default="runs")
    p.add_argument("--rom", default="../PokemonRed.gb")
    p.add_argument("--init-state", default="../init.state")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a .zip checkpoint to resume from")
    p.add_argument("--total-steps", type=int, default=100_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream", action="store_true",
                   help="Broadcast to the public stream endpoint")
    p.add_argument("--check-env", action="store_true",
                   help="Run gymnasium's env_checker on one instance and exit")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def make_env(rank, env_conf, seed=0, watch=False, stream=False):
    def _init():
        config = dict(env_conf)
        config["instance_id"] = f"env{rank:03d}"
        config["env_id"] = rank
        if watch and rank == 0:
            config["headless"] = False
        # only one worker may own the \r progress line; N of them share stdout
        config["print_rewards"] = bool(env_conf.get("print_rewards")) and rank == 0

        env = RedGymEnv(config)
        if stream:
            from stream_agent_wrapper import StreamWrapper
            env = StreamWrapper(env, stream_metadata={
                "user": "v2-default",
                "env_id": rank,
                "color": "#447799",
                "extra": "",
            })
        set_random_seed(seed + rank)
        return env
    return _init


def bootstrap_init_state(env_config):
    """Create/validate the init state in the parent, before workers spawn.

    PyBoy save states are version-locked, so a state written by 2.4.0 cannot be
    read by 2.7.0. RedGymEnv rebuilds one on load failure — but if N workers all
    discover that at once they race on the same file and corrupt it. Doing one
    probe reset here means every worker finds a valid state already on disk.
    """
    probe_cfg = dict(
        env_config,
        headless=True, save_video=False, print_rewards=False,
        session_path=Path(env_config["session_path"]) / "_probe",
    )
    env = RedGymEnv(probe_cfg)
    try:
        env.reset()
        print(f"init state OK: {env_config['init_state']}")
    finally:
        env.close()


def main():
    args = parse_args()
    sess_path = Path(args.sess_id)
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        "headless": True,
        "save_final_state": False,
        "action_freq": 24,
        "init_state": args.init_state,
        "max_steps": args.ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": args.rom,
        "reward_scale": 0.5,
        "explore_weight": 1.0,
        "debug_events": False,
        "stuck_threshold": 1200,
    }

    if not Path(args.rom).exists():
        sys.exit(f"ROM not found: {args.rom}")

    if args.check_env:
        from stable_baselines3.common.env_checker import check_env
        env = RedGymEnv(dict(env_config, print_rewards=False))
        try:
            check_env(env, warn=True)
            print("env_checker passed")
        finally:
            env.close()
        return

    bootstrap_init_state(env_config)

    num_cpu = max(1, args.num_cpu)
    n_steps = max(args.n_steps, 1)
    rollout = n_steps * num_cpu
    batch_size = args.batch_size
    if rollout % batch_size:
        print(f"warning: rollout {rollout} is not divisible by batch_size "
              f"{batch_size}; SB3 will drop the remainder each update")

    # DummyVecEnv is single-process: it would serialize all N emulators onto one
    # core, so --num-cpu had no effect on throughput.
    vec_cls = DummyVecEnv if num_cpu == 1 else SubprocVecEnv
    fns = [make_env(i, env_config, seed=args.seed, watch=args.watch,
                    stream=args.stream) for i in range(num_cpu)]
    env = vec_cls(fns) if vec_cls is DummyVecEnv else vec_cls(fns, start_method="spawn")
    env.seed(args.seed)                       # applied on the upcoming reset
    env = VecMonitor(env, filename=str(sess_path / "monitor.csv"))

    set_random_seed(args.seed)

    # CheckpointCallback counts vec-env steps, not agent timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq, 1),
        save_path=str(sess_path / "checkpoints"),
        name_prefix="poke",
    )
    callbacks = CallbackList([checkpoint_callback, TensorboardCallback(sess_path)])

    ckpt = args.checkpoint
    if ckpt and not ckpt.endswith(".zip"):
        ckpt += ".zip"
    resuming = bool(ckpt) and Path(ckpt).exists()

    if resuming:
        print(f"\nloading checkpoint {ckpt}")
        # kwargs override the saved hyperparameters, then _setup_model()
        # reallocates the rollout buffer to match. Poking .buffer_size /
        # .n_envs on a live buffer relies on reset() internals instead.
        model = PPO.load(
            ckpt, env=env, device=args.device,
            n_steps=n_steps, batch_size=batch_size, n_epochs=args.n_epochs,
            tensorboard_log=str(sess_path),
        )
    else:
        if ckpt:
            print(f"checkpoint {ckpt} not found; starting fresh")
        model = PPO(
            "MultiInputPolicy", env, verbose=1, device=args.device,
            n_steps=n_steps, batch_size=batch_size, n_epochs=args.n_epochs,
            gamma=0.999, ent_coef=0.01, seed=args.seed,
            tensorboard_log=str(sess_path),
        )

    print(f"envs={num_cpu} ({vec_cls.__name__})  n_steps={n_steps}  "
          f"rollout={rollout}  device={model.device}")
    print(model.policy)

    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=callbacks,
            tb_log_name="poke_ppo",
            reset_num_timesteps=not resuming,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; saving")
    finally:
        model.save(sess_path / "poke_final")
        env.close()


if __name__ == "__main__":
    main()