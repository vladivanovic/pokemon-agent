from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from red_gym_env_v2 import RedGymEnv

import io, os, threading, urllib.request, json
from PIL import Image

WEBHOOK = os.environ.get("POKE_WEBHOOK_URL")

def latest_checkpoint(sess_path: Path) -> Path | None:
    zips = list((sess_path / "checkpoints").glob("*.zip"))
    zips += list(sess_path.glob("poke_final.zip"))
    return max(zips, key=lambda p: p.stat().st_mtime) if zips else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sess-id", default="runs")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--rom", default="../PokemonRed.gb")
    p.add_argument("--init-state", default="../init.state")
    p.add_argument("--speed", type=int, default=3,
                   help="Emulation speed multiplier; 0 = unbounded")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--reload-every", type=int, default=0,
                   help="Reload the newest checkpoint every N steps (0 = never)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    sess_path = Path(args.sess_id)
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(sess_path)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"no checkpoint found under {sess_path}")
    print(f"loading {ckpt}")

    config = {
        "headless": False,          # -> window="SDL2" in RedGymEnv
        "save_final_state": False,
        "print_rewards": True,
        "action_freq": 24,
        "init_state": args.init_state,
        "max_steps": 10_000_000,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path / "_watch",
        "gb_path": args.rom,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    # Training logged "Wrapping the env in a VecTransposeImage", so the saved
    # policy expects channels-first images. Replicate that wrapper or the
    # screens/map tensors arrive with the wrong layout.
    venv = VecTransposeImage(DummyVecEnv([lambda: RedGymEnv(config)]))
    for pb in venv.get_attr("pyboy"):
        pb.set_emulation_speed(args.speed)

    model = PPO.load(str(ckpt), env=venv, device=args.device)

    obs = venv.reset()
    step = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, _, _ = venv.step(action)   # VecEnv auto-resets on done
            step += 1
            # uncomment these lines to send frames to webhook
            #if args.post_every and step % args.post_every == 0:
            #    base = venv.unwrapped.envs[0].unwrapped
            #    post_frame(base, step, base.total_reward)
            if args.reload_every and step % args.reload_every == 0:
                newest = latest_checkpoint(sess_path)
                if newest and newest != ckpt:
                    ckpt = newest
                    print(f"\nreloading {ckpt}")
                    model = PPO.load(str(ckpt), env=venv, device=args.device)
    except KeyboardInterrupt:
        pass
    finally:
        venv.close()

def post_frame(env, step, reward):
    """Fire-and-forget PNG upload. Never blocks the emulator loop."""
    if not WEBHOOK:
        return
    frame = env.render()                     # (144,160,3) uint8 RGB
    img = Image.fromarray(frame).resize((480, 432), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    payload = buf.getvalue()

    def _send():
        try:
            # Discord/Slack multipart form upload
            boundary = "----pokeframe"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="content"\r\n\r\n'
                f"step {step} | reward {reward:.1f}\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="f.png"\r\n'
                f"Content-Type: image/png\r\n\r\n"
            ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                WEBHOOK, data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as exc:
            print(f"\nwebhook failed: {exc}")

    threading.Thread(target=_send, daemon=True).start()


if __name__ == "__main__":
    main()