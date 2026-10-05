from __future__ import annotations

import json
import logging
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from einops import repeat
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from skimage.transform import downscale_local_mean

from global_map import GLOBAL_MAP_SHAPE, local_to_global

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory map (Pokemon Red/Blue USA)
# https://datacrystal.romhacking.net/wiki/Pokemon_Red/Blue:RAM_map
# ---------------------------------------------------------------------------
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E          # exclusive; expanded to cover S.S. Anne
MUSEUM_TICKET = (0xD754, 0)

IN_BATTLE = 0xD057                # 0 none, 1 wild, 2 trainer
ENEMY_MON_HP = 0xCFE6             # 2 bytes, big endian
PARTY_COUNT = 0xD163
PARTY_SPECIES = (0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169)
PARTY_LEVELS = (0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268)
PARTY_HP = (0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248)
PARTY_MAX_HP = (0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269)
BADGES = 0xD356
POKEDEX_OWNED = (0xD2F7, 0xD30A)  # 19 bytes of owned flags
X_POS, Y_POS, MAP_N = 0xD362, 0xD361, 0xD35E
REDS_HOUSE_2F = 38                # map id the game starts on

# built-in on 3.10+; user is on 3.12
def popcount(v: int) -> int:
    return int(v).bit_count()


class RedGymEnv(Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config=None, render_mode: str | None = "rgb_array"):
        config = dict(config or {})

        # ---- paths / io -----------------------------------------------------
        self.s_path = Path(config.get("session_path", "sessions/default"))
        self.s_path.mkdir(parents=True, exist_ok=True)
        self.gb_path = config["gb_path"]                     # required
        init_state = config.get("init_state")
        self.init_state = Path(init_state) if init_state else None

        # ---- run config -----------------------------------------------------
        self.save_final_state = bool(config.get("save_final_state", False))
        self.print_rewards = bool(config.get("print_rewards", False))
        self.headless = bool(config.get("headless", True))
        self.act_freq = int(config.get("action_freq", 24))
        self.press_step = int(config.get("press_step", 8))
        if self.act_freq <= self.press_step:
            raise ValueError(
                f"action_freq ({self.act_freq}) must exceed press_step ({self.press_step})"
            )
        self.max_steps = int(config.get("max_steps", 20480))
        self.save_video = bool(config.get("save_video", False))
        self.fast_video = bool(config.get("fast_video", True))
        self.explore_weight = float(config.get("explore_weight", 1.0))
        self.reward_scale = float(config.get("reward_scale", 1.0))
        self.terminate_on_wipe = bool(config.get("terminate_on_wipe", False))
        self.instance_id = str(config.get("instance_id", str(uuid.uuid4())[:8]))
        self.render_mode = render_mode

        # debug / logging knobs (off by default: these are hot-path expensive)
        self.debug_events = bool(config.get("debug_events", False))
        self.event_scan_freq = int(config.get("event_scan_freq", 1000))
        self.stats_log_freq = int(config.get("stats_log_freq", 1))
        self.stats_maxlen = int(config.get("stats_maxlen", 20000))

        # state-bootstrap knobs (only used when init_state is missing/stale)
        self.boot_max_frames = int(config.get("boot_max_frames", 12000))
        self.boot_settle_frames = int(config.get("boot_settle_frames", 240))

        self.frame_stacks = 3
        self.action_history_len = int(config.get("action_history_len", 3))
        self.enc_freqs = 8
        self.coords_pad = 12
        self.output_shape = (72, 80, self.frame_stacks)
        self.stuck_threshold = int(config.get("stuck_threshold", 600))
        self.env_rank = int(config.get("env_id", 0))
        self.explore_export_path = str(config.get(
            "explore_export_path", "/home/vlad/pokemon-agent/explore_map_union.json"))
        # 12 parallel envs share one export path -> writes race and corrupt the
        # file (two concatenated JSON dumps observed live). Per-rank files fix
        # both: no collision, and the union is a true merge across envs.
        self.explore_export_path = (
            f"/home/vlad/pokemon-agent/explore_map_union_rank{self.env_rank}.json")
        self.explore_export_every = int(config.get("explore_export_every", 5000))

        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0

        self.essential_map_locations = {
            v: i for i, v in enumerate(
                [40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65]
            )
        }

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START,
        ]

        self.event_names = self._load_event_names()

        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.reward_range = (-float("inf"), float("inf"))
        self.observation_space = spaces.Dict({
            "screens": spaces.Box(
                low=0, high=255, shape=self.output_shape, dtype=np.uint8),
            "health": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "level": spaces.Box(
                low=-1.0, high=1.0, shape=(self.enc_freqs,), dtype=np.float32),
            "badges": spaces.MultiBinary(8),
            "events": spaces.MultiBinary((EVENT_FLAGS_END - EVENT_FLAGS_START) * 8),
            "map": spaces.Box(
                low=0, high=255,
                shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                dtype=np.uint8),
            "recent_actions": spaces.MultiDiscrete(
                [len(self.valid_actions)] * self.action_history_len),
        })

        self.pyboy = None
        self._start_emulator()

        # every attribute read outside reset() must exist before the first reset
        self._init_episode_vars()

    # ------------------------------------------------------------------ setup
    def _load_event_names(self):
        path = Path(__file__).parent / "events.json"
        if not path.exists():
            logger.warning("events.json not found at %s; event naming disabled", path)
            return {}
        with open(path) as f:
            return json.load(f)

    def _start_emulator(self):
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
        self.pyboy = PyBoy(
            str(self.gb_path),
            window="null" if self.headless else "SDL2",
        )
        # 0 == unbounded. Without this, headless training runs at real time.
        self.pyboy.set_emulation_speed(0 if self.headless else 6)

    def _init_episode_vars(self):
        self.init_map_mem()
        self.agent_stats = deque(maxlen=self.stats_maxlen)
        self.explore_map = np.zeros(GLOBAL_MAP_SHAPE, dtype=np.uint8)
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.action_history_len,), dtype=np.int64)

        self.max_event_rew = 0.0
        self.max_level_rew = 0.0
        self.last_health = 1.0
        self.total_healing_rew = 0.0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        self.last_in_battle = False
        self.last_enemy_hp = 0
        self.enemy_fainted_pending = False
        self.battle_won_count = 0
        self.battles_entered = 0
        self.base_event_flags = 0
        self.current_event_flags_set = {}
        self.max_map_progress = 0
        self.progress_reward = {}
        self.total_reward = 0.0
        self.last_step_reward = 0.0
        self._faint_steps = 0
        self.skip_next_heal = False

    def init_map_mem(self):
        self.seen_coords = {}

    # --------------------------------------------------- initial state / boot
    def _load_initial_state(self):
        """Load init_state, or bootstrap one if it is missing/incompatible.

        PyBoy save states are version-locked, so a state written by 2.4.0
        cannot be read by 2.7.0. On failure we rebuild the emulator (a partial
        load leaves it corrupt), skip the intro, and rewrite the state file.
        """
        if self.init_state is not None and self.init_state.exists():
            try:
                with open(self.init_state, "rb") as f:
                    self.pyboy.load_state(f)
                return
            except Exception as exc:
                logger.warning(
                    "Could not load %s (%s). Rebuilding a fresh state — this is "
                    "expected after a PyBoy version bump.", self.init_state, exc
                )
                self._start_emulator()

        self._boot_and_skip_intro()
        if self.init_state is not None:
            self.init_state.parent.mkdir(parents=True, exist_ok=True)
            with open(self.init_state, "wb") as f:
                self.pyboy.save_state(f)
            logger.warning("Wrote a new init state to %s", self.init_state)

    def _boot_and_skip_intro(self):
        """Best-effort automatic new game.

        Mashes START/A until the player is standing in Red's bedroom. Blind
        A-mashing accepts the default naming screen, so the player and rival
        end up named 'AAAAAAA'. If you care about the start point, use
        make_init_state.py instead and point init_state at its output.
        """
        buttons = [
            (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
            (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
        ]
        frames = 0
        i = 0
        while frames < self.boot_max_frames:
            press, release = buttons[i % len(buttons)]
            self.pyboy.send_input(press)
            self.pyboy.tick(6, False)
            self.pyboy.send_input(release)
            self.pyboy.tick(10, False)
            frames += 16
            i += 1
            if self.read_m(MAP_N) == REDS_HOUSE_2F and self.read_m(PARTY_COUNT) == 0:
                break
        else:
            logger.warning(
                "Intro skip did not reach the overworld in %d frames; "
                "the emulator may be sitting on a menu.", self.boot_max_frames
            )
        self.pyboy.tick(self.boot_settle_frames, False)

    # ------------------------------------------------------------------ gym
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._close_video()
        self._load_initial_state()
        self._init_episode_vars()

        self.base_event_flags = int(
            np.unpackbits(self.read_range(EVENT_FLAGS_START, EVENT_FLAGS_END)).sum()
        )
        self.last_health = self.read_hp_fraction()
        self.party_size = self.read_m(PARTY_COUNT)
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum(self.progress_reward.values())
        self.reset_count += 1

        return self._get_obs(), {}

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)
        self.step_count += 1

        self.update_battle_tracking()
        self.update_recent_actions(action)
        self.update_seen_coords()
        self.update_explore_map()
        if self.step_count % self.explore_export_every == 0:
            self.export_explore_map_union()
        self.update_heal_reward()
        self.party_size = self.read_m(PARTY_COUNT)
        self.update_map_progress()

        reward = self.update_reward()
        self.last_health = self.read_hp_fraction()

        obs = self._get_obs()
        terminated = self.check_terminated()
        truncated = self.step_count >= self.max_steps

        if self.step_count % self.stats_log_freq == 0:
            self.append_agent_stats(action)
        if self.debug_events and self.step_count % self.event_scan_freq == 0:
            self.scan_event_flags()

        self.save_and_print_info(terminated or truncated, obs)
        if terminated or truncated:
            self._close_video()

        if terminated or truncated:
            print(f"\n[DONE] term={terminated} trunc={truncated} "
                  f"step={self.step_count} party={self.party_size} "
                  f"hp={self.read_hp_fraction():.3f} "
                  f"maxhp={self.party_max_hp_sum()} "
                  f"twipe={self.terminate_on_wipe}", flush=True)

        return obs, reward, terminated, truncated, {}

    def close(self):
        self._close_video()
        if self.pyboy is not None:
            self.pyboy.stop(save=False)
            self.pyboy = None

    def render(self):
        """Gymnasium-compliant render: full-res RGB frame."""
        return self.pyboy.screen.ndarray[:, :, :3].copy()

    # ------------------------------------------------------------ observation
    def _get_screen(self, reduce_res=True):
        frame = self.pyboy.screen.ndarray[:, :, 0:1].copy()
        if reduce_res:
            frame = np.round(
                downscale_local_mean(frame, (2, 2, 1))
            ).clip(0, 255).astype(np.uint8)
        return frame

    def _get_obs(self):
        self.update_recent_screens(self._get_screen(reduce_res=True))
        level_sum = 0.02 * self.get_levels_sum()
        return {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()], dtype=np.float32),
            "level": self.fourier_encode(level_sum),
            "badges": np.unpackbits(
                np.array([self.read_m(BADGES)], dtype=np.uint8)).astype(np.int8),
            "events": self.read_event_bits(),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.recent_actions,
        }

    def update_recent_screens(self, cur_screen):
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = cur_screen[:, :, 0]

    def update_recent_actions(self, action):
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = int(action)

    def fourier_encode(self, val):
        return np.sin(val * 2 ** np.arange(self.enc_freqs)).astype(np.float32)

    # ------------------------------------------------------------- emulation
    def run_action_on_emulator(self, action):
        render_screen = self.save_video or not self.headless
        press = self.valid_actions[action]
        release = self.release_actions[action]

        if self.save_video and not self.fast_video:
            # every emulated frame becomes a video frame
            self.pyboy.send_input(press)
            for i in range(self.act_freq):
                if i == self.press_step:
                    self.pyboy.send_input(release)
                self.pyboy.tick(1, True)
                self.add_video_frame()
            return

        self.pyboy.send_input(press)
        self.pyboy.tick(self.press_step, render_screen)
        self.pyboy.send_input(release)
        remaining = max(0, self.act_freq - self.press_step - 1)
        if remaining:
            self.pyboy.tick(remaining, render_screen)
        self.pyboy.tick(1, True)  # final frame always rendered: it feeds the obs
        if self.save_video and self.fast_video:
            self.add_video_frame()

    # -------------------------------------------------------------- memory io
    def read_m(self, addr):
        return self.pyboy.memory[addr]

    def read_range(self, start, end):
        return np.frombuffer(
            bytes(self.pyboy.memory[start:end]), dtype=np.uint8
        )

    def read_bit(self, addr, bit: int) -> bool:
        return bool((self.read_m(addr) >> bit) & 1)

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)

    def read_event_bits(self):
        return np.unpackbits(
            self.read_range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        ).astype(np.int8)

    def read_party(self):
        return [self.read_m(a) for a in PARTY_SPECIES]

    def get_game_coords(self):
        return (self.read_m(X_POS), self.read_m(Y_POS), self.read_m(MAP_N))

    def read_hp_fraction(self):
        hp_sum = sum(self.read_hp(a) for a in PARTY_HP)
        max_hp_sum = max(sum(self.read_hp(a) for a in PARTY_MAX_HP), 1)
        return float(np.clip(hp_sum / max_hp_sum, 0.0, 1.0))

    def get_badges(self):
        return popcount(self.read_m(BADGES))

    def get_pokedex_owned(self):
        return int(np.unpackbits(self.read_range(*POKEDEX_OWNED)).sum())

    # ---------------------------------------------------------- explore map
    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        gy, gx = local_to_global(y_pos, x_pos, map_n)
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        return None

    def update_explore_map(self):
        c = self.get_global_coords()
        if c is None:
            logger.debug("coord out of bounds: game=%s", self.get_game_coords())
            return
        self.explore_map[c[0], c[1]] = 255


    def export_explore_map_union(self):
        """Dump union explored-cells JSON for external consumers (Laya/Hermes)."""
        try:
            coords = np.argwhere(self.explore_map > 0)
            payload = {"updated_at_step": self.step_count,
                       "cells": [[int(y), int(x)] for y, x in coords]}
            tmp = self.explore_export_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f)
            Path(tmp).replace(self.explore_export_path)
        except Exception:
            logger.exception("explore map export failed")

    def get_explore_map(self):
        pad = self.coords_pad
        out = np.zeros((pad * 2, pad * 2), dtype=np.uint8)
        c = self.get_global_coords()
        if c is not None:
            gy, gx = c
            y0, y1 = gy - pad, gy + pad
            x0, x1 = gx - pad, gx + pad
            sy0, sx0 = max(0, y0), max(0, x0)
            sy1 = min(self.explore_map.shape[0], y1)
            sx1 = min(self.explore_map.shape[1], x1)
            if sy1 > sy0 and sx1 > sx0:
                out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                    self.explore_map[sy0:sy1, sx0:sx1]
        return repeat(out, "h w -> (h h2) (w w2)", h2=2, w2=2)

    def update_seen_coords(self):
        if self.read_m(IN_BATTLE) != 0:
            return
        x_pos, y_pos, map_n = self.get_game_coords()
        key = f"x:{x_pos} y:{y_pos} m:{map_n}"
        self.seen_coords[key] = self.seen_coords.get(key, 0) + 1

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        key = f"x:{x_pos} y:{y_pos} m:{map_n}"
        return 0 if self.seen_coords.get(key, 0) < self.stuck_threshold else 1

    def update_map_progress(self):
        self.max_map_progress = max(
            self.max_map_progress, self.get_map_progress(self.read_m(MAP_N))
        )

    def get_map_progress(self, map_idx):
        return self.essential_map_locations.get(map_idx, -1)

    # ------------------------------------------------------------- tracking
    def update_battle_tracking(self):
        """Count only genuine victories.

        The old check ('was in battle, now isn't, HP > 0') also fired on
        running away and on menu exits, which at 50 pts a pop was the biggest
        exploit in the reward function. We now require the enemy's HP to have
        actually hit zero during the battle.
        """
        in_battle = self.read_m(IN_BATTLE) != 0

        if in_battle:
            if not self.last_in_battle:
                self.battles_entered += 1
                self.enemy_fainted_pending = False
                self.last_enemy_hp = self.read_hp(ENEMY_MON_HP)
            enemy_hp = self.read_hp(ENEMY_MON_HP)
            if self.last_enemy_hp > 0 and enemy_hp == 0:
                self.enemy_fainted_pending = True
            self.last_enemy_hp = enemy_hp
        else:
            if (self.last_in_battle and self.enemy_fainted_pending
                    and self.read_hp_fraction() > 0):
                self.battle_won_count += 1
            self.enemy_fainted_pending = False
            self.last_enemy_hp = 0

        self.last_in_battle = in_battle

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        party_unchanged = self.read_m(PARTY_COUNT) == self.party_size

        if (self.last_health > 0 and cur_health <= 0
                and self.party_max_hp_sum() > 0):
            self.died_count += 1
            self.skip_next_heal = True      # the blackout auto-heal
        elif cur_health > self.last_health and party_unchanged:
            if self.skip_next_heal:
                self.skip_next_heal = False
            else:
                self.total_healing_rew += cur_health - self.last_health

    def party_max_hp_sum(self):
        return sum(self.read_hp(a) for a in PARTY_MAX_HP)

    def check_terminated(self):
        """True only on a confirmed party wipe.

        The party struct is written over several frames, so wPartyCount can be
        nonzero while max-HP is still 0. Requiring valid max-HP plus a few
        consecutive zero-HP steps avoids firing on that write window.
        """
        if not self.terminate_on_wipe:
            return False        # blackout auto-heals and respawns; let it play out
        if self.party_size == 0 or self.party_max_hp_sum() == 0:
            self._faint_steps = 0
            return False
        if self.read_hp_fraction() > 0.0:
            self._faint_steps = 0
            return False
        self._faint_steps += 1
        return self._faint_steps >= 4

    # -------------------------------------------------------------- rewards
    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [max(self.read_m(a) - min_poke_level, 0) for a in PARTY_LEVELS]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        explore_thresh, scale_factor = 12, 8
        level_sum = self.get_levels_sum()
        scaled = (level_sum if level_sum < explore_thresh
                  else (level_sum - explore_thresh) / scale_factor + explore_thresh)
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_all_events_reward(self):
        total = int(np.unpackbits(
            self.read_range(EVENT_FLAGS_START, EVENT_FLAGS_END)).sum())
        return max(
            total - self.base_event_flags
            - int(self.read_bit(MUSEUM_TICKET[0], MUSEUM_TICKET[1])),
            0,
        )

    def update_max_event_rew(self):
        self.max_event_rew = max(self.get_all_events_reward(), self.max_event_rew)
        return self.max_event_rew

    def get_game_state_reward(self):
        """Monotonic (non-decreasing) reward components only.

        Anything that can go back down must not live here — update_reward()
        differences this dict, so a term that falls refunds its own reward and
        becomes farmable. Transient terms go in get_instantaneous_reward().
        """
        return {
            "event": self.reward_scale * self.update_max_event_rew() * 4,
            "level": self.reward_scale * self.get_levels_reward() * 2,
            "heal": self.reward_scale * self.total_healing_rew * 10,
            "badge": self.reward_scale * self.get_badges() * 10,
            "explore": (self.reward_scale * self.explore_weight
                        * len(self.seen_coords) * 0.5),
            "pokedex": self.reward_scale * self.get_pokedex_owned() * 2,
            "battle": 0.0 * self.battles_entered,
            "win": self.reward_scale * (self.battle_won_count ** 0.5) * 8,
            "party": self.reward_scale * self.read_m(PARTY_COUNT) * 10,
            "map_progress": self.reward_scale * self.max_map_progress * 100,
        }

    def get_instantaneous_reward(self):
        return self.reward_scale * self.get_current_coord_count_reward() * -0.05

    def update_reward(self):
        self.progress_reward = self.get_game_state_reward()
        new_total = sum(self.progress_reward.values())
        delta = new_total - self.total_reward
        self.total_reward = new_total
        self.last_step_reward = delta + self.get_instantaneous_reward()
        return self.last_step_reward

    # ------------------------------------------------------------- reporting
    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [self.read_m(a) for a in PARTY_LEVELS]
        self.agent_stats.append({
            "step": self.step_count,
            "x": x_pos, "y": y_pos, "map": map_n,
            "max_map_progress": self.max_map_progress,
            "last_action": int(action),
            "pcount": self.read_m(PARTY_COUNT),
            "levels": levels,
            "levels_sum": sum(levels),
            "ptypes": self.read_party(),
            "hp": self.read_hp_fraction(),
            "coord_count": len(self.seen_coords),
            "deaths": self.died_count,
            "badge": self.get_badges(),
            "event": self.progress_reward.get("event", 0),
            "healr": self.total_healing_rew,
            "wins": self.battle_won_count,
        })

    def scan_event_flags(self):
        """Map set event flags to names.

        Bit indices in event_constants.asm are LSB-first, but enumerating an
        f'{val:08b}' string walks MSB-first — that inversion is why the old
        lookup never matched anything.
        """
        for address in range(EVENT_FLAGS_START, EVENT_FLAGS_END):
            val = self.read_m(address)
            if not val:
                continue
            for bit in range(8):
                if (val >> bit) & 1:
                    key = f"0x{address:X}-{bit}"
                    if key in self.event_names:
                        self.current_event_flags_set[key] = self.event_names[key]
                    else:
                        logger.debug("unnamed event flag: %s", key)

    def save_and_print_info(self, done, obs):
        if self.print_rewards:
            prog = " ".join(f"{k}: {v:5.2f}" for k, v in self.progress_reward.items())
            print(f"\rstep: {self.step_count:6d} {prog} sum: {self.total_reward:5.2f}",
                  end="", flush=True)

        if not (done and self.save_final_state):
            return

        import matplotlib.pyplot as plt  # imported lazily: heavy per worker
        if self.print_rewards:
            print("", flush=True)
        fs_path = self.s_path / "final_states"
        fs_path.mkdir(parents=True, exist_ok=True)
        stem = f"frame_r{self.total_reward:.4f}_{self.reset_count}"
        plt.imsave(fs_path / f"{stem}_explore_map.jpeg", obs["map"][:, :, 0])
        plt.imsave(fs_path / f"{stem}_full_explore_map.jpeg", self.explore_map)
        plt.imsave(fs_path / f"{stem}_full.jpeg",
                   self._get_screen(reduce_res=False)[:, :, 0])

    # ----------------------------------------------------------------- video
    def start_video(self):
        import mediapy as media  # imported lazily
        self._close_video()
        base_dir = self.s_path / "rollouts"
        base_dir.mkdir(parents=True, exist_ok=True)
        tag = f"reset_{self.reset_count}_id{self.instance_id}.mp4"

        self.full_frame_writer = media.VideoWriter(
            base_dir / f"full_{tag}", (144, 160), fps=60, input_format="gray")
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / f"model_{tag}", self.output_shape[:2], fps=60,
            input_format="gray")
        self.model_frame_writer.__enter__()
        self.map_frame_writer = media.VideoWriter(
            base_dir / f"map_{tag}",
            (self.coords_pad * 4, self.coords_pad * 4), fps=60,
            input_format="gray")
        self.map_frame_writer.__enter__()

    def add_video_frame(self):
        if self.full_frame_writer is None:
            return
        # telemetry is burned into the video only, never into the observation
        full = self._get_screen(reduce_res=False)[:, :, 0]
        if self.print_rewards:
            import cv2  # imported lazily
            full = np.ascontiguousarray(full)
            cv2.putText(
                full,
                f"HP: {self.read_hp_fraction():.2f} R: {self.total_reward:.1f}",
                (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        self.full_frame_writer.add_image(full)
        self.model_frame_writer.add_image(self._get_screen(reduce_res=True)[:, :, 0])
        self.map_frame_writer.add_image(self.get_explore_map())

    def _close_video(self):
        for name in ("full_frame_writer", "model_frame_writer", "map_frame_writer"):
            writer = getattr(self, name, None)
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    logger.debug("writer %s failed to close", name, exc_info=True)
                setattr(self, name, None)