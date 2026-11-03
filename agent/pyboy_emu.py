"""PyBoy backend: runs the agent against the real game.

Implements the same verbs as SimEmulator (press / read_state / save /
load / screenshot) so the Executor does not know or care which backend
it is driving.

Press semantics are *tile steps*, matching the simulator: a direction
press holds the button until the RAM coordinates change by one tile and
the walk settles (wWalkCounter back to 0). A press that changes nothing
(dialog open, wall, scripted event) simply times out - the executor's
stuck detection and the runner's dialog handling deal with that, exactly
as designed.

Booting a fresh game still needs the scripted intro (Oak's lecture and
the name-entry screens are menus, not gameplay); see run_real.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pyboy import PyBoy

from .state import ADDR_WALK_COUNTER, GameState, read_state

_STEP_TIMEOUT = 90      # frames to wait for one tile step
_BUTTON_HOLD = 5        # frames a button must be held to register (Gen 1)


class PyBoyEmulator:
    def __init__(self, rom_path: str | Path):
        self.pb = PyBoy(str(rom_path), window="null")
        self.press_count = 0
        self._last_map: Optional[int] = None
        self._world = None  # lazily loaded; only for warp-settle bounds

    # -- plumbing -------------------------------------------------------------
    def _read(self, address: int, length: int) -> bytes:
        m = self.pb.memory
        return bytes(m[address + i] for i in range(length))

    def tick(self, frames: int = 1) -> None:
        for _ in range(frames):
            self.pb.tick()

    def read_state(self) -> GameState:
        st = read_state(self._read)
        if self._world is None:
            from .world import default_world  # local: avoids a cycle
            self._world = default_world()
        gm = self._world.maps.get(st.map_id)
        out_of_bounds = gm is not None and not gm.in_bounds(st.x, st.y)
        if out_of_bounds or (self._last_map is not None
                             and st.map_id != self._last_map):
            # A warp just fired. The game writes the destination map id
            # and the destination coordinates at different points of the
            # transition, so a raw read can pair the NEW map with the
            # OLD map's coordinates (observed live: Pallet Town wearing
            # the house exit mat's (2, 7), and Viridian Mart wearing
            # (29, 19) - outside its own 8x8 grid). A settled position
            # is always inside its map, so: tick until the position is
            # in-bounds AND stable across consecutive reads.
            prev = None
            stable = 0
            for i in range(80):
                self.tick(4)
                st = read_state(self._read)
                gm = self._world.maps.get(st.map_id)
                cur = (st.map_id, st.x, st.y)
                if gm is not None and gm.in_bounds(st.x, st.y) and cur == prev:
                    stable += 1
                    # ~50 frames minimum, and the same position three
                    # reads running: stale coords can look stable for a
                    # few frames mid-fade (and can even be in-bounds for
                    # the new map, e.g. the Mart's (3, 7) in Viridian).
                    if stable >= 3 and i >= 12:
                        break
                else:
                    stable = 0
                prev = cur
        self._last_map = st.map_id
        return st

    def screenshot(self, path: str | Path) -> None:
        self.pb.screen.image.save(str(path))

    def save_state(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            self.pb.save_state(f)

    def load_state(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.pb.load_state(f)

    def stop(self) -> None:
        self.pb.stop()

    # -- verbs ------------------------------------------------------------------
    def press(self, button: str) -> None:
        self.press_count += 1
        if button in ("up", "down", "left", "right"):
            self._step(button)
        else:
            self.pb.button_press(button)
            self.tick(_BUTTON_HOLD)
            self.pb.button_release(button)
            self.tick(12)

    def _step(self, direction: str) -> None:
        before = self.read_state()
        start = (before.map_id, before.x, before.y)
        self.pb.button_press(direction)
        moved = False
        for _ in range(_STEP_TIMEOUT):
            self.pb.tick()
            st = self.read_state()
            if (st.map_id, st.x, st.y) != start:
                moved = True
                break
            if st.in_battle:
                break
        self.pb.button_release(direction)
        if moved:
            # Let the step finish so the next press starts clean.
            for _ in range(30):
                if self._read(ADDR_WALK_COUNTER, 1)[0] == 0:
                    break
                self.pb.tick()
            # If the step landed on a warp tile, the warp fires with a
            # fade that takes ~1-2 seconds of frames; callers that check
            # the map immediately race it (a whole family of desync
            # bugs). Block until the map actually changes.
            if self._world is None:
                from .world import default_world
                self._world = default_world()
            st = read_state(self._read)
            gm = self._world.maps.get(st.map_id)
            if gm is not None and any((wp.x, wp.y) == (st.x, st.y)
                                      for wp in gm.warps):
                for _ in range(60):
                    self.tick(3)
                    if read_state(self._read).map_id != st.map_id:
                        break
                self.read_state()  # settle the landing position
        else:
            self.tick(4)

    def mash_a(self, times: int = 6, settle: int = 30) -> None:
        """Advance dialog: press A repeatedly with time for text to print."""
        for _ in range(times):
            self.pb.button_press("a")
            self.tick(_BUTTON_HOLD)
            self.pb.button_release("a")
            self.tick(settle)

    def wait_free(self, max_rounds: int = 80, stable_needed: int = 3,
                  tick_len: int = 45) -> bool:
        """Wait until the game is genuinely idle: no text box open and
        the player not being walked by a scripted scene (Oak's
        intercept, the Mart parcel handover and the rival's scenes all
        alternate dialog beats with scripted movement that eats input).
        Returns True when control is really ours.

        stable_needed counts consecutive quiet checks (tick_len frames
        apart). The default (3) suits short scenes; Oak's intercept
        pauses mid-scene with the player briefly able to move, so its
        caller passes ~10: only the scene's true end stays quiet that
        long."""
        stable = 0
        last = None
        for _ in range(max_rounds):
            st = self.read_state()
            if st.in_dialog:
                self.close_dialog(max_presses=4)
                stable = 0
                last = None
                continue
            if st.in_battle:
                return False
            pos = (st.map_id, st.x, st.y)
            if pos == last:
                stable += 1
                if stable >= stable_needed:
                    return True
            else:
                stable = 0
                last = pos
            self.tick(tick_len)
        return False

    def close_dialog(self, max_presses: int = 30) -> int:
        """Press A until the text box is actually gone, then STOP.

        The stop matters: a blind extra press in front of an examinable
        object just opens its text again (the bedroom SNES trap).
        Returns the number of presses used."""
        presses = 0
        while presses < max_presses and self.read_state().in_dialog:
            self.pb.button_press("a")
            self.tick(_BUTTON_HOLD)
            self.pb.button_release("a")
            self.tick(30)
            presses += 1
        return presses

    def fly_to(self, dest_map: int) -> None:
        raise NotImplementedError(
            "Fly on hardware is a menu macro (START -> POKEMON -> Fly); "
            "not implemented yet")
