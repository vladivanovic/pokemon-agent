"""Unified emulator wrapper supporting PyBoy (GB/GBC) and PyGBA (GBA).

Provides a common interface for ROM loading, button input, frame advance,
screen capture, memory access, and save states across emulator backends.

Threading: these objects are NOT thread-safe in any useful sense. An internal
lock guards against the worst interleavings, but PyBoy's native core must have
a single owning thread. The server enforces this with a command queue — do not
call into an emulator from a request handler or an executor.
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

# SDL reads these when it initialises its drivers, which happens during the
# pyboy import chain — not when PyBoy() is constructed. They must therefore be
# set before any import below can reach pyboy. setdefault so a shell override
# still wins when debugging.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

logger = logging.getLogger("pokemon-agent.emulator")

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Emulator(ABC):
    """Abstract emulator interface.

    Subclasses wrap a concrete emulator library (PyBoy, PyGBA, etc.) and
    expose a uniform API for the agent layer.
    """

    BUTTONS: List[str] = ["a", "b", "start", "select", "up", "down", "left", "right"]

    def __init__(self) -> None:
        self.frame_count: int = 0
        self.rom_path: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------

    @abstractmethod
    def load(self, rom_path: str) -> None:
        """Load a ROM file and initialise the emulator."""

    @abstractmethod
    def close(self) -> None:
        """Shut down the emulator and release resources. Must be idempotent."""

    @property
    def is_loaded(self) -> bool:
        """True once a ROM is loaded and the backend is usable."""
        return self.rom_path is not None

    # -- input --------------------------------------------------------------

    @abstractmethod
    def press(self, button: str, frames: int = 1) -> None:
        """Press *button*, hold it for *frames* frames, then release.

        Parameters
        ----------
        button : str
            One of ``BUTTONS``.
        frames : int
            How many frames to hold the button before releasing. Gen 1 needs
            >= 4 for the vblank joypad poll to register reliably.
        """

    @abstractmethod
    def release_all(self) -> None:
        """Release every button."""

    # -- timing -------------------------------------------------------------

    @abstractmethod
    def tick(self, frames: int = 1, render_last: bool = True) -> None:
        """Advance the emulation by *frames* frames.

        Parameters
        ----------
        render_last : bool
            Render the final frame to the framebuffer. Pass False for bulk
            advancement — rendering is a significant per-frame cost and only
            the frame you are about to capture needs to be drawn. Note that
            ``get_screen()`` returns a stale image if the last tick did not
            render.
        """

    # -- video --------------------------------------------------------------

    @abstractmethod
    def get_screen(self) -> "Image.Image":
        """Return the current framebuffer as an owned RGB PIL Image."""

    # -- memory -------------------------------------------------------------

    @abstractmethod
    def read_u8(self, addr: int) -> int:
        """Read an unsigned 8-bit value from *addr*."""

    @abstractmethod
    def read_range(self, addr: int, size: int) -> bytes:
        """Read *size* bytes starting at *addr*."""

    def read_u16(self, addr: int) -> int:
        """Read an unsigned 16-bit LITTLE-endian value from *addr*."""
        raw = self.read_range(addr, 2)
        return raw[0] | (raw[1] << 8)

    def read_u32(self, addr: int) -> int:
        """Read an unsigned 32-bit LITTLE-endian value from *addr*."""
        return int.from_bytes(self.read_range(addr, 4), "little")

    def read_u16_be(self, addr: int) -> int:
        """Read an unsigned 16-bit BIG-endian value from *addr*.

        Gen 1 stores multi-byte values big-endian. Using read_u16() on Gen 1
        data silently byte-swaps it.
        """
        raw = self.read_range(addr, 2)
        return (raw[0] << 8) | raw[1]

    def read_u32_be(self, addr: int) -> int:
        """Read an unsigned 32-bit BIG-endian value from *addr*."""
        return int.from_bytes(self.read_range(addr, 4), "big")

    # -- save / load --------------------------------------------------------

    @abstractmethod
    def save_state(self, path: str) -> None:
        """Persist an emulator save-state to *path*."""

    @abstractmethod
    def load_state(self, path: str) -> None:
        """Restore an emulator save-state from *path*."""

    # -- info ---------------------------------------------------------------

    def get_info(self) -> Dict:
        """Return runtime metadata about the emulator."""
        return {
            "backend": self.__class__.__name__,
            "rom_path": self.rom_path,
            "frame_count": self.frame_count,
            "loaded": self.is_loaded,
        }


# ---------------------------------------------------------------------------
# PyBoy backend (Game Boy / Game Boy Color)
# ---------------------------------------------------------------------------

def _pyboy_sound_kwargs(cls) -> Dict[str, bool]:
    """Sound-related PyBoy kwargs set to False.

    PyBoy's __init__ is Cython-compiled and often has no introspectable
    signature, so fall back to probing the docstring for known names.
    """
    names = set()
    try:
        names = {n for n in inspect.signature(cls.__init__).parameters
                 if "sound" in n.lower() or "audio" in n.lower()}
    except (TypeError, ValueError):
        doc = (cls.__init__.__doc__ or "") + (cls.__doc__ or "")
        for cand in ("sound_emulated", "sound_volume", "sound"):
            if cand in doc:
                names.add(cand)
        if not names:
            names = {"sound", "sound_emulated"}
        logger.info("PyBoy signature not introspectable; trying %s", sorted(names))
    kwargs = {n: False for n in names}
    logger.info("PyBoy sound kwargs: %s", sorted(kwargs))
    return kwargs


class PyBoyEmulator(Emulator):
    """Wraps the *PyBoy* library for .gb / .gbc ROMs.

    Requires PyBoy >= 2.0 for ``window="null"`` and ``tick(count, render)``.
    Runs headless so no display server is required.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pyboy: Optional[object] = None
        # Reentrant because press() calls tick(). This is defence in depth
        # only: correctness depends on a single owning thread.
        self._lock = threading.RLock()

    # -- internal -----------------------------------------------------------

    def _pb(self):
        """Return the live PyBoy handle, or raise a message that says why not."""
        pb = self._pyboy
        if pb is None:
            raise RuntimeError("Emulator not loaded — call load(rom_path) first")
        return pb

    @property
    def is_loaded(self) -> bool:
        return self._pyboy is not None

    # -- lifecycle ----------------------------------------------------------

    def load(self, rom_path: str) -> None:
        """Load a Game Boy ROM via PyBoy."""
        try:
            from pyboy import PyBoy  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyBoy >= 2.0 is required for .gb/.gbc ROMs.  "
                'Install it with:  pip install "pyboy>=2.0"'
            ) from exc

        with self._lock:
            if self._pyboy is not None:
                raise RuntimeError("Emulator already loaded; call close() first")

            rom_path = str(Path(rom_path).expanduser().resolve())
            if not os.path.isfile(rom_path):
                raise FileNotFoundError(f"ROM not found: {rom_path}")

            kwargs = {"window": "null", **_pyboy_sound_kwargs(PyBoy)}
            try:
                pb = PyBoy(rom_path, **kwargs)
            except TypeError as exc:
                logger.warning("PyBoy rejected kwargs %s (%s); retrying minimal",
                               sorted(kwargs), exc)
                pb = PyBoy(rom_path, window="null", sound=False)

            # Remove the realtime throttle. Harmless if a null window already
            # runs unbounded; essential if it does not.
            try:
                pb.set_emulation_speed(0)
            except Exception as exc:
                logger.warning("set_emulation_speed(0) failed: %s", exc)

            self._pyboy = pb
            self.rom_path = rom_path
            self.frame_count = 0

        logger.info("PyBoy loaded: %s", rom_path)

    def close(self) -> None:
        """Stop PyBoy. Safe to call twice, and after a failed load()."""
        with self._lock:
            pb, self._pyboy = self._pyboy, None
        if pb is None:
            return
        try:
            pb.stop(save=False)  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("PyBoy.stop() failed: %s", exc)
        logger.info("PyBoy closed")

    # -- input --------------------------------------------------------------

    def press(self, button: str, frames: int = 1) -> None:
        """Press a button, hold it for *frames* frames, then release.

        Uses button_press/button_release rather than button(), which
        auto-releases after its own delay and is unreliable for the
        multi-frame holds Gen 1 movement needs.
        """
        button = button.lower()
        if button not in self.BUTTONS:
            raise ValueError(f"Unknown button '{button}'. Valid: {self.BUTTONS}")
        with self._lock:
            pb = self._pb()
            pb.button_press(button)  # type: ignore[union-attr]
            try:
                # No render while holding; the caller renders at capture time.
                self.tick(max(1, frames), render_last=False)
            finally:
                pb.button_release(button)  # type: ignore[union-attr]

    def release_all(self) -> None:
        """Release all buttons."""
        with self._lock:
            pb = self._pb()
            for btn in self.BUTTONS:
                try:
                    pb.button_release(btn)  # type: ignore[union-attr]
                except Exception:
                    pass

    # -- timing -------------------------------------------------------------

    def tick(self, frames: int = 1, render_last: bool = True) -> None:
        """Advance emulation by *frames* frames.

        Batched into at most two native calls. A per-frame Python loop costs
        N interpreter round-trips plus N full framebuffer renders; batching
        with render off is roughly an order of magnitude cheaper.

        PyBoy.tick() returns False when it wants to stop. Ignoring that means
        happily ticking a dead emulator forever, which presents as the game
        being extremely slow rather than as an error.
        """
        if frames < 1:
            return
        with self._lock:
            pb = self._pb()
            start = time.perf_counter()
            if frames > 1:
                if not pb.tick(frames - 1, False):  # type: ignore[union-attr]
                    raise RuntimeError("PyBoy requested stop during tick")
            if not pb.tick(1, render_last):  # type: ignore[union-attr]
                raise RuntimeError("PyBoy requested stop during tick")
            self.frame_count += frames
            elapsed = time.perf_counter() - start
        if elapsed > 0.1:
            logger.debug("tick(%d, render_last=%s) took %.3fs",
                         frames, render_last, elapsed)

    # -- video --------------------------------------------------------------

    def get_screen(self) -> "Image.Image":
        """Return the current frame as an owned RGB PIL Image (160x144).

        Copied because PyBoy's screen.image may share a buffer that the next
        tick mutates, and callers hand it to a PNG encoder. Converted to RGB
        because the alpha channel is constant and only inflates the PNG.

        Returns a STALE frame if the last tick used render_last=False — call
        ``tick(1, render_last=True)`` immediately before capturing.
        """
        with self._lock:
            return self._pb().screen.image.copy().convert("RGB")  # type: ignore[union-attr]

    # -- memory -------------------------------------------------------------

    def read_u8(self, addr: int) -> int:
        with self._lock:
            return self._pb().memory[addr] & 0xFF  # type: ignore[index]

    def read_range(self, addr: int, size: int) -> bytes:
        with self._lock:
            return bytes(self._pb().memory[addr:addr + size])  # type: ignore[index]

    # -- save / load --------------------------------------------------------

    def save_state(self, path: str) -> None:
        """Save emulator state to a file."""
        path = str(Path(path).expanduser().resolve())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            pb = self._pb()
            self.release_all()  # do not bake held buttons into the state
            with open(path, "wb") as f:
                pb.save_state(f)  # type: ignore[union-attr]
        logger.info("saved state: %s", path)

    def load_state(self, path: str) -> None:
        """Load emulator state from a file."""
        path = str(Path(path).expanduser().resolve())
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Save state not found: {path}")
        with self._lock:
            pb = self._pb()
            self.release_all()
            with open(path, "rb") as f:
                pb.load_state(f)  # type: ignore[union-attr]
            # Render one frame so get_screen() reflects the restored state.
            self.tick(1, render_last=True)
        logger.info("loaded state: %s", path)

    # -- info ---------------------------------------------------------------

    def get_info(self) -> Dict:
        info = super().get_info()
        info["platform"] = "GB/GBC"
        return info


# ---------------------------------------------------------------------------
# PyGBA backend (Game Boy Advance)
# ---------------------------------------------------------------------------

class PyGBAEmulator(Emulator):
    """Wraps the *PyGBA / mgba-py* library for .gba ROMs.

    Phase-2 backend, NOT exercised. The interface mirrors PyBoyEmulator so
    agent code stays backend-agnostic, but the button handling in particular
    is unverified.
    """

    _BUTTON_MAP = {
        "a": "press_a", "b": "press_b",
        "start": "press_start", "select": "press_select",
        "up": "press_up", "down": "press_down",
        "left": "press_left", "right": "press_right",
    }

    def __init__(self) -> None:
        super().__init__()
        self._gba: Optional[object] = None
        self._lock = threading.RLock()

    def _g(self):
        g = self._gba
        if g is None:
            raise RuntimeError("Emulator not loaded — call load(rom_path) first")
        return g

    @property
    def is_loaded(self) -> bool:
        return self._gba is not None

    # -- lifecycle ----------------------------------------------------------

    def load(self, rom_path: str) -> None:
        """Load a GBA ROM via PyGBA / mgba."""
        try:
            from pygba import PyGBA  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyGBA (mgba-py) is required for .gba ROMs.  "
                "Install it with:  pip install pygba"
            ) from exc

        rom_path = str(Path(rom_path).expanduser().resolve())
        if not os.path.isfile(rom_path):
            raise FileNotFoundError(f"ROM not found: {rom_path}")

        with self._lock:
            if self._gba is not None:
                raise RuntimeError("Emulator already loaded; call close() first")
            self._gba = PyGBA.load(rom_path)  # type: ignore[attr-defined]
            self.rom_path = rom_path
            self.frame_count = 0
        logger.info("PyGBA loaded: %s", rom_path)

    def close(self) -> None:
        with self._lock:
            self._gba = None

    # -- input --------------------------------------------------------------

    def press(self, button: str, frames: int = 1) -> None:
        button = button.lower()
        method = self._BUTTON_MAP.get(button)
        if method is None:
            raise ValueError(f"Unknown button '{button}'. Valid: {self.BUTTONS}")
        with self._lock:
            getattr(self._g(), method)()  # type: ignore[union-attr]
            self.tick(max(1, frames))

    def release_all(self) -> None:
        # PyGBA's press_* helpers auto-release after wait(); nothing to do.
        pass

    # -- timing -------------------------------------------------------------

    def tick(self, frames: int = 1, render_last: bool = True) -> None:
        """Advance *frames* frames. render_last is accepted for interface
        parity; mgba renders unconditionally."""
        if frames < 1:
            return
        with self._lock:
            self._g().wait(frames)  # type: ignore[union-attr]
            self.frame_count += frames

    # -- video --------------------------------------------------------------

    def get_screen(self) -> "Image.Image":
        with self._lock:
            return self._g().screen.to_pil().convert("RGB")  # type: ignore[union-attr]

    # -- memory -------------------------------------------------------------

    def read_u8(self, addr: int) -> int:
        with self._lock:
            return self._g().read_u8(addr)  # type: ignore[union-attr]

    def read_u16(self, addr: int) -> int:
        with self._lock:
            return self._g().read_u16(addr)  # type: ignore[union-attr]

    def read_u32(self, addr: int) -> int:
        with self._lock:
            return self._g().read_u32(addr)  # type: ignore[union-attr]

    def read_range(self, addr: int, size: int) -> bytes:
        with self._lock:
            g = self._g()
            return bytes(g.read_u8(addr + i) for i in range(size))  # type: ignore[union-attr]

    # -- save / load --------------------------------------------------------

    def save_state(self, path: str) -> None:
        with self._lock:
            self._g().save_state(path)  # type: ignore[union-attr]

    def load_state(self, path: str) -> None:
        with self._lock:
            self._g().load_state(path)  # type: ignore[union-attr]

    # -- info ---------------------------------------------------------------

    def get_info(self) -> Dict:
        info = super().get_info()
        info["platform"] = "GBA"
        return info


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_EXT_MAP = {
    ".gb": PyBoyEmulator,
    ".gbc": PyBoyEmulator,
    ".gba": PyGBAEmulator,
}


def create_emulator(rom_path: str, load: bool = True) -> Emulator:
    """Create the appropriate emulator for *rom_path* from its extension.

    Parameters
    ----------
    rom_path : str
        Path to a Game Boy (.gb/.gbc) or Game Boy Advance (.gba) ROM.
    load : bool
        When False, return an unloaded instance and leave booting to the
        caller via ``emu.load(rom_path)``. This is what lets session
        selection be separated from ROM boot — constructing an emulator must
        not be the same act as starting the game.

    Returns
    -------
    Emulator

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    """
    ext = Path(rom_path).suffix.lower()
    cls = _EXT_MAP.get(ext)
    if cls is None:
        raise ValueError(
            f"Unsupported ROM extension '{ext}'. "
            f"Supported: {', '.join(_EXT_MAP)}"
        )
    emu = cls()
    if load:
        try:
            emu.load(rom_path)
        except BaseException:
            emu.close()   # never leak a half-initialised backend
            raise
    return emu