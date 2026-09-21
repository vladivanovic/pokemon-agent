"""
Pokemon Agent — FastAPI Game Server

HTTP + WebSocket API for controlling a Game Boy / GBA emulator running a
Pokemon ROM, reading game state, and broadcasting events.

Two invariants hold this file together:

1. SINGLE-OWNER EMULATOR. Exactly one thread (``_emulator_worker``) may touch
   the emulator. Request handlers submit callables through ``emu_call`` and
   await the result. PyBoy's core is native and not thread-safe: calling into
   it from an executor while the worker ticks corrupts state and crashes in
   the audio path, which looks like a sound bug and is not one.

2. BOOT ONLY ON START. Creating or loading a game session only *arms* it
   (``_pending_state``). The ROM boots when /control is set to "running".
   Selecting a save must not start the game.
"""

import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Set

from fastapi import (
    FastAPI,
    HTTPException,
    Path as PathParam,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pokemon-agent.server")

__version__ = "0.1.0"

SID_PATTERN = r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GameConfig(BaseModel):
    """Server configuration — set before startup via configure()."""
    rom_path: str
    game_type: str = "auto"            # "red" | "firered" | "auto"
    port: int = 8765
    data_dir: str = "~/.pokemon-agent"
    load_state: Optional[str] = None   # save-state name to ARM at startup
    no_dashboard: bool = False


class ActionRequest(BaseModel):
    actions: list[str]


class EventRequest(BaseModel):
    """Agent narration pushed to the dashboard."""
    type: str                          # reasoning | decision | key_moment | alert
    text: Optional[str] = None
    description: Optional[str] = None  # for key_moment
    category: Optional[str] = None     # milestone | badge | catch | alert


class SaveRequest(BaseModel):
    name: str


class Objective(BaseModel):
    tier: str                          # primary | secondary | tertiary
    text: str
    done: bool = False


class ObjectivesRequest(BaseModel):
    objectives: list[Objective]


class ControlRequest(BaseModel):
    state: str                         # running | paused | stopped


class NewGameRequest(BaseModel):
    name: Optional[str] = None


class HermesSessionRequest(BaseModel):
    hermes_session_id: str


# ---------------------------------------------------------------------------
# Emulator thread — single-owner pattern
# ---------------------------------------------------------------------------

TARGET_FPS  = 60.0   # self-paced; the emulator itself runs unbounded
MAX_CATCHUP = 16     # frames per iteration ceiling after a stall
PUBLISH_HZ  = 10.0   # frames pushed to the dashboard per second while running


@dataclass
class _Cmd:
    fn: Callable                       # callable(emulator, *args) -> Any
    args: tuple
    future: "concurrent.futures.Future"


_emu_cmds: "queue.Queue[_Cmd]" = queue.Queue()
_emu_thread: Optional[threading.Thread] = None
_emu_stop = threading.Event()
_emu_state: str = "idle"               # idle | booting | ready | error
_emu_error: Optional[str] = None

_pending_state: Optional[str] = None   # save-state armed, awaiting START
_frame_png: Optional[bytes] = None     # latest encoded frame (served from cache)
_frame_no: int = 0
_frame_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_config: Optional[GameConfig] = None
_emulator = None                       # Emulator instance (worker-owned)
_reader = None                         # GameMemoryReader (worker-owned)
_start_time: float = 0.0
_loop: Optional[asyncio.AbstractEventLoop] = None

_objectives: list = [
    {"tier": "primary", "text": "Deliver Oak's Parcel · get Pokédex", "done": False},
    {"tier": "secondary", "text": "Reach Pewter City · Boulder Badge", "done": False},
    {"tier": "tertiary", "text": "Catch a Grass/Electric type", "done": False},
]

# Autopilot run state. Also gates emulation: the worker only ticks while
# "running", so PAUSE and STOP genuinely freeze the game.
_control_state: str = "stopped"

_session_mgr = None                    # GameSessionManager
_active_session = None                 # GameSession currently being played

_ws_clients: Set[WebSocket] = set()

# Replay buffer so a client connecting mid-run sees a populated Field Log.
# Display-worthy events only — not the high-frequency screenshot frames.
_event_history: deque = deque(maxlen=200)
_REPLAYABLE = {"reasoning", "decision", "thought", "key_moment", "moment",
               "alert", "battle", "action"}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pokemon Agent Server",
    version=__version__,
    description="HTTP + WebSocket API for Pokemon emulator control",
)

# allow_origins=["*"] with allow_credentials=True is spec-invalid and browsers
# reject it, so match local origins explicitly instead.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Emulator plumbing
# ---------------------------------------------------------------------------

def _fail(msg: str) -> None:
    """Enter the error state, log, and tell clients. The worker stays alive so
    a retry can succeed without restarting the server."""
    global _emu_state, _emu_error
    _emu_state, _emu_error = "error", msg
    logger.error("emulator: %s", msg)
    _emit_status()


def _emu_status() -> dict:
    return {"emulator_state": _emu_state, "error": _emu_error,
            "frame": _frame_no, "armed": _pending_state is not None,
            "control": _control_state}


def _publish(event: dict) -> None:
    """Broadcast from the worker thread onto the asyncio loop."""
    if _loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(broadcast(event), _loop)
    except RuntimeError:
        pass


def _emit_status() -> None:
    _publish({"type": "emulator", **_emu_status()})


def _ensure_emulator() -> None:
    """Raise 503 unless the emulator is booted and healthy."""
    if _emulator is None or _emu_state != "ready":
        detail = f"Emulator not ready ({_emu_state})"
        if _emu_error:
            detail += f": {_emu_error}"
        raise HTTPException(status_code=503, detail=detail)


async def emu_call(fn, *args, timeout: float = 30.0):
    """Run ``fn(emulator, *args)`` on the owner thread and await its result.

    The ONLY legal way for a request handler to touch the emulator. Anything
    using run_in_executor reintroduces the native-code race.
    """
    if _emu_thread is None or not _emu_thread.is_alive():
        raise HTTPException(status_code=503,
                            detail=f"Emulator thread not running ({_emu_state})")
    fut: "concurrent.futures.Future" = concurrent.futures.Future()
    _emu_cmds.put(_Cmd(fn, args, fut))
    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Emulator command timed out")


def _dispatch(cmd: _Cmd) -> None:
    """Execute one queued command on the owner thread."""
    if not cmd.future.set_running_or_notify_cancel():
        return
    try:
        cmd.future.set_result(cmd.fn(_emulator, *cmd.args))
    except Exception as exc:
        cmd.future.set_exception(exc)


def _publish_frame(force_render: bool = False) -> None:
    """Encode the current framebuffer and push it to clients.

    Must NOT advance emulation: this also runs while paused and stopped. The
    worker ticks with render_last=False for speed, so a render is requested
    only when emulation is live; otherwise the existing buffer is re-encoded.
    """
    global _frame_png, _frame_no
    try:
        if force_render:
            _emulator.tick(1, render_last=True)
        buf = io.BytesIO()
        _emulator.get_screen().save(buf, format="PNG")
        png = buf.getvalue()
    except Exception as exc:
        _fail(f"frame publish failed: {type(exc).__name__}: {exc}")
        return
    with _frame_lock:
        _frame_png, _frame_no = png, _emulator.frame_count
    _publish({"type": "screenshot", "frame": _frame_no,
              "data": {"image": base64.b64encode(png).decode("ascii"),
                       "format": "png"}})


def _emulator_worker() -> None:
    """Sole owner of the emulator. Nothing else may call into it."""
    global _emulator, _reader
    logger.info("emulator worker started")
    period = 1.0 / TARGET_FPS
    publish_period = 1.0 / PUBLISH_HZ
    next_due = time.perf_counter()
    next_publish = time.perf_counter()

    while not _emu_stop.is_set():
        # Commands first — boot, actions, save/load, screenshots. The blocking
        # get wakes immediately on arrival rather than busy-spinning.
        try:
            _dispatch(_emu_cmds.get(timeout=0.05))
            while True:
                try:
                    _dispatch(_emu_cmds.get_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass

        now = time.perf_counter()
        running = (_emulator is not None and _emu_state == "ready"
                   and _control_state == "running")

        if not running:
            next_due = now
        else:
            if now < next_due:
                continue
            due = min(int((now - next_due) / period) + 1, MAX_CATCHUP)
            try:
                _emulator.tick(due, render_last=False)
            except Exception as exc:
                _fail(f"tick failed: {type(exc).__name__}: {exc}")
                continue
            next_due += due * period
            if next_due < now - 0.25:   # fell badly behind; resync
                next_due = now

        # Publish whether or not the agent is running, so the dashboard stays
        # live. Slower cadence when frozen since the image cannot change.
        if _emulator is not None and _emu_state == "ready" and now >= next_publish:
            _publish_frame(force_render=running)
            next_publish = now + (publish_period if running else 1.0)

    if _emulator is not None:
        try:
            _emulator.close()
        finally:
            _emulator = None
            _reader = None
    logger.info("emulator worker stopped")


def _boot(_unused, rom_path: str, state_path: Optional[str]) -> dict:
    """Construct and load the emulator. Runs on the owner thread."""
    global _emulator, _reader, _emu_state, _emu_error
    from pokemon_agent.emulator import create_emulator

    if _emulator is not None:           # clear a wedged instance
        try:
            _emulator.close()
        except Exception:
            pass
        _emulator = None
        _reader = None

    _emu_state, _emu_error = "booting", None
    _emit_status()
    try:
        emu = create_emulator(rom_path, load=False)
        emu.load(rom_path)

        if _config.game_type == "red":
            from pokemon_agent.memory.red import PokemonRedReader
            _reader = PokemonRedReader(emu)
        else:
            from pokemon_agent.memory.firered import FireRedMemoryReader
            _reader = FireRedMemoryReader(emu)

        if state_path:
            emu.load_state(state_path)
        emu.tick(60, render_last=True)

        _emulator = emu
        _emu_state = "ready"
        _emit_status()
        _publish_frame()                # first frame lands immediately
        logger.info("emulator ready (frame %d)%s", emu.frame_count,
                    f", restored {Path(state_path).name}" if state_path else "")
        return {"state": _emu_state, "frame": emu.frame_count}
    except Exception as exc:
        _fail(f"boot failed: {type(exc).__name__}: {exc}")
        raise


def _start_emulator_thread() -> None:
    global _emu_thread
    if _emu_thread is not None and _emu_thread.is_alive():
        return
    _emu_stop.clear()
    _emu_thread = threading.Thread(target=_emulator_worker, name="emu",
                                   daemon=True)
    _emu_thread.start()


# ---------------------------------------------------------------------------
# Worker-side callables — every one takes the emulator as its first argument
# ---------------------------------------------------------------------------

def _screenshot_bytes(emu) -> bytes:
    """Fresh frame as PNG bytes. Renders first: the worker ticks unrendered."""
    emu.tick(1, render_last=True)
    buf = io.BytesIO()
    emu.get_screen().save(buf, format="PNG")
    return buf.getvalue()


def _grid_png(emu, scale: int) -> bytes:
    """Current frame with the A1..J9 grid overlay and walkability tint."""
    from pokemon_agent.collision import build_collision_grid
    from pokemon_agent.overlay import render_grid_overlay_bytes

    emu.tick(1, render_last=True)
    col = None
    try:
        player = _reader.read_player() or {}
        col = build_collision_grid(emu, facing=player.get("facing"),
                                   player_pos=player.get("position"))
    except Exception:
        logger.debug("collision unavailable for overlay", exc_info=True)
    return render_grid_overlay_bytes(emu.get_screen(), scale=scale,
                                     collision=col)


def _state_dict(emu) -> dict:
    """Full game state. The reader is worker-owned, hence the indirection."""
    return _get_state_dict()


def _ascii_map(emu) -> str:
    from pokemon_agent.collision import build_collision_grid, render_ascii_map
    from pokemon_agent.memory.red import MAP_NAMES

    player = _reader.read_player() or {}
    col = build_collision_grid(emu, facing=player.get("facing"),
                               player_pos=player.get("position"))
    for w in col.get("warps", []):
        w["dest_map_name"] = MAP_NAMES.get(w["dest_map"], f"Map {w['dest_map']}")
    return render_ascii_map(col, legend=True)


def _do_action(emu, action_str: str) -> None:
    """One action, executed atomically on the owner thread.

    Atomicity matters: Gen 1 needs the button held for >= 4 frames for the
    vblank joypad poll to register, then a settling wait. If another tick
    interleaves between the hold and the wait, the input is lost.
    """
    a = action_str.strip().lower()
    parts = a.split("_")

    if a == "a_until_dialog_end":
        for _ in range(10):
            emu.press("a", 8)
            emu.tick(30, render_last=False)
            # text_active, not active: `active` aliases input_locked, which is
            # also set during scripted cutscene movement, so the loop would
            # never exit in those.
            if not (_get_state_dict().get("dialog") or {}).get("text_active"):
                break
        return

    if parts[0] in ("press", "walk") and len(parts) >= 2:
        emu.press("_".join(parts[1:]), 8)
        emu.tick(12, render_last=False)
        return

    if parts[0] == "hold" and len(parts) >= 3:
        emu.press("_".join(parts[1:-1]), int(parts[-1]))
        return

    if parts[0] == "wait" and len(parts) == 2:
        emu.tick(int(parts[1]), render_last=False)
        return

    raise ValueError(f"Unknown action format: {action_str}")


# ---------------------------------------------------------------------------
# State assembly
# ---------------------------------------------------------------------------

def _get_state_dict() -> dict:
    """Build the full game state, including the collision grid for Red."""
    if _emulator is None or _reader is None or _emu_state != "ready":
        return {"status": "not_ready",
                "emulator_state": _emu_state,
                "error": _emu_error,
                "context": {"phase": _emu_state, "in_game": False}}

    from pokemon_agent.state.builder import build_game_state
    state = build_game_state(_reader, frame_count=_emulator.frame_count)

    # Ground-truth walkability from RAM. Only valid in the overworld with the
    # camera settled: wTileMap holds menu and text tiles when a box is open,
    # so the grid would otherwise be a picture of a menu rendered as a
    # mostly-blocked map — which reads as "you are trapped".
    try:
        ctx = state.get("context") or {}
        dlg = state.get("dialog") or {}
        if (_config and _config.game_type == "red" and ctx.get("in_game")
                and not (state.get("battle") or {}).get("in_battle")
                and not dlg.get("input_locked")):
            from pokemon_agent.collision import (build_collision_grid,
                                                 render_ascii_map)
            from pokemon_agent.memory.red import MAP_NAMES
            player = state.get("player") or {}
            col = build_collision_grid(_reader.emu,
                                       facing=player.get("facing"),
                                       player_pos=player.get("position"))
            for w in col.get("warps", []):
                w["dest_map_name"] = MAP_NAMES.get(w["dest_map"],
                                                   f"Map {w['dest_map']}")
            col["ascii"] = render_ascii_map(col, legend=True)
            state["collision"] = col
        else:
            state["collision"] = {
                "valid": False,
                "reason": "menu_dialog_battle_or_not_in_game",
            }
    except Exception as exc:  # noqa: BLE001
        state["collision"] = {"valid": False,
                              "reason": f"{type(exc).__name__}: {exc}"}
    return state


# ---------------------------------------------------------------------------
# Save-file resolution — one authoritative directory
# ---------------------------------------------------------------------------

def _saves_dir() -> Path:
    """Session-scoped when a game is active, else the legacy flat dir.

    Everything must go through this. The old split had /save writing to the
    session folder while /load and /saves read the flat one, so nothing you
    saved could be loaded back.
    """
    if _active_session is not None and _session_mgr is not None:
        return _session_mgr.saves_dir(_active_session.id)
    base = Path(_config.data_dir if _config else "~/.pokemon-agent")
    d = base.expanduser().resolve() / "saves"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_save(name: str) -> Path:
    """Locate a save by name: active session first, then the legacy dir."""
    p = _saves_dir() / f"{name}.state"
    if p.exists():
        return p
    base = Path(_config.data_dir if _config else "~/.pokemon-agent")
    legacy = base.expanduser().resolve() / "saves" / f"{name}.state"
    if legacy.exists():
        return legacy
    raise HTTPException(status_code=404, detail=f"Save not found: {name}")


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------

async def broadcast(event: dict):
    """Send a JSON event to every connected WebSocket client."""
    etype = event.get("type")
    if etype in _REPLAYABLE:
        rec = dict(event)
        rec.setdefault("ts", time.time())
        if etype == "action":
            rec.pop("state_after", None)   # keep the buffer small
        _event_history.append(rec)

    payload = json.dumps(event)
    dead: list[WebSocket] = []
    for ws in list(_ws_clients):           # copy: sending can mutate the set
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def configure(config: GameConfig):
    """Set server configuration. Call before app startup."""
    global _config
    _config = config
    global _start_time, _loop, _session_mgr, _pending_state
    global _active_session, _objectives


@app.on_event("startup")
async def _startup():
    global _start_time, _loop, _session_mgr, _pending_state
    _loop = asyncio.get_running_loop()
    _start_time = time.time()

    if _config is None:
        logger.warning("No GameConfig set — emulator will NOT start.")
        logger.warning("Call server.configure(GameConfig(...)) before startup.")
        return

    # The worker starts idle and cheap; it boots nothing until START.
    _start_emulator_thread()

    data_dir = Path(_config.data_dir).expanduser().resolve()
    (data_dir / "saves").mkdir(parents=True, exist_ok=True)

    from pokemon_agent.sessions import GameSessionManager
    _session_mgr = GameSessionManager(str(data_dir))

    # Re-adopt the most recently played session. Without this a server restart
    # leaves the dashboard pointing at a game the server has forgotten, and
    # START fails with 409 for no visible reason.
    try:
        recent = await asyncio.to_thread(_session_mgr.list)
    except Exception:
        logger.warning("could not list sessions", exc_info=True)
        recent = []
    if recent:
        gs = await asyncio.to_thread(_session_mgr.load, recent[0]["id"])
        if gs is not None:
            _active_session = gs
            _objectives = gs.objectives or _objectives
            latest = await asyncio.to_thread(_session_mgr.latest_save_path, gs.id)
            _pending_state = str(latest) if latest else None
            logger.info("restored session %s (%s)%s", gs.id, gs.name,
                        f", armed {latest.name}" if latest else " — no save yet")

    if not _config.no_dashboard:
        try:
            import pokemon_agent.dashboard as dashboard_mod  # noqa: F401
            from fastapi.staticfiles import StaticFiles
            dash_dir = Path(dashboard_mod.__file__).parent / "static"
            if dash_dir.is_dir():
                app.mount("/dashboard",
                          StaticFiles(directory=str(dash_dir), html=True),
                          name="dashboard")
                logger.info("Dashboard mounted at /dashboard")
            else:
                logger.warning("Dashboard module found but no static/ directory")
        except ImportError:
            logger.warning("Dashboard not installed — /dashboard unavailable")

    # Arm only. Booting here would defeat the whole point of START.
    if _config.load_state:
        state_path = data_dir / "saves" / f"{_config.load_state}.state"
        if state_path.exists():
            _pending_state = str(state_path)
            logger.info("Armed save state (press START to boot): %s",
                        _config.load_state)
        else:
            logger.warning("Save state not found: %s", state_path)

    logger.info("Ready — listening on port %s", _config.port)
    logger.info("Emulator is idle. POST /control {\"state\":\"running\"} to boot.")


@app.on_event("shutdown")
def _shutdown():
    """Stop the worker; it closes the emulator itself."""
    global _emu_thread
    _emu_stop.set()
    if _emu_thread is not None and _emu_thread.is_alive():
        _emu_thread.join(timeout=5.0)
    _emu_thread = None
    logger.info("Server shutdown complete.")


# ---------------------------------------------------------------------------
# Info endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    return {
        "name": "pokemon-agent",
        "version": __version__,
        "game": _config.game_type if _config else None,
        "rom": _config.rom_path if _config else None,
        "uptime_seconds": round(time.time() - _start_time, 1) if _start_time else 0,
        **_emu_status(),
    }


@app.get("/health")
async def health():
    return {"status": "ok", **_emu_status()}


@app.get("/state")
async def get_state():
    """Full game state JSON. Check `status` and `context.in_game` first."""
    return JSONResponse(content=await emu_call(_state_dict))


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

@app.get("/screenshot")
async def screenshot():
    """Latest published frame, served from cache — no emulator round-trip."""
    with _frame_lock:
        png, frame = _frame_png, _frame_no
    if png is None:
        raise HTTPException(status_code=503,
                            detail=f"No frame yet ({_emu_state})")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store",
                             "X-Frame-Number": str(frame)})


@app.get("/screenshot/base64")
async def screenshot_base64():
    with _frame_lock:
        png, frame = _frame_png, _frame_no
    if png is None:
        raise HTTPException(status_code=503,
                            detail=f"No frame yet ({_emu_state})")
    return {"image": base64.b64encode(png).decode("ascii"),
            "format": "png", "frame": frame}


@app.get("/screenshot/grid")
async def screenshot_grid(scale: int = 4):
    """Current frame with a labelled A1..J9 movement grid drawn on top.

    Unlike /screenshot this is a live render — it needs the collision grid, so
    it goes to the emulator thread.
    """
    if not 1 <= scale <= 8:
        raise HTTPException(status_code=400, detail="scale must be 1..8")
    _ensure_emulator()
    try:
        return Response(content=await emu_call(_grid_png, scale),
                        media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("grid screenshot failed")
        raise HTTPException(status_code=500,
                            detail=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

@app.get("/map/ascii")
async def map_ascii():
    """On-screen walkability as an ASCII map (text/plain).

    @ player · . walkable · # blocked · N person · D door/exit. Read from RAM,
    so it is ground truth rather than a guess from pixels.
    """
    _ensure_emulator()
    try:
        return Response(content=await emu_call(_ascii_map),
                        media_type="text/plain")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ASCII map error: {exc}")


@app.get("/minimap")
async def minimap():
    """Map name + player position as text."""
    _ensure_emulator()
    try:
        state = await emu_call(_state_dict)
        map_name = (state.get("map") or {}).get("map_name", "Unknown")
        pos = (state.get("player") or {}).get("position") or {}
        lines = [
            f"=== {map_name} ===",
            f"Player position: ({pos.get('x', '?')}, {pos.get('y', '?')})",
            "", "  N", "W + E", "  S",
        ]
        return Response(content="\n".join(lines), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Minimap error: {exc}")


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@app.post("/action")
async def execute_actions(req: ActionRequest):
    """Execute a sequence of game actions, one atomic step at a time."""
    _ensure_emulator()
    executed = 0
    try:
        for action_str in req.actions:
            await emu_call(_do_action, action_str)
            executed += 1
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Action error: {exc}")

    state_after = await emu_call(_state_dict)

    if _active_session is not None and _session_mgr is not None:
        s = _active_session.stats
        s["actions"] = s.get("actions", 0) + executed
        s["turns"] = s.get("turns", 0) + 1
        await asyncio.to_thread(_session_mgr.save, _active_session)

    # No screenshot broadcast here: the worker publishes at PUBLISH_HZ and its
    # frames carry a frame number the client uses for staleness detection.
    await broadcast({"type": "action", "actions": req.actions,
                     "actions_executed": executed, "state_after": state_after})

    return {"success": True, "actions_executed": executed,
            "state_after": state_after}


# ---------------------------------------------------------------------------
# Narration / objectives / control
# ---------------------------------------------------------------------------

@app.post("/event")
async def push_event(req: EventRequest):
    """Push agent narration to the dashboard (broadcast over WS).

    reasoning / decision / alert: send `text`.
    key_moment: send `description` and optionally `category`.
    Display-only — not stored in conversation history.
    """
    event: dict = {"type": req.type}
    if req.text is not None:
        event["text"] = req.text
    if req.description is not None:
        event["description"] = req.description
    if req.category is not None:
        event["category"] = req.category

    if (req.type in ("key_moment", "moment") and req.description
            and _active_session is not None and _session_mgr is not None):
        await asyncio.to_thread(_session_mgr.add_milestone, _active_session,
                                req.description, req.category or "milestone")

    await broadcast(event)
    return {"success": True, "broadcast_to": len(_ws_clients)}


@app.get("/objectives")
async def get_objectives():
    return {"objectives": _objectives}


@app.post("/objectives")
async def set_objectives(req: ObjectivesRequest):
    """Replace the objective list and broadcast it."""
    global _objectives
    _objectives = [o.model_dump() for o in req.objectives]
    if _active_session is not None and _session_mgr is not None:
        _active_session.objectives = _objectives
        await asyncio.to_thread(_session_mgr.save, _active_session)
    await broadcast({"type": "objectives", "objectives": _objectives})
    return {"success": True, "objectives": _objectives}


@app.get("/control")
async def get_control():
    return {"state": _control_state, **_emu_status()}


@app.post("/control")
async def set_control(req: ControlRequest):
    """Set the run state — and boot the emulator on the first "running".

    This is the single boot trigger. It also gates emulation: the worker only
    ticks while running, so "paused" and "stopped" freeze the game rather than
    merely idling the autopilot.
    """
    global _control_state
    valid = {"running", "paused", "stopped"}
    if req.state not in valid:
        raise HTTPException(status_code=400,
                            detail=f"state must be one of {sorted(valid)}")

    if req.state == "running" and _emu_state in ("idle", "error"):
        if _active_session is None:
            raise HTTPException(
                status_code=409,
                detail="No active game — create or load one first")
        _start_emulator_thread()
        await emu_call(_boot, _config.rom_path, _pending_state, timeout=120)

    _control_state = req.state
    await broadcast({"type": "control", "state": _control_state})
    _emit_status()
    return {"success": True, "state": _control_state, **_emu_status()}


# ---------------------------------------------------------------------------
# Game sessions
# ---------------------------------------------------------------------------

def _game_summary() -> dict:
    if _active_session is None:
        return {"active": None}
    gs = _active_session
    return {"active": {"id": gs.id, "name": gs.name, "game": gs.game,
                       "hermes_session_id": gs.hermes_session_id,
                       "objectives": gs.objectives, "stats": gs.stats}}


async def _activate(gs) -> None:
    """Make `gs` active: adopt objectives, persist, broadcast.

    Deliberately does NOT boot anything — see the module docstring.
    """
    global _active_session, _objectives
    _active_session = gs
    _objectives = gs.objectives or _objectives
    await asyncio.to_thread(_session_mgr.save, gs)
    await broadcast({"type": "objectives", "objectives": _objectives})
    await broadcast({"type": "game", **_game_summary()})


@app.get("/games")
async def list_games():
    if _session_mgr is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    games = await asyncio.to_thread(_session_mgr.list)
    return {"games": games,
            "active": _active_session.id if _active_session else None}


@app.get("/games/current")
async def current_game():
    return _game_summary()


@app.post("/games/new")
async def new_game(req: NewGameRequest):
    """Create a new session and arm a clean boot. Press START to play."""
    global _pending_state
    if _session_mgr is None or _config is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    gs = await asyncio.to_thread(_session_mgr.create, req.name, _config.game_type)
    _pending_state = None               # fresh game: boot to the title screen
    await _activate(gs)
    _emit_status()
    return {"success": True, "game": gs.to_dict(),
            "note": "press START to boot"}


@app.post("/games/{sid}/load")
async def load_game(sid: str = PathParam(..., pattern=SID_PATTERN)):
    """Arm an existing session: its latest save-state and its Hermes brain.

    Arming only. The ROM boots on START, so you can browse sessions without
    disturbing a running game.
    """
    global _pending_state
    if _session_mgr is None or _config is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    gs = await asyncio.to_thread(_session_mgr.load, sid)
    if gs is None:
        raise HTTPException(status_code=404,
                            detail=f"Game session not found: {sid}")
    latest = await asyncio.to_thread(_session_mgr.latest_save_path, sid)
    _pending_state = str(latest) if latest else None
    await _activate(gs)
    _emit_status()
    return {"success": True, "game": gs.to_dict(),
            "restored_save": latest.stem if latest else None,
            "armed": True, "note": "press START to boot"}


@app.post("/games/{sid}/hermes")
async def bind_hermes(req: HermesSessionRequest,
                      sid: str = PathParam(..., pattern=SID_PATTERN)):
    """Bind the Hermes session id so the run's brain persists in the manifest."""
    if _session_mgr is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    gs = (_active_session
          if (_active_session and _active_session.id == sid)
          else await asyncio.to_thread(_session_mgr.load, sid))
    if gs is None:
        raise HTTPException(status_code=404,
                            detail=f"Game session not found: {sid}")
    gs.hermes_session_id = req.hermes_session_id
    await asyncio.to_thread(_session_mgr.save, gs)
    await broadcast({"type": "game", **_game_summary()})
    return {"success": True, "hermes_session_id": gs.hermes_session_id}


@app.delete("/games/{sid}")
async def delete_game(sid: str = PathParam(..., pattern=SID_PATTERN)):
    """Delete a session and its saves. The active session is protected."""
    if _session_mgr is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    if _active_session and _active_session.id == sid:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the active game; load another first.")
    ok = await asyncio.to_thread(_session_mgr.delete, sid)
    if not ok:
        raise HTTPException(status_code=404,
                            detail=f"Game session not found: {sid}")
    return {"success": True, "deleted": sid}


# ---------------------------------------------------------------------------
# Save states
# ---------------------------------------------------------------------------

@app.post("/save")
async def save_state(req: SaveRequest):
    """Save emulator state into the active session's folder."""
    _ensure_emulator()
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    path = _saves_dir() / f"{req.name}.state"
    await emu_call(lambda e, p=str(path): e.save_state(p))
    if _active_session is not None and _session_mgr is not None:
        _active_session.stats["saves"] = _active_session.stats.get("saves", 0) + 1
        await asyncio.to_thread(_session_mgr.save, _active_session)
        await asyncio.to_thread(_session_mgr.prune_saves, _active_session.id, 20)
    return {"success": True, "path": str(path), "name": req.name,
            "session": _active_session.id if _active_session else None}


@app.post("/load")
async def load_state(req: SaveRequest):
    """Load a save state into the running emulator."""
    _ensure_emulator()
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    path = _resolve_save(req.name)
    await emu_call(lambda e, p=str(path): e.load_state(p))
    state_after = await emu_call(_state_dict)
    await broadcast({"type": "state_update", "reason": "load",
                     "state": state_after})
    return {"success": True, "name": req.name, "state_after": state_after}


@app.get("/saves")
async def list_saves():
    """List save states in the authoritative directory for the active session."""
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    d = _saves_dir()
    files = sorted(d.glob("*.state"), key=lambda f: f.stat().st_mtime,
                   reverse=True)
    return {"dir": str(d),
            "session": _active_session.id if _active_session else None,
            "saves": [{"name": f.stem, "file": f.name,
                       "size_bytes": f.stat().st_size,
                       "modified": f.stat().st_mtime} for f in files]}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Live event stream. Frames are pushed by the emulator worker."""
    await ws.accept()
    _ws_clients.add(ws)
    try:
        await ws.send_json({"type": "connected", "version": __version__})
        await ws.send_json({"type": "emulator", **_emu_status()})
        if _event_history:
            await ws.send_json({"type": "replay",
                                "events": list(_event_history)})
        await ws.send_json({"type": "objectives", "objectives": _objectives})
        await ws.send_json({"type": "control", "state": _control_state})
        await ws.send_json({"type": "game", **_game_summary()})

        # Send the cached frame immediately so a late client is not blank
        # until the next publish tick.
        with _frame_lock:
            png, frame = _frame_png, _frame_no
        if png is not None:
            await ws.send_json({
                "type": "screenshot", "frame": frame,
                "data": {"image": base64.b64encode(png).decode("ascii"),
                         "format": "png"}})

        while True:
            data = await ws.receive_text()
            if data.strip().lower() == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("websocket closed with error", exc_info=True)
    finally:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Dashboard fallback — registered only if static files are missing
# ---------------------------------------------------------------------------

def _register_dashboard_fallback():
    try:
        import pokemon_agent.dashboard as _dm
        static_dir = Path(_dm.__file__).parent / "static"
        if static_dir.is_dir() and (static_dir / "index.html").exists():
            return
    except ImportError:
        pass

    @app.get("/dashboard")
    @app.get("/dashboard/{path:path}")
    async def dashboard_fallback(path: str = ""):
        raise HTTPException(
            status_code=404,
            detail="Dashboard not installed. "
                   "Install with: pip install pokemon-agent[dashboard]")


_register_dashboard_fallback()