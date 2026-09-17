"""
Pokemon Agent — FastAPI Game Server

Provides HTTP + WebSocket API for controlling a Game Boy / GBA emulator
running a Pokemon ROM, reading game state, and broadcasting events.
"""

import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

SID_PATTERN = r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pokemon-agent.server")

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GameConfig(BaseModel):
    """Server configuration — set before startup."""
    rom_path: str
    game_type: str = "auto"       # "red", "firered", or "auto"
    port: int = 8765
    data_dir: str = "~/.pokemon-agent"
    load_state: Optional[str] = None  # Save-state name to auto-load on startup
    no_dashboard: bool = False


class ActionRequest(BaseModel):
    """Body for POST /action."""
    actions: list[str]


class EventRequest(BaseModel):
    """Body for POST /event — the agent pushes narration to the dashboard."""
    type: str                       # "reasoning" | "decision" | "key_moment" | "alert"
    text: Optional[str] = None      # for reasoning / decision / alert
    description: Optional[str] = None  # for key_moment
    category: Optional[str] = None     # key_moment category: milestone/badge/catch/alert


class SaveRequest(BaseModel):
    """Body for POST /save and POST /load."""
    name: str


class Objective(BaseModel):
    """A single objective shown on the dashboard."""
    tier: str            # "primary" | "secondary" | "tertiary"
    text: str
    done: bool = False


class ObjectivesRequest(BaseModel):
    """Body for POST /objectives — replace the full objective list."""
    objectives: list[Objective]


class ControlRequest(BaseModel):
    """Body for POST /control — set the autopilot run state."""
    state: str           # "running" | "paused" | "stopped"


class NewGameRequest(BaseModel):
    """Body for POST /games/new."""
    name: Optional[str] = None


class HermesSessionRequest(BaseModel):
    """Body for POST /games/{id}/hermes — bind the Hermes session id."""
    hermes_session_id: str


# ---------------------------------------------------------------------------
# Emulator thread — single-owner pattern
# ---------------------------------------------------------------------------

TARGET_FPS = 60.0        # self-paced; PyBoy runs unbounded
MAX_CATCHUP = 16         # frames per iteration ceiling after a stall
PUBLISH_HZ = 10.0                 # frames pushed to dashboard per second

@dataclass
class _Cmd:
    fn: Callable           # callable(emulator, *args) -> Any
    args: tuple
    future: "concurrent.futures.Future"

_emu_cmds: "queue.Queue[_Cmd]" = queue.Queue()
_emu_thread: Optional[threading.Thread] = None
_emu_stop = threading.Event()
_emu_state: str = "idle"          # idle|booting|ready|error
_emu_error: Optional[str] = None

# Lifecycle / frame publisher
_pending_state: Optional[str] = None   # save-state armed by /games/*/load
_frame_png: Optional[bytes] = None     # latest encoded frame
_frame_no: int = 0
_frame_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_config: Optional[GameConfig] = None
_emulator = None          # Emulator instance
_reader = None            # GameMemoryReader subclass instance
_start_time: float = 0.0
_loop: Optional[asyncio.AbstractEventLoop] = None

# Dynamic objectives shown on the dashboard (default = Kanto opening goals).
_objectives: list = [
    {"tier": "primary", "text": "Deliver Oak's Parcel · get Pokédex", "done": False},
    {"tier": "secondary", "text": "Reach Pewter City · Boulder Badge", "done": False},
    {"tier": "tertiary", "text": "Catch a Grass/Electric type", "done": False},
]

# Autopilot run state. "stopped" (default) | "running" | "paused".
# A standalone `pokemon-agent play` loop reads this and only acts when running.
_control_state: str = "stopped"

# Game-session layer (binds Hermes brain + emulator saves + objectives/stats).
_session_mgr = None       # GameSessionManager
_active_session = None     # GameSession currently being played

# WebSocket clients
_ws_clients: Set[WebSocket] = set()

# Replay buffer — recent narration/milestone events so a client that connects
# mid-run sees the Field Log already populated instead of an empty panel.
# Only display-worthy events are kept (reasoning/decision/key_moment/alert/
# action), not the high-frequency screenshot/state_update frames.
from collections import deque
_event_history: deque = deque(maxlen=200)
_REPLAYABLE = {"reasoning", "decision", "thought", "key_moment", "moment", "alert", "battle", "action"}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pokemon Agent Server",
    version=__version__,
    description="HTTP + WebSocket API for Pokemon emulator control",
)

# CORS — allow everything for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_game_type(rom_path: str) -> str:
    """Pick reader type based on file extension."""
    ext = Path(rom_path).suffix.lower()
    if ext in (".gb", ".gbc"):
        return "red"
    elif ext == ".gba":
        return "firered"
    return "unknown"


def _ensure_emulator():
    """Raise 503 if the emulator isn't ready."""
    if _emulator is None:
        raise HTTPException(status_code=503, detail="Emulator not initialised")


async def emu_call(fn, *args, timeout: float = 30.0):
    """Run fn(emulator, *args) on the owner thread and await its result.

    This is the ONLY legal way for a request handler to touch the emulator.
    Anything using run_in_executor reintroduces the native-code race.
    """
    if _emu_thread is None or not _emu_thread.is_alive():
        raise HTTPException(status_code=503, detail=f"Emulator not running ({_emu_state})")
    fut: "concurrent.futures.Future" = concurrent.futures.Future()
    _emu_cmds.put(_Cmd(fn, args, fut))
    try:
        return await asyncio.wait_for(asyncio.wrap_future(fut), timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Emulator command timed out")


async def broadcast(event: dict):
    """Send a JSON event to every connected WebSocket client.

    Display-worthy events (narration, milestones, actions) are also recorded
    in a replay buffer so a client connecting mid-run can backfill the log.
    """
    etype = event.get("type")
    if etype in _REPLAYABLE:
        rec = dict(event)
        rec.setdefault("ts", time.time())
        if etype == "action":
            # Don't store the full state snapshot in the buffer — just the moves.
            rec.pop("state_after", None)
        _event_history.append(rec)

    dead: list[WebSocket] = []
    payload = json.dumps(event)
    for ws in list(_ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _screenshot_broadcast_task():
    """Periodically broadcast current frame to all WebSocket clients.

    Runs every 1 second while there are connected clients. This keeps the
    dashboard's game screen live even when no actions are being executed
    (e.g., while the model is thinking).
    """
    while True:
        await asyncio.sleep(1.0)
        if not _ws_clients or _emulator is None:
            continue
        try:
            png_bytes = await emu_call(_screenshot_bytes)
            b64 = base64.b64encode(png_bytes).decode("ascii")
            await broadcast({
                "type": "screenshot",
                "data": {"image": b64, "format": "png"},
            })
        except Exception:
            # Don't spam logs if emulator isn't ready
            pass


# Global task reference
_screenshot_task: asyncio.Task | None = None


def _saves_dir() -> Path:
    """The single authoritative save directory.

    Session-scoped when a game is active, otherwise the legacy flat dir.
    Every save/load path must go through this — the old split meant
    /save wrote somewhere /load never looked.
    """
    if _active_session is not None and _session_mgr is not None:
        return _session_mgr.saves_dir(_active_session.id)
    if _config is None:
        # Config not set yet — use default
        d = Path("~/.pokemon-agent").expanduser().resolve() / "saves"
    else:
        d = Path(_config.data_dir).expanduser().resolve() / "saves"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_save(name: str) -> Path:
    """Locate a save by name: active session first, then legacy dir."""
    p = _saves_dir() / f"{name}.state"
    if p.exists():
        return p
    legacy = Path(_config.data_dir).expanduser().resolve() / "saves" / f"{name}.state"
    if legacy.exists():
        return legacy
    raise HTTPException(status_code=404, detail=f"Save not found: {name}")


def _get_state_dict() -> dict:
    """Build full game state from the memory reader."""
    if _emulator is None or _emu_state != "ready":
        return {"status": "not_ready", "emulator_state": _emu_state,
                "error": _emu_error, "context": {"phase": _emu_state,
                                                 "in_game": False}}
    # At this point _reader is guaranteed to be initialized
    from pokemon_agent.state.builder import build_game_state
    state = build_game_state(_reader, frame_count=_emulator.frame_count)
    # Attach the on-screen walkability grid for Red/Blue (overworld tilesets).
    # This is ground-truth collision read from RAM — far more reliable than
    # inferring walkability from pixels.
    # Only build collision when: in game, not in battle, no input-locked dialog.
    # wTileMap holds menu and text tiles when a box is open, so the grid
    # would otherwise be a picture of a menu rendered as a mostly-blocked map.
    try:
        ctx = state.get("context") or {}
        dlg = state.get("dialog") or {}
        if (_config and _config.game_type == "red" and ctx.get("in_game")
                and not (state.get("battle") or {}).get("in_battle")
                and not dlg.get("input_locked")):
            from pokemon_agent.collision import build_collision_grid, render_ascii_map
            from pokemon_agent.memory.red import MAP_NAMES
            player = state.get("player") or {}
            col = build_collision_grid(
                _reader.emu,
                facing=player.get("facing"),
                player_pos=player.get("position"))
            
            for w in col.get("warps", []):
                w["dest_map_name"] = MAP_NAMES.get(w["dest_map"], f"Map {w['dest_map']}")

            col["ascii"] = render_ascii_map(col, legend=True)
            state["collision"] = col
        else:
            state["collision"] = {"valid": False, "reason": "menu_dialog_or_not_in_game"}
    except Exception as exc:  # noqa: BLE001
        state["collision"] = {"valid": False,
                              "reason": f"{type(exc).__name__}: {exc}"}
    return state


def _screenshot_bytes(emu) -> bytes:
    """Grab the current frame as PNG bytes. Tick once with render_last to ensure fresh framebuffer."""
    emu.tick(1, render_last=True)
    screen = emu.get_screen()
    buf = io.BytesIO()
    # If it's a numpy array, convert to PIL first
    try:
        from PIL import Image
        if not isinstance(screen, Image.Image):
            import numpy as np
            screen = Image.fromarray(screen)
        screen.save(buf, format="PNG")
    except ImportError:
        # Fallback: assume screen already has save()
        screen.save(buf, format="PNG")
    return buf.getvalue()


def _state_dict(emu) -> dict:
    """Return full game state. Reader is worker-owned."""
    return _get_state_dict()


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

def _do_action(emu, action_str: str) -> None:
    """One action, executed atomically on the owner thread."""
    a = action_str.strip().lower()
    parts = a.split("_")
    if a == "a_until_dialog_end":
        for _ in range(10):
            emu.press("a", 8)
            emu.tick(30, render_last=False)
            if not (_get_state_dict().get("dialog") or {}).get("active"):
                break
    elif parts[0] in ("press", "walk") and len(parts) >= 2:
        emu.press("_".join(parts[1:]), 8)
        emu.tick(12, render_last=False)
    elif parts[0] == "hold" and len(parts) >= 3:
        emu.press("_".join(parts[1:-1]), int(parts[-1]))
    elif parts[0] == "wait" and len(parts) == 2:
        emu.tick(int(parts[1]), render_last=False)
    else:
        raise ValueError(f"Unknown action format: {action_str}")


# ---------------------------------------------------------------------------
# Server lifecycle — single-owner emulator thread
# ---------------------------------------------------------------------------


def _fail(msg: str) -> None:
    global _emu_state, _emu_error
    _emu_state, _emu_error = "error", msg
    logger.error("emulator: %s", msg)


def _dispatch(cmd: _Cmd) -> None:
    """Execute one queued command on the owner thread."""
    if not cmd.future.set_running_or_notify_cancel():
        return
    try:
        cmd.future.set_result(cmd.fn(_emulator, *cmd.args))
    except Exception as exc:
        cmd.future.set_exception(exc)


def _publish(event: dict) -> None:
    """Broadcast from the worker thread onto the asyncio loop."""
    if _loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(broadcast(event), _loop)
    except RuntimeError:
        pass


def _emu_status() -> dict:
    return {"emulator_state": _emu_state, "error": _emu_error,
            "frame": _frame_no, "armed": _pending_state is not None}


def _emit_status() -> None:
    _publish({"type": "emulator", **_emu_status()})


def _emit_frame(png_bytes: bytes) -> None:
    """Encode and publish a frame at PUBLISH_HZ."""
    global _frame_png, _frame_no
    with _frame_lock:
        _frame_png = png_bytes
        _frame_no += 1
    _publish({
        "type": "screenshot",
        "data": {"image": base64.b64encode(png_bytes).decode("ascii"), "format": "png"},
    })


def _fail(msg: str) -> None:
    global _emu_state, _emu_error
    _emu_state, _emu_error = "error", msg
    logger.error("emulator: %s", msg)
    _emit_status()


def _publish_frame() -> None:
    """Render one frame, cache the PNG, push it to clients."""
    global _frame_png, _frame_no
    try:
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
    """Sole owner of the emulator. Nothing else may call into PyBoy."""
    global _emulator, _reader
    logger.info("emulator worker started")
    period = 1.0 / TARGET_FPS
    publish_period = 1.0 / PUBLISH_HZ
    next_due = time.perf_counter()
    next_publish = time.perf_counter()

    while not _emu_stop.is_set():
        # Commands first — boot, load_state, button presses.
        try:
            _dispatch(_emu_cmds.get(timeout=0.05))
            while True:
                try:
                    _dispatch(_emu_cmds.get_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass

        running = (_emulator is not None and _emu_state == "ready"
                   and _control_state == "running")
        now = time.perf_counter()
        
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
            if next_due < now - 0.25:
                next_due = now

        # Publish frame at PUBLISH_HZ regardless of agent running
        if _emulator is not None and _emu_state == "ready" and now >= next_publish:
            _publish_frame()
            next_publish = now + (1.0 / PUBLISH_HZ if running else 1.0)

    if _emulator is not None:
        try:
            _emulator.close()
        finally:
            _emulator = None
            _reader = None
    logger.info("emulator worker stopped")


def _boot(_unused, rom_path: str, state_path: Optional[str]) -> dict:
    """Construct + load the emulator. Runs on the owner thread."""
    global _emulator, _reader, _emu_state, _emu_error
    from pokemon_agent.emulator import create_emulator
    if _emulator is not None:          # wedged instance from a failed boot
        try:
            _emulator.close()
        except Exception:
            pass
        _emulator = None
    _emu_state, _emu_error = "booting", None
    _emit_status()
    try:
        emu = create_emulator(rom_path)
        if _config.game_type == "red":
            from pokemon_agent.memory.red import PokemonRedReader
            _reader = PokemonRedReader(emu)
        else:
            from pokemon_agent.memory.firered import FireRedMemoryReader
            _reader = FireRedMemoryReader(emu)
        if state_path:
            emu.load_state(state_path)
        emu.tick(60)
        _emulator = emu
        _emu_state = "ready"
        _emit_status()
        _publish_frame()  # first frame lands immediately
        return {"state": _emu_state, "frame": emu.frame_count}
    except Exception as exc:
        _fail(f"boot failed: {type(exc).__name__}: {exc}")
        raise


def _start_emulator_thread() -> None:
    global _emu_thread
    if _emu_thread is not None and _emu_thread.is_alive():
        return
    _emu_stop.clear()
    _emu_thread = threading.Thread(target=_emulator_worker, name="emu", daemon=True)
    _emu_thread.start()


def configure(config: GameConfig):
    """Set server configuration (call before app startup)."""
    global _config
    _config = config


@app.on_event("startup")
async def _startup():
    global _emulator, _reader, _start_time, _config, _loop, _emu_thread
    _loop = asyncio.get_running_loop()
    _start_time = time.time()

    if _config is None:
        # Config can be injected via environment or set beforehand
        logger.warning("No GameConfig set — emulator will NOT start.")
        logger.warning("Call server.configure(GameConfig(...)) before startup.")
        return

    # Start the emulator thread
    _start_emulator_thread()

    # Create data directories
    data_dir = Path(_config.data_dir).expanduser().resolve()
    (data_dir / "saves").mkdir(parents=True, exist_ok=True)

    # Initialise the game-session manager.
    global _session_mgr
    from pokemon_agent.sessions import GameSessionManager
    _session_mgr = GameSessionManager(str(data_dir))

    # Try mounting dashboard
    if not _config.no_dashboard:
        try:
            import pokemon_agent.dashboard as dashboard_mod  # noqa: F401
            from fastapi.staticfiles import StaticFiles
            dash_dir = Path(dashboard_mod.__file__).parent / "static"
            if dash_dir.is_dir():
                app.mount("/dashboard", StaticFiles(directory=str(dash_dir), html=True), name="dashboard")
                logger.info("Dashboard mounted at /dashboard")
            else:
                logger.warning("Dashboard module found but no static/ directory")
        except ImportError:
            logger.warning("Dashboard not installed — /dashboard unavailable")
            logger.warning("Install with: pip install pokemon-agent[dashboard]")

    # Auto-load a save state if specified — ARM ONLY, don't boot yet
    if _config.load_state:
        saves_dir = data_dir / "saves"
        state_path = saves_dir / f"{_config.load_state}.state"
        if state_path.exists():
            global _pending_state
            _pending_state = str(state_path)
            logger.info("Armed save state (press START to boot): %s", _config.load_state)
        else:
            logger.warning("Save state not found: %s", state_path)

    logger.info(f"Ready — listening on port {_config.port}")
    logger.info("Endpoints: /, /state, /screenshot, /action, /save, /load, /saves, /minimap, /health, /ws")


@app.on_event("shutdown")
def _shutdown():
    """Cleanup emulator on server shutdown."""
    global _emulator, _emu_thread, _emu_stop
    _emu_stop.set()
    if _emu_thread and _emu_thread.is_alive():
        _emu_thread.join(timeout=2.0)
    if _emulator:
        logger.info("Shutting down emulator...")
        try:
            _emulator.close()
        except Exception:
            pass
        _emulator = None
    logger.info("Server shutdown complete.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Server info."""
    return {
        "name": "pokemon-agent",
        "version": __version__,
        "game": _config.game_type if _config else None,
        "rom": _config.rom_path if _config else None,
        "uptime_seconds": round(time.time() - _start_time, 1) if _start_time else 0,
        "emulator_ready": _emulator is not None,
        **_emu_status(),
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "emulator_ready": _emulator is not None, **_emu_status()}


@app.get("/state")
async def get_state():
    """Full game state JSON."""
    state = await emu_call(_state_dict)
    return JSONResponse(content=state)


def _grid_png(emu, scale: int) -> bytes:
    """Render the current frame with the A1..J9 grid overlay.
    Runs on the emulator owner thread. The worker ticks with rendering off,
    so refresh the framebuffer before capture.
    """
    from pokemon_agent.collision import build_collision_grid
    from pokemon_agent.overlay import render_grid_overlay_bytes
    emu.tick(1, render_last=True)
    walkable = None
    try:
        player = _reader.read_player() or {}
        col = build_collision_grid(emu, facing=player.get("facing"),
                                   player_pos=player.get("position"))
        if col.get("valid"):
            walkable = col["walkable"]
    except Exception:
        logger.debug("collision unavailable for overlay", exc_info=True)
    return render_grid_overlay_bytes(emu.get_screen(), scale=scale,
                                     walkable=walkable)


@app.get("/screenshot/grid")
async def screenshot_grid(scale: int = 4):
    """Current frame with a labelled A1..J9 movement grid drawn on top."""
    if not 1 <= scale <= 8:
        raise HTTPException(status_code=400, detail="scale must be 1..8")
    try:
        return Response(content=await emu_call(_grid_png, scale),
                        media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("grid screenshot failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")



@app.get("/screenshot")
async def screenshot():
    """Current emulator frame as PNG image (served from cache)."""
    with _frame_lock:
        png = _frame_png
    if png is None:
        raise HTTPException(status_code=503,
                            detail=f"No frame yet ({_emu_state})")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store",
                             "X-Frame-Number": str(_frame_no)})


@app.get("/screenshot/base64")
async def screenshot_base64():
    """Current emulator frame as base64-encoded PNG in JSON (served from cache)."""
    with _frame_lock:
        png = _frame_png
    if png is None:
        raise HTTPException(status_code=503,
                            detail=f"No frame yet ({_emu_state})")
    b64 = base64.b64encode(png).decode("ascii")
    return {"image": b64, "format": "png", "frame": _frame_no}


@app.post("/event")
async def push_event(req: EventRequest):
    """Push an agent-narration event to the dashboard (broadcast over WS).

    The agent calls this to make its reasoning visible on the stream:
      - type "reasoning" / "decision" / "alert": send `text`
      - type "key_moment": send `description` (+ optional `category`:
        milestone | badge | catch | alert)
    These are display-only; they are NOT stored in conversation history.
    """
    event: dict = {"type": req.type}
    if req.text is not None:
        event["text"] = req.text
    if req.description is not None:
        event["description"] = req.description
    if req.category is not None:
        event["category"] = req.category
    # Persist real milestones into the active session's timeline.
    if req.type in ("key_moment", "moment") and req.description \
            and _active_session is not None and _session_mgr is not None:
        _session_mgr.add_milestone(_active_session, req.description,
                                   req.category or "milestone")
    await broadcast(event)
    return {"success": True, "broadcast_to": len(_ws_clients)}


@app.get("/objectives")
async def get_objectives():
    """Current objective list (primary/secondary/tertiary + done flags)."""
    return {"objectives": _objectives}


@app.post("/objectives")
async def set_objectives(req: ObjectivesRequest):
    """Replace the full objective list and broadcast it to the dashboard.

    The player (agent or autopilot) sets real goals here so the dashboard
    reflects the actual plan instead of static placeholder text.
    """
    global _objectives
    _objectives = [o.model_dump() for o in req.objectives]
    if _active_session is not None and _session_mgr is not None:
        _active_session.objectives = _objectives
        _session_mgr.save(_active_session)
    await broadcast({"type": "objectives", "objectives": _objectives})
    return {"success": True, "objectives": _objectives}


@app.get("/control")
async def get_control():
    """Current autopilot run state: running | paused | stopped."""
    return {"state": _control_state}


@app.post("/control")
async def set_control(req: ControlRequest):
    """Set the autopilot run state (drives the Start/Pause/Stop buttons).

    A standalone `pokemon-agent play` loop polls this and only takes actions
    while the state is "running". This endpoint is the wiring behind the
    dashboard's control buttons; it does not itself drive the emulator.
    """
    global _control_state, _pending_state
    valid = {"running", "paused", "stopped"}
    if req.state not in valid:
        raise HTTPException(status_code=400,
                            detail=f"state must be one of {sorted(valid)}")

    if req.state == "running" and _emu_state in ("idle", "error"):
        if _active_session is None:
            raise HTTPException(status_code=409,
                                detail="No active game — create or load one first")
        _start_emulator_thread()
        await emu_call(_boot, _config.rom_path, _pending_state, timeout=120)

    _control_state = req.state
    await broadcast({"type": "control", "state": _control_state})
    return {"success": True, "state": _control_state, **_emu_status()}


# ---------------------------------------------------------------------------
# Game sessions — new game / load game / list / delete
# ---------------------------------------------------------------------------

def _game_summary() -> dict:
    if _active_session is None:
        return {"active": None}
    gs = _active_session
    return {"active": {"id": gs.id, "name": gs.name, "game": gs.game,
                       "hermes_session_id": gs.hermes_session_id,
                       "objectives": gs.objectives, "stats": gs.stats}}


async def _activate(gs) -> None:
    """Make `gs` the active session: sync objectives, broadcast, persist."""
    global _active_session, _objectives
    _active_session = gs
    _objectives = gs.objectives or _objectives
    _session_mgr.save(gs)

    await broadcast({"type": "objectives", "objectives": _objectives})
    await broadcast({"type": "game", **_game_summary()})


@app.get("/games")
async def list_games():
    """List all game sessions (newest first) + which one is active."""
    if _session_mgr is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    return {"games": _session_mgr.list(),
            "active": _active_session.id if _active_session else None}


@app.get("/games/current")
async def current_game():
    """The active game session summary (or {active: null})."""
    return _game_summary()


@app.post("/games/new")
async def new_game(req: NewGameRequest):
    """Start a NEW game: fresh emulator boot + a fresh session manifest.

    Resets the emulator to the ROM's title/boot (no save loaded) and creates a
    new GameSession (new Hermes brain — hermes_session_id starts null and is
    bound on the autopilot's first turn).
    """
    global _pending_state
    if _session_mgr is None or _config is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    gs = _session_mgr.create(name=req.name, game=_config.game_type)
    _pending_state = None
    await _activate(gs)

    await broadcast({"type": "control", "state": _control_state})
    return {"success": True, "game": gs.to_dict()}


@app.post("/games/{sid}/load")
async def load_game(sid: str = PathParam(..., pattern=SID_PATTERN)):
    """Load an existing game session: restore its latest save-state and make
    it active (its Hermes session id is restored too, so the autopilot resumes
    the SAME brain). If the session has no save yet, just activate it."""
    global _pending_state
    if _session_mgr is None or _config is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    gs = _session_mgr.load(sid)
    if gs is None:
        raise HTTPException(status_code=404, detail=f"Game session not found: {sid}")

    latest = _session_mgr.latest_save_path(sid)
    _pending_state = str(latest) if latest else None
    await _activate(gs)
    _emit_status()
    return {"success": True, "game": gs.to_dict(),
            "restored_save": latest.stem if latest else None,
            "armed": True, "note": "press START to boot"}


@app.post("/games/{sid}/hermes")
async def bind_hermes(sid: str = PathParam(..., pattern=SID_PATTERN), req: HermesSessionRequest = None):
    """Bind/refresh the Hermes session id for a game (autopilot calls this on
    its first turn so the run's brain memory is persisted in the manifest)."""
    if _session_mgr is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    gs = (_active_session if (_active_session and _active_session.id == sid)
          else _session_mgr.load(sid))
    if gs is None:
        raise HTTPException(status_code=404, detail=f"Game session not found: {sid}")
    gs.hermes_session_id = req.hermes_session_id
    _session_mgr.save(gs)
    await broadcast({"type": "game", **_game_summary()})
    return {"success": True, "hermes_session_id": gs.hermes_session_id}


@app.delete("/games/{sid}")
async def delete_game(sid: str = PathParam(..., pattern=SID_PATTERN)):
    """Delete a game session and its saves (cannot delete the active one)."""
    if _session_mgr is None:
        raise HTTPException(status_code=503, detail="Session manager not ready")
    if _active_session and _active_session.id == sid:
        raise HTTPException(status_code=400, detail="Cannot delete the active game; load another first.")
    ok = _session_mgr.delete(sid)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Game session not found: {sid}")
    return {"success": True, "deleted": sid}


@app.post("/action")
async def execute_actions(req: ActionRequest):
    """Execute a sequence of game actions."""
    _ensure_emulator()
    try:
        executed = 0
        for action_str in req.actions:
            await emu_call(_do_action, action_str)
            executed += 1

        state_after = await emu_call(_state_dict)

        # Bump per-session stats.
        if _active_session is not None and _session_mgr is not None:
            s = _active_session.stats
            s["actions"] = s.get("actions", 0) + executed
            s["turns"] = s.get("turns", 0) + 1
            _session_mgr.save(_active_session)

        try:
            png_bytes = await emu_call(_screenshot_bytes)
            screenshot_b64 = base64.b64encode(png_bytes).decode("ascii")
        except Exception:
            screenshot_b64 = None

        # Broadcast to WebSocket clients
        await broadcast({
            "type": "action",
            "actions": req.actions,
            "actions_executed": executed,
            "state_after": state_after,
        })
        # Also push the latest frame so the dashboard updates immediately
        if screenshot_b64:
            await broadcast({
                "type": "screenshot",
                "data": {"image": screenshot_b64, "format": "png"},
            })

        return {
            "success": True,
            "actions_executed": executed,
            "state_after": state_after,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action error: {e}")


@app.post("/save")
async def save_state(req: SaveRequest):
    """Save emulator state. Routed into the active game session's folder when
    one is active; otherwise the legacy flat saves/ dir."""
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
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    path = _resolve_save(req.name)
    await emu_call(lambda e, p=str(path): e.load_state(p))
    state_after = await emu_call(_state_dict)
    await broadcast({"type": "state_update", "reason": "load", "state": state_after})
    return {"success": True, "name": req.name, "state_after": state_after}


@app.get("/saves")
async def list_saves():
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    d = _saves_dir()
    files = sorted(d.glob("*.state"), key=lambda f: f.stat().st_mtime, reverse=True)
    return {"dir": str(d),
            "session": _active_session.id if _active_session else None,
            "saves": [{"name": f.stem, "file": f.name,
                       "size_bytes": f.stat().st_size,
                       "modified": f.stat().st_mtime} for f in files]}


@app.get("/map/ascii")
async def map_ascii():
    """The current on-screen walkability grid as an ASCII map (text/plain).

    @ = player (E5), . = walkable, # = blocked. Read from RAM collision data,
    so it is ground truth — not a guess from pixels.
    """
    _ensure_emulator()
    try:
        def _ascii(emu) -> str:
            from pokemon_agent.collision import build_collision_grid, render_ascii_map
            return render_ascii_map(build_collision_grid(emu), legend=True)
        text = await emu_call(_ascii)
        return Response(content=text, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASCII map error: {e}")


@app.get("/minimap")
async def minimap():
    """Simple ASCII minimap — current map name + player position."""
    _ensure_emulator()
    try:
        state = await emu_call(_state_dict)
        map_info = state.get("map", {})
        player = state.get("player", {})
        map_name = map_info.get("map_name", "Unknown")
        pos = player.get("position", {})
        x = pos.get("x", "?")
        y = pos.get("y", "?")

        lines = [
            f"=== {map_name} ===",
            f"Player position: ({x}, {y})",
            "",
            "  N",
            "W + E",
            "  S",
        ]
        text = "\n".join(lines)
        return Response(content=text, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Minimap error: {e}")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Live event stream via WebSocket."""
    global _screenshot_task
    
    await ws.accept()
    _ws_clients.add(ws)
    
    # Start screenshot broadcast task on first client
    if len(_ws_clients) == 1 and _screenshot_task is None:
        _screenshot_task = asyncio.create_task(_screenshot_broadcast_task())
    
    try:
        # Send a welcome message
        await ws.send_json({
            "type": "connected",
            "version": __version__,
            "emulator_ready": _emulator is not None,
        })
        # Also send current emulator status so dashboard shows correct state immediately
        await ws.send_json({"type": "emulator", **_emu_status()})
        # Backfill: replay recent narration/milestone/action events so the
        # Field Log is populated immediately instead of starting empty.
        if _event_history:
            await ws.send_json({
                "type": "replay",
                "events": list(_event_history),
            })
        # Send current objectives + control state so the panel + buttons sync.
        await ws.send_json({"type": "objectives", "objectives": _objectives})
        await ws.send_json({"type": "control", "state": _control_state})
        await ws.send_json({"type": "game", **_game_summary()})
        # Keep alive — wait for client messages (or disconnect)
        while True:
            data = await ws.receive_text()
            # Clients can send a "ping" to keep alive
            if data.strip().lower() == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)
        # Stop screenshot broadcast task when last client disconnects
        if len(_ws_clients) == 0 and _screenshot_task is not None:
            _screenshot_task.cancel()
            _screenshot_task = None


# ---------------------------------------------------------------------------
# Dashboard fallback — only registered if dashboard static files are missing
# ---------------------------------------------------------------------------

def _register_dashboard_fallback():
    """Register a fallback route for /dashboard if static files aren't available."""
    try:
        import pokemon_agent.dashboard as _dm
        static_dir = Path(_dm.__file__).parent / "static"
        if static_dir.is_dir() and (static_dir / "index.html").exists():
            return  # Dashboard exists — don't register fallback
    except ImportError:
        pass

    @app.get("/dashboard")
    @app.get("/dashboard/{path:path}")
    async def dashboard_fallback(path: str = ""):
        raise HTTPException(
            status_code=404,
            detail="Dashboard not installed. Install with: pip install pokemon-agent[dashboard]",
        )

_register_dashboard_fallback()
