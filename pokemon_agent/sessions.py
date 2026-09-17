import copy
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pokemon-agent.sessions")

_SID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[0-9a-f]{6}$")
SCHEMA_VERSION = 1


class InvalidSessionId(ValueError):
    """Session id failed validation — never touch the filesystem with it."""


DEFAULT_OBJECTIVES = [
    {"tier": "primary", "text": "Become Pokémon League Champion — earn all 8 badges", "done": False},
    {"tier": "secondary", "text": "Deliver Oak's Parcel · get the Pokédex", "done": False},
    {"tier": "tertiary", "text": "Build a balanced team", "done": False},
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GameSession:
    id: str
    name: str
    game: str = "red"
    hermes_session_id: Optional[str] = None
    objectives: List[Dict[str, Any]] = field(
        default_factory=lambda: copy.deepcopy(DEFAULT_OBJECTIVES))
    milestones: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "turns": 0, "actions": 0, "blackouts": 0, "saves": 0,
    })
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameSession":
        known = set(cls.__dataclass_fields__)
        kw = {k: v for k, v in d.items() if k in known}
        if "id" not in kw:
            raise ValueError("manifest missing 'id'")
        kw.setdefault("name", kw["id"])
        return cls(**kw)


class GameSessionManager:
    """Disk-backed CRUD for game sessions under <data_dir>/games/."""

    def __init__(self, data_dir: str):
        self.root = Path(data_dir).expanduser().resolve() / "games"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # --- paths ---
    def _dir(self, sid: str) -> Path:
        if not isinstance(sid, str) or not _SID_RE.match(sid):
            raise InvalidSessionId(f"invalid session id: {sid!r}")
        d = (self.root / sid).resolve()
        if d != self.root / sid or not d.is_relative_to(self.root):
            raise InvalidSessionId(f"session id escapes root: {sid!r}")
        return d

    def _manifest(self, sid: str) -> Path:
        return self._dir(sid) / "manifest.json"

    def saves_dir(self, sid: str, create: bool = True) -> Path:
        d = self._dir(sid) / "saves"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # --- persistence ---
    def save(self, gs: GameSession) -> GameSession:
        gs.updated_at = _now_iso()
        gs.schema_version = SCHEMA_VERSION
        d = self._dir(gs.id)
        d.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(gs.to_dict(), indent=2)
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=str(d), prefix=".manifest-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self._manifest(gs.id))
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return gs

    def load(self, sid: str) -> Optional[GameSession]:
        try:
            mf = self._manifest(sid)
        except InvalidSessionId:
            logger.warning("rejected session id: %r", sid)
            return None
        if not mf.exists():
            return None
        try:
            return GameSession.from_dict(json.loads(mf.read_text()))
        except Exception as exc:
            logger.error("corrupt manifest %s: %s: %s", mf, type(exc).__name__, exc)
            return None

    def exists(self, sid: str) -> bool:
        try:
            return self._manifest(sid).exists()
        except InvalidSessionId:
            return False

    def create(self, name: Optional[str] = None, game: str = "red") -> GameSession:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        gs = GameSession(id=sid, name=name or f"Run {sid}", game=game)
        self.saves_dir(sid)  # make the saves folder
        return self.save(gs)

    def delete(self, sid: str) -> bool:
        d = self._dir(sid)
        if d.exists():
            shutil.rmtree(d)
            return True
        return False

    def list(self) -> List[Dict[str, Any]]:
        """Summaries of all sessions, newest first."""
        out: List[Dict[str, Any]] = []
        for d in self.root.iterdir():
            if not d.is_dir():
                continue
            gs = self.load(d.name)
            if not gs:
                continue
            saves = list(self.saves_dir(gs.id, create=False).glob("*.state"))
            latest = self.latest_save_path(gs.id)
            out.append({
                "id": gs.id, "name": gs.name, "game": gs.game,
                "hermes_session_id": gs.hermes_session_id,
                "badges": _latest_badges(gs),
                "save_count": len(saves),
                "latest_save": latest.stem if latest else None,
                "turns": gs.stats.get("turns", 0),
                "milestones": len(gs.milestones),
                "created_at": gs.created_at, "updated_at": gs.updated_at,
            })
        out.sort(key=lambda x: x["updated_at"], reverse=True)
        return out

    # --- per-session save-state listing ---
    def list_saves(self, sid: str) -> List[Dict[str, Any]]:
        d = self.saves_dir(sid, create=False)
        if not d.exists():
            return []
        out = []
        for f in sorted(d.glob("*.state")):
            st = f.stat()
            out.append({"name": f.stem, "size_bytes": st.st_size, "modified": st.st_mtime})
        out.sort(key=lambda x: x["modified"], reverse=True)
        return out

    def latest_save_path(self, sid: str) -> Optional[Path]:
        """Newest save-state by mtime, or None."""
        saves = list(self.saves_dir(sid, create=False).glob("*.state"))
        if not saves:
            return None
        return max(saves, key=lambda f: f.stat().st_mtime)

    def next_save_name(self, sid: str, turn: int = 0) -> str:
        """Zero-padded so lexical and chronological order agree."""
        return f"turn_{turn:06d}"

    def prune_saves(self, sid: str, keep: int = 20) -> int:
        """Delete all but the *keep* newest saves. Returns count removed."""
        saves = sorted(self.saves_dir(sid, create=False).glob("*.state"),
                       key=lambda f: f.stat().st_mtime, reverse=True)
        removed = 0
        for f in saves[keep:]:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    # --- milestone helper ---
    def add_milestone(self, gs: GameSession, description: str, category: str = "milestone"):
        gs.milestones.insert(0, {
            "description": description, "category": category,
            "turn": gs.stats.get("turns", 0), "at": _now_iso(),
        })
        gs.milestones = gs.milestones[:100]
        self.save(gs)


def _latest_badges(gs: GameSession) -> int:
    return sum(1 for m in gs.milestones if m.get("category") == "badge")
