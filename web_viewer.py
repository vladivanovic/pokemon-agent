#!/usr/bin/env python3
"""Minimal live viewer for play_full_game.py runs (stdlib only).

Why not the legacy dashboard? legacy/pokemon_agent/server.py is a
1,300-line FastAPI app built around the LEGACY emulator/session
machinery (pokemon_agent.emulator.create_emulator, game sessions,
objectives, a WebSocket contract, pydantic models). Feeding it from
agent/pyboy_emu.py means re-platforming its emulator worker, and it
drags in fastapi/uvicorn/pydantic - against this repo's zero-dependency
core and the "runs on a Nano" requirement. This module keeps the
legacy page's shape (live screen + status panel + start/stop) with
nothing but http.server:

    GET  /           the page (screen <img>, status panel, buttons)
    GET  /frame.png  the current game frame (PNG, 160x144 x3)
    GET  /state.json map / badges / money / party / current phase
    POST /control    {"action": "start" | "stop"}
    POST /directive  one live-redirect directive (the same JSON the
                     --directives file channel takes: {"goal": ...}
                     or {"action": "pause" | "resume" | "stop"});
                     the page's redirect form posts here

The provider is duck-typed: anything with frame_png() -> bytes|None,
state_dict() -> dict and control(action) -> dict; /directive also
uses directive(obj) -> dict when the provider offers one (the runner
appends to the run's directive file - one channel, two doors).
Frame reads are
unsynchronised reads of the emulator's screen buffer from the HTTP
thread while the game thread ticks - worst case is a torn frame in
the viewer, never a control input. State reads are WRAM reads, the
same thing every probe logger did. Frame generation is throttled by
the provider (play_full_game caches at ~250 ms).
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>pokemon-agent - live run</title>
<style>
  body { background:#101418; color:#d7dee6; font-family:ui-monospace,
         Menlo, Consolas, monospace; margin:0; padding:18px; }
  h1 { font-size:16px; margin:0 0 12px; letter-spacing:.08em; }
  .wrap { display:flex; gap:18px; flex-wrap:wrap; }
  .screen-shell { background:#1b2129; border:1px solid #39424e;
                  padding:10px; border-radius:6px; }
  img.screen { width:480px; height:432px; image-rendering:pixelated;
               background:#0a0c0e; display:block; }
  .panel { min-width:300px; max-width:460px; }
  .panel h2 { font-size:12px; text-transform:uppercase;
              letter-spacing:.12em; color:#8fa1b3; margin:14px 0 6px; }
  table { border-collapse:collapse; font-size:13px; }
  td { padding:2px 10px 2px 0; vertical-align:top; }
  td.k { color:#8fa1b3; }
  button { background:#233140; color:#d7dee6; border:1px solid #39424e;
           border-radius:4px; padding:6px 14px; cursor:pointer;
           font:inherit; }
  button:hover { background:#2d405a; }
  #log { font-size:11px; color:#9fb0c0; white-space:pre-wrap;
         max-height:220px; overflow-y:auto; }
  .badge { color:#ffd76a; }
</style>
</head>
<body>
<h1>POK&Eacute;MON RED &mdash; DETERMINISTIC AGENT, LIVE</h1>
<div class="wrap">
  <div class="screen-shell">
    <img id="gameScreen" class="screen" alt="game screen">
  </div>
  <div class="panel">
    <button id="btnStart">Start run</button>
    <button id="btnStop">Stop after phase</button>
    <h2>Status</h2>
    <table>
      <tr><td class="k">run</td><td id="stRun">-</td></tr>
      <tr><td class="k">phase</td><td id="stPhase">-</td></tr>
      <tr><td class="k">map</td><td id="stMap">-</td></tr>
      <tr><td class="k">money</td><td id="stMoney">-</td></tr>
      <tr><td class="k">badges</td><td id="stBadges" class="badge">-</td></tr>
      <tr><td class="k">dex</td><td id="stDex">-</td></tr>
    </table>
    <h2>Party</h2>
    <table id="party"></table>
    <h2>Redirect this run</h2>
    <div style="font-size:12px; color:#8fa1b3; margin-bottom:6px;">
      A goal is validated against the live state before it runs;
      rejections are logged with the reason. It takes effect at the
      next phase boundary or executor round.</div>
    <textarea id="dirGoal" rows="3" style="width:100%; box-sizing:border-box;
      background:#0a0c0e; color:#d7dee6; border:1px solid #39424e;
      border-radius:4px; font:inherit; font-size:12px;"
      >{"target_map_name": "ViridianCity", "reason": "detour"}</textarea>
    <div style="margin-top:6px;">
      <button id="btnGoal">Send goal</button>
      <button id="btnPause">Pause</button>
      <button id="btnResume">Resume</button>
      <button id="btnDirStop">Stop run</button>
      <span id="dirMsg" style="font-size:12px; color:#9fb0c0;"></span>
    </div>
    <h2>Progress</h2>
    <div id="log"></div>
  </div>
</div>
<script>
  var frame = document.getElementById('gameScreen');
  function tickFrame() { frame.src = '/frame.png?t=' + Date.now(); }
  function ctl(action) {
    fetch('/control', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action: action})}).then(tickState);
  }
  document.getElementById('btnStart').onclick = function(){ ctl('start'); };
  document.getElementById('btnStop').onclick = function(){ ctl('stop'); };
  function directive(obj) {
    fetch('/directive', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(obj)}).then(function(r){ return r.json(); })
    .then(function(d){
      document.getElementById('dirMsg').textContent =
        d.ok ? 'queued' : ('refused: ' + (d.error || '?'));
      tickState();
    }).catch(function(){});
  }
  function sendGoal() {
    var raw = document.getElementById('dirGoal').value;
    var obj;
    try { obj = JSON.parse(raw); }
    catch(e) {
      document.getElementById('dirMsg').textContent = 'bad JSON: ' + e;
      return;
    }
    // A bare goal object is wrapped; a full {"goal":...} or
    // {"action":...} directive is sent as written.
    if (obj && !obj.goal && !obj.action) obj = {goal: obj};
    directive(obj);
  }
  document.getElementById('btnGoal').onclick = sendGoal;
  document.getElementById('btnPause').onclick =
    function(){ directive({action: 'pause'}); };
  document.getElementById('btnResume').onclick =
    function(){ directive({action: 'resume'}); };
  document.getElementById('btnDirStop').onclick =
    function(){ directive({action: 'stop'}); };
  function tickState() {
    fetch('/state.json').then(function(r){ return r.json(); })
    .then(function(d){
      document.getElementById('stRun').textContent = d.run_status || '-';
      document.getElementById('stPhase').textContent =
        d.phase ? (d.phase + ' (' + (d.phase_index+1) + '/' + d.phase_total + ')') : '-';
      document.getElementById('stMap').textContent =
        (d.map || d.map_id) + ' (' + d.x + ',' + d.y + ')';
      document.getElementById('stMoney').textContent = d.money;
      document.getElementById('stBadges').textContent =
        (d.badges || []).join(', ') || 'none';
      document.getElementById('stDex').textContent = d.dex != null ? d.dex : '-';
      var rows = '';
      (d.party || []).forEach(function(m){
        rows += '<tr><td>sp 0x' + m.species_id.toString(16) + '</td>' +
                '<td>Lv ' + m.level + '</td>' +
                '<td>' + m.hp + '/' + m.max_hp + ' HP</td></tr>';
      });
      document.getElementById('party').innerHTML = rows;
      document.getElementById('log').textContent =
        (d.recent_log || []).join('\\n');
    }).catch(function(){});
  }
  setInterval(tickFrame, 500);
  setInterval(tickState, 1000);
  tickFrame(); tickState();
</script>
</body>
</html>
"""


def state_to_dict(state, world=None, info=None) -> dict:
    """Serialise a GameState (or duck-typed stand-in) for /state.json.

    Pure function of its inputs - unit-tested without an emulator.
    `info` carries runner bookkeeping (phase, run status, log tail)
    and wins on key collisions.
    """
    d = {
        "map_id": state.map_id,
        "map": None,
        "x": state.x,
        "y": state.y,
        "in_battle": bool(state.in_battle),
        "money": state.money,
        "badges": sorted(state.badges),
        "party": [
            {
                "species_id": m.species_id,
                "level": m.level,
                "hp": m.hp,
                "max_hp": m.max_hp,
            }
            for m in state.party
        ],
    }
    if world is not None:
        try:
            d["map"] = world.map_name(state.map_id)
        except Exception:
            d["map"] = None
    if info:
        d.update(info)
    return d


def make_server(host: str, port: int, provider):
    """Build (but do not start) the viewer HTTP server.

    provider: object with frame_png() -> bytes | None,
    state_dict() -> dict, control(action: str) -> dict, and
    optionally directive(obj) -> dict for POST /directive.
    """

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console for the run log
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path.startswith("/frame.png"):
                png = provider.frame_png()
                if png is None:
                    self._send(503, b"no frame yet", "text/plain")
                else:
                    self._send(200, png, "image/png")
            elif self.path.startswith("/state.json"):
                body = json.dumps(provider.state_dict()).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.startswith("/control"):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    action = str(req.get("action", ""))
                except Exception:
                    action = ""
                body = json.dumps(provider.control(action)).encode()
                self._send(200, body, "application/json")
            elif self.path.startswith("/directive"):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                except Exception:
                    req = None
                fn = getattr(provider, "directive", None)
                if req is None:
                    res = {"ok": False, "error": "body is not JSON"}
                elif fn is None:
                    res = {"ok": False,
                           "error": "no directive channel on this run"}
                else:
                    res = fn(req)
                body = json.dumps(res).encode()
                self._send(200, body, "application/json")
            else:
                self._send(404, b"not found", "text/plain")

    return ThreadingHTTPServer((host, port), Handler)
