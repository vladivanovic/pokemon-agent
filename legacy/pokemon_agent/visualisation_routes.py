"""
Visualisation endpoints for the FastAPI server.
Adds:
  - /map/visualisation  -> self-contained HTML page with interactive path
  - /map/visualisation/data  -> JSON (regions + path segments) for dynamic loading
"""
import json
import os
from pathlib import Path
from typing import Optional
from typing import Optional

from fastapi import APIRouter, Response
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(prefix="/map", tags=["visualisation"])

REPO = Path("/home/vlad/pokemon-agent")
MAP_DATA = REPO / "v2" / "map_data.json"
TRACE = Path(os.path.expanduser("~/.pokemon-agent/flight_recorder.jsonl"))
PORTALS = Path(os.path.expanduser("~/.pokemon-agent/portals.json"))


def load_map_data():
    with MAP_DATA.open() as f:
        md = json.load(f)
    regions = []
    for r in md.get("regions", []):
        if r.get("id") == "-1":
            continue
        x, y = r["coordinates"]
        w, h = r["tileSize"]
        regions.append({"name": r["name"], "x": x, "y": y, "w": w, "h": h, "id": r["id"]})
    return regions


def load_path_segments():
    if not TRACE.exists():
        return []
    path = []
    with TRACE.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln.startswith("{"):
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("pos") is None:
                continue
            path.append({
                "turn": d.get("turn"),
                "ts": d.get("ts"),
                "map": d.get("map"),
                "choice": d.get("choice"),
                "x": d["pos"][0],
                "y": d["pos"][1],
            })
    # Split on map change
    segments = []
    if path:
        cur = [path[0]]
        for p in path[1:]:
            if p["map"] == cur[-1]["map"]:
                cur.append(p)
            else:
                segments.append(cur)
                cur = [p]
        segments.append(cur)
    return segments


def load_portal_offsets():
    """Portal offsets from the map graph: map -> (mx, my) world coords of map's (0,0)."""
    if not PORTALS.exists():
        return {}
    with PORTALS.open() as f:
        raw = json.load(f)
    offsets = {}
    for map_name, portals in raw.items():
        if not portals:
            continue
        first = next(iter(portals.values()))
        px = first.get("player_x")
        py = first.get("player_y")
        if px is not None and py is not None:
            # player is always at E5 (col=4, row=4 in 0-indexed 10x9 grid)
            # so map(0,0) = player_pos - (4,4) in screen cells
            # screen cell size is 8x8 pixels in world coords? Actually map_data.json coords are already world pixels.
            # The portal's player_x/player_y ARE world coordinates of the player at record time.
            # Each step on the map moves 8 world pixels (tile size).
            offsets[map_name] = (px - 4 * 8, py - 4 * 8)
    return offsets


def to_world(map_name: str, local_x: int, local_y: int, offsets: dict) -> Optional[tuple]:
    """Convert local map position to world coordinates using portal offsets."""
    off = offsets.get(map_name)
    if off is None:
        # Fallback: use map_data.json region coordinates if available
        regions = load_map_data()
        for r in regions:
            if r["name"] == map_name:
                # local (0,0) = region top-left
                return (r["x"] + local_x * 8, r["y"] + local_y * 8)
        return None
    mx, my = off
    return (mx + local_x * 8, my + local_y * 8)


@router.get("/visualisation", response_class=HTMLResponse)
async def visualisation_page():
    """Serve the self-contained visualisation HTML."""
    regions = load_map_data()
    segments = load_path_segments()
    offsets = load_portal_offsets()

    # Convert path to world coordinates
    world_segments = []
    for seg in segments:
        wseg = []
        for p in seg:
            world = to_world(p["map"], p["x"], p["y"], offsets)
            if world:
                wseg.append({
                    "turn": p["turn"],
                    "ts": p["ts"],
                    "map": p["map"],
                    "choice": p["choice"],
                    "wx": world[0],
                    "wy": world[1],
                })
        if wseg:
            world_segments.append(wseg)

    # Load the world map image as base64 for embedding
    import base64
    WORLD_MAP = Path("/tmp/world_map_1024.png")
    world_map_b64 = ""
    if WORLD_MAP.exists():
        with WORLD_MAP.open("rb") as f:
            world_map_b64 = base64.b64encode(f.read()).decode("ascii")

    # Build HTML with real world map background
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pokémon Red — Agent Path Visualisation</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:0; padding:20px; }}
  h1 {{ margin:0 0 10px; font-size:1.4rem; }}
  .wrap {{ max-width:1280px; margin:0 auto; }}
  .canvas-wrap {{ position:relative; width:1024px; height:1024px; margin:0 auto; border:1px solid #333; }}
  canvas {{ position:absolute; top:0; left:0; background:transparent; }}
  #bg {{ z-index:0; }}
  #fg {{ z-index:1; }}
  .legend {{ margin-top:12px; display:flex; flex-wrap:wrap; gap:8px; font-size:0.8rem; justify-content:center; }}
  .leg {{ display:flex; align-items:center; gap:4px; }}
  .swatch {{ width:14px; height:14px; border-radius:2px; border:1px solid #444; }}
  .info {{ margin-top:8px; font-size:0.85rem; color:#aaa; text-align:center; }}
  .tooltip {{ position:fixed; pointer-events:none; background:#222; border:1px solid #555;
               padding:8px 10px; border-radius:4px; font-size:0.75rem; z-index:10;
               display:none; max-width:260px; }}
  .tooltip .turn {{ font-weight:bold; color:#ffd866; }}
  .tooltip .map {{ color:#9fd89f; }}
  .tooltip .choice {{ color:#d8a8ff; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Pokémon Red — Agent Path Visualisation</h1>
  <div class="canvas-wrap">
    <img id="bg" src="data:image/png;base64,{world_map_b64}" width="1024" height="1024" style="display:block;">
    <canvas id="fg" width="1024" height="1024"></canvas>
  </div>
  <div class="legend" id="legend"></div>
  <div class="info" id="info">
    {len(world_segments)} map segments · turns {world_segments[0][0]["turn"] if world_segments else "?"}–{world_segments[-1][-1]["turn"] if world_segments else "?"}
  </div>
  <div class="tooltip" id="tt"></div>
</div>
<script>
const regions = {json.dumps(load_map_data())};
const segments = {json.dumps(world_segments)};
// World map is 1024x1024 pixels. The flight recorder gives local map coordinates
// (0-15, 0-15-ish per map). We need to convert using the map_data.json region
// rectangles which are already in world pixels (0-1023 range).
const WORLD_W = 1024, WORLD_H = 1024;
const cv = document.getElementById("fg");
const ctx = cv.getContext("2d");
cv.width = WORLD_W; cv.height = WORLD_H;
const tt = document.getElementById("tt");
const legDiv = document.getElementById("legend");

function wx(x) {{ return x; }}
function wy(y) {{ return y; }}

// --- draw path segments on top of the world map ---
segments.forEach((seg, si) => {{
  if (seg.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(wx(seg[0].wx), wy(seg[0].wy));
  for (let i = 1; i < seg.length; i++) {{
    ctx.lineTo(wx(seg[i].wx), wy(seg[i].wy));
  }}
  const hue = (si * 137) % 360;
  ctx.strokeStyle = `hsl(${{hue}}, 80%, 55%)`;
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fillStyle = `hsl(${{hue}}, 80%, 55%)`;
  ctx.beginPath();
  ctx.arc(wx(seg[0].wx), wy(seg[0].wy), 5, 0, Math.PI*2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(wx(seg[seg.length-1].wx), wy(seg[seg.length-1].wy), 5, 0, Math.PI*2);
  ctx.fill();
}});

// --- legend ---
const seen = new Set();
segments.forEach((seg, si) => {{
  if (!seg.length) return;
  const m = seg[0].map;
  if (seen.has(m)) return;
  seen.add(m);
  const hue = (si * 137) % 360;
  const div = document.createElement("div");
  div.className = "leg";
  div.innerHTML = `<span class="swatch" style="background:hsl(${{hue}}, 80%, 55%)"></span> ${{m}}`;
  legDiv.appendChild(div);
}});

// --- tooltip ---
cv.addEventListener("mousemove", e => {{
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let hit = null, best = Infinity;
  segments.forEach(seg => {{
    if (!seg.length) return;
    for (let i = 1; i < seg.length; i++) {{
      const x1 = wx(seg[i-1].wx), y1 = wy(seg[i-1].wy);
      const x2 = wx(seg[i].wx),   y2 = wy(seg[i].wy);
      const dx = x2 - x1, dy = y2 - y1;
      const len2 = dx*dx + dy*dy;
      let t = ((mx - x1)*dx + (my - y1)*dy) / len2;
      t = Math.max(0, Math.min(1, t));
      const px = x1 + t*dx, py = y1 + t*dy;
      const d2 = (mx-px)**2 + (my-py)**2;
      if (d2 < best && d2 < 144) {{ best = d2; hit = seg[i]; }}
    }}
  }});
  if (hit) {{
    tt.style.display = "block";
    tt.style.left = (e.clientX + 14) + "px";
    tt.style.top  = (e.clientY + 14) + "px";
    tt.innerHTML = `<span class="turn">turn ${{hit.turn}}</span><br>
                    <span class="map">${{hit.map}}</span><br>
                    <span class="choice">${{hit.choice}}</span>`;
  }} else {{
    tt.style.display = "none";
  }}
}});
cv.addEventListener("mouseleave", () => {{ tt.style.display = "none"; }});
</script>
</body>
</html>
"""
    return HTMLResponse(content=html)


@router.get("/visualisation/data")
async def visualisation_data():
    """JSON API: regions + path segments with world coordinates."""
    regions = load_map_data()
    segments = load_path_segments()
    offsets = load_portal_offsets()

    world_segments = []
    for seg in segments:
        wseg = []
        for p in seg:
            world = to_world(p["map"], p["x"], p["y"], offsets)
            if world:
                wseg.append({
                    "turn": p["turn"],
                    "ts": p["ts"],
                    "map": p["map"],
                    "choice": p["choice"],
                    "wx": world[0],
                    "wy": world[1],
                })
        if wseg:
            world_segments.append(wseg)

    return JSONResponse({
        "regions": load_map_data(),
        "segments": world_segments,
        "counts": {"regions": len(load_map_data()), "positions": sum(len(s) for s in world_segments), "segments": len(world_segments)}
    })


@router.get("/visualisation/portals")
async def visualisation_portals():
    """Debug: portal offsets for coordinate conversion."""
    offsets = load_portal_offsets()
    return JSONResponse({"offsets": offsets})