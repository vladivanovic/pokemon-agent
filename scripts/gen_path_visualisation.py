#!/usr/bin/env python3
"""
Generate a static HTML visualisation of the agent's path overlaid on the
Gen 1 Kanto world map.

Sources:
  - v2/map_data.json  (region rectangles in world pixels)
  - ~/.pokemon-agent/flight_recorder.jsonl  (turn-by-turn positions)
Output: /home/vlad/pokemon-agent/docs/path_visualisation.html
"""
import json
import os
from pathlib import Path

REPO = Path("/home/vlad/pokemon-agent")
MAP_DATA = REPO / "v2" / "map_data.json"
TRACE = Path(os.path.expanduser("~/.pokemon-agent/flight_recorder.jsonl"))
OUT = REPO / "docs" / "path_visualisation.html"

# --------------------------------------------------------------------------
# 1. Load region rectangles
# --------------------------------------------------------------------------
with MAP_DATA.open() as f:
    md = json.load(f)

regions = []
for r in md.get("regions", []):
    if r.get("id") == "-1":   # the Kanto pseudo-region
        continue
    x, y = r["coordinates"]
    w, h = r["tileSize"]
    regions.append({
        "name": r["name"],
        "x": x, "y": y,
        "w": w, "h": h,
        "id": r["id"],
    })

# World bounds
min_x = min(r["x"] for r in regions)
min_y = min(r["y"] for r in regions)
max_x = max(r["x"] + r["w"] for r in regions)
max_y = max(r["y"] + r["h"] for r in regions)

# --------------------------------------------------------------------------
# 2. Load agent path
# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# 3. Build HTML
# --------------------------------------------------------------------------
# Scale to fit in 1200px wide, keep aspect
MARGIN = 40
CANVAS_W = 1200
SCALE = (CANVAS_W - 2 * MARGIN) / (max_x - min_x)
CANVAS_H = int((max_y - min_y) * SCALE + 2 * MARGIN)

# Colour map
MAP_COLOURS = {
    "town": "#e8dcc8",
    "city": "#d4c4a8",
    "route": "#a8d4a8",
    "gate": "#c8b898",
    "forest": "#88c888",
    "cave": "#988878",
    "building": "#b8a898",
    "sea": "#88b8e8",
    "default": "#d0d0d0",
}

def classify(name):
    n = name.lower()
    if "town" in n: return "town"
    if "city" in n: return "city"
    if n.startswith("route "): return "route"
    if "gate" in n: return "gate"
    if "forest" in n: return "forest"
    if "cave" in n or "tunnel" in n or "rock" in n: return "cave"
    if any(k in n for k in ["house", "lab", "mart", "center", "gym", "mansion", "tower"]): return "building"
    if "sea" in n or "route" in n and any(x in n for x in ["19","20","21","22","23","24","25"]): return "sea"
    return "default"

for r in regions:
    r["colour"] = MAP_COLOURS[classify(r["name"])]

# Path polylines - split on map change
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

# --------------------------------------------------------------------------
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pokémon Red — Agent Path Visualisation</title>
<style>
  body {{ font-family: system-ui, sans-serif; background:#111; color:#eee; margin:0; padding:20px; }}
  h1 {{ margin:0 0 10px; font-size:1.4rem; }}
  .wrap {{ max-width:1280px; margin:0 auto; }}
  canvas {{ background:#0a0a0a; border:1px solid #333; display:block; }}
  .legend {{ margin-top:12px; display:flex; flex-wrap:wrap; gap:8px; font-size:0.8rem; }}
  .leg {{ display:flex; align-items:center; gap:4px; }}
  .swatch {{ width:14px; height:14px; border-radius:2px; border:1px solid #444; }}
  .info {{ margin-top:8px; font-size:0.85rem; color:#aaa; }}
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
  <canvas id="cv" width="{CANVAS_W}" height="{CANVAS_H}"></canvas>
  <div class="legend" id="legend"></div>
  <div class="info" id="info">
    {len(path)} recorded positions · {len(segments)} map segments · turns {path[0]["turn"] if path else "?"}–{path[-1]["turn"] if path else "?"}
  </div>
  <div class="tooltip" id="tt"></div>
</div>
<script>
const regions = {json.dumps(regions)};
const segments = {json.dumps(segments)};
const SCALE = {SCALE};
const MIN_X = {min_x}, MIN_Y = {min_y};
const MARGIN = {MARGIN};

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const tt = document.getElementById("tt");
const legDiv = document.getElementById("legend");

function wx(x) {{ return MARGIN + (x - MIN_X) * SCALE; }}
function wy(y) {{ return MARGIN + (y - MIN_Y) * SCALE; }}

// --- draw regions ---
regions.forEach(r => {{
  ctx.fillStyle = r.colour;
  ctx.fillRect(wx(r.x), wy(r.y), r.w * SCALE, r.h * SCALE);
  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;
  ctx.strokeRect(wx(r.x), wy(r.y), r.w * SCALE, r.h * SCALE);
  // label
  if (r.w * SCALE > 60 && r.h * SCALE > 30) {{
    ctx.fillStyle = "#222";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(r.name, wx(r.x + r.w/2), wy(r.y + r.h/2 + 3));
  }}
}});

// --- draw path segments ---
segments.forEach((seg, si) => {{
  if (seg.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(wx(seg[0].x), wy(seg[0].y));
  for (let i = 1; i < seg.length; i++) {{
    ctx.lineTo(wx(seg[i].x), wy(seg[i].y));
  }}
  // colour by map
  const hue = (si * 137) % 360;
  ctx.strokeStyle = `hsl(${{hue}}, 80%, 55%)`;
  ctx.lineWidth = 2;
  ctx.stroke();

  // start/end markers
  ctx.fillStyle = `hsl(${{hue}}, 80%, 55%)`;
  ctx.beginPath();
  ctx.arc(wx(seg[0].x), wy(seg[0].y), 4, 0, Math.PI*2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(wx(seg[seg.length-1].x), wy(seg[seg.length-1].y), 4, 0, Math.PI*2);
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

// --- tooltip on hover ---
cv.addEventListener("mousemove", e => {{
  const rect = cv.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let hit = null, best = Infinity;
  segments.forEach(seg => {{
    if (!seg.length) return;
    for (let i = 1; i < seg.length; i++) {{
      const x1 = wx(seg[i-1].x), y1 = wy(seg[i-1].y);
      const x2 = wx(seg[i].x),   y2 = wy(seg[i].y);
      // point-to-segment distance
      const dx = x2 - x1, dy = y2 - y1;
      const len2 = dx*dx + dy*dy;
      let t = ((mx - x1)*dx + (my - y1)*dy) / len2;
      t = Math.max(0, Math.min(1, t));
      const px = x1 + t*dx, py = y1 + t*dy;
      const d2 = (mx-px)**2 + (my-py)**2;
      if (d2 < best && d2 < 100) {{ best = d2; hit = seg[i]; }}
    }}
  }});
  if (hit) {{
    tt.style.display = "block";
    tt.style.left = (e.clientX + 14) + "px";
    tt.style.top  = (e.clientY + 14) + "px";
    tt.innerHTML = `<span class="turn">turn ${{hit.turn}}</span><br>
                    <span class="map">${{hit.map}}</span><br>
                    <span class="choice">${{hit.choice}}</span><br>
                    pos: [${{hit.x}}, ${{hit.y}}]`;
  }} else {{
    tt.style.display = "none";
  }}
}});
cv.addEventListener("mouseleave", () => {{ tt.style.display = "none"; }});
</script>
</body>
</html>
"""

OUT.write_text(html)
print(f"Written {OUT}  ({len(regions)} regions, {len(path)} positions, {len(segments)} segments)")
print(f"World bounds: x {min_x}..{max_x}  y {min_y}..{max_y}")
print(f"Canvas: {CANVAS_W}x{CANVAS_H}  scale={SCALE:.3f}")