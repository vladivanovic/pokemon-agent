#!/usr/bin/env python3
"""Purge poisoned portal entries from portals.json.

A poisoned entry is one of:
  1. dest_map == 255          - the "return to last outdoor map" sentinel; its
                                recorded name is whatever was in transit, never
                                the real destination.
  2. A source map (e.g. Viridian Mart) with implausibly many cells pointing at
     ONE destination - real doors are 1-2 tiles wide, so >4 cells to one dest
     is stale-frame garbage. Keep the first 4 only when the cells are tightly
     clustered (a genuine gate interior connector), else drop all.
  3. Junk names ("Map N") that resolve to nothing.

Run standalone:  python3 scripts/purge_portals.py [--dry-run]
The MapGraph constructor calls the same clean_portals_data() at load, so this
script is for a one-time deep clean + verification.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pokemon_agent.map_graph import _is_interior  # noqa: E402

PORTALS = os.environ.get(
    "POKEMON_PORTALS",
    os.path.expanduser("~/.pokemon-agent/portals.json"))

# A real door is 1-2 tiles wide: a map never legitimately has more than a
# couple of cells leading to the same destination, EXCEPT gate interiors
# (pass-through connectors). A connector spans the whole room - many rows AND
# columns - but each ROW has at most 2-3 door cells. Stale-frame garbage has
# whole ROWS of "doors" (every tile in transit decodes as one).
MAX_SAME_DEST = 4


def canon(name):
    return " ".join(str(name or "").lower().split())


def clean_portals_data(raw: dict) -> tuple:
    """Return (clean, stats). Stats is a dict of counts for reporting."""
    clean = {}
    stats = {"dropped_255": 0, "dropped_junkname": 0, "dropped_interior": 0,
             "dropped_overflow": 0, "dropped_maps": 0, "kept_cells": 0}
    for map_name, cells in (raw or {}).items():
        if not isinstance(cells, dict):
            stats["dropped_maps"] += 1
            continue
        # Building interiors have no named doors at all.
        if _is_interior(map_name):
            stats["dropped_interior"] += len(cells)
            continue
        grouped = {}
        for cell, entry in cells.items():
            if not isinstance(entry, dict):
                stats["dropped_junkname"] += 1
                continue
            if int(entry.get("dest_map", -1)) == 255:
                stats["dropped_255"] += 1
                continue
            dn = entry.get("dest_map_name") or ""
            if not dn or dn.startswith("Map "):
                stats["dropped_junkname"] += 1
                continue
            grouped.setdefault(dn, {})[cell] = entry
        kept = {}
        for dst, group in grouped.items():
            if len(group) > MAX_SAME_DEST:
                # Connector check: a genuine gate interior has door cells on
                # few columns but many rows. Stale garbage has many of both.
                rows = {int(c[1:]) for c in group}
                cols = {c[0].upper() for c in group}
                if len(cols) > 3:
                    # Door cells on >3 columns for ONE destination is not a
                    # connector - stale-frame garbage.
                    stats["dropped_overflow"] += len(group)
                    continue
                # Plausible connector: keep the first MAX_SAME_DEST cells.
                for c in list(group)[:MAX_SAME_DEST]:
                    kept[c] = group[c]
                stats["dropped_overflow"] += len(group) - MAX_SAME_DEST
                continue
            kept.update(group)
        if kept:
            clean[map_name] = kept
            stats["kept_cells"] += len(kept)
        else:
            stats["dropped_maps"] += 1
    return clean, stats


def main():
    dry = "--dry-run" in sys.argv
    with open(PORTALS, "r", encoding="utf-8") as f:
        raw = json.load(f)
    before = sum(len(v) for v in raw.values() if isinstance(v, dict))
    clean, stats = clean_portals_data(raw)
    after = stats["kept_cells"]

    print(f"portals: {PORTALS}")
    print(f"  before: {before} cells across {len(raw)} maps")
    print(f"  after : {after} cells across {len(clean)} maps")
    print(f"  dropped: {stats['dropped_255']} dest255, "
          f"{stats['dropped_junkname']} junk-name, "
          f"{stats['dropped_overflow']} overflow, "
          f"{stats['dropped_maps']} empty maps")

    print("\n  per-map after clean:")
    for m in sorted(clean):
        dests = {}
        for c, e in clean[m].items():
            dests.setdefault(e["dest_map_name"], []).append(c)
        dstr = "; ".join(f"{k}: {len(v)}" for k, v in dests.items())
        print(f"    {m}: {dstr}")

    if dry:
        print("\n(dry run - not written)")
        return
    tmp = PORTALS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=1, sort_keys=True)
    os.replace(tmp, PORTALS)
    print(f"\nwritten: {PORTALS}")


if __name__ == "__main__":
    main()
