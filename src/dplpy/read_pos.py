__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2024  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
__license__ = "GNU GPLv3"

# Title: read_pos.py
# Project: OpenDendro dplPy
# Description: Read CooRecorder (.pos) coordinate files into dplPy ring-width
#              frames plus the per-series attributes CooRecorder records
#              (distance to pith, radius, comment, ...). A .pos file stores the
#              x,y positions (in mm) of ring boundaries along a measured path, so
#              a ring width is the Euclidean distance between consecutive
#              boundaries. Modeled on the read_pos() function in Colin Maher's
#              modendro R package (MIT), re-implemented for pandas. Requires
#              CooRecorder 7.8 or later (the first to store pith coordinates).

import os
import re
import math

import numpy as np
import pandas as pd

from ._validate import _warn

_MIN_COO_VERSION = 780   # 7.8.0, encoded as an int (see _coo_version)


def _coo_version(text):
    """CooRecorder version as a zero-padded int (7.8.1 -> 781, 7.7 -> 770)."""
    m = re.search(r"CooRecorder\s*=\s*([0-9]+(?:\.[0-9]+)*)", text, re.I)
    if not m:
        return None
    digits = m.group(1).replace(".", "")
    digits = (digits + "00")[:3]       # 78 -> 780, 781 -> 781
    try:
        return int(digits)
    except ValueError:
        return None


def _first_number(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else None


def _parse_header(lines):
    """Pull the metadata we care about out of the header. Keys vary by version
    in case and spelling, so every lookup is case-insensitive."""
    joined = "\n".join(lines)
    meta = {"comment": None, "coo_version": _coo_version(joined)}

    # outer date: a "DATED <year>" line
    od = None
    for ln in lines:
        if re.search(r"\bDATED\b", ln, re.I):
            od = _first_number(ln)
            break
    meta["od"] = int(od) if od is not None else None

    # millimetre marker (coordinates are already in mm when present)
    meta["mm"] = any("millimet" in ln.lower() for ln in lines)

    # DPI (informational unless the mm marker is absent)
    dpi = None
    for ln in lines:
        if re.search(r"#\s*DPI", ln, re.I):
            dpi = _first_number(ln)
            break
    meta["dpi"] = dpi

    # pith coordinates and the values CooRecorder itself computed
    m = re.search(r"PithCoordinates\s*=\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
                  joined, re.I)
    meta["pith"] = (float(m.group(1)), float(m.group(2))) if m else None
    m = re.search(r"Distance\s*To\s*Pith\s*=\s*(-?\d+(?:\.\d+)?)", joined, re.I)
    meta["d2pith_file"] = float(m.group(1)) if m else None
    m = re.search(r"Years\s*To\s*Pith\s*=\s*(-?\d+(?:\.\d+)?)", joined, re.I)
    meta["yrs_to_pith_file"] = int(float(m.group(1))) if m else None
    m = re.search(r"\bRadius\s*=\s*(-?\d+(?:\.\d+)?)", joined, re.I)
    meta["radius_file"] = float(m.group(1)) if m else None

    # free comment: a "#C ..." line that is not one of the metadata lines
    meta_pat = re.compile(
        r"DATED |Written\s*=|PithCoordinates\s*=|CalcRadius\s*=|CooRecorder\s*=|"
        r"licen[cs]edTo\s*=|YearsToPith\s*=|Radius\s*=|DistanceToPith\s*=", re.I)
    comments = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("#C "):
            body = s[3:].strip()
            if body and not meta_pat.search(body):
                comments.append(body)
    meta["comment"] = "; ".join(comments) if comments else None
    return meta


_COORD_RE = re.compile(r"^-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?")


def _parse_points(lines):
    """Return a list of points (dicts: x, y, type, label) in file order.

    type is one of reg (ring boundary), multi1/multi2 (a bent-path boundary and
    its via-point), W (seasonwood), gap (crack/marker). #ZEROn repeats a point
    n times to encode locally-absent (zero-width) rings.
    """
    points = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#") or s.upper().startswith("SCALE"):
            continue
        is_d = s.startswith("D")
        body = s[1:] if is_d else s

        # #ZERO repeats
        rep = 1
        zm = re.search(r"#ZERO(\d*)", body, re.I)
        if zm:
            rep = 1 + (int(zm.group(1)) if zm.group(1) else 0)
            body = body[:zm.start()].rstrip()

        coord_part, _, label = body.partition("#")
        coord_part = coord_part.strip()
        label = label.strip() or None
        if not _COORD_RE.match(coord_part):
            continue   # not a coordinate line

        pairs = coord_part.split()
        xy = []
        for p in pairs:
            if "," not in p:
                continue
            a, b = p.split(",")[:2]
            try:
                xy.append((float(a), float(b)))
            except ValueError:
                pass
        if not xy:
            continue

        if is_d:
            lab = (label or "").lower()
            ptype = "gap" if ("gap" in lab or "%" in lab) else "W"
            for _ in range(rep):
                points.append({"x": xy[0][0], "y": xy[0][1], "type": ptype,
                               "label": label})
        elif len(xy) >= 2:                        # multipoint: boundary + via-point
            for _ in range(rep):
                points.append({"x": xy[0][0], "y": xy[0][1], "type": "multi1",
                               "label": label})
                points.append({"x": xy[1][0], "y": xy[1][1], "type": "multi2",
                               "label": None})
        else:
            for _ in range(rep):
                points.append({"x": xy[0][0], "y": xy[0][1], "type": "reg",
                               "label": label})

    # a file that ends on a W or gap point has a dangling marker: drop it
    while points and points[-1]["type"] in ("W", "gap"):
        points.pop()
    return points


def _dist(p, q):
    return math.hypot(p["x"] - q["x"], p["y"] - q["y"])


def _parse_pos_file(path, default_od=None):
    """Parse one .pos file. Returns a dict of results, or raises ValueError with
    a short reason (collected into not_read by read_pos)."""
    with open(path, "r", encoding="latin-1") as fh:
        lines = fh.read().splitlines()
    if not lines or "DENDRO" not in lines[0]:
        raise ValueError("not a DENDRO .pos file (first line)")

    meta = _parse_header(lines)
    ver = meta["coo_version"]
    if ver is not None and ver < _MIN_COO_VERSION:
        raise ValueError("CooRecorder %s predates 7.8 (no pith coordinates)"
                         % (ver / 100.0))

    od = meta["od"] if meta["od"] is not None else default_od
    relative = od is None
    if relative:
        od = 0   # relative years: outermost ring = 0, counting inward negative

    pts = _parse_points(lines)
    boundaries = [p for p in pts if p["type"] in ("reg", "multi1", "multi2")]
    if len(boundaries) < 2:
        raise ValueError("fewer than two ring boundaries parsed")

    # distance (mm) of each point's incoming segment; first is undefined
    seg = [np.nan]
    for i in range(1, len(pts)):
        seg.append(_dist(pts[i], pts[i - 1]))

    # gap handling (modendro): keep boundary->firstgap, drop between-gap and, for
    # a lone gap, drop the leaving segment, so a crack does not count as growth.
    i = 0
    n = len(pts)
    while i < n:
        if pts[i]["type"] == "gap":
            j = i
            while j < n and pts[j]["type"] == "gap":
                j += 1
            run = j - i                      # gap points i..j-1
            for k in range(i + 1, j):        # between consecutive gaps
                seg[k] = np.nan
            if run == 1 and j < n:
                seg[j] = np.nan              # lone gap: drop the leaving segment
            i = j
        else:
            i += 1

    # assign calendar (or relative) years: outermost reg/multi1 boundary = od,
    # counting inward; via-points (multi2) and seasonwood (W) inherit; then shift
    # so each segment is attributed to the ring at its inner end.
    years = [None] * n
    y = od
    for i in range(n):
        if pts[i]["type"] in ("reg", "multi1"):
            years[i] = y
            y -= 1
    for i in range(n):                        # forward-fill inherited points
        if years[i] is None and i > 0:
            years[i] = years[i - 1]
    shifted = [None] + years[:-1]             # segment -> inner-endpoint ring

    # whole-ring widths, and seasonwood split where W points occur
    ring = {}
    ew = {}
    lw = {}
    for i in range(1, n):
        yr = shifted[i]
        d = seg[i]
        # a multipoint's inner pair (multi2) is a bent-path via-point, not growth;
        # modendro excludes its incoming distance from the ring widths.
        if yr is None or not np.isfinite(d) or pts[i]["type"] == "multi2":
            continue
        ring[yr] = ring.get(yr, 0.0) + d
        if pts[i]["type"] == "W":             # outer part so far = latewood
            lw[yr] = lw.get(yr, 0.0) + d
        elif pts[i - 1]["type"] == "W":       # segment leaving a W = earlywood
            ew[yr] = ew.get(yr, 0.0) + d

    # the innermost ring (its inner boundary is the last measured point) has no
    # inner neighbour among the boundaries, so drop the partial last-year bucket
    # only if it never received a closing segment -- handled naturally above.

    years_sorted = sorted(ring)
    rw = pd.Series([ring[k] for k in years_sorted], index=years_sorted, dtype=float)

    # recomputed distance to pith: innermost boundary -> pith coordinate
    d2pith_calc = None
    if meta["pith"] is not None:
        inner = boundaries[-1]
        d2pith_calc = math.hypot(inner["x"] - meta["pith"][0],
                                 inner["y"] - meta["pith"][1])

    radius_calc = float(rw.sum()) + (d2pith_calc or 0.0)

    # Point-order QC is left to the user: the raw coordinates are returned so they
    # can be plotted (the reliable check). A simple automatic monotonicity test was
    # tried and dropped -- it flagged a benign curved core while missing a genuine
    # point swap, so it was worse than nothing. Robust auto-detection is deferred.
    coords = pd.DataFrame(pts)

    attrs = {
        "distance_to_pith": d2pith_calc,
        "distance_to_pith_file": meta["d2pith_file"],
        "years_to_pith_file": meta["yrs_to_pith_file"],
        "radius_file": meta["radius_file"],
        "radius_calc": radius_calc,
        "outer_date": None if relative else int(od),
        "inner_year": int(min(years_sorted)) if not relative else int(min(years_sorted)),
        "n_rings": int(len(rw)),
        "relative_dates": relative,
        "comment": meta["comment"],
        "coo_version": None if ver is None else ver / 100.0,
    }
    return {"rw": rw,
            "ew": pd.Series(ew, dtype=float).sort_index() if ew else None,
            "lw": pd.Series(lw, dtype=float).sort_index() if lw else None,
            "attrs": attrs,
            "coords": coords}


class PosCollection:
    """Result of read_pos(): whole-ring widths plus the pieces a .pos file carries
    that a Tucson .rwl drops. Mirrors TridasProject's role for TRiDaS."""

    def __init__(self, rwl, earlywood, latewood, attributes, coordinates, not_read):
        self._rwl = rwl
        self._ew = earlywood
        self._lw = latewood
        self.attributes = attributes
        self.coordinates = coordinates
        self.not_read = not_read

    def rwl(self):
        return self._rwl

    def earlywood(self):
        return self._ew

    def latewood(self):
        return self._lw

    def __repr__(self):
        n = 0 if self._rwl is None else self._rwl.shape[1]
        return ("PosCollection(%d series, %d not read)"
                % (n, 0 if self.not_read is None else len(self.not_read)))


def _gather_pos_files(path):
    if os.path.isdir(path):
        found = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(".pos"):
                    found.append(os.path.join(root, f))
        return sorted(found)
    return [path]


def read_pos(path, default_od=None):
    """Read CooRecorder .pos file(s) into a PosCollection.

    path may be a single .pos file or a directory tree (every .pos under it is
    read and stacked into one ring-width frame). Series are named by file stem.
    """
    path = os.fspath(path)
    files = _gather_pos_files(path)
    if not files:
        raise ValueError("no .pos files found at: %s" % path)

    rw_cols, ew_cols, lw_cols = {}, {}, {}
    attr_rows, coord_frames, not_read = [], [], []
    seen = {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        name = stem
        if name in seen:                         # disambiguate duplicate stems
            seen[name] += 1
            name = "%s_%d" % (stem, seen[stem])
            _warn("read_pos: duplicate series name '%s'; renamed to '%s'."
                  % (stem, name))
        else:
            seen[name] = 1
        try:
            res = _parse_pos_file(f, default_od=default_od)
        except Exception as exc:                 # never abort the whole run
            not_read.append({"file": f, "reason": str(exc)})
            continue
        rw_cols[name] = res["rw"]
        if res["ew"] is not None:
            ew_cols[name] = res["ew"]
        if res["lw"] is not None:
            lw_cols[name] = res["lw"]
        row = {"series": name}
        row.update(res["attrs"])
        attr_rows.append(row)
        c = res["coords"].copy()
        c.insert(0, "series", name)
        coord_frames.append(c)

    def _frame(cols):
        if not cols:
            return None
        df = pd.DataFrame(cols).sort_index()
        df.index.name = "Year"
        return df

    rwl = _frame(rw_cols)
    if rwl is not None:
        rwl.attrs["kind"] = "rwl"
    attributes = pd.DataFrame(attr_rows) if attr_rows else None
    coordinates = pd.concat(coord_frames, ignore_index=True) if coord_frames else None
    not_read_df = pd.DataFrame(not_read) if not_read else None
    return PosCollection(rwl, _frame(ew_cols), _frame(lw_cols),
                         attributes, coordinates, not_read_df)


def read_pos_single(path, default_od=None):
    """Return just the whole-ring rwl DataFrame for dpl.readers("x.pos").

    Accepts a single file or a directory of .pos files. The per-series CooRecorder
    attributes (distance to pith, radius, comment, ...) ride along on
    ``.attrs["dplpy_pos_attributes"]``; use dpl.read_pos() for the full
    PosCollection (earlywood/latewood, coordinates, not_read)."""
    pc = read_pos(path, default_od=default_od)
    rwl = pc.rwl()
    if rwl is None:
        reasons = ""
        if pc.not_read is not None and len(pc.not_read):
            reasons = " (" + "; ".join(pc.not_read["reason"].astype(str)) + ")"
        raise ValueError("no readable .pos series found at: %s%s" % (path, reasons))
    rwl.attrs["dplpy_pos_attributes"] = pc.attributes
    return rwl
