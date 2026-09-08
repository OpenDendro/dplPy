__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2026  OpenDendro

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
__license__ = "GNU GPLv3"

#!/usr/bin/python
# -*- coding: utf-8 -*-

# Title: read_crn.py
# Project: OpenDendro dplPy
# Description: Read Tucson chronology (.crn) files -- the standardized site
#              chronologies produced by ARSTAN and friends. Handles the standard
#              single-chronology ITRDB file (as dplR's read.crn does) AND the
#              "combined" multi-block files that dplR cannot: an ARSTAN run's
#              stacked std/res/ars(/trn) chronologies for one site, and files
#              that concatenate many sites' chronologies (e.g. SSF output), each
#              block tagged with a trailing chronology-type label.
#
# Format (ITRDB, see treeinfo.pdf):
#   * Optional header records (1 or 3 lines) precede each block; they are skipped
#     automatically (a line is data only if cols 7-10 hold a plausible year).
#   * A data row is: site ID (cols 1-6), decade (cols 7-10), then ten
#     (index I4, sample-count I3) pairs (cols 11-80), optionally a trailing TRL
#     ID or, in combined files, a chronology-type label (trn/std/res/ars/...).
#   * Index values are x1000 (1000 == 1.000 mean growth); 9990 is missing.
#   * An optional statistics line may close a block (nYears, AC1, StdDev, ...);
#     it is recognized by a decimal point in the value region and skipped
#     (captured on df.attrs["dplpy_crn_stats"]).
#
# Returns one DataFrame indexed by year (or, with split_by_site=True, a dict of
# them keyed by site). Each chronology contributes a value column (named by its
# type for a single-site file -- 'std','res','ars',... -- or by site for a
# multi-site file) plus a sample-depth column ('samp_depth', shared when several
# types of one site carry identical depth). See dev/dplR_gap_analysis_2026-08.md.
#
# example usage:
# >>> import dplpy as dpl
# >>> crn = dpl.read_crn("ga004.crn")                 # -> columns ['std','samp_depth']
# >>> crn = dpl.read_crn("site.rwl_crns.txt")         # -> ['trn','std','res','ars','samp_depth']
# >>> by_site = dpl.read_crn("ALL.CRNS", split_by_site=True)   # -> {site: DataFrame}

import os
import re
import warnings
from datetime import date

import numpy as np
import pandas as pd

_MISSING = 9990          # ITRDB missing-value code (before /1000 scaling)


def read_crn(filename, strict=True, split_by_site=False):
    """Read a Tucson (.crn) chronology file into a DataFrame.

    Parameters
    ----------
    filename : str
        Path (or http/https URL) to a .crn chronology file. Standard single
        chronologies and combined multi-block files (stacked ARSTAN types, or
        many sites concatenated) are both handled.
    strict : bool, default True
        ``True`` (the default) refuses a file with no readable chronology;
        ``False`` returns None instead. (Individual malformed rows are skipped
        with a warning either way.)
    split_by_site : bool, default False
        If True, return a dict mapping each site ID to its own DataFrame (each
        framed as a single-site file: value columns named by chronology type
        plus a shared ``samp_depth``). Useful for multi-site bundles, where the
        default single wide frame is unwieldy. A single-site file yields a
        one-entry dict.

    Returns
    -------
    pandas.DataFrame or dict of pandas.DataFrame
        By default one DataFrame indexed by year: one value column per
        chronology (named by chronology type for a single-site file, e.g.
        'std'/'res'/'ars'; by site, or 'site.type', for a multi-site file),
        index values divided by 1000 and 9990 mapped to NaN, with sample depths
        as 'samp_depth' (shared when several chronologies carry identical depth)
        or '<name> samp_depth' otherwise. With ``split_by_site=True``, a dict
        {site_id: DataFrame}. ``df.attrs`` carries ``dplpy_crn`` (a summary) and
        ``dplpy_crn_stats`` (embedded statistics lines).

    Examples
    --------
    >>> dpl.read_crn("ga004.crn")
    >>> dpl.read_crn("celaque_prelim_v01.rwl_crns.txt")
    >>> by_site = dpl.read_crn("ALL.CRNS", split_by_site=True)   # {site: DataFrame}

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#read_crn
    """
    if not isinstance(strict, bool):
        raise ValueError("strict must be True (strict) or False (salvage), got "
                         + repr(strict))

    lines = _read_lines(filename)
    decade_pos = _detect_decade_pos(lines)
    blocks, stats = _parse_blocks(lines, decade_pos)

    # No usable chronology: no data rows at all, or rows that parsed a decade but no
    # actual values (e.g. a non-fixed-width file). Fail cleanly, honouring strict,
    # instead of crashing downstream when the year set is empty.
    if not blocks or not any(b["values"] for b in blocks):
        if not strict:
            warnings.warn("No chronology data found in "
                          + os.path.basename(filename) + "; returning None.")
            return None
        raise ValueError("No chronology data found in " + os.path.basename(filename)
                         + " -- is this a Tucson .crn file?")

    if decade_pos != 7:
        warnings.warn("Using columns " + str(decade_pos) + "-10 for the decade field "
                      "(the site ID is shorter than 6 characters, or a BC year "
                      "borrows column 6 for its sign).")

    sites = list(dict.fromkeys(b["site"] for b in blocks))

    if split_by_site:
        result = {}
        for site in sites:
            sub = [b for b in blocks if b["site"] == site]
            d = _assemble(sub)
            if d is None:
                continue                               # this site had no usable values
            d.attrs["dplpy_crn"] = {
                "sites": [site],
                "types": list(dict.fromkeys(b["type"] for b in sub)),
                "n_chronologies": len(sub),
            }
            d.attrs["dplpy_crn_stats"] = [s for s in stats if s[:6].strip() == site]
            result[site] = d
        print(os.path.basename(filename) + " successfully read as crn file: "
              + str(len(blocks)) + (" chronology" if len(blocks) == 1 else " chronologies")
              + " across " + str(len(sites)) + (" site" if len(sites) == 1 else " sites")
              + " (split_by_site)")
        return result

    df = _assemble(blocks)
    df.attrs["dplpy_crn"] = {
        "sites": sites,
        "types": list(dict.fromkeys(b["type"] for b in blocks)),
        "n_chronologies": len(blocks),
    }
    df.attrs["dplpy_crn_stats"] = stats

    # one-line success summary, in the style of dpl.readers()
    info = df.attrs["dplpy_crn"]
    n = info["n_chronologies"]
    print(os.path.basename(filename) + " successfully read as crn file with "
          + str(n) + (" chronology" if n == 1 else " chronologies")
          + " [" + ", ".join(str(t) for t in info["types"]) + "]"
          + (" from " + str(len(info["sites"])) + " sites" if len(info["sites"]) > 1 else "")
          + " covering the period from " + str(int(df.index.min()))
          + " to " + str(int(df.index.max())))
    return df


# --- line acquisition -------------------------------------------------------

def _read_lines(filename):
    if str(filename).lower().startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(filename) as resp:
            text = resp.read().decode("latin-1", errors="replace")
        return text.splitlines()
    with open(filename, encoding="latin-1") as fh:
        return [ln.rstrip("\n").rstrip("\r") for ln in fh]


# --- parsing ----------------------------------------------------------------

def _decade_field(ln, decade_pos=7):
    """Return (id_width, decade_year) for a data line, or (None, None).

    The site ID occupies columns 1..decade_pos-1 and the decade columns
    decade_pos..10. ``decade_pos`` is 7 for a standard file (ID in cols 1-6, decade
    in cols 7-10); ``_detect_decade_pos`` shifts it left for a BC chronology that
    borrows column 6 for the year's minus sign (e.g. 'MWK51-6000'). The value pairs
    always begin at column 11, so only the id/decade split within cols 1-10 moves.
    A line whose decade columns are not a plausible year (a header, a blank, or a
    statistics line's site label) is not a data line.
    """
    id_width = decade_pos - 1
    if len(ln) < 10 or not ln[:id_width].strip():
        return None, None
    seg = ln[id_width:10]
    if not seg.strip():
        return None, None
    try:
        yr = int(seg)
    except ValueError:
        return None, None
    if -12000 <= yr <= 12000:
        return id_width, yr
    return None, None


def _type_label(ln):
    """Trailing alphabetic chronology-type label (trn/std/res/ars/ssfcrn/...), or None."""
    toks = ln.rstrip().split()
    if toks and re.fullmatch(r"[A-Za-z]+", toks[-1]):
        return toks[-1].lower()
    return None


def _to_int(seg):
    seg = seg.strip()
    if not seg:
        return None
    try:
        return int(seg)
    except ValueError:
        return None


def _detect_decade_pos(lines):
    """Find the 1-based column where the decade field starts, following dplR's
    read.crn adaptive logic.

    Normally 7 (site ID in cols 1-6, decade in cols 7-10). A BC chronology carries
    the year's minus sign in column 6, pushing the decade field left (the ID is then
    <= 5 chars) -- e.g. 'MWK51-6000'. dplR detects this when the first data line's
    cols 7-10 read as a *future* year and a trailing negative number is present in
    cols 2-10; the decade field then starts at that negative's column. Returns the
    decade start column (7 for a standard file)."""
    this_year = date.today().year
    for ln in lines:
        if len(ln) < 14:
            continue
        seg = ln[6:10]
        try:
            yr = int(seg)
        except ValueError:
            continue
        # confirm this is a real data line: the first value pair (cols 11-14) parses
        if _to_int(ln[10:14]) is None:
            continue
        if -12000 <= yr <= this_year:
            return 7                                   # standard layout
        # cols 7-10 read as a future year -> look for a trailing negative (a BC
        # year borrowing col 6 for its sign) in cols 2-10, exactly as dplR does.
        m = re.search(r" *-\d+$", ln[1:10])
        if m:
            return m.start() + 2                       # 1-based col of the decade field
        return 7                                       # future year, no negative: leave as-is
    return 7                                           # no data line found


def _parse_blocks(lines, decade_pos=7):
    """Group data rows into chronology blocks keyed by (site, type).

    Returns (blocks, stats) where each block is a dict with 'site', 'type',
    'values' {year: index/1000 or NaN} and 'depths' {year: sample count}, in
    first-seen order; stats is a list of raw embedded-statistics lines skipped.
    ``decade_pos`` (from ``_detect_decade_pos``) sets the id/decade column split.
    """
    blocks = {}
    order = []
    stats = []
    for ln in lines:
        id_width, decade = _decade_field(ln, decade_pos)
        if id_width is None:
            continue                                  # header / blank / non-data
        region = ln[10:]                               # the (I4,I3) pair region (always col 11+)
        # An embedded statistics line looks like a data row but carries decimals
        # in its value region (e.g. ".106  .358"). Skip it (keep for attrs).
        if "." in region[:70]:
            stats.append(ln.rstrip())
            continue
        site = ln[:id_width].strip()
        typ = _type_label(ln) or "std"
        key = (site, typ)
        if key not in blocks:
            blocks[key] = {"site": site, "type": typ, "values": {}, "depths": {}}
            order.append(key)
        blk = blocks[key]
        base = (decade // 10) * 10                     # floor to the decade start
        for k in range(10):
            v = _to_int(region[7 * k:7 * k + 4])
            d = _to_int(region[7 * k + 4:7 * k + 7])
            if v is None:
                continue
            year = base + k
            blk["values"][year] = np.nan if v == _MISSING else v / 1000.0
            blk["depths"][year] = 0 if d is None else d
    return [blocks[k] for k in order], stats


# --- assembly into one DataFrame --------------------------------------------

def _assemble(blocks):
    sites = list(dict.fromkeys(b["site"] for b in blocks))
    single_site = len(sites) == 1

    # value-column name for each block
    for b in blocks:
        same_site = [x for x in blocks if x["site"] == b["site"]]
        if single_site:
            b["vcol"] = b["type"]
        elif len(same_site) == 1:
            b["vcol"] = b["site"]
        else:
            b["vcol"] = b["site"] + "." + b["type"]

    # depth-column name: share one column across a site's blocks when their
    # sample-depth series are identical (the usual ARSTAN std/res/ars case).
    for site in sites:
        sb = [b for b in blocks if b["site"] == site]
        identical = all(b["depths"] == sb[0]["depths"] for b in sb)
        if identical:
            name = "samp_depth" if single_site else (site + " samp_depth")
            for b in sb:
                b["dcol"] = name
        else:
            for b in sb:
                b["dcol"] = b["vcol"] + " samp_depth"

    # column order: each site's value column(s), then its depth column(s)
    col_order, seen = [], set()
    for site in sites:
        for b in [x for x in blocks if x["site"] == site]:
            if b["vcol"] not in seen:
                col_order.append(b["vcol"]); seen.add(b["vcol"])
        for b in [x for x in blocks if x["site"] == site]:
            if b["dcol"] not in seen:
                col_order.append(b["dcol"]); seen.add(b["dcol"])

    value_src = {}
    depth_src = {}
    for b in blocks:
        value_src.setdefault(b["vcol"], {}).update(b["values"])
        depth_src.setdefault(b["dcol"], {}).update(b["depths"])

    all_years = set()
    for s in value_src.values():
        all_years.update(s)
    if not all_years:
        return None                                    # nothing usable to frame
    lo, hi = min(all_years), max(all_years)
    index = pd.Index(range(lo, hi + 1), name="Year")

    # Build every column at once. Assigning columns one at a time (df[c] = ...) in a
    # loop copies the whole frame each time and, for a wide multi-site bundle
    # (hundreds of columns), triggers pandas' "highly fragmented DataFrame"
    # PerformanceWarning on every insert -- a real slowdown and a flood of noise.
    data = {}
    for c in col_order:
        src = value_src[c] if c in value_src else depth_src[c]
        data[c] = [src.get(y, np.nan) for y in index]
    df = pd.DataFrame(data, index=index, columns=col_order)

    # trim leading/trailing years where every chronology value is NaN, but keep
    # the index contiguous (interior gaps stay as NaN rows).
    val_cols = [c for c in col_order if c in value_src]
    present = df.index[~df[val_cols].isna().all(axis=1)]
    if len(present):
        df = df.loc[present.min():present.max()]

    # Follow dplR: a sample-depth column whose recorded values are all 1 means the
    # sample size was not actually recorded (a chronology assembled or imported from
    # a source that lacked depths, filled with 1). Keeping it is misleading, so drop
    # it -- exactly as dplR's read.crn does ("All embedded sample depths are one").
    # A depth of 0 marks a padding/missing cell (it pairs with a 9990 -> NaN value),
    # not a real sample count, so ignore 0s (and gap-year NaNs): drop the column when
    # every *recorded* sample depth is 1.
    depth_cols = [c for c in df.columns if c in depth_src]
    dropped = []
    for c in depth_cols:
        recorded = df[c].dropna()
        recorded = recorded[recorded != 0]
        if len(recorded) and (recorded == 1).all():
            dropped.append(c)
    if dropped:
        df = df.drop(columns=dropped)
        warnings.warn("Dropped sample-depth column"
                      + ("s " if len(dropped) > 1 else " ")
                      + ", ".join(dropped)
                      + ": all recorded sample depths are 1 (the sample size was not "
                        "actually recorded).")
    return df
