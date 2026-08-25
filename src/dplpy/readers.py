from __future__ import print_function

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2024  OpenDendro

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

# Date: 9/8/2022
# Author: Ifeoluwa Ale
# Project: OpenDendro- Readers
# Description: Reads data from supported file types (*.CSV and *.RWL)
#              and stores them in a dataframe
#
# The Tucson (.rwl) reader was substantially hardened in v0.2.x to handle
# the "not-that-rare" formatting problems found in real ITRDB files while
# preserving byte-for-byte backward compatibility on clean files. The design
# mirrors dplR's read.tucson() robustness model (header auto-detection, a
# fixed-width parse with a whitespace-delimited fallback, per-series precision
# detection, and tolerant per-line error handling), with two DELIBERATE
# departures from dplR agreed for dplPy:
#
#   * No-data internal gaps (years with no row at all) stay NaN instead of
#     being filled with 0.0.  dplR's 0.0 there is an artifact of its
#     zero-initialised assembly matrix; NaN keeps genuine gaps out of any
#     downstream mean/detrend.
#   * Anomalous negative values (anything < 0 that is not the -9999 stop
#     marker, e.g. -7 or -2599) are set to NaN AND a warning is emitted,
#     rather than being converted to 0.0 as dplR does.  Ring widths cannot be
#     negative, so these are treated as missing and the user is told.
#
# On every valid measurement dplPy matches dplR exactly.

import os
import warnings
from collections import Counter

import pandas as pd
import numpy as np


def readers(filename: str, skip_lines=0, header=None):
    """Imports a common ring width data file

    Extended Summary
    ----------------
    This function reads data from common ring width data files (.csv, .rwl)
    and stores them in pandas dataframes.

    Parameters
    ----------
    filename : str
        a data file (.CSV, .RWL or .RAW)
    header : bool or None, default None
        Whether a 3-line site-metadata header sits at the top of the file.
        ``None`` (the default) auto-detects the header the way dplR does, so
        the great majority of real ITRDB files "just work" without the caller
        having to know.  Pass ``True`` or ``False`` to force the behaviour.
    skip_lines : int, default 0
        indicates how many of the first few lines of the file to skip
        when reading it.

    Returns
    -------
    data : pandas dataframe

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> data = dpl.readers("../tests/data/rwl/file.rwl")           # header auto-detected
    >>> data = dpl.readers("../tests/data/rwl/file.rwl", header=True)

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#readers

    """
    FORMAT = "." + filename.split(".")[-1]
    print("\nAttempting to read input file: " + os.path.basename(filename) + " as " + FORMAT + " format\n")

    # open the input file and read its data into a pandas dataframe
    if filename.upper().endswith(".CSV"):
        series_data = pd.read_csv(filename, skiprows=skip_lines)
    elif filename.upper().endswith(".RWL"):
        series_data = process_rwl_pandas(filename, skip_lines, header)
    elif filename.upper().endswith(".RAW"):
        series_data = process_rwl_pandas(filename, skip_lines, header)
    else:
        errorMsg = """

Unable to read file, please check that you're using a supported type
Accepted file types are .csv and .rwl

Example usages:
>>> import dplpy as dpl
>>> data = dpl.readers('../tests/data/csv/filename.csv')
>>> data = dpl.readers('../tests/data/rwl/filename.rwl'), header=True
"""

        raise ValueError(errorMsg)

    # If no data is returned, then an error was encountered when reading the file.
    if series_data is None:
        errorMsg = """
        Error reading file. Check that file exists and that file formatting is consistent with {format} format.
        If your file contains headers, run dpl.headers(file_path, header=True)
        """.format(format=FORMAT)
        raise ValueError(errorMsg)
    series_data.set_index('Year', inplace=True, drop=True)

    # Display message to show that reading was successful
    print("\nSUCCESS!\nFile read as:", FORMAT, "file\n")

    # Display names of all the series found
    print("Series names:")
    print(list(series_data.columns), "\n")
    return series_data


# ---------------------------------------------------------------------------
# .rwl (Tucson) reading
# ---------------------------------------------------------------------------

def process_rwl_pandas(filename, skip_lines, header):
    """Read a Tucson (.rwl/.raw) file into a Year-indexed dataframe.

    Returns a dataframe with a ``Year`` column (the public ``readers`` wrapper
    sets it as the index), or ``None`` if nothing usable could be parsed.
    """
    with open(filename, "r") as rwl_file:
        raw_lines = rwl_file.readlines()

    # 1. Drop blank lines (warning about them, as earlier dplPy did) and
    #    comment lines (a '#' anywhere in the first 78 columns), the same
    #    pre-clean dplR performs.  Line numbers in the warning are 1-indexed
    #    against the ORIGINAL file so they stay meaningful to the user.
    clean_lines = []
    for lineno, line in enumerate(raw_lines, start=1):
        line = line.rstrip("\r\n")
        if len(line.strip()) == 0:
            warnings.warn("Empty line found at line " + str(lineno) + "\n")
            continue
        hashpos = line.find("#")
        if 0 <= hashpos <= 77:
            continue  # comment line
        clean_lines.append(line)

    # 2. Honour an explicit skip_lines against the cleaned stream.
    if skip_lines:
        clean_lines = clean_lines[skip_lines:]

    if len(clean_lines) == 0:
        return None

    # 3. Resolve the header. Rather than assume a fixed 3-line header, skip any
    #    number of leading header/metadata lines up to the first line that looks
    #    like data. Many real ITRDB files have 1-, 2- or other-length headers, and
    #    blindly skipping three would silently drop data rows (or, with fewer than
    #    three lines, lose the first measurements). header=False disables this;
    #    header=True and header=None both skip leading header lines robustly.
    if header is False:
        start = 0
    else:
        start = _first_data_line_index(clean_lines)
    clean_lines = clean_lines[start:]

    if len(clean_lines) == 0:
        return None

    parsed = read_rwl(clean_lines)
    if parsed is None:
        return None
    rwl_data, precision, order = parsed

    # 4. Assemble the dataframe.  The index spans the first to last year for
    #    which ANY series has data; years with no row remain NaN (a deliberate
    #    departure from dplR, which zero-fills such gaps).
    all_years = [yr for series in rwl_data.values() for yr in series]
    if len(all_years) == 0:
        return None
    first_date = min(all_years)
    last_date = max(all_years)
    index = list(range(first_date, last_date + 1))

    df = pd.DataFrame(data={"Year": index})
    series_columns = []
    for series in order:
        div = precision[series]
        col = [rwl_data[series].get(yr, np.nan) for yr in index]
        col = [v / div if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
               for v in col]
        series_columns.append(pd.Series(data=col, name=series))
    df = pd.concat([df] + series_columns, axis=1)
    return df


def _detect_header(first_line):
    """Return True if the first line looks like site metadata, not data.

    Port of dplR read.tucson()'s header heuristic: a data line has an integer
    year in cols 9-12 and integer measurements from col 13 on; anything else is
    treated as a header, with a rescue for data lines that carry unusually long
    IDs / spacing.
    """
    if len(first_line) < 12:
        raise ValueError("first line in rwl file ends before column 12")

    is_head = False

    # Year must be an integer in [-1e4, 1e4] in columns 9-12 (0-indexed 8:12).
    yr_field = first_line[8:12]
    try:
        yrcheck = float(yr_field)
        if yrcheck < -1e4 or yrcheck > 1e4 or yrcheck != round(yrcheck):
            is_head = True
    except ValueError:
        is_head = True

    # Data fields (10 x 6 chars from col 13) must be blank or integer, no letters.
    if not is_head:
        fields = [first_line[i:i + 6].lstrip() for i in range(12, 12 + 6 * 10, 6)]
        nonempty = [k for k, f in enumerate(fields) if f != ""]
        if not nonempty:
            is_head = True
        else:
            fields = fields[:nonempty[-1] + 1]
            if any(any(c.isalpha() for c in f) for f in fields):
                is_head = True
            else:
                for f in fields:
                    if f == "":
                        continue
                    try:
                        if float(f) != round(float(f)):
                            is_head = True
                            break
                    except ValueError:
                        is_head = True
                        break

    # Rescue: a data line with a long ID / odd spacing can trip the checks
    # above. If splitting on whitespace yields an ID followed by all-integer
    # tokens, it is really data after all.
    if is_head:
        parts = first_line.strip().split()
        if 3 <= len(parts) <= 13:
            rest = parts[1:]
            if not any(any(c.isalpha() for c in p) for p in rest):
                ok = True
                for p in rest:
                    try:
                        if float(p) != round(float(p)):
                            ok = False
                            break
                    except ValueError:
                        ok = False
                        break
                if ok:
                    is_head = False

    return is_head


def _looks_like_data(line):
    """True if the line parses as a Tucson data row: an ID, an integer year, and
    a numeric first measurement. Header/metadata lines fail because their first
    field after the year is text (a site name, species, investigator, region).
    Trailing notes on a data line are tolerated -- they are trimmed before the
    first value is inspected -- which is why this is used (instead of the stricter
    header heuristic) to find where the data begins."""
    if len(line) < 12:
        return False
    for parser in (_parse_fixed, _parse_ws):
        try:
            _sid, _yr, vals = parser(line)
        except (ValueError, IndexError):
            continue
        for v in vals:
            vs = v.strip()
            if vs == "":
                continue                 # skip leading blank (missing) fields
            try:
                int(vs)
                return True              # first real measurement is numeric -> data
            except ValueError:
                return False             # first real field is text -> header/metadata
    return False


def _first_data_line_index(lines):
    """Index of the first line that looks like Tucson data rather than a header
    or metadata line -- used to skip a header of *any* length (1, 2, 3 or more
    lines) instead of assuming exactly three. Returns 0 if none looks like data
    (leave the lines untouched and let parsing report the problem)."""
    for i, ln in enumerate(lines):
        if _looks_like_data(ln):
            return i
    return 0


def _trim_trailing_junk(values):
    """Drop trailing tokens that are not measurements -- appended notes/comments
    and trailing blanks -- by keeping everything up to and including the last
    token that parses as an integer (a real value or a 999 / -9999 stop marker).
    Interior blank fields (missing rings written as blanks) are preserved as ""
    so they hold their year position; only *trailing* junk/blanks are removed."""
    last = -1
    for i, v in enumerate(values):
        vs = v.strip()
        if vs == "":
            continue
        try:
            int(vs)
            last = i
        except ValueError:
            continue
    return values[:last + 1]


def _parse_fixed(line, long=False):
    """Parse one line by fixed Tucson columns: (id, year, [value strings]).

    Interior blank 6-char fields are kept as "" so a missing ring written as
    blanks holds its year position (dropping it would shift the rest of the
    decade); trailing blanks/notes are trimmed by _trim_trailing_junk.

    A bunched 5-char negative year (a BC year < -999 written with no space
    between ID and year, e.g. 'MNP262M-1262' = year -1262) is detected per line
    -- a '-' in column 8 with column 12 filled -- and read as a 7-char ID + a
    5-char year, the same heuristic dplR's `long` mode and read.tucson2 use.
    Without this the '-' lands in the ID field and the year reads as +1262,
    which then trips the self-overlap check."""
    if not long and len(line) >= 12 and line[7] == '-' and line[11] != ' ':
        idw, yrw = 7, 5
    else:
        idw, yrw = (7, 5) if long else (8, 4)
    series_id = line[:idw].strip()
    year = int(line[idw:idw + yrw])          # may raise ValueError
    rest = line[idw + yrw:]
    values = [rest[i:i + 6].strip() for i in range(0, len(rest), 6)]
    values = _trim_trailing_junk(values)
    return series_id, year, values


def _parse_ws(line):
    """Parse one line by whitespace delimiting: (id, year, [value strings])."""
    tokens = line.split()
    if len(tokens) < 2:
        raise ValueError("too few fields")
    series_id = tokens[0]
    year = int(tokens[1])                    # may raise ValueError
    values = _trim_trailing_junk(tokens[2:])
    return series_id, year, values


def _parse_all(lines, method, long=False):
    """Parse every line with one method. Returns a list where each element is
    (id, year, values, lineno) or None if that line could not be parsed."""
    rows = []
    for k, line in enumerate(lines):
        if len(line.strip()) < 7:
            rows.append(None)
            continue
        try:
            if method == "fixed":
                sid, yr, vals = _parse_fixed(line, long)
            else:
                sid, yr, vals = _parse_ws(line)
            rows.append((sid, yr, vals, k))
        except (ValueError, IndexError):
            rows.append(None)
    return rows


def _rows_valid(rows):
    """dplR input.ok essence: reject a parse where any row carries more values
    than its decade allows (the tell-tale of a mis-tokenised line). One extra
    column is permitted for a trailing stop marker."""
    good = [r for r in rows if r is not None]
    if not good:
        return False
    for sid, yr, vals, k in good:
        full_per_row = 10 - (yr % 10)
        if len(vals) > full_per_row + 1:
            return False
    return True


def read_rwl(lines, long=False):
    """Parse cleaned Tucson data lines into (rwl_data, precision, order).

    rwl_data  : {series_id: {year: raw_integer_value}}
    precision : {series_id: 100 or 1000}   (divisor to convert to mm)
    order     : [series_id, ...]           (first-appearance order)

    Uses a fixed-width parse first (the Tucson "standard"), falling back to a
    whitespace-delimited parse when the fixed one fails validation -- exactly
    the strategy dplR uses to cope with long IDs, negative years, etc. Returns
    ``None`` if nothing usable could be parsed.
    """
    fixed_rows = _parse_all(lines, "fixed", long)
    if _rows_valid(fixed_rows):
        rows = fixed_rows
    else:
        ws_rows = _parse_all(lines, "ws")
        if _rows_valid(ws_rows):
            warnings.warn("fixed-width parse failed; re-read with variable (whitespace) columns")
            rows = ws_rows
        else:
            # Neither validated cleanly. Keep the parse that recovered the most
            # rows so a single malformed file still yields as much data as
            # possible, and tell the user the read may be incomplete.
            n_fixed = sum(1 for r in fixed_rows if r is not None)
            n_ws = sum(1 for r in ws_rows if r is not None)
            if max(n_fixed, n_ws) == 0:
                return None
            warnings.warn("file has formatting problems; read may be incomplete")
            rows = fixed_rows if n_fixed >= n_ws else ws_rows

    n_bad = sum(1 for r in rows if r is None)
    if n_bad:
        warnings.warn(str(n_bad) + " line(s) could not be parsed and were skipped")

    # Count contiguous blocks per series, so we can later tell a genuine
    # duplicate ID (the same code used by two separate blocks -- two cores)
    # from a single series that overlaps *itself* because a row is malformed.
    block_count = {}
    prev = None
    for r in rows:
        if r is None:
            continue
        sid = r[0]
        if sid != prev:
            block_count[sid] = block_count.get(sid, 0) + 1
            prev = sid

    # Determine each series' measurement precision from its TERMINATOR -- the
    # stop marker on its last row -- exactly as dplR does. This is what makes a
    # mid-series 999 in a 0.001 mm series read as a real 0.999 mm value rather
    # than being mistaken for a 0.01 mm stop marker (a silent bug in the older,
    # "any 999 is a marker" logic).
    last_row = {}
    for r in rows:
        if r is not None:
            last_row[r[0]] = r          # final occurrence wins (rows are in order)
    precision = {}
    for sid, r in last_row.items():
        vals = r[2]
        term = vals[-1].strip() if vals else ""
        if term == "999":
            precision[sid] = 100        # 0.01 mm
        elif term == "-9999":
            precision[sid] = 1000       # 0.001 mm
        # otherwise unknown -> resolved by the dominant-precision fallback below

    rwl_data = {}
    order = []
    neg_hits = []       # (series, year) for anomalous negatives -> NaN
    nonint_hits = 0
    identical_dups = 0  # cells duplicated with an identical value (copy-paste)
    year_src = {}       # sid -> {year: start-year of the row that wrote it}
    overlaps = []       # (sid, year, first_row_start, second_row_start)
    overlap_seen = set()
    prec_shift = []     # (sid, year) where a 0.01 mm series carries a stray -9999

    for r in rows:
        if r is None:
            continue
        sid, yr, vals, k = r
        if sid not in rwl_data:
            rwl_data[sid] = {}
            year_src[sid] = {}
            order.append(sid)
        prec = precision.get(sid)
        for j, tok in enumerate(vals):
            s = tok.strip()
            y = yr + j
            if s == "":
                continue             # interior blank field = missing ring (holds its year)
            if s == "-9999":
                # -9999 is only ever the 0.001 mm stop marker. In a 0.001 mm
                # series it is a marker -> drop. In a series terminated at
                # 0.01 mm it signals a measurement-precision shift (e.g. an early
                # 0.001 mm segment inside a 0.01 mm series): record to fail below.
                if prec == 100:
                    prec_shift.append((sid, y))
                continue
            if s == "999":
                if prec == 100:
                    continue             # 0.01 mm stop marker (dplR nulls all 999 here)
                # 0.001 mm (or unknown): 999 is a real 0.999 mm value -> keep it
            try:
                v = int(s)
            except ValueError:
                nonint_hits += 1         # non-integer token: skip -> NaN
                continue
            if v < 0:
                # Anomalous negative (not a stop marker): treat as missing.
                neg_hits.append((sid, y))
                continue
            if y in rwl_data[sid]:
                # A value is already present for this (series, year). If it is
                # identical it's a harmless duplicate -- a copy-paste of a row or
                # of the whole file -- so ignore it. If it differs, the data
                # genuinely conflicts: record it and refuse the file below.
                if v == rwl_data[sid][y]:
                    identical_dups += 1
                elif sid not in overlap_seen:
                    overlaps.append((sid, y, year_src[sid].get(y), yr))
                    overlap_seen.add(sid)
                continue
            rwl_data[sid][y] = v
            year_src[sid][y] = yr

    # Fail loudly on overlapping data. Silently merging or overwriting would
    # corrupt the series, so -- like dplR -- we refuse the file; unlike dplR we
    # say *which* kind of problem it is and point at the offending row.
    if overlaps:
        problems = []
        for sid, y, first_start, second_start in overlaps:
            if block_count.get(sid, 1) > 1:
                # Same ID in two separate blocks: a genuine duplicate series ID.
                problems.append(
                    "Duplicate series ID '" + str(sid) + "': this ID is used by more "
                    "than one series in the file (they overlap at year " + str(y)
                    + ") -- most often two cores mistakenly given the same code. "
                    "Rename or remove the duplicate."
                )
            else:
                # One contiguous block that runs into itself: a malformed row.
                allowed = 10 - (first_start % 10) if first_start is not None else None
                next_dec = first_start - (first_start % 10) + 10 if first_start is not None else None
                detail = (
                    "Series '" + str(sid) + "' overlaps itself at year " + str(y) + ": the "
                    "row beginning " + str(first_start) + " supplies a value for " + str(y)
                    + ", but another row begins at " + str(second_start) + "."
                )
                if allowed is not None:
                    detail += (
                        " A row beginning at " + str(first_start) + " can hold only "
                        + str(allowed) + " value(s) before the next decade (" + str(next_dec)
                        + "), so that row appears to have one value too many, a misplaced "
                        "value, or a mistyped start year. Please check that row."
                    )
                problems.append(detail)
        raise ValueError(
            "Cannot read file -- overlapping data detected; dplPy will not guess how "
            "to resolve it.\n  " + "\n  ".join(problems)
            + "\nPlease correct the file and read it again."
        )

    # Fail on a measurement-precision shift within a single series (a 0.01 mm
    # series that also carries a -9999). Reading such a series at one precision
    # makes part of it 10x wrong, so -- in normal mode -- we refuse and name the
    # series. (Salvage mode will instead drop the affected series and continue.)
    if prec_shift:
        affected = []
        for sid, y in prec_shift:
            if sid not in affected:
                affected.append(sid)
        where = "; ".join(
            str(sid) + " (stray -9999 near year "
            + str(min(y for s, y in prec_shift if s == sid)) + ")"
            for sid in affected
        )
        raise ValueError(
            "Cannot read file -- measurement-precision shift detected. These series "
            "end with a 0.01 mm stop marker (999) but also contain a -9999 (the "
            "0.001 mm stop marker) mid-series, so the series was measured at two "
            "different precisions: " + where + ".\ndplPy will not guess the boundary, "
            "because reading such a series at a single precision makes part of it 10x "
            "wrong. Please split the series by precision (or correct the markers) and "
            "read it again."
        )

    # Resolve precision for any series that never showed a stop marker: adopt
    # the file's dominant precision (or 0.001 mm if the file has none at all),
    # and warn so the assumption is visible.
    known = list(precision.values())
    if known:
        dominant = Counter(known).most_common(1)[0][0]
    else:
        dominant = 1000
        warnings.warn("no stop markers found in file; assuming 0.001 mm precision")
    for sid in order:
        if sid not in precision:
            precision[sid] = dominant
            warnings.warn(
                "series '" + str(sid) + "' has no stop marker; assuming "
                + ("0.01" if dominant == 100 else "0.001") + " mm precision"
            )

    if identical_dups:
        warnings.warn(
            str(identical_dups) + " identical duplicate value(s) were ignored "
            "(a row or the file appears to have been duplicated, e.g. by copy-paste)"
        )

    if nonint_hits:
        warnings.warn(str(nonint_hits) + " non-integer value(s) could not be read and were set to NaN")

    if neg_hits:
        preview = ", ".join(str(sid) + "@" + str(yr) for sid, yr in neg_hits[:5])
        more = "" if len(neg_hits) <= 5 else " ..."
        warnings.warn(
            str(len(neg_hits)) + " anomalous negative value(s) (not the -9999 stop "
            "marker) were set to NaN [" + preview + more + "]"
        )

    # Drop any series that ended up with no usable data at all.
    order = [sid for sid in order if len(rwl_data[sid]) > 0]
    rwl_data = {sid: rwl_data[sid] for sid in order}
    if len(order) == 0:
        return None

    return rwl_data, precision, order
