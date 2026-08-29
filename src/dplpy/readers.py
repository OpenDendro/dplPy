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
import re
import warnings
import urllib.request
from collections import Counter
from datetime import date

import pandas as pd
import numpy as np


def readers(filename: str, skip_lines=0, header=None, on_error="raise", format=None):
    """Imports a common ring width data file

    Extended Summary
    ----------------
    This function reads data from common ring width data files (.csv, .rwl)
    and stores them in pandas dataframes.

    Parameters
    ----------
    filename : str
        a data file (.CSV, .RWL or .RAW). May be a local path or an http(s) URL
        (e.g. a file served from the NOAA/ITRDB archive) -- both are read the
        same way.
    header : bool or None, default None
        Whether a 3-line site-metadata header sits at the top of the file.
        ``None`` (the default) auto-detects the header the way dplR does, so
        the great majority of real ITRDB files "just work" without the caller
        having to know.  Pass ``True`` or ``False`` to force the behaviour.
    skip_lines : int, default 0
        indicates how many of the first few lines of the file to skip
        when reading it.
    format : {None, "tucson", "csv"}, default None
        Force the file format regardless of suffix. ``None`` infers it: a .csv
        suffix is read as CSV; .rwl/.raw as Tucson; and any other suffix (e.g.
        .txt, or none at all) is decided by sniffing the content, so a valid
        Tucson file with a nonstandard extension still reads. Pass "tucson" or
        "csv" to override entirely. ("rwl"/"raw" are accepted as aliases for
        "tucson".)
    on_error : {"raise", "warn"}, default "raise"
        "raise" (strict) refuses a file with an unrecoverable problem, as before.
        "warn" (salvage) recovers as much as possible instead of raising: a series
        with a self-overlap or a measurement-precision shift is dropped, a
        duplicate series ID (two overlapping cores) has its later block(s)
        renamed and kept, and a file with nothing usable returns ``None``. Every
        such action is warned about and recorded on ``df.attrs["dplpy_salvage"]``
        (a list of {series, issue, action, detail}).

    Returns
    -------
    data : pandas dataframe

    Examples
    --------
    >>> import dplpy as dpl
    >>> data = dpl.readers("../tests/data/csv/file.csv")
    >>> data = dpl.readers("../tests/data/rwl/file.rwl")           # header auto-detected
    >>> data = dpl.readers("../tests/data/rwl/file.rwl", header=True)
    >>> data = dpl.readers("https://www.ncei.noaa.gov/pub/data/paleo/treering/"
    ...                    "measurements/northamerica/usa/ak132x.rwl")  # read from a URL

    References
    ----------
    .. [1] https:/opendendro.org/dplpy-man/#readers

    """
    if on_error not in ("raise", "warn"):
        raise ValueError("on_error must be 'raise' or 'warn'")

    # `filename` may be a local path or an http(s) URL -- read either the same way.
    is_url = filename.lower().startswith(("http://", "https://"))
    FORMAT = "." + filename.split(".")[-1]

    # Resolve the format: explicit `format` wins; otherwise infer from a known
    # suffix; otherwise sniff the content so a Tucson file with a nonstandard
    # extension (e.g. .txt) still reads.
    if format is not None:
        f = str(format).strip().lower()
        if f in ("tucson", "rwl", "raw"):
            fmt = "tucson"
        elif f == "csv":
            fmt = "csv"
        else:
            raise ValueError("format must be None, 'tucson', or 'csv'")
    elif filename.upper().endswith(".CSV"):
        fmt = "csv"
    elif filename.upper().endswith((".RWL", ".RAW")):
        fmt = "tucson"
    else:
        fmt = _sniff_format(filename, is_url)
        if fmt is None:
            raise ValueError(
                "Could not determine the file format from its suffix or its content. "
                "If this is a ring-width file, pass format='tucson' (or 'csv'). "
                "Recognized suffixes are .csv, .rwl and .raw."
            )
        warnings.warn("File suffix '" + FORMAT + "' not recognized; inferred "
                      + fmt + " format from the file contents.")

    # open the input file and read its data into a pandas dataframe
    if fmt == "csv":
        series_data = pd.read_csv(filename, skiprows=skip_lines)  # pandas reads paths and URLs
    else:  # tucson
        if is_url:
            raw_lines = _fetch_url_lines(filename)
            series_data = _lines_to_dataframe(raw_lines, skip_lines, header, on_error,
                                              os.path.basename(filename))
        else:
            series_data = process_rwl_pandas(filename, skip_lines, header, on_error)

    # If no data is returned, then an error was encountered when reading the file.
    if series_data is None:
        if on_error == "warn":
            # Salvage mode: nothing usable, but don't derail a batch -- warn and
            # return None so the caller can simply skip this file.
            warnings.warn(
                "No usable data could be read from " + os.path.basename(filename)
                + "; returning None (on_error='warn')."
            )
            return None
        errorMsg = """
        Error reading file. Check that file exists and that file formatting is consistent with {format} format.
        If your file contains headers, run dpl.headers(file_path, header=True)
        """.format(format=FORMAT)
        raise ValueError(errorMsg)
    salvage_report = series_data.attrs.get("dplpy_salvage", [])
    hdr_skipped = series_data.attrs.get("dplpy_header_lines_skipped", None)
    meta = series_data.attrs.get("dplpy_metadata", None)
    dropped = series_data.attrs.get("dplpy_dropped", 0)
    series_data.set_index('Year', inplace=True, drop=True)
    series_data.attrs["dplpy_salvage"] = salvage_report            # re-attach (survives set_index)
    series_data.attrs["dplpy_dropped"] = dropped
    if hdr_skipped is not None:
        series_data.attrs["dplpy_header_lines_skipped"] = hdr_skipped
    if meta is not None:
        series_data.attrs["dplpy_metadata"] = meta

    basename = os.path.basename(filename)
    first_year = int(series_data.index.min())
    last_year = int(series_data.index.max())

    # Sanity check: a ring cannot post-date the present. A most-recent year in the
    # future flags a misdated file or a mis-parsed year (e.g. a 5-digit year from
    # a bad field split). Strict (on_error='raise') stops; salvage warns and keeps.
    this_year = date.today().year
    if last_year > this_year:
        msg = (basename + ": the most recent year is " + str(last_year)
               + ", which is in the future (the current year is " + str(this_year)
               + "). A ring cannot post-date the present -- the file may be "
               "misdated or a year was mis-parsed.")
        if on_error == "warn":
            warnings.warn(msg)
        else:
            raise ValueError(msg)

    # One-line success summary (concise for notebooks): file, format, series
    # count, and the period covered, plus a note if a header was auto-detected.
    label = {"csv": "csv", "tucson": "rwl"}.get(fmt, fmt)
    summary = (basename + " successfully extracted as " + label + " file with "
               + str(series_data.shape[1]) + " series covering the period from "
               + str(first_year) + " to " + str(last_year))
    if hdr_skipped:
        summary += (" (auto-detected " + str(hdr_skipped) + " header line"
                    + ("s" if hdr_skipped != 1 else "") + ")")
    if dropped:
        summary += (", with " + str(dropped) + " unreadable value"
                    + ("s" if dropped != 1 else "") + " set to NaN")
    print(summary)
    return series_data


# ---------------------------------------------------------------------------
# .rwl (Tucson) reading
# ---------------------------------------------------------------------------

def process_rwl_pandas(filename, skip_lines, header, on_error="raise"):
    """Read a Tucson (.rwl/.raw) file into a Year-indexed dataframe.

    Returns a dataframe with a ``Year`` column (the public ``readers`` wrapper
    sets it as the index), or ``None`` if nothing usable could be parsed. In
    salvage mode (on_error="warn") the returned frame carries a report of what
    was dropped/renamed on ``df.attrs["dplpy_salvage"]``.
    """
    with open(filename, "r") as rwl_file:
        raw_lines = rwl_file.readlines()
    return _lines_to_dataframe(raw_lines, skip_lines, header, on_error,
                               os.path.basename(filename))


# A Tucson .rwl has at most 3 header/metadata lines before the first data row (the
# ITRDB spec's site-name / region-species / investigators records; 0-2 also occur).
# More than 3 means the file is not a clean Tucson .rwl: either a nonstandard file
# (e.g. the fl0xx sites' accidentally doubled 3-line header) or an altogether
# different format (most often a NOAA Template file -- a tab-delimited table under a
# ~100-line "# ..." metadata header). Forcing those through the decadal grammar
# yields silent garbage, so strict mode raises and salvage mode warns + returns None
# when the header runs past this limit. A caller who is sure can pass header=False.
_MAX_AUTO_HEADER = 3


def _lines_to_dataframe(raw_lines, skip_lines, header, on_error, source_name):
    """Shared pipeline for the file and URL readers: clean lines, resolve the
    header, parse, and assemble the Year-column dataframe (with salvage report on
    ``df.attrs['dplpy_salvage']``). Returns None if nothing usable is present."""
    # 1. Drop blank lines (warning about them, as earlier dplPy did) and comment
    #    lines (a '#' anywhere in the first 78 columns). Line numbers in the
    #    warning are 1-indexed against the original input.
    clean_lines = []
    for lineno, line in enumerate(raw_lines, start=1):
        line = line.rstrip("\r\n")
        if len(line.strip()) == 0:
            warnings.warn("Empty line found at line " + str(lineno) + "\n")
            continue
        if _is_comment_line(line):
            continue  # a '#'-comment or a '#'-marked note row (not a '#' inside an ID)
        clean_lines.append(line)

    # 2. Honour an explicit skip_lines against the cleaned stream.
    if skip_lines:
        clean_lines = clean_lines[skip_lines:]
    if len(clean_lines) == 0:
        return None

    # 3. Resolve the header. Rather than assume a fixed 3-line header, skip any
    #    number of leading header/metadata lines up to the first line that looks
    #    like data (robust to 1-, 2- or other-length headers). header=False
    #    disables this; header=True and header=None both skip robustly.
    if header is False:
        start = 0
    else:
        start = _first_data_line_index(clean_lines)

    # Guard: refuse a file whose header is too long to be a Tucson .rwl. This
    # catches non-Tucson formats (e.g. NOAA Template files) before they are
    # force-fit into the decadal grammar and returned as garbage. header=False
    # (start forced to 0) bypasses the guard for a caller who is sure.
    if start > _MAX_AUTO_HEADER:
        msg = (source_name + " does not look like a Tucson .rwl file: " + str(start)
               + " header/metadata lines precede the first data row (a Tucson .rwl "
               "has at most 3). It may be a nonstandard file (e.g. a doubled header) "
               "or a different format -- for example a NOAA Template file (a "
               "tab-delimited table under a large '#' metadata header). If it really "
               "is Tucson data, pass header=False to force the read.")
        if on_error == "warn":
            warnings.warn(msg + " Returning None.")
            return None
        raise ValueError(msg)

    header_block = clean_lines[:start]      # the lines auto-skipped as header
    clean_lines = clean_lines[start:]
    if len(clean_lines) == 0:
        return None

    parsed = read_rwl(clean_lines, on_error=on_error)
    if parsed is None:
        return None
    rwl_data, precision, order, report, dropped = parsed

    # 4. Assemble the dataframe. The index spans the first to last year with
    #    data; years with no row stay NaN (a deliberate departure from dplR,
    #    which zero-fills such gaps).
    df = _assemble_dataframe(rwl_data, precision, order)
    if df is None:
        return None
    df.attrs["dplpy_salvage"] = report
    df.attrs["dplpy_dropped"] = dropped              # non-integer + anomalous-negative cells set to NaN
    df.attrs["dplpy_header_lines_skipped"] = start   # header lines auto-skipped
    df.attrs["dplpy_metadata"] = _extract_header_metadata(header_block)
    if on_error == "warn" and report:
        _warn_salvage_summary(source_name, report)
    return df


# ---------------------------------------------------------------------------
# Header metadata extraction (prototype). The classic ITRDB Tucson header is 3
# lines, each prefixed by the site code + a line number (1/2/3):
#   line 1: site name ............................................ SPCODE
#   line 2: region   species-name   ELEVm   LAT LONG   __   firstYr lastYr
#   line 3: investigator(s)
# Fields are best-effort: anything not confidently found is left None, and the
# raw header lines are always retained for provenance.
# ---------------------------------------------------------------------------

_METADATA_FIELDS = ("site_id", "site_name", "species_code", "species_name",
                    "country_region", "elevation_m", "latitude", "longitude",
                    "first_year", "last_year", "investigators")


_NORTH_AMERICA = frozenset({
    # US states
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
    "usa", "u.s.a.", "us", "united states",
    # Canada + provinces/territories
    "canada", "alberta", "british columbia", "manitoba", "new brunswick",
    "newfoundland", "labrador", "nova scotia", "ontario", "quebec",
    "saskatchewan", "prince edward island", "yukon", "northwest territories",
    "nunavut",
    # Mexico + Central America (all North + West)
    "mexico", "guatemala", "belize", "honduras", "el salvador", "nicaragua",
    "costa rica", "panama",
})
# South America: entirely West longitude, but latitude straddles the equator.
_SOUTH_AMERICA = frozenset({
    "colombia", "venezuela", "guyana", "suriname", "french guiana", "ecuador",
    "peru", "brazil", "bolivia", "paraguay", "chile", "argentina", "uruguay",
})


def _region_hemisphere(region):
    """Expected (latitude_sign, longitude_sign) for a *standardized ITRDB*
    country/state name, or (None, None) if the name is not recognized. Only the
    Americas are handled -- their longitude is unambiguously West -- because
    Eastern-hemisphere longitude is not determinable from country alone (UK,
    Portugal, Morocco are West while their neighbours are East)."""
    r = (region or "").strip().lower()
    if r in _NORTH_AMERICA:
        return (1, -1)          # North, West
    if r in _SOUTH_AMERICA:
        return (None, -1)       # latitude ambiguous (equator), longitude West
    return (None, None)


def _dm_to_decimal(token):
    """Decode a packed degrees-minutes coordinate (DDMM / DDDMM, optional leading
    '-') to decimal degrees: '3627' -> 36.45, '-2053' -> -20.8833, '00406' -> 4.1."""
    neg = token.startswith("-")
    digits = token.lstrip("-")
    if not digits.isdigit() or len(digits) < 3:
        return None
    minutes = int(digits[-2:])
    degrees = int(digits[:-2])
    if minutes >= 60:
        return None                      # not a valid DDMM value
    val = degrees + minutes / 60.0
    return round(-val if neg else val, 4)


# Precompiled ITRDB Tucson header-parsing patterns (compiled once, not per file).
_RE_HEADER_PREFIX = re.compile(r"^\s*(\S+)\s+(?:\d\s)?(.*)$")
_RE_TRAILING_JUNK = re.compile(r"[\s\-]+$")     # trailing spaces / placeholder dashes
_RE_SPECIES_CODE = re.compile(r"[A-Z]{2,4}")
_RE_YEAR_PAIR = re.compile(r"(-?\d{3,4})\s*(-?\d{3,4})$")
_RE_ELEVATION = re.compile(r"(\d{2,5})\s*[Mm]\b")
_RE_LATLONG = re.compile(r"(-?\d{4})\s*(-?\d{4,5})")
_RE_FIELD_SPLIT = re.compile(r"\s{2,}")
_RE_LEADING_DIGIT = re.compile(r"^-?\d")


def _split_header_prefix(line):
    """Strip the leading 'SITEID  N ' prefix from a header line, returning
    (site_id, remaining_content). The line number (1/2/3) is optional -- some
    ITRDB files omit it -- so a header like 'SFP519AA ARIZONA ...' still splits
    into ('SFP519AA', 'ARIZONA ...')."""
    m = _RE_HEADER_PREFIX.match(line)
    if m:
        return m.group(1), m.group(2)
    toks = line.split()
    return (toks[0] if toks else None), line


def _extract_header_metadata(header_lines):
    """Parse ITRDB Tucson header lines into a metadata dict (best-effort)."""
    md = {k: None for k in _METADATA_FIELDS}
    md["n_header_lines"] = len(header_lines)
    md["header_raw"] = list(header_lines)
    if not header_lines:
        return md

    # ---- line 1: site id, site name, species code ----
    sid, c1 = _split_header_prefix(header_lines[0])
    md["site_id"] = sid
    c1 = _RE_TRAILING_JUNK.sub("", c1)          # strip trailing spaces / placeholder dashes
    toks1 = c1.split()
    if toks1 and _RE_SPECIES_CODE.fullmatch(toks1[-1]):
        md["species_code"] = toks1[-1]
        md["site_name"] = c1[:c1.rfind(toks1[-1])].strip() or None
    elif c1.strip():
        md["site_name"] = c1.strip()

    # ---- line 2: region, species name, elevation, lat/long, year range ----
    if len(header_lines) >= 2:
        _, c2 = _split_header_prefix(header_lines[1])
        c2 = _RE_TRAILING_JUNK.sub("", c2)      # strip trailing junk (e.g. ' -')
        # trailing year pair: space-separated OR bunched negatives ('-2220-1890')
        ym = _RE_YEAR_PAIR.search(c2)
        if ym:
            md["first_year"] = int(ym.group(1))
            md["last_year"] = int(ym.group(2))
            c2_geo = c2[:ym.start()]
        else:
            c2_geo = c2
        em = _RE_ELEVATION.search(c2_geo)          # elevation
        if em:
            md["elevation_m"] = int(em.group(1))
        gm = _RE_LATLONG.search(c2_geo)      # packed lat/long
        if gm:
            md["latitude"] = _dm_to_decimal(gm.group(1))
            md["longitude"] = _dm_to_decimal(gm.group(2))
        fields2 = [f for f in _RE_FIELD_SPLIT.split(c2.strip()) if f]
        if fields2:
            md["country_region"] = fields2[0]
            if len(fields2) >= 2 and not _RE_LEADING_DIGIT.match(fields2[1]):
                md["species_name"] = fields2[1]

    # ---- line 3: investigator(s) ----
    if len(header_lines) >= 3:
        _, c3 = _split_header_prefix(header_lines[2])
        c3 = _RE_TRAILING_JUNK.sub("", c3)
        md["investigators"] = c3.strip() or None

    # ---- hemisphere correction ----
    # The packed lat/long carries an unreliable sign (ITRDB files variably omit
    # the '-' on West longitudes or use '-' as a separator). Where the header's
    # country/state is a recognized standardized ITRDB name, force the correct
    # hemisphere. Otherwise leave the decoded sign and flag it unverified, so a
    # caller (e.g. a user's own non-ITRDB file) knows the coordinate sign was not
    # checked.
    md["hemisphere_verified"] = False
    lat_sign, lon_sign = _region_hemisphere(md["country_region"])
    if lat_sign is not None or lon_sign is not None:
        md["hemisphere_verified"] = True
        if lon_sign is not None and md["longitude"] is not None:
            md["longitude"] = lon_sign * abs(md["longitude"])
        if lat_sign is not None and md["latitude"] is not None:
            md["latitude"] = lat_sign * abs(md["latitude"])

    return md


def metadata(filename, header=None, skip_lines=0):
    """Extract site/sample metadata from a Tucson (.rwl) file's header.

    Returns a dict with best-effort fields (site_id, site_name, species_code,
    species_name, country_region, elevation_m, latitude, longitude, first_year,
    last_year, investigators) plus n_header_lines and the raw header lines.
    Unreadable fields are None. Reads only the header, so it is cheap and
    independent of loading the data; accepts a local path or an http(s) URL.
    This is a prototype -- see df.attrs['dplpy_metadata'] for the same result
    captured at read time.
    """
    is_url = filename.lower().startswith(("http://", "https://"))
    if is_url:
        raw = _fetch_url_lines(filename)
    else:
        with open(filename, "r") as fh:
            raw = fh.read().split("\n")
    clean = []
    for line in raw:
        line = line.rstrip("\r\n")
        if len(line.strip()) == 0:
            continue
        if 0 <= line.find("#") <= 77:
            continue
        clean.append(line)
    if skip_lines:
        clean = clean[skip_lines:]
    if not clean:
        return _extract_header_metadata([])
    start = 0 if header is False else _first_data_line_index(clean)
    return _extract_header_metadata(clean[:start])


def _fetch_url_lines(url):
    """Download a Tucson file from an http(s) URL and return it as a list of
    lines, matching what ``open(...).readlines()`` yields for a local file."""
    with urllib.request.urlopen(url) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return text.split("\n")


def _sniff_format(filename, is_url):
    """Decide 'tucson' vs 'csv' from a file's content when the suffix is not one
    of the recognized ones. Returns 'tucson', 'csv', or None if undetermined.
    A Tucson file has a line that parses as ID + integer year + numeric values;
    a CSV does not (its comma-joined fields fail that parse) but contains commas.
    File-open / network errors are left to propagate (so a missing file raises
    FileNotFoundError rather than a vague 'unknown format')."""
    if is_url:
        lines = _fetch_url_lines(filename)
    else:
        with open(filename, "r") as fh:
            lines = fh.read().split("\n")
    sample = []
    for ln in lines:
        ln = ln.rstrip("\r\n")
        if len(ln.strip()) == 0:
            continue
        if _is_comment_line(ln):               # '#'-comment or '#'-marked note row
            continue
        sample.append(ln)
        if len(sample) >= 50:
            break
    if not sample:
        return None
    start = _first_data_line_index(sample)
    for ln in sample[start:]:
        if _looks_like_data(ln):
            return "tucson"
    if any("," in ln for ln in sample):
        return "csv"
    return None


def _assemble_dataframe(rwl_data, precision, order):
    """Build a Year-column DataFrame from parsed rwl data. Shared by the file and
    URL readers. Years with no value are NaN. Returns None if there is no data."""
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
        # Vectorized: the {year: value} dict becomes a Series indexed by year,
        # divided to mm and reindexed to the full year span (NaN for missing
        # years). Values line up positionally with `index`, exactly as the old
        # per-year comprehension did, but at C speed rather than interpreted.
        col = (pd.Series(rwl_data[series]) / div).reindex(index).to_numpy()
        series_columns.append(pd.Series(data=col, name=series))
    return pd.concat([df] + series_columns, axis=1)


def _warn_salvage_summary(basename, report):
    """One concise per-file warning summarising salvage actions, so a large batch
    stays legible. Full detail lives in df.attrs['dplpy_salvage']."""
    dropped = [r for r in report if r["action"] == "dropped"]
    renamed = [r for r in report if r["action"].startswith("renamed")]
    parts = []
    if dropped:
        preview = "; ".join(r["series"] + ":" + r["issue"] for r in dropped[:5])
        parts.append("dropped " + str(len(dropped)) + " series (" + preview
                     + (" ..." if len(dropped) > 5 else "") + ")")
    if renamed:
        preview = "; ".join(r["action"] for r in renamed[:5])
        parts.append("renamed " + str(len(renamed)) + " (" + preview
                     + (" ..." if len(renamed) > 5 else "") + ")")
    if parts:
        warnings.warn("Salvaged " + basename + ": " + "; ".join(parts))


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


def _parse_fixed(line):
    """Parse one line by fixed Tucson columns: (id, year, [value strings]).

    Interior blank 6-char fields are kept as "" so a missing ring written as
    blanks holds its year position (dropping it would shift the rest of the
    decade); trailing blanks/notes are trimmed by _trim_trailing_junk.

    A bunched 5-char negative year (a BC year < -999 written with no space
    between ID and year, e.g. 'MNP262M-1262' = year -1262) is detected per line
    -- a '-' in column 8 with column 12 filled -- and read as a 7-char ID + a
    5-char year, the same handling dplR's `long` mode and read.tucson2 give the
    long-chronology case. Without this the '-' lands in the ID field and the year
    reads as +1262, which then trips the self-overlap check."""
    if len(line) >= 12 and line[7] == '-' and line[11] != ' ':
        idw, yrw = 7, 5                       # bunched 5-char BC year
    else:
        idw, yrw = 8, 4                        # standard 8-char id + 4-char year
    series_id = line[:idw].strip()
    year = int(line[idw:idw + yrw])          # may raise ValueError
    # The ITRDB decadal record holds exactly ten 6-char value fields (cols
    # 13-72); cols 74-78 are an optional site ID and anything further is junk.
    # Read only those ten fields so a nonstandard trailing column (e.g. a
    # per-row value count) or a joined next record does not inflate the value
    # count and force a fall back to whitespace parsing.
    rest = line[idw + yrw:idw + yrw + 60]
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


def _parses_as_data(line):
    """True if ``line`` parses as a Tucson data row (an id, a plausible year, values).

    Used to tell a data row whose series ID contains '#' (e.g. 'SP#1', 'GFI48C#H')
    from a '#'-marked note/annotation line that does not (e.g. a CDendro line
    'BAZENA  #### corrC GT 0.7...'). The former is data; the latter is a comment.
    """
    for parser in (_parse_fixed, _parse_ws):
        try:
            _sid, yr, vals = parser(line)
        except (ValueError, IndexError):
            continue
        if -12000 <= yr <= 12000 and vals:
            return True
    return False


def _is_comment_line(line):
    """A comment is a line whose first non-blank character is '#', or any other
    '#'-bearing line that is not a valid data row (a '#'-marked note/annotation).

    A '#' inside a real series ID is data (the row parses), so a parseable row is
    never treated as a comment -- this keeps ITRDB series like 'SP#1' / 'GFI48C#H'
    while still dropping interspersed annotation rows like 'BAZENA  #### ...'.
    """
    if line.lstrip().startswith("#"):
        return True
    return "#" in line and not _parses_as_data(line)


def _parse_all(lines, method):
    """Parse every line with one method. Returns a list where each element is
    (id, year, values, lineno) or None if that line could not be parsed."""
    rows = []
    for k, line in enumerate(lines):
        if len(line.strip()) < 7:
            rows.append(None)
            continue
        try:
            if method == "fixed":
                sid, yr, vals = _parse_fixed(line)
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


def _block_year_values(rows, row_indices):
    """Map of {year: raw integer value} a block assigns -- used to tell a true
    duplicate (two blocks giving DIFFERENT values for the same year) from an
    identical copy-paste (same values, handled by dedup) or a disjoint segment.
    Blank fields, trailing stop markers, and negatives (which become NaN) are
    excluded so they do not create spurious conflicts."""
    yv = {}
    for i in row_indices:
        sid, yr, vals, k = rows[i]
        n = len(vals)
        for j in range(n):
            tok = vals[j].strip()
            if tok == "":
                continue
            if j == n - 1 and tok in ("999", "-9999"):
                continue
            try:
                v = int(tok)
            except ValueError:
                continue
            if v < 0:
                continue
            yv[yr + j] = v
    return yv


def _rename_overlapping_duplicates(rows):
    """Salvage helper. When a series ID appears in more than one contiguous block
    and those blocks actually share years (two cores sharing a code), rename the
    second and later blocks (ID2, ID3, ...) so both are kept as distinct series.
    A single ID split into disjoint segments (a gap, e.g. dplR's BT006 case) is
    left untouched so it still merges. Mutates and returns `rows`, plus a list of
    {series, issue, action, detail} records for the report."""
    # Contiguous blocks, in file order, as lists of row indices.
    blocks = []
    cur = None
    for i, r in enumerate(rows):
        if r is None:
            continue
        sid = r[0]
        if cur is not None and cur["sid"] == sid:
            cur["rows"].append(i)
        else:
            if cur is not None:
                blocks.append(cur)
            cur = {"sid": sid, "rows": [i]}
    if cur is not None:
        blocks.append(cur)

    by_sid = {}
    for b in blocks:
        by_sid.setdefault(b["sid"], []).append(b)

    existing = set(by_sid.keys())
    records = []
    for sid, bl in by_sid.items():
        if len(bl) < 2:
            continue
        # A block is a true duplicate only if it gives a DIFFERENT value for a
        # year an earlier block already covered. Sharing years with identical
        # values is a copy-paste (dedup handles it); disjoint years is a
        # legitimately segmented series (merge). Only genuine conflicts rename.
        accum = {}
        rename_flags = []
        for b in bl:
            yv = _block_year_values(rows, b["rows"])
            conflict = any(y in accum and accum[y] != v for y, v in yv.items())
            rename_flags.append(conflict)
            if not conflict:
                for y, v in yv.items():
                    accum.setdefault(y, v)
        if not any(rename_flags):
            continue
        suffix = 2
        for b, needs_rename in zip(bl, rename_flags):
            if not needs_rename:
                continue  # keep the first (and any disjoint) block under the ID
            newid = sid + str(suffix)
            while newid in existing:
                suffix += 1
                newid = sid + str(suffix)
            existing.add(newid)
            for ri in b["rows"]:
                s, y, v, k = rows[ri]
                rows[ri] = (newid, y, v, k)
            records.append({"series": sid, "issue": "duplicate_id",
                            "action": "renamed to " + newid,
                            "detail": "overlapping duplicate block kept as " + newid})
            suffix += 1
    return rows, records


def _looks_like_record(s):
    """True if ``s`` begins a Tucson data record: an ID, a plausible year, a value."""
    toks = s.split()
    if len(toks) < 3:
        return False
    try:
        yr = int(toks[1])
    except ValueError:
        return False
    return -12000 <= yr <= 12000


def _split_joined_records(line, _depth=0):
    """Split a line where a stop marker is butted directly against a new core ID.

    A missing line break in the file can join one series' terminal row to the
    next series' first row, e.g. ``...   999FR-002  1660  ...``. Left alone this
    collapses into a mis-tokenised row (the values of the second record land on
    the first, inventing impossible years). We split at a stop marker (999 or
    -9999) that sits at a field boundary (not inside a value) and is immediately
    followed by a letter, but only when the remainder actually parses as a fresh
    record -- so a normal trailing site ID or note is left untouched. Returns a
    list of one or more lines.
    """
    if _depth >= 50:
        return [line]
    for m in re.finditer(r"(?<!\d)(?:999|-9999)(?=[A-Za-z])", line):
        cut = m.end()
        tail = line[cut:]
        if _looks_like_record(tail):
            return [line[:cut]] + _split_joined_records(tail, _depth + 1)
    return [line]


def read_rwl(lines, on_error="raise"):
    """Parse cleaned Tucson data lines into (rwl_data, precision, order, report).

    rwl_data  : {series_id: {year: raw_integer_value}}
    precision : {series_id: 100 or 1000}   (divisor to convert to mm)
    order     : [series_id, ...]           (first-appearance order)
    report    : [{series, issue, action, detail}, ...]  (salvage actions taken)

    Uses a fixed-width parse first (the Tucson "standard"), falling back to a
    whitespace-delimited parse when the fixed one fails validation -- exactly
    the strategy dplR uses to cope with long IDs, negative years, etc. Returns
    ``None`` if nothing usable could be parsed.

    on_error="warn" enables salvage mode: instead of raising on an unrecoverable
    per-series problem, a self-overlap or precision-shift series is dropped and a
    duplicate-ID's later block(s) are renamed and kept, each recorded in report.
    """
    salvage = (on_error == "warn")
    report = []

    # Repair joined records: a missing line break that butts a stop marker
    # against the next series' ID (e.g. '...999FR-002  1660...') is split so both
    # records parse independently instead of collapsing into a mis-tokenised line.
    repaired = []
    n_joined = 0
    for ln in lines:
        parts = _split_joined_records(ln)
        n_joined += len(parts) - 1
        repaired.extend(parts)
    if n_joined:
        warnings.warn(str(n_joined) + " joined record(s) split -- a stop marker was "
                      "directly followed by a new series ID (a missing line break in "
                      "the file)")
    lines = repaired

    fixed_rows = _parse_all(lines, "fixed")
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

    # Salvage: rename the later block(s) of any duplicate series ID whose blocks
    # overlap in time (two cores sharing a code) so both are kept as distinct
    # series. A single ID split into disjoint segments is left to merge, as in
    # strict mode. Done before block/precision analysis so renamed blocks are
    # treated as independent series downstream.
    if salvage:
        rows, rename_recs = _rename_overlapping_duplicates(rows)
        report.extend(rename_recs)

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
    nonint_cells = []   # (series, year, token) for non-integer tokens -> NaN
    identical_dups = 0  # cells duplicated with an identical value (copy-paste)
    year_src = {}       # sid -> {year: start-year of the row that wrote it}
    overlaps = []       # (sid, year, first_row_start, second_row_start)
    overlap_seen = set()
    prec_shift = []     # (sid, year) where a 0.01 mm series carries a stray -9999
    drop_series = set() # series to drop in salvage mode

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
                nonint_cells.append((sid, y, s))   # non-integer token -> NaN
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
    if overlaps and salvage:
        # In salvage mode the duplicate-ID overlaps were already resolved by the
        # rename pre-pass, so anything left is a series overlapping itself (a
        # malformed row). Drop those series and record it.
        for sid, y, first_start, second_start in overlaps:
            drop_series.add(sid)
            report.append({"series": sid, "issue": "self_overlap",
                           "action": "dropped",
                           "detail": "overlaps itself at year " + str(y)})
    elif overlaps:
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
        if salvage:
            for sid in affected:
                drop_series.add(sid)
                yy = min(y for s, y in prec_shift if s == sid)
                report.append({"series": sid, "issue": "precision_shift",
                               "action": "dropped",
                               "detail": "stray -9999 near year " + str(yy)})
        else:
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

    if nonint_cells:
        # Distinguish the recognizable "space-split" case -- a large ring width
        # (>= 1000, i.e. >= 10 mm) whose digits are separated by spaces inside its
        # fixed-width field, e.g. '1  065' meaning 1065. dplPy does not guess these
        # (neither does dplR, whose as.numeric() also returns NA), but names them so
        # the loss is actionable rather than an opaque count.
        split = [(sid, y, tok) for sid, y, tok in nonint_cells
                 if " " in tok.strip() and re.sub(r"\s+", "", tok).lstrip("-").isdigit()]
        msg = (str(len(nonint_cells)) + " non-integer value(s) could not be read "
               "and were set to NaN")
        if split:
            ex = "; ".join(str(sid) + "@" + str(y) + " '" + tok + "'="
                           + re.sub(r"\s+", "", tok) for sid, y, tok in split[:3])
            more = " ..." if len(split) > 3 else ""
            msg += ("; " + str(len(split)) + " look like space-split values whose digits "
                    "are separated by spaces in the fixed-width field (e.g. " + ex + more + ")")
        warnings.warn(msg)

    if neg_hits:
        preview = ", ".join(str(sid) + "@" + str(yr) for sid, yr in neg_hits[:5])
        more = "" if len(neg_hits) <= 5 else " ..."
        warnings.warn(
            str(len(neg_hits)) + " anomalous negative value(s) (not the -9999 stop "
            "marker) were set to NaN [" + preview + more + "]"
        )

    # Drop salvage-flagged series (self-overlap / precision-shift), then any
    # series that ended up with no usable data at all.
    if drop_series:
        order = [sid for sid in order if sid not in drop_series]
    order = [sid for sid in order if len(rwl_data.get(sid, {})) > 0]
    rwl_data = {sid: rwl_data[sid] for sid in order}
    if len(order) == 0:
        return None

    dropped = len(nonint_cells) + len(neg_hits)   # values the file had but couldn't be used
    return rwl_data, precision, order, report, dropped
