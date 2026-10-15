from __future__ import print_function

__copyright__ = """
   dplPy for tree ring width time series analyses
   Copyright (C) 2021  OpenDendro

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

# Date: 11/17/2021
# Author: Tyson Lee Swetnam
# Project: OpenDendro- Writers
# Description: Writers for all supported file types (*.CSV, *.RWL, and *.TXT)
#
# example usages:
# >>> import dplpy as dpl
# >>> rwl = dpl.readers("./data/in_file.rwl")
# >>> dpl.writers(rwl, "./data/out_file", "csv")        # -> ./data/out_file.csv
# >>> dpl.writers(rwl, "./data/out_file", "rwl")        # -> ./data/out_file.rwl
# >>> crn = dpl.chron(dpl.detrend(rwl, plot=False), plot=False)
# >>> dpl.writers(crn, "./data/site", "crn", header=dpl.SiteMetadata.from_rwl(rwl))

import pandas as pd
import numpy as np
from ._validate import (_require_dataframe, _warn, _normalize_choice,
                        _require_year_index)
from .site_metadata import SiteMetadata, pack_crn_coordinates, pack_crn_elevation


def _tridas_metadata_from_header(header):
    """Map a writers() `header` (a SiteMetadata, a dict, or None) onto what
    write_tridas() wants: a (metadata, taxon) pair. SiteMetadata is site-level,
    so its site name becomes the object title and its species the element taxon.
    Returns (metadata_or_None, taxon_or_None); the convenience path writes with
    validate=False, so missing fields become empty placeholders."""
    if header is None:
        return (None, None)
    if isinstance(header, SiteMetadata):
        md = {
            "project_title": header.site_name,
            "investigator": header.investigators,
            "objects": [{"title": header.site_name, "type": "unknown"}],
        }
        return (md, header.species_name)
    if isinstance(header, dict):
        return (header, header.get("taxon"))
    # a TridasMetadata (or anything write_tridas accepts) passes through
    return (header, None)


def _resolve_precision(data, prec, allowed, what):
    """Resolve the measurement precision to write at (audit C2).

    ``prec=None`` looks up ``data.attrs["dplpy_precision"]`` (what the series
    were read at); one precision -> use it; several -> raise asking for an
    explicit ``prec``; none -> the ``what``-specific default is returned as None
    for the caller to apply. An explicit ``prec`` must be in ``allowed``."""
    if prec is not None:
        if prec not in allowed:
            raise ValueError("prec must be one of " + ", ".join(str(a) for a in allowed)
                             + ", got " + repr(prec))
        return prec
    attrs = getattr(data, "attrs", {}) or {}
    found = attrs.get("dplpy_precision")
    if isinstance(found, dict):
        values = set(found.values())
    elif found is None:
        values = set()
    else:
        values = {found}
    values = {v for v in values if v in allowed}
    if len(values) == 1:
        return values.pop()
    if len(values) > 1:
        raise ValueError(
            what + ": the data carry mixed measurement precisions "
            + str(sorted(values)) + " (df.attrs['dplpy_precision']); pass prec= "
            "explicitly to write every series at one precision.")
    return None


def _check_unique_columns(data, what):
    """Raise a clear ValueError on duplicate column names (audit B6)."""
    dups = pd.Index(data.columns)[pd.Index(data.columns).duplicated()].unique().tolist()
    if dups:
        raise ValueError(
            what + ": duplicate series names " + str([str(d) for d in dups][:8])
            + " cannot be written (each column must have a unique name); rename "
            "them first, e.g. data.columns = [...].")


def writers(data, label, format, header=None, chronology_type="standard",
            column="std", prec=None, gaps=-99, sep="\t", float_format="%.4f",
            na_rep="NA"):
    """Output dplpy datasets to .csv, .rwl, .crn, .txt and .tridas (.xml) files.

    Write a dataframe to a file as set by ``format``, created in the current
    directory unless a path is included in ``label``. For 'csv'/'rwl', ``data``
    is a ring-width dataframe (years x series); for 'crn', ``data`` is a
    *chronology* (from dpl.chron()) and a ``header`` of site metadata is
    REQUIRED to populate the Tucson .crn header.

    Parameters
    ----------
    data : pandas dataframe
        ring widths (csv/rwl) or a chronology from dpl.chron() (crn).
    label : str
        name (can include file path) to give the file; no extension.
    format : str
        'csv', 'rwl', 'crn', 'txt' or 'tridas' (case-insensitive).
    header : dict or SiteMetadata, required for format='crn'
        site metadata with the keys: 'site_id', 'site_name', 'species_code',
        'state_country', 'species', 'elevation', 'latitude', 'longitude',
        'investigators'; optionally 'completion_date'. First/last year are taken
        from the chronology. Coordinates may be decimal degrees (packed to the
        ITRDB DDMM/DDDMM field) or already-packed strings; elevation in metres.
    chronology_type : str, default 'standard'
        for format='crn', the ITRDB chronology type written in header record 2:
        'standard' (blank code), 'arstan' ('A'), or 'residual' ('R').
    column : str, default 'std'
        for format='crn', which chronology column to write (e.g. 'std', 'res').
    prec : float or None, default None
        for format='rwl'/'tridas', the measurement precision in mm. ``None`` uses
        the precision the data were read at (``df.attrs["dplpy_precision"]``;
        mixed precisions raise, pass ``prec`` explicitly), else 0.001 for rwl
        (values x1000, end marker -9999; 0.01 writes x100 with a 999 marker, the
        dplR mapping) and unrounded millimetres for tridas.
    gaps : {int, "split"}, default -99
        for format='rwl', how to encode a *true interior gap* (a NaN inside a series;
        a real 0 is a locally absent ring and is always written as 0): a negative
        integer (default -99) writes that sentinel in the gap (it must be negative so
        it is not read as a ring width, and not a stop marker; the reader turns it
        back into NaN), or "split" closes the block with the end marker and reopens a
        new block at the next present year.
    sep, float_format, na_rep :
        for format='txt', the column delimiter (default tab), printf float format
        (default '%.4f', None for full precision), and missing-value text (default
        'NA'). For 'txt', ``data`` may be a DataFrame, a single Series, or a list of
        year-aligned Series.

    Returns
    -------
    str
        the path of the file written.

    References
    ----------
    .. [1] https://opendendro.github.io/dplPy/reference/io/#writers
    """
    if not isinstance(label, str):
        raise TypeError("Expected label to be of type str, not " + str(type(label)))
    if not isinstance(format, str):
        raise TypeError("Expected format to be of type str, not " + str(type(format)))

    # 'txt' accepts a DataFrame, a single Series, or a list of year-aligned Series
    # (the author chooses the columns). Every other format requires a DataFrame.
    if format.strip().lower() != "txt":
        _require_dataframe(data)
    format = _normalize_choice(format, ("csv", "rwl", "crn", "txt", "tridas"), "format")

    # TRiDaS is written by the standalone writer (its metadata surface dwarfs the
    # other formats). This is the convenience path: treat `data` as ring widths
    # unless it carries a samp_depth column (a chronology). The full-control path,
    # with mandatory-field validation, is dpl.write_tridas().
    if format == "tridas":
        from .tridas import write_tridas
        cols = [str(c).lower() for c in data.columns]
        is_crn = any(c.startswith("samp_depth") or c.startswith("samp.depth")
                     for c in cols)
        md, taxon = _tridas_metadata_from_header(header)
        return write_tridas(label,
                            rwl=None if is_crn else data,
                            crn=data if is_crn else None,
                            metadata=md, taxon=taxon, prec=prec, validate=False)

    filename = label + "." + format
    with open(filename, "w") as output:
        if format == "csv":
            write_csv(data, output)
        elif format == "rwl":
            write_rwl(data, output, prec=prec, gaps=gaps)
        elif format == "crn":
            write_crn(data, output, header=header,
                      chronology_type=chronology_type, column=column)
        else:
            write_txt(data, output, sep=sep, float_format=float_format, na_rep=na_rep)
    return filename


def conv_data(data):
    if np.isnan(data):
        return "NA"
    else:
        return str(data)


def write_csv(data, file):
    """Write a year-indexed frame as a dplPy CSV ('Year' then one column per
    series, NaN as 'NA'). Duplicate column names raise (audit B6)."""
    _check_unique_columns(data, "write_csv")
    file.write('"Year","')
    file.write('","'.join(str(c) for c in data.columns.tolist()))
    file.write('"\n')

    for year, row in data.iterrows():
        file.write(str(year))
        file.write(",")
        file.write(",".join(map(conv_data, row)))
        file.write('\n')


def _tucson_names(columns, bc):
    """Series IDs for the Tucson file: at most 8 characters (7 when any year is
    BC, since the sign borrows a column). Longer IDs are truncated with a warning,
    as dplR does, and kept unique after truncation (audit C1)."""
    maxlen = 7 if bc else 8
    names = [str(c) for c in columns]
    out = []
    renamed = []
    for name in names:
        new = name
        if len(name) > maxlen:
            new = name[:maxlen]
            k = 2
            while new in out or (new != name and new in names):
                suffix = str(k)
                new = name[:maxlen - len(suffix)] + suffix
                k += 1
            renamed.append(name + " -> " + new)
        out.append(new)
    if renamed:
        _warn("write_rwl: " + str(len(renamed)) + " series ID(s) longer than "
              + str(maxlen) + " characters were truncated for the Tucson format: "
              + ", ".join(renamed[:6]) + (" ..." if len(renamed) > 6 else ""),
              stacklevel=4)
    return out


def write_rwl(data, file, prec=None, gaps=-99):
    """Write a ring-width dataframe (years x series) to a Tucson decadal .rwl file.

    ``prec`` is the measurement precision in mm; the precision->marker mapping
    follows dplR's write.tucson: 0.001 writes values * 1000 with a -9999 end marker,
    0.01 writes values * 100 with a 999 end marker. ``None`` (default) uses the
    precision the data were read at (``data.attrs["dplpy_precision"]``), else
    0.001. Values are right-justified in 6-column fields. dplPy encodes interior
    gaps via ``gaps`` rather than dplR's 0 / -9.99 missing string.

    A real 0 in ``data`` is a *locally absent ring* and is always written as 0.
    ``gaps`` controls how a *true interior gap* (a NaN inside a series' span) is
    encoded:

    * a negative integer (default ``-99``): write that sentinel in the gap within
      one continuous block. It must be negative (so the reader does not read it as a
      width) and must not collide with a stop marker; the reader turns it back into
      NaN on read.
    * ``"split"``: close the current block with the end marker and reopen a new
      block at the next present year, restoring the gap as NaN on read.

    Caveats. The Tucson format cannot carry a negative value: a negative RWI
    (difference detrending) is written, with a warning, but reads back as NaN.
    A value that rounds to the stop code (9.99 mm at prec=0.01, -9.999 at 0.001)
    is ambiguous on read; a warning suggests ``prec=0.001``. The year index must
    be contiguous (reindex to a full year range first; missing years as NaN) and
    series names must be unique and at most 8 characters (7 with BC years;
    longer names are truncated with a warning).
    """
    prec = _resolve_precision(data, prec, (0.01, 0.001), "write_rwl")
    if prec is None:
        prec = 0.001
    if prec == 0.01:
        rproc, stop_code = 100, 999
    else:
        rproc, stop_code = 1000, -9999

    if gaps != "split":
        # A gap sentinel: must be a plain integer, negative (so it is never read
        # as a width), and not a value the reader treats specially (999 / -9999
        # are stop / precision-shift markers at these precisions).
        if isinstance(gaps, bool) or not isinstance(gaps, (int, np.integer)):
            raise ValueError('gaps must be "split" or a negative integer '
                             '(e.g. -9 or -99), got ' + repr(gaps))
        gaps = int(gaps)
        if gaps >= 0:
            raise ValueError("gaps sentinel must be negative so it is not read as "
                             "a ring width, got " + repr(gaps))
        if gaps in (-9999, 999, stop_code):
            raise ValueError("gaps sentinel " + repr(gaps) + " collides with a "
                             "stop marker; use e.g. -9 or -99")

    _check_unique_columns(data, "write_rwl")
    data = _require_year_index(data, "data", require_contiguous=True)
    years = np.asarray(data.index, dtype=int)
    bc = bool(len(years) and years.min() < 0)
    names = _tucson_names(data.columns, bc)

    def head(name, year):
        # The series id + year end at column 12. A BC (negative) year needs one
        # extra column, taken from the id field.
        if year < 0:
            return name.ljust(7) + str(year).rjust(5)
        return name.ljust(8) + str(year).rjust(4)

    def cell(value):
        return "%6d" % int(value)

    matrix = data.to_numpy(dtype=float)
    n_negative = 0
    stop_hits = []
    for j, name in enumerate(names):
        col = matrix[:, j]
        present = np.flatnonzero(~np.isnan(col))
        if len(present) == 0:                       # all-NaN series: skip
            continue
        first, last = int(present[0]), int(present[-1])
        vals = col[first:last + 1]
        yrs = years[first:last + 1]
        codes = np.where(np.isnan(vals), 0, np.round(vals * rproc)).astype(int)
        n_negative += int((vals < 0).sum())
        if np.any(codes[~np.isnan(vals)] == stop_code):
            stop_hits.append(name)
        k, n = 0, len(vals)
        file.write(head(name, int(yrs[0])))
        while k < n:
            year = int(yrs[k])
            if np.isnan(vals[k]) and gaps == "split":
                # split mode: close this block, skip the NaN run, reopen at the
                # next present year.
                file.write(cell(stop_code) + "\n")
                while k < n and np.isnan(vals[k]):
                    k += 1
                if k < n:
                    file.write(head(name, int(yrs[k])))
                continue
            if np.isnan(vals[k]):
                file.write(cell(gaps))              # sentinel; stays in the block
            else:
                file.write(cell(codes[k]))
            k += 1
            if (year + 1) % 10 == 0:                # new decade line (the stop
                file.write("\n" + head(name, year + 1))   # marker may sit on it)
        file.write(cell(stop_code) + "\n")

    if n_negative:
        _warn("write_rwl: " + str(n_negative) + " negative value(s) were written; the "
              "Tucson format cannot carry negative widths or indices, so they read "
              "back as NaN (anomalous negatives). Use csv or tridas to keep them.",
              stacklevel=4)
    if stop_hits:
        _warn("write_rwl: a value in series " + ", ".join(stop_hits[:6])
              + (" ..." if len(stop_hits) > 6 else "") + " rounds to the stop code "
              + str(stop_code) + " at prec=" + str(prec) + " and will be read as the "
              "end of the series; write with prec=0.001 (or csv) to avoid the "
              "ambiguity.", stacklevel=4)


# Required site-metadata keys for the .crn header, and the ITRDB chronology
# type codes written in header record 2 (columns 62-63).
_CRN_HEADER_KEYS = ["site_id", "site_name", "species_code", "state_country",
                    "species", "elevation", "latitude", "longitude",
                    "investigators"]
_CRN_TYPE_CODE = {"standard": "  ", "arstan": "A ", "residual": "R "}
_CRN_MISSING = "9990  0"       # missing-value marker + 0 depth


def write_crn(chron_data, file, header=None, chronology_type="standard", column="std"):
    """Write a chronology to a Tucson .crn file.

    ``chron_data`` is a chronology -- the output of dpl.chron() -- with an index
    column (``column``, default 'std') and a 'samp_depth' column, indexed by
    year. Index values are written as integers = round(index, 3) * 1000 (so 1000
    is mean growth), with the sample depth alongside, in the ITRDB decadal layout;
    the missing-value code is 9990 and a NaN depth is written as 0. ``header``
    populates the three ITRDB header records; it may be a plain dict (keys below)
    or a ``SiteMetadata`` object (e.g. ``dpl.SiteMetadata.from_rwl(rwl)``).
    Decimal-degree coordinates and a numeric elevation are packed into the ITRDB
    fields (DDMM / DDDMM, '%dM').

    Field widths are enforced: an index must round to at most 9999 (|index| <
    10) and a depth lie in 0-999, else ValueError naming the year. A chronology
    with any BC year uses dplR's layout (a 5-character site ID, the sign borrowing
    column 6); a year below -9999 cannot be written.
    """
    if column not in chron_data.columns:
        raise ValueError("chronology column '" + str(column) + "' not found; "
                         "available columns: " + str(list(chron_data.columns)))
    if "samp_depth" not in chron_data.columns:
        raise ValueError("chronology must have a 'samp_depth' column -- pass the "
                         "output of dpl.chron().")
    # Accept a SiteMetadata (the shared metadata object) as well as a plain dict.
    # A SiteMetadata is rendered to the .crn header dict; a dict is used as-is.
    if isinstance(header, SiteMetadata):
        header = header.to_crn_header()
    if not isinstance(header, dict):
        raise ValueError("writing a .crn file requires a 'header' dict (or a "
                         "SiteMetadata) of site metadata with keys: "
                         + ", ".join(_CRN_HEADER_KEYS))
    missing = [k for k in _CRN_HEADER_KEYS if k not in header]
    if missing:
        raise ValueError("header is missing required key(s): " + ", ".join(missing))
    ct = _normalize_choice(chronology_type, tuple(_CRN_TYPE_CODE), "chronology_type")

    chron_data = _require_year_index(chron_data, "chron_data")
    years = np.asarray(chron_data.index, dtype=int)
    idx = chron_data[column].to_numpy(dtype=float)
    depth = chron_data["samp_depth"].to_numpy(dtype=float)
    first_yr, last_yr = int(years.min()), int(years.max())

    # --- layout: a BC chronology borrows column 6 for the year's sign (dplR) ---
    bc = first_yr < 0
    if bc and first_yr < -9999:
        raise ValueError("write_crn: year " + str(first_yr) + " needs more than 5 "
                         "characters; the Tucson .crn format cannot carry it.")
    id_width = 5 if bc else 6
    full_id = str(header["site_id"])
    site_id = full_id[:id_width]
    if len(full_id) > id_width:
        _warn("write_crn: site id '" + full_id + "' truncated to '" + site_id + "' ("
              + str(id_width) + " characters" + (" with BC years" if bc else "")
              + ").", stacklevel=4)

    # --- field-width validation up front, naming the year (audit A7) ---
    codes = np.full(len(idx), 0, dtype=int)
    ok = ~np.isnan(idx)
    codes[ok] = np.round(idx[ok] * 1000).astype(int)
    bad = ok & ((codes > 9999) | (codes < -999))
    if bad.any():
        y = int(years[np.flatnonzero(bad)[0]])
        raise ValueError("write_crn: index value " + repr(float(idx[years == y][0]))
                         + " at year " + str(y) + " does not fit the 4-character "
                         "field (|index| must be < 10).")
    depth_i = np.where(np.isnan(depth), 0, np.round(depth)).astype(int)
    badd = (depth_i > 999) | (depth_i < 0)
    if badd.any():
        y = int(years[np.flatnonzero(badd)[0]])
        raise ValueError("write_crn: sample depth " + str(int(depth_i[years == y][0]))
                         + " at year " + str(y) + " is outside 0-999 (3-character "
                         "field).")

    # --- three ITRDB header records ---
    def fld(v, w):                                  # left-justified fixed field
        return str(v)[:w].ljust(w)
    lat_long = pack_crn_coordinates(header["latitude"], header["longitude"])
    elev = pack_crn_elevation(header["elevation"])
    yrs = ("%d %d" % (first_yr, last_yr)).ljust(9)
    rec1 = fld(site_id, 6) + "   " + fld(header["site_name"], 52) + fld(header["species_code"], 4)
    rec2 = (fld(site_id, 6) + "   " + fld(header["state_country"], 13)
            + fld(header["species"], 18) + fld(elev, 5) + "  "
            + fld(lat_long, 10) + "    " + _CRN_TYPE_CODE[ct] + "    " + yrs)
    rec3 = (fld(site_id, 6) + "   " + fld(header["investigators"], 63)
            + fld(header.get("completion_date", ""), 8))
    file.write(rec1 + "\n" + rec2 + "\n" + rec3 + "\n")

    # --- decadal data records ---
    def pair(j):
        if not ok[j]:
            return _CRN_MISSING
        return ("%4d" % codes[j]) + ("%3d" % depth_i[j])

    def head(year):
        if bc:
            return site_id.ljust(5) + ("%5d" % year)
        return site_id.ljust(6) + ("%4d" % year)

    decades = years // 10 * 10
    unique_decades = sorted(set(decades.tolist()))
    lines = []
    for di, dec in enumerate(unique_decades):
        pos = np.flatnonzero(decades == dec)
        n = len(pos)
        pairs = "".join(pair(j) for j in pos)
        # the opening (partial) decade is padded at the FRONT to 10 pairs
        front = (_CRN_MISSING * (10 - n)) if di == 0 else ""
        lines.append(head(int(years[pos[0]])) + front + pairs)
    # pad the final (partial) decade at the END to 10 pairs
    n_last = int((decades == unique_decades[-1]).sum())
    lines[-1] = lines[-1] + _CRN_MISSING * (10 - n_last)
    file.write("\n".join(lines) + "\n")


def write_txt(data, file, sep="\t", float_format="%.4f", na_rep="NA"):
    """Write a set of year-aligned variables to a tab/space-delimited text table.

    This is a general "columns of your choosing" writer, modelled on ARSTAN's
    "tabs" file but not fixed to its columns: you decide what to output. Typical
    ARSTAN columns -- sample depth, mean segment length, mean age, and the raw /
    standard / residual / ARSTAN chronologies -- are assembled by the caller from
    dpl.samp_stats() and dpl.chron_ars() (and a raw-width mean), e.g.::

        meta = dpl.samp_stats(data)                     # samp_depth, seg, age
        ars  = dpl.chron_ars(rwi)                        # std, res, ars, samp_depth
        raw  = data.mean(axis=1).rename("raw")          # raw-width mean chronology
        tabs = pd.concat([meta[["samp_depth", "seg", "age"]], raw,
                          ars[["std", "res", "ars"]]], axis=1)
        dpl.writers(tabs, "site_tabs", "txt")

    Parameters
    ----------
    data : pandas dataframe, or a single Series, or a list/tuple of Series
        the variables to write, one per output column. A list/tuple of named
        Series is aligned on the union of their year indexes; each column's header
        is its Series/column name. An unnamed index is written as 'Year'.
    sep : str, default "\\t"
        column delimiter (tab by default; pass e.g. " " or "," for other layouts).
    float_format : str or None, default "%.4f"
        printf-style format for floating-point values; None writes full precision.
    na_rep : str, default "NA"
        text written for missing (NaN) cells.
    """
    if isinstance(data, pd.Series):
        frame = data.to_frame()
    elif isinstance(data, (list, tuple)):
        if not all(isinstance(s, pd.Series) for s in data):
            raise TypeError("For 'txt', a list input must contain only pandas "
                            "Series (one per column).")
        frame = pd.concat(list(data), axis=1)
    elif isinstance(data, pd.DataFrame):
        frame = data
    else:
        raise TypeError("Expected a pandas DataFrame, Series, or a list of Series, "
                        "not " + str(type(data)))

    frame = frame.sort_index()
    frame.index.name = frame.index.name or "Year"
    frame.to_csv(file, sep=sep, float_format=float_format, na_rep=na_rep)
