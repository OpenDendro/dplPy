"""Fire-history (FHX2) data model, reader/writer, and accessors.

A port of the foundational layer of the R package burnr (Malevich, Guiterman &
Margolis; Malevich et al. 2018, Dendrochronologia,
doi:10.1016/j.dendro.2018.02.005) for reading, writing, and manipulating forest
fire-scar history data.

An ``fhx`` object here is a plain :class:`pandas.DataFrame` (tagged
``attrs["kind"] = "fhx"``) with three columns:

* ``year``     -- the calendar year of the observation (int)
* ``series``   -- the sample/series id (str)
* ``rec_type`` -- the record type, an ordered Categorical over the controlled
  vocabulary in :mod:`dplpy._fhx_vocab` (fire scars/injuries by season, recorder
  years, pith/bark/inner/outer, estimate)

This is a tidy event table (one row per observed year x series), unlike dplPy's
wide ring-width matrix.
"""
import re

import numpy as np
import pandas as pd

from ._validate import _warn
from ._fhx_vocab import (REC_TYPE_LEVELS, REC_TYPE_ABRV, REC_TYPE_ALL,
                         SCAR, INJURY, CANON, _NODATA)


# --- the rec_type factor --------------------------------------------------

def _make_rec_type(x):
    """Turn a rec_type name (or iterable of names) into an ordered Categorical
    with burnr's canonical levels. Raises on an unknown name."""
    vals = [x] if isinstance(x, str) else list(x)
    bad = sorted(set(map(str, vals)) - set(REC_TYPE_LEVELS))
    if bad:
        raise ValueError("unknown rec_type(s): " + ", ".join(bad)
                         + ". See dplpy._fhx_vocab.REC_TYPE_LEVELS.")
    return pd.Categorical(vals, categories=REC_TYPE_LEVELS, ordered=False)


# --- construction / coercion / predicate ----------------------------------

def frame(year, series, rec_type):
    """Construct a validated ``fhx`` DataFrame from parallel arrays of year,
    series, and rec_type (names or a Categorical)."""
    year = np.asarray(list(year), dtype="int64")
    series = np.asarray([str(s) for s in series], dtype=object)
    if not isinstance(rec_type, pd.Categorical):
        rec_type = _make_rec_type(list(rec_type))
    else:
        rec_type = pd.Categorical(rec_type, categories=REC_TYPE_LEVELS)
    if not (len(year) == len(series) == len(rec_type)):
        raise ValueError("year, series, and rec_type must be the same length.")
    out = pd.DataFrame({"year": year, "series": series, "rec_type": rec_type})
    out.attrs["kind"] = "fhx"
    _check_duplicates(out)
    return out


def as_frame(df):
    """Cast a DataFrame-like with year/series/rec_type columns to an ``fhx``."""
    df = pd.DataFrame(df)
    missing = {"year", "series", "rec_type"} - set(df.columns)
    if missing:
        raise ValueError("cannot cast to fhx; missing column(s): "
                         + ", ".join(sorted(missing)))
    return frame(df["year"], df["series"], df["rec_type"])


def is_fhx(x):
    """True if ``x`` is an fhx-style frame (year/series/rec_type, rec_type a
    Categorical over the controlled vocabulary)."""
    return (isinstance(x, pd.DataFrame)
            and {"year", "series", "rec_type"}.issubset(x.columns)
            and isinstance(x["rec_type"].dtype, pd.CategoricalDtype))


def _require_fhx(x):
    if not is_fhx(x):
        raise TypeError("expected an fhx object (year/series/rec_type frame); "
                        "use dpl.as_frame() to cast.")


def _check_duplicates(x):
    dup = x.duplicated(subset=["year", "series"])
    if dup.any():
        bad = x.loc[dup, ["series", "year"]].head(5)
        _warn("fhx has duplicate (series, year) observations, e.g.:\n"
              + bad.to_string(index=False))


def _violates_canon(x):
    """True if any rec_type is outside the original FHX2 canon (others may not be
    readable by older software)."""
    _require_fhx(x)
    return bool((~x["rec_type"].isin(CANON)).any())


# --- reader ---------------------------------------------------------------

_HEADER_RE = re.compile(r"^(FHX2|FIRE2)\s+FORMAT", re.IGNORECASE)


def read(fname, encoding="utf-8"):
    """Read an FHX2 ``.fhx`` fire-history file into an ``fhx`` DataFrame.

    Parses the ``FHX2 FORMAT`` (or legacy ``FIRE2 FORMAT``) body: the
    ``first_year n_series id_length`` line, the vertically-stacked series names,
    and the year x series grid of single-character codes. The free-text metadata
    header is ignored, as in burnr.
    """
    with open(fname, encoding=encoding) as fh:
        text = fh.read()
    return _parse_fhx(text)


def _parse_fhx(text):
    lines = text.split("\n")
    first = next((i for i, ln in enumerate(lines) if _HEADER_RE.match(ln)), None)
    if first is None:
        raise ValueError("Cannot find line 'FHX2 FORMAT' or 'FIRE2 FORMAT'.")
    desc = lines[first + 1].split()
    if len(desc) != 3:
        raise ValueError("The descriptive line after 'FHX2 FORMAT' must have "
                         "three space-separated integers "
                         "(first_year n_series id_length).")
    first_year, n_series, id_len = (int(desc[0]), int(desc[1]), int(desc[2]))

    # series names: id_len lines, n_series wide, read down each column
    id_block = lines[first + 2: first + 2 + id_len]
    names = []
    for j in range(n_series):
        chars = [(id_block[i][j] if i < len(id_block) and j < len(id_block[i])
                  else " ") for i in range(id_len)]
        names.append("".join(chars).strip())

    # body starts after the blank line(s) following the id block
    k = first + 2 + id_len
    while k < len(lines) and lines[k].strip() == "":
        k += 1

    years, series_out, codes = [], [], []
    for r, line in enumerate(lines[k:]):
        yr = first_year + r
        for j in range(n_series):
            if j >= len(line):
                continue                      # short line -> no data
            c = line[j]
            if c in _NODATA or c == " ":
                continue
            name = REC_TYPE_ALL.get(c)
            if name is None:
                raise ValueError("unknown FHX2 code %r at year %d, series %r"
                                 % (c, yr, names[j]))
            years.append(yr)
            series_out.append(names[j])
            codes.append(name)
    return frame(years, series_out, _make_rec_type(codes))


# --- writer ---------------------------------------------------------------

def write(x, fname):
    """Write an ``fhx`` object to an FHX2 ``.fhx`` file. Round-trips with
    :func:`read`. Warns if any rec_type falls outside the FHX2 canon."""
    _require_fhx(x)
    if _violates_canon(x):
        _warn("write: object has rec_types outside the FHX2 canon; other "
              "software may not read the output.")
    series = _series_names(x)
    n = len(series)
    id_len = max((len(s) for s in series), default=1)
    ymin, ymax = int(x["year"].min()), int(x["year"].max())
    code = {(int(row.year), str(row.series)): REC_TYPE_ABRV[str(row.rec_type)]
            for row in x.itertuples(index=False)}

    out = ["FHX2 FORMAT", "%d %d %d" % (ymin, n, id_len)]
    for i in range(id_len):                   # series names written vertically
        out.append("".join((s[i] if i < len(s) else " ") for s in series))
    out.append("")                            # blank line before the body
    for yr in range(ymin, ymax + 1):
        row = "".join(code.get((yr, s), ".") for s in series)
        out.append("%s %d" % (row, yr))       # trailing year, as burnr writes
    with open(fname, "w") as fh:
        fh.write("\n".join(out) + "\n")


# --- accessors ------------------------------------------------------------

def _series_names(x):
    """The series ids in ``x``, in order of first appearance."""
    _require_fhx(x)
    return list(pd.unique(x["series"].astype(str)))


def _year_range(x):
    """``(first_year, last_year)`` spanned by ``x``."""
    _require_fhx(x)
    return (int(x["year"].min()), int(x["year"].max()))


def events(x, scar_event=True, injury_event=False, custom_search=None):
    """Per-series event years, as ``{series: [years]}``. By default a fire scar
    (``_fs``) is an event; set ``injury_event=True`` to also count injuries
    (``_fi``). ``custom_search`` overrides with a regex matched against rec_type.
    """
    _require_fhx(x)
    if custom_search is not None:
        pat = custom_search
    else:
        parts = (["_fs"] if scar_event else []) + (["_fi"] if injury_event else [])
        pat = "|".join(parts) if parts else r"(?!x)x"   # match nothing
    rt = x["rec_type"].astype(str)
    hit = rt.str.contains(pat, regex=True)
    out = {}
    for s in _series_names(x):
        m = hit & (x["series"].astype(str) == s)
        out[s] = x.loc[m, "year"].to_numpy()
    return out


def sort(x, by="first_year", decreasing=False):
    """Reorder the series of ``x`` by their first (or last) year. ``by`` is
    ``"first_year"`` or ``"last_year"``."""
    _require_fhx(x)
    if by not in ("first_year", "last_year"):
        raise ValueError("by must be 'first_year' or 'last_year'.")
    agg = x.groupby("series", observed=True)["year"]
    key = (agg.min() if by == "first_year" else agg.max()).sort_values(
        ascending=not decreasing)
    order = {s: i for i, s in enumerate(key.index)}
    out = x.assign(_o=x["series"].map(order)).sort_values(
        ["_o", "year"]).drop(columns="_o").reset_index(drop=True)
    return as_frame(out)
