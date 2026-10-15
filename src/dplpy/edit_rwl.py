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

# Title: edit_rwl.py
# Project: OpenDendro dplPy
# Description: Insert, delete, or change a single ring in one series, the fix
#              that pairs with the crossdating checks. One dispatch function with
#              an action= argument, folding dplR's insert.ring and delete.ring
#              (and adding a value edit) into dpl.edit_rwl.

import numpy as np
import pandas as pd

from ._validate import _require_year_index, _normalize_choice, _carry_attrs, _warn

_ACTIONS = ("insert", "delete", "change")


def edit_rwl(data, series=None, year=None, action="insert", value=None,
             fix_last=True, fix_length=None):
    """Insert, delete, or change one ring in a single series.

    Returns a **new** object; the input is never modified in place (unlike the
    old Fortran programs, where the edit mutated the record). Assign the result
    to keep it. The fix that pairs with the crossdating checks: `xdate`,
    `series_corr`, and `check` find a misdated ring (a NEGATIVE lag there means
    a MISSING ring, so ``action="insert"``), and `edit_rwl` corrects it.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        a ring-width collection (then `series` names the column to edit) or a
        single year-indexed series. Leading/trailing NaN rows of a Series are kept.
    series : str or None
        the series (column) to edit; required when `data` is a DataFrame.
    year : int
        the calendar year to act on (a whole number; NaN or 2003.7 is an error).
        For `action="insert"` it may also be the year just before the series
        (prepend).
    action : {"insert", "delete", "change"}, default "insert"
        `"insert"` adds a ring (a missed ring), `"delete"` removes one (a false
        or double-counted ring), `"change"` replaces a measured value without
        changing the series length or dating. Case-insensitive.
    value : float or None
        the ring width, a finite non-negative number. Required for `"change"`;
        for `"insert"` it defaults to the series mean; ignored for `"delete"`.
    fix_last : bool, default True
        insert/delete only. `True` keeps the most recent year dated and lets the
        one-ring change roll off the inner (oldest) end; `False` keeps the inner
        end dated and changes the recent end. Matches dplR's `fix.last`.
    fix_length : bool or None, default None
        insert/delete on a **Series** only. `True` (the default for a Series, as
        dplR's `fix.length`) keeps the series' original length by dropping
        (insert) or NA-padding (delete) one ring at the end opposite `fix_last`;
        a warning names the measured ring an insert drops. For a **DataFrame**
        it is not applied, following dplR's ``rwl`` method: no measurement is
        ever discarded, the collection simply gains a year (insert) or the
        series loses one (delete), and all-NaN rows a delete leaves at either
        end are trimmed. Passing ``fix_length=True`` with a DataFrame is an
        error, as in dplR.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        a copy with the one edit applied, same type as `data`. For a DataFrame
        the edited column is realigned onto the (possibly one-year-extended)
        year index and the other series are untouched. The input's ``attrs``
        are kept (the result is still raw widths).

    Examples
    --------
    >>> import dplpy as dpl
    >>> rwl = dpl.readers("co021.rwl")
    >>> fixed = dpl.edit_rwl(rwl, series="641114", year=1900, action="insert")
    >>> fixed = dpl.edit_rwl(rwl, series="641114", year=1905, action="delete")
    >>> fixed = dpl.edit_rwl(rwl, series="641114", year=1912, action="change", value=0.44)

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/insert.ring.html
    .. [2] https://opendendro.github.io/dplPy/reference/utilities/#edit_rwl
    """
    action = _normalize_choice(action, _ACTIONS, "action")
    year = _whole_year(year)

    if isinstance(data, pd.DataFrame):
        if series is None:
            raise ValueError("series is required when data is a DataFrame")
        if series not in data.columns:
            raise ValueError("series %r not found in the collection" % series)
        if fix_length:
            raise ValueError("fix_length=True does not apply to a DataFrame: as in "
                             "dplR's rwl method no measurement is discarded -- the "
                             "collection gains or loses a year instead. Pass the "
                             "single Series to use fix_length.")
        data = _require_year_index(data, "data")
        new_span = _edit_span(data[series], year, action, value, fix_last, False)
        new_index = data.index.union(new_span.index)
        out = data.reindex(new_index).copy()
        out[series] = new_span.reindex(new_index)
        out.index.name = data.index.name
        if action == "delete":
            out = _trim_empty_ends(out)
        return _carry_attrs(data, out, kind="rwl", keep_measurement=True)
    elif isinstance(data, pd.Series):
        data = _require_year_index(data, "data")
        if fix_length is None:
            fix_length = True                      # dplR's insert.ring / delete.ring default
        new_span = _edit_span(data, year, action, value, fix_last, fix_length)
        # keep the Series' own leading/trailing NaN rows (e.g. a column taken from
        # a collection), but let a delete vacate a year at the end it shortens
        new_index = data.index.union(new_span.index)
        out = new_span.reindex(new_index)
        out.name = data.name
        out.index.name = data.index.name
        if action == "delete" and not fix_length:    # (fix_length's NA pad is deliberate)
            out = _trim_vacated_ends(data.reindex(new_index), out)
        return _carry_attrs(data, out, kind="rwl", keep_measurement=True)
    else:
        raise TypeError("Expected dataframe or series input, got "
                        + str(type(data)) + " instead.")


def _whole_year(year):
    """``year`` as an int, or a clear ValueError for None, NaN or a fraction."""
    if year is None:
        raise ValueError("year is required")
    try:
        y = float(year)
    except (TypeError, ValueError):
        raise ValueError("year must be a whole calendar year, got %r" % (year,))
    if not np.isfinite(y):
        raise ValueError("year must be a whole calendar year, got %r" % (year,))
    if y != int(y):
        raise ValueError("year must be a whole calendar year, got %r" % (year,))
    return int(y)


def _ring_value(value, what):
    """``value`` as a finite, non-negative float."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number, got %r" % (what, value))
    if not np.isfinite(v):
        raise ValueError("%s must be a finite number, got %r" % (what, value))
    if v < 0:
        raise ValueError("ring value must be non-negative")
    return v


def _trim_empty_ends(df):
    """Drop leading and trailing rows with no measurement in any series (dplR's
    rwl method trims the all-NA years a delete leaves at the ends)."""
    has = df.notna().any(axis=1).to_numpy()
    if not has.any():
        return df
    keep = np.flatnonzero(has)
    return df.iloc[keep[0]:keep[-1] + 1]


def _trim_vacated_ends(before, after):
    """Drop end rows of a Series that were measured before the edit and are NaN
    now (a year the delete vacated); rows that were NaN in the input are kept."""
    vacated = (before.notna() & after.isna()).to_numpy()
    lo, hi = 0, len(after)
    while lo < hi and vacated[lo]:
        lo += 1
    while hi > lo and vacated[hi - 1]:
        hi -= 1
    return after.iloc[lo:hi]


def _edit_span(col, year, action, value, fix_last, fix_length):
    """Apply one edit to a single series' gap-free measured span and return the
    edited span as a year-indexed Series."""
    name = col.name
    idx_name = col.index.name
    s = col.dropna()
    if s.empty:
        raise ValueError("series has no measurements to edit")
    yrs = s.index.to_numpy().astype(np.int64)
    first, last, n = int(yrs[0]), int(yrs[-1]), len(s)
    if last - first != n - 1:
        raise ValueError("series %r has interior gaps; edit_rwl needs a gap-free "
                         "measured span -- close gaps first with fill_internal"
                         % (name,))
    vals = s.to_numpy(dtype=float)

    def out_series(v, y):
        return pd.Series(np.asarray(v, dtype=float),
                         index=pd.Index(np.asarray(y, dtype=np.int64), name=idx_name),
                         name=name)

    if action == "change":
        if value is None:
            raise ValueError("action='change' requires a value")
        value = _ring_value(value, "value")
        if not (first <= year <= last):
            raise ValueError("year %d is outside the series span %d-%d"
                             % (year, first, last))
        v = vals.copy()
        v[year - first] = value
        return out_series(v, np.arange(first, last + 1))

    if action == "insert":
        value = float(np.mean(vals)) if value is None else _ring_value(value, "value")
        if year == first - 1:
            pos = 0
        elif first <= year <= last:
            pos = (year - first) + 1          # insert after this year
        else:
            raise ValueError("year %d must be within %d-%d, or %d to prepend"
                             % (year, first, last, first - 1))
        v = np.concatenate([vals[:pos], [value], vals[pos:]])
        y = np.arange(first - 1, last + 1) if fix_last else np.arange(first, last + 2)
        if fix_length:
            # dplR's fix.length: keep the original length by dropping the ring at
            # the end opposite fix_last -- a MEASURED ring, so say so (audit A16)
            dropped_year, dropped = (int(y[0]), v[0]) if fix_last else (int(y[-1]), v[-1])
            if np.isfinite(dropped):
                _warn("edit_rwl: fix_length=True dropped the measured ring %g at %d "
                      "of series %r to keep the series length; pass fix_length=False "
                      "to keep every measurement." % (dropped, dropped_year, name),
                      stacklevel=4)
            v, y = (v[1:], y[1:]) if fix_last else (v[:-1], y[:-1])
        return out_series(v, y)

    # action == "delete"
    if not (first <= year <= last):
        raise ValueError("year %d is outside the series span %d-%d"
                         % (year, first, last))
    v = np.delete(vals, year - first)
    if fix_length and fix_last:
        v = np.concatenate([[np.nan], v])
        y = np.arange(first, last + 1)
    elif fix_length and not fix_last:
        v = np.concatenate([v, [np.nan]])
        y = np.arange(first, last + 1)
    elif fix_last:
        y = np.arange(first + 1, last + 1)
    else:
        y = np.arange(first, last)
    return out_series(v, y)
