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

import warnings

import numpy as np
import pandas as pd

_ACTIONS = ("insert", "delete", "change")


def edit_rwl(data, series=None, year=None, action="insert", value=None,
             fix_last=True, fix_length=True):
    """Insert, delete, or change one ring in a single series.

    Returns a **new** object; the input is never modified in place (unlike the
    old Fortran programs, where the edit mutated the record). Assign the result
    to keep it. The fix that pairs with the crossdating checks: `xdate`,
    `series_corr`, and `check` find a misdated ring, and `edit_rwl` corrects it.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        a ring-width collection (then `series` names the column to edit) or a
        single year-indexed series.
    series : str or None
        the series (column) to edit; required when `data` is a DataFrame.
    year : int
        the calendar year to act on. For `action="insert"` it may also be the
        year just before the series (prepend).
    action : {"insert", "delete", "change"}, default "insert"
        `"insert"` adds a ring (a missed ring), `"delete"` removes one (a false
        or double-counted ring), `"change"` replaces a measured value without
        changing the series length or dating.
    value : float or None
        the ring width. Required for `"change"`; for `"insert"` it defaults to
        the series mean; ignored for `"delete"`.
    fix_last : bool, default True
        insert/delete only. `True` keeps the most recent year dated and lets the
        one-ring change roll off the inner (oldest) end; `False` keeps the inner
        end dated and changes the recent end. Matches dplR's `fix.last`.
    fix_length : bool, default True
        insert/delete only. `True` keeps the series' original length by dropping
        (insert) or NA-padding (delete) one ring at the end opposite `fix_last`.
        Matches dplR's `fix.length`.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        a copy with the one edit applied, same type as `data`. For a DataFrame
        the edited column is realigned onto the (possibly one-year-extended)
        year index and the other series are untouched.

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
    """
    action = str(action).lower()
    if action not in _ACTIONS:
        raise ValueError("action must be one of %s, got %r"
                         % (", ".join(repr(a) for a in _ACTIONS), action))
    if year is None:
        raise ValueError("year is required")
    year = int(year)

    if isinstance(data, pd.DataFrame):
        if series is None:
            raise ValueError("series is required when data is a DataFrame")
        if series not in data.columns:
            raise ValueError("series %r not found in the collection" % series)
        new_span = _edit_span(data[series], year, action, value, fix_last, fix_length)
        new_index = data.index.union(new_span.index)
        out = data.reindex(new_index).copy()
        out[series] = new_span.reindex(new_index)
        out.index.name = data.index.name
        return out
    elif isinstance(data, pd.Series):
        return _edit_span(data, year, action, value, fix_last, fix_length)
    else:
        raise TypeError("Expected dataframe or series input, got "
                        + str(type(data)) + " instead.")


def _edit_span(col, year, action, value, fix_last, fix_length):
    """Apply one edit to a single series' gap-free measured span and return the
    edited span as a year-indexed Series."""
    name = col.name
    idx_name = col.index.name
    s = col.dropna()
    if s.empty:
        raise ValueError("series has no measurements to edit")
    years = s.index.to_numpy()
    yrs = years.astype(np.int64)
    if not np.array_equal(years.astype(float), yrs.astype(float)):
        raise ValueError("series years must be whole numbers")
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
        if not (first <= year <= last):
            raise ValueError("year %d is outside the series span %d-%d"
                             % (year, first, last))
        v = vals.copy()
        v[year - first] = float(value)
        return out_series(v, np.arange(first, last + 1))

    if action == "insert":
        value = float(np.mean(vals)) if value is None else float(value)
        if value < 0:
            raise ValueError("ring value must be non-negative")
        if year == first - 1:
            pos = 0
        elif first <= year <= last:
            pos = (year - first) + 1          # insert after this year
        else:
            raise ValueError("year %d must be within %d-%d, or %d to prepend"
                             % (year, first, last, first - 1))
        v = np.concatenate([vals[:pos], [value], vals[pos:]])
        y = np.arange(first - 1, last + 1) if fix_last else np.arange(first, last + 2)
        if fix_length and fix_last:
            v, y = v[1:], y[1:]               # drop the oldest ring
        elif fix_length and not fix_last:
            v, y = v[:-1], y[:-1]             # drop the most recent ring
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
