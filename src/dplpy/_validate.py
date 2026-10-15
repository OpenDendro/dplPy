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

# Title: _validate.py
# Project: OpenDendro dplPy
# Description: Small shared input-validation helpers, so the many public
#              functions that require a ring-width DataFrame -- or that accept a
#              DataFrame or a path to a file -- do it one consistent way with one
#              consistent error message, instead of each re-implementing the
#              check with slightly different wording.

import warnings

import pandas as pd


def _require_dataframe(x):
    """Return ``x`` unchanged if it is a pandas DataFrame, else raise TypeError.

    The single home for the ``isinstance(x, pd.DataFrame)`` guard that opens most
    of dplPy's public functions.
    """
    if not isinstance(x, pd.DataFrame):
        raise TypeError("Expected dataframe input, got " + str(type(x)) + " instead.")
    return x


def _coerce_to_frame(inp):
    """Return ``inp`` as a DataFrame.

    A DataFrame is returned as-is; a string is treated as a path (or URL) and
    read with ``readers()``; anything else raises TypeError. Used by the summary
    functions (summary, stats, report, plot) that accept either a frame or a file
    path.
    """
    if isinstance(inp, pd.DataFrame):
        return inp
    if isinstance(inp, str):
        from .readers import readers          # local import avoids an import cycle
        return readers(inp)
    raise TypeError("Input must be a pandas DataFrame or a path (str) to a "
                    "CSV/RWL file, not " + str(type(inp)) + ".")


# Canonical correlation-method names; the single home for the case-insensitive
# alias table the crossdating / rbar functions share.
_CORR_ALIASES = {"spearman": "spearman", "pearson": "pearson", "kendall": "kendall"}


def _normalize_corr(corr, allowed=("spearman", "pearson", "kendall")):
    """Return the canonical (lower-case) correlation-method name for ``corr``.

    Case-insensitive, so "Spearman", "spearman" and "SPEARMAN" are all accepted.
    ``allowed`` restricts the methods a given caller supports -- xdate/series_corr
    accept all three, while interseries_corr/rwi_stats support only spearman and
    pearson. Raises ValueError if ``corr`` is unknown or outside ``allowed``.
    """
    method = _CORR_ALIASES.get(str(corr).strip().lower())
    if method is None or method not in allowed:
        raise ValueError("corr must be one of "
                         + " / ".join(m.capitalize() for m in allowed)
                         + ", got '" + str(corr) + "'.")
    return method


def _plot_kwarg(plot, kwargs):
    """Resolve the plot flag, accepting the deprecated ``make_plot=`` alias.

    dplPy standardized every public function's plotting switch on ``plot=`` in
    0.8.0. ``make_plot=`` is kept for one release as a deprecated alias: if it is
    present it emits a ``DeprecationWarning`` and its value is used (a user who
    passes the old name clearly wants that value). Any other stray keyword is a
    ``TypeError``, exactly as it would be without ``**kwargs``.
    """
    if "make_plot" in kwargs:
        plot = kwargs.pop("make_plot")
        warnings.warn(
            "make_plot= is deprecated and will be removed in a future release; "
            "use plot= instead.", DeprecationWarning, stacklevel=3)
    if kwargs:
        raise TypeError("unexpected keyword argument(s): "
                        + ", ".join(repr(k) for k in sorted(kwargs)))
    return plot


# --------------------------------------------------------------------------- #
# v0.9.0 shared helpers (audit follow-ups). One home each, so the many public
# functions that need the same check or policy do it one way.
# --------------------------------------------------------------------------- #
import numpy as np


def _warn(msg, category=UserWarning, stacklevel=3):
    """warnings.warn that points at the USER's call, not a dplPy internal line.

    Default stacklevel=3 is right for a warning raised directly inside a public
    function (public function -> _warn). A helper called by a public function
    should pass stacklevel=4, and so on. (Audit G1: 60 of 62 warnings pointed
    at library internals.)
    """
    warnings.warn(msg, category, stacklevel=stacklevel)


def _alias_kwarg(kwargs, old, new, current, func):
    """Honour a deprecated keyword ``old`` for parameter ``new``.

    If ``old`` is in ``kwargs`` it is popped, a DeprecationWarning names the
    function and the replacement, and its value is returned (a caller who
    passes the old name clearly wants that value). Otherwise ``current`` (the
    value bound to the new name) is returned. Call once per renamed parameter;
    then call ``_reject_kwargs(kwargs)`` so stray names still raise TypeError.
    """
    if old in kwargs:
        val = kwargs.pop(old)
        warnings.warn(
            func + "(): " + old + "= is deprecated and will be removed in a "
            "future release; use " + new + "= instead.",
            DeprecationWarning, stacklevel=3)
        return val
    return current


def _reject_kwargs(kwargs):
    """Raise TypeError for any keyword left in ``kwargs`` after aliases are
    consumed, exactly as a function without ``**kwargs`` would."""
    if kwargs:
        raise TypeError("unexpected keyword argument(s): "
                        + ", ".join(repr(k) for k in sorted(kwargs)))


def _normalize_choice(value, allowed, param, none_ok=False):
    """Case-insensitive match of a string option against ``allowed``.

    Returns the canonical spelling from ``allowed``. Raises ValueError naming
    the parameter and the accepted values otherwise. ``none_ok`` lets None pass
    through. (Audit D11: option strings were case-sensitive in a dozen places.)
    """
    if value is None and none_ok:
        return None
    key = str(value).strip().lower()
    for a in allowed:
        if str(a).lower() == key:
            return a
    raise ValueError(param + " must be one of "
                     + ", ".join(repr(a) for a in allowed)
                     + ", got " + repr(value) + ".")


def _require_year_index(df, param="data", require_contiguous=False):
    """Validate (and tidy) the year index of a year-indexed frame or series.

    - a float index of whole numbers is cast to int;
    - a non-integer index raises ValueError;
    - duplicate years raise ValueError;
    - an unsorted index is SORTED with a warning (readers always sort; concat
      and slicing can unsort, and several functions silently return wrong
      numbers on an unsorted index -- audit A4);
    - with ``require_contiguous=True`` a gap in the years raises ValueError
      (audit B5: the Tucson writer needs a dense grid).
    Returns the (possibly re-sorted, re-cast) object; the input is not modified.
    """
    idx = df.index
    if not (pd.api.types.is_integer_dtype(idx) or pd.api.types.is_float_dtype(idx)):
        raise ValueError(param + " must have an integer year index, got "
                         + str(idx.dtype) + ".")
    if pd.api.types.is_float_dtype(idx):
        arr = np.asarray(idx, dtype=float)
        if np.isnan(arr).any() or not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(param + " year index must be whole numbers.")
        df = df.copy()
        df.index = pd.Index(arr.astype(int), name=idx.name)
        idx = df.index
    if idx.has_duplicates:
        dups = sorted(set(idx[idx.duplicated()]))[:5]
        raise ValueError(param + " has duplicate years (e.g. " + str(dups) + ").")
    if not idx.is_monotonic_increasing:
        _warn(param + " year index was not sorted; sorting it (results on an "
              "unsorted index are undefined).", stacklevel=4)
        df = df.sort_index()
        idx = df.index
    if require_contiguous:
        yrs = np.asarray(idx)
        if len(yrs) > 1 and not np.all(np.diff(yrs) == 1):
            gaps = int(((np.diff(yrs) != 1)).sum())
            raise ValueError(param + " year index must be contiguous (one row per "
                             "year); found " + str(gaps) + " gap(s). Reindex to a "
                             "full year range first (missing years as NaN).")
    return df


def _drop_allnan_columns(df, param="data", stacklevel=4):
    """Drop columns with no measured values, warning which were dropped.

    An all-NaN column (common after a year slice) crashed six functions with
    cryptic errors (audit B1). If EVERY column is all-NaN, raise ValueError.
    Returns a new frame; the input is not modified.
    """
    empty = [c for c in df.columns if df[c].isna().all()]
    if not empty:
        return df
    if len(empty) == len(df.columns):
        raise ValueError(param + ": every series is entirely NaN; nothing to "
                         "compute.")
    _warn(param + ": dropped " + str(len(empty)) + " series with no measured "
          "values (all NaN): " + ", ".join(str(c) for c in empty[:8])
          + (" ..." if len(empty) > 8 else ""), stacklevel=stacklevel)
    return df.drop(columns=empty)


def _series_table(x, param, value_col=None):
    """Coerce a per-series side table (pith offsets, diameters, tree ids...) to
    a pandas Series indexed by series name.

    Accepts: a DataFrame indexed by series name (one value column, or
    ``value_col``); a DataFrame with a ``series`` column; a dict; a Series.
    (Audit D15: po/diam/d2pith/ids each accepted a different subset.)
    """
    if isinstance(x, pd.Series):
        s = x
    elif isinstance(x, dict):
        s = pd.Series(x)
    elif isinstance(x, pd.DataFrame):
        df = x
        if "series" in df.columns:
            df = df.set_index("series")
        if value_col is not None:
            if value_col not in df.columns:
                raise ValueError(param + " must have a '" + str(value_col)
                                 + "' column; has " + str(list(df.columns)) + ".")
            s = df[value_col]
        elif df.shape[1] == 1:
            s = df.iloc[:, 0]
        else:
            raise ValueError(param + ": could not tell which column holds the "
                             "values; columns are " + str(list(df.columns))
                             + ". Pass a single-column table or a dict.")
    else:
        raise TypeError(param + " must be a DataFrame (indexed by series or with "
                        "a 'series' column), a dict, or a Series, not "
                        + str(type(x)) + ".")
    s = s.copy()
    s.index = s.index.astype(str)
    s.index.name = "series"
    return s


# attrs keys that describe RAW MEASUREMENTS and must not survive a transform
# into indices, chronologies, or other units (audit G10).
_MEASUREMENT_ATTRS = ("dplpy_precision", "dplpy_mixed_precision")


def _carry_attrs(src, out, kind=None, keep_measurement=False):
    """Copy provenance attrs from ``src`` to ``out`` deliberately.

    pandas drops ``attrs`` on most operations, so provenance propagation was
    accidental: some transforms kept everything (including a now-false
    precision on indices), others dropped it all. This copies metadata,
    encoding, salvage and reader diagnostics, removes the measurement-only keys
    unless ``keep_measurement`` (the output is still raw widths), and sets
    ``out.attrs["kind"]`` when given ("rwl", "rwi", "crn", "bai", ...).
    Returns ``out``.
    """
    if src is None or not hasattr(src, "attrs"):
        return out
    for k, v in src.attrs.items():
        if not keep_measurement and k in _MEASUREMENT_ATTRS:
            continue
        if k == "kind":
            continue
        out.attrs[k] = v
    if kind is not None:
        out.attrs["kind"] = kind
    return out
