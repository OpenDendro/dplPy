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

# Title: bai.py
# Description: Convert ring widths to basal area increment (BAI) -- the annual
#   cross-sectional area of wood added each year, BAI_t = pi * (R_t^2 - R_{t-1}^2),
#   where R_t is the stem radius at the end of year t. BAI removes much of the
#   geometric age/size trend that ring width carries (a constant ring width means
#   an ever-larger annual area on a widening stem), and is common in forest-growth
#   and ecological studies. Ports dplR's bai.out() and bai.in().
#
#   bai_out works from the OUTSIDE in: the outermost measured ring sits at the
#   stem radius (half the diameter, or the summed ring widths if no diameter is
#   given), and radii decrease inward. bai_in works from the pith OUT: the radius
#   accumulates from an optional pith offset (d2pith) plus the cumulative ring
#   widths.
#
# example usage:
#   >>> import dplpy as dpl
#   >>> rwl = dpl.readers("gp.rwl")
#   >>> bai = dpl.bai_out(rwl)                       # radius = sum of ring widths
#   >>> bai = dpl.bai_out(rwl, diam=diam_df)         # measured stem diameters
#   >>> bai = dpl.bai_in(rwl, d2pith=d2pith_df)      # distance-to-pith offsets

import numpy as np
import pandas as pd

from ._validate import _require_dataframe


def _resolve_per_series(param, rwl, value_name):
    """Resolve a per-series scalar (diam or d2pith) to a {series: float} map,
    validated against the rwl columns. Accepts None, a dict / pandas Series
    (series -> value), or a DataFrame whose first column (or a 'series' column)
    holds names and whose ``value_name`` column (or second column) holds values --
    matching dplR's two-column data.frame."""
    if param is None:
        return None
    cols = [str(c) for c in rwl.columns]
    if isinstance(param, pd.DataFrame):
        names = param["series"] if "series" in param.columns else param.iloc[:, 0]
        if value_name in param.columns:
            values = param[value_name]
        elif param.shape[1] >= 2:
            values = param.iloc[:, 1]
        else:
            raise ValueError("'%s' DataFrame needs a value column." % value_name)
        mapping = {str(a): float(b) for a, b in zip(names, values)}
    elif isinstance(param, (dict, pd.Series)):
        items = param.items()
        mapping = {str(a): float(b) for a, b in items}
    else:
        raise TypeError("'%s' must be None, a dict/Series, or a DataFrame, not %s."
                        % (value_name, type(param)))
    missing = [c for c in cols if c not in mapping]
    if missing:
        raise ValueError("'%s' is missing values for series: %s"
                         % (value_name, ", ".join(missing)))
    return mapping


def bai_out(rwl: pd.DataFrame, diam=None) -> pd.DataFrame:
    """Basal area increment computed from the outside in (dplR ``bai.out``).

    Parameters
    ----------
    rwl : pandas.DataFrame
        Year-indexed ring widths (one column per series).
    diam : DataFrame, dict, or pandas.Series, optional
        Stem diameter for each series, in the same length units as the ring
        widths. A DataFrame uses a ``series`` column (or the first column) for the
        names and a ``diam`` column (or the second column) for the diameters; a
        dict/Series maps series name -> diameter. If omitted, each series' radius
        is taken as the sum of its ring widths (diameter = twice that).

    Returns
    -------
    pandas.DataFrame
        BAI in squared length units, same shape/index as ``rwl`` (NaN where ring
        width was missing).
    """
    _require_dataframe(rwl)
    dmap = _resolve_per_series(diam, rwl, "diam")
    out = rwl.copy()
    for col in rwl.columns:
        dat = rwl[col].dropna()
        vals = dat.to_numpy(dtype=float)
        if vals.size == 0:
            continue
        d = dmap[str(col)] if dmap is not None else vals.sum() * 2.0
        # radii from the outer edge inward: d/2, d/2 - rw_outer, ...
        r0 = d / 2.0 - np.concatenate(([0.0], np.cumsum(vals[::-1])))
        bai = -np.pi * np.diff(r0 * r0)[::-1]
        out.loc[dat.index, col] = bai
    return out


def bai_in(rwl: pd.DataFrame, d2pith=None) -> pd.DataFrame:
    """Basal area increment computed from the pith out (dplR ``bai.in``).

    Parameters
    ----------
    rwl : pandas.DataFrame
        Year-indexed ring widths (one column per series).
    d2pith : DataFrame, dict, or pandas.Series, optional
        Distance from the innermost measured ring to the pith, per series, in the
        same length units as the ring widths. A DataFrame uses a ``series`` column
        (or the first column) for the names and a ``d2pith`` column (or the second
        column) for the offsets; a dict/Series maps series name -> offset. If
        omitted, every offset is 0 (the innermost ring is assumed to reach the
        pith).

    Returns
    -------
    pandas.DataFrame
        BAI in squared length units, same shape/index as ``rwl`` (NaN where ring
        width was missing).
    """
    _require_dataframe(rwl)
    dmap = _resolve_per_series(d2pith, rwl, "d2pith")
    out = rwl.copy()
    for col in rwl.columns:
        dat = rwl[col].dropna()
        vals = dat.to_numpy(dtype=float)
        if vals.size == 0:
            continue
        dp = dmap[str(col)] if dmap is not None else 0.0
        cum = np.cumsum(vals)
        # BAI_t = pi (R_t^2 - R_{t-1}^2) = pi * rw * (R_t + R_{t-1}),
        # with R_t = d2pith + cum, R_{t-1} = d2pith + cum - rw
        bai = np.pi * vals * (vals + 2.0 * (cum + dp - vals))
        out.loc[dat.index, col] = bai
    return out
