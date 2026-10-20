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

# Title: pith.py
# Project: OpenDendro dplPy
# Description: Pith-offset helpers, ports of dplR's po.to.wc() and wc.to.po().
#              A pith offset is the estimated number of rings from the innermost
#              measured ring to the pith (a true OFFSET: 0 means the pith is
#              present at the first measured ring). It is what age-based Regional
#              Curve Standardisation needs
#              to align every series on a common cambial-age axis. These two
#              functions convert between the pith-offset representation and the
#              (TRIDAS-style) wood-completeness representation.

import numpy as np
import pandas as pd
from ._validate import _require_dataframe, _normalize_choice, _warn


def po_to_wc(po: pd.DataFrame) -> pd.DataFrame:
    """Convert pith offset to (partial) wood completeness (dplR's po.to.wc()).

    The number of missing heartwood rings equals the pith offset (dplPy's
    pith offset is itself the count of rings between pith and first measured
    ring; 0 = pith present).

    Parameters
    ----------
    po : pandas.DataFrame
        pith-offset table with a ``series`` column (series IDs) and a
        ``pith_offset`` column (integer rings between the pith and the first
        measured ring; 0 = pith present).

    Returns
    -------
    pandas.DataFrame
        one column, ``n_missing_heartwood`` (integer), indexed by series.

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/po.to.wc.html
    """
    _require_dataframe(po)
    series = po["series"] if "series" in po.columns else po.iloc[:, 0]
    pith_offset = po["pith_offset"] if "pith_offset" in po.columns else po.iloc[:, 1]
    return pd.DataFrame(
        {"n_missing_heartwood": pd.array(np.asarray(pith_offset), dtype="Int64")},
        index=pd.Index(np.asarray(series), name="series"),
    )


def wc_to_po(wc: pd.DataFrame) -> pd.DataFrame:
    """Convert wood completeness to pith offset (dplR's wc.to.po()).

    The pith offset is the number of missing heartwood rings plus the number of
    unmeasured inner rings (0 = pith present), for every series whose pith
    presence is known (``complete``/``incomplete``) or whose missing-heartwood
    count is given; other series get a missing pith offset.

    Parameters
    ----------
    wc : pandas.DataFrame
        wood-completeness table indexed by series, with any of the optional
        columns ``pith_presence`` (``"complete"``/``"incomplete"``),
        ``n_missing_heartwood`` (int) and ``n_unmeasured_inner`` (int).

    Returns
    -------
    pandas.DataFrame
        a pith-offset table with ``series`` and ``pith_offset`` columns
        (nullable integer; 0 = pith present), suitable as input to RCS.

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/wc.to.po.html
    """
    _require_dataframe(wc)
    n = len(wc)
    nan = pd.Series(np.full(n, np.nan))
    pith = wc["pith_presence"].reset_index(drop=True) if "pith_presence" in wc.columns else nan
    missing = wc["n_missing_heartwood"].reset_index(drop=True) if "n_missing_heartwood" in wc.columns else nan
    unmeasured = wc["n_unmeasured_inner"].reset_index(drop=True) if "n_unmeasured_inner" in wc.columns else nan

    known = pith.isin(["complete", "incomplete"]) | missing.notna()
    # rowSums(cbind(missing, unmeasured), na.rm=TRUE), only for known rows;
    # dplPy's offset is 0 when the pith is present (no +1, unlike dplR).
    total = missing.fillna(0).to_numpy() + unmeasured.fillna(0).to_numpy()
    pith_offset = pd.array([pd.NA] * n, dtype="Int64")
    pith_offset[known.to_numpy()] = np.asarray(total[known.to_numpy()], dtype="int64")

    return pd.DataFrame({"series": np.asarray(wc.index), "pith_offset": pith_offset})


def yrs_to_pith(rwl, d2pith, method="dist", n_rings=10):
    """Estimate years (rings) to pith from an estimated distance to pith.

    Ports modendro's ``yrs_to_pith()`` (C. Maher). The number of unmeasured
    rings between the pith and the innermost measured ring is estimated as the
    distance to pith divided by a mean innermost ring width, rounded to a whole
    number of rings. The distance to pith is whatever you measured (e.g. in
    CooRecorder, as returned by ``dpl.read_pos().attributes``); only the table
    format matters.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring widths, years as the index and series IDs as columns.
    d2pith : pandas.DataFrame
        one row per series, with a ``series`` column and a distance-to-pith
        column named ``d2pith`` or ``distance_to_pith`` (same units as the ring
        widths, typically mm).
    method : {"dist", "rings"}, default "dist"
        ``"rings"`` divides the distance by the mean of the innermost ``n_rings``
        ring widths. ``"dist"`` divides by the mean of as many innermost rings as
        fit within the distance (at least ``n_rings``), which tracks a changing
        growth rate near the pith better.
    n_rings : int, default 10
        number of innermost rings to average (``"rings"``), or the floor on the
        number averaged (``"dist"``).

    Returns
    -------
    pandas.DataFrame
        columns ``series``, ``d2pith``, ``mean_rw``, and ``pith_offset`` (the
        estimated whole-ring offset, 0 = pith present), ready to pass to
        ``dpl.rcs``/``dpl.po_to_wc`` as a pith-offset table.
    """
    _require_dataframe(rwl)
    _require_dataframe(d2pith)
    method = _normalize_choice(method, ("dist", "rings"), "method")
    if "series" not in d2pith.columns:
        raise ValueError("d2pith must have a 'series' column.")
    dcol = ("d2pith" if "d2pith" in d2pith.columns
            else "distance_to_pith" if "distance_to_pith" in d2pith.columns else None)
    if dcol is None:
        raise ValueError("d2pith must have a 'd2pith' (or 'distance_to_pith') column.")
    dmap = dict(zip(d2pith["series"].astype(str),
                    pd.to_numeric(d2pith[dcol], errors="coerce")))

    rows = []
    for col in rwl.columns:
        name = str(col)
        s = pd.to_numeric(rwl[col], errors="coerce").dropna().sort_index()
        d = dmap.get(name, np.nan)
        inner = s.to_numpy()                         # sorted by year = innermost first
        if inner.size == 0 or not np.isfinite(d):
            if name not in dmap:
                _warn("yrs_to_pith: no distance to pith for series '%s'." % name)
            rows.append({"series": name, "d2pith": d, "mean_rw": np.nan,
                         "pith_offset": pd.NA})
            continue
        if method == "rings":
            used = inner[:n_rings]
        else:
            cum = np.cumsum(inner)
            used = inner[cum < d]
            if used.size < n_rings:                  # too few within the distance
                used = inner[:n_rings]
        mean_rw = float(np.nanmean(used)) if used.size else np.nan
        po = int(round(d / mean_rw)) if (mean_rw and np.isfinite(mean_rw) and mean_rw > 0) else pd.NA
        rows.append({"series": name, "d2pith": float(d), "mean_rw": mean_rw,
                     "pith_offset": po})
    out = pd.DataFrame(rows)
    out["pith_offset"] = out["pith_offset"].astype("Int64")
    return out
