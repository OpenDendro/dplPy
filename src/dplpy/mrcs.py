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

# Title: mrcs.py
# Project: OpenDendro dplPy
# Description: Multiple-curve Regional Curve Standardisation, a port of CRUST's
#              multiple-RCS option (Melvin & Briffa 2014, Part 2). Trees are
#              sorted by relative growth rate (fast vs slow), split into a few
#              equal-count classes, and each class is standardised by its own
#              regional curve -- correcting the slope ("modern sample bias") that
#              a single average curve injects when the sample's growth-rate mix
#              changes over time ("trees that grow fast die young"). Multiple RCS
#              also removes some genuine low-frequency variance; the mean-reset
#              option (on by default) reinstates each tree's one-curve mean to
#              recover it. Each class curve is built by the single-curve rcs(),
#              so mrcs() shares rcs()'s full fit/method/period/preset vocabulary
#              and mrcs(n_curves=1) is identical to rcs().

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._validate import (_require_dataframe, _plot_kwarg, _warn, _alias_kwarg,
                        _normalize_choice, _carry_attrs)
from ._plot_style import style_axes, finalize_font
from .detrend import _normalize_method, _finish_figure
from .rcs import rcs, _resolve_po, _align, _reset_means

_MIN_TREES_PER_CURVE = 40   # CRUST rule of thumb for a reliable RCS curve
_SORTS = ("growth", "age", "diameter", "none")


def _sort_key(rwl, po_arr, sort):
    """Per-series sort key for curve allocation (ascending).

    "growth"   : CRUST growth rate -- the tree's total measured radius (plus the
                 regional-curve-estimated radius of any missing pith rings)
                 divided by the UNSMOOTHED single regional curve's cumulative
                 radius at the tree's full cambial age. >1 = faster than average.
    "age"      : cambial age = pith offset + number of rings (span, gaps included).
    "diameter" : total measured radius (sum of ring widths + estimated pith).
    "none"     : input order.
    """
    M = rwl.to_numpy(dtype=float)
    ncol = M.shape[1]
    offs = np.asarray(po_arr, dtype=int)

    if sort == "none":
        return np.arange(ncol, dtype=float)

    # the unsmoothed single regional curve (arithmetic mean by cambial age) and
    # its cumulative radius -- shared by "growth" and the pith estimate.
    rwca = _align(M, offs)
    ca_n = np.sum(~np.isnan(rwca), axis=1)
    with np.errstate(invalid="ignore"):
        ca_mean = np.where(ca_n > 0, np.nan_to_num(np.nanmean(rwca, axis=1)), 0.0)
    cum = np.cumsum(ca_mean)                       # expected radius by cambial age

    ages = np.array([int(np.where(~np.isnan(rwca[:, i]))[0].max()) + 1
                     if np.any(~np.isnan(rwca[:, i])) else offs[i]
                     for i in range(ncol)])        # full cambial age of each tree
    if sort == "age":
        return ages.astype(float)

    radius = np.empty(ncol)
    age_rad = np.empty(ncol)
    for i in range(ncol):
        pith = cum[offs[i] - 1] if offs[i] > 0 else 0.0   # est. missing-pith radius
        radius[i] = np.nansum(M[:, i]) + pith
        age_rad[i] = max(cum[min(ages[i] - 1, len(cum) - 1)], 1e-9)
    if sort == "diameter":
        return radius
    # "growth": radius relative to the single-curve expectation at that age
    return radius / age_rad


def _allocate(rwl, po_arr, n_curves, sort, label="mrcs"):
    """Rank trees by `sort` and split into `n_curves` near-equal classes.
    Returns (alloc, groups): ``alloc[col] = curve index``, and ``groups`` the
    list of column-index arrays. Shared by mrcs() and sfmrcs()."""
    cols = list(rwl.columns)
    key = _sort_key(rwl, po_arr, sort)
    order = np.argsort(key, kind="stable")
    groups = [g for g in np.array_split(order, n_curves) if len(g) > 0]
    if len(groups) < n_curves:
        _warn("%s: only %d non-empty curve(s) for n_curves=%d (too few series)."
              % (label, len(groups), n_curves), stacklevel=4)
    alloc = np.zeros(len(cols), dtype=int)
    for gi, g in enumerate(groups):
        alloc[g] = gi
        if len(g) < _MIN_TREES_PER_CURVE:
            _warn("%s: curve %d has %d trees (< %d); multiple RCS is unreliable "
                  "with sparsely-replicated curves."
                  % (label, gi, len(g), _MIN_TREES_PER_CURVE), stacklevel=4)
    return alloc, groups


def mrcs(rwl: pd.DataFrame, po=None, n_curves=2, sort="growth",
         fit="AgeDepSpline", method="ratio", spline_period=0.1, nyrs0=50, f=0.5,
         pos_slope=False, biweight=True, min_n=None, preset=None,
         reset_means=True, rescale_difference=False, rc_out=False,
         plot=True, **kwargs):
    """Multiple-curve Regional Curve Standardisation.

    Sort trees by relative growth rate, split them into ``n_curves`` equal-count
    classes, build a regional curve for each class, and standardise each tree by its
    own class curve. This removes the modern-sample-bias slope that a single
    regional curve injects when the sample's growth-rate (or age) mix changes over
    time. Each class curve is built by `rcs()`, so ``mrcs`` shares its
    ``fit``/``method``/``spline_period``/``nyrs0``/``f``/``pos_slope``/``preset``
    vocabulary, and ``mrcs(rwl, n_curves=1)`` is identical to ``rcs(rwl)``. Port of
    CRUST's multiple RCS.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame, dict, Series or None, default None
        pith offsets (``series`` and ``pith_offset``; a true offset, 0 = pith
        present), as in `rcs()`. None assumes 0 for every series.
    n_curves : int, default 2
        number of regional curves. ``1`` reduces exactly to `rcs()`.
    sort : {"growth", "age", "diameter", "none"}, default "growth"
        how trees are ranked before the equal-count split: "growth" by relative
        growth rate, "age" by cambial age, "diameter" by total radius, "none" keeps
        input order. Case-insensitive.
    fit, method, spline_period, nyrs0, f, pos_slope, biweight, min_n, preset,
    rescale_difference :
        passed through to `rcs()` to build each class curve; see `rcs()`.
        Defaults match `rcs()`. The former name ``period=`` is accepted with a
        DeprecationWarning.
    reset_means : bool, default True
        if True (ratios only), rescale each tree's indices to the mean it would have
        had under one-curve RCS, recovering the low-frequency variance multiple RCS
        removes. Skipped with a warning for ``method="difference"``.
    rc_out : bool, default False
        if True, return a dict ``{"rwi", "rc", "alloc"}`` -- the indices, the list of
        per-class regional curves (by cambial age), and the series->class map.
    plot : bool, default True
        draw the per-class regional-curve figure. ``make_plot=`` is accepted as a
        deprecated alias.

    Returns
    -------
    pandas.DataFrame or dict
        the detrended ``rwi`` (calendar-year indexed), or ``{"rwi", "rc", "alloc"}``
        when ``rc_out`` is True.

    References
    ----------
    .. [1] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further
       RCS options and recommendations. Dendrochronologia, 32, 343-356.
    """
    _require_dataframe(rwl)
    spline_period = _alias_kwarg(kwargs, "period", "spline_period", spline_period, "mrcs")
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)
    sort = _normalize_choice(sort, _SORTS, "sort")
    n_curves = int(n_curves)
    if n_curves < 1:
        raise ValueError("n_curves must be >= 1, got %d." % n_curves)

    cols = list(rwl.columns)
    common = dict(fit=fit, method=method, spline_period=spline_period, nyrs0=nyrs0,
                  f=f, pos_slope=pos_slope, biweight=biweight, min_n=min_n,
                  preset=preset, rescale_difference=rescale_difference)

    # n_curves == 1 is exactly single-curve RCS
    if n_curves == 1:
        res = rcs(rwl, po=po, plot=plot, rc_out=True, **common)
        rwi = res["rwi"]
        if rc_out:
            return {"rwi": rwi, "rc": [res["rc"]], "alloc": {c: 0 for c in cols}}
        return rwi

    po_arr = _resolve_po(rwl, po, label="mrcs")
    po_tab = pd.DataFrame({"series": cols, "pith_offset": po_arr})

    # 1. rank trees and split into n_curves near-equal classes (shared helper)
    alloc_arr, groups = _allocate(rwl, po_arr, n_curves, sort, label="mrcs")

    # 2. standardise each class by its own regional curve (via single-curve rcs)
    idx = np.full(rwl.shape, np.nan)
    rc_list = []
    alloc = {}
    for gi, g in enumerate(groups):
        gcols = [cols[i] for i in g]
        res = rcs(rwl[gcols], po=po_tab.iloc[g], plot=False, rc_out=True, **common)
        idx[:, g] = res["rwi"].to_numpy(dtype=float)
        rc_list.append(res["rc"])
        for c in gcols:
            alloc[c] = gi

    # 3. mean-reset: put each tree back on its one-curve mean (ratios only)
    if reset_means:
        if method != "ratio":
            _warn("mrcs: reset_means is only applied with method='ratio'; "
                  "skipped for method='difference'.")
        else:
            single = rcs(rwl, po=po_tab, plot=False, **common)
            idx = _reset_means(idx, single.to_numpy(dtype=float))

    rwi = pd.DataFrame(idx, index=rwl.index, columns=cols)
    _carry_attrs(rwl, rwi, kind="rwi")

    if plot:
        _plot_mrcs(rc_list)

    if rc_out:
        return {"rwi": rwi, "rc": rc_list, "alloc": alloc}
    return rwi


def _plot_mrcs(rc_list):
    """Overlay the per-class regional curves vs cambial age."""
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(rc_list)
    cmap = plt.get_cmap("viridis")
    for i, rc in enumerate(rc_list):
        x = np.arange(1, len(rc) + 1)
        ax.plot(x, rc, lw=2, color=cmap(i / max(n - 1, 1)),
                label="Curve %d" % i)
    ax.set_xlabel("Cambial age (years)")
    ax.set_ylabel("Ring width (mm)")
    ax.set_title("Multiple-curve RCS: regional curves by growth class", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axes(ax, xgrid=True, ygrid=True)
    fig.tight_layout()
    finalize_font(fig)
    _finish_figure(fig)
    return ax
