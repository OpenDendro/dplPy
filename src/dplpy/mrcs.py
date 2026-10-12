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

import warnings

import numpy as np
import pandas as pd

from ._validate import _require_dataframe, _plot_kwarg
from .detrend import _normalize_method
from .rcs import rcs

_MIN_TREES_PER_CURVE = 40   # CRUST rule of thumb for a reliable RCS curve


def _po_map(rwl, po):
    """Pith-offset lookup (true offset; 0 = pith present), as in rcs()."""
    cols = list(rwl.columns)
    if po is None:
        return {c: 0 for c in cols}
    s = po["series"] if "series" in po.columns else po.iloc[:, 0]
    pv = po["pith_offset"] if "pith_offset" in po.columns else po.iloc[:, 1]
    return dict(zip(np.asarray(s), np.asarray(pv)))


def _sort_key(rwl, po_map, sort):
    """Per-series sort key for curve allocation (ascending).

    "growth"   : CRUST growth rate -- the tree's total measured radius (plus the
                 regional-curve-estimated radius of any missing pith rings)
                 divided by the UNSMOOTHED single regional curve's cumulative
                 radius at the tree's full cambial age. >1 = faster than average.
    "age"      : cambial age = pith offset + number of measured rings.
    "diameter" : total measured radius (sum of ring widths + estimated pith).
    "none"     : input order.
    """
    cols = list(rwl.columns)
    M = rwl.to_numpy(dtype=float)
    counts = np.sum(~np.isnan(M), axis=0)
    offs = np.array([int(po_map[c]) for c in cols])

    if sort == "none":
        return np.arange(len(cols), dtype=float)
    if sort == "age":
        return (offs + counts).astype(float)

    # build the unsmoothed single regional curve (arithmetic mean by cambial age)
    # and its cumulative radius -- shared by "growth" and the pith estimate.
    nrow = int(counts.max()) + int(offs.max())
    rwca = np.full((nrow, len(cols)), np.nan)
    for i in range(len(cols)):
        vals = M[:, i][~np.isnan(M[:, i])]
        rwca[offs[i]:offs[i] + len(vals), i] = vals
    with np.errstate(invalid="ignore"):
        ca_mean = np.nanmean(rwca, axis=1)
    ca_n = np.sum(~np.isnan(rwca), axis=1)
    ca_mean = np.where(ca_n > 0, np.nan_to_num(ca_mean), 0.0)
    cum = np.cumsum(ca_mean)                       # expected radius by cambial age

    radius = np.empty(len(cols))
    age_rad = np.empty(len(cols))
    for i in range(len(cols)):
        vals = M[:, i][~np.isnan(M[:, i])]
        pith = cum[offs[i] - 1] if offs[i] > 0 else 0.0   # est. missing-pith radius
        radius[i] = vals.sum() + pith
        age_idx = min(offs[i] + len(vals) - 1, len(cum) - 1)
        age_rad[i] = max(cum[age_idx], 1e-9)
    if sort == "diameter":
        return radius
    # "growth": radius relative to the single-curve expectation at that age
    return radius / age_rad


def mrcs(rwl: pd.DataFrame, po=None, n_curves=2, sort="growth",
          fit="AgeDepSpline", method="ratio", period=0.1, nyrs0=50, f=0.5,
          pos_slope=False, biweight=True, min_n=None, preset=None,
          reset_means=True, rescale_difference=False, rc_out=False,
          plot=True, **kwargs):
    """Multiple-curve Regional Curve Standardisation (CRUST multiple RCS).

    Extended Summary
    ----------------
    Single-curve RCS (:func:`rcs`) assumes every tree shares one age->growth
    expectation. Because "trees that grow fast die young", fast growers acquire a
    negative slope in their one-curve indices and slow growers a positive slope,
    so one average regional curve injects a systematic slope ("modern sample
    bias") whenever the sample's growth-rate (or age) mix changes over calendar
    time. ``mrcs`` sorts the trees by relative growth rate, splits them into
    ``n_curves`` equal-count classes, builds a separate regional curve for each
    class, and standardises each tree by its own class curve. A port of CRUST's
    multiple-RCS option (Melvin & Briffa 2014, Part 2).

    Multiple RCS corrects the slope bias but also removes some genuine
    low-frequency variance (on timescales beyond the mean tree length; almost all
    of the loss occurs going from one to two curves). ``reset_means`` (default
    ``True``) rescales each tree's indices back to the mean it would have had under
    one-curve RCS, reinstating the long-timescale amplitude while keeping the
    slope correction -- CRUST's recommended compromise. It requires
    ``method="ratio"`` and is skipped (with a warning) for ``method="difference"``.

    Each class curve is built by the single-curve :func:`rcs`, so ``mrcs`` shares
    its full ``fit``/``method``/``period``/``nyrs0``/``f``/``pos_slope``/``preset``
    vocabulary, and ``mrcs(rwl, n_curves=1)`` is identical to ``rcs(rwl)``.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame or None, default None
        pith offsets (columns ``series`` and ``pith_offset``; a true offset,
        0 = pith present), as in :func:`rcs`. None assumes 0 for every series.
    n_curves : int, default 2
        number of regional curves (CRUST ``srcno``). ``1`` reduces exactly to
        :func:`rcs`. Most of the low-frequency correction/loss is already present
        at 2 curves.
    sort : {"growth", "age", "diameter", "none"}, default "growth"
        how trees are ranked before the equal-count split. "growth" is CRUST's
        recommended growth-rate ranking (see Notes); "age" uses cambial age;
        "diameter" uses total radius; "none" keeps input order.
    fit, method, period, nyrs0, f, pos_slope, biweight, min_n, preset,
    rescale_difference :
        passed through to :func:`rcs` to build each class curve; see
        :func:`rcs` for their meaning. Defaults match :func:`rcs`.
    reset_means : bool, default True
        if True (and ``method="ratio"``), rescale each tree's indices to the mean
        it would have had under one-curve RCS, recovering low-frequency variance
        (CRUST "mean single"). Skipped with a warning for ``method="difference"``.
    rc_out : bool, default False
        if True, return a dict ``{"rwi", "rc", "alloc"}`` -- the indices, the list
        of per-class regional curves (by cambial age), and the series->class map.
    plot : bool, default True
        draw the per-class regional-curve figure. ``make_plot=`` is accepted as a
        deprecated alias.

    Returns
    -------
    pandas.DataFrame or dict
        the detrended ``rwi`` (calendar-year indexed), or
        ``{"rwi", "rc", "alloc"}`` when ``rc_out`` is True.

    Notes
    -----
    Growth rate (``sort="growth"``) is the tree's total measured radius -- plus
    the regional-curve-estimated radius of any missing pith rings -- divided by
    the cumulative radius of the *unsmoothed* single regional curve (the
    age-aligned mean of all trees) at the tree's full cambial age, after CRUST's
    ``growth_rate`` routine. Trees are allocated to classes in near-equal numbers;
    a class with fewer than %d trees is flagged, following CRUST's guidance that
    multiple RCS needs well-replicated curves.

    References
    ----------
    .. [1] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further
       RCS options and recommendations. Dendrochronologia, 32, 343-356.
    """ % _MIN_TREES_PER_CURVE
    _require_dataframe(rwl)
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)
    if sort not in ("growth", "age", "diameter", "none"):
        raise ValueError("sort must be 'growth', 'age', 'diameter', or 'none', "
                         "got '" + str(sort) + "'.")
    n_curves = int(n_curves)
    if n_curves < 1:
        raise ValueError("n_curves must be >= 1, got %d." % n_curves)

    cols = list(rwl.columns)
    common = dict(fit=fit, method=method, period=period, nyrs0=nyrs0, f=f,
                  pos_slope=pos_slope, biweight=biweight, min_n=min_n,
                  preset=preset, rescale_difference=rescale_difference)

    # n_curves == 1 is exactly single-curve RCS
    if n_curves == 1:
        res = rcs(rwl, po=po, plot=plot, rc_out=True, **common)
        rwi = res["rwi"]
        if rc_out:
            return {"rwi": rwi, "rc": [res["rc"]], "alloc": {c: 0 for c in cols}}
        return rwi

    po_map = _po_map(rwl, po)

    # 1. rank trees and split into n_curves near-equal classes
    key = _sort_key(rwl, po_map, sort)
    order = list(np.argsort(key, kind="stable"))
    groups = [g for g in np.array_split(np.array(order), n_curves) if len(g) > 0]
    if len(groups) < n_curves:
        warnings.warn("mrcs: only %d non-empty curve(s) for n_curves=%d "
                      "(too few series)." % (len(groups), n_curves))
    for gi, g in enumerate(groups):
        if len(g) < _MIN_TREES_PER_CURVE:
            warnings.warn("mrcs: curve %d has %d trees (< %d); multiple RCS is "
                          "unreliable with sparsely-replicated curves."
                          % (gi, len(g), _MIN_TREES_PER_CURVE))

    # 2. standardise each class by its own regional curve (via single-curve rcs)
    rwi = rwl.copy().astype(float)
    rwi.iloc[:, :] = np.nan
    rc_list = []
    alloc = {}
    for gi, g in enumerate(groups):
        gcols = [cols[i] for i in g]
        sub = rwl[gcols]
        sub_po = pd.DataFrame({"series": gcols,
                               "pith_offset": [int(po_map[c]) for c in gcols]})
        res = rcs(sub, po=sub_po, plot=False, rc_out=True, **common)
        rwi[gcols] = res["rwi"]
        rc_list.append(res["rc"])
        for c in gcols:
            alloc[c] = gi

    # 3. mean-reset: put each tree back on its one-curve mean (ratios only)
    if reset_means:
        if method != "ratio":
            warnings.warn("mrcs: reset_means is only applied with method='ratio'; "
                          "skipped for method='difference'.")
        else:
            single = rcs(rwl, po=po, plot=False, **common)
            for c in cols:
                m = rwi[c].to_numpy(); s = single[c].to_numpy()
                mm = np.nanmean(m); ms = np.nanmean(s)
                if np.isfinite(mm) and np.isfinite(ms) and mm != 0:
                    rwi[c] = rwi[c] * (ms / mm)

    if plot:
        _plot_mrcs(rc_list)

    if rc_out:
        return {"rwi": rwi, "rc": rc_list, "alloc": alloc}
    return rwi


def _plot_mrcs(rc_list):
    """Overlay the per-class regional curves vs cambial age."""
    import matplotlib.pyplot as plt
    from ._plot_style import style_axes, finalize_font, ACCENT, NEUTRAL
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
    plt.show()
    return ax
