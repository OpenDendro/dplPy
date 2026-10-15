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

# Title: sfmrcs.py
# Project: OpenDendro dplPy
# Description: Signal-Free multiple-curve Regional Curve Standardisation -- the
#              signal-free iteration (as in sfrcs) wrapped around the growth-rate
#              curve allocation (as in mrcs), a port of CRUST's signal-free
#              multiple-RCS. Trees are sorted by relative growth rate and split
#              into n_curves equal-count classes once; then, each iteration, the
#              per-class regional curves are rebuilt from signal-free measurements
#              (the data with the current common chronology divided out) while the
#              tree indices are formed from the original measurements, iterating to
#              convergence. reset_means (default) restores each tree's one-curve
#              mean to recover low-frequency variance. sfmrcs(n_curves=1) is
#              identical to sfrcs().

import numpy as np
import pandas as pd

from ._validate import (_require_dataframe, _plot_kwarg, _warn, _alias_kwarg,
                        _normalize_choice)
from .detrend import _normalize_method
from .rcs import _sfrcs_run, _resolve_po, _as_rwi_frame
from .mrcs import _allocate, _plot_mrcs, _SORTS
from .sfrcs import sfrcs, _sf_options, _curve_settings


def sfmrcs(rwl: pd.DataFrame, po=None, n_curves=2, sort="growth",
           fit="AgeDepSpline", method="ratio", spline_period=0.1, nyrs0=50,
           f=0.5, pos_slope=False, biweight=False, min_n=None, preset="crust",
           ss=10, rise=None, max_iterations=40, tol=1e-3, reset_means=True,
           rescale_difference=False, plot=True, return_info=False,
           verbose=True, **kwargs):
    """Signal-Free multiple-curve Regional Curve Standardisation.

    Combine `sfrcs()` (signal-free iteration) with `mrcs()` (multiple
    curves): sort trees by growth rate into ``n_curves`` equal-count classes, then
    iteratively rebuild each class's regional curve from measurements that have had
    the current chronology divided out, while forming the tree indices from the
    original measurements, until the chronology converges. ``reset_means`` (ratios
    only) restores each tree's one-curve mean to recover the low-frequency variance
    multiple RCS removes. ``sfmrcs(rwl, n_curves=1)`` is identical to
    ``sfrcs(rwl)``. Port of CRUST's signal-free multiple RCS.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame, dict, Series or None, default None
        pith offsets (true offset, 0 = pith present), as in `rcs()`.
    n_curves : int, default 2
        number of regional curves. ``1`` reduces exactly to `sfrcs()`.
    sort : {"growth", "age", "diameter", "none"}, default "growth"
        how trees are ranked before the equal-count split (see `mrcs()`).
    fit, spline_period, nyrs0, f, pos_slope, min_n, preset, ss, rise :
        the regional-curve settings, identical to `sfrcs()` (``preset="crust"``
        by default; ``period=`` is a deprecated alias of ``spline_period``).
    method : {"ratio", "difference"}, default "ratio"
        ratio (division) or difference (subtraction) indices.
    biweight, max_iterations, tol, verbose :
        the signal-free settings, identical to `sfrcs()` (``biweight_curve=`` /
        ``biweight_crn=`` are deprecated aliases of ``biweight``).
    reset_means : bool, default True
        restore each tree's one-curve mean (ratios only; skipped with a warning
        for ``method="difference"``).
    rescale_difference : bool, default False
        ``method="difference"`` only: rescale residual indices onto the ratio
        scale, as in `sfrcs()` (needed for signal-free difference indices; see
        there). Before 0.9.0 the rescale was always applied.
    plot : bool, default True
        draw the per-class regional-curve figure (``make_plot=`` is a deprecated
        alias).
    return_info : bool, default False
        if True, return a dict of diagnostics (see Returns) instead of ``rwi``.

    Returns
    -------
    pandas.DataFrame or dict
        the detrended ``rwi`` (calendar-year indexed), or, with
        ``return_info=True``, a dict with ``rwi``, ``sfc`` (the signal-free
        chronology + sample depth), ``samp_depth``, ``conv`` (per-iteration max
        change), ``n_iter``, ``converged``, ``rc`` (list of per-class regional
        curves that produced ``rwi``) and ``alloc`` (series -> class).

    References
    ----------
    .. [1] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further
       RCS options and recommendations. Dendrochronologia, 32, 343-356.
    """
    _require_dataframe(rwl)
    spline_period = _alias_kwarg(kwargs, "period", "spline_period", spline_period, "sfmrcs")
    biweight = _sf_options(kwargs, biweight, max_iterations, "sfmrcs")
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)
    sort = _normalize_choice(sort, _SORTS, "sort")
    n_curves = int(n_curves)
    if n_curves < 1:
        raise ValueError("n_curves must be >= 1, got %d." % n_curves)
    curve = _curve_settings(fit, spline_period, nyrs0, f, pos_slope, preset, ss,
                            rise, min_n)

    # n_curves == 1 is exactly single-curve signal-free RCS
    if n_curves == 1:
        return sfrcs(rwl, po=po, method=method, biweight=biweight,
                     max_iterations=max_iterations, tol=tol, plot=plot,
                     return_info=return_info, verbose=verbose,
                     rescale_difference=rescale_difference, **curve)

    if reset_means and method != "ratio":
        _warn("sfmrcs: reset_means is only applied with method='ratio'; "
              "skipped for method='difference'.")
    do_reset = reset_means and method == "ratio"

    cols = list(rwl.columns)
    M = rwl.to_numpy(dtype=float)
    po_arr = _resolve_po(rwl, po, label="sfmrcs")
    alloc, groups = _allocate(rwl, po_arr, n_curves, sort, label="sfmrcs")

    run = _sfrcs_run(M, po_arr, method=method, biweight=biweight, curve=curve,
                     max_iterations=int(max_iterations), tol=float(tol),
                     verbose=verbose, alloc=alloc, reset_means=do_reset,
                     rescale_difference=rescale_difference, label="sfmrcs")
    rwi = _as_rwi_frame(rwl, run["rwi"])
    rc_list = run["rc_list"]

    if plot:
        _plot_mrcs(rc_list)

    if return_info:
        sfc = pd.DataFrame({"sfc": run["crn"], "samp.depth": run["samp_depth"]},
                           index=rwl.index)
        return {"rwi": rwi, "sfc": sfc, "samp_depth": run["samp_depth"],
                "conv": np.asarray(run["history"]), "n_iter": run["n_iter"],
                "converged": run["converged"], "rc": rc_list,
                "alloc": {cols[i]: int(alloc[i]) for i in range(len(cols))}}
    return rwi
