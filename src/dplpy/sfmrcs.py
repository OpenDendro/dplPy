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

import warnings

import numpy as np
import pandas as pd

from ._validate import _require_dataframe, _plot_kwarg
from .detrend import _normalize_method
from .rcs import _sfrcs_run, _build_rwca, _crust_regional_curve, _row_biweight
from .mrcs import _allocate, _po_map, _plot_mrcs
from .sfrcs import sfrcs


def sfmrcs(rwl: pd.DataFrame, po=None, n_curves=2, sort="growth",
           method="ratio", biweight_curve=False, biweight_crn=False,
           ss=10, rise=None, max_iterations=40, tol=1e-3,
           reset_means=True, plot=True, return_info=False, verbose=True,
           **kwargs):
    """Signal-Free multiple-curve Regional Curve Standardisation.

    Extended Summary
    ----------------
    Combines :func:`sfrcs` (signal-free RCS) with :func:`mrcs` (multiple-curve
    RCS): trees are sorted by relative growth rate and split into ``n_curves``
    equal-count classes, then the signal-free iteration rebuilds each class's
    regional curve from measurements that have had the current chronology divided
    out, while forming the tree indices from the original measurements, until the
    chronology converges. A port of CRUST's signal-free multiple-RCS.

    Multiple RCS corrects the growth-rate slope bias but removes some genuine
    low-frequency variance; ``reset_means`` (default ``True``, ratios only)
    restores each tree's one-curve mean to recover it (CRUST "mean single").
    ``sfmrcs(rwl, n_curves=1)`` is identical to ``sfrcs(rwl)``.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame or None, default None
        pith offsets (true offset, 0 = pith present), as in :func:`rcs`.
    n_curves : int, default 2
        number of regional curves. ``1`` reduces exactly to :func:`sfrcs`.
    sort : {"growth", "age", "diameter", "none"}, default "growth"
        how trees are ranked before the equal-count split (see :func:`mrcs`).
    method : {"ratio", "difference"}, default "ratio"
        ratio (division) or difference (subtraction) indices.
    biweight_curve, biweight_crn, ss, rise, max_iterations, tol, verbose :
        the signal-free settings, identical to :func:`sfrcs`.
    reset_means : bool, default True
        restore each tree's one-curve mean (ratios only; skipped with a warning
        for ``method="difference"``).
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
        curves) and ``alloc`` (series -> class).

    References
    ----------
    .. [1] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further
       RCS options and recommendations. Dendrochronologia, 32, 343-356.
    """
    _require_dataframe(rwl)
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)
    if sort not in ("growth", "age", "diameter", "none"):
        raise ValueError("sort must be 'growth', 'age', 'diameter', or 'none', "
                         "got '" + str(sort) + "'.")
    n_curves = int(n_curves)
    if n_curves < 1:
        raise ValueError("n_curves must be >= 1, got %d." % n_curves)
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1.")

    # n_curves == 1 is exactly single-curve signal-free RCS
    if n_curves == 1:
        return sfrcs(rwl, po=po, method=method, biweight_curve=biweight_curve,
                     biweight_crn=biweight_crn, ss=ss, rise=rise,
                     max_iterations=max_iterations, tol=tol, plot=plot,
                     return_info=return_info, verbose=verbose)

    if reset_means and method != "ratio":
        warnings.warn("sfmrcs: reset_means is only applied with method='ratio'; "
                      "skipped for method='difference'.")
    do_reset = reset_means and method == "ratio"

    cols = list(rwl.columns)
    M = rwl.to_numpy(dtype=float)
    po_map = _po_map(rwl, po)
    alloc, groups = _allocate(rwl, po_map, n_curves, sort, label="sfmrcs")

    idx, crn, samp_depth, history, n_iter, converged = _sfrcs_run(
        M, cols, po_map, ratios=(method == "ratio"),
        biweight_curve=biweight_curve, biweight_crn=biweight_crn, ss=ss,
        rise=rise, max_iterations=int(max_iterations), tol=float(tol),
        verbose=verbose, alloc=alloc, reset_means=do_reset)

    rwi = rwl.copy().astype(float)
    rwi.iloc[:, :] = idx

    if plot or return_info:
        # rebuild each class's final regional curve from the converged
        # signal-free measurements (as sfrcs does for its single curve)
        present = ~np.isnan(M)
        po_arr = np.array([int(po_map[c]) for c in cols])
        counts = present.sum(axis=0)
        nrow = int(counts.max()) + int(po_arr.max())
        qok = (present.sum(axis=1) > 1)
        div = present & qok[:, None] & (crn[:, None] >= 0.01)
        fx = np.where(div, M / np.where(crn == 0, np.nan, crn)[:, None], M)
        fx[~present] = np.nan
        rwca_sf = _build_rwca(fx, po_arr, nrow)
        rise_flag = (nrow > 1500) if rise is None else bool(rise)
        row_mean = _row_biweight if biweight_curve else (
            lambda a: np.nanmean(np.where(np.isnan(a), np.nan, a), axis=1))
        rc_list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for gi in range(len(groups)):
                sel = np.where(alloc == gi)[0]
                sub = rwca_sf[:, sel]
                rc_list.append(_crust_regional_curve(
                    row_mean(sub), np.sum(~np.isnan(sub), axis=1), nrow,
                    ss=ss, rise=rise_flag))
        if plot:
            _plot_mrcs(rc_list)

    if return_info:
        sfc = pd.DataFrame({"sfc": crn, "samp.depth": samp_depth}, index=rwl.index)
        return {"rwi": rwi, "sfc": sfc, "samp_depth": samp_depth,
                "conv": np.asarray(history), "n_iter": n_iter,
                "converged": converged, "rc": rc_list,
                "alloc": {cols[i]: int(alloc[i]) for i in range(len(cols))}}
    return rwi
