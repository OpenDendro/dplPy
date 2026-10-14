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

# Title: sfrcs.py
# Project: OpenDendro dplPy
# Description: Signal-Free Regional Curve Standardisation (single curve), a port
#              of CRUST's rcs_detrend/det_sf_rcs (Melvin & Briffa 2014). Wraps the
#              signal-free iteration around RCS-by-cambial-age: at each pass the
#              regional curve is rebuilt from measurements with the current common
#              signal divided out, which removes the "trend distortion" that biases
#              ordinary RCS when the age structure of the sample is uneven over
#              calendar time. NEW to the dpl family -- no dplR counterpart exists,
#              so it is faithful to CRUST rather than validated against dplR.

import numpy as np
import pandas as pd

from .rcs import _sfrcs_run, _plot_rcs, _build_rwca, _crust_regional_curve
from ._validate import _plot_kwarg
from .detrend import _normalize_method


def sfrcs(rwl: pd.DataFrame, po=None, method="ratio", biweight_curve=False,
          biweight_crn=False, ss=10, rise=None, max_iterations=40, tol=1e-3,
          plot=True, return_info=False, verbose=True, **kwargs):
    """Signal-Free Regional Curve Standardization of a set of ring-width series.

    Ordinary RCS lets the single regional curve absorb some of the common climate
    signal when the sample's age structure varies over time, biasing the low
    frequencies. Signal-Free RCS breaks that feedback: it iterates, each pass
    rebuilding the regional curve from measurements with the current chronology
    divided out, while still forming the tree indices from the original
    measurements. Iteration stops when the chronology stops changing (``tol``). Port
    of CRUST's single-curve signal-free RCS (Melvin & Briffa 2014).

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame or None, default None
        pith offsets, columns ``series`` and ``pith_offset`` -- rings between the pith
        and the first measured ring (0 = pith present). None assumes 0 for all. (A dplR
        ``po`` table stores the cambial age of the first ring, 1 = pith present, so
        subtract 1 first.)
    method : {"ratio", "difference"}, default "ratio"
        how indices are formed: "ratio" divides, "difference" subtracts.
    biweight_curve : bool, default False
        build the regional curve with a biweight robust mean by cambial age. Default
        False (arithmetic mean by age) matches CRUST.
    biweight_crn : bool, default False
        build the yearly chronology mean with a biweight robust mean. Default False
        (arithmetic) matches CRUST's default.
    ss : int, default 10
        age-dependent spline stiffness offset for the regional curve; per-point
        stiffness is ``ss + cambial age`` (ss=10 -> 11-year minimum), as in CRUST.
    rise : bool or None, default None
        allow the smoothed regional curve to rise in its final third. None uses CRUST's
        rule (allowed only for curves longer than 1500 years); otherwise the tail is
        flattened to prevent an artefactual old-age upturn.
    max_iterations : int, default 40
        maximum signal-free iterations.
    tol : float, default 1e-3
        convergence threshold on the maximum chronology change between iterations.
    plot : bool, default True
        draw the regional-curve figure (series and final curve vs cambial age). The
        former name ``make_plot=`` is accepted for now with a DeprecationWarning.
    return_info : bool, default False
        if True, return a dict of diagnostics (see Returns) instead of just ``rwi``.
    verbose : bool, default True
        print a convergence line.

    Returns
    -------
    pandas.DataFrame or dict
        the detrended ``rwi`` (calendar x series). With ``return_info=True``, a dict::

            {"rwi":        detrended series (calendar x series),
             "sfc":        the signal-free chronology (Series + samp.depth),
             "samp_depth": number of trees per calendar year,
             "conv":       max absolute chronology change at each iteration,
             "n_iter":     iterations run,
             "converged":  whether tol was reached within max_iterations,
             "rc":         the final regional curve (by cambial age)}

    Notes
    -----
    Single-curve case (CRUST ``trc=1, src=1``); multiple-curve signal-free RCS is
    ``sfmrcs``. There is no dplR reference for this method: the port follows CRUST's
    ``stand.f90`` and its spline smoother is validated against CRUST's compiled
    kernel, but the full iteration is not checked against a gold standard, so treat
    results as CRUST-faithful rather than machine-verified.

    References
    ----------
    .. [1] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 1, Signal-Free
       RCS. Dendrochronologia, 32, 7-20.
    .. [2] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further RCS
       options and recommendations. Dendrochronologia, 32, 343-356.
    """
    if not isinstance(rwl, pd.DataFrame):
        raise TypeError("Expected dataframe input, got " + str(rwl.__class__) + " instead.")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1.")
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)
    ratios = (method == "ratio")

    cols = list(rwl.columns)
    M = rwl.to_numpy(dtype=float)

    if po is None:
        po_map = {c: 0 for c in cols}
    else:
        s = po["series"] if "series" in po.columns else po.iloc[:, 0]
        pv = po["pith_offset"] if "pith_offset" in po.columns else po.iloc[:, 1]
        po_map = dict(zip(np.asarray(s), np.asarray(pv)))

    idx, crn, samp_depth, history, n_iter, converged = _sfrcs_run(
        M, cols, po_map, ratios=ratios, biweight_curve=biweight_curve,
        biweight_crn=biweight_crn, ss=ss, rise=rise,
        max_iterations=int(max_iterations), tol=float(tol), verbose=verbose)

    # detrended series back into a calendar-indexed frame
    rwi = rwl.copy().astype(float)
    rwi.iloc[:, :] = idx

    if plot or return_info:
        # rebuild the final regional curve (from the converged signal-free
        # measurements) for plotting / return
        present = ~np.isnan(M)
        po_arr = np.array([int(po_map[c]) for c in cols])
        counts = present.sum(axis=0)
        nrow = int(counts.max()) + int(po_arr.max())
        qok = (present.sum(axis=1) > 1)
        crn_col = crn.copy()
        div = present & qok[:, None] & (crn_col[:, None] >= 0.01)
        fx = np.where(div, M / np.where(crn_col == 0, np.nan, crn_col)[:, None], M)
        fx[~present] = np.nan
        rwca_sf = _build_rwca(fx, po_arr, nrow)
        rwca_orig = _build_rwca(M, po_arr, nrow)
        if biweight_curve:
            from .xdate import _row_biweight
            ca_m = _row_biweight(rwca_sf)
        else:
            with np.errstate(invalid="ignore"):
                ca_m = np.nanmean(rwca_sf, axis=1)
        ca_n = np.sum(~np.isnan(rwca_sf), axis=1)
        rise_flag = (nrow > 1500) if rise is None else bool(rise)
        rc = _crust_regional_curve(ca_m, ca_n, nrow, ss=ss, rise=rise_flag)
        if plot:
            _plot_rcs(rwca_orig, ca_m, rc, len(cols))

    if return_info:
        sfc = pd.DataFrame({"sfc": crn, "samp.depth": samp_depth}, index=rwl.index)
        return {"rwi": rwi, "sfc": sfc, "samp_depth": samp_depth,
                "conv": np.asarray(history), "n_iter": n_iter,
                "converged": converged, "rc": rc}
    return rwi
