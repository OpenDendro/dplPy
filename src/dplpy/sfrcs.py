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

from .rcs import _sfrcs_run, _plot_rcs, _resolve_po, _as_rwi_frame
from ._validate import (_require_dataframe, _plot_kwarg, _warn, _alias_kwarg,
                        _normalize_choice)
from .detrend import _normalize_method, _normalize_fit


def _sf_options(kwargs, biweight, max_iterations, func):
    """Shared argument plumbing for sfrcs/sfmrcs: the deprecated
    ``biweight_curve``/``biweight_crn`` pair collapses onto ``biweight`` (if both
    are given, ``biweight_crn`` wins with a warning), and ``max_iterations`` must
    be >= 1."""
    old = {k: kwargs.pop(k) for k in ("biweight_curve", "biweight_crn") if k in kwargs}
    for k in old:
        _warn(func + "(): " + k + "= is deprecated and will be removed in a future "
              "release; use biweight= instead.", DeprecationWarning, stacklevel=4)
    if len(old) == 2 and bool(old["biweight_curve"]) != bool(old["biweight_crn"]):
        _warn(func + "(): biweight_curve and biweight_crn disagree; a single "
              "biweight= now controls both the regional-curve mean and the "
              "chronology mean -- using biweight_crn="
              + str(bool(old["biweight_crn"])) + ".", stacklevel=4)
    if "biweight_crn" in old:
        biweight = bool(old["biweight_crn"])
    elif "biweight_curve" in old:
        biweight = bool(old["biweight_curve"])
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1.")
    return biweight


def _curve_settings(fit, spline_period, nyrs0, f, pos_slope, preset, ss, rise, min_n):
    """Normalise the regional-curve options shared by sfrcs/sfmrcs into the dict
    ``_sfrcs_run`` hands to ``_regional_curve``."""
    return dict(fit=None if fit is None else _normalize_fit(fit),
                spline_period=spline_period, nyrs0=nyrs0, f=f, pos_slope=pos_slope,
                preset=_normalize_choice(preset, ("crust",), "preset", none_ok=True),
                ss=ss, rise=rise, min_n=min_n)


def sfrcs(rwl: pd.DataFrame, po=None, fit="AgeDepSpline", method="ratio",
          spline_period=0.1, nyrs0=50, f=0.5, pos_slope=False, biweight=False,
          min_n=None, preset="crust", ss=10, rise=None, max_iterations=40,
          tol=1e-3, rescale_difference=False, plot=True, return_info=False,
          verbose=True, **kwargs):
    """Signal-Free Regional Curve Standardization of a set of ring-width series.

    Ordinary RCS lets the single regional curve absorb some of the common climate
    signal when the sample's age structure varies over time, biasing the low
    frequencies. Signal-Free RCS breaks that feedback: it iterates, each pass
    rebuilding the regional curve from measurements with the current chronology
    divided out, while still forming the tree indices from the original
    measurements. Iteration stops when the chronology stops changing (``tol``). Port
    of CRUST's single-curve signal-free RCS (Melvin & Briffa 2014). The regional
    curve is built exactly as ``rcs`` builds it, so ``sfrcs(max_iterations=1)``
    equals ``rcs`` (with the same curve settings) up to the signal-free
    normalisation of the chronology to mean 1.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame, dict, Series or None, default None
        pith offsets (true offset, 0 = pith present), as in ``rcs``. None assumes
        0 for all; a missing or NA offset is taken as 0 with a warning.
    fit, spline_period, nyrs0, f, pos_slope, min_n :
        regional-curve settings, as in ``rcs``; used when ``preset`` is None.
    method : {"ratio", "difference"}, default "ratio"
        how indices are formed: "ratio" divides, "difference" subtracts.
    biweight : bool, default False
        use a Tukey biweight robust mean both for the regional curve by cambial age
        and for the yearly chronology. Default False (arithmetic means) matches
        CRUST. The former pair ``biweight_curve=``/``biweight_crn=`` is accepted
        with a DeprecationWarning (if both are given, ``biweight_crn`` wins).
    preset : {None, "crust"}, default "crust"
        "crust" (the default, unlike ``rcs``) builds the regional curve the CRUST way
        -- an age-dependent spline with per-point stiffness ``ss + cambial age``
        smoothed only where sample depth >= 4, the no-rise tail rule, gaps infilled,
        a 0.02 mm floor and flat tail extension -- ignoring ``fit`` and its
        companions. None uses the ``fit`` vocabulary of ``rcs`` instead.
    ss : int, default 10
        CRUST's age-dependent spline stiffness offset (``preset="crust"`` only).
    rise : bool or None, default None
        allow the CRUST curve to rise in its final third. None uses CRUST's rule
        (allowed only for curves longer than 1500 years).
    max_iterations : int, default 40
        maximum signal-free iterations.
    tol : float, default 1e-3
        convergence threshold on the maximum chronology change between iterations.
    rescale_difference : bool, default False
        ``method="difference"`` only. False (as in ``rcs``) returns raw residual
        indices centred on ~0. True rescales each series' residuals onto the mean
        and SD of its own ratio index, so difference indices sit on the ~1 ratio
        scale -- CRUST's residual mode (``ind==2``) and what signal-free difference
        indices need before the chronology is divided out, so prefer True for
        difference-mode signal-free work. (Before 0.9.0 sfrcs always rescaled.)
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
             "rc":         the regional curve (by cambial age) that produced rwi}

    Notes
    -----
    Single-curve case (CRUST ``trc=1, src=1``); multiple-curve signal-free RCS is
    ``sfmrcs``. There is no dplR reference for this method: the port follows CRUST's
    ``stand.f90`` and its spline smoother is validated against CRUST's compiled
    kernel, but the full iteration is not checked against a gold standard, so treat
    results as CRUST-faithful rather than machine-verified. An interior NaN is an
    unmeasured ring: it advances cambial age and stays NaN in the output.

    References
    ----------
    .. [1] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 1, Signal-Free
       RCS. Dendrochronologia, 32, 7-20.
    .. [2] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further RCS
       options and recommendations. Dendrochronologia, 32, 343-356.
    """
    _require_dataframe(rwl)
    spline_period = _alias_kwarg(kwargs, "period", "spline_period", spline_period, "sfrcs")
    biweight = _sf_options(kwargs, biweight, max_iterations, "sfrcs")
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)
    curve = _curve_settings(fit, spline_period, nyrs0, f, pos_slope, preset, ss,
                            rise, min_n)

    M = rwl.to_numpy(dtype=float)
    po_arr = _resolve_po(rwl, po, label="sfrcs")

    run = _sfrcs_run(M, po_arr, method=method, biweight=biweight, curve=curve,
                     max_iterations=int(max_iterations), tol=float(tol),
                     verbose=verbose, rescale_difference=rescale_difference,
                     label="sfrcs")
    rwi = _as_rwi_frame(rwl, run["rwi"])
    rc = run["rc_list"][0]

    if plot:
        _plot_rcs(run["rwca"], run["ca_m_list"][0], rc, len(rwl.columns))

    if return_info:
        sfc = pd.DataFrame({"sfc": run["crn"], "samp.depth": run["samp_depth"]},
                           index=rwl.index)
        return {"rwi": rwi, "sfc": sfc, "samp_depth": run["samp_depth"],
                "conv": np.asarray(run["history"]), "n_iter": run["n_iter"],
                "converged": run["converged"], "rc": rc}
    return rwi
