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

# Title: rcs.py
# Project: OpenDendro dplPy
# Description: Regional Curve Standardisation (single curve), a port of dplR's
#              rcs(). Series are aligned on a common cambial-age axis using their
#              pith offsets; the average growth at each cambial age forms the
#              "regional curve"; that curve is smoothed and each ring is divided
#              by (or has subtracted) the curve value at its cambial age. Unlike
#              curve-fitting standardisation this preserves low-frequency
#              (e.g. multi-centennial) signal. The regional curve can be fit
#              with any curve in detrend()'s vocabulary (fit=), mirroring how
#              ARSTAN and CRUST build it; fit='Spline' reproduces dplR's rcs().

import warnings

import numpy as np
import pandas as pd
from ._validate import _require_dataframe, _plot_kwarg
import matplotlib.pyplot as plt
from ._plot_style import style_axes, finalize_font, ACCENT, NEUTRAL
from .agedepspline import ads, _ads_curve
from .smoothingspline import _smooth_csaps, get_period
from .xdate import _row_biweight
from . import curvefit
from .detrend import _normalize_fit, _normalize_method


# --- CRUST-style regional-curve refinements (Melvin & Briffa 2014) ----------
# A native re-implementation of CRUST's spline3 / splinec kernels.

def _spline3(rws, cnt, ss=10, rise=False):
    """CRUST's age-dependent RCS smoother (stand.f90 spline3).

    Fits an age-dependent spline (per-point stiffness ``ss + cambial age``, i.e.
    our ads with nyrs0 = ss+1) only over the well-replicated part of the curve
    -- up to the last cambial age with sample depth >= 4 -- then holds the curve
    flat beyond. Unless ``rise`` (kept only for very old trees / basal area), the
    tail is flattened from the curve's minimum in its final third, preventing an
    artefactual old-age upturn.
    """
    n = len(rws)
    j = 0
    for jj in range(n, 0, -1):
        if cnt[jj - 1] >= 4:
            j = jj
            break
    if j < 4:                                   # too little replication to smooth
        j = n
    stiff = ss + np.arange(1, j + 1)            # ss+1, ss+2, ... = age-dependent
    rwp = np.full(n, np.nan)
    fit = _ads_curve(rws[:j], stiff)
    rwp[:j] = rws[:j] if fit is None else fit
    if rise:
        rwp[j:] = rwp[j - 1]                     # flat extend beyond the fit
    else:
        start = 2 * j // 3                       # final third (1-based start)
        q = int(np.argmin(rwp[start - 1:j])) + start   # 1-based index of the min
        rwp[q:] = rwp[q - 1]                     # flatten after it
    return rwp


def _crust_regional_curve(ca_m, ca_n, nrow, ss=10, rise=False, floor=0.02):
    """Build the CRUST-style regional curve: infill gaps, smooth with spline3,
    apply the minimum-value floor, and flat-extend to the full cambial-age grid."""
    last = int(np.max(np.where(ca_n > 0)[0]))    # last cambial age with data
    span = ca_m[:last + 1].copy()
    cnt = ca_n[:last + 1]
    # infill gaps: leading with the first value, interior by carry-forward
    valid = np.where(~np.isnan(span))[0]
    span[:valid[0]] = span[valid[0]]
    for m in range(valid[0] + 1, len(span)):
        if np.isnan(span[m]):
            span[m] = span[m - 1]
    rc_span = np.maximum(_spline3(span, cnt, ss=ss, rise=rise), floor)
    rc = np.full(nrow, np.nan)
    rc[:last + 1] = rc_span
    rc[last + 1:] = rc_span[-1]                   # flat extension
    return rc


def _caps(y, nyrs, f):
    """dplR's caps(): a cubic smoothing spline with an ``f`` (default 0.5)
    amplitude cutoff at wavelength ``nyrs``. dplR truncates nyrs to an integer
    (caps.R: as.integer(nyrs)), so we do too."""
    x = np.arange(1, len(y) + 1)
    return np.asarray(_smooth_csaps(x, y, int(nyrs), f))


# --- Signal-Free RCS engine (Melvin & Briffa 2014, CRUST stand.f90) ----------
# The signal-free iteration of CRUST's rcs_detrend/det_sf_rcs, single-curve case
# (trc=1, src=1). NOT validated against dplR -- dplR has no signal-free RCS -- so
# this is faithful to stand.f90 with its numerical core (spline3) validated
# against CRUST's compiled kernel to ~1e-11. See sfrcs.py for the public API.

def _build_rwca(X, po_arr, nrow):
    """Place each series' rings on the common cambial-age axis: series i's
    measurements start at cambial-age row ``po_arr[i]`` (a true pith offset;
    0 = pith present). Mirrors the alignment
    in rcs(); the NaN pattern of ``X`` selects the rings (so a signal-free copy
    of the measurement matrix aligns identically to the raw one)."""
    ncol = X.shape[1]
    rwca = np.full((nrow, ncol), np.nan)
    for i in range(ncol):
        vals = X[:, i][~np.isnan(X[:, i])]
        s = po_arr[i]
        rwca[s:s + len(vals), i] = vals
    return rwca


def _sfrcs_run(M, cols, po_map, ratios=True, biweight_curve=False,
               biweight_crn=False, ss=10, rise=None, max_iterations=40,
               tol=1e-3, verbose=True, alloc=None, reset_means=False):
    """Single-curve signal-free RCS. Returns (rwi_matrix, crn, samp_depth,
    history, n_iter, converged).

    Each iteration: form signal-free measurements ``fx = tx/crn`` (only where the
    calendar year has >1 tree and ``crn >= 0.01``, else keep raw ``tx`` -- CRUST's
    division guard); build the regional curve by cambial age from ``fx`` (so the
    age curve is not biased by the age structure of the sample); but form the tree
    indices from the *original* ``tx`` divided by that curve; rescale so the whole
    chronology has mean 1; recompute the chronology. Iterate until
    ``max|crn_k - crn_(k-1)| < tol`` (CRUST's convergence measure).
    """
    nyear, ncol = M.shape
    present = ~np.isnan(M)
    samp_depth = present.sum(axis=1)
    qok = samp_depth > 1                          # >1 tree in the calendar year
    firsts = np.array([int(np.argmax(present[:, i])) for i in range(ncol)])
    counts = present.sum(axis=0)
    po_arr = np.array([int(po_map[c]) for c in cols])
    nrow = int(counts.max()) + int(po_arr.max())
    rise_flag = (nrow > 1500) if rise is None else bool(rise)

    rwca_orig = _build_rwca(M, po_arr, nrow)      # raw measurements by cambial age
    row_mean = _row_biweight if biweight_curve else (
        lambda a: np.nanmean(np.where(np.isnan(a), np.nan, a), axis=1))

    # tree -> regional-curve allocation; a single group is single-curve SF-RCS
    alloc_arr = np.zeros(ncol, dtype=int) if alloc is None else np.asarray(alloc, dtype=int)
    curve_ids = sorted(set(alloc_arr.tolist()))
    multi = len(curve_ids) > 1

    crn = np.ones(nyear)
    prev = None
    history = []
    converged = False
    k = 0
    idx = None
    for k in range(1, max_iterations + 1):
        # 1. signal-free measurements (calendar space); iteration 1 = raw
        if k == 1:
            fx = M
        else:
            div = present & qok[:, None] & (crn[:, None] >= 0.01)
            fx = np.where(div, M / np.where(crn == 0, np.nan, crn)[:, None], M)
            fx[~present] = np.nan
        # 2. regional curve(s) by cambial age, from the signal-free measurements.
        # One curve per allocation group (a single group == single-curve SF-RCS).
        rwca_sf = _build_rwca(fx, po_arr, nrow)
        rc_mat = np.empty((nrow, ncol))
        rc_single = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for c in curve_ids:
                sel = np.where(alloc_arr == c)[0]
                sub = rwca_sf[:, sel]
                rcc = _crust_regional_curve(row_mean(sub),
                                            np.sum(~np.isnan(sub), axis=1),
                                            nrow, ss=ss, rise=rise_flag)
                rc_mat[:, sel] = rcc[:, None]
            if reset_means and ratios and multi:
                rc_single = _crust_regional_curve(
                    row_mean(rwca_sf), np.sum(~np.isnan(rwca_sf), axis=1),
                    nrow, ss=ss, rise=rise_flag)
        # 3. tree indices from the ORIGINAL measurements and each tree's curve
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = rwca_orig / rc_mat
        if ratios:
            rwica = ratio
        else:
            # CRUST residual mode (tree_ind ind==2): per tree, z-score the
            # residuals (tx - cx) and rescale them to the mean and SD of that
            # tree's ratios -- so residual indices sit on the ratio scale.
            rwica = np.full_like(ratio, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for j in range(ncol):
                    r = ratio[:, j]
                    d = rwca_orig[:, j] - rc_mat[:, j]
                    ok = ~np.isnan(r)
                    if ok.sum() < 2:
                        rwica[:, j] = d
                        continue
                    mn1, sd1 = np.nanmean(r), np.nanstd(r, ddof=1)
                    mn, sd = np.nanmean(d), np.nanstd(d, ddof=1)
                    rwica[:, j] = ((d - mn) / sd) * sd1 + mn1 if sd > 0 else r
        # 3b. mean-reset (multi-curve, ratios only): put each tree back on its
        # single-curve mean (CRUST "mean single") to recover low-freq variance.
        if rc_single is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                single_ratio = rwca_orig / rc_single[:, None]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for j in range(ncol):
                    ms = np.nanmean(single_ratio[:, j])
                    mm = np.nanmean(rwica[:, j])
                    if np.isfinite(ms) and np.isfinite(mm) and mm != 0:
                        rwica[:, j] = rwica[:, j] * (ms / mm)
        # 4. rescale so the whole chronology has mean 1 (CRUST "prevent drift")
        gm = np.nanmean(rwica)
        if np.isfinite(gm) and gm != 0:
            rwica = rwica / gm
        # 5. map indices back to calendar space, form the chronology
        idx = np.full((nyear, ncol), np.nan)
        for i in range(ncol):
            s = po_arr[i]
            n = int(counts[i])
            idx[firsts[i]:firsts[i] + n, i] = rwica[s:s + n, i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            crn_new = _row_biweight(idx) if biweight_crn else np.nanmean(idx, axis=1)
        # 6. convergence (max absolute change, over years with a chronology value)
        if prev is not None:
            d = np.abs(crn_new - prev)
            diff = float(np.nanmax(d)) if np.any(np.isfinite(d)) else 0.0
            history.append(diff)
            if diff < tol:
                crn = crn_new
                converged = True
                break
        prev = crn_new
        crn = crn_new

    if verbose:
        if converged:
            print(f"Signal-free RCS converged in {k} iterations "
                  f"(max change {history[-1]:.2e} < {tol:g}).")
        else:
            print(f"Signal-free RCS did NOT converge in {max_iterations} "
                  f"iterations (last max change "
                  f"{history[-1] if history else float('nan'):.2e}).")
    return idx, crn, samp_depth, history, k, converged


def _fit_regional_curve(ym, fit, period, nyrs0, f, pos_slope):
    """Smooth the age-aligned regional mean ``ym`` with the curve named by
    ``fit`` (canonical spelling, or None for no smoothing); returns an array the
    same length as ``ym``.

    ``fit`` draws on the same vocabulary as detrend(): the two splines plus the
    parametric growth curves in curvefit.py. This mirrors how ARSTAN and CRUST
    build the regional curve -- the RC is just another curve fit, including the
    "no smoothing" option (fit=None: ARSTAN's "no smooth" / CRUST rdt 2).
    """
    if fit is None:
        return np.asarray(ym, dtype=float)
    x = np.arange(1, len(ym) + 1)
    if fit == "Spline":
        return _caps(ym, get_period(period, len(ym)), f)
    if fit == "AgeDepSpline":
        return ads(ym, nyrs0=nyrs0, pos_slope=pos_slope)
    if fit == "ModNegExp":
        return curvefit.mod_neg_exp(x, ym, pos_slope)
    if fit == "NegExp":
        return curvefit.neg_exp_arstan(x, ym, pos_slope)
    if fit == "ModHugershoff":
        return curvefit.mod_hugershoff(x, ym, pos_slope)
    if fit == "Hugershoff":
        return curvefit.hugershoff_arstan(x, ym)
    if fit == "GeneralExp":
        return curvefit.general_exp(x, ym)
    if fit == "LinearAny":
        return curvefit.linear(x, ym)
    if fit == "LinearNegative":
        return curvefit.linear_neg(x, ym)
    if fit == "Mean":
        return curvefit.horizontal(x, ym)
    raise ValueError("unsupported fit for the regional curve: " + str(fit))


def rcs(rwl: pd.DataFrame, po=None, fit="AgeDepSpline", method="ratio",
        period=0.1, nyrs0=50, f=0.5, pos_slope=False, biweight=True,
        min_n=None, preset=None, rescale_difference=False, rc_out=False,
        plot=True, **kwargs):
    """Regional Curve Standardisation of a set of ring-width series.

    Extended Summary
    ----------------
    Detrends by a single "regional curve": every series is placed on a common
    cambial-age axis according to its pith offset, the average ring width at
    each cambial age is taken (the regional curve), that curve is smoothed, and
    each ring is divided by (ratios) or has subtracted (difference) the curve
    value at its cambial age. This preserves low-frequency growth signal that
    per-series detrending would remove. A port of dplR's rcs().

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame or None, default None
        pith offsets, with columns ``series`` and ``pith_offset``. The pith
        offset is a TRUE offset: the number of rings between the pith and the
        first measured ring, so **0 means the pith is present**. None assumes a
        pith offset of 0 for every series. **Convention (changed in 0.8.0):**
        this now equals ARSTAN's ``nshft`` exactly, so an ARSTAN pith-offset file
        can be used directly. dplR instead stores the cambial age of the first
        ring (1 = pith present), so a dplR ``po`` table needs 1 subtracted first.
    fit : str or None, default "AgeDepSpline"
        curve fitted to the regional (age-aligned mean) curve, from the same
        vocabulary as ``detrend()``: ``"AgeDepSpline"`` (default; age-dependent
        spline, ARSTAN's recommended RCS curve, honours ``pos_slope``),
        ``"Spline"`` (fixed-stiffness smoothing spline; dplR's ``rcs`` curve),
        ``"ModNegExp"``, ``"NegExp"``, ``"ModHugershoff"``, ``"Hugershoff"``,
        ``"GeneralExp"``, ``"LinearAny"``, ``"LinearNegative"``, ``"Mean"``, or
        ``None`` for no smoothing (the age-aligned mean itself is the regional
        curve -- ARSTAN's "no smooth" / CRUST's unsmoothed option). Case-insensitive.
    method : {"ratio", "difference"}, default "ratio"
        how each index is formed from the ring width and the regional-curve value
        at its cambial age: ``"ratio"`` divides (default), ``"difference"``
        subtracts. Matches ``detrend()``.
    period : int or float, default 0.1
        smoothing wavelength for ``fit="Spline"`` (same ``get_period`` rule as
        ``detrend``): ``0 < period <= 1`` is a fraction of the curve length
        (default ``0.1`` = 10%, reproducing dplR / pre-0.8 ``caps``);
        ``period < 0`` is a percent-of-length ("ARSTAN") spline (``-10`` = 10%);
        ``period > 1`` is a fixed wavelength in years.
    nyrs0 : int, default 50
        initial stiffness for ``fit="AgeDepSpline"`` (see ``dpl.ads``).
    f : float, default 0.5
        frequency-response amplitude for ``fit="Spline"``.
    pos_slope : bool, default False
        whether the fitted regional curve may rise at the old-age end. Default
        False gives the non-increasing end used by ARSTAN ("non-increasing end
        fitting") and CRUST ("no rise in the final third"), avoiding an
        artefactual old-age upturn. Applies to ``fit="AgeDepSpline"`` and the
        ``ModNegExp``/``NegExp``/``ModHugershoff`` linear fallbacks.
    biweight : bool, default True
        build the regional curve with Tukey's biweight robust mean (C=9) rather
        than the arithmetic mean.
    min_n : int or None, default None
        truncate the regional-curve tail where the by-age sample depth falls
        below this.
    preset : {None, "crust"}, default None
        None follows the parameters above. "crust" builds the regional curve the
        CRUST way (Melvin & Briffa 2014): an age-dependent spline smoothed only
        where sample depth >= 4 with the "no rise in the final third" tail rule,
        gaps infilled, a 0.02 mm minimum-value floor, and flat tail extension.
        When set, ``fit``/``period``/``nyrs0``/``f``/``pos_slope`` are ignored
        (it is a self-contained bundle of choices).
    rescale_difference : bool, default False
        only used when ``method="difference"``. If True, rescale each series'
        difference (residual) index onto the mean and standard deviation of that
        series' own ratio index, so difference indices sit on the ~1 ratio scale
        -- ARSTAN's ``calind`` convention. Default False returns residuals
        centred on ~0.
    rc_out : bool, default False
        if True, return a dict with both the detrended ``rwi`` and the regional
        curve ``rc``; otherwise return just ``rwi``.
    plot : bool, default True
        draw the regional-curve figure (series and curve vs cambial age). The
        former name ``make_plot=`` is accepted for now with a DeprecationWarning.

    Returns
    -------
    pandas.DataFrame or dict
        the detrended ``rwi`` (calendar-year indexed), or ``{"rwi", "rc"}`` when
        ``rc_out`` is True.

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/rcs.html
    .. [2] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 1,
       Signal-Free RCS. Dendrochronologia, 32, 7-20.
    .. [3] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further
       RCS options and recommendations. Dendrochronologia, 32, 343-356.
    """
    _require_dataframe(rwl)
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)               # 'ratio' / 'difference'
    fit = None if fit is None else _normalize_fit(fit)
    if preset not in (None, "crust"):
        raise ValueError("preset must be None or 'crust', got '" + str(preset) + "'.")

    cols = list(rwl.columns)
    ncol = len(cols)
    M = rwl.to_numpy(dtype=float)                    # (nyears, nseries)

    # pith offset (true offset; 0 = pith present): default 0 for every series
    if po is None:
        po_map = {c: 0 for c in cols}
    else:
        s = po["series"] if "series" in po.columns else po.iloc[:, 0]
        pv = po["pith_offset"] if "pith_offset" in po.columns else po.iloc[:, 1]
        po_map = dict(zip(np.asarray(s), np.asarray(pv)))

    counts = np.sum(~np.isnan(M), axis=0)            # non-NA rings per series
    max_len = int(counts.max())
    max_po = int(max(int(po_map[c]) for c in cols))

    # ring width by cambial age: each series' (gap-compacted) values placed
    # starting at row = its pith offset.
    rwca = np.full((max_len + max_po, ncol), np.nan)
    for i, c in enumerate(cols):
        vals = M[:, i][~np.isnan(M[:, i])]
        p = int(po_map[c])
        rwca[p:p + len(vals), i] = vals

    # regional curve = average by cambial age
    if biweight:
        ca_m = _row_biweight(rwca)
    else:
        with np.errstate(invalid="ignore"):
            ca_m = np.nanmean(rwca, axis=1)
    ca_n = np.sum(~np.isnan(rwca), axis=1)

    if min_n is not None:
        last_valid = int(np.max(np.where(ca_n >= min_n)[0]))
        ca_m[np.arange(len(ca_m)) > last_valid] = np.nan

    if preset == "crust":
        # CRUST regional curve: infill, age-dependent spline smoothed only where
        # depth >= 4 (no-rise-final-third tail), 0.02 floor, flat extension.
        ss = 10
        rise = rwca.shape[0] > 1500              # "except trees > 1500 years old"
        rc = _crust_regional_curve(ca_m, ca_n, rwca.shape[0], ss=ss, rise=rise)
    else:
        # smooth (or not) the regional curve over its non-NA extent, with the
        # curve chosen by fit= (shared vocabulary with detrend())
        valid = ~np.isnan(ca_m)
        ym = ca_m[valid]
        smoothed = _fit_regional_curve(ym, fit, period, nyrs0, f, pos_slope)
        rc = np.full(rwca.shape[0], np.nan)
        rc[valid] = np.asarray(smoothed, dtype=float)

    # Non-positive regional curve: a flexible RC can dip to or below zero, which
    # makes ratio indices blow up or turn negative. ARSTAN halts here ("rcs growth
    # curve goes negative"); as a library dplPy warns and continues so batch runs
    # are not interrupted -- the user can pick a stiffer curve or difference mode.
    _rc_vals = rc[~np.isnan(rc)]
    if np.any(_rc_vals <= 0):
        warnings.warn(
            "rcs: the regional curve is <= 0 at %d cambial age(s); ratio indices "
            "there will be unstable or negative. Consider a stiffer smoothing "
            "(a larger period), fit='AgeDepSpline', or method='difference'."
            % int(np.sum(_rc_vals <= 0)))

    # detrend in cambial-age space, then map back to calendar years
    rwica = rwca / rc[:, None] if method == "ratio" else rwca - rc[:, None]

    if method != "ratio" and rescale_difference:
        # ARSTAN calind (ind=2): put each series' difference (residual) index onto
        # the mean & SD of that series' own ratio index, so difference indices sit
        # on the ~1 ratio scale. This is the faithful per-series version of the
        # chronology-level "re-mean" sometimes applied downstream.
        with np.errstate(invalid="ignore", divide="ignore"):
            _ratio_idx = rwca / rc[:, None]
        for i in range(ncol):
            d = rwica[:, i]
            r = _ratio_idx[:, i]
            ok = np.isfinite(d) & np.isfinite(r)
            if ok.sum() < 2:
                continue
            sd = d[ok].std()
            if sd == 0 or not np.isfinite(sd):
                continue
            rwica[:, i] = (d - d[ok].mean()) / sd * r[ok].std() + r[ok].mean()
    rwi = rwl.copy().astype(float)
    for i, c in enumerate(cols):
        mask = ~np.isnan(M[:, i])
        if not mask.any():
            continue
        first = int(np.argmax(mask))
        last = len(mask) - 1 - int(np.argmax(mask[::-1]))
        p = int(po_map[c])
        rwi.iloc[first:last + 1, i] = rwica[p:p + int(counts[i]), i]

    if plot:
        _plot_rcs(rwca, ca_m, rc, ncol)

    if rc_out:
        return {"rwi": rwi, "rc": rc}
    return rwi


def _plot_rcs(rwca, ca_m, rc, ncol):
    """Regional-curve figure: each series and the regional curve vs cambial age
    (mirrors dplR's rcs plot), base-R style."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(1, rwca.shape[0] + 1)
    alpha = min(0.15, 5.0 / ncol)
    for i in range(ncol):
        ax.plot(x, rwca[:, i], color="0.6", lw=1, alpha=alpha, zorder=1)
    ax.plot(x, ca_m, color=NEUTRAL, lw=1, zorder=2, label="Mean")
    ax.plot(x, rc, color=ACCENT, lw=2.5, zorder=3, label="Regional curve")
    ax.plot([], [], color="0.6", lw=1, label="Series")
    ax.set_xlabel("Cambial age (years)")
    ax.set_ylabel("Ring width (mm)")
    ax.set_title("Regional curve standardization", fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    style_axes(ax, xgrid=True, ygrid=True)
    fig.tight_layout()
    finalize_font(fig)
    plt.show()
    return ax
