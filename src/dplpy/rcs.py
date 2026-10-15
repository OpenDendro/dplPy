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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._plot_style import style_axes, finalize_font, ACCENT, NEUTRAL
from ._validate import (_require_dataframe, _plot_kwarg, _warn, _alias_kwarg,
                        _normalize_choice, _series_table, _carry_attrs)
from .agedepspline import ads, _ads_curve
from .smoothingspline import _caps, get_period
from .xdate import _row_biweight
from . import curvefit
from .detrend import _normalize_fit, _normalize_method, _finish_figure, _MIN_N


# --- Pith offsets (audit B8 / E2) -------------------------------------------

def _resolve_po(rwl, po, label="rcs"):
    """Resolve a pith-offset table to an int array aligned with ``rwl.columns``.

    ``po`` is None (0 for every series) or a per-series table (DataFrame with
    ``series``/``pith_offset`` columns -- or two positional columns -- a dict or a
    Series). Offsets are TRUE offsets: 0 = pith present. Policy: a series missing
    from the table, or with a NaN/<NA> offset (what ``wc_to_po`` gives an
    unknown-pith series), gets 0 with a warning naming it; a non-integer or
    negative offset raises ValueError.
    """
    cols = [str(c) for c in rwl.columns]
    if po is None:
        return np.zeros(len(cols), dtype=int)
    if isinstance(po, pd.DataFrame) and po.shape[1] == 2 \
            and "series" not in po.columns and "pith_offset" not in po.columns:
        po = po.set_axis(["series", "pith_offset"], axis=1)      # positional table
    value_col = "pith_offset" if isinstance(po, pd.DataFrame) \
        and "pith_offset" in po.columns else None
    tab = _series_table(po, "po", value_col=value_col)
    missing = [c for c in cols if c not in tab.index]
    if missing:
        _warn(label + ": no pith offset for " + str(len(missing)) + " series ("
              + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
              + "); assuming 0 (pith present).", stacklevel=4)
    out = np.zeros(len(cols), dtype=int)
    na = []
    for i, c in enumerate(cols):
        if c not in tab.index:
            continue
        v = tab.loc[c]
        if isinstance(v, pd.Series):                   # duplicated series name
            v = v.iloc[0]
        if pd.isna(v):
            na.append(c)
            continue
        fv = float(v)
        if fv != np.floor(fv):
            raise ValueError(label + ": pith offset for '" + c + "' must be a "
                             "whole number of rings, got " + repr(v) + ".")
        if fv < 0:
            raise ValueError(label + ": pith offset for '" + c + "' is negative ("
                             + repr(v) + "); offsets count rings from the pith "
                             "to the first measured ring (0 = pith present).")
        out[i] = int(fv)
    if na:
        _warn(label + ": pith offset is NA for " + str(len(na)) + " series ("
              + ", ".join(na[:8]) + (" ..." if len(na) > 8 else "")
              + "); assuming 0 (pith present).", stacklevel=4)
    return out


# --- Cambial-age alignment (audit A1 / E1) ----------------------------------

def _align(M, po_arr):
    """Place a calendar matrix ``M`` (years x series) on the cambial-age axis.

    Series i's first measured ring sits at row ``po_arr[i]`` (a true pith
    offset; 0 = pith present) and every later ring at ``po_arr[i] + (its
    calendar position - the first ring's)``: an interior NaN is an unmeasured
    ring, so it advances cambial age and stays NaN (as in CRUST and dplR). The
    NaN pattern of ``M`` selects the rings, so a signal-free copy of the
    measurement matrix aligns identically to the raw one. Returns ``rwca``
    (max cambial age x series).
    """
    ncol = M.shape[1]
    nrow = 0
    pos_list = []
    for i in range(ncol):
        pos = np.where(~np.isnan(M[:, i]))[0]
        pos_list.append(pos)
        if len(pos):
            nrow = max(nrow, int(po_arr[i]) + int(pos[-1] - pos[0]) + 1)
    rwca = np.full((max(nrow, 1), ncol), np.nan)
    for i, pos in enumerate(pos_list):
        if len(pos):
            rwca[int(po_arr[i]) + (pos - pos[0]), i] = M[pos, i]
    return rwca


def _unalign(rwica, M, po_arr):
    """Inverse of ``_align``: map a cambial-age matrix back onto the calendar
    rows where ``M`` is measured (same shape as ``M``, NaN elsewhere)."""
    out = np.full(M.shape, np.nan)
    for i in range(M.shape[1]):
        pos = np.where(~np.isnan(M[:, i]))[0]
        if len(pos):
            out[pos, i] = rwica[int(po_arr[i]) + (pos - pos[0]), i]
    return out


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
        start = max(2 * j // 3, 1)               # final third (1-based start)
        q = int(np.argmin(rwp[start - 1:j])) + start   # 1-based index of the min
        rwp[q:] = rwp[q - 1]                     # flatten after it
    return rwp


def _infill_forward(span):
    """Carry-forward infill of a cambial-age mean over its span: leading NaN
    take the first value, interior NaN the previous age's value (CRUST)."""
    span = np.asarray(span, dtype=float).copy()
    valid = np.where(~np.isnan(span))[0]
    span[:valid[0]] = span[valid[0]]
    for m in range(valid[0] + 1, len(span)):
        if np.isnan(span[m]):
            span[m] = span[m - 1]
    return span


def _crust_regional_curve(ca_m, ca_n, nrow, ss=10, rise=False, floor=0.02):
    """Build the CRUST-style regional curve: infill gaps, smooth with spline3,
    apply the minimum-value floor, and flat-extend to the full cambial-age grid."""
    last = int(np.max(np.where(ca_n > 0)[0]))    # last cambial age with data
    span = _infill_forward(ca_m[:last + 1])
    cnt = ca_n[:last + 1]
    rc_span = np.maximum(_spline3(span, cnt, ss=ss, rise=rise), floor)
    rc = np.full(nrow, np.nan)
    rc[:last + 1] = rc_span
    rc[last + 1:] = rc_span[-1]                   # flat extension
    return rc


# --- Regional curve = average by cambial age, smoothed (shared by all four) --

def _fit_regional_curve(ca_m, fit, spline_period, nyrs0, f, pos_slope, label="rcs"):
    """Smooth the age-aligned regional mean ``ca_m`` (NaN where no series
    reaches that cambial age) with the curve named by ``fit`` (canonical
    spelling, or None for no smoothing). Returns an array the same length as
    ``ca_m``, NaN at ages without data.

    ``fit`` draws on the same vocabulary as detrend(): the two splines plus the
    parametric growth curves in curvefit.py. This mirrors how ARSTAN and CRUST
    build the regional curve -- the RC is just another curve fit, including the
    "no smoothing" option (fit=None: ARSTAN's "no smooth" / CRUST rdt 2).

    Ages with zero coverage inside the span (audit A15): the parametric fits are
    given the TRUE cambial-age axis, and the splines are fit over the span with
    the gaps infilled by carry-forward, so a gap never compresses the age axis.
    """
    ca_m = np.asarray(ca_m, dtype=float)
    valid = ~np.isnan(ca_m)
    rc = np.full(len(ca_m), np.nan)
    if not valid.any():
        raise ValueError(label + ": the regional curve has no data at any "
                         "cambial age.")
    if fit is None:
        rc[valid] = ca_m[valid]
        return rc
    idx = np.where(valid)[0]
    first, last = int(idx[0]), int(idx[-1])
    if last - first + 1 != len(idx):
        _warn(label + ": no series covers " + str(last - first + 1 - len(idx))
              + " cambial age(s) inside the regional curve's span; the curve is "
              "fit on the true age axis (spline fits carry the previous age's "
              "mean across the gap).", stacklevel=5)
    ym = ca_m[valid]
    n = len(ym)
    if n < _MIN_N.get(fit, 1):
        _warn(label + ": the regional curve has only " + str(n) + " cambial "
              "age(s) with data, too few for fit='" + fit + "'; using the mean "
              "(dplR's 'dirty dog').", stacklevel=5)
        fit = "Mean"
    x = idx + 1.0                                 # true cambial age of each mean
    if fit in ("Spline", "AgeDepSpline"):
        span = _infill_forward(ca_m[first:last + 1])
        if fit == "Spline":
            sm = _caps(span, get_period(spline_period, len(span)), f)
        else:
            sm = ads(span, nyrs0=nyrs0, pos_slope=pos_slope)
        rc[valid] = np.asarray(sm, dtype=float)[idx - first]
        return rc
    if fit == "ModNegExp":
        sm = curvefit.mod_neg_exp(x, ym, pos_slope, label + " regional curve", t=x)
    elif fit == "NegExp":
        sm = curvefit.neg_exp_arstan(x, ym, pos_slope, label + " regional curve", t=x)
    elif fit == "ModHugershoff":
        sm = curvefit.mod_hugershoff(x, ym, pos_slope, label + " regional curve", t=x)
    elif fit == "Hugershoff":
        sm = curvefit.hugershoff_arstan(x, ym, label + " regional curve", t=x)
    elif fit == "GeneralExp":
        sm = curvefit.general_exp(x, ym, label + " regional curve", t=x)
    elif fit == "LinearAny":
        sm = curvefit.linear(x, ym)
    elif fit == "LinearNegative":
        sm = curvefit.linear_neg(x, ym, label + " regional curve")
    elif fit == "Mean":
        sm = curvefit.horizontal(x, ym)
    else:
        raise ValueError("unsupported fit for the regional curve: " + str(fit))
    rc[valid] = np.asarray(sm, dtype=float)
    return rc


def _regional_curve(rwca, biweight, min_n, preset, fit, spline_period, nyrs0, f,
                    pos_slope, ss=10, rise=None, label="rcs"):
    """Average ``rwca`` by cambial age and smooth it. The ONE regional-curve
    builder behind rcs, sfrcs, mrcs and sfmrcs. Returns (ca_m, ca_n, rc)."""
    nrow = rwca.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ca_m = _row_biweight(rwca) if biweight else np.nanmean(rwca, axis=1)
    ca_m = np.asarray(ca_m, dtype=float)
    ca_n = np.sum(~np.isnan(rwca), axis=1)
    if min_n is not None:
        deep = np.where(ca_n >= min_n)[0]
        if len(deep) == 0:
            raise ValueError(label + ": min_n=" + str(min_n) + " exceeds the "
                             "sample depth at every cambial age (max "
                             + str(int(ca_n.max())) + ").")
        ca_m = ca_m.copy()
        ca_m[np.arange(nrow) > int(deep.max())] = np.nan
    if preset == "crust":
        # CRUST regional curve: infill, age-dependent spline smoothed only where
        # depth >= 4 (no-rise-final-third tail), 0.02 floor, flat extension.
        rise_flag = (nrow > 1500) if rise is None else bool(rise)
        rc = _crust_regional_curve(ca_m, ca_n, nrow, ss=ss, rise=rise_flag)
    else:
        rc = _fit_regional_curve(ca_m, fit, spline_period, nyrs0, f, pos_slope,
                                 label=label)
    return ca_m, ca_n, rc


def _warn_nonpositive_rc(rc, label="rcs"):
    """Non-positive regional curve: a flexible RC can dip to or below zero, which
    makes ratio indices blow up or turn negative. ARSTAN halts here ("rcs growth
    curve goes negative"); as a library dplPy warns and continues so batch runs
    are not interrupted -- the user can pick a stiffer curve or difference mode."""
    vals = rc[~np.isnan(rc)]
    if np.any(vals <= 0):
        _warn(label + ": the regional curve is <= 0 at %d cambial age(s); ratio "
              "indices there will be unstable or negative. Consider a stiffer "
              "smoothing (a larger spline_period), fit='AgeDepSpline', or "
              "method='difference'." % int(np.sum(vals <= 0)), stacklevel=4)


# --- Index arithmetic shared by the four functions (audit E5 / E6) ----------

def _rescale_difference(diff, ratio):
    """ARSTAN calind (ind=2) / CRUST tree_ind (ind==2): per series, put the
    difference (residual) index onto the mean and SD of that series' own ratio
    index, so difference indices sit on the ~1 ratio scale. Degenerate series
    (fewer than 2 values, or zero spread) keep their raw differences."""
    out = diff.copy()
    for i in range(diff.shape[1]):
        d, r = diff[:, i], ratio[:, i]
        ok = np.isfinite(d) & np.isfinite(r)
        if ok.sum() < 2:
            continue
        sd = d[ok].std()
        if sd == 0 or not np.isfinite(sd):
            continue
        out[:, i] = (d - d[ok].mean()) / sd * r[ok].std() + r[ok].mean()
    return out


def _reset_means(rwi, single):
    """Multiple-curve mean reset (CRUST "mean single"): rescale each series'
    indices so its mean equals the mean it has under one-curve RCS, recovering
    the low-frequency variance multiple RCS removes. Arrays in, array out."""
    out = rwi.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j in range(rwi.shape[1]):
            ms, mm = np.nanmean(single[:, j]), np.nanmean(rwi[:, j])
            if np.isfinite(ms) and np.isfinite(mm) and mm != 0:
                out[:, j] = rwi[:, j] * (ms / mm)
    return out


def _indices(rwca, rc_mat, method, rescale_difference):
    """Form indices in cambial-age space: ratio (division) or difference
    (subtraction), the latter optionally rescaled onto the ratio scale."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = rwca / rc_mat
    if method == "ratio":
        return ratio
    diff = rwca - rc_mat
    return _rescale_difference(diff, ratio) if rescale_difference else diff


# --- Signal-Free RCS engine (Melvin & Briffa 2014, CRUST stand.f90) ----------
# The signal-free iteration of CRUST's rcs_detrend/det_sf_rcs. NOT validated
# against dplR -- dplR has no signal-free RCS -- so this is faithful to stand.f90
# with its numerical core (spline3) validated against CRUST's compiled kernel to
# ~1e-11. See sfrcs.py / sfmrcs.py for the public API.

def _signal_free(M, crn, present, qok, additive=False):
    """Signal-free measurements ``M / crn`` -- only where the calendar year has
    more than one tree and ``crn >= 0.01``, else the raw measurement (CRUST's
    division guard, audit E3). With ``additive`` (difference indices left on
    their own scale) the chronology is a mean departure and is SUBTRACTED
    instead, as in ssf's difference mode."""
    if additive:
        fx = np.where(present & qok[:, None], M - crn[:, None], M)
    else:
        div = present & qok[:, None] & (crn[:, None] >= 0.01)
        fx = np.where(div, M / np.where(crn == 0, np.nan, crn)[:, None], M)
    fx[~present] = np.nan
    return fx


def _sfrcs_run(M, po_arr, method="ratio", biweight=False, curve=None,
               max_iterations=40, tol=1e-3, verbose=True, alloc=None,
               reset_means=False, rescale_difference=False, label="sfrcs"):
    """Signal-free RCS (single or multiple curve). ``curve`` is the dict of
    regional-curve settings handed to ``_regional_curve`` (fit, spline_period,
    nyrs0, f, pos_slope, preset, ss, rise, min_n).

    Each iteration: form signal-free measurements ``fx = tx/crn`` (CRUST's
    division guard); build the regional curve(s) by cambial age from ``fx`` (so
    the age curve is not biased by the age structure of the sample); form the
    tree indices from the *original* ``tx`` and that curve; rescale so the whole
    chronology has mean 1; recompute the chronology. Iterate until
    ``max|crn_k - crn_(k-1)| < tol`` (CRUST's convergence measure).

    ``method="difference"`` with ``rescale_difference=True`` is CRUST's residual
    mode (indices rescaled onto the ratio scale, so the division above applies).
    Without the rescale the residuals stay in measurement units, so the engine
    runs additively: the chronology is a mean departure, subtracted from the
    measurements to form the signal-free series, and the indices are centred on
    0 instead of normalised to mean 1.

    Returns a dict: rwi (calendar x series), crn, samp_depth, history, n_iter,
    converged, rc_list (one regional curve per allocation group, the curves that
    produced rwi), ca_m_list (their unsmoothed age means) and rwca (the raw
    measurements by cambial age).
    """
    curve = dict(curve or {})
    additive = (method == "difference" and not rescale_difference)
    present = ~np.isnan(M)
    samp_depth = present.sum(axis=1)
    qok = samp_depth > 1                          # >1 tree in the calendar year
    nyear, ncol = M.shape
    rwca_orig = _align(M, po_arr)                 # raw measurements by cambial age
    nrow = rwca_orig.shape[0]

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
    rc_list, ca_m_list = [], []
    for k in range(1, max_iterations + 1):
        # 1. signal-free measurements (calendar space); iteration 1 = raw
        fx = M if k == 1 else _signal_free(M, crn, present, qok, additive)
        # 2. regional curve(s) by cambial age, from the signal-free measurements
        rwca_sf = _align(fx, po_arr)
        rc_mat = np.empty((nrow, ncol))
        rc_list, ca_m_list = [], []
        for c in curve_ids:
            sel = np.where(alloc_arr == c)[0]
            ca_m, _, rcc = _regional_curve(rwca_sf[:, sel], biweight, label=label,
                                           **curve)
            rc_mat[:, sel] = rcc[:, None]
            rc_list.append(rcc)
            ca_m_list.append(ca_m)
        # 3. tree indices from the ORIGINAL measurements and each tree's curve
        rwica = _indices(rwca_orig, rc_mat, method, rescale_difference)
        # 3b. mean-reset (multi-curve, ratios only): put each tree back on its
        # single-curve mean (CRUST "mean single") to recover low-freq variance.
        if reset_means and method == "ratio" and multi:
            _, _, rc_single = _regional_curve(rwca_sf, biweight, label=label, **curve)
            with np.errstate(invalid="ignore", divide="ignore"):
                rwica = _reset_means(rwica, rwca_orig / rc_single[:, None])
        # 4. rescale so the whole chronology has mean 1 (CRUST "prevent drift");
        #    additive residuals are centred on 0 instead
        gm = np.nanmean(rwica)
        if additive:
            rwica = rwica - (gm if np.isfinite(gm) else 0.0)
        elif np.isfinite(gm) and gm != 0:
            rwica = rwica / gm
        # 5. map indices back to calendar space, form the chronology
        idx = _unalign(rwica, M, po_arr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            crn_new = _row_biweight(idx) if biweight else np.nanmean(idx, axis=1)
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

    for rcc in rc_list:
        _warn_nonpositive_rc(rcc, label=label)
    if verbose:
        if converged:
            print(f"Signal-free RCS converged in {k} iterations "
                  f"(max change {history[-1]:.2e} < {tol:g}).")
        else:
            print(f"Signal-free RCS did NOT converge in {max_iterations} "
                  f"iterations (last max change "
                  f"{history[-1] if history else float('nan'):.2e}).")
    return {"rwi": idx, "crn": crn, "samp_depth": samp_depth,
            "history": history, "n_iter": k, "converged": converged,
            "rc_list": rc_list, "ca_m_list": ca_m_list, "rwca": rwca_orig}


def _as_rwi_frame(rwl, idx):
    """Wrap a calendar x series index matrix as a frame like ``rwl``, carrying
    provenance attrs and tagging the output as indices (audit G10)."""
    rwi = pd.DataFrame(idx, index=rwl.index, columns=rwl.columns)
    return _carry_attrs(rwl, rwi, kind="rwi")


def rcs(rwl: pd.DataFrame, po=None, fit="AgeDepSpline", method="ratio",
        spline_period=0.1, nyrs0=50, f=0.5, pos_slope=False, biweight=True,
        min_n=None, preset=None, rescale_difference=False, rc_out=False,
        plot=True, **kwargs):
    """Regional Curve Standardization of a set of ring-width series.

    Align every series on a common cambial-age axis using its pith offset, average
    to form the regional curve, smooth it, and divide (ratio) or subtract
    (difference) the curve value at each ring's cambial age. This preserves the
    low-frequency growth signal that per-series detrending removes. Shares the
    ``fit`` / ``method`` / ``spline_period`` vocabulary of ``detrend``. Analog of
    dplR's ``rcs``.

    Parameters
    ----------
    rwl : pandas.DataFrame
        ring-width series, years as the index and series as columns.
    po : pandas.DataFrame, dict, Series or None, default None
        pith offsets: a table with ``series`` and ``pith_offset`` columns (or a
        dict / Series keyed by series) giving the number of rings between the pith
        and the first measured ring, so 0 means the pith is present. None assumes 0
        for every series; a series missing from the table, or with an NA offset,
        is taken as 0 with a warning; negative or fractional offsets raise. (A dplR
        ``po`` table stores the cambial age of the first ring, where 1 = pith
        present, so subtract 1 first.)
    fit : str or None, default "AgeDepSpline"
        curve fitted to the age-aligned mean, from the ``detrend`` vocabulary:
        ``"AgeDepSpline"`` (default; age-dependent spline, honours ``pos_slope``),
        ``"Spline"`` (fixed-stiffness smoothing spline), ``"ModNegExp"``, ``"NegExp"``,
        ``"ModHugershoff"``, ``"Hugershoff"``, ``"GeneralExp"``, ``"LinearAny"``,
        ``"LinearNegative"``, ``"Mean"``, or ``None`` for no smoothing (the age-aligned
        mean itself is the regional curve). Case-insensitive.
    method : {"ratio", "difference"}, default "ratio"
        how each index is formed from the ring width and the regional-curve value at
        its cambial age: ``"ratio"`` divides, ``"difference"`` subtracts.
    spline_period : int or float, default 0.1
        smoothing wavelength for ``fit="Spline"`` (``get_period`` rule): a fraction of
        length in ``(0, 1]`` (default 0.1 = 10%), a percent-of-length spline if
        negative (-10 = 10%), or a fixed wavelength in years if > 1. The former name
        ``period=`` is accepted with a DeprecationWarning.
    nyrs0 : int, default 50
        initial stiffness for ``fit="AgeDepSpline"`` (see ``dpl.ads``).
    f : float, default 0.5
        frequency-response amplitude for ``fit="Spline"``.
    pos_slope : bool, default False
        whether the regional curve may rise at the old-age end. False keeps a
        non-increasing end, avoiding an artefactual old-age upturn. Applies to
        ``AgeDepSpline`` and the ``ModNegExp``/``NegExp``/``ModHugershoff`` linear
        fallbacks.
    biweight : bool, default True
        build the regional curve with a Tukey biweight robust mean (else arithmetic).
    min_n : int or None, default None
        truncate the regional-curve tail where by-age sample depth falls below this.
        Raises if no cambial age reaches ``min_n``.
    preset : {None, "crust"}, default None
        None follows the parameters above. "crust" builds the regional curve the CRUST
        way (Melvin & Briffa 2014): an age-dependent spline smoothed only where sample
        depth >= 4 with the no-rise tail rule, gaps infilled, a 0.02 mm floor, and flat
        tail extension. When set, ``fit``/``spline_period``/``nyrs0``/``f``/``pos_slope``
        are ignored. Case-insensitive.
    rescale_difference : bool, default False
        used only when ``method="difference"``. If True, rescale each series'
        difference index onto the mean and standard deviation of its own ratio index,
        so differences sit on the ~1 ratio scale. Default False returns residuals
        centred on ~0.
    rc_out : bool, default False
        if True, return a dict with both the detrended ``rwi`` and the regional curve
        ``rc``; otherwise return just ``rwi``.
    plot : bool, default True
        draw the regional-curve figure (series and curve vs cambial age). The former
        name ``make_plot=`` is accepted for now with a DeprecationWarning.

    Returns
    -------
    pandas.DataFrame or dict
        the detrended ``rwi`` (calendar-year indexed), or ``{"rwi", "rc"}`` when
        ``rc_out`` is True.

    Notes
    -----
    An interior NaN is an unmeasured ring: it advances the series' cambial age
    and its index stays NaN, so the years after a gap keep their true age (as in
    CRUST and dplR). Where no series reaches a cambial age inside the curve's
    span, the parametric fits use the true age axis and the splines carry the
    previous age's mean forward (with a warning).

    References
    ----------
    .. [1] https://rdrr.io/cran/dplR/man/rcs.html
    .. [2] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 1, Signal-Free
       RCS. Dendrochronologia, 32, 7-20.
    .. [3] Melvin, T. M. & Briffa, K. R. (2014) CRUST: Software for the
       implementation of Regional Chronology Standardisation: Part 2, Further RCS
       options and recommendations. Dendrochronologia, 32, 343-356.
    """
    _require_dataframe(rwl)
    spline_period = _alias_kwarg(kwargs, "period", "spline_period", spline_period, "rcs")
    plot = _plot_kwarg(plot, kwargs)
    method = _normalize_method(method)               # 'ratio' / 'difference'
    fit = None if fit is None else _normalize_fit(fit)
    preset = _normalize_choice(preset, ("crust",), "preset", none_ok=True)

    M = rwl.to_numpy(dtype=float)                    # (nyears, nseries)
    po_arr = _resolve_po(rwl, po, label="rcs")

    # ring width by cambial age: each series placed starting at its pith offset
    rwca = _align(M, po_arr)
    ca_m, ca_n, rc = _regional_curve(rwca, biweight, min_n, preset, fit,
                                     spline_period, nyrs0, f, pos_slope, label="rcs")
    _warn_nonpositive_rc(rc, label="rcs")

    # detrend in cambial-age space, then map back to calendar years
    rwica = _indices(rwca, rc[:, None], method, rescale_difference)
    rwi = _as_rwi_frame(rwl, _unalign(rwica, M, po_arr))

    if plot:
        _plot_rcs(rwca, ca_m, rc, rwca.shape[1])

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
    _finish_figure(fig)
    return ax
